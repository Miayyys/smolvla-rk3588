"""Differentiable numerical QAT references for frozen local HAQ assignments.

FP master weights remain trainable. INT16/DFP are local reference schemes,
not a guarantee of equality with RKNN's internal quantizer.
"""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import math
import torch
from torch import nn
from torch.nn import functional as F


class ExactForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, exact, surrogate):
        return exact

    @staticmethod
    def backward(ctx, gradient):
        return None,gradient


class MixedQATOperator(nn.Module):
    def __init__(self, original, fmt, stats=None):
        super().__init__()
        if fmt not in ('w8a8','w16a16i','w16a16i_dfp','cpu_int8_row_lookup','float16','bfloat16'):
            raise ValueError('Unknown QAT format')
        self.fmt=fmt
        self.kind=('linear' if isinstance(original,nn.Linear) else
                   'conv2d' if isinstance(original,nn.Conv2d) else 'embedding')
        self.original_dtype=original.weight.dtype
        self.register_buffer('_dtype_marker',torch.empty((),dtype=self.original_dtype),persistent=False)
        self.master_weight=nn.Parameter(original.weight.detach().float().clone())
        bias=getattr(original,'bias',None)
        self.bias=None if bias is None else nn.Parameter(bias.detach().float().clone())
        self.fake_quant_enabled=True
        self._quantized_version=None
        self.register_buffer('_q_weight',None,persistent=False)
        self.register_buffer('_weight_scale',None,persistent=False)
        if self.kind=='linear':self.in_features,self.out_features=original.in_features,original.out_features
        elif self.kind=='conv2d':
            self.stride,self.padding,self.dilation,self.groups=original.stride,original.padding,original.dilation,original.groups
        else:self.num_embeddings,self.embedding_dim,self.padding_idx=original.num_embeddings,original.embedding_dim,original.padding_idx
        if fmt in ('w8a8','w16a16i','w16a16i_dfp','cpu_int8_row_lookup'):
            self.bits=8 if fmt in ('w8a8','cpu_int8_row_lookup') else 16
            bound=2**(self.bits-1)-1
            if self.kind!='embedding':
                if stats is None:raise ValueError('Missing isolated activation calibration')
                lo,hi=min(stats['min'],0.),max(stats['max'],0.)
                if not math.isfinite(lo+hi) or hi<=lo:raise ValueError('Invalid calibration')
                if fmt=='w16a16i_dfp':
                    scale=2**math.ceil(math.log2(max(abs(lo),abs(hi),1e-12)/bound));zero=0
                else:
                    scale=max((hi-lo)/(2*bound+1),1e-12)
                    zero=max(-bound-1,min(bound,round(-bound-1-lo/scale)))
                self.register_buffer('activation_scale',torch.tensor(scale),persistent=False)
                self.register_buffer('activation_zero',torch.tensor(zero,dtype=torch.int32),persistent=False)

    @property
    def weight(self):
        # Parent SmolVLA dispatch must retain its original BF16/F32 interface.
        return self._dtype_marker

    def weight_scales(self):
        self.refresh_quantized_weight()
        return self._weight_scale

    def refresh_quantized_weight(self):
        if self._quantized_version==self.master_weight._version:return
        bound=2**(self.bits-1)-1
        # Packing computes rounding on CPU. Use that same detached path once
        # per optimizer update to avoid CPU/CUDA division differences at ties.
        flat=self.master_weight.detach().float().cpu().flatten(1)
        scales=flat.abs().amax(1).clamp_min(1e-12)/bound
        if self.fmt=='w16a16i_dfp':
            scale=2**math.ceil(math.log2(float(scales.max())))
            scales=torch.full_like(scales,scale)
        q=torch.round(flat/scales[:,None]).clamp(-bound,bound)
        self._q_weight=q.reshape(self.master_weight.shape).to(
            device=self.master_weight.device,dtype=torch.int8 if self.bits==8 else torch.int16)
        self._weight_scale=scales.to(self.master_weight.device)
        self._quantized_version=self.master_weight._version

    @staticmethod
    def ste_round(values,scale,zero,low,high):
        coordinates=values/scale+zero
        # Match the packer's round-before-zero operation and clipping derivative.
        decoded=(torch.round(values/scale)+zero).clamp(low,high).sub(zero)*scale
        surrogate=torch.where((coordinates>=low)&(coordinates<=high),values,values.detach())
        return ExactForward.apply(decoded.detach(),surrogate)

    def quantized_weight(self):
        if not self.fake_quant_enabled:return self.master_weight.to(self.original_dtype)
        if self.fmt in ('float16','bfloat16'):
            return self.master_weight.to(torch.float16 if self.fmt=='float16' else torch.bfloat16)
        bound=2**(self.bits-1)-1
        scales=self.weight_scales().reshape(-1,*([1]*(self.master_weight.ndim-1)))
        decoded=self._q_weight.float()*scales
        coordinates=self.master_weight/scales
        surrogate=torch.where((coordinates>=-bound)&(coordinates<=bound),self.master_weight,self.master_weight.detach())
        return ExactForward.apply(decoded,surrogate)

    def forward(self,x):
        w=self.quantized_weight()
        if self.kind=='embedding':return F.embedding(x,w,padding_idx=self.padding_idx).to(self.original_dtype)
        dtype=x.dtype
        raw_x=x
        if self.fake_quant_enabled and self.fmt in ('w8a8','w16a16i','w16a16i_dfp'):
            bound=2**(self.bits-1)-1
            x=self.ste_round(x.float(),self.activation_scale,self.activation_zero,-bound-1,bound)
        x=x.to(w.dtype);bias=None if self.bias is None else self.bias.to(w.dtype)
        y=(F.linear(x,w,bias) if self.kind=='linear' else F.conv2d(x,w,bias,self.stride,self.padding,self.dilation,self.groups))
        if self.fake_quant_enabled and self.kind=='linear' and self.fmt=='w8a8' and x.device.type=='cuda':
            # Integer forward exactly matches the deployed local reference;
            # floating fake-quant matmul supplies the STE backward gradient.
            with torch.no_grad():
                q=(torch.round(raw_x.float()/self.activation_scale)+self.activation_zero).clamp(-128,127)
                scales=self.weight_scales()
                qw=self._q_weight
                ni,no=math.ceil(self.in_features/8)*8,math.ceil(self.out_features/8)*8
                qt=F.pad(qw,(0,ni-self.in_features,0,no-self.out_features)).t().contiguous()
                matrix=q.reshape(-1,self.in_features).to(torch.int8);rows=matrix.shape[0]
                matrix=F.pad(matrix,(0,ni-self.in_features,0,max(32,math.ceil(rows/8)*8)-rows))
                accum=torch._int_mm(matrix.contiguous(),qt)[:rows,:self.out_features]
                exact=(accum-self.activation_zero*qw.sum(1,dtype=torch.int32)).float()
                exact*=self.activation_scale*scales
                if self.bias is not None:exact+=self.bias
                exact=exact.reshape(*raw_x.shape[:-1],self.out_features)
            y=ExactForward.apply(exact,y)
        return y.to(dtype)


def prepare_mixed_qat(policy,space,assignment,stats):
    from qvla.haq.search_space import validate_assignment
    validate_assignment(space,assignment)
    for parameter in policy.parameters():parameter.requires_grad_(False)
    names=[]
    for site in space['action_sites']:
        name=site['module'];original=policy.get_submodule(name)
        if list(original.weight.shape)!=site['weight_shape']:raise ValueError('Unexpected module shape')
        replacement=MixedQATOperator(original,assignment[name],stats.get(name)).to(original.weight.device)
        parent,leaf=name.rsplit('.',1);setattr(policy.get_submodule(parent),leaf,replacement);names.append(name)
    return names
