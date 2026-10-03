"""Mixed PTQ runtime: CUDA INT8 GEMM and row-wise INT8 embedding storage."""
import re
import torch
from torch import nn
from torch.nn import functional as F
from real_int8_linear import RealInt8Linear


class MixedInt8Linear(RealInt8Linear):
    def forward(self, x):
        if x.device.type != 'cuda':
            raise RuntimeError('MixedInt8Linear requires CUDA integer GEMM')
        q = (torch.round(x.float() / self.activation_scale) + self.activation_zero).clamp(-128,127).to(torch.int8)
        matrix = q.reshape(-1, self.in_features).contiguous()
        rows = matrix.shape[0]
        if rows == 0:
            return x.new_empty(*x.shape[:-1], self.out_features)
        # CUDA int8 GEMM requires aligned matrix dimensions; token counts need not align.
        padded_rows = max(32, ((rows + 7) // 8) * 8)
        matrix = F.pad(matrix, (0,0,0,padded_rows - rows))
        acc = torch._int_mm(matrix, self.weight_q_t)[:rows]
        y = (acc - self.activation_zero * self.weight_sum).float()
        y = y * (self.activation_scale * self.weight_scale)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*x.shape[:-1], self.out_features).to(x.dtype)


class Int8Embedding(nn.Module):
    def __init__(self, rows, width, dtype, padding_idx=None):
        super().__init__()
        self.num_embeddings, self.embedding_dim = rows, width
        self.padding_idx = padding_idx
        self.register_buffer('weight_q', torch.empty(rows,width,dtype=torch.int8))
        self.register_buffer('weight_scale', torch.empty(rows,dtype=torch.float32))
        self.register_buffer('_dtype_marker', torch.empty((),dtype=dtype), persistent=False)

    @property
    def weight(self):
        # SmolVLA consults embedding.weight.dtype; never materialize the full FP table.
        return self._dtype_marker

    def forward(self, ids):
        return (self.weight_q[ids].float() * self.weight_scale[ids,None]).to(self._dtype_marker.dtype)


def select(policy, config):
    groups = [g for g in config['groups'] if g['v0_format']=='w8a8']
    selected = {}
    for name, mod in policy.named_modules():
        if isinstance(mod, nn.Linear) and any(re.search(g['pattern'],name+'.weight') for g in groups):
            selected[name] = mod
    embedding = 'model.vlm_with_expert.vlm.model.text_model.embed_tokens'
    if not isinstance(policy.get_submodule(embedding), nn.Embedding):
        raise ValueError('Expected standard token embedding')
    if len(selected) != 291:
        raise ValueError(f'Expected 291 Linear modules, got {len(selected)}')
    return selected, embedding


def replace_from_manifest(policy, manifest):
    names = []
    for spec in manifest:
        name = spec['name']; old = policy.get_submodule(name)
        dtype = getattr(torch, spec['compute_dtype'])
        if spec['kind']=='linear':
            if not isinstance(old,nn.Linear) or [old.out_features,old.in_features]!=spec['shape']:
                raise ValueError(f'Linear structure mismatch: {name}')
            new = MixedInt8Linear(old.in_features,old.out_features,old.bias is not None,dtype)
            names.append(name)
        elif spec['kind']=='embedding':
            if not isinstance(old,nn.Embedding) or list(old.weight.shape)!=spec['shape']:
                raise ValueError(f'Embedding structure mismatch: {name}')
            new = Int8Embedding(*spec['shape'],dtype,old.padding_idx)
        else:
            raise ValueError(spec['kind'])
        parent, leaf = name.rsplit('.',1)
        setattr(policy.get_submodule(parent),leaf,new)
    return names
