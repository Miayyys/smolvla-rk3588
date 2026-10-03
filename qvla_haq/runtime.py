"""Full selectable operator assignment, with explicitly scoped local execution.

INT8 Linear uses CUDA integer GEMM. Integer16 and quantized Conv use decoded
weights in floating kernels: numerical references, not RKNN kernel replicas.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ConfiguredOperator(nn.Module):
    def __init__(self, original, fmt, stats=None):
        super().__init__()
        self.fmt = fmt
        self.kind = ('linear' if isinstance(original, nn.Linear) else
                     'conv2d' if isinstance(original, nn.Conv2d) else 'embedding')
        self.original_dtype = original.weight.dtype
        self.shape = list(original.weight.shape)
        self.register_buffer('_dtype_marker', torch.empty((), dtype=self.original_dtype), persistent=False)
        if self.kind == 'linear':
            self.in_features, self.out_features = original.in_features, original.out_features
        elif self.kind == 'conv2d':
            self.stride, self.padding = original.stride, original.padding
            self.dilation, self.groups = original.dilation, original.groups
        else:
            self.num_embeddings, self.embedding_dim = self.shape
            self.padding_idx = original.padding_idx
        integer = fmt in ('w8a8', 'w16a16i', 'w16a16i_dfp', 'cpu_int8_row_lookup')
        w = original.weight.detach().float()
        bias = getattr(original, 'bias', None)
        self.register_buffer('bias', None if bias is None else bias.detach().float().clone())
        if not integer:
            dtype = torch.float16 if fmt == 'float16' else torch.bfloat16
            self.register_buffer('stored_weight', w.to(dtype))
        else:
            self.bits = 8 if fmt in ('w8a8', 'cpu_int8_row_lookup') else 16
            bound = 2 ** (self.bits - 1) - 1
            flat = w.flatten(1)
            scale = flat.abs().amax(1).clamp_min(1e-12) / bound
            if fmt == 'w16a16i_dfp':
                # Tensor-wide power-of-two scale; explicit approximation to RKNN DFP.
                scale = torch.full_like(scale, 2 ** math.ceil(math.log2(float(scale.max()))))
            q = torch.round(flat / scale[:, None]).clamp(-bound, bound)
            self.register_buffer('weight_scale', scale)
            self.register_buffer('stored_weight', q.reshape(w.shape).to(
                torch.int8 if self.bits == 8 else torch.int16))
            if self.kind != 'embedding':
                if stats is None:
                    raise ValueError('Integer operator requires isolated calibration statistics')
                lo, hi = min(stats['min'], 0.), max(stats['max'], 0.)
                if not math.isfinite(lo + hi):
                    raise ValueError('Nonfinite activation calibration range')
                amin, amax = -2 ** (self.bits - 1), bound
                if fmt == 'w16a16i_dfp':
                    a = 2 ** math.ceil(math.log2(max(abs(lo), abs(hi), 1e-12) / bound))
                    z = 0
                else:
                    a = max((hi - lo) / (amax - amin), 1e-12)
                    z = max(amin, min(amax, round(amin - lo / a)))
                self.register_buffer('activation_scale', torch.tensor(a, dtype=torch.float32))
                self.register_buffer('activation_zero', torch.tensor(z, dtype=torch.int32))
        if self.kind == 'linear' and fmt == 'w8a8':
            # A runtime-only transpose/alignment cache, excluded from file size.
            ni, no = math.ceil(self.shape[1] / 8) * 8, math.ceil(self.shape[0] / 8) * 8
            self.register_buffer('q_t', F.pad(self.stored_weight, (0, ni-self.shape[1], 0,
                                                                no-self.shape[0])).t().contiguous(),
                                 persistent=False)
            self.register_buffer('weight_sum', self.stored_weight.sum(1, dtype=torch.int32), persistent=False)

    @property
    def weight(self):
        # SmolVLA checks dispatch dtype without requiring a decoded full weight.
        return self._dtype_marker

    def refresh(self):
        if self.kind == 'linear' and self.fmt == 'w8a8':
            ni, no = self.q_t.shape
            self.q_t = F.pad(self.stored_weight, (0, ni-self.shape[1], 0,
                                                no-self.shape[0])).t().contiguous()
            self.weight_sum = self.stored_weight.sum(1, dtype=torch.int32)

    def forward(self, x):
        if self.kind == 'embedding':
            w = self.stored_weight[x]
            if self.fmt == 'cpu_int8_row_lookup':
                w = w.float() * self.weight_scale[x, None]
            return w.to(self.original_dtype)
        output_dtype = x.dtype
        if self.fmt in ('float16', 'bfloat16'):
            dtype = self.stored_weight.dtype
            w, inp = self.stored_weight, x.to(dtype)
            b = None if self.bias is None else self.bias.to(dtype)
        else:
            bound = 2 ** (self.bits - 1) - 1
            q = (torch.round(x.float()/self.activation_scale) + self.activation_zero).clamp(-bound-1,bound)
            if self.kind == 'linear' and self.fmt == 'w8a8':
                if x.device.type != 'cuda':
                    raise RuntimeError('W8A8 Linear requires CUDA integer GEMM')
                matrix = q.reshape(-1, self.in_features).to(torch.int8)
                rows = matrix.shape[0]
                matrix = F.pad(matrix, (0, self.q_t.shape[0]-self.in_features,
                                        0, max(32, math.ceil(rows/8)*8)-rows))
                acc = torch._int_mm(matrix.contiguous(), self.q_t)[:rows, :self.out_features]
                y = (acc-self.activation_zero*self.weight_sum).float()
                y *= self.activation_scale*self.weight_scale
                if self.bias is not None:
                    y += self.bias
                return y.reshape(*x.shape[:-1], self.out_features).to(output_dtype)
            inp = (q-self.activation_zero)*self.activation_scale
            w = self.stored_weight.float()*self.weight_scale.reshape(-1,*([1]*(len(self.shape)-1)))
            b = self.bias
        y = (F.linear(inp,w,b) if self.kind == 'linear' else
             F.conv2d(inp,w,b,self.stride,self.padding,self.dilation,self.groups))
        return y.to(output_dtype)


def apply_assignment(policy, space, assignment, stats, originals):
    from qvla_haq.search_space import validate_assignment
    validate_assignment(space, assignment)
    for site in space['action_sites']:
        name = site['module']
        old = originals[name]
        if list(old.weight.shape) != site['weight_shape']:
            raise ValueError(f'Weight shape mismatch: {name}')
        op = ConfiguredOperator(old, assignment[name], stats.get(name))
        parent, leaf = name.rsplit('.',1)
        setattr(policy.get_submodule(parent),leaf,op)


def standalone_table_cost(space, tables, assignment, calls):
    """Sum measured isolated median call costs, not end-to-end NPU latency."""
    costs = {r['case_id']: r for r in tables['measured_costs']}
    costs.update({r['case_id']:r for r in tables['supplemental_costs']})
    rows = []
    for site in space['action_sites']:
        name = site['module']
        option = next(o for o in site['options'] if o['format'] == assignment[name])
        ids = option['cost_case_ids']
        if len(ids) != 1 or name not in calls or calls[name]['calls_per_observation'] <= 0:
            raise ValueError(f'Missing/ambiguous cost or graph call count: {name}')
        row = costs[ids[0]]
        count = calls[name]['calls_per_observation']
        rows.append({'module':name,'format':assignment[name], 'case_id':ids[0],
                     'p50_ms':row['p50_ms'],'calls_per_observation':count,
                     'cost_ms':count*row['p50_ms'],
                     'measured_input_shape':row['input_shape'],
                     'observed_input_shapes':calls[name]['shapes'],
                     'shape_exact':calls[name]['shapes'] == [row['input_shape']],
                     'timing_stable':row.get('timing_stable_20pct')})
    return {'scope':'sum_of_isolated_signature_medians_proxy_not_end_to_end_latency',
            'total_ms':sum(r['cost_ms'] for r in rows),
            'shape_mismatch_sites':sum(not r['shape_exact'] for r in rows),
            'fusion_adjustment':'not_applied; no fused costs double counted',
            'unmeasured_graph_costs':'norm/softmax/attention/flow bookkeeping; not included',
            'boundary_cost':'standalone float32 IO included; actual graph conversions not calibrated',
            'rows':rows}
