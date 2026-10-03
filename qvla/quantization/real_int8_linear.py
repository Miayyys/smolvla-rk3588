"""Packed per-channel W8A8 Linear using CUDA integer GEMM, not fake quantization."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))


import torch
from torch import nn


class RealInt8Linear(nn.Module):
    def __init__(self, in_features, out_features, has_bias, compute_dtype):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_q_t", torch.empty(in_features, out_features, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32))
        self.register_buffer("weight_sum", torch.empty(out_features, dtype=torch.int32))
        self.register_buffer("activation_scale", torch.empty((), dtype=torch.float32))
        self.register_buffer("activation_zero", torch.empty((), dtype=torch.int32))
        self.register_buffer("_dtype_marker", torch.empty((), dtype=compute_dtype), persistent=False)
        if has_bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=torch.float32))
        else:
            self.bias = None

    @property
    def weight(self):
        """LeRobot reads only weight.dtype for expert attention dispatch."""
        return self._dtype_marker

    def forward(self, x):
        if x.device.type != "cuda":
            raise RuntimeError("RealInt8Linear requires a CUDA device for torch._int_mm")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected {self.in_features} input channels, got {x.shape[-1]}")
        quantized = torch.round(x.float() / self.activation_scale)
        quantized = (quantized + self.activation_zero).clamp(-128, 127).to(torch.int8)
        matrix = quantized.reshape(-1, self.in_features)
        accumulated = torch._int_mm(matrix, self.weight_q_t)
        corrected = accumulated - self.activation_zero * self.weight_sum
        output = corrected.float() * (self.activation_scale * self.weight_scale)
        if self.bias is not None:
            output = output + self.bias
        return output.reshape(*x.shape[:-1], self.out_features).to(dtype=x.dtype)


def replace_expert_linears(policy):
    """Replace all 112 expert Linears before strict loading of a packed checkpoint."""
    prefix = "model.vlm_with_expert.lm_expert"
    selected = [(name, module) for name, module in policy.named_modules()
                if name.startswith(prefix) and isinstance(module, nn.Linear)]
    if len(selected) != 112:
        raise ValueError(f"Expected 112 expert Linears, got {len(selected)}")
    for name, linear in selected:
        parent, leaf = name.rsplit(".", 1)
        setattr(policy.get_submodule(parent), leaf,
                RealInt8Linear(linear.in_features, linear.out_features,
                               linear.bias is not None, torch.bfloat16))
    return [name for name, _ in selected]
