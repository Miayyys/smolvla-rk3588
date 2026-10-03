#!/usr/bin/env python3
"""Probe whether a real SmolVLA Linear fake-quant graph reaches RKNN as Q/DQ.

This exports an untrained fake-quant subgraph only; it is a QAT conversion
feasibility check, not a trained or evaluated quantized model.
"""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))


import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx.reference import ReferenceEvaluator
from torch import nn

from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


class StaticFakeQuantLinear(nn.Module):
    def __init__(self, source: nn.Linear, act_min: float, act_max: float,
                 out_min: float, out_max: float):
        super().__init__()
        self.weight = nn.Parameter(source.weight.detach().float().clone())
        self.bias = (nn.Parameter(source.bias.detach().float().clone())
                     if source.bias is not None else None)
        act_scale = (act_max - act_min) / 255.0
        act_zero = int(np.clip(np.rint(-128 - act_min / act_scale), -128, 127))
        self.register_buffer("act_scale", torch.tensor(act_scale, dtype=torch.float32))
        self.register_buffer("act_zero", torch.tensor(act_zero, dtype=torch.int32))
        out_scale = (out_max - out_min) / 255.0
        out_zero = int(np.clip(np.rint(-128 - out_min / out_scale), -128, 127))
        self.register_buffer("out_scale", torch.tensor(out_scale, dtype=torch.float32))
        self.register_buffer("out_zero", torch.tensor(out_zero, dtype=torch.int32))
        max_abs = self.weight.detach().abs().amax(dim=1)
        self.register_buffer("weight_scales", torch.clamp(max_abs / 127.0, min=1e-12))
        self.register_buffer("weight_zeros", torch.zeros(self.weight.shape[0], dtype=torch.int32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.fake_quantize_per_tensor_affine(
            x, self.act_scale, self.act_zero, -128, 127)
        w = torch.fake_quantize_per_channel_affine(
            self.weight, self.weight_scales, self.weight_zeros, 0, -128, 127)
        y = F.linear(x, w, self.bias)
        return torch.fake_quantize_per_tensor_affine(
            y, self.out_scale, self.out_zero, -128, 127)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--vlm-assets-dir", type=Path, required=True)
    parser.add_argument("--module", default=("model.vlm_with_expert.vlm.model."
                 "vision_model.encoder.layers.10.mlp.fc1"))
    parser.add_argument("--calibration-inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = SmolVLAConfig.from_pretrained(args.model_dir)
    config.device = "cpu"
    config.vlm_model_name = str(args.vlm_assets_dir.resolve())
    config.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(args.model_dir, config=config, strict=True)
    source = policy.get_submodule(args.module)
    if not isinstance(source, nn.Linear):
        raise TypeError("Expected a Linear submodule")
    paths = [Path(line) for line in args.calibration_inputs.read_text().splitlines()
             if line.strip()]
    if not paths:
        raise ValueError("No calibration inputs")
    act_min, act_max = float("inf"), -float("inf")
    out_min, out_max = float("inf"), -float("inf")
    sample = None
    for path in paths:
        arr = np.load(path, allow_pickle=False).astype(np.float32)
        if not np.isfinite(arr).all():
            raise ValueError(f"Nonfinite input: {path}")
        if arr.shape[-1] != source.in_features:
            raise ValueError(f"Input feature mismatch: {path}")
        act_min = min(act_min, float(arr.min()))
        act_max = max(act_max, float(arr.max()))
        with torch.inference_mode():
            source_out = source(torch.from_numpy(arr)).float()
        out_min = min(out_min, float(source_out.amin()))
        out_max = max(out_max, float(source_out.amax()))
        if sample is None:
            sample = torch.from_numpy(arr)
    if act_max <= act_min or out_max <= out_min:
        raise ValueError("Calibration range has no width")
    module = StaticFakeQuantLinear(source, act_min, act_max, out_min, out_max).eval()
    with torch.inference_mode():
        expected = module(sample).numpy()
        onnx_path = args.output_dir / "qat_qdq_probe.onnx"
        torch.onnx.export(module, sample, onnx_path, opset_version=19,
                          input_names=["activation"], output_names=["output"],
                          dynamo=False)
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    ops = [node.op_type for node in model.graph.node]
    actual = ReferenceEvaluator(model).run(None, {"activation": sample.numpy()})[0]
    max_abs = float(np.max(np.abs(expected - actual)))
    mean_abs = float(np.mean(np.abs(expected - actual)))
    output_rms = float(np.sqrt(np.mean(expected ** 2)))
    np.save(args.output_dir / "sample_input.npy", sample.numpy(), allow_pickle=False)
    np.save(args.output_dir / "torch_fake_quant_output.npy", expected, allow_pickle=False)
    report = {
        "scope": "untrained_actual_checkpoint_linear_qat_qdq_export_feasibility_only",
        "module": args.module,
        "calibration_inputs": [str(path) for path in paths],
        "calibration_count": len(paths),
        "activation_range": [act_min, act_max],
        "output_range": [out_min, out_max],
        "activation_scale": float(module.act_scale),
        "activation_zero_point": int(module.act_zero),
        "output_scale": float(module.out_scale),
        "output_zero_point": int(module.out_zero),
        "weight_quantization": "symmetric per-output-channel INT8 -128..127 with maxabs/127 scale",
        "activation_quantization": "static asymmetric per-tensor INT8 -128..127",
        "input_shape": list(sample.shape),
        "onnx_ops": sorted(set(ops)),
        "quantize_linear_count": ops.count("QuantizeLinear"),
        "dequantize_linear_count": ops.count("DequantizeLinear"),
        "torch_onnx_max_abs": max_abs,
        "torch_onnx_mean_abs": mean_abs,
        "torch_output_rms": output_rms,
        "onnx_bytes": onnx_path.stat().st_size,
        "trained": False,
        "rknn_conversion": "not_measured",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "calibration_inputs"}), flush=True)
    if report["quantize_linear_count"] < 3 or report["dequantize_linear_count"] < 3:
        raise ValueError("ONNX export did not preserve input/weight/output Q/DQ")
    if mean_abs > 0.01 * output_rms:
        raise ValueError("Torch/ONNX fake-quant mean error exceeds 1% output RMS")


if __name__ == "__main__":
    main()
