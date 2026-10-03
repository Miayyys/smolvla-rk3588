#!/usr/bin/env python3
"""Export one real SmolVLA single-input module as a static FP32 ONNX probe.

This checks an RKNN candidate subgraph. The random tensor is for export and
operator parity only; it is not PTQ calibration data or a quality test.
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
from safetensors import safe_open
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from onnx.reference import ReferenceEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--vlm-assets-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--module", default="model.vlm_with_expert.lm_expert.layers.0.mlp")
    parser.add_argument("--sample-input", type=Path,
                        help="Captured module input; random sample is only for operator feasibility")
    parser.add_argument("--master", type=Path,
                        help="Optional QAT float-master safetensors; selected module weights replace original")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = SmolVLAConfig.from_pretrained(args.model_dir)
    config.device = "cpu"
    config.vlm_model_name = str(args.vlm_assets_dir.resolve())
    config.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(args.model_dir, config=config, strict=True)
    module_name = args.module
    module = policy.get_submodule(module_name)
    source_weight_dtype = str(next(module.parameters()).dtype)
    mlp = module.float().eval()
    replaced = []
    if args.master:
        with safe_open(str(args.master), framework="pt", device="cpu") as source:
            for local_name, parameter in mlp.named_parameters():
                full_name = f"{module_name}.{local_name}"
                if full_name not in source.keys():
                    raise ValueError(f"Missing trained master weight: {full_name}")
                with torch.no_grad():
                    parameter.copy_(source.get_tensor(full_name).float())
                replaced.append(full_name)
    if args.sample_input:
        sample_array = np.load(args.sample_input, allow_pickle=False).astype(np.float32)
        if not np.isfinite(sample_array).all():
            raise ValueError("Nonfinite sample input")
        sample = torch.from_numpy(sample_array)
    else:
        if module_name != "model.vlm_with_expert.lm_expert.layers.0.mlp":
            parser.error("--sample-input required for modules other than expert layer 0")
        torch.manual_seed(0)
        sample = torch.randn(1, 50, 720)
    with torch.no_grad():
        expected = mlp(sample).numpy()
        onnx_name = ("expert_layer0_mlp_fp32.onnx" if args.sample_input is None
                     else "mlp_fp32.onnx")
        onnx_path = args.output_dir / onnx_name
        torch.onnx.export(
            mlp, sample, onnx_path, export_params=True,
            input_names=["hidden_states"], output_names=["hidden_states_out"],
            opset_version=17, dynamo=False,
        )
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    actual = ReferenceEvaluator(model).run(None, {"hidden_states": sample.numpy()})[0]
    max_abs = float(np.max(np.abs(expected - actual)))
    if not np.allclose(expected, actual, rtol=1e-4, atol=1e-4):
        raise ValueError(f"ONNX reference output diverged: max_abs={max_abs}")
    report = {
        "purpose": "FP32 export/operator feasibility only; sample input is not PTQ calibration",
        "module": module_name,
        "sample_input": str(args.sample_input.resolve()) if args.sample_input else None,
        "input_shape": list(sample.shape),
        "input_dtype": "float32",
        "source_weight_dtype": source_weight_dtype,
        "qat_float_master": str(args.master.resolve()) if args.master else None,
        "replaced_parameters": replaced,
        "onnx_opset": 17,
        "onnx_ops": sorted({node.op_type for node in model.graph.node}),
        "onnx_bytes": onnx_path.stat().st_size,
        "onnx_reference_max_abs": max_abs,
        "target_platform": "rk3588",
        "rknn_conversion": "not_measured",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
