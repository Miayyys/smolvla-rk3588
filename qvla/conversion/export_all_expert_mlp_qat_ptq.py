#!/usr/bin/env python3
"""Export original-FP PTQ and trained-master QAT ONNX for all expert MLPs in one load."""

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
from onnx.reference import ReferenceEvaluator
from safetensors import safe_open
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

from qvla.evaluation.probe_action_sensitivity import sha256


PREFIX = "model.vlm_with_expert.lm_expert.layers"


def export(module, sample, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        expected = module(sample).numpy()
        torch.onnx.export(module, sample, output, export_params=True,
                          input_names=["hidden_states"], output_names=["hidden_states_out"],
                          opset_version=17, dynamo=False)
    model = onnx.load(str(output))
    onnx.checker.check_model(model)
    actual = ReferenceEvaluator(model).run(None, {"hidden_states": sample.numpy()})[0]
    max_abs = float(np.max(np.abs(actual - expected)))
    if not np.allclose(actual, expected, rtol=1e-4, atol=1e-4):
        raise ValueError(f"ONNX parity failed: {output}, max_abs={max_abs}")
    return {"onnx": str(output), "onnx_sha256": sha256(output),
            "onnx_bytes": output.stat().st_size, "reference_max_abs": max_abs,
            "onnx_ops": sorted({node.op_type for node in model.graph.node})}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "vlm-assets-dir", "qat-snapshot", "partition", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    partition = json.loads(args.partition.read_text())
    if partition["source_weight_sha256"] != sha256(args.model_dir / "model.safetensors"):
        raise ValueError("Original checkpoint differs from frozen partition")
    config = SmolVLAConfig.from_pretrained(args.model_dir)
    config.device = "cpu"
    config.vlm_model_name = str(args.vlm_assets_dir.resolve())
    config.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(args.model_dir, config=config, strict=True).eval()
    n_layers = len(policy.get_submodule(PREFIX))
    if n_layers != 16:
        raise ValueError(f"Expected 16 expert layers, got {n_layers}")
    torch.manual_seed(0)
    sample = torch.randn(1, 50, 720)
    records = []
    with safe_open(str(args.qat_snapshot), framework="pt", device="cpu") as source:
        available = set(source.keys())
        for layer in range(n_layers):
            name = f"{PREFIX}.{layer}.mlp"
            module = policy.get_submodule(name).float().eval()
            base = args.output_dir / f"layer_{layer:02d}"
            ptq = export(module, sample, base / "ptq" / "mlp_fp32.onnx")
            replaced = []
            for local_name, parameter in module.named_parameters():
                key = f"{name}.{local_name}"
                if key not in available:
                    raise ValueError(f"QAT master lacks {key}")
                with torch.no_grad():
                    parameter.copy_(source.get_tensor(key).float())
                replaced.append(key)
            qat = export(module, sample, base / "qat" / "mlp_fp32.onnx")
            records.append({"layer": layer, "module": name,
                            "source_weight_dtype": "torch.bfloat16",
                            "qat_replaced_parameters": replaced, "ptq": ptq, "qat": qat})
            print(f"exported PTQ/QAT expert MLP {layer+1}/{n_layers}", flush=True)
    report = {"scope": "16 expert MLPs; QAT FP master versus original FP, no quantization yet",
              "source_weight_sha256": partition["source_weight_sha256"],
              "partition_sha256": sha256(args.partition),
              "qat_snapshot_sha256": sha256(args.qat_snapshot),
              "onnx_opset": 17, "sample_shape": list(sample.shape), "records": records}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Exported {len(records) * 2} ONNX graphs", flush=True)


if __name__ == "__main__":
    main()
