#!/usr/bin/env python3
"""Export odd expert K and V separately, each with its actual independent input."""

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

import torch
from safetensors import safe_open
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

from qvla.conversion.export_all_expert_attention_qat_ptq import export
from qvla.evaluation.probe_action_sensitivity import sha256


PREFIX = "model.vlm_with_expert.lm_expert.layers"


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
    torch.manual_seed(0)
    sample = torch.randn(1, 256, 320)
    records = []
    with safe_open(str(args.qat_snapshot), framework="pt", device="cpu") as source:
        for layer in range(1, 16, 2):
            for kind in ("k", "v"):
                name = f"{PREFIX}.{layer}.self_attn.{kind}_proj"
                module = policy.get_submodule(name).float().eval()
                base = args.output_dir / f"layer_{layer:02d}" / kind
                ptq = export(module, sample, base / "ptq" / "projection_fp32.onnx")
                replaced = []
                for local_name, parameter in module.named_parameters():
                    key = f"{name}.{local_name}"
                    if key not in source.keys():
                        raise ValueError(f"QAT master lacks {key}")
                    with torch.no_grad():
                        parameter.copy_(source.get_tensor(key).float())
                    replaced.append(key)
                qat = export(module, sample, base / "qat" / "projection_fp32.onnx")
                records.append({"layer": layer, "kind": kind, "module": name,
                                "qat_replaced_parameters": replaced,
                                "ptq": ptq, "qat": qat})
            print(f"exported separate K/V layer {layer}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"scope": "odd expert K/V separately; independent real inputs required",
              "source_weight_sha256": partition["source_weight_sha256"],
              "partition_sha256": sha256(args.partition),
              "qat_snapshot_sha256": sha256(args.qat_snapshot),
              "onnx_opset": 17, "sample_shape": list(sample.shape),
              "records": records}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Exported {len(records) * 2} separate ONNX graphs", flush=True)


if __name__ == "__main__":
    main()
