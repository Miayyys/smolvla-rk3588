#!/usr/bin/env python3
"""Export QKV and attention output projections for all expert layers in one load."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx.reference import ReferenceEvaluator
from safetensors import safe_open
from torch import nn
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

from export_expert_qkv_probe import QKV
from probe_action_sensitivity import sha256


PREFIX = "model.vlm_with_expert.lm_expert.layers"


class KV(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.k_proj = attention.k_proj.float()
        self.v_proj = attention.v_proj.float()

    def forward(self, x):
        return torch.cat((self.k_proj(x), self.v_proj(x)), dim=-1)


def export(module, sample, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        expected = module(sample).numpy()
        torch.onnx.export(module, sample, output, export_params=True,
                          input_names=["hidden_states"], output_names=["projection_output"],
                          opset_version=17, dynamo=False)
    model = onnx.load(str(output))
    onnx.checker.check_model(model)
    actual = ReferenceEvaluator(model).run(None, {"hidden_states": sample.numpy()})[0]
    max_abs = float(np.max(np.abs(expected - actual)))
    if not np.allclose(expected, actual, rtol=1e-4, atol=1e-4):
        raise ValueError(f"ONNX parity failed: {output}, {max_abs}")
    return {"onnx": str(output), "onnx_sha256": sha256(output),
            "onnx_bytes": output.stat().st_size, "output_shape": list(expected.shape),
            "reference_max_abs": max_abs,
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
    if len(policy.get_submodule(PREFIX)) != 16:
        raise ValueError("Unexpected expert layer count")
    torch.manual_seed(0)
    samples = {"qkv": torch.randn(1, 50, 720), "q": torch.randn(1, 50, 720),
               "kv": torch.randn(1, 256, 320), "out": torch.randn(1, 50, 960)}
    records = []
    with safe_open(str(args.qat_snapshot), framework="pt", device="cpu") as source:
        available = set(source.keys())
        for layer in range(16):
            attention_name = f"{PREFIX}.{layer}.self_attn"
            attention = policy.get_submodule(attention_name)
            for kind in (("qkv", "out") if layer % 2 == 0 else ("q", "kv", "out")):
                module = (QKV(attention).eval() if kind == "qkv" else
                          KV(attention).eval() if kind == "kv" else
                          attention.q_proj.float().eval() if kind == "q" else
                          attention.o_proj.float().eval())
                prefix = (attention_name if kind in ("qkv", "kv") else
                          attention_name + f".{kind}_proj" if kind == "q" else
                          attention_name + ".o_proj")
                base = args.output_dir / f"layer_{layer:02d}" / kind
                sample = samples[kind]
                ptq = export(module, sample, base / "ptq" / "projection_fp32.onnx")
                replaced = []
                for local_name, parameter in module.named_parameters():
                    key = prefix + "." + local_name
                    if key not in available:
                        raise ValueError(f"QAT master lacks {key}")
                    with torch.no_grad():
                        parameter.copy_(source.get_tensor(key).float())
                    replaced.append(key)
                qat = export(module, sample, base / "qat" / "projection_fp32.onnx")
                records.append({"layer": layer, "kind": kind,
                                "module": attention_name, "qat_replaced_parameters": replaced,
                                "ptq": ptq, "qat": qat})
            print(f"exported attention QKV/out PTQ/QAT {layer+1}/16", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"scope": "16 expert attention layers, QKV concat and output projection; no quantization yet",
              "source_weight_sha256": partition["source_weight_sha256"],
              "partition_sha256": sha256(args.partition),
              "qat_snapshot_sha256": sha256(args.qat_snapshot),
              "onnx_opset": 17,
              "sample_shapes": {kind: list(sample.shape) for kind, sample in samples.items()},
              "records": records}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Exported {len(records) * 2} ONNX graphs", flush=True)


if __name__ == "__main__":
    main()
