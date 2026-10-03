#!/usr/bin/env python3
"""Export expert layer-0 self-attention Q/K/V projections as one RKNN probe."""

import argparse
import hashlib
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


PREFIX = "model.vlm_with_expert.lm_expert.layers.0.self_attn"


class QKV(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.q_proj = attention.q_proj.float()
        self.k_proj = attention.k_proj.float()
        self.v_proj = attention.v_proj.float()

    def forward(self, x):
        return torch.cat((self.q_proj(x), self.k_proj(x), self.v_proj(x)), dim=-1)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "vlm-assets-dir", "sample-input", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--master", type=Path)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = SmolVLAConfig.from_pretrained(args.model_dir)
    config.device = "cpu"
    config.vlm_model_name = str(args.vlm_assets_dir.resolve())
    config.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(args.model_dir, config=config, strict=True)
    qkv = QKV(policy.get_submodule(PREFIX)).eval()
    replaced = []
    if args.master:
        with safe_open(str(args.master), framework="pt", device="cpu") as source:
            for local_name, parameter in qkv.named_parameters():
                key = f"{PREFIX}.{local_name}"
                if key not in source.keys():
                    raise ValueError(f"Missing {key}")
                with torch.no_grad():
                    parameter.copy_(source.get_tensor(key).float())
                replaced.append(key)
    sample_array = np.load(args.sample_input, allow_pickle=False).astype(np.float32)
    if sample_array.shape != (1, 50, 720) or not np.isfinite(sample_array).all():
        raise ValueError(f"Unexpected sample input {sample_array.shape}")
    sample = torch.from_numpy(sample_array)
    with torch.no_grad():
        expected = qkv(sample).numpy()
        output = args.output_dir / "expert_layer0_qkv_fp32.onnx"
        torch.onnx.export(qkv, sample, output, export_params=True,
                          input_names=["hidden_states"], output_names=["qkv"],
                          opset_version=17, dynamo=False)
    model = onnx.load(str(output))
    onnx.checker.check_model(model)
    actual = ReferenceEvaluator(model).run(None, {"hidden_states": sample_array})[0]
    max_abs = float(np.max(np.abs(actual - expected)))
    if not np.allclose(actual, expected, rtol=1e-4, atol=1e-4):
        raise ValueError(f"ONNX mismatch {max_abs}")
    report = {"module": PREFIX, "projection_count": 3, "input_shape": list(sample.shape),
              "output_shape": list(expected.shape), "sample_input": str(args.sample_input),
              "sample_sha256": sha256(args.sample_input),
              "source_weight_sha256": sha256(args.model_dir / "model.safetensors"),
              "master": str(args.master) if args.master else None,
              "master_sha256": sha256(args.master) if args.master else None,
              "replaced_parameters": replaced, "onnx_sha256": sha256(output),
              "onnx_bytes": output.stat().st_size, "onnx_ops": sorted({n.op_type for n in model.graph.node}),
              "onnx_reference_max_abs": max_abs}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
