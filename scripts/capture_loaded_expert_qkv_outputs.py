#!/usr/bin/env python3
"""Save original loaded BF16 expert QKV outputs for paired RKNN comparisons."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


PREFIX = "model.vlm_with_expert.lm_expert.layers.0.self_attn"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "vlm-assets-dir", "inputs", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(line) for line in args.inputs.read_text().splitlines() if line.strip()]
    if len(paths) != 40 or len({p.name for p in paths}) != 40:
        raise ValueError("Expected 40 unique development input files")
    config = SmolVLAConfig.from_pretrained(args.model_dir)
    config.device = "cuda"
    config.vlm_model_name = str(args.vlm_assets_dir.resolve())
    config.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(args.model_dir, config=config, strict=True).eval()
    attention = policy.get_submodule(PREFIX)
    projections = [attention.q_proj, attention.k_proj, attention.v_proj]
    dtypes = [str(p.weight.dtype) for p in projections]
    if any(dtype != "torch.bfloat16" for dtype in dtypes):
        raise ValueError(f"Unexpected loaded dtype: {dtypes}")
    records = []
    with torch.inference_mode():
        for path in paths:
            sample = np.load(path, allow_pickle=False).astype(np.float32)
            x = torch.from_numpy(sample).to("cuda")
            original = torch.cat([p(x.to(p.weight.dtype)).float() for p in projections], dim=-1)
            fp32 = torch.cat([torch.nn.functional.linear(x, p.weight.float(),
                                                        p.bias.float() if p.bias is not None else None)
                              for p in projections], dim=-1)
            loaded = original.cpu().numpy()
            target = args.output_dir / path.name
            np.save(target, loaded, allow_pickle=False)
            records.append({"input": path.name, "input_sha256": sha256(path),
                            "output_sha256": sha256(target), "shape": list(loaded.shape),
                            "loaded_bf16_mae_vs_fp32": float(torch.mean(torch.abs(original - fp32)))})
    report = {"module": PREFIX, "source_weight_sha256": sha256(args.model_dir / "model.safetensors"),
              "loaded_dtype": dtypes, "inputs": str(args.inputs), "samples": len(records),
              "mean_loaded_bf16_mae_vs_fp32": float(np.mean([r["loaded_bf16_mae_vs_fp32"] for r in records])),
              "records": records}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
