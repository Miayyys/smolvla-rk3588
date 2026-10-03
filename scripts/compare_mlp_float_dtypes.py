#!/usr/bin/env python3
"""Compare loaded, BF16, FP16 and FP32 MLP arithmetic on captured inputs."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


MODULES = {
    "vision10": "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.10.mlp",
    "vision11": "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.11.mlp",
    "language3": "model.vlm_with_expert.vlm.model.text_model.layers.3.mlp",
    "language4": "model.vlm_with_expert.vlm.model.text_model.layers.4.mlp",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mae(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().mean().item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--vlm-assets-dir", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-loaded-outputs-dir", type=Path,
                        help="Save original loaded-module outputs for paired RKNN parity")
    args = parser.parse_args()
    config = SmolVLAConfig.from_pretrained(args.model_dir)
    config.device = "cuda"
    config.vlm_model_name = str(args.vlm_assets_dir.resolve())
    config.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(args.model_dir, config=config, strict=True).eval()
    report = {"scope": "isolated MLP arithmetic only, no RKNN or full-action inference",
              "source_weight_sha256": sha256(args.model_dir / "model.safetensors"),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "reference": "FP32 copy of the loaded weights on captured inputs",
              "modules": {}}
    for short, module_name in MODULES.items():
        original = policy.get_submodule(module_name).eval()
        source_dtypes = sorted({str(p.dtype) for p in original.parameters()})
        if source_dtypes not in (["torch.float32"], ["torch.bfloat16"]):
            raise ValueError(f"Unexpected source parameter dtype for {short}: {source_dtypes}")
        bf16 = copy.deepcopy(original).bfloat16().eval().cuda()
        fp16 = copy.deepcopy(original).half().eval().cuda()
        fp32 = copy.deepcopy(original).float().eval().cuda()
        param_count = sum(p.numel() for p in original.parameters())
        source_weight_values = torch.cat([p.detach().float().flatten() for p in original.parameters()])
        cast_weight_values = source_weight_values.half().float()
        bf16_weight_values = source_weight_values.bfloat16().float()
        weight_overflow = int(torch.isinf(source_weight_values.half()).sum().item())
        weight_zero_underflow = int(((source_weight_values != 0) & (cast_weight_values == 0)).sum().item())
        paths = sorted((args.runs_dir / f"rknn_{short}_dev_probe").glob("task_*.npy"))
        if len(paths) != 4:
            raise ValueError(f"Expected 4 development inputs for {short}, got {len(paths)}")
        rows = []
        for path in paths:
            x = torch.from_numpy(np.load(path, allow_pickle=False).astype(np.float32)).cuda()
            with torch.inference_mode():
                y32 = fp32(x).float()
                y16 = fp16(x.half()).float()
                ybf = bf16(x.bfloat16()).float()
                ysource = original(x.to(next(original.parameters()).dtype)).float()
            loaded_output = None
            if args.save_loaded_outputs_dir:
                output_dir = args.save_loaded_outputs_dir / short
                output_dir.mkdir(parents=True, exist_ok=True)
                loaded_output = output_dir / path.name
                np.save(loaded_output, ysource.cpu().numpy(), allow_pickle=False)
            rows.append({"input": path.name, "input_sha256": sha256(path),
                         "input_shape": list(x.shape),
                         "input_bf16_recast_max_abs": float((x-x.bfloat16().float()).abs().max().item()),
                         "input_min": float(x.min().item()), "input_max": float(x.max().item()),
                         "input_fp16_overflow_elements": int(torch.isinf(x.half()).sum().item()),
                         "input_fp16_zero_underflow_elements": int(((x != 0) & (x.half().float() == 0)).sum().item()),
                         "output_fp32_min": float(y32.min().item()),
                         "output_fp32_max": float(y32.max().item()),
                         "bf16_vs_fp32_mae": mae(ybf, y32),
                         "fp16_vs_fp32_mae": mae(y16, y32),
                         "fp16_vs_bf16_mae": mae(y16, ybf),
                         "loaded_vs_fp32_mae": mae(ysource, y32),
                         "loaded_output": str(loaded_output) if loaded_output else None,
                         "loaded_output_sha256": sha256(loaded_output) if loaded_output else None})
        metrics = ("bf16_vs_fp32_mae", "fp16_vs_fp32_mae", "fp16_vs_bf16_mae",
                   "loaded_vs_fp32_mae")
        report["modules"][short] = {
            "module": module_name, "source_parameter_dtypes": source_dtypes,
            "parameter_count": param_count,
            "source_runtime_parameter_bytes": param_count * (4 if source_dtypes == ["torch.float32"] else 2),
            "fp16_parameter_bytes": param_count * 2,
            "bf16_parameter_bytes": param_count * 2,
            "weight_min": float(source_weight_values.min().item()),
            "weight_max": float(source_weight_values.max().item()),
            "weight_fp16_cast_mae": mae(cast_weight_values, source_weight_values),
            "weight_bf16_cast_mae": mae(bf16_weight_values, source_weight_values),
            "weight_fp16_overflow_elements": weight_overflow,
            "weight_fp16_zero_underflow_elements": weight_zero_underflow,
            "samples": len(rows),
            **{f"mean_{key}": float(np.mean([row[key] for row in rows])) for key in metrics},
            "records": rows,
        }
        print(short, {key: report["modules"][short][f"mean_{key}"] for key in metrics}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
