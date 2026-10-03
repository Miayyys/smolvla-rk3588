#!/usr/bin/env python3
"""Pack SmolVLA as QAT-expert + PTQ-VLM, or independent PTQ-only control."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))


import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from torchao.quantization import Int8WeightOnlyConfig, quantize_


EXPERT_PREFIX = "model.vlm_with_expert.lm_expert"
VLM_PREFIX = "model.vlm_with_expert.vlm"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expert_linear(module: nn.Module, name: str) -> bool:
    return (name.startswith(EXPERT_PREFIX) and isinstance(module, nn.Linear)
            and module.in_features % 16 == 0)


def vlm_linear(module: nn.Module, name: str) -> bool:
    return name.startswith(VLM_PREFIX) and isinstance(module, nn.Linear)


def load_float_model(model_dir: Path, vlm_dir: Path) -> SmolVLAPolicy:
    config = SmolVLAConfig.from_pretrained(model_dir)
    config.device = "cuda"
    config.vlm_model_name = str(vlm_dir.resolve())
    config.load_vlm_weights = False
    return SmolVLAPolicy.from_pretrained(model_dir, config=config, strict=True)


def build_quantized(model_dir: Path, vlm_dir: Path, qat_checkpoint: Path | None) -> SmolVLAPolicy:
    policy = load_float_model(model_dir, vlm_dir)
    quantize_(policy, Int8WeightOnlyConfig(group_size=16), filter_fn=expert_linear)
    if qat_checkpoint:
        # This is a locally generated checkpoint, not an arbitrary third-party pickle.
        saved = torch.load(qat_checkpoint, map_location="cpu", weights_only=False)
        policy.load_state_dict(saved, strict=True)
    quantize_(policy, Int8WeightOnlyConfig(), filter_fn=vlm_linear)
    policy.eval()
    return policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--vlm-assets-dir", type=Path, required=True)
    parser.add_argument("--qat-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy = build_quantized(args.model_dir, args.vlm_assets_dir, args.qat_checkpoint)
    expert_names = [name for name, module in policy.named_modules()
                    if name.startswith(EXPERT_PREFIX) and isinstance(module, nn.Linear)]
    vlm_names = [name for name, module in policy.named_modules()
                 if name.startswith(VLM_PREFIX) and isinstance(module, nn.Linear)]
    expert_q = sum(type(policy.get_submodule(name).weight).__name__ == "AffineQuantizedTensor"
                   for name in expert_names)
    vlm_q = sum(type(policy.get_submodule(name).weight).__name__ == "AffineQuantizedTensor"
                for name in vlm_names)
    if expert_q != 112 or vlm_q < 50:
        raise ValueError(f"Unexpected quantization coverage: expert={expert_q}, vlm={vlm_q}")

    # Preserve original BF16/F32 storage for untouched tensors. The model loader
    # may upcast BF16 tensors to F32 in memory; saving those as F32 hides savings.
    state = policy.state_dict()
    with safe_open(args.model_dir / "model.safetensors", framework="pt", device="cpu") as source:
        source_dtypes = {key: source.get_slice(key).get_dtype() for key in source.keys()}
    for key, value in state.items():
        if type(value).__name__ == "AffineQuantizedTensor":
            continue
        original = source_dtypes.get(key)
        if original == "BF16" and value.dtype == torch.float32:
            state[key] = value.to(torch.bfloat16)
    checkpoint = args.output_dir / "mixed_w8.pt"
    torch.save(state, checkpoint)
    del state
    # Verify actual reload and one representative quantized layer's packed type.
    reloaded = build_quantized(args.model_dir, args.vlm_assets_dir, None)
    reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False), strict=True)
    first_expert = reloaded.get_submodule(expert_names[0]).weight
    first_vlm = reloaded.get_submodule(vlm_names[0]).weight
    if any(type(weight).__name__ != "AffineQuantizedTensor" for weight in (first_expert, first_vlm)):
        raise ValueError("Reloaded model lost quantized weights")
    report = {"method": "QAT_expert_PTQ_VLM" if args.qat_checkpoint else "PTQ_only_control",
              "expert_int8_linears": expert_q, "vlm_int8_linears": vlm_q,
              "expert_group_size": 16, "vlm_group_size": None,
              "source_model_bytes": (args.model_dir / "model.safetensors").stat().st_size,
              "checkpoint_bytes": checkpoint.stat().st_size,
              "checkpoint_sha256": sha256(checkpoint),
              "reloaded": True}
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
