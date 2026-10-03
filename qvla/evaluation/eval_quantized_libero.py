#!/usr/bin/env python3
"""Run a packed QVLA checkpoint through LeRobot's unchanged rollout/eval code.

Pass the usual lerobot-eval arguments plus --qvla-checkpoint=/path/to/mixed_w8.pt.
The checkpoint is produced locally by qvla/quantization/pack_mixed.py, not an arbitrary
third-party pickle.
"""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))


import hashlib
import sys
from pathlib import Path

import torch
from lerobot.scripts import lerobot_eval

from qvla.quantization.pack_mixed import build_quantized


def pop_checkpoint_arg() -> Path:
    matches = [arg for arg in sys.argv[1:] if arg.startswith("--qvla-checkpoint=")]
    if len(matches) != 1:
        raise SystemExit("Exactly one --qvla-checkpoint=/path/to/mixed_w8.pt is required")
    sys.argv.remove(matches[0])
    checkpoint = Path(matches[0].split("=", 1)[1]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    checkpoint = pop_checkpoint_arg()
    print(f"QVLA packed checkpoint: {checkpoint} sha256={sha256(checkpoint)}", flush=True)

    def make_quantized_policy(*, cfg, env_cfg, rename_map):
        if cfg.type != "smolvla" or cfg.pretrained_path is None:
            raise ValueError("Packed QVLA evaluation requires a staged SmolVLA --policy.path")
        model_dir = Path(cfg.pretrained_path)
        vlm_dir = Path(cfg.vlm_model_name)
        policy = build_quantized(model_dir, vlm_dir, None)
        # torchao tensor subclasses currently use torch.save/torch.load rather
        # than safetensors. Only load checkpoints produced by this project.
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        policy.load_state_dict(state, strict=True)
        policy.eval()
        return policy

    lerobot_eval.make_policy = make_quantized_policy
    lerobot_eval.main()


if __name__ == "__main__":
    main()
