#!/usr/bin/env python3
"""Run a strictly reloaded real-W8A8 checkpoint in standard LeRobot LIBERO rollout."""

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
import random
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.scripts import lerobot_eval

from qvla.evaluation.probe_action_sensitivity import sha256
from qvla.quantization.real_int8_linear import replace_expert_linears
from qvla.quantization.mixed_int8 import replace_from_manifest


def pop_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--qvla-pack-report", type=Path, required=True)
    p.add_argument("--qvla-mode", choices=("fp", "ptq", "qat"), required=True)
    p.add_argument("--qvla-vlm-assets-dir", type=Path, required=True)
    ours, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    for i, token in enumerate(rest):
        if token == "--output_dir" and i + 1 < len(rest):
            Path(rest[i + 1]).mkdir(parents=True, exist_ok=True)
        elif token.startswith("--output_dir="):
            Path(token.split("=", 1)[1]).mkdir(parents=True, exist_ok=True)
    return ours


def main():
    args = pop_args()
    report = json.loads(args.qvla_pack_report.read_text())
    if args.qvla_mode == "fp":
        path = None
        print(json.dumps({"mode": "fp", "source_weight_sha256": report["source_weight_sha256"]}),
              flush=True)
    else:
        item = report["outputs"][args.qvla_mode]
        path = Path(item["path"]).resolve()
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError("Packed checkpoint hash/size mismatch")
        print(json.dumps({"mode": args.qvla_mode, "checkpoint": str(path),
                          "sha256": item["sha256"], "bytes": item["bytes"]}), flush=True)

    def make_quantized_policy(*, cfg, env_cfg, rename_map):
        if cfg.type != "smolvla" or cfg.pretrained_path is None:
            raise ValueError("Expected staged SmolVLA policy")
        cfg.vlm_model_name = str(args.qvla_vlm_assets_dir.resolve())
        cfg.load_vlm_weights = False
        cfg.device = "cuda"
        if args.qvla_mode == "fp":
            source = Path(cfg.pretrained_path) / "model.safetensors"
            if sha256(source) != report["source_weight_sha256"]:
                raise ValueError("FP source weight hash changed")
            return SmolVLAPolicy.from_pretrained(cfg.pretrained_path, config=cfg, strict=True).eval()
        policy = SmolVLAPolicy(cfg)
        names = (replace_from_manifest(policy, report["manifest"])
                 if report.get("format") == "mixed-int8-v1" else replace_expert_linears(policy))
        if len(names) != report["selected_linears"]:
            raise ValueError("Selected expert Linear count changed")
        state = load_file(str(path), device="cpu")
        policy.load_state_dict(state, strict=True)
        policy.to("cuda").eval()
        print(f"Strictly loaded {len(names)} real INT8 expert Linears", flush=True)
        return policy

    original_eval_policy_all = lerobot_eval.eval_policy_all
    original_processors = lerobot_eval.make_pre_post_processors
    original_run_one = lerobot_eval.run_one
    assets_path = str(args.qvla_vlm_assets_dir.resolve())

    def make_local_processors(*args, **kwargs):
        overrides = dict(kwargs.get("preprocessor_overrides") or {})
        tokenizer = dict(overrides.get("tokenizer_processor") or {})
        tokenizer["tokenizer_name"] = assets_path
        overrides["tokenizer_processor"] = tokenizer
        overrides["rename_observations_processor"] = {"rename_map": {
            "observation.images.image": "observation.images.camera1",
            "observation.images.image2": "observation.images.camera2",
        }}
        kwargs["preprocessor_overrides"] = overrides
        return original_processors(*args, **kwargs)

    def evaluate_without_video(*args, **kwargs):
        kwargs["max_episodes_rendered"] = 0
        kwargs["videos_dir"] = None
        return original_eval_policy_all(*args, **kwargs)

    def run_one_with_paired_noise(*run_args, **run_kwargs):
        task_group = run_kwargs.get("task_group", run_args[0] if run_args else None)
        task_id = run_kwargs.get("task_id", run_args[1] if len(run_args) > 1 else None)
        groups = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
        if task_group not in groups or task_id is None:
            raise ValueError(f"Unexpected LIBERO task for paired noise: {task_group}/{task_id}")
        base_seed = int(run_kwargs.get("start_seed") or 0)
        task_seed = base_seed + 100000 * (groups.index(task_group) + 1) + int(task_id)
        random.seed(task_seed)
        np.random.seed(task_seed)
        torch.manual_seed(task_seed)
        torch.cuda.manual_seed_all(task_seed)
        print(f"paired policy noise seed {task_group}/{task_id}: {task_seed}", flush=True)
        return original_run_one(*run_args, **run_kwargs)

    lerobot_eval.make_policy = make_quantized_policy
    lerobot_eval.eval_policy_all = evaluate_without_video
    lerobot_eval.run_one = run_one_with_paired_noise
    lerobot_eval.make_pre_post_processors = make_local_processors
    lerobot_eval.main()


if __name__ == "__main__":
    main()
