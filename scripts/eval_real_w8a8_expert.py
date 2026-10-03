#!/usr/bin/env python3
"""Strictly reload packed PTQ/QAT full checkpoints and compare 40 development actions."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

from probe_action_sensitivity import run_action, selected_frames, sha256
from qat_train_w8a8_stage1 import load_policy
from real_int8_linear import replace_expert_linears


def action_metrics(actions, original, rows):
    delta = np.abs(actions - original)
    first = np.array([np.abs(actions[i, 0] - row["recorded_action"]).mean()
                      for i, row in enumerate(rows)])
    return {"chunk_mae_vs_original_fp": float(delta.mean()),
            "task_chunk_mae_vs_original_fp": delta.mean(axis=(1, 2)).tolist(),
            "first_action_mae_vs_recorded": float(first.mean()),
            "task_first_action_mae_vs_recorded": first.tolist()}


def evaluate(policy, pre, post, rows, label):
    actions = []
    times = []
    for i, row in enumerate(rows):
        torch.cuda.synchronize()
        start = time.perf_counter()
        actions.append(run_action(policy, pre, post, row))
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        if (i + 1) % 10 == 0:
            print(f"{label} action {i+1}/{len(rows)}", flush=True)
    return np.stack(actions), {"per_action_seconds": times,
                               "mean_seconds": float(np.mean(times)),
                               "p50_seconds": float(np.percentile(times, 50)),
                               "p95_seconds": float(np.percentile(times, 95))}


def load_packed(args, path):
    config = SmolVLAConfig.from_pretrained(args.model_dir)
    config.device = "cuda"
    config.vlm_model_name = str(args.vlm_assets_dir.resolve())
    config.load_vlm_weights = False
    policy = SmolVLAPolicy(config)
    names = replace_expert_linears(policy)
    state = load_file(str(path), device="cpu")
    policy.load_state_dict(state, strict=True)
    del state
    policy.to("cuda").eval()
    pre, post = make_pre_post_processors(
        config, str(args.model_dir),
        preprocessor_overrides={"tokenizer_processor": {
            "tokenizer_name": str(args.vlm_assets_dir.resolve())}},
    )
    return policy, pre, post, names


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "vlm-assets-dir", "dataset-root", "splits", "partition",
                 "map", "pack-report", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--max-tasks", type=int, default=40)
    args = p.parse_args()
    if not 1 <= args.max_tasks <= 40:
        p.error("--max-tasks must be 1..40")
    partition = json.loads(args.partition.read_text())
    pack = json.loads(args.pack_report.read_text())
    if (partition["source_split_sha256"] != sha256(args.splits)
            or partition["source_weight_sha256"] != sha256(args.model_dir / "model.safetensors")
            or pack["partition_sha256"] != sha256(args.partition)
            or pack["map_sha256"] != sha256(args.map)):
        raise ValueError("Frozen model/split/map/packed identity mismatch")
    rows = selected_frames(args.dataset_root,
                           partition["development_episode_ids_from_qat_train"], 1, 40)[:args.max_tasks]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_policy, pre, post = load_policy(args)
    original_policy.to("cuda").eval()
    original, original_timing = evaluate(original_policy, pre, post, rows, "original FP")
    del original_policy
    torch.cuda.empty_cache()
    results = {}
    arrays = {"original": original,
              "task_index": np.array([row["task_index"] for row in rows]),
              "episode_index": np.array([row["episode_index"] for row in rows]),
              "recorded_first": np.stack([row["recorded_action"] for row in rows])}
    for mode in ("ptq", "qat"):
        item = pack["outputs"][mode]
        path = Path(item["path"])
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Packed checkpoint identity changed: {mode}")
        policy, pre, post, names = load_packed(args, path)
        actions, timing = evaluate(policy, pre, post, rows, mode)
        arrays[mode] = actions
        results[mode] = {"checkpoint": str(path), "checkpoint_sha256": item["sha256"],
                         "checkpoint_bytes": item["bytes"], "quantized_linears": len(names),
                         "metrics": action_metrics(actions, original, rows), "timing": timing}
        del policy
        torch.cuda.empty_cache()
    npz = args.output_dir / "actions.npz"
    np.savez_compressed(npz, **arrays)
    report = {"scope": "full action with 112 expert Linear real CUDA INT8 GEMMs, other modules original dtype",
              "board_execution": "not_measured", "closed_loop": "not_measured",
              "source_weight_sha256": partition["source_weight_sha256"],
              "partition_sha256": sha256(args.partition), "pack_report_sha256": sha256(args.pack_report),
              "development_episode_ids": [row["episode_index"] for row in rows],
              "tasks": len(rows), "torch_version": torch.__version__,
              "original_fp": {"metrics": action_metrics(original, original, rows),
                              "timing": original_timing},
              "ptq": results["ptq"], "qat": results["qat"],
              "actions": str(npz), "actions_sha256": sha256(npz)}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({mode: {"first_action_mae_vs_recorded": report[mode]["metrics"]["first_action_mae_vs_recorded"],
                             "chunk_mae_vs_original_fp": report[mode]["metrics"]["chunk_mae_vs_original_fp"],
                             "p50_seconds": report[mode]["timing"]["p50_seconds"]}
                      for mode in ("original_fp", "ptq", "qat")}), flush=True)


if __name__ == "__main__":
    main()
