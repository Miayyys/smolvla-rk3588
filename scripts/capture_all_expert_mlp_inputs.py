#!/usr/bin/env python3
"""Capture all expert MLP inputs in one pass over frozen calibration or development episodes."""

import argparse
import json
from pathlib import Path

import numpy as np

from probe_action_sensitivity import run_action, selected_frames, sha256
from qat_train_w8a8_stage1 import load_policy


PREFIX = "model.vlm_with_expert.lm_expert.layers"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "vlm-assets-dir", "dataset-root", "splits", "partition", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--split", choices=("calibration", "development"), required=True)
    parser.add_argument("--frames-per-task", type=int, default=2)
    parser.add_argument("--step-samples", type=int, default=3)
    args = parser.parse_args()
    if args.frames_per_task not in (1, 2) or not 1 <= args.step_samples <= 10:
        parser.error("frames-per-task must be 1 or 2 and step-samples must be 1..10")
    partition = json.loads(args.partition.read_text())
    if (partition["source_split_sha256"] != sha256(args.splits) or
            partition["source_weight_sha256"] != sha256(args.model_dir / "model.safetensors")):
        raise ValueError("Frozen model/split identity mismatch")
    ids = partition["calibration_episode_ids_from_ptq_calibration"] if args.split == "calibration" else partition["development_episode_ids_from_qat_train"]
    rows = selected_frames(args.dataset_root, ids, args.frames_per_task, 40)
    if len(rows) != 40 * args.frames_per_task:
        raise ValueError("Expected 40 tasks at specified frame count")
    policy, pre, post = load_policy(args)
    n_layers = len(policy.get_submodule(PREFIX))
    if n_layers != 16:
        raise ValueError(f"Expected 16 expert layers, got {n_layers}")
    captured = [[] for _ in range(n_layers)]
    handles = []
    for layer in range(n_layers):
        def capture(_module, inputs, layer=layer):
            captured[layer].append(inputs[0].detach().float().cpu().numpy())
        handles.append(policy.get_submodule(f"{PREFIX}.{layer}.mlp").register_forward_pre_hook(capture))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    lists = [[] for _ in range(n_layers)]
    try:
        for i, row in enumerate(rows):
            for values in captured:
                values.clear()
            run_action(policy, pre, post, row)
            if any(len(values) != 10 for values in captured):
                raise ValueError(f"Unexpected MLP call count at row {i}")
            indices = np.linspace(0, 9, args.step_samples, dtype=int)
            for layer, values in enumerate(captured):
                directory = args.output_dir / f"layer_{layer:02d}"
                directory.mkdir(exist_ok=True)
                for rank, index in enumerate(indices):
                    array = values[index]
                    if array.shape != (1, 50, 720) or not np.isfinite(array).all():
                        raise ValueError(f"Invalid layer {layer} input at row {i}: {array.shape}")
                    path = directory / (f"task_{row['task_index']:02d}_frame_{row['frame_rank']:02d}_step_{rank:02d}.npy")
                    np.save(path, array, allow_pickle=False)
                    lists[layer].append(str(path.resolve()))
                    records.append({"layer": layer, "task_index": row["task_index"],
                                    "episode_index": row["episode_index"],
                                    "frame_rank": row["frame_rank"], "step_rank": rank,
                                    "mlp_call_index": int(index), "file": str(path),
                                    "sha256": sha256(path)})
            if (i + 1) % 10 == 0:
                print(f"{args.split}: {i+1}/{len(rows)} action rows captured", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    name = "rknn_dataset.txt" if args.split == "calibration" else "heldout_inputs.txt"
    for layer, paths in enumerate(lists):
        (args.output_dir / f"layer_{layer:02d}" / name).write_text("\n".join(paths) + "\n")
    report = {"scope": "all 16 expert MLPs, original loaded policy",
              "split": args.split, "partition_sha256": sha256(args.partition),
              "source_weight_sha256": partition["source_weight_sha256"],
              "source_split_sha256": sha256(args.splits), "frames_per_task": args.frames_per_task,
              "step_samples": args.step_samples, "n_layers": n_layers,
              "samples_per_layer": len(lists[0]), "records": records}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {len(records)} inputs across {n_layers} layers", flush=True)


if __name__ == "__main__":
    main()
