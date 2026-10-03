#!/usr/bin/env python3
"""Capture real SmolVLA MLP inputs from one episode per LIBERO task.

The default split is for RKNN calibration. Test captures are for held-out
numerical checks only and must never be supplied as a quantization dataset.
"""

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

import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--vlm-assets-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--partition", type=Path,
                        help="Frozen evaluation partition; limits capture to its 40 calibration/development episodes")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--module", default="model.vlm_with_expert.lm_expert.layers.0.mlp")
    parser.add_argument("--max-tasks", type=int, default=40)
    parser.add_argument("--split-name", choices=("qat_train", "ptq_calibration", "test"),
                        default="ptq_calibration")
    parser.add_argument("--frames-per-task", type=int, default=1)
    parser.add_argument("--step-samples", type=int, default=1,
                        help="Number of MLP calls to retain across one action generation")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.frames_per_task < 1 or args.step_samples < 1:
        parser.error("--frames-per-task and --step-samples must be positive")
    if not 1 <= args.max_tasks <= 40:
        parser.error("--max-tasks must be 1..40")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads(args.splits.read_text())
    partitions = {name: set(split["splits"][name])
                  for name in ("qat_train", "ptq_calibration", "test")}
    if any(partitions[left] & partitions[right]
           for left, right in (("qat_train", "ptq_calibration"),
                               ("qat_train", "test"),
                               ("ptq_calibration", "test"))):
        raise ValueError("Training, calibration, and test episodes overlap")
    selected_episodes = split["splits"][args.split_name]
    if args.partition:
        partition = json.loads(args.partition.read_text())
        if partition["source_split_sha256"] != sha256(args.splits):
            raise ValueError("Split hash differs from frozen partition")
        if partition["source_weight_sha256"] != sha256(args.model_dir / "model.safetensors"):
            raise ValueError("Weight hash differs from frozen partition")
        key = {"ptq_calibration": "calibration_episode_ids_from_ptq_calibration",
               "qat_train": "development_episode_ids_from_qat_train"}.get(args.split_name)
        if key is None:
            parser.error("Frozen test episodes must not be used for this probe")
        selected_episodes = partition[key]
        if not set(selected_episodes) <= set(split["splits"][args.split_name]):
            raise ValueError("Partition includes episodes outside requested split")

    lengths = {}
    for path in sorted((args.dataset_root / "meta/episodes").rglob("*.parquet")):
        for row in pq.read_table(path, columns=["episode_index", "length"]).to_pylist():
            lengths[row["episode_index"]] = row["length"]
    dataset = LeRobotDataset("lerobot/libero", root=args.dataset_root,
                             episodes=selected_episodes, video_backend="pyav", return_uint8=True)
    selected = {}
    offset = 0
    for episode in selected_episodes:
        raw = dataset.get_raw_item(offset)
        if int(raw["episode_index"]) != episode or int(raw["frame_index"]) != 0:
            raise ValueError(f"Unexpected calibration episode at dataset index {offset}")
        selected.setdefault(int(raw["task_index"]), (episode, offset, lengths[episode]))
        offset += lengths[episode]
    if len(selected) != len(split["task_counts"]) or offset != len(dataset):
        raise ValueError("Calibration dataset does not cover all tasks")

    config = SmolVLAConfig.from_pretrained(args.model_dir)
    config.device = "cuda"
    config.vlm_model_name = str(args.vlm_assets_dir.resolve())
    config.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(args.model_dir, config=config, strict=True).eval()
    preprocess, _ = make_pre_post_processors(
        config, str(args.model_dir),
        preprocessor_overrides={"tokenizer_processor": {
            "tokenizer_name": str(args.vlm_assets_dir.resolve())}},
    )
    module_name = args.module
    captured = []

    def capture(_module, inputs):
        if args.step_samples > 1 or not captured:
            captured.append(inputs[0].detach().float().cpu().numpy())

    handle = policy.get_submodule(module_name).register_forward_pre_hook(capture)
    records = []
    try:
        for task_index, (episode, dataset_index, episode_length) in sorted(selected.items()):
            if task_index >= args.max_tasks:
                continue
            frame_offsets = np.linspace(0, episode_length - 1,
                                        args.frames_per_task, dtype=int)
            if len(set(frame_offsets)) != args.frames_per_task:
                raise ValueError(f"Episode {episode} too short for frame sampling")
            for frame_rank, frame_offset in enumerate(frame_offsets):
                frame = dataset[dataset_index + int(frame_offset)]
                observation = {
                    "observation.state": frame["observation.state"],
                    "observation.images.image": frame["observation.images.image"].float() / 255.0,
                    "observation.images.image2": frame["observation.images.image2"].float() / 255.0,
                    "task": frame["task"],
                }
                sample_seed = (args.seed + 1009 * episode + 9176 * frame_rank
                               + 17 * task_index)
                torch.manual_seed(sample_seed)
                torch.cuda.manual_seed_all(sample_seed)
                policy.reset()
                captured.clear()
                with torch.inference_mode():
                    policy.select_action(preprocess(observation))
                if len(captured) < args.step_samples:
                    raise ValueError(f"Only {len(captured)} MLP calls for task {task_index}")
                step_indices = np.linspace(0, len(captured) - 1,
                                           args.step_samples, dtype=int)
                if len(set(step_indices)) != args.step_samples:
                    raise ValueError(f"Duplicate MLP call selection for task {task_index}")
                for step_rank, step_index in enumerate(step_indices):
                    activation = captured[step_index]
                    if not np.isfinite(activation).all():
                        raise ValueError(f"Invalid MLP activation for task {task_index}")
                    file_name = (f"task_{task_index:02d}.npy" if
                                 args.frames_per_task == args.step_samples == 1 else
                                 f"task_{task_index:02d}_frame_{frame_rank:02d}_step_{step_rank:02d}.npy")
                    path = args.output_dir / file_name
                    np.save(path, activation, allow_pickle=False)
                    records.append({"task_index": task_index, "episode_index": episode,
                                    "frame_index": int(frame_offset), "frame_rank": frame_rank,
                                    "step_rank": step_rank, "mlp_call_index": int(step_index),
                                    "mlp_call_count": len(captured), "noise_seed": sample_seed,
                                    "file": file_name, "shape": list(activation.shape),
                                    "min": float(activation.min()),
                                    "max": float(activation.max()),
                                    "sha256": sha256(path)})
                print(f"task {task_index}, frame {frame_offset}: "
                      f"saved {len(step_indices)}/{len(captured)} MLP calls", flush=True)
    finally:
        handle.remove()
    shapes = {tuple(record["shape"]) for record in records}
    if len(shapes) != 1:
        raise ValueError(f"MLP activation shape varies by task: {sorted(shapes)}")
    list_name = ("rknn_dataset.txt" if args.split_name == "ptq_calibration"
                 else "heldout_inputs.txt")
    (args.output_dir / list_name).write_text(
        "".join(str((args.output_dir / record["file"]).resolve()) + "\n" for record in records)
    )
    report = {"split": args.split_name, "samples": len(records),
              "frames_per_task": args.frames_per_task,
              "step_samples": args.step_samples, "base_seed": args.seed,
              "partition_sha256": sha256(args.partition) if args.partition else None,
              "module": module_name, "source_weight_sha256": sha256(args.model_dir / "model.safetensors"),
              "split_sha256": sha256(args.splits), "records": records}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {len(records)} isolated {args.split_name} activations; "
          f"shape={next(iter(shapes))}", flush=True)


if __name__ == "__main__":
    main()
