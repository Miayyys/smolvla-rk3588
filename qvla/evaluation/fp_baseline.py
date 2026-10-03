#!/usr/bin/env python3
"""Evaluate one frozen test episode per LIBERO task with the FP SmolVLA policy."""

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
import statistics
import time
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--vlm-assets-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--mixed-checkpoint", type=Path,
                        help="Evaluate a saved real W8 mixed checkpoint on the same frozen samples")
    args = parser.parse_args()

    weight_hash = sha256(args.model_dir / "model.safetensors")
    if weight_hash != args.expected_weight_sha256:
        raise ValueError(f"Model hash mismatch: {weight_hash}")
    split = json.loads(args.splits.read_text())
    test_episodes = split["splits"]["test"]
    length_by_episode = {}
    for path in sorted((args.dataset_root / "meta/episodes").rglob("*.parquet")):
        for row in pq.read_table(path, columns=["episode_index", "length"]).to_pylist():
            length_by_episode[row["episode_index"]] = row["length"]

    dataset = LeRobotDataset("lerobot/libero", root=args.dataset_root,
                             episodes=test_episodes, video_backend="pyav", return_uint8=True)
    if len(dataset) != sum(length_by_episode[episode] for episode in test_episodes):
        raise ValueError("Selected episode frame count mismatch")
    selected = {}
    offset = 0
    for episode in test_episodes:
        raw = dataset.get_raw_item(offset)
        if int(raw["episode_index"]) != episode or int(raw["frame_index"]) != 0:
            raise ValueError(f"Unexpected episode order at index {offset}: {raw}")
        task_index = int(raw["task_index"])
        selected.setdefault(task_index, (episode, offset))
        offset += length_by_episode[episode]
    if len(selected) != len(split["task_counts"]):
        raise ValueError(f"Expected {len(split['task_counts'])} tasks, got {len(selected)}")

    config = SmolVLAConfig.from_pretrained(args.model_dir)
    config.device = "cuda"
    config.vlm_model_name = str(args.vlm_assets_dir.resolve())
    config.load_vlm_weights = False
    if args.mixed_checkpoint:
        from qvla.quantization.pack_mixed import build_quantized

        policy = build_quantized(args.model_dir, args.vlm_assets_dir, None)
        policy.load_state_dict(torch.load(args.mixed_checkpoint, map_location="cpu", weights_only=False),
                               strict=True)
    else:
        policy = SmolVLAPolicy.from_pretrained(args.model_dir, config=config, strict=True)
    preprocess, postprocess = make_pre_post_processors(
        config, str(args.model_dir),
        preprocessor_overrides={"tokenizer_processor": {
            "tokenizer_name": str(args.vlm_assets_dir.resolve())}},
    )
    policy.eval()
    records = []
    for task_index, (episode, dataset_index) in sorted(selected.items()):
        frame = dataset[dataset_index]
        observation = {
            "observation.state": frame["observation.state"],
            "observation.images.image": frame["observation.images.image"].float() / 255.0,
            "observation.images.image2": frame["observation.images.image2"].float() / 255.0,
            "task": frame["task"],
        }
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        policy.reset()
        batch = preprocess(observation)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            action = postprocess(policy.select_action(batch))
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000
        prediction = action.detach().cpu().numpy().reshape(-1)
        target = frame["action"].detach().cpu().numpy().reshape(-1)
        if prediction.shape != (7,) or not np.isfinite(prediction).all():
            raise ValueError(f"Invalid action for task {task_index}: {prediction}")
        records.append({"task_index": task_index, "episode_index": episode,
                        "frame_index": 0, "task": frame["task"],
                        "action": prediction.tolist(), "recorded_action": target.tolist(),
                        "action_mae_vs_recorded": float(np.mean(np.abs(prediction - target))),
                        "latency_ms": latency_ms})
        print(f"task {task_index}: episode {episode}, MAE {records[-1]['action_mae_vs_recorded']:.4f}, "
              f"latency {latency_ms:.1f} ms", flush=True)
    latencies = [row["latency_ms"] for row in records]
    report = {"model_weight_sha256": weight_hash, "split_sha256": sha256(args.splits),
              "method": "mixed_W8" if args.mixed_checkpoint else "FP",
              "checkpoint_sha256": sha256(args.mixed_checkpoint) if args.mixed_checkpoint else weight_hash,
              "checkpoint_bytes": args.mixed_checkpoint.stat().st_size if args.mixed_checkpoint
              else (args.model_dir / "model.safetensors").stat().st_size,
              "split": "test", "samples": len(records), "tasks": len(selected),
              "seed_per_sample": 0, "torch": torch.__version__, "cuda": torch.version.cuda,
              "mean_action_mae_vs_recorded": statistics.mean(
                  row["action_mae_vs_recorded"] for row in records),
              "latency_p50_ms": float(np.percentile(latencies, 50)),
              "latency_p95_ms": float(np.percentile(latencies, 95)),
              "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
              "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
