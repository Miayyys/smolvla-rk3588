#!/usr/bin/env python3
"""Create reproducible, task-stratified episode splits for QAT, calibration and test."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = args.dataset_root
    episode_rows = []
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        episode_rows.extend(pq.read_table(path, columns=["episode_index", "length"]).to_pylist())
    expected = {row["episode_index"] for row in episode_rows}
    if len(expected) != len(episode_rows):
        raise ValueError("Duplicate episode indices in metadata")

    episode_tasks = {}
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "task_index"])
        for row in table.to_pylist():
            episode = row["episode_index"]
            task = row["task_index"]
            previous = episode_tasks.setdefault(episode, task)
            if previous != task:
                raise ValueError(f"Episode {episode} contains multiple tasks")
    if set(episode_tasks) != expected:
        raise ValueError(f"Episode metadata/data mismatch: {len(expected)} vs {len(episode_tasks)}")

    groups = defaultdict(list)
    for episode, task in episode_tasks.items():
        groups[task].append(episode)
    splits = {"qat_train": [], "ptq_calibration": [], "test": []}
    counts = {}
    for task in sorted(groups):
        episodes = sorted(groups[task])
        random.Random(args.seed + task).shuffle(episodes)
        count = len(episodes)
        if count < 3:
            raise ValueError(f"Task {task} has fewer than 3 episodes")
        n_test = max(1, round(count * 0.15))
        n_cal = max(1, round(count * 0.15))
        splits["test"].extend(episodes[:n_test])
        splits["ptq_calibration"].extend(episodes[n_test:n_test + n_cal])
        splits["qat_train"].extend(episodes[n_test + n_cal:])
        counts[str(task)] = {"total": count, "qat_train": count - n_test - n_cal,
                             "ptq_calibration": n_cal, "test": n_test}
    splits = {name: sorted(episodes) for name, episodes in splits.items()}
    if sum(len(value) for value in splits.values()) != len(expected):
        raise ValueError("Split coverage mismatch")
    result = {"seed": args.seed, "dataset_repo": "lerobot/libero",
              "dataset_revision": "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4",
              "splits": splits, "task_counts": counts}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"episodes": len(expected), "tasks": len(groups),
                      "split_sizes": {key: len(value) for key, value in splits.items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
