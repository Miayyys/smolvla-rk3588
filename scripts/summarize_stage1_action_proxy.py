#!/usr/bin/env python3
"""Summarize 40-task paired action proxy without treating fake quant as deployment."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_difference(a, b):
    delta = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(0)
    bootstrap = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
    return {"mean": float(delta.mean()), "tasks_lower": int(np.sum(delta < 0)),
            "task_bootstrap_95_ci": np.quantile(bootstrap, [0.025, 0.975]).tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = json.loads(args.report.read_text())
    arrays_path = args.report.parent / "actions.npz"
    if sha256(arrays_path) != report["arrays_sha256"]:
        raise ValueError("Action archive hash mismatch")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if arrays["original"].shape != (40, 50, 7):
        raise ValueError("Unexpected action shape")
    if len(set(arrays["task_index"].tolist())) != 40:
        raise ValueError("Expected one action observation per task")
    first = {key: np.abs(arrays[key][:, 0, :] - arrays["recorded_first"]).mean(axis=1)
             for key in ("original", "ptq_fake", "qat_fake", "qat_master")}
    result = {"scope": "full action path, 40 development tasks, fake W8A8 diagnostic only",
              "mean_first_action_mae_vs_recorded": {key: float(value.mean()) for key, value in first.items()},
              "paired_first_action_mae": {
                  "qat_fake_minus_original": paired_difference(first["qat_fake"], first["original"]),
                  "qat_fake_minus_ptq_fake": paired_difference(first["qat_fake"], first["ptq_fake"]),
                  "ptq_fake_minus_original": paired_difference(first["ptq_fake"], first["original"]),
              },
              "chunk_mae_vs_original": {key: float(np.abs(arrays[key] - arrays["original"]).mean())
                                        for key in first},
              "per_task": [{"task": int(arrays["task_index"][i]),
                            "episode": int(arrays["episode_index"][i]),
                            **{f"{key}_first_action_mae": float(value[i]) for key, value in first.items()}}
                           for i in range(40)],
              "source_report": str(args.report), "actions_sha256": report["arrays_sha256"],
              "bootstrap_seed": 0, "bootstrap_replicates": 10000}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ids = arrays["task_index"]
    for key, label in (("original", "Original FP"), ("ptq_fake", "PTQ fake W8A8"),
                       ("qat_fake", "QAT fake W8A8")):
        ax.plot(ids, first[key], marker="o", markersize=3, linewidth=1, label=label)
    ax.set(xlabel="LIBERO development task index", ylabel="First-action MAE vs recorded action",
           title="40 paired development tasks; training quantizer only")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output.with_suffix(".png"), dpi=180)
    fig.savefig(args.output.with_suffix(".svg"))
    print(json.dumps({k: v for k, v in result.items() if k != "per_task"}))


if __name__ == "__main__":
    main()
