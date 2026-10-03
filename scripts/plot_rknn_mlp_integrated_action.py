#!/usr/bin/env python3
"""Plot paired development action errors from one-MLP RKNN integration."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    ptq = report["ptq"]["scores"]
    qat = report["qat"]["scores"]
    baseline = np.asarray(report["baseline_task_first_action_mae_vs_recorded"])
    ptq_first = np.asarray(ptq["task_first_action_mae_vs_recorded"])
    qat_first = np.asarray(qat["task_first_action_mae_vs_recorded"])
    task = np.arange(len(baseline))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained")
    axes[0].plot(task, baseline, "o-", markersize=3, label="Original FP")
    axes[0].plot(task, ptq_first, "o-", markersize=3, label="PTQ: one RKNN INT8 MLP")
    axes[0].plot(task, qat_first, "o-", markersize=3, label="QAT: one RKNN INT8 MLP")
    axes[0].set_ylabel("First action MAE vs recorded")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.25)
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].bar(task, qat_first - ptq_first, color=np.where(qat_first < ptq_first, "#138a68", "#bb5959"))
    axes[1].set_ylabel("QAT − PTQ first action MAE")
    axes[1].set_xlabel("Frozen development task index (one episode per task)")
    axes[1].grid(axis="y", alpha=.25)
    fig.suptitle("One RKNN W8A8 expert MLP in full SmolVLA action path (host simulator)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)
    differences = qat_first - ptq_first
    rng = np.random.default_rng(0)
    samples = rng.integers(0, len(differences), size=(10000, len(differences)))
    bootstrap = differences[samples].mean(axis=1)
    summary = {
        "quantity": "QAT minus PTQ first action MAE versus recorded, per task",
        "tasks": len(differences), "bootstrap_seed": 0, "bootstrap_replicates": 10000,
        "paired_mean": float(differences.mean()),
        "paired_ci95_percentile": np.quantile(bootstrap, [.025, .975]).tolist(),
        "qat_better_tasks": int(np.count_nonzero(differences < 0)),
        "ptq_better_tasks": int(np.count_nonzero(differences > 0)),
        "paired_task_differences": differences.tolist(),
        "plot": str(args.output), "source_report": str(args.report)}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
