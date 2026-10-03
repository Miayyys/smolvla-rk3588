#!/usr/bin/env python3
"""Plot paired full-expert real-W8A8 actions and save task bootstrap statistics."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--plot", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    args = p.parse_args()
    report = json.loads(args.report.read_text())
    original = np.array(report["original_fp"]["metrics"]["task_first_action_mae_vs_recorded"])
    ptq = np.array(report["ptq"]["metrics"]["task_first_action_mae_vs_recorded"])
    qat = np.array(report["qat"]["metrics"]["task_first_action_mae_vs_recorded"])
    delta = qat - ptq
    rng = np.random.default_rng(0)
    draws = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(axis=1)
    summary = {"quantity": "QAT minus independent PTQ first action MAE versus recorded",
               "tasks": len(delta), "bootstrap_seed": 0, "bootstrap_replicates": 10000,
               "paired_mean": float(delta.mean()),
               "paired_ci95_percentile": np.quantile(draws, [.025, .975]).tolist(),
               "qat_better_tasks": int(np.count_nonzero(delta < 0)),
               "ptq_better_tasks": int(np.count_nonzero(delta > 0)),
               "paired_task_differences": delta.tolist(),
               "source_report": str(args.report), "plot": str(args.plot)}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    task = np.arange(len(delta))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained")
    for label, values in (("Original FP", original), ("Independent PTQ W8A8", ptq),
                          ("QAT converted W8A8", qat)):
        axes[0].plot(task, values, "o-", markersize=3, label=label)
    axes[0].set_ylabel("First action MAE vs recorded")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.2)
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].bar(task, delta, color=np.where(delta < 0, "#138a68", "#bb5959"))
    axes[1].set_xlabel("Frozen development task index (one episode per task)")
    axes[1].set_ylabel("QAT − PTQ MAE")
    axes[1].grid(axis="y", alpha=.2)
    fig.suptitle("112 real INT8 expert Linears on NVIDIA A10; 40 development tasks")
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
