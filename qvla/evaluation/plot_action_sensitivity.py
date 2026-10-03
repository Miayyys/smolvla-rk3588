#!/usr/bin/env python3
"""Plot quality-first SmolVLA development diagnostics from fake INT8 MLP outputs."""

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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ORDER = ["vision_0_5", "vision_6_11", "language_0_7", "language_8_15",
         "expert_0_7", "expert_8_15", "expert_0_only"]
COLORS = ["#4b79a7", "#4b79a7", "#9b70aa", "#9b70aa",
          "#138879", "#138879", "#2b6a60"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report["not_real_quantization"] is not True or report["not_closed_loop"] is not True:
        raise ValueError("Expected fake-quant, open-loop diagnostic report")
    if set(report["groups"]) != set(ORDER) or report["tasks"] != 40 or report["development_samples"] != 80:
        raise ValueError("Expected all seven groups and 80 development frames")
    groups = report["groups"]
    means = [groups[name]["mean_full_chunk_mae_vs_fp"] for name in ORDER]
    first = [groups[name]["mean_first_action_mae_vs_fp"] for name in ORDER]
    task_values = np.empty((len(ORDER), 40), dtype=float)
    for i, name in enumerate(ORDER):
        rows = groups[name]["records"]
        if len(rows) != 80:
            raise ValueError(f"Expected 80 records for {name}")
        for task in range(40):
            values = [row["full_chunk_mae_vs_fp"] for row in rows
                      if row["task_index"] == task]
            if len(values) != 2:
                raise ValueError(f"Expected two frames for task {task}, {name}")
            task_values[i, task] = np.mean(values)
    bootstrap_indices = np.random.default_rng(20260927).integers(0, 40, size=(10000, 40))
    task_bootstrap_ci = {name: np.quantile(
        task_values[i][bootstrap_indices].mean(axis=1), [0.025, 0.975]).tolist()
        for i, name in enumerate(ORDER)}
    summary = {"scope": report["scope"], "not_real_quantization": True,
               "not_closed_loop": True, "development_samples": 80,
               "task_bootstrap_resamples": 10000,
               "task_bootstrap_seed": 20260927,
               "task_bootstrap_mean_95pct_interval": task_bootstrap_ci,
               "mean_full_chunk_mae_vs_fp": dict(zip(ORDER, means)),
               "mean_first_action_mae_vs_fp": dict(zip(ORDER, first)),
               "per_task_full_chunk_mae_vs_fp": {
                   name: task_values[i].tolist() for i, name in enumerate(ORDER)}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "action_sensitivity_v1_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(15.5, 6.6),
                                     gridspec_kw={"width_ratios": [0.72, 1.5]})
    y = np.arange(len(ORDER))
    left.barh(y, means, color=COLORS, alpha=0.9)
    left.set_yticks(y, ORDER)
    left.invert_yaxis()
    left.set_xlabel("Mean full-chunk action MAE vs FP")
    left.grid(axis="x", color="#e3e8ed")
    left.set_axisbelow(True)
    left.set_xlim(0, max(means) * 1.26)
    for i, value in enumerate(means):
        left.text(value + max(means) * 0.02, i, f"{value:.4f}", va="center")

    im = right.imshow(task_values, cmap="YlOrRd", aspect="auto",
                      vmin=0, vmax=float(np.quantile(task_values, 0.95)))
    right.set_yticks(y, ORDER)
    right.set_xticks(range(0, 40, 5))
    right.set_xlabel("LIBERO task index (mean of first and last development frame)")
    right.set_ylabel("MLP group with output rounding")
    cbar = fig.colorbar(im, ax=right, pad=0.015)
    cbar.set_label("Full-chunk MAE vs FP")
    fig.subplots_adjust(top=0.77, bottom=0.15, left=0.13, right=0.96, wspace=0.23)
    fig.text(0.13, 0.96, "SmolVLA · action sensitivity by MLP group",
             fontsize=19, fontweight="bold", va="top")
    fig.text(0.13, 0.90,
             "Static min/max INT8 output rounding · 40 disjoint development episodes × 2 frames · paired noise",
             color="#526170", fontsize=10.5, va="top")
    fig.text(0.13, 0.055,
             "Diagnostic fake quant only. No packed model, RKNN execution, closed-loop success or board latency measured.",
             color="#526170", fontsize=9)
    for suffix in ("png", "svg"):
        fig.savefig(args.output_dir / f"action_sensitivity_v1.{suffix}",
                    dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps({"means": dict(zip(ORDER, means))}), flush=True)


if __name__ == "__main__":
    main()
