#!/usr/bin/env python3
"""Plot paired RKNN simulator errors for three INT8 calibration methods."""

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


METHODS = [
    ("fp16", "FP16", "#65717d"),
    ("int8_normal", "INT8 normal", "#3f6fa8"),
    ("int8_kl_divergence", "INT8 KL", "#bd7653"),
    ("int8_mmse", "INT8 MMSE", "#14866d"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--heldout-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-name", choices=("test", "qat_train"), default="test")
    parser.add_argument("--report-label", default="heldout")
    parser.add_argument("--basename", default="rknn_mlp_calibration_v2")
    parser.add_argument("--summary-name", default="calibration_compare.json")
    args = parser.parse_args()
    heldout = json.loads((args.heldout_dir / "report.json").read_text())
    by_file = {row["file"]: row for row in heldout["records"]}
    if heldout["split"] != args.split_name or len(by_file) != 240:
        raise ValueError("Expected 240 isolated held-out activations")
    reports = {}
    for name, _, _ in METHODS:
        report = json.loads((args.probe_dir / f"{name}_{args.report_label}_parity.json").read_text())
        if report["samples"] != 240 or {r["input"] for r in report["records"]} != set(by_file):
            raise ValueError(f"Different held-out samples for {name}")
        reports[name] = report

    group_keys = sorted({(row["frame_rank"], row["step_rank"])
                         for row in heldout["records"]})
    summary = {"scope": "one MLP; host simulator numerical error vs FP32 ONNX",
               "split": args.split_name,
               "heldout_samples": 240, "tasks": 40,
               "methods": {name: {"mean_mae": reports[name]["mean_mae"],
                                  "min_cosine": reports[name]["min_cosine"],
                                  "group_mean_mae": {f"frame{f}_step{s}": float(np.mean([
                                      row["mae"] for row in reports[name]["records"]
                                      if (by_file[row["input"]]["frame_rank"],
                                          by_file[row["input"]]["step_rank"]) == (f, s)]))
                                      for f, s in group_keys}}
                           for name, _, _ in METHODS}}
    task_means = {}
    for name, _, _ in METHODS:
        task_means[name] = {task: float(np.mean([
            row["mae"] for row in reports[name]["records"]
            if by_file[row["input"]]["task_index"] == task])) for task in range(40)}
    summary["mmse_better_than_normal_tasks"] = sum(
        task_means["int8_mmse"][task] < task_means["int8_normal"][task]
        for task in range(40))
    summary["mmse_better_than_kl_tasks"] = sum(
        task_means["int8_mmse"][task] < task_means["int8_kl_divergence"][task]
        for task in range(40))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / args.summary_name).write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.7, 5.8),
                                     gridspec_kw={"width_ratios": [0.85, 1.25]})
    fig.patch.set_facecolor("white")
    for i, (name, label, color) in enumerate(METHODS):
        value = reports[name]["mean_mae"]
        left.bar(i, value, color=color, width=0.66)
        left.text(i, value * 1.11, f"{value:.5f}", ha="center", va="bottom", fontsize=9)
    left.set_xticks(range(4), [label for _, label, _ in METHODS], rotation=15)
    left.set_yscale("log")
    left.set_ylim(4e-5, 3e-2)
    left.set_ylabel("Mean output MAE vs FP32 ONNX (log scale)")
    left.grid(axis="y", color="#e5eaed")
    left.set_axisbelow(True)
    labels = [f"F{f} · S{s}" for f, s in group_keys]
    x = np.arange(len(group_keys))
    for name, label, color in METHODS[1:]:
        values = [summary["methods"][name]["group_mean_mae"][f"frame{f}_step{s}"]
                  for f, s in group_keys]
        right.plot(x, values, marker="o", linewidth=1.8, color=color, label=label)
    right.set_xticks(x, labels)
    right.set_ylabel("Mean output MAE vs FP32 ONNX")
    right.set_xlabel("Frame and selected MLP call (40 tasks each)")
    right.grid(axis="y", color="#e5eaed")
    right.legend(frameon=False)
    fig.subplots_adjust(top=0.79, bottom=0.2, left=0.09, right=0.97, wspace=0.25)
    fig.text(0.09, 0.96, "RK3588 MLP · calibration comparison", fontsize=19,
             fontweight="bold", va="top")
    fig.text(0.09, 0.905, f"240 {args.split_name} activations · RKNN 2.3.2 host simulator · same ONNX and input split",
             color="#526170", fontsize=11, va="top")
    fig.text(0.09, 0.06,
             "F0/F1: first/last sampled episode frame. S0/S1/S2: first/middle/last of 10 MLP calls. "
             "Board execution and action quality are unmeasured.",
             color="#526170", fontsize=8.5)
    for suffix in ("png", "svg"):
        fig.savefig(args.output_dir / f"{args.basename}.{suffix}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps({"means": {n: summary["methods"][n]["mean_mae"]
                                for n, _, _ in METHODS},
                      "mmse_better_than_normal_tasks": summary["mmse_better_than_normal_tasks"],
                      "mmse_better_than_kl_tasks": summary["mmse_better_than_kl_tasks"]}), flush=True)


if __name__ == "__main__":
    main()
