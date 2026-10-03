#!/usr/bin/env python3
"""Plot individual vision and language MLP action sensitivities."""

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


GROUPS = ([f"vision_{i}_only" for i in range(6, 12)] +
          [f"language_{i}_only" for i in range(8)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if set(report["groups"]) != set(GROUPS) or report["development_samples"] != 80:
        raise ValueError("Expected 14 individual layers and 80 paired development frames")
    summary = {"scope": report["scope"], "not_real_quantization": True,
               "development_samples": 80, "groups": {}}
    for name in GROUPS:
        group = report["groups"][name]
        if len(group["modules"]) != 1 or len(group["records"]) != 80:
            raise ValueError(f"Unexpected group shape for {name}")
        summary["groups"][name] = {
            "mean_full_chunk_mae_vs_fp": group["mean_full_chunk_mae_vs_fp"],
            "mean_first_action_mae_vs_recorded_delta":
                group["mean_first_action_mae_vs_recorded_delta"],
            "parameter_count": group["module_parameter_count"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "action_layer_sensitivity_v1.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.7))
    for ax, names, color, title in (
        (axes[0], GROUPS[:6], "#3d78a4", "Vision MLP layers 6–11"),
        (axes[1], GROUPS[6:], "#9567a3", "Language MLP layers 0–7")):
        values = [summary["groups"][name]["mean_full_chunk_mae_vs_fp"] for name in names]
        labels = [name.split("_")[1] for name in names]
        bars = ax.bar(labels, values, color=color, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Mean full action-chunk MAE vs FP")
        ax.grid(axis="y", color="#e3e8ed")
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(values) * 1.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value + max(values)*0.025,
                    f"{value:.4f}", ha="center", fontsize=8)
    fig.subplots_adjust(top=0.77, bottom=0.18, left=0.09, right=0.97, wspace=0.25)
    fig.text(0.09, 0.96, "Action sensitivity concentrates in two MLP layers",
             fontsize=18, fontweight="bold", va="top")
    fig.text(0.09, 0.90,
             "Static min/max output INT8 rounding · 40 development episodes × 2 frames · identical FP noise",
             color="#526170", fontsize=10.5, va="top")
    fig.text(0.09, 0.055,
             "Layer-wise fake quant only; no packed model, RKNN board execution or closed-loop task quality measured.",
             color="#526170", fontsize=9)
    for suffix in ("png", "svg"):
        fig.savefig(args.output_dir / f"action_layer_sensitivity_v1.{suffix}",
                    dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps({name: summary["groups"][name]["mean_full_chunk_mae_vs_fp"]
                      for name in GROUPS}), flush=True)


if __name__ == "__main__":
    main()
