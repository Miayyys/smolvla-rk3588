#!/usr/bin/env python3
"""Plot paired action sensitivity for non-MLP SmolVLA module groups."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GROUPS = (
    ("vision_patch_projection", "Vision patch projection"),
    ("vision_attention_0_5", "Vision attention 0–5"),
    ("vision_attention_6_11", "Vision attention 6–11"),
    ("language_attention_0_7", "Language attention 0–7"),
    ("language_attention_8_15", "Language attention 8–15"),
    ("expert_attention_0_7", "Expert attention 0–7"),
    ("expert_attention_8_15", "Expert attention 8–15"),
    ("action_interface_projections", "State/action projections"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if set(report["groups"]) != {name for name, _ in GROUPS} or report["development_samples"] != 80:
        raise ValueError("Expected eight non-MLP groups and 80 paired development frames")
    summary = {"scope": report["scope"], "not_real_quantization": True,
               "development_samples": 80, "groups": {}}
    for name, label in GROUPS:
        group = report["groups"][name]
        if len(group["records"]) != 80:
            raise ValueError(f"Unexpected sample count for {name}")
        summary["groups"][name] = {
            "label": label,
            "mean_full_chunk_mae_vs_fp": group["mean_full_chunk_mae_vs_fp"],
            "mean_first_action_mae_vs_recorded_delta":
                group["mean_first_action_mae_vs_recorded_delta"],
            "module_parameter_count": group["module_parameter_count"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "non_mlp_action_sensitivity_v1.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(11.8, 6.4))
    labels = [label for _, label in GROUPS]
    values = [summary["groups"][name]["mean_full_chunk_mae_vs_fp"] for name, _ in GROUPS]
    colors = ["#4d87aa"] * 3 + ["#926aab"] * 2 + ["#538e73"] * 2 + ["#bb8050"]
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    xmax = max(values) * 1.22
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Mean full action chunk MAE vs original FP")
    ax.grid(axis="x", color="#e3e8ed")
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values[::-1]):
        ax.text(value + xmax * 0.012, bar.get_y() + bar.get_height() / 2,
                f"{value:.5f}", va="center")
    fig.subplots_adjust(top=0.78, bottom=0.17, left=0.24, right=0.96)
    fig.text(0.24, 0.96, "Non-MLP module output sensitivity", fontsize=18,
             fontweight="bold", va="top")
    fig.text(0.24, 0.90,
             "Static min/max INT8 output rounding · 40 calibration and 40 development episodes × 2 frames",
             color="#526170", fontsize=10, va="top")
    fig.text(0.24, 0.055,
             "Diagnostic fake quant only; weights, packed size, RKNN and closed-loop quality were not measured.",
             color="#526170", fontsize=9)
    for suffix in ("png", "svg"):
        fig.savefig(args.output_dir / f"non_mlp_action_sensitivity_v1.{suffix}",
                    dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps({name: summary["groups"][name]["mean_full_chunk_mae_vs_fp"]
                      for name, _ in GROUPS}), flush=True)


if __name__ == "__main__":
    main()
