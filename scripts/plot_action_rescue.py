#!/usr/bin/env python3
"""Plot action error when sensitive MLP outputs remain FP in a fake-INT8 probe."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-report", type=Path, required=True)
    parser.add_argument("--layer-report", type=Path, required=True)
    parser.add_argument("--rescue-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    original, single, rescued = [json.loads(path.read_text()) for path in
                                 (args.group_report, args.layer_report, args.rescue_report)]
    for key in ("source_weight_sha256", "split_sha256", "partition_sha256",
                "calibration_episodes", "development_episodes", "frames_per_task",
                "development_samples"):
        if len({json.dumps(report[key], sort_keys=True) for report in (original, single, rescued)}) != 1:
            raise ValueError(f"Inputs differ: {key}")
    pairs = [
        ("Vision MLP 6–11", "vision_6_11", "vision_11_only", "vision_6_10"),
        ("Language MLP 0–7", "language_0_7", "language_3_only", "language_0_2_4_7"),
    ]
    summary = {"scope": "paired full-action-chunk error under output fake INT8",
               "not_real_quantization": True, "not_closed_loop": True,
               "development_samples": 80, "groups": {}}
    for title, all_name, sensitive_name, rest_name in pairs:
        summary["groups"][all_name] = {
            "title": title,
            "all_layers_mae": original["groups"][all_name]["mean_full_chunk_mae_vs_fp"],
            "sensitive_layer_mae": single["groups"][sensitive_name]["mean_full_chunk_mae_vs_fp"],
            "other_layers_mae": rescued["groups"][rest_name]["mean_full_chunk_mae_vs_fp"],
            "sensitive_layer": sensitive_name,
            "other_layers": rest_name}
    summary["combined_12_layer_mae"] = rescued["groups"][
        "vision_6_10_language_without_3"]["mean_full_chunk_mae_vs_fp"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "action_rescue_v1.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(10, 5.7))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    x = np.arange(2)
    width = 0.24
    keys = (("all_layers_mae", "all layers rounded", "#b25c53"),
            ("sensitive_layer_mae", "sensitive layer only", "#d29150"),
            ("other_layers_mae", "other layers; sensitive stays FP", "#198575"))
    for i, (key, label, color) in enumerate(keys):
        values = [summary["groups"][row[1]][key] for row in pairs]
        bars = ax.bar(x + (i - 1) * width, values, width, color=color, label=label)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, value+0.0005,
                    f"{value:.4f}", ha="center", fontsize=9)
    ax.set_xticks(x, [row[0] for row in pairs])
    ax.set_ylabel("Mean full action-chunk MAE vs FP")
    ax.set_ylim(0, 0.037)
    ax.grid(axis="y", color="#e3e8ed")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.subplots_adjust(top=0.77, bottom=0.17, left=0.12, right=0.98)
    fig.text(0.12, 0.96, "Keeping two sensitive MLP outputs in FP",
             fontsize=18, fontweight="bold", va="top")
    fig.text(0.12, 0.9,
             "40 development episodes × 2 frames · paired noise · static min/max output fake INT8",
             color="#526170", fontsize=10.5, va="top")
    fig.text(0.12, 0.05,
             "The two green groups together have MAE %.4f. No packed model, closed-loop or board resource result." %
             summary["combined_12_layer_mae"], color="#526170", fontsize=9)
    for suffix in ("png", "svg"):
        fig.savefig(args.output_dir / f"action_rescue_v1.{suffix}",
                    dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps(summary["groups"]), flush=True)


if __name__ == "__main__":
    main()
