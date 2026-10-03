#!/usr/bin/env python3
"""Plot sampled calibration MSE scans beside paired full-action clipping results."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GROUPS = ("vision_6_11", "language_0_7")
LABELS = ("vision 6–11", "language 0–7")
COLORS = ("#3d78a4", "#9567a3")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minmax-report", type=Path, required=True)
    parser.add_argument("--mse-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.minmax_report.read_text())
    mse = json.loads(args.mse_report.read_text())
    for key in ("source_weight_sha256", "split_sha256", "partition_sha256",
                "calibration_episodes", "development_episodes", "frames_per_task",
                "development_samples"):
        if base[key] != mse[key]:
            raise ValueError(f"Different experiment inputs: {key}")
    if base["development_samples"] != 80 or set(mse["groups"]) != set(GROUPS):
        raise ValueError("Expected 80 paired development frames and two target groups")
    percentiles = [row["retained_fraction"] for row in next(iter(
        mse["quantizers"].values()))["calibration_candidate_scan"]]
    summary = {"scope": "sampled calibration reconstruction vs paired action error",
               "calibration_sampling": mse["mse_sampling"],
               "not_real_quantization": True, "groups": {}}
    for group in GROUPS:
        modules = mse["groups"][group]["modules"]
        scans = [mse["quantizers"][name]["calibration_candidate_scan"] for name in modules]
        if any([row["retained_fraction"] for row in scan] != percentiles for scan in scans):
            raise ValueError("Candidate percentiles differ across modules")
        ratios = [[row["sample_mse"] / scan[-1]["sample_mse"] for row in scan]
                  for scan in scans]
        base_rows = {(row["episode_index"], row["frame_index"]): row
                     for row in base["groups"][group]["records"]}
        pairs = [(base_rows[(row["episode_index"], row["frame_index"])]["full_chunk_mae_vs_fp"],
                  row["full_chunk_mae_vs_fp"]) for row in mse["groups"][group]["records"]]
        summary["groups"][group] = {
            "median_sample_mse_ratio_by_retained_fraction": np.median(ratios, axis=0).tolist(),
            "chosen_retained_fraction_per_module": {
                name: mse["quantizers"][name]["selected_retained_fraction"] for name in modules},
            "minmax_action_mae": base["groups"][group]["mean_full_chunk_mae_vs_fp"],
            "mse_action_mae": mse["groups"][group]["mean_full_chunk_mae_vs_fp"],
            "mse_better_frames": sum(new < old for old, new in pairs),
            "paired_frames": len(pairs),
            "minmax_recorded_action_delta": base["groups"][group]["mean_first_action_mae_vs_recorded_delta"],
            "mse_recorded_action_delta": mse["groups"][group]["mean_first_action_mae_vs_recorded_delta"]}
    summary["retained_fractions"] = percentiles
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "action_clipping_compare_v1.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.4, 5.5),
                                     gridspec_kw={"width_ratios": [1.15, 1]})
    for group, label, color in zip(GROUPS, LABELS, COLORS):
        values = summary["groups"][group]["median_sample_mse_ratio_by_retained_fraction"]
        left.plot(np.arange(len(percentiles)), values, color=color,
                  linewidth=2, marker="o", label=label)
    left.set_yscale("log")
    left.set_xlabel("Retained calibration fraction (symmetric tail quantiles)")
    left.set_ylabel("Median sampled activation MSE / min/max MSE")
    left.grid(color="#e3e8ed")
    left.legend(frameon=False)
    left.set_xticks(np.arange(len(percentiles)),
                    [f"{100*p:g}%" for p in percentiles], rotation=35)

    x = np.arange(len(GROUPS))
    width = 0.35
    normal = [summary["groups"][group]["minmax_action_mae"] for group in GROUPS]
    clipped = [summary["groups"][group]["mse_action_mae"] for group in GROUPS]
    right.bar(x - width/2, normal, width, label="min/max", color="#637585")
    right.bar(x + width/2, clipped, width, label="sampled MSE choice", color="#c47951")
    right.set_xticks(x, LABELS)
    right.set_ylabel("Mean full action-chunk MAE vs FP (80 development frames)")
    right.grid(axis="y", color="#e3e8ed")
    right.set_axisbelow(True)
    right.legend(frameon=False)
    for i, (a, b) in enumerate(zip(normal, clipped)):
        right.text(i-width/2, a+0.0004, f"{a:.4f}", ha="center", fontsize=9)
        right.text(i+width/2, b+0.0004, f"{b:.4f}", ha="center", fontsize=9)
    right.set_ylim(0, max(normal + clipped) * 1.23)
    fig.subplots_adjust(top=0.78, bottom=0.22, left=0.09, right=0.98, wspace=0.31)
    fig.text(0.09, 0.96, "Calibration MSE and action response can disagree",
             fontsize=18, fontweight="bold", va="top")
    fig.text(0.09, 0.90,
             "Same 40 development episodes · 2 frames each · output fake INT8, not RKNN or a packed model",
             color="#526170", fontsize=10.5, va="top")
    fig.text(0.09, 0.055,
             "Left: sampled calibration outputs; right: complete open-loop action chunks. Closed-loop task success unmeasured.",
             color="#526170", fontsize=9)
    for suffix in ("png", "svg"):
        fig.savefig(args.output_dir / f"action_clipping_compare_v1.{suffix}",
                    dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps({name: summary["groups"][name] for name in GROUPS}), flush=True)


if __name__ == "__main__":
    main()
