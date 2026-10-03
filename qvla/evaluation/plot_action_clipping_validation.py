#!/usr/bin/env python3
"""Show why sampled activation MSE selected invalid clipping for SmolVLA."""

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


GROUPS = ("vision_6_11", "language_0_7")
LABELS = ("vision 6–11", "language 0–7")
COLORS = ("#3d78a4", "#9567a3")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minmax-report", type=Path, required=True)
    parser.add_argument("--sampled-report", type=Path, required=True)
    parser.add_argument("--exact-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in
               (args.minmax_report, args.sampled_report, args.exact_report)]
    base, sampled, exact = reports
    for key in ("source_weight_sha256", "split_sha256", "partition_sha256",
                "calibration_episodes", "development_episodes", "frames_per_task",
                "development_samples"):
        if len({json.dumps(report[key], sort_keys=True) for report in reports}) != 1:
            raise ValueError(f"Different experiment inputs: {key}")
    if base["development_samples"] != 80 or set(sampled["groups"]) != set(GROUPS) or set(exact["groups"]) != set(GROUPS):
        raise ValueError("Expected same 80 samples and two target groups")
    percentiles = [row["retained_fraction"] for row in next(iter(
        sampled["quantizers"].values()))["calibration_candidate_scan"]]
    summary = {"scope": "sampled vs all-elements calibration MSE and action response",
               "not_real_quantization": True, "retained_fractions": percentiles,
               "groups": {}}
    for group in GROUPS:
        modules = sampled["groups"][group]["modules"]
        sampled_ratios, exact_ratios = [], []
        for name in modules:
            a = sampled["quantizers"][name]["calibration_candidate_scan"]
            b = exact["quantizers"][name]["full_calibration_candidate_scan"]
            if [r["retained_fraction"] for r in a] != percentiles or [r["retained_fraction"] for r in b] != percentiles:
                raise ValueError("Different candidate grids")
            sampled_ratios.append([row["sample_mse"] / a[-1]["sample_mse"] for row in a])
            exact_ratios.append([row["full_calibration_mse"] / b[-1]["full_calibration_mse"] for row in b])
        summary["groups"][group] = {
            "sampled_median_mse_ratio": np.median(sampled_ratios, axis=0).tolist(),
            "exact_median_mse_ratio": np.median(exact_ratios, axis=0).tolist(),
            "sampled_selected_fraction": {n: sampled["quantizers"][n]["selected_retained_fraction"] for n in modules},
            "exact_selected_fraction": {n: exact["quantizers"][n]["selected_retained_fraction"] for n in modules},
            "action_mae": {"minmax": base["groups"][group]["mean_full_chunk_mae_vs_fp"],
                           "sampled_mse": sampled["groups"][group]["mean_full_chunk_mae_vs_fp"],
                           "exact_mse": exact["groups"][group]["mean_full_chunk_mae_vs_fp"]}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "action_clipping_validation_v1.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6),
                             gridspec_kw={"width_ratios": [1, 1, 1.05]})
    x = np.arange(len(percentiles))
    for group, label, color in zip(GROUPS, LABELS, COLORS):
        axes[0].plot(x, summary["groups"][group]["sampled_median_mse_ratio"],
                     marker="o", color=color, linewidth=1.8, label=label)
        axes[1].plot(x, summary["groups"][group]["exact_median_mse_ratio"],
                     marker="o", color=color, linewidth=1.8, label=label)
    for ax, title in zip(axes[:2], ("Sampled calibration values", "All calibration values")):
        ax.set_title(title, fontsize=12)
        ax.set_yscale("log")
        ax.set_xticks(x, [f"{100*p:g}%" for p in percentiles], rotation=35)
        ax.set_xlabel("Retained fraction")
        ax.set_ylabel("Median activation MSE / min/max MSE")
        ax.grid(color="#e3e8ed")
    axes[0].legend(frameon=False)
    width = 0.23
    bars = [("minmax", "min/max", "#627487"),
            ("sampled_mse", "sampled MSE", "#d27c51"),
            ("exact_mse", "exact MSE", "#168678")]
    for i, (key, label, color) in enumerate(bars):
        values = [summary["groups"][group]["action_mae"][key] for group in GROUPS]
        axes[2].bar(np.arange(2) + (i - 1) * width, values, width,
                    color=color, label=label)
    axes[2].set_title("Paired development actions", fontsize=12)
    axes[2].set_xticks(range(2), LABELS)
    axes[2].set_ylabel("Mean full action-chunk MAE vs FP")
    axes[2].grid(axis="y", color="#e3e8ed")
    axes[2].set_axisbelow(True)
    axes[2].legend(frameon=False)
    fig.subplots_adjust(top=0.77, bottom=0.2, left=0.07, right=0.98, wspace=0.3)
    fig.text(0.07, 0.96, "Sampled MSE picked thresholds that failed full calibration",
             fontsize=18, fontweight="bold", va="top")
    fig.text(0.07, 0.905,
             "Same 40 calibration and 40 development episodes · 2 frames each · output fake INT8",
             color="#526170", fontsize=10.5, va="top")
    fig.text(0.07, 0.05,
             "Exact MSE was measured on every calibration element; all tested modules selected min/max. No RKNN or closed-loop result.",
             color="#526170", fontsize=9)
    for suffix in ("png", "svg"):
        fig.savefig(args.output_dir / f"action_clipping_validation_v1.{suffix}",
                    dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps({g: summary["groups"][g]["action_mae"] for g in GROUPS}), flush=True)


if __name__ == "__main__":
    main()
