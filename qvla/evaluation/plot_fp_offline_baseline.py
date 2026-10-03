#!/usr/bin/env python3
"""Plot recorded FP action error and latency from the 40-sample offline test."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    records = sorted(report["records"], key=lambda row: row["task_index"])
    if report["samples"] != 40 or report["tasks"] != 40:
        raise ValueError("Expected one frozen first-frame sample per task")
    if [row["task_index"] for row in records] != list(range(40)):
        raise ValueError("Task IDs must cover 0 through 39 exactly once")
    ids = np.arange(40)
    errors = np.array([row["action_mae_vs_recorded"] for row in records])
    latency = np.array([row["latency_ms"] for row in records])
    if not np.isclose(errors.mean(), report["mean_action_mae_vs_recorded"]):
        raise ValueError("Per-task errors disagree with report mean")
    if not np.isclose(np.percentile(latency, 50), report["latency_p50_ms"]):
        raise ValueError("Per-task latencies disagree with report median")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.3), sharex=True,
                             gridspec_kw={"hspace": 0.24})
    fig.patch.set_facecolor("white")
    green, blue, gray = "#16876d", "#466aaa", "#536171"
    axes[0].bar(ids, errors, width=0.76, color=green)
    axes[0].axhline(report["mean_action_mae_vs_recorded"], color="#b25545",
                    linestyle="--", linewidth=1.5,
                    label=f"Mean = {report['mean_action_mae_vs_recorded']:.4f}")
    axes[0].set_ylabel("Action MAE vs recorded action")
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].scatter(ids, latency, s=28, color=blue, zorder=3)
    axes[1].plot(ids, latency, linewidth=0.8, alpha=0.55, color=blue)
    axes[1].annotate(f"First sample: {latency[0]:.1f} ms", xy=(0, latency[0]),
                     xytext=(5, latency[0] - 33), fontsize=9, color=gray,
                     arrowprops={"arrowstyle": "->", "color": gray, "lw": 0.8})
    axes[1].axhline(report["latency_p50_ms"], color=green, linestyle="--",
                    linewidth=1.5, label=f"p50 = {report['latency_p50_ms']:.1f} ms")
    axes[1].axhline(report["latency_p95_ms"], color="#b25545", linestyle=":",
                    linewidth=1.7, label=f"p95 = {report['latency_p95_ms']:.1f} ms")
    axes[1].set_ylabel("Action inference latency (ms)")
    axes[1].set_xlabel("Task index (one first frame per task)")
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].set_xlim(-0.8, 39.8)
    axes[1].set_xticks(range(0, 40, 5))
    for ax in axes:
        for boundary in (9.5, 19.5, 29.5):
            ax.axvline(boundary, color="#dfe5ea", linewidth=0.9)
        ax.grid(axis="y", color="#e5eaed", linewidth=0.7)
        ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.79, bottom=0.18)
    fig.text(0.12, 0.96, "SmolVLA FP · offline action baseline", fontsize=20,
             fontweight="bold", color="#15252e", va="top")
    fig.text(0.12, 0.905, "40 held-out first frames  ·  fixed seed 0  ·  prior GPU environment",
             fontsize=12, color=gray, va="top")
    fig.text(0.12, 0.08,
             "Action MAE compares one predicted action with the dataset action. "
             "It is not task success or a closed-loop score.", fontsize=9, color=gray)
    fig.text(0.12, 0.052, f"Source: {args.report.as_posix()}", fontsize=8, color=gray)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(args.output_dir / f"fp_offline_40.{suffix}", dpi=220,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
