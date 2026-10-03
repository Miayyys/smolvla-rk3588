#!/usr/bin/env python3
"""Plot the recorded 40-task, one-episode SmolVLA FP LIBERO pilot."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


SUITES = [
    ("libero_spatial", "Spatial"),
    ("libero_object", "Object"),
    ("libero_goal", "Goal"),
    ("libero_10", "LIBERO-10"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report["overall"]["episodes"] != 40 or report["seed"] != 0:
        raise ValueError("This figure is defined for the frozen 40-task, seed-0 pilot")

    matrix = np.full((4, 10), np.nan)
    for task in report["tasks"]:
        row = next((i for i, (key, _) in enumerate(SUITES) if key == task["suite"]), None)
        if row is None or not 0 <= task["task_id"] < 10 or not np.isnan(matrix[row, task["task_id"]]):
            raise ValueError(f"Unexpected or duplicate task record: {task}")
        matrix[row, task["task_id"]] = int(task["success"])
    if np.isnan(matrix).any():
        raise ValueError("Some task records are missing")
    wins = matrix.sum(axis=1).astype(int)
    for (suite, _), win in zip(SUITES, wins):
        if report["suite_results"][suite] != {"successes": int(win), "episodes": 10}:
            raise ValueError(f"Suite summary disagrees with task records: {suite}")
    if wins.sum() != report["overall"]["successes"]:
        raise ValueError("Overall summary disagrees with task records")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    green, red, gray = "#16876d", "#d85e54", "#536171"
    fig = plt.figure(figsize=(11.5, 7.4), facecolor="white")
    grid = fig.add_gridspec(2, 1, height_ratios=[1.15, 1], hspace=0.37,
                           left=0.14, right=0.94, top=0.83, bottom=0.16)
    ax = fig.add_subplot(grid[0])
    y = np.arange(4)
    ax.barh(y, wins, color=green, height=0.58, label="Success")
    ax.barh(y, 10 - wins, left=wins, color=red, height=0.58, label="Failure")
    for i, win in enumerate(wins):
        ax.text(10.2, i, f"{win}/10", va="center", color=gray, fontweight="bold")
    ax.set_yticks(y, [label for _, label in SUITES])
    ax.invert_yaxis()
    ax.set_xlim(0, 11.3)
    ax.set_xticks(range(0, 11, 2))
    ax.set_xlabel("Successful episodes out of 10")
    ax.grid(axis="x", color="#e2e8ec", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.01))

    heat = fig.add_subplot(grid[1])
    heat.imshow(matrix, cmap=ListedColormap([red, green]), vmin=0, vmax=1, aspect="auto")
    heat.set_yticks(range(4), [label for _, label in SUITES])
    heat.set_xticks(range(10), [str(i) for i in range(10)])
    heat.set_xlabel("Task ID within suite")
    heat.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    heat.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    heat.grid(which="minor", color="white", linewidth=3)
    heat.tick_params(which="minor", bottom=False, left=False)
    for row in range(4):
        for col in range(10):
            heat.text(col, row, "✓" if matrix[row, col] else "×",
                      ha="center", va="center", color="white", fontsize=14,
                      fontweight="bold")
    for spine in heat.spines.values():
        spine.set_visible(False)

    fig.text(0.14, 0.96, "SmolVLA FP baseline · LIBERO", fontsize=20,
             fontweight="bold", color="#15252e", va="top")
    fig.text(0.14, 0.905, "31 / 40 successful episodes  ·  77.5% observed success",
             fontsize=12, color=gray, va="top")
    fig.text(0.14, 0.07,
             "Development pilot: one initial state per task, seed 0, one episode per task. "
             "No multi-seed uncertainty estimate.", fontsize=9, color=gray)
    fig.text(0.14, 0.043, f"Source: {args.report.as_posix()}", fontsize=8, color=gray)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(args.output_dir / f"fp_libero_40x1.{suffix}", dpi=220,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
