#!/usr/bin/env python3
"""Plot measured paired-action development error for stage-1 QAT learning rates."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    plotted = []
    for path in args.reports:
        report = json.loads(path.read_text())
        rate = report["learning_rate"]
        history = report["development_history"]
        initial = report["baseline_vs_prepared"]["mae"]
        x = [0] + [entry["step"] for entry in history]
        y = [initial] + [entry["baseline_vs_fake"]["mae"] for entry in history]
        ax.plot(x, y, marker="o", label=f"learning rate {rate:g}")
        plotted.append({"report": str(path), "learning_rate": rate, "step": x,
                        "action_mae": y, "development_episodes": report["development_episode_ids"]})
    ax.set(xlabel="QAT training step", ylabel="Mean |action − original FP action|",
           title="Stage-1 W8A8 fake-quant QAT: 40 development tasks")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    fig.savefig(args.output.with_suffix(".svg"))
    args.output.with_suffix(".json").write_text(json.dumps(plotted, indent=2) + "\n")


if __name__ == "__main__":
    main()
