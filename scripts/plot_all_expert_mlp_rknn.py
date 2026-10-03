#!/usr/bin/env python3
"""Plot all 16 expert MLP RKNN output errors from saved held-out reports."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--plot", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.report.read_text())
    rows = data["records"]
    if len(rows) != 32 or any(row["status"] not in ("success", "resumed") for row in rows):
        raise ValueError("Expected 32 completed QAT/PTQ layer results")
    by_key = {(row["layer"], row["mode"]): row for row in rows}
    pairs = [{"layer": layer, "ptq_mae": by_key[(layer, "ptq")]["mean_mae"],
              "qat_mae": by_key[(layer, "qat")]["mean_mae"],
              "qat_minus_ptq": by_key[(layer, "qat")]["mean_mae"] -
                               by_key[(layer, "ptq")]["mean_mae"]}
             for layer in range(16)]
    summary = {"source_report": str(args.report), "plot": str(args.plot),
               "development_inputs_per_layer": 240,
               "qat_lower_mae_layers": sum(row["qat_minus_ptq"] < 0 for row in pairs),
               "ptq_lower_mae_layers": sum(row["qat_minus_ptq"] > 0 for row in pairs),
               "pairs": pairs}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    x = np.arange(16)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True,
                             height_ratios=(2, 1))
    axes[0].plot(x, [row["ptq_mae"] for row in pairs], "o-", label="Original FP → PTQ W8A8")
    axes[0].plot(x, [row["qat_mae"] for row in pairs], "o-", label="QAT master → W8A8")
    axes[0].set_ylabel("Output MAE vs original FP32 ONNX")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    differences = [row["qat_minus_ptq"] for row in pairs]
    axes[1].bar(x, differences, color=["#158765" if value < 0 else "#b95454"
                                       for value in differences])
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("QAT − PTQ MAE")
    axes[1].set_xticks(x)
    axes[1].set_xlabel("Expert MLP layer (240 independent development inputs/layer)")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("RK3588 W8A8 host simulator: all expert MLPs")
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=180)


if __name__ == "__main__":
    main()
