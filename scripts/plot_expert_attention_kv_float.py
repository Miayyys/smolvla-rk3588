#!/usr/bin/env python3
"""Plot paired RKNN KV precision scan from saved numeric reports."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--w8-report", type=Path, required=True)
    p.add_argument("--float-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    w8 = json.loads(args.w8_report.read_text())["per_group"]
    floats = json.loads(args.float_report.read_text())["records"]
    layers = (1, 11, 13, 15)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    x = np.arange(len(layers))
    colors = {"w8a8": "#d5791c", "bf16": "#4a7b9d", "fp16": "#167b63"}
    for i, mode in enumerate(("ptq", "qat")):
        ax = axes[i]
        for j, precision in enumerate(("w8a8", "bf16", "fp16")):
            if precision == "w8a8":
                values = [next(row[f"{mode}_mae"] for row in w8
                               if row["layer"] == layer and row["kind"] == "kv") for layer in layers]
            else:
                values = [next(row["mean_mae_vs_original_fp32"] for row in floats
                               if row["layer"] == layer and row["mode"] == mode
                               and row["precision"] == precision) for layer in layers]
            ax.bar(x + (j - 1) * 0.26, values, width=0.24, label=precision.upper(),
                   color=colors[precision])
        ax.set_yscale("log")
        ax.set_xticks(x, [f"layer {layer}" for layer in layers])
        ax.set_title(mode.upper())
        ax.set_ylabel("Mean output MAE vs original FP32 ONNX (log scale)")
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
    fig.suptitle("RK3588 host simulator: expert KV precision scan, 240 development inputs/group")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
