#!/usr/bin/env python3
"""Plot measured BF16/FP16 errors for direct PyTorch and RKNN host simulation."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
FIGURES = ROOT / "figures"
MODULES = ("vision10", "vision11", "language3", "language4")


def main() -> None:
    torch_report = json.loads((RUNS / "rknn_float_dtype_compare/report.json").read_text())
    rows = []
    for name in MODULES:
        module = torch_report["modules"][name]
        rknn = {}
        for dtype in ("bf16", "fp16", "int8"):
            base = RUNS / f"rknn_{name}_{dtype}_probe"
            parity = json.loads((base / "parity.json").read_text())
            compile_report = json.loads(next(base.glob("*compile_report.json")).read_text())
            rknn[dtype] = {"mean_mae": parity["mean_mae"],
                           "rknn_bytes": compile_report["rknn_bytes"],
                           "parity_source": str((base / "parity.json").relative_to(ROOT)),
                           "compile_source": str(next(base.glob("*compile_report.json")).relative_to(ROOT))}
        rows.append({"module": name, "source_runtime_dtype": module["source_parameter_dtypes"][0],
                     "direct_bf16_mae": module["mean_bf16_vs_fp32_mae"],
                     "direct_fp16_mae": module["mean_fp16_vs_fp32_mae"],
                     "loaded_mae": module["mean_loaded_vs_fp32_mae"],
                     "rknn": rknn})
    FIGURES.mkdir(exist_ok=True)
    (FIGURES / "rknn_float_dtypes_v1.json").write_text(json.dumps(rows, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
    positions = np.arange(len(rows))
    width = 0.25
    axes[0].bar(positions - width / 2, [r["direct_bf16_mae"] for r in rows], width,
                label="BF16", color="#8b5fbf")
    axes[0].bar(positions + width / 2, [r["direct_fp16_mae"] for r in rows], width,
                label="FP16", color="#31688e")
    axes[0].set_title("PyTorch MLP arithmetic")
    for offset, dtype, color in ((-width, "bf16", "#8b5fbf"),
                                  (0, "fp16", "#31688e"),
                                  (width, "int8", "#d95f02")):
        axes[1].bar(positions + offset,
                    [r["rknn"][dtype]["mean_mae"] for r in rows], width,
                    label=dtype.upper(), color=color)
    axes[1].set_title("RKNN host simulator")
    for ax in axes:
        ax.set_xticks(positions, [r["module"].replace("vision", "vision ").replace("language", "language ")
                                  for r in rows], rotation=20)
        ax.set_yscale("log")
        ax.set_ylim(0.00005, 2)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("Mean output MAE vs FP32 copy (log scale)")
    fig.suptitle("Four held-out inputs per MLP · stored/runtime dtypes may differ")
    fig.tight_layout()
    fig.savefig(FIGURES / "rknn_float_dtypes_v1.png", dpi=180)
    fig.savefig(FIGURES / "rknn_float_dtypes_v1.svg")


if __name__ == "__main__":
    main()
