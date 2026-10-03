#!/usr/bin/env python3
"""Plot RKNN float formats against the originally loaded MLP outputs."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
FIGURES = ROOT / "figures"
MODULES = ("vision10", "vision11", "language3", "language4")


def main() -> None:
    original = json.loads((RUNS / "rknn_float_dtype_compare/report_vs_loaded.json").read_text())
    rows = []
    for name in MODULES:
        modes = {}
        for mode in ("bf16", "fp16"):
            source = RUNS / f"rknn_{name}_{mode}_probe/parity_vs_loaded.json"
            parity = json.loads(source.read_text())
            modes[mode] = {"mean_mae_vs_original_loaded": parity["mean_mae_vs_original_loaded"],
                           "source": str(source.relative_to(ROOT)),
                           "samples": parity["samples"]}
        rows.append({"module": name,
                     "original_loaded_dtype": original["modules"][name]["source_parameter_dtypes"][0],
                     "modes": modes})
    FIGURES.mkdir(exist_ok=True)
    (FIGURES / "rknn_vs_original_v1.json").write_text(json.dumps(rows, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(rows))
    width = 0.32
    for offset, mode, color in ((-width / 2, "bf16", "#8b5fbf"),
                                (width / 2, "fp16", "#31688e")):
        ax.bar(x + offset, [row["modes"][mode]["mean_mae_vs_original_loaded"]
                            for row in rows], width, color=color, label=mode.upper())
    labels = [f'{r["module"].replace("vision", "vision ").replace("language", "language ")}\n'
              f'original {"FP32" if r["original_loaded_dtype"] == "torch.float32" else "BF16"}'
              for r in rows]
    ax.set_xticks(x, labels)
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1e-2)
    ax.set_ylabel("Mean output MAE vs originally loaded MLP (log scale)")
    ax.set_title("RKNN host simulator · four held-out inputs per MLP")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "rknn_vs_original_v1.png", dpi=180)
    fig.savefig(FIGURES / "rknn_vs_original_v1.svg")


if __name__ == "__main__":
    main()
