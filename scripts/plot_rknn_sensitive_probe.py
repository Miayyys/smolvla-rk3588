#!/usr/bin/env python3
"""Plot held-out RKNN host-simulator MLP output error from measured reports."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
FIGURES = ROOT / "figures"
CASES = [
    ("vision 10", "FP16", "rknn_vision10_fp16_probe"),
    ("vision 10", "INT8 MMSE", "rknn_vision10_int8_probe"),
    ("vision 11", "FP16", "rknn_vision11_fp16_probe"),
    ("vision 11", "INT8 MMSE", "rknn_vision11_int8_probe"),
    ("language 3", "FP16", "rknn_language3_fp16_probe"),
    ("language 3", "INT8 MMSE", "rknn_language3_int8_probe"),
    ("language 4", "FP16", "rknn_language4_fp16_probe"),
    ("language 4", "INT8 MMSE", "rknn_language4_int8_probe"),
]


def main() -> None:
    rows = []
    for module, mode, folder in CASES:
        report = json.loads((RUNS / folder / "parity.json").read_text())
        rows.append({"module": module, "mode": mode, "mean_mae": report["mean_mae"],
                     "mean_cosine": report["mean_cosine"], "samples": report["samples"],
                     "source": str(Path("runs") / folder / "parity.json")})
    FIGURES.mkdir(exist_ok=True)
    (FIGURES / "rknn_sensitive_probe_v1.json").write_text(
        json.dumps(rows, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(10.8, 4.5))
    x = range(len(rows))
    colors = ["#31688e" if row["mode"] == "FP16" else "#d95f02" for row in rows]
    ax.bar(x, [row["mean_mae"] for row in rows], color=colors, width=0.64)
    ax.set_yscale("log")
    ax.set_ylim(0.0002, 2)
    ax.set_xticks(list(x), [f'{row["module"]}\n{row["mode"]}' for row in rows])
    ax.set_ylabel("Mean absolute output error vs FP32 ONNX (log scale)")
    ax.set_title("RKNN host simulator · four held-out development inputs per MLP")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "rknn_sensitive_probe_v1.png", dpi=180)
    fig.savefig(FIGURES / "rknn_sensitive_probe_v1.svg")


if __name__ == "__main__":
    main()
