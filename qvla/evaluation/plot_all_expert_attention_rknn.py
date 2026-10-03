#!/usr/bin/env python3
"""Plot 40 expert attention projection groups' W8A8 size and held-out output MAE."""

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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--plot", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    args = p.parse_args()
    report = json.loads(args.report.read_text())
    records = report["records"]
    if len(records) != 80 or any(row["status"] != "success" for row in records):
        raise ValueError("Expected 80 successful projection builds")
    values = {(r["layer"], r["kind"], r["mode"]): r for r in records}
    groups = sorted({(r["layer"], r["kind"]) for r in records})
    if len(groups) != 40:
        raise ValueError("Expected 40 attention projection groups")
    ptq = np.array([values[(layer, kind, "ptq")]["mean_mae_vs_original_fp32"]
                    for layer, kind in groups])
    qat = np.array([values[(layer, kind, "qat")]["mean_mae_vs_original_fp32"]
                    for layer, kind in groups])
    delta = qat - ptq
    labels = [f"{layer}:{kind}" for layer, kind in groups]
    x = np.arange(len(groups))
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, layout="constrained")
    axes[0].plot(x, ptq, "o-", markersize=3, label="Original FP → PTQ W8A8")
    axes[0].plot(x, qat, "o-", markersize=3, label="QAT master → W8A8")
    axes[0].set_ylabel("Output MAE vs original FP32 ONNX")
    axes[0].grid(alpha=.25)
    axes[0].legend()
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].bar(x, delta, color=np.where(delta < 0, "#138a68", "#bb5959"))
    axes[1].set_ylabel("QAT − PTQ output MAE")
    axes[1].set_xlabel("Expert layer : projection group (240 independent development inputs/group)")
    axes[1].set_xticks(x, labels, rotation=90, fontsize=7)
    axes[1].grid(axis="y", alpha=.25)
    fig.suptitle("RK3588 W8A8 host simulator: all expert attention projection groups")
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=170)
    plt.close(fig)
    summary = {"scope": "40 attention projection subgraphs; FP32 ONNX reference only",
               "source_report": str(args.report), "plot": str(args.plot),
               "calibration_inputs_per_group": 240, "development_inputs_per_group": 240,
               "total_rknn_bytes_by_mode": {mode: sum(r["rknn_bytes"] for r in records if r["mode"] == mode)
                                            for mode in ("ptq", "qat")},
               "qat_lower_mae_groups": int(np.count_nonzero(delta < 0)),
               "ptq_lower_mae_groups": int(np.count_nonzero(delta > 0)),
               "per_group": [{"layer": layer, "kind": kind,
                              "ptq_mae": float(ptq[i]), "qat_mae": float(qat[i]),
                              "qat_minus_ptq": float(delta[i]),
                              "rknn_bytes": values[(layer, kind, "ptq")]["rknn_bytes"]}
                             for i, (layer, kind) in enumerate(groups)]}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
