#!/usr/bin/env python3
"""Plot measured per-task RKNN simulator error for paired QAT and PTQ."""

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
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--module-label", default="Expert MLP")
    args = parser.parse_args()
    data = json.loads(args.comparison.read_text())
    by_task = defaultdict(list)
    for row in data["paired_samples"]:
        by_task[row["task"]].append(row)
    ids = sorted(by_task)
    qat = np.array([np.mean([r["qat_mae"] for r in by_task[i]]) for i in ids])
    ptq = np.array([np.mean([r["ptq_mae"] for r in by_task[i]]) for i in ids])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ids, ptq, marker="o", markersize=3, linewidth=1, label="PTQ from original FP")
    ax.plot(ids, qat, marker="o", markersize=3, linewidth=1, label="QAT step 50 → RKNN")
    ax.set(xlabel="LIBERO development task index", ylabel="Mean output MAE vs original FP32 ONNX",
           title=(f"{args.module_label}: RKNN W8A8 host simulator, "
                  f"{data.get('samples_per_task', data['samples'] // data['tasks'])} input(s) per task"))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    fig.savefig(args.output.with_suffix(".svg"))


if __name__ == "__main__":
    main()
