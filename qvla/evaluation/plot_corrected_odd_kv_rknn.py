#!/usr/bin/env python3
"""Plot corrected independent K/V RKNN W8A8 results from the host simulator."""

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
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    records = report["records"]
    if len(records) != 32 or any(r.get("status") != "success" or r.get("samples") != 240
                                  for r in records):
        raise ValueError("Expected 32 successful K/V×QAT/PTQ results with 240 inputs each")
    values = {(r["layer"], r["kind"], r["mode"]): r["mean_mae_vs_original_loaded_bf16"]
              for r in records}
    layers = sorted({r["layer"] for r in records})
    if layers != list(range(1, 16, 2)):
        raise ValueError(f"Unexpected odd expert layer set: {layers}")

    summary = {"scope": report["scope"], "source_report": str(args.report),
               "development_inputs_per_projection": 240, "layers": layers, "by_kind": {}}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, kind in zip(axes, ("k", "v")):
        ptq = [values[(layer, kind, "ptq")] for layer in layers]
        qat = [values[(layer, kind, "qat")] for layer in layers]
        ax.plot(layers, ptq, "o-", label="Original FP → PTQ W8A8")
        ax.plot(layers, qat, "s-", label="QAT master → W8A8")
        ax.set_title(f"{kind.upper()} projection")
        ax.set_xlabel("Odd action-expert layer")
        ax.set_ylabel("Output MAE vs loaded BF16 reference")
        ax.set_xticks(layers)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        paired = [(p, q) for p, q in zip(ptq, qat)]
        summary["by_kind"][kind] = {
            "ptq_mean_mae": statistics.fmean(ptq),
            "qat_mean_mae": statistics.fmean(qat),
            "qat_lower_mae_layers": sum(q < p for p, q in paired),
            "ptq_lower_mae_layers": sum(p < q for p, q in paired),
            "per_layer": [{"layer": layer, "ptq_mae": p, "qat_mae": q}
                          for layer, p, q in zip(layers, ptq, qat)],
        }
    fig.suptitle("Corrected RKNN W8A8 K/V subgraphs (host simulator)")
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=180)
    summary["plot"] = str(args.plot)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
