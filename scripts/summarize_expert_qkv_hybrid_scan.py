#!/usr/bin/env python3
"""Summarize real RKNN file bytes and paired simulator error for QKV bit choices."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path, label, error_key, size_override=None):
    data = json.loads(path.read_text())
    rows = {Path(row["input"]).name: row[error_key] for row in data["records"]}
    if len(rows) != 40:
        raise ValueError(f"Expected 40 unique development inputs for {label}")
    return {"label": label, "bytes": size_override if size_override is not None else data["rknn_bytes"],
            "mae": float(np.mean(list(rows.values()))),
            "report": str(path), "per_input": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", choices=("loaded", "fp32"), default="loaded")
    args = parser.parse_args()
    root = args.runs
    loaded = args.reference == "loaded"
    base_file = "development_loaded_parity.json" if loaded else "development_simulator_parity.json"
    base_key = "rknn_mae_vs_original_loaded" if loaded else "rknn_mae_vs_original_fp"
    hybrid_key = "mae_vs_original_loaded" if loaded else "mae"
    fp16_key = "mae_vs_original_loaded" if loaded else "mae"
    candidates = [
        load(root / "expert_qkv0_ptq_v1" / base_file, "all INT8", base_key),
        load(root / "expert_qkv0_qat50_v1" / base_file, "QAT all INT8", base_key),
    ]
    for part in ("q", "k", "v", "qk", "qv", "kv"):
        label = "qfp16" if part == "q" else part + "fp16"
        directory = "expert_qkv0_hybrid_" + ("qfp16" if part == "q" else part) + "_v1"
        candidates.append(load(root / directory / f"hybrid_{label}_development_parity.json",
                               f"{part.upper()} FP16", hybrid_key))
    fp16_bytes = json.loads((root / "expert_qkv0_fp16_v1/fp16_compile_report.json").read_text())["rknn_bytes"]
    candidates.append(load(root / "expert_qkv0_fp16_v1" / base_file,
                           "all FP16", fp16_key, fp16_bytes))
    names = candidates[0]["per_input"].keys()
    if any(item["per_input"].keys() != names for item in candidates):
        raise ValueError("Candidate input names differ")
    for row in candidates:
        row["pareto"] = not any(
            other is not row and other["bytes"] <= row["bytes"] and other["mae"] <= row["mae"]
            and (other["bytes"] < row["bytes"] or other["mae"] < row["mae"])
            for other in candidates)
    baseline = candidates[0]["per_input"]
    for row in candidates:
        diffs = np.array([row["per_input"][name] - baseline[name] for name in sorted(names)])
        rng = np.random.default_rng(0)
        boots = rng.choice(diffs, size=(10000, len(diffs)), replace=True).mean(axis=1)
        row["paired_mean_delta_vs_ptq_int8"] = float(diffs.mean())
        row["paired_bootstrap_95_ci"] = np.quantile(boots, [0.025, 0.975]).tolist()
        row["tasks_lower_error_than_ptq_int8"] = int(np.sum(diffs < 0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reference = "original loaded BF16 projections" if loaded else "original FP32 ONNX"
    args.output.write_text(json.dumps({"scope": "one expert layer-0 QKV subgraph, 40 development tasks",
                                       "metric": f"host simulator MAE vs {reference}",
                                       "board_ram_latency": "not_measured",
                                       "bootstrap_seed": 0, "bootstrap_replicates": 10000,
                                       "candidates": candidates}, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(8, 5))
    offsets = {"all INT8": (4, -15), "QAT all INT8": (4, 5),
               "K FP16": (4, 7), "V FP16": (4, -15),
               "QK FP16": (-48, 9), "QV FP16": (5, -15),
               "all FP16": (5, 4)}
    for row in candidates:
        ax.scatter(row["bytes"] / 1e6, row["mae"], s=85 if row["pareto"] else 45,
                   color="#0868ac" if row["pareto"] else "#8c8c8c")
        offset = offsets.get(row["label"], (4, 4))
        ax.annotate(row["label"], (row["bytes"] / 1e6, row["mae"]),
                    xytext=offset, textcoords="offset points", fontsize=9)
    frontier = sorted((r for r in candidates if r["pareto"]), key=lambda r: r["bytes"])
    ax.plot([r["bytes"] / 1e6 for r in frontier], [r["mae"] for r in frontier],
            color="#0868ac", alpha=0.5)
    ax.set(xlabel="Exported RKNN file size (MB)", ylabel=f"Mean output MAE vs {reference}",
           title="Expert layer-0 QKV: 40 held-out inputs, RKNN host simulator")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output.with_suffix(".png"), dpi=180)
    fig.savefig(args.output.with_suffix(".svg"))
    print(json.dumps([{k: r[k] for k in ("label", "bytes", "mae", "pareto",
                                          "paired_mean_delta_vs_ptq_int8")}
                      for r in candidates]))


if __name__ == "__main__":
    main()
