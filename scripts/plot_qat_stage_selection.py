#!/usr/bin/env python3
"""Plot measured output-rounding candidate stages; not real W8A8 quality."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STAGES = [
    ("Expert only", "mixed_stage1_expert_outputs", 199720960,
     "runs/action_mixed_stages_v1/report.json"),
    ("+ late VLM", "mixed_stage2_late_vlm_outputs", 432580096,
     "runs/action_mixed_stages_v1/report.json"),
    ("All v0 candidates", "mixed_v0_active_outputs", 660138496,
     "runs/action_mixed_v0_combo/report.json"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-stem", type=Path,
                        default=Path("figures/qat_stage_selection_v1"))
    args = parser.parse_args()
    rows = []
    for label, key, source_bytes, path in STAGES:
        report = json.loads((args.project_root / path).read_text())
        if report["not_real_quantization"] is not True or report["tasks"] != 40:
            raise ValueError("Expected 40-task fake-quant diagnostic")
        records = report["groups"][key]["records"]
        values = np.asarray([row["full_chunk_mae_vs_fp"] for row in records])
        rows.append({"label": label, "group": key, "source_weight_bytes": source_bytes,
                     "module_count": len(report["groups"][key]["modules"]),
                     "samples": len(records), "mean_action_mae_vs_fp": float(values.mean()),
                     "p95_row_action_mae_vs_fp": float(np.percentile(values, 95)),
                     "max_row_action_mae_vs_fp": float(values.max())})
    stem = args.project_root / args.output_stem
    stem.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    colors = ["#256783", "#d69b36", "#aa4c43"]
    axes[0].bar(x, [row["mean_action_mae_vs_fp"] for row in rows],
                color=colors, width=0.56, label="Mean")
    axes[0].scatter(x, [row["p95_row_action_mae_vs_fp"] for row in rows],
                    color="#1c2731", zorder=3, label="p95 observation")
    axes[0].set_ylabel("Full action chunk MAE vs FP")
    axes[0].legend(frameon=False)
    axes[1].bar(x, [row["source_weight_bytes"] / 1e6 for row in rows],
                color=colors, width=0.56)
    axes[1].set_ylabel("Source weights in candidate groups (MB)")
    for ax in axes:
        ax.set_xticks(x, [row["label"] for row in rows], rotation=10, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Mixed-precision QAT stage screening", fontsize=15, y=0.98)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.24, top=0.85, wspace=0.28)
    fig.text(0.5, 0.045,
             "40 disjoint development observations; output fake INT8 only. "
             "Source bytes are not compression savings.",
             ha="center", fontsize=8, color="#555555")
    fig.savefig(stem.with_suffix(".png"), dpi=190)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)
    stem.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
