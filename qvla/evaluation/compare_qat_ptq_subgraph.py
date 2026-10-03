#!/usr/bin/env python3
"""Compare paired QAT/PTQ RKNN host-simulator errors, resampling by task."""

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
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qat", type=Path, required=True)
    parser.add_argument("--ptq", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    qat = json.loads(args.qat.read_text())
    ptq = json.loads(args.ptq.read_text())
    if qat["inputs"] != ptq["inputs"] or qat["calibration_dataset"] != ptq["calibration_dataset"]:
        raise ValueError("QAT/PTQ inputs or calibration dataset differ")
    qrows = {Path(row["input"]).name: row for row in qat["records"]}
    prows = {Path(row["input"]).name: row for row in ptq["records"]}
    if qrows.keys() != prows.keys():
        raise ValueError("QAT/PTQ held-out input files differ")
    tasks = defaultdict(list)
    paired = []
    for name in sorted(qrows):
        matched = re.match(r"task_(\d+)(?:_|\.)", name)
        if matched is None:
            raise ValueError(f"Missing task ID in {name}")
        task = int(matched.group(1))
        qerr = qrows[name]["rknn_mae_vs_original_fp"]
        perr = prows[name]["rknn_mae_vs_original_fp"]
        difference = qerr - perr
        tasks[task].append(difference)
        paired.append({"input": name, "task": task, "qat_mae": qerr, "ptq_mae": perr,
                       "qat_minus_ptq_mae": difference})
    task_means = {task: float(np.mean(values)) for task, values in tasks.items()}
    counts = {len(values) for values in tasks.values()}
    if len(task_means) != 40 or len(counts) != 1:
        raise ValueError("Expected 40 tasks with equal held-out input counts")
    values = np.array(list(task_means.values()))
    rng = np.random.default_rng(0)
    boot = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    result = {
        "metric": "RKNN simulator mean output MAE vs original FP32 ONNX; QAT minus PTQ",
        "negative_favors_qat": True, "samples": len(paired), "tasks": len(task_means),
        "samples_per_task": counts.pop(),
        "qat_mean_mae": qat["mean_rknn_mae_vs_original_fp"],
        "ptq_mean_mae": ptq["mean_rknn_mae_vs_original_fp"],
        "mean_difference": float(values.mean()),
        "relative_change": float(values.mean() / ptq["mean_rknn_mae_vs_original_fp"]),
        "tasks_favoring_qat": int(np.sum(values < 0)),
        "task_bootstrap_95_percentile_ci": np.quantile(boot, [0.025, 0.975]).tolist(),
        "bootstrap_seed": 0, "bootstrap_replicates": 10000,
        "task_mean_differences": task_means, "paired_samples": paired,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("task_mean_differences", "paired_samples")}))


if __name__ == "__main__":
    main()
