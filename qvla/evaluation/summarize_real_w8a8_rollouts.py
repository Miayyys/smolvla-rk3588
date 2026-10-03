#!/usr/bin/env python3
"""Summarize paired LIBERO task success from saved FP/PTQ/QAT eval_info files."""

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

import matplotlib.pyplot as plt
import numpy as np


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MODES = ("fp", "ptq", "qat")


def load(root):
    outcomes = {}
    metadata = {}
    for mode in MODES:
        for suite in SUITES:
            path = root / f"expert_real_w8a8_rollout_{mode}_v1" / suite / "eval_info.json"
            data = json.loads(path.read_text())
            if data["overall"]["n_episodes"] != 10 or len(data["per_task"]) != 10:
                raise ValueError(f"Expected 10 one-episode tasks in {path}")
            for task in data["per_task"]:
                task_id = task["task_id"]
                successes = task["metrics"]["successes"]
                if len(successes) != 1 or task["task_group"] != suite:
                    raise ValueError(f"Unexpected task format in {path}")
                key = (suite, task_id)
                if (mode, key) in outcomes:
                    raise ValueError(f"Duplicate task {mode} {key}")
                outcomes[(mode, key)] = bool(successes[0])
            metadata[(mode, suite)] = {"eval_seconds": data["overall"]["eval_s"],
                                       "pc_success": data["overall"]["pc_success"]}
    keys = sorted({key for mode, key in outcomes})
    if len(keys) != 40 or any((mode, key) not in outcomes for key in keys for mode in MODES):
        raise ValueError("FP/PTQ/QAT task IDs do not align")
    return keys, outcomes, metadata


def bootstrap_paired(keys, outcomes, mode, n=20000):
    rng = np.random.default_rng(0)
    by_suite = [[key for key in keys if key[0] == suite] for suite in SUITES]
    differences = []
    for suite_keys in by_suite:
        differences.append(np.array([int(outcomes[(mode, key)]) - int(outcomes[("fp", key)])
                                     for key in suite_keys], dtype=np.float64))
    draws = np.stack([np.mean(values[rng.integers(0, len(values), (n, len(values)))], axis=1)
                      for values in differences], axis=1).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--plot", type=Path, required=True)
    args = p.parse_args()
    keys, outcomes, metadata = load(args.runs_root)
    suites = {}
    for suite in SUITES:
        suite_keys = [key for key in keys if key[0] == suite]
        suites[suite] = {mode: sum(outcomes[(mode, key)] for key in suite_keys)
                         for mode in MODES}
    totals = {mode: sum(outcomes[(mode, key)] for key in keys) for mode in MODES}
    comparisons = {}
    for mode in ("ptq", "qat"):
        delta = [int(outcomes[(mode, key)]) - int(outcomes[("fp", key)]) for key in keys]
        comparisons[mode] = {"delta_successes_out_of_40": sum(delta),
                             "delta_success_rate": sum(delta) / len(delta),
                             "improved_tasks": sum(value > 0 for value in delta),
                             "worsened_tasks": sum(value < 0 for value in delta),
                             "unchanged_tasks": sum(value == 0 for value in delta),
                             "stratified_task_bootstrap_95pct_descriptive": bootstrap_paired(
                                 keys, outcomes, mode)}
    report = {"scope": "LIBERO four suites, 10 tasks × 1 episode, seed 0, 256×256",
              "interval_warning": "one initial state per task; task bootstrap is descriptive, not a quality gate",
              "totals": totals, "per_suite_successes_out_of_10": suites,
              "comparisons_vs_fp": comparisons,
              "per_task": [{"suite": key[0], "task_id": key[1],
                            **{mode: outcomes[(mode, key)] for mode in MODES}} for key in keys],
              "eval_metadata": {f"{mode}/{suite}": metadata[(mode, suite)]
                                for mode in MODES for suite in SUITES}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    x = np.arange(len(SUITES))
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for i, mode in enumerate(MODES):
        values = [suites[suite][mode] for suite in SUITES]
        bars = ax.bar(x + (i - 1) * 0.25, values, width=0.23, label=mode.upper())
        ax.bar_label(bars, labels=[str(value) for value in values], padding=3)
    ax.set_xticks(x, [suite.removeprefix("libero_") for suite in SUITES])
    ax.set_ylim(0, 11)
    ax.set_ylabel("Successful tasks out of 10 (one episode/task)")
    ax.set_title("Paired LIBERO rollout: FP, real INT8 PTQ, real INT8 QAT")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=180)
    print(json.dumps({"totals": totals, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
