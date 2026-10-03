#!/usr/bin/env python3
"""Run paired FP/PTQ/QAT LIBERO development suites concurrently and resumably."""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path


MODES = ("fp", "ptq", "qat")


def run_mode(task):
    args, mode = task
    output = args.output_root / f"expert_real_w8a8_rollout_{mode}_v1"
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).with_name("run_real_w8a8_libero_suites.py")),
               "--mode", mode, "--model-dir", str(args.model_dir),
               "--vlm-assets-dir", str(args.vlm_assets_dir),
               "--pack-report", str(args.pack_report), "--output-dir", str(output),
               "--seed", str(args.seed)]
    env = dict(os.environ, HF_HUB_OFFLINE="1", OMP_NUM_THREADS="1")
    started = time.time()
    log_path = args.output_root / f"{mode}.log"
    with log_path.open("w") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                check=False)
    return {"mode": mode, "returncode": result.returncode,
            "status": "success" if result.returncode == 0 else "failed",
            "elapsed_seconds": time.time() - started, "output_dir": str(output),
            "log": str(log_path)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "vlm-assets-dir", "pack-report", "output-root"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    args = p.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_mode, (args, mode)): mode for mode in MODES}
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            (args.output_root / "progress.json").write_text(json.dumps({
                "scope": "per-task paired-noise FP/PTQ/QAT LIBERO rollouts",
                "seed": args.seed,
                "records": sorted(records, key=lambda row: MODES.index(row["mode"]))
            }, indent=2) + "\n")
            print(f"{record['mode']}: {record['status']} in {record['elapsed_seconds']:.1f}s",
                  flush=True)
    if any(record["status"] != "success" for record in records):
        raise RuntimeError("At least one mode failed; see per-mode logs")


if __name__ == "__main__":
    main()
