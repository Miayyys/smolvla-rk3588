#!/usr/bin/env python3
"""Resume paired FP/PTQ/QAT LIBERO development rollouts, one suite at a time."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("fp", "ptq", "qat"), required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--vlm-assets-dir", type=Path, required=True)
    p.add_argument("--pack-report", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for suite in SUITES:
        output = args.output_dir / suite
        output.mkdir(parents=True, exist_ok=True)
        result_path = output / "eval_info.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            records.append({"suite": suite, "status": "resumed", "overall": result.get("overall")})
            print(f"{args.mode} {suite}: existing result reused", flush=True)
            continue
        command = [sys.executable, str(Path(__file__).with_name("eval_real_w8a8_libero.py")),
                   "--qvla-pack-report", str(args.pack_report),
                   "--qvla-mode", args.mode,
                   "--qvla-vlm-assets-dir", str(args.vlm_assets_dir),
                   "--policy.path", str(args.model_dir),
                   "--env.type", "libero", "--env.task", suite,
                   "--env.observation_height", "256", "--env.observation_width", "256",
                   "--eval.n_episodes", "1", "--eval.batch_size", "1",
                   "--seed", str(args.seed), "--output_dir", str(output)]
        environment = dict(os.environ, HF_HUB_OFFLINE="1", OMP_NUM_THREADS="1")
        started = time.time()
        print(f"{args.mode} {suite}: running 10 tasks × 1 episode", flush=True)
        with (output / "rollout.log").open("w") as log:
            completed = subprocess.run(command, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, check=False)
        record = {"suite": suite, "command": command, "seed": args.seed,
                  "elapsed_seconds": time.time() - started,
                  "returncode": completed.returncode,
                  "status": "success" if completed.returncode == 0 and result_path.is_file() else "failed"}
        if result_path.is_file():
            record["overall"] = json.loads(result_path.read_text()).get("overall")
        records.append(record)
        (args.output_dir / "progress.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"{args.mode} {suite}: {record['status']} after {record['elapsed_seconds']:.1f}s", flush=True)
        if record["status"] != "success":
            raise RuntimeError(f"Rollout failed: {output / 'rollout.log'}")
    print(f"{args.mode}: all four suites complete", flush=True)


if __name__ == "__main__":
    main()
