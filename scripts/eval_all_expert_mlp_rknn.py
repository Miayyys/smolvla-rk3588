#!/usr/bin/env python3
"""Evaluate 16 exported MLP RKNN recipes on held-out expert activations."""

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path


def run_one(task):
    export, calibration, development, builds, output, layer, mode = task
    root = output / f"layer_{layer:02d}" / mode
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "report.json"
    if report_path.is_file():
        old = json.loads(report_path.read_text())
        if old.get("samples") == 240:
            return {"layer": layer, "mode": mode, "status": "resumed",
                    "mean_mae": old["mean_rknn_mae_vs_original_fp"],
                    "rknn_bytes": old["rknn_bytes"]}
    group = f"layer_{layer:02d}"
    command = [sys.executable, str(Path(__file__).with_name("check_exported_rknn_parity.py")),
               "--rknn", str(builds / group / mode / "mlp_int8_mmse_rk3588.rknn"),
               "--onnx", str(export / group / mode / "mlp_fp32.onnx"),
               "--reference-onnx", str(export / group / "ptq/mlp_fp32.onnx"),
               "--inputs", str(development / group / "heldout_inputs.txt"),
               "--dataset", str(calibration / group / "rknn_dataset.txt"),
               "--output", str(report_path)]
    with (root / "eval.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    return {"layer": layer, "mode": mode,
            "status": "success" if result.returncode == 0 and report.get("samples") == 240 else "failed",
            "returncode": result.returncode,
            "mean_mae": report.get("mean_rknn_mae_vs_original_fp"),
            "mean_master_mae": report.get("mean_master_mae_vs_original_fp"),
            "rknn_bytes": report.get("rknn_bytes")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("export-dir", "calibration-dir", "development-dir", "build-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(args.export_dir, args.calibration_dir, args.development_dir,
              args.build_dir, args.output_dir, layer, mode)
             for layer in range(16) for mode in ("ptq", "qat")]
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            (args.output_dir / "report.json").write_text(json.dumps({
                "scope": "16 expert MLP W8A8 RKNN host numeric vs original FP32 ONNX",
                "records": sorted(records, key=lambda row: (row["layer"], row["mode"]))
            }, indent=2) + "\n")
            print(f"{len(records)}/{len(tasks)} {record}", flush=True)
    if any(row["status"] == "failed" for row in records):
        raise RuntimeError("At least one MLP numeric evaluation failed")


if __name__ == "__main__":
    main()
