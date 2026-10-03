#!/usr/bin/env python3
"""Compare four KV precision candidates against actual loaded BF16 outputs."""

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def run_one(task):
    export, calibration, development, loaded, output, layer, mode, precision = task
    group = f"layer_{layer:02d}/kv"
    destination = output / group / mode / precision
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "report.json"
    command = [sys.executable, str(Path(__file__).with_name("compile_rknn_projection_with_eval.py")),
               "--onnx", str(export / group / mode / "projection_fp32.onnx"),
               "--reference-onnx", str(export / group / "ptq/projection_fp32.onnx"),
               "--dataset", str(calibration / group / "rknn_dataset.txt"),
               "--development-inputs", str(development / group / "heldout_inputs.txt"),
               "--development-report", str(development / "report.json"),
               "--loaded-outputs-dir", str(loaded / group),
               "--output-dir", str(destination), "--precision", precision]
    with (destination / "compile_and_eval.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    return {"layer": layer, "mode": mode, "precision": precision,
            "status": report.get("status", "failed") if result.returncode == 0 else "failed",
            "returncode": result.returncode, "rknn_bytes": report.get("rknn_bytes"),
            "mean_mae_vs_original_fp32": report.get("mean_mae_vs_original_fp32"),
            "mean_mae_vs_original_loaded_bf16": report.get("mean_mae_vs_original_loaded_bf16")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("export-dir", "calibration-dir", "development-dir", "loaded-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    loaded = json.loads((args.loaded_dir / "report.json").read_text())
    development = json.loads((args.development_dir / "report.json").read_text())
    if (loaded["development_report_sha256"] !=
            hashlib.sha256((args.development_dir / "report.json").read_bytes()).hexdigest()
            or loaded["partition_sha256"] != development["partition_sha256"]
            or loaded["samples_per_layer"] != 240):
        raise ValueError("Loaded outputs do not match development input partition")
    tasks = [(args.export_dir, args.calibration_dir, args.development_dir,
              args.loaded_dir, args.output_dir, layer, mode, precision)
             for layer in (1, 11, 13, 15) for mode in ("ptq", "qat")
             for precision in ("w8a8", "bf16", "fp16")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            (args.output_dir / "report.json").write_text(json.dumps({
                "scope": "four expert KV groups vs actual loaded BF16 outputs",
                "records": sorted(records, key=lambda r: (r["layer"], r["mode"], r["precision"]))
            }, indent=2) + "\n")
            print(f"{len(records)}/{len(tasks)} {record}", flush=True)
    if any(record["status"] != "success" for record in records):
        raise RuntimeError("At least one KV candidate failed")


if __name__ == "__main__":
    main()
