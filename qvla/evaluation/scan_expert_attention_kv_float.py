#!/usr/bin/env python3
"""Compare FP16/BF16 RKNN host output on four high-error expert KV groups."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))
from qvla.paths import source_path


import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path


def run_one(task):
    export, calibration, development, root, layer, mode, precision = task
    group = f"layer_{layer:02d}/kv"
    output = root / group / mode / precision
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(source_path('compile_rknn_projection_with_eval.py')),
               "--onnx", str(export / group / mode / "projection_fp32.onnx"),
               "--reference-onnx", str(export / group / "ptq/projection_fp32.onnx"),
               "--dataset", str(calibration / group / "rknn_dataset.txt"),
               "--development-inputs", str(development / group / "heldout_inputs.txt"),
               "--development-report", str(development / "report.json"),
               "--output-dir", str(output), "--precision", precision]
    with (output / "compile_and_eval.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    report_path = output / "report.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    return {"layer": layer, "mode": mode, "precision": precision,
            "status": report.get("status", "failed") if result.returncode == 0 else "failed",
            "returncode": result.returncode, "rknn_bytes": report.get("rknn_bytes"),
            "rknn_sha256": report.get("rknn_sha256"),
            "mean_mae_vs_original_fp32": report.get("mean_mae_vs_original_fp32"),
            "mean_mae_vs_own_fp32": report.get("mean_mae_vs_own_fp32"),
            "elapsed_seconds": report.get("elapsed_seconds")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("export-dir", "calibration-dir", "development-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    tasks = [(args.export_dir, args.calibration_dir, args.development_dir, args.output_dir,
              layer, mode, precision)
             for layer in (1, 11, 13, 15)
             for mode in ("ptq", "qat") for precision in ("fp16", "bf16")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            (args.output_dir / "report.json").write_text(json.dumps({
                "scope": "four expert KV groups, RK3588 FP16/BF16 host simulator",
                "records": sorted(records, key=lambda row: (row["layer"], row["mode"], row["precision"]))
            }, indent=2) + "\n")
            print(f"{len(records)}/{len(tasks)} {record}", flush=True)
    if any(row["status"] != "success" for row in records):
        raise RuntimeError("Some KV float compilations failed")


if __name__ == "__main__":
    main()
