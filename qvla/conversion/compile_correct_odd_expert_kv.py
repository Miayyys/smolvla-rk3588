#!/usr/bin/env python3
"""Build true separate-input K/V RKNN graphs for odd expert layers."""

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
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_one(task):
    args, layer, kind, mode, precision = task
    group = f"layer_{layer:02d}"
    dataset_root = args.k_calibration_dir if kind == "k" else args.v_calibration_dir
    development_root = args.k_development_dir if kind == "k" else args.v_development_dir
    input_kind = "kv" if kind == "k" else "v"
    onnx = args.export_dir / group / kind / mode / "projection_fp32.onnx"
    reference = args.export_dir / group / kind / "ptq/projection_fp32.onnx"
    dataset = dataset_root / group / input_kind / "rknn_dataset.txt"
    development_inputs = development_root / group / input_kind / "heldout_inputs.txt"
    development_report = development_root / "report.json"
    loaded_dir = args.loaded_dir / group / "kv"
    output = args.output_dir / group / kind / mode / precision
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(source_path('compile_rknn_projection_with_eval.py')),
               "--onnx", str(onnx), "--reference-onnx", str(reference),
               "--dataset", str(dataset), "--development-inputs", str(development_inputs),
               "--development-report", str(development_report),
               "--loaded-outputs-dir", str(loaded_dir), "--loaded-output-half", kind,
               "--output-dir", str(output), "--precision", precision]
    with (output / "compile_and_eval.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    path = output / "report.json"
    report = json.loads(path.read_text()) if path.is_file() else {}
    return {"layer": layer, "kind": kind, "mode": mode, "precision": precision,
            "status": report.get("status", "failed") if result.returncode == 0 else "failed",
            "returncode": result.returncode, "samples": report.get("samples"),
            "rknn_bytes": report.get("rknn_bytes"),
            "rknn_sha256": report.get("rknn_sha256"),
            "mean_mae_vs_original_fp32": report.get("mean_mae_vs_original_fp32"),
            "mean_mae_vs_original_loaded_bf16": report.get("mean_mae_vs_original_loaded_bf16")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("export-dir", "k-calibration-dir", "v-calibration-dir",
                 "k-development-dir", "v-development-dir", "loaded-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--precision", choices=("w8a8", "bf16", "fp16"), default="w8a8")
    args = p.parse_args()
    reports = [json.loads((directory / "report.json").read_text()) for directory in
               (args.export_dir, args.k_calibration_dir, args.v_calibration_dir,
                args.k_development_dir, args.v_development_dir, args.loaded_dir)]
    partition = reports[0]["partition_sha256"]
    source = reports[0]["source_weight_sha256"]
    if any(report["partition_sha256"] != partition or
           report["source_weight_sha256"] != source for report in reports):
        raise ValueError("Export/input identity mismatch")
    if (reports[1]["split"] != "calibration" or reports[2]["split"] != "calibration"
            or reports[3]["split"] != "development" or reports[4]["split"] != "development"):
        raise ValueError("Calibration and development partitions are mixed")
    tasks = [(args, layer, kind, mode, args.precision) for layer in range(1, 16, 2)
             for kind in ("k", "v") for mode in ("ptq", "qat")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            (args.output_dir / f"report_{args.precision}.json").write_text(json.dumps({
                "scope": "correct separate-input odd expert K/V, RK3588 host simulator",
                "source_weight_sha256": source, "partition_sha256": partition,
                "loaded_report_sha256": sha256(args.loaded_dir / "report.json"),
                "precision": args.precision,
                "records": sorted(records, key=lambda r: (r["layer"], r["kind"], r["mode"]))
            }, indent=2) + "\n")
            print(f"{len(records)}/{len(tasks)} {record}", flush=True)
    if any(record["status"] != "success" or record["samples"] != 240 for record in records):
        raise RuntimeError("At least one corrected K/V build or evaluation failed")


if __name__ == "__main__":
    main()
