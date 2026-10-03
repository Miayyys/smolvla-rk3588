#!/usr/bin/env python3
"""Parallel, resumable RK3588 compilation and development evaluation for expert attention."""

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
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def process(task):
    layer, kind, mode, onnx, reference, dataset, inputs, development_report, output, compiler = task
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    artifact = output / "projection_int8_mmse_rk3588.rknn"
    ids = {"onnx_sha256": sha256(onnx), "reference_onnx_sha256": sha256(reference),
           "dataset_sha256": sha256(dataset),
           "development_inputs_sha256": sha256(inputs),
           "development_report_sha256": sha256(development_report)}
    if report_path.is_file() and artifact.is_file():
        old = json.loads(report_path.read_text())
        if (old.get("status") == "success" and all(old.get(k) == v for k, v in ids.items())
                and old.get("rknn_sha256") == sha256(artifact)):
            return {"layer": layer, "kind": kind, "mode": mode,
                    "status": "success", "resumed": True,
                    "rknn_bytes": artifact.stat().st_size,
                    "mean_mae_vs_original_fp32": old["mean_mae_vs_original_fp32"],
                    "elapsed_seconds": old["elapsed_seconds"]}
    command = [sys.executable, str(compiler), "--onnx", str(onnx),
               "--reference-onnx", str(reference), "--dataset", str(dataset),
               "--development-inputs", str(inputs),
               "--development-report", str(development_report), "--output-dir", str(output)]
    with (output / "compile_and_eval.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    return {"layer": layer, "kind": kind, "mode": mode,
            "status": report.get("status", "failed") if result.returncode == 0 else "failed",
            "resumed": False, "returncode": result.returncode,
            "rknn_bytes": report.get("rknn_bytes"),
            "mean_mae_vs_original_fp32": report.get("mean_mae_vs_original_fp32"),
            "elapsed_seconds": report.get("elapsed_seconds")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("export-dir", "calibration-dir", "development-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--start-layer", type=int, default=0)
    p.add_argument("--end-layer", type=int, default=16)
    args = p.parse_args()
    if not 1 <= args.workers <= 8 or not 0 <= args.start_layer < args.end_layer <= 16:
        p.error("Invalid workers or layer range")
    export = json.loads((args.export_dir / "report.json").read_text())
    calibration = json.loads((args.calibration_dir / "report.json").read_text())
    development = json.loads((args.development_dir / "report.json").read_text())
    if (export["partition_sha256"] != calibration["partition_sha256"]
            or export["partition_sha256"] != development["partition_sha256"]
            or export["source_weight_sha256"] != calibration["source_weight_sha256"]
            or export["source_weight_sha256"] != development["source_weight_sha256"]
            or calibration["split"] != "calibration" or development["split"] != "development"
            or calibration["samples_per_group"] != 240
            or development["samples_per_group"] != 240):
        raise ValueError("Export/calibration/development identity mismatch")
    compiler = source_path('compile_rknn_projection_with_eval.py').resolve()
    tasks = []
    for layer in range(args.start_layer, args.end_layer):
        for kind in (("qkv", "out") if layer % 2 == 0 else ("q", "kv", "out")):
            dataset = (args.calibration_dir / f"layer_{layer:02d}" / kind / "rknn_dataset.txt").resolve()
            inputs = (args.development_dir / f"layer_{layer:02d}" / kind / "heldout_inputs.txt").resolve()
            if len(dataset.read_text().splitlines()) != 240 or len(inputs.read_text().splitlines()) != 240:
                raise ValueError(f"Unexpected input count for {layer} {kind}")
            reference = (args.export_dir / f"layer_{layer:02d}" / kind / "ptq" / "projection_fp32.onnx").resolve()
            for mode in ("ptq", "qat"):
                onnx = (args.export_dir / f"layer_{layer:02d}" / kind / mode / "projection_fp32.onnx").resolve()
                output = (args.output_dir / f"layer_{layer:02d}" / kind / mode).resolve()
                tasks.append((layer, kind, mode, onnx, reference, dataset, inputs,
                              (args.development_dir / "report.json").resolve(), output, compiler))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    summary = args.output_dir / f"report_layers_{args.start_layer:02d}_{args.end_layer:02d}.json"
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            data = {"scope": "expert attention QKV/output projection RK3588 W8A8, host simulator",
                    "workers": args.workers, "start_layer": args.start_layer,
                    "end_layer": args.end_layer, "tasks": len(tasks),
                    "partition_sha256": export["partition_sha256"],
                    "source_weight_sha256": export["source_weight_sha256"],
                    "qat_snapshot_sha256": export["qat_snapshot_sha256"],
                    "records": sorted(records, key=lambda r: (r["layer"], r["kind"], r["mode"]))}
            temporary = summary.with_suffix(".tmp")
            temporary.write_text(json.dumps(data, indent=2) + "\n")
            temporary.replace(summary)
            print(f"{len(records)}/{len(tasks)} layer {record['layer']} {record['kind']} {record['mode']}: "
                  f"{record['status']} in {record['elapsed_seconds']}s", flush=True)
    if any(record["status"] != "success" for record in records):
        raise RuntimeError("At least one attention build failed; see reports and logs")


if __name__ == "__main__":
    main()
