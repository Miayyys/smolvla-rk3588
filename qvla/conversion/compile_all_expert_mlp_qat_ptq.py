#!/usr/bin/env python3
"""Compile 16 expert MLPs for QAT/PTQ with bounded CPU parallelism and resume."""

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
import shutil
import subprocess
import sys
import time
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compile_one(task):
    layer, mode, onnx, dataset, output, compiler, python = task
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "mlp_int8_mmse_rk3588.rknn"
    marker = output / "done.json"
    source_hash = sha256(onnx)
    dataset_hash = sha256(dataset)
    if marker.is_file() and artifact.is_file():
        old = json.loads(marker.read_text())
        if (old.get("onnx_sha256") == source_hash and
                old.get("dataset_sha256") == dataset_hash and
                old.get("rknn_sha256") == sha256(artifact)):
            return old | {"resumed": True}
    command = [python, str(compiler), "--onnx", str(onnx), "--output-dir", str(output),
               "--dataset", str(dataset), "--mode", "int8", "--algorithm", "mmse"]
    started = time.perf_counter()
    with (output / "compile.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    record = {"layer": layer, "mode": mode, "onnx": str(onnx),
              "onnx_sha256": source_hash, "dataset": str(dataset),
              "dataset_sha256": dataset_hash, "command": command,
              "elapsed_seconds": time.perf_counter() - started,
              "returncode": result.returncode, "resumed": False}
    if result.returncode == 0 and artifact.is_file():
        record.update(status="success", rknn=str(artifact),
                      rknn_bytes=artifact.stat().st_size, rknn_sha256=sha256(artifact))
        marker.write_text(json.dumps(record, indent=2) + "\n")
    else:
        record["status"] = "failed"
    return record


def reuse_previous_layer0(args):
    """Reuse validated previous layer-0 builds when all 240 calibration arrays match."""
    old_dir = Path("runs/rknn_expert_mlp_calibration_v2")
    old_list = old_dir / "rknn_dataset.txt"
    new_list = args.calibration_dir / "layer_00" / "rknn_dataset.txt"
    if not old_list.is_file() or not new_list.is_file():
        return
    old_paths = [Path(line) for line in old_list.read_text().splitlines()]
    new_paths = [Path(line) for line in new_list.read_text().splitlines()]
    if len(old_paths) != len(new_paths) or len(new_paths) != 240:
        return
    if any(not old.is_file() or not new.is_file() or sha256(old) != sha256(new)
           for old, new in zip(old_paths, new_paths)):
        return
    for mode, old_output in (("ptq", Path("runs/qat_ptq_expert0_original")),
                             ("qat", Path("runs/qat_ptq_expert0_qat50"))):
        onnx = args.export_dir / "layer_00" / mode / "mlp_fp32.onnx"
        old_onnx = old_output / "expert_layer0_mlp_fp32.onnx"
        old_rknn = old_output / "expert_layer0_mlp_int8_mmse_rk3588.rknn"
        old_report = old_output / "int8_mmse_compile_report.json"
        if not all(path.is_file() for path in (onnx, old_onnx, old_rknn, old_report)):
            continue
        report = json.loads(old_report.read_text())
        if (sha256(onnx) != sha256(old_onnx) or report.get("status") != "success"
                or report.get("quantized_algorithm") != "mmse"
                or report.get("quantized_dtype") != "w8a8"
                or report.get("quantized_method") != "channel"
                or report.get("optimization_level") != 3):
            continue
        target = args.output_dir / "layer_00" / mode
        target.mkdir(parents=True, exist_ok=True)
        copied = target / "mlp_int8_mmse_rk3588.rknn"
        shutil.copyfile(old_rknn, copied)
        marker = {"layer": 0, "mode": mode, "onnx": str(onnx.resolve()),
                  "onnx_sha256": sha256(onnx), "dataset": str(new_list.resolve()),
                  "dataset_sha256": sha256(new_list), "elapsed_seconds": 0.0,
                  "returncode": 0, "resumed": True, "reused_from": str(old_rknn.resolve()),
                  "status": "success", "rknn": str(copied.resolve()),
                  "rknn_bytes": copied.stat().st_size, "rknn_sha256": sha256(copied)}
        (target / "done.json").write_text(json.dumps(marker, indent=2) + "\n")
        print(f"Reused prior layer 0 {mode}: all 240 calibration arrays and recipe match", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("export-dir", "calibration-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if not 1 <= args.workers <= 8:
        p.error("workers must be 1..8")
    compiler = source_path('compile_rknn_mlp_probe.py').resolve()
    export_report = json.loads((args.export_dir / "report.json").read_text())
    calibration_report = json.loads((args.calibration_dir / "report.json").read_text())
    if (export_report["source_weight_sha256"] != calibration_report["source_weight_sha256"]
            or export_report["partition_sha256"] != calibration_report["partition_sha256"]
            or calibration_report["split"] != "calibration"
            or calibration_report["samples_per_layer"] != 240):
        raise ValueError("Export/calibration identity mismatch")
    reuse_previous_layer0(args)
    tasks = []
    for layer in range(16):
        dataset = args.calibration_dir / f"layer_{layer:02d}" / "rknn_dataset.txt"
        if len(dataset.read_text().splitlines()) != 240:
            raise ValueError(f"Missing calibration inputs for layer {layer}")
        for mode in ("ptq", "qat"):
            onnx = args.export_dir / f"layer_{layer:02d}" / mode / "mlp_fp32.onnx"
            if not onnx.is_file():
                raise FileNotFoundError(onnx)
            tasks.append((layer, mode, onnx.resolve(), dataset.resolve(),
                          (args.output_dir / f"layer_{layer:02d}" / mode).resolve(),
                          compiler, sys.executable))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(compile_one, task): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            report = {"scope": "16 expert MLPs, independent QAT and original-FP PTQ W8A8",
                      "workers": args.workers, "tasks": len(tasks),
                      "source_weight_sha256": export_report["source_weight_sha256"],
                      "qat_snapshot_sha256": export_report["qat_snapshot_sha256"],
                      "partition_sha256": export_report["partition_sha256"],
                      "records": sorted(records, key=lambda r: (r["layer"], r["mode"]))}
            temporary = args.output_dir / "report.json.tmp"
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(args.output_dir / "report.json")
            print(f"{len(records)}/{len(tasks)} layer {record['layer']} {record['mode']}: "
                  f"{record['status']} in {record['elapsed_seconds']:.1f}s", flush=True)
    failures = [record for record in records if record["status"] != "success"]
    if failures:
        raise RuntimeError(f"{len(failures)} RKNN builds failed; see report and per-job logs")


if __name__ == "__main__":
    main()
