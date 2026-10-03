#!/usr/bin/env python3
"""Compile one attention projection group and evaluate in the same RKNN host runtime."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from rknn.api import RKNN


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_ok(name, value):
    if value != 0:
        raise RuntimeError(f"RKNN {name} failed: {value}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("onnx", "reference-onnx", "dataset", "development-inputs",
                 "development-report", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--precision", choices=("w8a8", "fp16", "bf16"), default="w8a8")
    p.add_argument("--loaded-outputs-dir", type=Path,
                   help="Saved original loaded-model outputs, matched by input basename")
    p.add_argument("--loaded-output-half", choices=("k", "v"),
                   help="Select one half of a saved concatenated K/V output")
    args = p.parse_args()
    paths = [Path(line) for line in args.development_inputs.read_text().splitlines()]
    if len(paths) != 240 or any(not path.is_file() for path in paths):
        raise ValueError("Expected 240 existing development activations")
    development = json.loads(args.development_report.read_text())
    lengths = {str(Path(record["file"]).resolve()): record["original_token_length"]
               for record in development["records"]}
    if any(str(path.resolve()) not in lengths for path in paths):
        raise ValueError("Development report does not cover every input")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = args.output_dir / ("projection_int8_mmse_rk3588.rknn" if args.precision == "w8a8"
                                  else f"projection_{args.precision}_rk3588.rknn")
    report = {"scope": "one attention projection group, RK3588 host simulator",
              "board_execution": "not_measured", "onnx": str(args.onnx),
              "onnx_sha256": sha256(args.onnx),
              "reference_onnx": str(args.reference_onnx),
              "reference_onnx_sha256": sha256(args.reference_onnx),
              "dataset": str(args.dataset), "dataset_sha256": sha256(args.dataset),
              "development_inputs": str(args.development_inputs),
              "development_inputs_sha256": sha256(args.development_inputs),
              "development_report_sha256": sha256(args.development_report),
              "recipe": {"target_platform": "rk3588", "quantized_algorithm": "mmse",
                         "quantized_method": "channel", "quantized_dtype": "w8a8",
                         "optimization_level": 3}}
    if args.precision != "w8a8":
        report["recipe"]["float_dtype"] = "float16" if args.precision == "fp16" else "bfloat16"
    report["precision"] = args.precision
    rknn = RKNN(verbose=False)
    started = time.perf_counter()
    try:
        require_ok("config", rknn.config(**report["recipe"]))
        require_ok("load_onnx", rknn.load_onnx(model=str(args.onnx)))
        require_ok("build", rknn.build(do_quantization=args.precision == "w8a8",
                                        dataset=str(args.dataset) if args.precision == "w8a8" else None))
        require_ok("export_rknn", rknn.export_rknn(str(artifact)))
        require_ok("init_runtime", rknn.init_runtime())
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        reference = ort.InferenceSession(str(args.reference_onnx), options,
                                         providers=["CPUExecutionProvider"])
        candidate = ort.InferenceSession(str(args.onnx), options,
                                         providers=["CPUExecutionProvider"])
        expected_shape = tuple(candidate.get_inputs()[0].shape)
        if expected_shape not in ((1, 256, 320), (1, 50, 720), (1, 50, 960)):
            raise ValueError(f"Unexpected ONNX input shape {expected_shape}")
        records = []
        inference_seconds = 0.0
        for path in paths:
            sample = np.load(path, allow_pickle=False).astype(np.float32)
            if sample.shape != expected_shape:
                raise ValueError(f"Unexpected input shape {sample.shape}")
            original = reference.run(None, {reference.get_inputs()[0].name: sample})[0]
            master = candidate.run(None, {candidate.get_inputs()[0].name: sample})[0]
            t = time.perf_counter()
            result = rknn.inference(inputs=[sample])
            inference_seconds += time.perf_counter() - t
            if result is None or len(result) != 1:
                raise RuntimeError("RKNN simulator inference failed")
            quantized = np.asarray(result[0], dtype=np.float32).reshape(original.shape)
            active_length = lengths[str(path.resolve())]
            if not 1 <= active_length <= sample.shape[1]:
                raise ValueError(f"Invalid active length for {path}")
            original, master, quantized = (array[:, :active_length]
                                           for array in (original, master, quantized))
            loaded_mae = None
            if args.loaded_outputs_dir is not None:
                loaded_path = args.loaded_outputs_dir / path.name
                loaded = np.load(loaded_path, allow_pickle=False).astype(np.float32)
                if args.loaded_output_half is not None:
                    if loaded.shape[-1] != 2 * quantized.shape[-1]:
                        raise ValueError(f"Cannot select {args.loaded_output_half} from {loaded_path}")
                    loaded = (loaded[..., :quantized.shape[-1]] if args.loaded_output_half == "k"
                              else loaded[..., quantized.shape[-1]:])
                if loaded.shape != quantized.shape:
                    raise ValueError(f"Loaded output shape differs: {loaded_path}, {loaded.shape}, {quantized.shape}")
                loaded_mae = float(np.mean(np.abs(quantized - loaded)))
            records.append({"input": str(path),
                            "active_length": active_length,
                            "mae_vs_original_fp32": float(np.mean(np.abs(quantized - original))),
                            "mae_vs_own_fp32": float(np.mean(np.abs(quantized - master))),
                            "master_mae_vs_original_fp32": float(np.mean(np.abs(master - original))),
                            "mae_vs_original_loaded_bf16": loaded_mae})
        report.update({"status": "success", "rknn": str(artifact),
                       "rknn_sha256": sha256(artifact), "rknn_bytes": artifact.stat().st_size,
                       "samples": len(records), "simulator_seconds": inference_seconds,
                       "mean_mae_vs_original_fp32": float(np.mean([r["mae_vs_original_fp32"] for r in records])),
                       "mean_mae_vs_own_fp32": float(np.mean([r["mae_vs_own_fp32"] for r in records])),
                       "loaded_outputs_dir": str(args.loaded_outputs_dir) if args.loaded_outputs_dir else None,
                       "loaded_output_half": args.loaded_output_half,
                       "mean_mae_vs_original_loaded_bf16": (
                           float(np.mean([r["mae_vs_original_loaded_bf16"] for r in records]))
                           if args.loaded_outputs_dir else None),
                       "records": records})
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        rknn.release()
        print(json.dumps({k: v for k, v in report.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
