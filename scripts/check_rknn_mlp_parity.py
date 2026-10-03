#!/usr/bin/env python3
"""Compare RKNN simulator output with ONNX on held-out submodule activations."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
from rknn.api import RKNN


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fp16", "bf16", "int8"), required=True)
    parser.add_argument("--algorithm", choices=("normal", "mmse", "kl_divergence"),
                        default="normal")
    parser.add_argument("--dataset", type=Path,
                        help="Calibration list for rebuilding the INT8 simulator model")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--loaded-outputs", type=Path,
                        help="Directory of paired outputs from the originally loaded MLP")
    parser.add_argument("--reference-onnx", type=Path,
                        help="Original FP ONNX for paired QAT/PTQ subgraph comparison")
    args = parser.parse_args()
    if args.mode == "int8" and (args.dataset is None or not args.dataset.is_file()):
        parser.error("INT8 requires an existing --dataset")
    if args.mode != "int8" and args.algorithm != "normal":
        parser.error("--algorithm only applies to INT8")
    float_dtype = "bfloat16" if args.mode == "bf16" else "float16"
    paths = [Path(line) for line in args.inputs.read_text().splitlines() if line.strip()]
    if not paths:
        parser.error("Input list is empty")
    onnx = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    reference = (ort.InferenceSession(str(args.reference_onnx), providers=["CPUExecutionProvider"])
                 if args.reference_onnx else None)
    rknn = RKNN(verbose=False)
    if rknn.config(target_platform="rk3588", quantized_algorithm=args.algorithm,
                   quantized_method="channel", quantized_dtype="w8a8",
                   float_dtype=float_dtype) != 0:
        raise RuntimeError("config failed")
    if rknn.load_onnx(model=str(args.onnx)) != 0:
        raise RuntimeError("load_onnx failed")
    if rknn.build(do_quantization=args.mode == "int8",
                  dataset=str(args.dataset) if args.dataset else None) != 0:
        raise RuntimeError("build failed")
    if rknn.init_runtime() != 0:
        raise RuntimeError("init_runtime failed")
    records = []
    start = time.monotonic()
    try:
        for path in paths:
            sample = np.load(path, allow_pickle=False).astype(np.float32)
            expected = onnx.run(None, {onnx.get_inputs()[0].name: sample})[0]
            outputs = rknn.inference(inputs=[sample])
            if outputs is None or len(outputs) != 1:
                raise RuntimeError(f"RKNN inference failed: {path}")
            actual = np.asarray(outputs[0], dtype=np.float32).reshape(expected.shape)
            delta = actual - expected
            reference_error = None
            master_error = None
            if reference:
                target = reference.run(None, {reference.get_inputs()[0].name: sample})[0]
                reference_error = float(np.mean(np.abs(actual - target)))
                master_error = float(np.mean(np.abs(expected - target)))
            loaded_error = None
            if args.loaded_outputs:
                loaded_path = args.loaded_outputs / path.name
                if not loaded_path.is_file():
                    raise FileNotFoundError(loaded_path)
                loaded = np.load(loaded_path, allow_pickle=False).astype(np.float32)
                if loaded.shape != expected.shape:
                    raise ValueError(f"Loaded output shape differs for {path.name}")
                loaded_error = float(np.mean(np.abs(actual - loaded)))
            records.append({"input": path.name, "mae": float(np.mean(np.abs(delta))),
                            "mae_vs_reference_onnx": reference_error,
                            "master_onnx_mae_vs_reference": master_error,
                            "rmse": float(np.sqrt(np.mean(delta ** 2))),
                            "max_abs": float(np.max(np.abs(delta))),
                            "mae_vs_original_loaded": loaded_error,
                            "cosine": float(np.dot(actual.ravel(), expected.ravel()) /
                                            (np.linalg.norm(actual) * np.linalg.norm(expected)))})
    finally:
        rknn.release()
    report = {"scope": "one SmolVLA submodule; numerical parity only",
              "execution": "RKNN host simulator rebuilt from ONNX; not exported file or RK3588 board",
              "mode": args.mode, "quantized_algorithm": args.algorithm,
              "quantized_method": "channel", "quantized_dtype": "w8a8",
              "float_dtype": float_dtype,
              "calibration_dataset": str(args.dataset) if args.dataset else None,
              "onnx": str(args.onnx),
              "inputs": str(args.inputs), "samples": len(records),
              "mean_mae": float(np.mean([row["mae"] for row in records])),
              "mean_mae_vs_reference_onnx": (float(np.mean([row["mae_vs_reference_onnx"] for row in records]))
                                             if reference else None),
              "mean_master_onnx_mae_vs_reference": (
                  float(np.mean([row["master_onnx_mae_vs_reference"] for row in records]))
                  if reference else None),
              "mean_mae_vs_original_loaded": (
                  float(np.mean([row["mae_vs_original_loaded"] for row in records]))
                  if args.loaded_outputs else None),
              "loaded_outputs": str(args.loaded_outputs) if args.loaded_outputs else None,
              "reference_onnx": str(args.reference_onnx) if args.reference_onnx else None,
              "max_mae": max(row["mae"] for row in records),
              "mean_cosine": float(np.mean([row["cosine"] for row in records])),
              "min_cosine": min(row["cosine"] for row in records),
              "elapsed_seconds": round(time.monotonic() - start, 3),
              "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}), flush=True)


if __name__ == "__main__":
    main()
