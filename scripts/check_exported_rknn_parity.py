#!/usr/bin/env python3
"""Compare an exported RKNN build recipe in the host simulator on held-out inputs.

Toolkit2 cannot load_rknn into the host simulator. This rebuilds from ONNX with
the same config and calibration inputs; the exported file is checked for size.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from rknn.api import RKNN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rknn", "onnx", "reference-onnx", "inputs", "output", "dataset"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--loaded-outputs", type=Path,
                        help="Original loaded-runtime outputs with the same input basenames")
    args = parser.parse_args()
    paths = [Path(line) for line in args.inputs.read_text().splitlines() if line.strip()]
    if not paths or any(not p.is_file() for p in paths):
        raise ValueError("Missing held-out input")
    candidate = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    reference = ort.InferenceSession(str(args.reference_onnx), providers=["CPUExecutionProvider"])
    rknn = RKNN(verbose=False)
    if rknn.config(target_platform="rk3588", quantized_algorithm="mmse",
                   quantized_method="channel", quantized_dtype="w8a8",
                   float_dtype="float16", optimization_level=3) != 0:
        raise RuntimeError("RKNN config failed")
    if rknn.load_onnx(model=str(args.onnx)) != 0:
        raise RuntimeError("RKNN ONNX load failed")
    if rknn.build(do_quantization=True, dataset=str(args.dataset)) != 0:
        raise RuntimeError("RKNN simulator rebuild failed")
    if rknn.init_runtime() != 0:
        raise RuntimeError("RKNN host runtime failed")
    records = []
    try:
        for path in paths:
            sample = np.load(path, allow_pickle=False).astype(np.float32)
            fp = reference.run(None, {reference.get_inputs()[0].name: sample})[0]
            master = candidate.run(None, {candidate.get_inputs()[0].name: sample})[0]
            outputs = rknn.inference(inputs=[sample])
            if outputs is None or len(outputs) != 1:
                raise RuntimeError(f"RKNN inference failed on {path}")
            quantized = np.asarray(outputs[0], dtype=np.float32).reshape(fp.shape)
            loaded_error = None
            if args.loaded_outputs:
                loaded = np.load(args.loaded_outputs / path.name, allow_pickle=False).astype(np.float32)
                if loaded.shape != quantized.shape:
                    raise ValueError(f"Loaded output shape differs for {path.name}")
                loaded_error = float(np.mean(np.abs(quantized - loaded)))
            records.append({"input": str(path),
                            "master_mae_vs_original_fp": float(np.mean(np.abs(master - fp))),
                            "rknn_mae_vs_master": float(np.mean(np.abs(quantized - master))),
                            "rknn_mae_vs_original_fp": float(np.mean(np.abs(quantized - fp))),
                            "rknn_mae_vs_original_loaded": loaded_error})
    finally:
        rknn.release()
    report = {"scope": "one subgraph, host simulator rebuilt using export recipe",
              "execution": "rebuilt from ONNX; Toolkit2 2.3.2 cannot reload exported RKNN in host simulator",
              "board_execution": "not_measured", "rknn": str(args.rknn),
              "rknn_bytes": args.rknn.stat().st_size, "onnx": str(args.onnx),
              "reference_onnx": str(args.reference_onnx), "inputs": str(args.inputs),
              "loaded_outputs": str(args.loaded_outputs) if args.loaded_outputs else None,
              "calibration_dataset": str(args.dataset),
              "samples": len(records),
              "mean_master_mae_vs_original_fp": float(np.mean([r["master_mae_vs_original_fp"] for r in records])),
              "mean_rknn_mae_vs_master": float(np.mean([r["rknn_mae_vs_master"] for r in records])),
              "mean_rknn_mae_vs_original_fp": float(np.mean([r["rknn_mae_vs_original_fp"] for r in records])),
              "mean_rknn_mae_vs_original_loaded": (
                  float(np.mean([r["rknn_mae_vs_original_loaded"] for r in records]))
                  if args.loaded_outputs else None),
              "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
