#!/usr/bin/env python3
"""Compile and numerically probe one RKNN hybrid-quantized MLP candidate."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
from rknn.api import RKNN


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_ok(stage: str, result: int) -> None:
    if result != 0:
        raise RuntimeError(f"RKNN {stage} failed with code {result}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-stem", default="expert_layer0_mlp")
    parser.add_argument("--scope", default="one action-expert MLP")
    parser.add_argument("--loaded-outputs", type=Path,
                        help="Original loaded-runtime outputs with matching input basenames")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.output_stem}_hybrid_{args.label}_rk3588.rknn"
    report_path = args.output_dir / f"hybrid_{args.label}_development_parity.json"
    paths = [Path(line) for line in args.inputs.read_text().splitlines() if line.strip()]
    if not paths:
        parser.error("No development inputs")
    config_text = args.config.read_text()
    if "custom_quantize_layers: {}" in config_text:
        parser.error("No FP16 layer selected in config")
    report = {"scope": args.scope, "stage": "hybrid_step2",
              "execution": "RKNN host simulator; RK3588 board not measured",
              "label": args.label, "config": str(args.config.resolve()),
              "config_sha256": sha256(args.config), "input_count": len(paths)}
    rknn = RKNN(verbose=True)
    started = time.monotonic()
    try:
        require_ok("hybrid_quantization_step2", rknn.hybrid_quantization_step2(
            model_input=str(args.model.resolve(strict=True)),
            data_input=str(args.data.resolve(strict=True)),
            model_quantization_cfg=str(args.config.resolve(strict=True))))
        require_ok("export_rknn", rknn.export_rknn(str(output)))
        report.update(rknn_bytes=output.stat().st_size, rknn_sha256=sha256(output))
        require_ok("init_runtime", rknn.init_runtime())
        onnx = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
        records = []
        for path in paths:
            sample = np.load(path, allow_pickle=False).astype(np.float32)
            expected = onnx.run(None, {onnx.get_inputs()[0].name: sample})[0]
            outputs = rknn.inference(inputs=[sample])
            if outputs is None or len(outputs) != 1:
                raise RuntimeError(f"RKNN inference failed: {path}")
            actual = np.asarray(outputs[0], dtype=np.float32).reshape(expected.shape)
            delta = actual - expected
            loaded_error = None
            if args.loaded_outputs:
                loaded = np.load(args.loaded_outputs / path.name, allow_pickle=False).astype(np.float32)
                if loaded.shape != actual.shape:
                    raise ValueError(f"Loaded output shape differs for {path.name}")
                loaded_error = float(np.mean(np.abs(actual - loaded)))
            records.append({"input": path.name,
                            "mae": float(np.mean(np.abs(delta))),
                            "mae_vs_original_loaded": loaded_error,
                            "rmse": float(np.sqrt(np.mean(delta ** 2))),
                            "max_abs": float(np.max(np.abs(delta))),
                            "cosine": float(np.dot(actual.ravel(), expected.ravel()) /
                                            (np.linalg.norm(actual) * np.linalg.norm(expected)))})
        report.update(status="success", samples=len(records),
              mean_mae=float(np.mean([row["mae"] for row in records])),
              mean_mae_vs_original_loaded=(float(np.mean([row["mae_vs_original_loaded"] for row in records]))
                                           if args.loaded_outputs else None),
              loaded_outputs=str(args.loaded_outputs) if args.loaded_outputs else None,
                      max_mae=max(row["mae"] for row in records),
                      min_cosine=min(row["cosine"] for row in records), records=records)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        rknn.release()
        print(json.dumps({key: value for key, value in report.items() if key != "records"}), flush=True)


if __name__ == "__main__":
    main()
