#!/usr/bin/env python3
"""Run RKNN's official hybrid quantization step 1 on the SmolVLA MLP ONNX."""

import argparse
import json
import os
from pathlib import Path
import time

from rknn.api import RKNN


def require_ok(stage: str, result: int) -> None:
    if result != 0:
        raise RuntimeError(f"RKNN {stage} failed with code {result}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--algorithm", choices=("normal", "mmse", "kl_divergence"),
                        default="mmse")
    args = parser.parse_args()
    onnx = args.onnx.resolve(strict=True)
    dataset = args.dataset.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"scope": "one action-expert MLP", "stage": "hybrid_step1",
              "platform": "rk3588", "algorithm": args.algorithm,
              "quantized_method": "channel", "quantized_dtype": "w8a8",
              "float_dtype": "float16", "onnx": str(onnx), "dataset": str(dataset),
              "proposal": False, "board_execution": "not_measured"}
    rknn = RKNN(verbose=True)
    started = time.monotonic()
    previous = Path.cwd()
    try:
        os.chdir(output)
        require_ok("config", rknn.config(target_platform="rk3588",
                                          quantized_algorithm=args.algorithm,
                                          quantized_method="channel",
                                          quantized_dtype="w8a8", float_dtype="float16"))
        require_ok("load_onnx", rknn.load_onnx(model=str(onnx)))
        require_ok("hybrid_quantization_step1",
                   rknn.hybrid_quantization_step1(dataset=str(dataset), proposal=False))
        report.update(status="success", generated_files=[
            {"name": path.name, "bytes": path.stat().st_size}
            for path in sorted(output.iterdir()) if path.is_file()
            and path.name not in ("hybrid_step1_report.json", "hybrid_step1.log")])
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        (output / "hybrid_step1_report.json").write_text(json.dumps(report, indent=2) + "\n")
        os.chdir(previous)
        rknn.release()
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
