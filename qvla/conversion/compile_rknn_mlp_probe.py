#!/usr/bin/env python3
"""Compile a real SmolVLA single-input subgraph for RK3588 feasibility testing.

This host-side probe cannot establish board execution or full-VLA coverage.
The INT8 dataset must contain captured calibration activations, never test data.
"""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))


import argparse
import json
from pathlib import Path
import time

from rknn.api import RKNN


def require_ok(stage: str, result: int) -> None:
    if result != 0:
        raise RuntimeError(f"RKNN {stage} failed with code {result}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path,
                        help="RKNN calibration list; required for INT8")
    parser.add_argument("--mode", choices=("fp16", "bf16", "int8"), required=True)
    parser.add_argument("--algorithm", choices=("normal", "mmse", "kl_divergence"),
                        default="normal", help="RKNN INT8 calibration algorithm")
    parser.add_argument("--optimization-level", type=int, choices=(0, 1, 2, 3), default=3,
                        help="RKNN compiler optimization level; default 3")
    args = parser.parse_args()
    if args.mode == "int8" and (args.dataset is None or not args.dataset.is_file()):
        parser.error("INT8 requires an existing --dataset")
    if args.mode != "int8" and args.algorithm != "normal":
        parser.error("--algorithm only applies to INT8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    label = args.mode + (f"_{args.algorithm}" if args.algorithm != "normal" else "")
    stem = ("expert_layer0_mlp" if args.onnx.name == "expert_layer0_mlp_fp32.onnx"
            else "mlp")
    output = args.output_dir / f"{stem}_{label}_rk3588.rknn"
    report = {"scope": "one SmolVLA submodule only", "platform": "rk3588",
              "mode": args.mode, "quantized_algorithm": args.algorithm,
              "optimization_level": args.optimization_level,
              "quantized_method": "channel", "quantized_dtype": "w8a8",
              "float_dtype": "bfloat16" if args.mode == "bf16" else "float16",
              "onnx": str(args.onnx.resolve()),
              "calibration_dataset": str(args.dataset.resolve()) if args.dataset else None,
              "board_execution": "not_measured"}
    rknn = RKNN(verbose=True)
    started = time.monotonic()
    try:
        require_ok("config", rknn.config(target_platform="rk3588",
                                          quantized_algorithm=args.algorithm,
                                          quantized_method="channel",
                                          quantized_dtype="w8a8", float_dtype=report["float_dtype"],
                                          optimization_level=args.optimization_level))
        require_ok("load_onnx", rknn.load_onnx(model=str(args.onnx)))
        require_ok("build", rknn.build(do_quantization=args.mode == "int8",
                                        dataset=str(args.dataset) if args.dataset else None))
        require_ok("export_rknn", rknn.export_rknn(str(output)))
        report.update(status="success", rknn_path=str(output.resolve()),
                      rknn_bytes=output.stat().st_size)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        (args.output_dir / f"{label}_compile_report.json").write_text(
            json.dumps(report, indent=2) + "\n")
        rknn.release()
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
