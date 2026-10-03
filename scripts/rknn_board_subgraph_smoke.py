#!/usr/bin/env python3
"""Run a saved RKNN subgraph on a Rockchip board and record latency/parity.

This intentionally measures one RKNN graph through RKNN-Toolkit-Lite2. It does
not claim that the full SmolVLA policy runs on the board.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import statistics
import time
from pathlib import Path

import numpy as np
from rknnlite.api import RKNNLite


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mem_available_kib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    return None


def rss_max_kib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True,
                        help="One held-out float32 .npy activation")
    parser.add_argument("--reference", type=Path,
                        help="Matching original-FP output .npy for numerical parity")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--data-format", choices=("nchw", "nhwc"),
                        help="Explicit layout of the supplied 4D input")
    parser.add_argument("--save-output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be >= 0 and --repeats must be >= 1")

    sample = np.load(args.input, allow_pickle=False)
    if sample.dtype != np.float32:
        sample = sample.astype(np.float32)
    sample = np.ascontiguousarray(sample)
    expected = None
    if args.reference is not None:
        expected = np.load(args.reference, allow_pickle=False).astype(np.float32)

    report = {
        "scope": "single RKNN subgraph on RK3588 board",
        "status": "running",
        "model": str(args.model),
        "model_bytes": args.model.stat().st_size,
        "model_sha256": sha256(args.model),
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "input_shape": list(sample.shape),
        "input_dtype": str(sample.dtype),
        "input_data_format": args.data_format,
        "reference": str(args.reference) if args.reference else None,
        "reference_sha256": sha256(args.reference) if args.reference else None,
        "warmup_runs": args.warmup,
        "timed_runs": args.repeats,
        "npu_core_mask": "NPU_CORE_0",
        "board": platform.platform(),
        "python": platform.python_version(),
        "rknn_lite_version": importlib.metadata.version("rknn-toolkit-lite2"),
        "mem_available_kib_before": mem_available_kib(),
        "rusage_maxrss_kib_before": rss_max_kib(),
    }
    rknn = RKNNLite()
    try:
        ret = rknn.load_rknn(str(args.model))
        if ret != 0:
            raise RuntimeError(f"RKNNLite.load_rknn returned {ret}")
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if ret != 0:
            raise RuntimeError(f"RKNNLite.init_runtime returned {ret}")

        outputs = None
        def infer():
            # Lite2 mutates the format list to internal integer enums.
            options = {"data_format": [args.data_format]} if args.data_format else {}
            return rknn.inference(inputs=[sample], **options)
        for _ in range(args.warmup):
            outputs = infer()
        durations_ms = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            outputs = infer()
            durations_ms.append((time.perf_counter() - start) * 1000.0)
        if outputs is None or len(outputs) != 1:
            raise RuntimeError("Expected one non-empty RKNN output")
        actual = np.asarray(outputs[0], dtype=np.float32)
        if args.save_output:
            args.save_output.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.save_output, actual, allow_pickle=False)
        report.update({
            "status": "success",
            "output_shape": list(actual.shape),
            "output_dtype": str(actual.dtype),
            "durations_ms": durations_ms,
            "latency_ms": {
                "mean": statistics.fmean(durations_ms),
                "p50": float(np.percentile(durations_ms, 50)),
                "p95": float(np.percentile(durations_ms, 95)),
                "min": min(durations_ms),
                "max": max(durations_ms),
            },
            "latency_scope": "RKNNLite.inference call; excludes model loading and preprocessing",
        })
        if expected is not None:
            if actual.shape != expected.shape:
                raise ValueError(f"Output/reference shape mismatch: {actual.shape} vs {expected.shape}")
            delta = actual - expected
            report["parity_vs_reference"] = {
                "mae": float(np.mean(np.abs(delta))),
                "rmse": float(np.sqrt(np.mean(np.square(delta)))),
                "max_abs_error": float(np.max(np.abs(delta))),
                "reference_shape": list(expected.shape),
            }
    except Exception as exc:
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        report.update({
            "mem_available_kib_after": mem_available_kib(),
            "rusage_maxrss_kib_after": rss_max_kib(),
        })
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        rknn.release()
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
