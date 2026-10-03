#!/usr/bin/env python3
"""Measure native-BF16 and row-wise-INT8 CPU embedding lookup on RK3588."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import argparse
import hashlib
import json
import platform
import resource
import statistics
import time
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--rounds", type=int, default=3)
    args = p.parse_args()
    root = args.root.resolve()
    bf16_path, q_path = root / "embedding_bf16.npy", root / "embedding_int8.npy"
    scale_path, ids_path = root / "embedding_scale.npy", root / "token_ids.npy"
    bf16 = np.load(bf16_path, mmap_mode="r", allow_pickle=False)
    q = np.load(q_path, mmap_mode="r", allow_pickle=False)
    scale = np.load(scale_path, mmap_mode="r", allow_pickle=False)
    ids = np.load(ids_path, allow_pickle=False)
    if bf16.shape != (49280, 960) or q.shape != bf16.shape or scale.shape != (49280,):
        raise ValueError("Unexpected embedding table shape")

    def native_lookup():
        rows = np.take(bf16, ids, axis=0)
        return (rows.astype(np.uint32) << 16).view(np.float32)

    def int8_lookup():
        rows = np.take(q, ids, axis=0)
        row_scale = np.take(scale, ids, axis=0)[..., None]
        values = rows.astype(np.float32) * row_scale
        return (((values.view(np.uint32) >> 16) << 16).view(np.float32))

    fn = {"native_bf16": native_lookup, "cpu_int8_row_lookup": int8_lookup}
    case_info = {
        "native_bf16": {"table_bytes": bf16.nbytes,
                        "serialized_file_bytes": bf16_path.stat().st_size,
                        "weight_precision": "BF16", "operation": "row gather + BF16 decode to FP32"},
        "cpu_int8_row_lookup": {"table_bytes": q.nbytes + scale.nbytes,
                                "serialized_file_bytes": q_path.stat().st_size + scale_path.stat().st_size,
                                "weight_precision": "INT8 per-row scale", "operation": "row gather + dequantize + BF16 output rounding"},
    }
    result = {name: {"durations_ms": [], "round_p50_ms": []} for name in fn}
    for rnd in range(args.rounds):
        order = list(fn) if rnd % 2 == 0 else list(reversed(fn))
        for name in order:
            lookup = fn[name]
            for _ in range(args.warmup):
                lookup()
            times = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                output = lookup()
                times.append((time.perf_counter() - start) * 1000.0)
            result[name]["durations_ms"].extend(times)
            result[name]["round_p50_ms"].append(float(np.percentile(times, 50)))
            result[name]["last_output_shape"] = list(output.shape)
            result[name]["last_output_sha256"] = hashlib.sha256(output.tobytes()).hexdigest()
            result[name]["mae_vs_bf16_reference"] = float(np.mean(np.abs(
                output - np.load(root / "reference_bf16.npy", allow_pickle=False))))
            print(rnd, name, "done", flush=True)
    report = {
        "status": "success",
        "scope": "CPU NumPy embedding gather/dequant only; excludes model loading, tokenizer, and CPU-to-NPU transfer",
        "board": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
        "cpu_affinity": sorted(__import__("os").sched_getaffinity(0)),
        "input_shape": list(ids.shape), "input_ids_sha256": sha(ids_path),
        "warmup": args.warmup, "repeats_per_round": args.repeats, "rounds": args.rounds,
        "checkpoint_sha256": json.loads((root / "prepare_report.json").read_text())["source_checkpoint_sha256"],
        "tensor_sha256": {p.name: sha(p) for p in (bf16_path, q_path, scale_path)},
        "cases": {},
    }
    for name, record in result.items():
        all_times = record["durations_ms"]
        round_p50 = record["round_p50_ms"]
        report["cases"][name] = {
            **case_info[name], **record,
            "p50_ms": float(np.percentile(all_times, 50)),
            "p95_ms": float(np.percentile(all_times, 95)),
            "mean_ms": statistics.fmean(all_times),
            "timing_stable_20pct": max(round_p50) / min(round_p50) <= 1.2,
        }
    out = root / "board_embedding_lookup.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
