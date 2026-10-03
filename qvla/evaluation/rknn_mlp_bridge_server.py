#!/usr/bin/env python3
"""Serve two RK3588 MLP W8A8 host simulators to a LeRobot process.

Toolkit2 2.3.2 cannot reload exported RKNN files in its host simulator. We
rebuild each ONNX with the same frozen recipe, and verify the exported files.
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
import hashlib
import json
import time
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
from rknn.api import RKNN


SHAPE = (1, 50, 720)
OUTPUT_SHAPE = (1, 50, 720)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(onnx, dataset):
    instance = RKNN(verbose=False)
    operations = (
        ("config", lambda: instance.config(target_platform="rk3588", quantized_algorithm="mmse",
                                           quantized_method="channel", quantized_dtype="w8a8")),
        ("load_onnx", lambda: instance.load_onnx(model=str(onnx))),
        ("build", lambda: instance.build(do_quantization=True, dataset=str(dataset))),
        ("init_runtime", instance.init_runtime),
    )
    for name, operation in operations:
        status = operation()
        if status != 0:
            raise RuntimeError(f"RKNN {name} failed for {onnx}: {status}")
    return instance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptq-onnx", type=Path, required=True)
    parser.add_argument("--ptq-rknn", type=Path, required=True)
    parser.add_argument("--qat-onnx", type=Path, required=True)
    parser.add_argument("--qat-rknn", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.ptq_onnx, args.ptq_rknn, args.qat_onnx, args.qat_rknn, args.dataset):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.socket.parent.mkdir(parents=True, exist_ok=True)
    if args.socket.exists():
        raise FileExistsError(f"Remove stale socket after verifying no server runs: {args.socket}")
    models = {}
    report = {"scope": "host simulator rebuilt from same ONNX, calibration and config as exported RKNN",
              "board_execution": "not_measured", "recipe": {
                  "target_platform": "rk3588", "quantized_algorithm": "mmse",
                  "quantized_method": "channel", "quantized_dtype": "w8a8",
                  "do_quantization": True, "input_shape": SHAPE,
                  "calibration_dataset": str(args.dataset), "calibration_dataset_sha256": sha256(args.dataset)},
              "models": {}, "connections": 0}
    try:
        for key, onnx, rknn in (("P", args.ptq_onnx, args.ptq_rknn),
                                ("Q", args.qat_onnx, args.qat_rknn)):
            started = time.perf_counter()
            models[key] = build(onnx, args.dataset)
            report["models"][key] = {"onnx": str(onnx), "onnx_sha256": sha256(onnx),
                                      "rknn": str(rknn), "rknn_sha256": sha256(rknn),
                                      "rknn_bytes": rknn.stat().st_size,
                                      "build_seconds": time.perf_counter() - started,
                                      "inference_count": 0, "simulator_seconds": 0.0}
            print(f"READY_MODEL {key}", flush=True)
        with Listener(str(args.socket), family="AF_UNIX") as listener:
            print(f"READY_SOCKET {args.socket}", flush=True)
            stopping = False
            while not stopping:
                with listener.accept() as conn:
                    report["connections"] += 1
                    while True:
                        try:
                            payload = conn.recv_bytes()
                        except EOFError:
                            break
                        if payload == b"X":
                            conn.send_bytes(b"X")
                            stopping = True
                            break
                        key = payload[:1].decode("ascii")
                        if key not in models or len(payload) != 1 + np.prod(SHAPE) * 4:
                            raise ValueError("Invalid bridge request")
                        sample = np.frombuffer(payload, dtype="<f4", offset=1).reshape(SHAPE).copy()
                        if not np.isfinite(sample).all():
                            raise ValueError("Non-finite bridge input")
                        started = time.perf_counter()
                        outputs = models[key].inference(inputs=[sample])
                        elapsed = time.perf_counter() - started
                        if outputs is None or len(outputs) != 1:
                            raise RuntimeError("RKNN simulator inference failed")
                        output = np.asarray(outputs[0], dtype="<f4")
                        if output.shape != OUTPUT_SHAPE or not np.isfinite(output).all():
                            raise ValueError(f"Invalid RKNN output: {output.shape}")
                        report["models"][key]["inference_count"] += 1
                        report["models"][key]["simulator_seconds"] += elapsed
                        conn.send_bytes(output.tobytes())
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        for instance in models.values():
            instance.release()
        if args.socket.exists():
            args.socket.unlink()
        print("SERVER_STOPPED", flush=True)


if __name__ == "__main__":
    main()
