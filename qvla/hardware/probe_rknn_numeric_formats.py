#!/usr/bin/env python3
"""Check Toolkit2 config acceptance of MX names; no hardware-support claim."""

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

from rknn.api import RKNN


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    attempts = [("quantized_dtype", name) for name in
                ("w8a8", "w8a16", "w4a8", "w4a16", "mx4", "mx8",
                 "mxfp4", "mxfp8", "mxint8")]
    attempts += [("float_dtype", name) for name in
                 ("float16", "bfloat16", "mx4", "mx8", "mxfp4", "mxfp8")]
    attempts += [("quantized_method", "group32")]
    rows = []
    for field, value in attempts:
        rknn = RKNN(verbose=False)
        row = {"field": field, "value": value, "target_platform": "rk3588"}
        try:
            row["config_return"] = rknn.config(target_platform="rk3588", **{field: value})
        except Exception as exc:
            row["exception"] = f"{type(exc).__name__}: {exc}"
        finally:
            rknn.release()
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {"scope": "Toolkit2 config validation only, no ONNX build or RK3588 execution",
              "toolkit_version": "2.3.2", "attempts": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
