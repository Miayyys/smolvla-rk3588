#!/usr/bin/env python3
"""Report safetensors storage dtypes without loading tensor data."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))


import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import struct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.weights.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(length))
    counts = defaultdict(lambda: {"tensors": 0, "elements": 0, "data_bytes": 0})
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        row = counts[tensor["dtype"]]
        row["tensors"] += 1
        elements = 1
        for dim in tensor["shape"]:
            elements *= dim
        row["elements"] += elements
        row["data_bytes"] += tensor["data_offsets"][1] - tensor["data_offsets"][0]
    sha = hashlib.sha256()
    with args.weights.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(block)
    result = {"weights": str(args.weights), "sha256": sha.hexdigest(),
              "file_bytes": args.weights.stat().st_size, "storage_dtypes": dict(counts)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
