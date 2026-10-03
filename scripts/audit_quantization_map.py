#!/usr/bin/env python3
"""Audit a provisional quantization map against every safetensors weight.

Reports source checkpoint bytes only. It does not estimate packed model size,
RK3588 RAM, latency, action quality, or feasibility.
"""

import argparse
import hashlib
import json
import re
import struct
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, default=Path("config/quantization_map_v0.json"))
    parser.add_argument("--weights", type=Path,
                        default=Path("artifacts/transfer/model/model.safetensors"))
    parser.add_argument("--output", type=Path,
                        default=Path("figures/quantization_map_v0_inventory.json"))
    args = parser.parse_args()
    spec = json.loads(args.map.read_text())
    actual_hash = sha256(args.weights)
    if actual_hash != spec["checkpoint_weight_sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {actual_hash}")
    groups = spec["groups"]
    ids = [group["id"] for group in groups]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate group ID")
    patterns = [(group, re.compile(group["pattern"])) for group in groups]
    with args.weights.open("rb") as stream:
        header_len = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_len))
    rows = {group["id"]: {"source_tensor_count": 0, "source_elements": 0,
                          "source_payload_bytes": 0, "source_dtypes": Counter(),
                          "v0_format": group["v0_format"],
                          "next_probe": group["next_probe"]} for group in groups}
    total_tensors = 0
    total_bytes = 0
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        matches = [group["id"] for group, pattern in patterns if pattern.match(name)]
        if len(matches) != 1:
            raise ValueError(f"Tensor must match exactly one group: {name}: {matches}")
        row = rows[matches[0]]
        nbytes = tensor["data_offsets"][1] - tensor["data_offsets"][0]
        elements = 1
        for dimension in tensor["shape"]:
            elements *= dimension
        row["source_tensor_count"] += 1
        row["source_elements"] += elements
        row["source_payload_bytes"] += nbytes
        row["source_dtypes"][tensor["dtype"]] += 1
        total_tensors += 1
        total_bytes += nbytes
    for group in groups:
        if rows[group["id"]]["source_tensor_count"] == 0:
            raise ValueError(f"Group has no source tensors: {group['id']}")
    result = {
        "scope": "source_safetensors_inventory_only_not_quantization_benefit",
        "map_version": spec["version"],
        "checkpoint_weight_sha256": actual_hash,
        "source_file_bytes": args.weights.stat().st_size,
        "source_payload_bytes": total_bytes,
        "source_tensor_count": total_tensors,
        "groups": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Audited {total_tensors} tensors in {len(groups)} groups; source payload {total_bytes:,} B")
    for name, row in rows.items():
        print(f"{name:26s} {row['source_payload_bytes']:>11,} B  {row['v0_format']}")
    print(f"Written {args.output}")


if __name__ == "__main__":
    main()
