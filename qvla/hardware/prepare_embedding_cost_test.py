#!/usr/bin/env python3
"""Prepare a real-weight CPU token-embedding lookup comparison for RK3588."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import hashlib
import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/hardware_embedding_v1"
SOURCE = ROOT / "artifacts/transfer/model/model.safetensors"
NAME = "model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight"
SEED = 20260929
LENGTH = 177  # Same sequence length as the downstream captured text MLP probe.


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with SOURCE.open("rb") as f:
        header_bytes = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_bytes))
        spec = header[NAME]
        if spec["dtype"] != "BF16" or spec["shape"] != [49280, 960]:
            raise ValueError(f"Unexpected embedding tensor: {spec}")
        data_start = 8 + header_bytes
        f.seek(data_start + spec["data_offsets"][0])
        raw = f.read(spec["data_offsets"][1] - spec["data_offsets"][0])
    if len(raw) != 94617600:
        raise ValueError(f"Unexpected raw tensor bytes: {len(raw)}")
    bf16 = np.frombuffer(raw, dtype="<u2").reshape(spec["shape"]).copy()
    original = (bf16.astype(np.uint32) << 16).view(np.float32)
    scale = np.maximum(np.max(np.abs(original), axis=1), 1e-10) / 127.0
    q = np.clip(np.rint(original / scale[:, None]), -127, 127).astype(np.int8)
    rng = np.random.default_rng(SEED)
    ids = rng.integers(0, spec["shape"][0], size=(1, LENGTH), dtype=np.int32)
    native = (bf16[ids].astype(np.uint32) << 16).view(np.float32)
    dequant = q[ids].astype(np.float32) * scale[ids, None]
    # The reference runtime stores/returns BF16 activations after embedding lookup.
    quantized = ((dequant.view(np.uint32) >> 16) << 16).view(np.float32)
    np.save(OUT / "embedding_bf16.npy", bf16)
    np.save(OUT / "embedding_int8.npy", q)
    np.save(OUT / "embedding_scale.npy", scale.astype(np.float32))
    np.save(OUT / "token_ids.npy", ids)
    np.save(OUT / "reference_bf16.npy", native)
    np.save(OUT / "reference_int8_row.npy", quantized)
    report = {
        "source_checkpoint": "artifacts/transfer/model/model.safetensors",
        "source_checkpoint_sha256": sha(SOURCE),
        "tensor": NAME,
        "source_dtype": "BF16",
        "source_shape": spec["shape"],
        "source_tensor_bytes": len(raw),
        "int8_method": "symmetric row-wise: scale=max(abs(row))/127; round-to-nearest; clamp[-127,127]",
        "int8_weight_bytes": q.nbytes,
        "scale_bytes": scale.nbytes,
        "packed_int8_plus_scales_bytes": q.nbytes + scale.nbytes,
        "input_ids_shape": list(ids.shape),
        "input_ids_seed": SEED,
        "sequence_length_basis": "177, matching text MLP held-out activation length; token ID list is synthetic cost-only",
        "native_lookup_output_bytes_float32_expanded": native.nbytes,
        "int8_lookup_output_bytes_float32_expanded": quantized.nbytes,
        "lookup_error_vs_bf16": {
            "mae": float(np.mean(np.abs(quantized - native))),
            "rmse": float(np.sqrt(np.mean((quantized - native) ** 2))),
            "max_abs": float(np.max(np.abs(quantized - native))),
        },
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)}
                  for p in sorted(OUT.glob("*.npy"))},
    }
    (OUT / "prepare_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
