#!/usr/bin/env python3
"""Generate RKNN step-2 QKV hybrid configs from an unchanged step-1 config."""

import argparse
import hashlib
import json
from pathlib import Path


NODES = {
    "q": "/q_proj/MatMul_output_0_mm",
    "k": "/k_proj/MatMul_output_0_mm",
    "v": "/v_proj/MatMul_output_0_mm",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step1-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.step1_config.read_text()
    heading = "custom_quantize_layers: {}\n"
    if not source.startswith(heading):
        raise ValueError("Expected untouched RKNN hybrid step-1 config")
    for name in NODES.values():
        if f"    {name}:\n" not in source:
            raise ValueError(f"Missing calibrated output {name}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for selected in ("q", "k", "v", "qk", "qv", "kv"):
        entries = "".join(f"    {NODES[key]}: float16\n" for key in selected)
        text = source.replace(heading, "custom_quantize_layers:\n" + entries, 1)
        path = args.output_dir / f"rknn_expert_qkv0_mmse_{selected}_fp16.cfg"
        path.write_text(text)
        manifest.append({"candidate": selected, "fp16_outputs": [NODES[k] for k in selected],
                         "config": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
