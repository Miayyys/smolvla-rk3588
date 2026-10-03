#!/usr/bin/env python3
"""Pack all 112 expert Linear weights as actual W8A8 integer-GEMM model files."""

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

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from qvla.evaluation.probe_action_sensitivity import sha256


def pack_linear(name, source, ranges, mode, master):
    weight_key = name + ".weight"
    source_weight = (master.get_tensor(weight_key) if mode == "qat"
                     else source.get_tensor(weight_key)).float().contiguous()
    if source_weight.ndim != 2 or source_weight.shape[0] % 8 or source_weight.shape[1] % 8:
        raise ValueError(f"INT8 GEMM shape unsupported: {name} {source_weight.shape}")
    weight_scale = source_weight.abs().amax(dim=1).clamp_min(1e-10) / 127
    weight_q_t = torch.round(source_weight / weight_scale[:, None]).clamp(-127, 127).to(torch.int8).t().contiguous()
    record = ranges[name]
    low, high = float(record["min"]), float(record["max"])
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError(f"Invalid activation range for {name}")
    input_scale = max((high - low) / 255, 1e-12)
    input_zero = int(np.clip(np.rint(-128 - low / input_scale), -128, 127))
    result = {name + ".weight_q_t": weight_q_t,
              name + ".weight_scale": weight_scale.contiguous(),
              name + ".weight_sum": weight_q_t.int().sum(dim=0).contiguous(),
              name + ".activation_scale": torch.tensor(input_scale, dtype=torch.float32),
              name + ".activation_zero": torch.tensor(input_zero, dtype=torch.int32)}
    bias_key = name + ".bias"
    if bias_key in source.keys():
        bias = (master.get_tensor(bias_key) if mode == "qat" else source.get_tensor(bias_key))
        result[bias_key] = bias.float().contiguous()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "qat-snapshot", "calibration-ranges", "partition", "map", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    partition = json.loads(args.partition.read_text())
    target = json.loads(args.map.read_text())
    source_path = args.model_dir / "model.safetensors"
    if (partition["source_weight_sha256"] != sha256(source_path)
            or target["evaluation_partition_sha256"] != sha256(args.partition)):
        raise ValueError("Frozen source/map identity mismatch")
    ranges = json.loads(args.calibration_ranges.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    with safe_open(str(source_path), framework="pt", device="cpu") as source, \
         safe_open(str(args.qat_snapshot), framework="pt", device="cpu") as master:
        names = sorted({key[:-7] for key in master.keys() if key.endswith(".weight")})
        if len(names) != target["selected_linear_count"] or set(names) != set(ranges):
            raise ValueError("QAT master, map and calibration ranges differ")
        for name in names:
            if name + ".weight" not in source.keys():
                raise ValueError(f"Original weight missing: {name}")
        excluded = {name + suffix for name in names for suffix in (".weight", ".bias")}
        common = {key: source.get_tensor(key).contiguous()
                  for key in source.keys() if key not in excluded}
        for mode in ("ptq", "qat"):
            state = dict(common)
            for i, name in enumerate(names):
                state.update(pack_linear(name, source, ranges, mode, master))
                if (i + 1) % 28 == 0:
                    print(f"{mode}: packed {i+1}/{len(names)} linears", flush=True)
            path = args.output_dir / f"expert_{mode}_real_w8a8.safetensors"
            save_file(state, str(path))
            outputs[mode] = {"path": str(path), "sha256": sha256(path),
                             "bytes": path.stat().st_size,
                             "ratio_vs_original_file": path.stat().st_size / source_path.stat().st_size}
            del state
    report = {"scope": "full checkpoint with 112 expert Linear INT8 weights and static INT8 activation metadata",
              "runtime": "torch._int_mm CUDA; full action reload must be verified separately",
              "source_weight_sha256": partition["source_weight_sha256"],
              "source_weight_bytes": source_path.stat().st_size,
              "qat_snapshot_sha256": sha256(args.qat_snapshot),
              "calibration_ranges_sha256": sha256(args.calibration_ranges),
              "partition_sha256": sha256(args.partition), "map_sha256": sha256(args.map),
              "selected_linears": len(names), "names": names,
              "quantization": {"activation": "asymmetric per-tensor INT8, fixed calibration min/max",
                               "weight": "symmetric per-output-channel INT8, range [-127,127]",
                               "accumulator": "INT32, dequantized to FP32 then original dtype"},
              "outputs": outputs}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_linears": len(names), "outputs": outputs}), flush=True)


if __name__ == "__main__":
    main()
