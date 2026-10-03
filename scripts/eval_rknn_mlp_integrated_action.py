#!/usr/bin/env python3
"""Run full SmolVLA action chunks with one real RKNN host-simulator MLP."""

import argparse
import json
import time
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np
import torch
from torch import nn
from safetensors import safe_open

from probe_action_sensitivity import run_action, selected_frames, sha256
from qat_train_w8a8_stage1 import QATLinear, load_policy, selected_linears


MLP_NAME = "model.vlm_with_expert.lm_expert.layers.0.mlp"


class BridgeMLP(nn.Module):
    def __init__(self, connection, model_key):
        super().__init__()
        self.connection = connection
        self.model_key = model_key.encode("ascii")
        self.calls = 0
        self.roundtrip_seconds = 0.0

    def forward(self, x):
        if tuple(x.shape) != (1, 50, 720):
            raise ValueError(f"Unexpected MLP input shape {tuple(x.shape)}")
        sample = np.ascontiguousarray(x.detach().float().cpu().numpy(), dtype="<f4")
        started = time.perf_counter()
        self.connection.send_bytes(self.model_key + sample.tobytes())
        raw = self.connection.recv_bytes()
        self.roundtrip_seconds += time.perf_counter() - started
        if len(raw) != sample.nbytes:
            raise ValueError(f"Unexpected bridge response size {len(raw)}")
        output = np.frombuffer(raw, dtype="<f4").copy().reshape(sample.shape)
        if not np.isfinite(output).all():
            raise ValueError("Non-finite bridge output")
        self.calls += 1
        return torch.from_numpy(output).to(device=x.device, dtype=x.dtype)


def score(rows, actions, original, own_float):
    return {
        "chunk_mae_vs_original_fp": float(np.abs(actions - original).mean()),
        "task_chunk_mae_vs_original_fp": np.abs(actions - original).mean(axis=(1, 2)).tolist(),
        "chunk_mae_vs_own_float": float(np.abs(actions - own_float).mean()),
        "task_chunk_mae_vs_own_float": np.abs(actions - own_float).mean(axis=(1, 2)).tolist(),
        "first_action_mae_vs_recorded": float(np.mean([
            np.abs(actions[i, 0] - row["recorded_action"]).mean()
            for i, row in enumerate(rows)])),
        "task_first_action_mae_vs_recorded": [float(np.abs(actions[i, 0] - row["recorded_action"]).mean())
                                               for i, row in enumerate(rows)],
    }


def evaluate(policy, pre, post, rows, conn, key):
    parent_name, leaf = MLP_NAME.rsplit(".", 1)
    parent = policy.get_submodule(parent_name)
    previous = getattr(parent, leaf)
    bridge = BridgeMLP(conn, key)
    setattr(parent, leaf, bridge)
    try:
        started = time.perf_counter()
        actions = []
        for i, row in enumerate(rows):
            actions.append(run_action(policy, pre, post, row))
            if (i + 1) % 10 == 0:
                print(f"{key} actions {i+1}/{len(rows)}", flush=True)
        elapsed = time.perf_counter() - started
    finally:
        setattr(parent, leaf, previous)
    if bridge.calls != 10 * len(rows):
        raise ValueError(f"Expected {10 * len(rows)} RKNN calls, got {bridge.calls}")
    return np.stack(actions), {"calls": bridge.calls,
                               "host_wall_seconds": elapsed,
                               "bridge_roundtrip_seconds": bridge.roundtrip_seconds}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "vlm-assets-dir", "dataset-root", "splits", "partition",
                 "map", "calibration-ranges", "qat-snapshot", "proxy-report", "proxy-arrays",
                 "socket", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--max-tasks", type=int, default=40)
    p.add_argument("--stop-server", action="store_true")
    args = p.parse_args()
    if not 1 <= args.max_tasks <= 40:
        p.error("--max-tasks must be 1..40")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partition = json.loads(args.partition.read_text())
    target = json.loads(args.map.read_text())
    proxy = json.loads(args.proxy_report.read_text())
    identities = ((partition["source_split_sha256"], sha256(args.splits)),
                  (partition["source_weight_sha256"], sha256(args.model_dir / "model.safetensors")),
                  (target["evaluation_partition_sha256"], sha256(args.partition)),
                  (proxy["arrays_sha256"], sha256(args.proxy_arrays)),
                  (proxy["qat_snapshot_sha256"], sha256(args.qat_snapshot)),
                  (proxy["map_sha256"], sha256(args.map)),
                  (proxy["calibration_ranges_sha256"], sha256(args.calibration_ranges)))
    if any(expected != actual for expected, actual in identities):
        raise ValueError("Frozen model, split, map, QAT or proxy identity mismatch")
    rows = selected_frames(args.dataset_root, partition["development_episode_ids_from_qat_train"], 1, 40)
    rows = rows[:args.max_tasks]
    if len(rows) != args.max_tasks:
        raise ValueError("Missing development rows")
    with np.load(args.proxy_arrays, allow_pickle=False) as archive:
        original = archive["original"][:args.max_tasks]
        qat_master = archive["qat_master"][:args.max_tasks]
        if (not np.array_equal(archive["episode_index"][:args.max_tasks],
                               [r["episode_index"] for r in rows])
                or not np.array_equal(archive["task_index"][:args.max_tasks],
                                      [r["task_index"] for r in rows])):
            raise ValueError("Saved FP actions do not match selected episodes")
    policy, pre, post = load_policy(args)
    fresh = run_action(policy, pre, post, rows[0])
    baseline_parity = float(np.max(np.abs(fresh - original[0])))
    if baseline_parity > 1e-5:
        raise ValueError(f"Saved FP baseline changed: max diff {baseline_parity}")
    with Client(str(args.socket), family="AF_UNIX") as conn:
        ptq, ptq_timing = evaluate(policy, pre, post, rows, conn, "P")
        selected = selected_linears(policy)
        if len(selected) != target["selected_linear_count"]:
            raise ValueError("Selected QAT Linear count changed")
        ranges = json.loads(args.calibration_ranges.read_text())
        if set(ranges) != set(selected):
            raise ValueError("QAT activation ranges changed")
        for name, linear in selected.items():
            parent, leaf = name.rsplit(".", 1)
            wrapper = QATLinear(linear, ranges[name]["min"], ranges[name]["max"])
            wrapper.fake_quant_enabled = False
            setattr(policy.get_submodule(parent), leaf, wrapper)
        with safe_open(str(args.qat_snapshot), framework="pt", device="cpu") as source:
            expected = {f"{name}.{suffix}" for name in selected for suffix in
                        (["weight", "bias"] if policy.get_submodule(name).bias is not None else ["weight"])}
            if set(source.keys()) != expected:
                raise ValueError("QAT snapshot parameter names changed")
            for name in selected:
                module = policy.get_submodule(name)
                with torch.no_grad():
                    module.weight.copy_(source.get_tensor(name + ".weight").to(module.weight.device))
                    if module.bias is not None:
                        module.bias.copy_(source.get_tensor(name + ".bias").to(module.bias.device))
        fresh_qat = run_action(policy, pre, post, rows[0])
        qat_parity = float(np.max(np.abs(fresh_qat - qat_master[0])))
        if qat_parity > 1e-5:
            raise ValueError(f"Saved QAT master baseline changed: max diff {qat_parity}")
        qat, qat_timing = evaluate(policy, pre, post, rows, conn, "Q")
        if args.stop_server:
            conn.send_bytes(b"X")
            if conn.recv_bytes() != b"X":
                raise ValueError("Server stop acknowledgement failed")
    arrays_path = args.output_dir / "actions.npz"
    np.savez_compressed(arrays_path, original=original, qat_master=qat_master,
                        ptq_integrated=ptq, qat_integrated=qat,
                        recorded_first=np.stack([row["recorded_action"] for row in rows]),
                        task_index=np.array([row["task_index"] for row in rows]),
                        episode_index=np.array([row["episode_index"] for row in rows]))
    result = {"scope": "full action inference with one RKNN simulator MLP; other modules float",
              "board_execution": "not_measured", "closed_loop": "not_measured",
              "reference": "original loaded BF16 policy, fixed pre/postprocessor and noise",
              "source_weight_sha256": partition["source_weight_sha256"],
              "split_sha256": sha256(args.splits), "partition_sha256": sha256(args.partition),
              "map_sha256": sha256(args.map), "qat_snapshot_sha256": sha256(args.qat_snapshot),
              "proxy_arrays_sha256": sha256(args.proxy_arrays), "tasks": len(rows),
              "development_episode_ids": [row["episode_index"] for row in rows],
              "baseline_parity_max_abs": baseline_parity,
              "qat_float_master_parity_max_abs": qat_parity,
              "baseline_task_first_action_mae_vs_recorded": [
                  float(np.abs(original[i, 0] - row["recorded_action"]).mean())
                  for i, row in enumerate(rows)],
              "actions": str(arrays_path), "actions_sha256": sha256(arrays_path),
              "ptq": {"scores": score(rows, ptq, original, original), "timing": ptq_timing},
              "qat": {"scores": score(rows, qat, original, qat_master), "timing": qat_timing},
              "torch_version": torch.__version__}
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"tasks": len(rows), "baseline_parity_max_abs": baseline_parity,
                      "qat_float_master_parity_max_abs": qat_parity,
                      "ptq": result["ptq"]["scores"]["first_action_mae_vs_recorded"],
                      "qat": result["qat"]["scores"]["first_action_mae_vs_recorded"],
                      "actions": str(arrays_path)}), flush=True)


if __name__ == "__main__":
    main()
