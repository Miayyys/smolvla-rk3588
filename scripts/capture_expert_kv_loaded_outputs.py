#!/usr/bin/env python3
"""Capture actual loaded-policy odd-layer K/V outputs and separate V inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from probe_action_sensitivity import run_action, selected_frames, sha256
from qat_train_w8a8_stage1 import load_policy


LAYERS = tuple(range(1, 16, 2))
PREFIX = "model.vlm_with_expert.lm_expert.layers"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "vlm-assets-dir", "dataset-root", "splits", "partition",
                 "development-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    partition = json.loads(args.partition.read_text())
    saved = json.loads((args.development_dir / "report.json").read_text())
    if (saved["split"] != "development" or
            saved["partition_sha256"] != sha256(args.partition) or
            partition["source_weight_sha256"] != sha256(args.model_dir / "model.safetensors") or
            partition["source_split_sha256"] != sha256(args.splits)):
        raise ValueError("Frozen source/development identity mismatch")
    ids = partition["development_episode_ids_from_qat_train"]
    rows = selected_frames(args.dataset_root, ids, 2, 40)
    policy, pre, post = load_policy(args)
    captured = {(layer, kind): [] for layer in LAYERS
                for kind in ("k_input", "v_input", "k", "v")}
    handles = []
    for layer in LAYERS:
        attention = policy.get_submodule(f"{PREFIX}.{layer}.self_attn")
        def input_hook(_module, inputs, layer=layer, kind="k_input"):
            captured[(layer, kind)].append(inputs[0].detach().float().cpu().numpy())
        def output_hook(_module, _inputs, output, layer=layer, kind="k"):
            captured[(layer, kind)].append(output.detach().float().cpu().numpy())
        handles.append(attention.k_proj.register_forward_pre_hook(input_hook))
        handles.append(attention.v_proj.register_forward_pre_hook(
            lambda mod, inp, layer=layer: input_hook(mod, inp, layer, "v_input")))
        handles.append(attention.k_proj.register_forward_hook(output_hook))
        handles.append(attention.v_proj.register_forward_hook(
            lambda mod, inp, out, layer=layer: output_hook(mod, inp, out, layer, "v")))
    records = []
    try:
        for index, row in enumerate(rows):
            for values in captured.values():
                values.clear()
            run_action(policy, pre, post, row)
            if any(len(values) != 10 for values in captured.values()):
                raise ValueError(f"Unexpected KV call count at row {index}")
            for layer in LAYERS:
                for step_rank, call_index in enumerate(np.linspace(0, 9, 3, dtype=int)):
                    basename = (f"task_{row['task_index']:02d}_frame_{row['frame_rank']:02d}_"
                                f"step_{step_rank:02d}.npy")
                    input_path = args.development_dir / f"layer_{layer:02d}" / "kv" / basename
                    saved_input = np.load(input_path, allow_pickle=False)
                    actual_input = captured[(layer, "k_input")][call_index]
                    v_input = captured[(layer, "v_input")][call_index]
                    if (saved_input.shape[1] < actual_input.shape[1] or
                            not np.array_equal(saved_input[:, :actual_input.shape[1]], actual_input)):
                        raise ValueError(f"Loaded input differs from saved development input: {input_path}")
                    loaded = np.concatenate((captured[(layer, "k")][call_index],
                                             captured[(layer, "v")][call_index]), axis=-1)
                    if loaded.shape[1] != actual_input.shape[1] or v_input.shape[1] != actual_input.shape[1]:
                        raise ValueError("KV output token length mismatch")
                    output = args.output_dir / f"layer_{layer:02d}" / "kv" / basename
                    output.parent.mkdir(parents=True, exist_ok=True)
                    np.save(output, loaded, allow_pickle=False)
                    v_input_path = args.output_dir / f"layer_{layer:02d}" / "v_input" / basename
                    v_input_path.parent.mkdir(parents=True, exist_ok=True)
                    if v_input.shape[1] > 256:
                        raise ValueError("V input exceeds fixed 256 tokens")
                    np.save(v_input_path, np.pad(v_input,
                             ((0, 0), (0, 256 - v_input.shape[1]), (0, 0))), allow_pickle=False)
                    records.append({"layer": layer, "task_index": row["task_index"],
                                    "episode_index": row["episode_index"],
                                    "frame_rank": row["frame_rank"], "step_rank": step_rank,
                                    "call_index": int(call_index), "input": str(input_path),
                                    "input_sha256": sha256(input_path),
                                    "output": str(output), "output_sha256": sha256(output),
                                    "v_input": str(v_input_path),
                                    "v_input_sha256": sha256(v_input_path),
                                    "kv_inputs_equal": bool(np.array_equal(actual_input, v_input)),
                                    "shape": list(loaded.shape),
                                    "source_dtype": "torch.bfloat16"})
            if (index + 1) % 10 == 0:
                print(f"captured {index+1}/{len(rows)} rows", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    report = {"scope": "actual loaded KV outputs and separate V inputs for all odd expert layers",
              "source_weight_sha256": partition["source_weight_sha256"],
              "partition_sha256": sha256(args.partition),
              "development_report_sha256": sha256(args.development_dir / "report.json"),
              "samples_per_layer": 240, "records": records}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {len(records)} loaded KV outputs", flush=True)


if __name__ == "__main__":
    main()
