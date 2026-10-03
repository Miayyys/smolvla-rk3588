#!/usr/bin/env python3
"""Capture shape-correct attention projection inputs from all 16 expert layers."""

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

from qvla.evaluation.probe_action_sensitivity import run_action, selected_frames, sha256
from qvla.quantization.qat_train_w8a8_stage1 import load_policy


PREFIX = "model.vlm_with_expert.lm_expert.layers"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "vlm-assets-dir", "dataset-root", "splits", "partition", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--split", choices=("calibration", "development"), required=True)
    p.add_argument("--frames-per-task", type=int, default=2)
    p.add_argument("--step-samples", type=int, default=3)
    p.add_argument("--odd-v-only", action="store_true",
                   help="Capture separate V inputs on odd expert layers")
    args = p.parse_args()
    partition = json.loads(args.partition.read_text())
    if (partition["source_weight_sha256"] != sha256(args.model_dir / "model.safetensors")
            or partition["source_split_sha256"] != sha256(args.splits)):
        raise ValueError("Frozen model/split identity mismatch")
    ids = (partition["calibration_episode_ids_from_ptq_calibration"] if args.split == "calibration"
           else partition["development_episode_ids_from_qat_train"])
    rows = selected_frames(args.dataset_root, ids, args.frames_per_task, 40)
    if len(rows) != 40 * args.frames_per_task:
        raise ValueError("Expected one episode per 40 tasks")
    policy, pre, post = load_policy(args)
    if len(policy.get_submodule(PREFIX)) != 16:
        raise ValueError("Unexpected expert layer count")
    groups = ([(layer, "v") for layer in range(1, 16, 2)] if args.odd_v_only else
              [(layer, kind) for layer in range(16)
               for kind in (("qkv", "out") if layer % 2 == 0 else ("q", "kv", "out"))])
    captured = {group: [] for group in groups}
    handles = []
    for layer, kind in groups:
        projection = {"qkv": "q_proj", "q": "q_proj", "kv": "k_proj",
                      "v": "v_proj", "out": "o_proj"}[kind]
        module = f"{PREFIX}.{layer}.self_attn.{projection}"
        def capture(_module, inputs, group=(layer, kind)):
            captured[group].append(inputs[0].detach().float().cpu().numpy())
        handles.append(policy.get_submodule(module).register_forward_pre_hook(capture))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lists = {group: [] for group in groups}
    records = []
    try:
        for i, row in enumerate(rows):
            for values in captured.values():
                values.clear()
            run_action(policy, pre, post, row)
            if any(len(values) != 10 for values in captured.values()):
                raise ValueError(f"Unexpected attention call count at row {i}")
            indices = np.linspace(0, 9, args.step_samples, dtype=int)
            for group, values in captured.items():
                layer, kind = group
                directory = args.output_dir / f"layer_{layer:02d}" / kind
                directory.mkdir(parents=True, exist_ok=True)
                for rank, index in enumerate(indices):
                    array = values[index]
                    original_length = array.shape[1]
                    if kind in ("kv", "v"):
                        if (array.ndim != 3 or array.shape[0] != 1 or
                                array.shape[2] != 320 or original_length > 256):
                            raise ValueError(f"KV input cannot fit fixed 256 tokens: {array.shape}")
                        array = np.pad(array, ((0, 0), (0, 256 - original_length), (0, 0)))
                    expected_shape = ((1, 50, 960) if kind == "out" else
                                      (1, 256, 320) if kind in ("kv", "v") else (1, 50, 720))
                    if array.shape != expected_shape or not np.isfinite(array).all():
                        raise ValueError(f"Invalid attention input {group}: {array.shape}")
                    path = directory / (f"task_{row['task_index']:02d}_frame_{row['frame_rank']:02d}_step_{rank:02d}.npy")
                    np.save(path, array, allow_pickle=False)
                    lists[group].append(str(path.resolve()))
                    records.append({"layer": layer, "kind": kind,
                                    "task_index": row["task_index"],
                                    "episode_index": row["episode_index"],
                                    "frame_rank": row["frame_rank"], "step_rank": rank,
                                    "attention_call_index": int(index),
                                    "original_token_length": int(original_length),
                                    "file": str(path), "sha256": sha256(path)})
            if (i + 1) % 10 == 0:
                print(f"{args.split}: {i+1}/{len(rows)} attention rows captured", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    list_name = "rknn_dataset.txt" if args.split == "calibration" else "heldout_inputs.txt"
    for (layer, kind), paths in lists.items():
        (args.output_dir / f"layer_{layer:02d}" / kind / list_name).write_text("\n".join(paths) + "\n")
    report = {"scope": ("odd expert V projection inputs, independent of K" if args.odd_v_only else
                        "16 expert attention layers: even QKV; odd Q and legacy KV; all output projections"),
              "odd_kv_padding": "zero-pad token dimension to 256; original_token_length recorded; Linear output must be cropped",
              "split": args.split, "partition_sha256": sha256(args.partition),
              "source_weight_sha256": partition["source_weight_sha256"],
              "source_split_sha256": sha256(args.splits),
              "frames_per_task": args.frames_per_task, "step_samples": args.step_samples,
              "samples_per_group": len(next(iter(lists.values()))), "groups": len(groups),
              "records": records}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {len(records)} attention inputs across {len(groups)} groups", flush=True)


if __name__ == "__main__":
    main()
