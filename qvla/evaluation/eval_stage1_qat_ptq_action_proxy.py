#!/usr/bin/env python3
"""Paired 40-task action diagnostic for original FP, PTQ fake W8A8 and QAT fake W8A8.

This is a training-quantizer proxy. It does not execute exported RKNN files.
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
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

from qvla.evaluation.probe_action_sensitivity import run_action, selected_frames, sha256
from qvla.quantization.qat_train_w8a8_stage1 import QATLinear, selected_linears


def score(rows, actions, reference):
    abs_diff = np.abs(actions - reference)
    first_vs_recorded = np.array([
        np.abs(actions[i, 0] - row["recorded_action"]).mean()
        for i, row in enumerate(rows)
    ])
    return {"chunk_mae_vs_original_fp": float(abs_diff.mean()),
            "task_chunk_mae_vs_original_fp": abs_diff.mean(axis=(1, 2)).tolist(),
            "first_action_mae_vs_recorded": float(first_vs_recorded.mean()),
            "task_first_action_mae_vs_recorded": first_vs_recorded.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "vlm-assets-dir", "dataset-root", "splits", "partition",
                 "map", "calibration-ranges", "qat-snapshot", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partition = json.loads(args.partition.read_text())
    target = json.loads(args.map.read_text())
    if (partition["source_split_sha256"] != sha256(args.splits)
            or partition["source_weight_sha256"] != sha256(args.model_dir / "model.safetensors")
            or target["evaluation_partition_sha256"] != sha256(args.partition)):
        raise ValueError("Model/split/map identity mismatch")
    episodes = partition["development_episode_ids_from_qat_train"]
    rows = selected_frames(args.dataset_root, episodes, 1, 40)
    if len(rows) != 40:
        raise ValueError("Expected 40 development rows")
    config = SmolVLAConfig.from_pretrained(args.model_dir)
    config.device = "cuda"
    config.vlm_model_name = str(args.vlm_assets_dir.resolve())
    config.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(args.model_dir, config=config, strict=True).eval()
    pre, post = make_pre_post_processors(
        config, str(args.model_dir),
        preprocessor_overrides={"tokenizer_processor": {
            "tokenizer_name": str(args.vlm_assets_dir.resolve())}},
    )
    original = np.stack([run_action(policy, pre, post, row) for row in rows])
    selected = selected_linears(policy)
    if len(selected) != target["selected_linear_count"]:
        raise ValueError("Unexpected selected Linear count")
    ranges = json.loads(args.calibration_ranges.read_text())
    if set(ranges) != set(selected):
        raise ValueError("Activation ranges do not cover the selected Linears")
    for name, linear in selected.items():
        parent, leaf = name.rsplit(".", 1)
        record = ranges[name]
        setattr(policy.get_submodule(parent), leaf,
                QATLinear(linear, record["min"], record["max"]))
    ptq_fake = np.stack([run_action(policy, pre, post, row) for row in rows])
    with safe_open(str(args.qat_snapshot), framework="pt", device="cpu") as source:
        expected = {f"{name}.{suffix}" for name in selected for suffix in
                    (["weight", "bias"] if policy.get_submodule(name).bias is not None else ["weight"])}
        if set(source.keys()) != expected:
            raise ValueError("QAT snapshot parameter names differ from selected modules")
        for name in selected:
            module = policy.get_submodule(name)
            with torch.no_grad():
                module.weight.copy_(source.get_tensor(name + ".weight").to(module.weight.device))
                if module.bias is not None:
                    module.bias.copy_(source.get_tensor(name + ".bias").to(module.bias.device))
    qat_fake = np.stack([run_action(policy, pre, post, row) for row in rows])
    for name in selected:
        policy.get_submodule(name).fake_quant_enabled = False
    qat_master = np.stack([run_action(policy, pre, post, row) for row in rows])
    array_path = args.output_dir / "actions.npz"
    np.savez_compressed(array_path, original=original, ptq_fake=ptq_fake,
                        qat_fake=qat_fake, qat_master=qat_master,
                        recorded_first=np.stack([r["recorded_action"] for r in rows]),
                        task_index=np.array([r["task_index"] for r in rows]),
                        episode_index=np.array([r["episode_index"] for r in rows]))
    report = {"scope": "full action path, training-quantizer fake W8A8 proxy, not RKNN",
              "source_weight_sha256": partition["source_weight_sha256"],
              "split_sha256": sha256(args.splits), "partition_sha256": sha256(args.partition),
              "map_sha256": sha256(args.map),
              "calibration_ranges_sha256": sha256(args.calibration_ranges),
              "qat_snapshot_sha256": sha256(args.qat_snapshot),
              "development_episode_ids": [r["episode_index"] for r in rows],
              "tasks": len(rows), "selected_linears": len(selected),
              "action_shape": list(original.shape), "arrays": str(array_path),
              "arrays_sha256": sha256(array_path),
              "original_fp": score(rows, original, original),
              "ptq_fake": score(rows, ptq_fake, original),
              "qat_fake": score(rows, qat_fake, original),
              "qat_float_master": score(rows, qat_master, original),
              "torch_version": torch.__version__}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: ({mode: value[mode] for mode in
                            ("chunk_mae_vs_original_fp", "first_action_mae_vs_recorded")}
                            if key in ("original_fp", "ptq_fake", "qat_fake", "qat_float_master")
                            else value) for key, value in report.items()}), flush=True)


if __name__ == "__main__":
    main()
