#!/usr/bin/env python3
"""Check checkpoint module participation during paired SmolVLA action inference."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

from probe_action_sensitivity import run_action, selected_frames


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--vlm-assets-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int, default=40)
    args = parser.parse_args()
    if not 1 <= args.max_tasks <= 40:
        parser.error("--max-tasks must be 1..40")
    partition = json.loads(args.partition.read_text())
    if partition["source_split_sha256"] != sha256(args.splits):
        raise ValueError("Split hash mismatch")
    if partition["source_weight_sha256"] != sha256(args.model_dir / "model.safetensors"):
        raise ValueError("Weight hash mismatch")
    selected = partition["development_episode_ids_from_qat_train"]
    split = json.loads(args.splits.read_text())["splits"]
    if len(selected) != 40 or not set(selected) <= set(split["qat_train"]):
        raise ValueError("Invalid development episodes")
    rows = [row for row in selected_frames(args.dataset_root, selected, 1, 40)
            if row["task_index"] < args.max_tasks]
    config = SmolVLAConfig.from_pretrained(args.model_dir)
    config.device = "cuda"
    config.vlm_model_name = str(args.vlm_assets_dir.resolve())
    config.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(args.model_dir, config=config, strict=True).eval()
    preprocess, postprocess = make_pre_post_processors(
        config, str(args.model_dir), preprocessor_overrides={"tokenizer_processor": {
            "tokenizer_name": str(args.vlm_assets_dir.resolve())}})
    names = {
        "lm_head": "model.vlm_with_expert.vlm.lm_head",
        "token_embedding": "model.vlm_with_expert.vlm.model.text_model.embed_tokens",
        "connector": "model.vlm_with_expert.vlm.model.connector.modality_projection.proj",
        "action_time_in": "model.action_time_mlp_in",
        "action_time_out": "model.action_time_mlp_out",
        "vision_mlp_11": "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.11.mlp",
        "language_mlp_3": "model.vlm_with_expert.vlm.model.text_model.layers.3.mlp",
    }
    counts = {key: 0 for key in names}
    per_task = []
    handles = []
    current = None

    def hook(key):
        def count(_module, _inputs, _output):
            counts[key] += 1
            current[key] += 1
        return count

    try:
        for key, name in names.items():
            handles.append(policy.get_submodule(name).register_forward_hook(hook(key)))
        for i, row in enumerate(rows, 1):
            current = {key: 0 for key in names}
            action = run_action(policy, preprocess, postprocess, row)
            per_task.append({"task_index": row["task_index"],
                             "episode_index": row["episode_index"],
                             "action_shape": list(action.shape), "calls": current})
            if i % 10 == 0:
                print(f"action path {i}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    embed = policy.get_submodule(names["token_embedding"]).weight
    head = policy.get_submodule(names["lm_head"]).weight
    result = {"scope": "actual_action_chunk_module_liveness_not_board_execution",
              "weight_sha256": sha256(args.model_dir / "model.safetensors"),
              "split_sha256": sha256(args.splits),
              "partition_sha256": sha256(args.partition),
              "development_episodes": [r["episode_index"] for r in rows],
              "modules": names, "calls": counts, "per_task": per_task,
              "lm_head_and_embedding_same_storage":
                  bool(head.untyped_storage().data_ptr() == embed.untyped_storage().data_ptr()),
              "lm_head_and_embedding_same_parameter": bool(head is embed),
              "torch": torch.__version__, "cuda": torch.version.cuda}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"calls": counts,
                      "lm_head_and_embedding_same_storage":
                          result["lm_head_and_embedding_same_storage"]}), flush=True)


if __name__ == "__main__":
    main()
