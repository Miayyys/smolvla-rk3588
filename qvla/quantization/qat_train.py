#!/usr/bin/env python3
"""Task-balanced SmolVLA action-expert QAT, followed by real W8 packing."""

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
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from torchao.quantization import Int8WeightOnlyConfig, quantize_
from torchao.quantization.qat import (
    FakeQuantizeConfig, FromIntXQuantizationAwareTrainingConfig,
    IntXQuantizationAwareTrainingConfig,
)


EXPERT_PREFIX = "model.vlm_with_expert.lm_expert"
GROUP_SIZE = 16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expert_linear(module: nn.Module, name: str) -> bool:
    return (name.startswith(EXPERT_PREFIX) and isinstance(module, nn.Linear)
            and module.in_features % GROUP_SIZE == 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--vlm-assets-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--evaluation-partition", type=Path,
                        help="Exclude development episodes from QAT training")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    split = json.loads(args.splits.read_text())
    train_episodes = split["splits"]["qat_train"]
    partition_hash = None
    if args.evaluation_partition is not None:
        partition = json.loads(args.evaluation_partition.read_text())
        if partition["source_split_sha256"] != sha256(args.splits):
            raise ValueError("Evaluation partition does not match dataset split")
        if partition["source_weight_sha256"] != sha256(args.model_dir / "model.safetensors"):
            raise ValueError("Evaluation partition does not match source weights")
        development = set(partition["development_episode_ids_from_qat_train"])
        if not development or not development <= set(train_episodes):
            raise ValueError("Development episodes are missing from QAT train split")
        train_episodes = [episode for episode in train_episodes if episode not in development]
        partition_hash = sha256(args.evaluation_partition)
    dataset = LeRobotDataset(
        "lerobot/libero", root=args.dataset_root, episodes=train_episodes,
        delta_timestamps={"action": [i / 10 for i in range(50)]},
        video_backend="pyav", return_uint8=True,
    )
    by_task = defaultdict(list)
    episode_starts = []
    offset = 0
    for episode in train_episodes:
        length = dataset.meta.episodes["length"][episode]
        raw = dataset.get_raw_item(offset)
        if int(raw["episode_index"]) != episode or int(raw["frame_index"]) != 0:
            raise ValueError(f"Unexpected train episode order at {offset}")
        by_task[int(raw["task_index"])].append((offset, length, episode))
        episode_starts.append(offset)
        offset += length
    if offset != len(dataset) or len(by_task) != len(split["task_counts"]):
        raise ValueError("Training dataset coverage mismatch")

    config = SmolVLAConfig.from_pretrained(args.model_dir)
    config.device = "cuda"
    config.vlm_model_name = str(args.vlm_assets_dir.resolve())
    config.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(args.model_dir, config=config, strict=True)
    preprocess, _ = make_pre_post_processors(
        config, str(args.model_dir),
        preprocessor_overrides={"tokenizer_processor": {
            "tokenizer_name": str(args.vlm_assets_dir.resolve())}},
    )
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    selected = [name for name, module in policy.named_modules() if expert_linear(module, name)]
    if len(selected) != 112:
        raise ValueError(f"Expected 112 action expert Linear layers, got {len(selected)}")
    quantize_(policy, IntXQuantizationAwareTrainingConfig(
        weight_config=FakeQuantizeConfig(torch.int8, group_size=GROUP_SIZE)),
        filter_fn=expert_linear)
    for name, parameter in policy.named_parameters():
        if name.startswith(EXPERT_PREFIX):
            parameter.requires_grad_(True)
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    policy.train()
    policy.model.vlm_with_expert.vlm.eval()
    torch.cuda.reset_peak_memory_stats()
    task_ids = sorted(by_task)
    losses = []
    began = time.perf_counter()
    for step in range(args.steps):
        task_index = task_ids[step % len(task_ids)]
        start, length, episode = rng.choice(by_task[task_index])
        frame = dataset[start + rng.randrange(length)]
        observation = {key: frame[key] for key in (
            "observation.state", "action", "task", "action_is_pad") if key in frame}
        for key in ("observation.images.image", "observation.images.image2"):
            observation[key] = frame[key].float() / 255.0
        batch = preprocess(observation)
        if batch["action"].ndim == 2:
            batch["action"] = batch["action"].unsqueeze(0)
        if "action_is_pad" in batch and batch["action_is_pad"].ndim == 1:
            batch["action_is_pad"] = batch["action_is_pad"].unsqueeze(0)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = policy.forward(batch)
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss at step {step}: {loss}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % 20 == 0 or step + 1 == args.steps:
            print(f"step {step + 1}/{args.steps}: mean_loss_last20={sum(losses[-20:])/len(losses[-20:]):.6f}, "
                  f"elapsed_s={time.perf_counter()-began:.1f}", flush=True)

    # Fake quant is a training aid. Pack actual W8 tensors and save a reloadable
    # checkpoint with all scales and tensor subclass metadata.
    policy.eval()
    quantize_(policy, FromIntXQuantizationAwareTrainingConfig())
    quantize_(policy, Int8WeightOnlyConfig(group_size=GROUP_SIZE), filter_fn=expert_linear)
    packed = [name for name, module in policy.named_modules()
              if name in selected and type(module.weight).__name__ == "AffineQuantizedTensor"]
    if len(packed) != len(selected):
        raise ValueError(f"Packed {len(packed)}/{len(selected)} expert linears")
    checkpoint_path = args.output_dir / "qat_w8_expert.pt"
    torch.save(policy.state_dict(), checkpoint_path)
    report = {"method": "QAT_then_real_W8", "seed": args.seed, "steps": args.steps,
              "learning_rate": args.learning_rate, "weight_bits": 8, "group_size": GROUP_SIZE,
              "selected_linears": len(selected), "trainable_parameters": sum(p.numel() for p in trainable),
              "train_episode_count": len(train_episodes), "tasks": len(task_ids),
              "split_sha256": sha256(args.splits), "source_weight_sha256": sha256(args.model_dir / "model.safetensors"),
              "evaluation_partition_sha256": partition_hash,
              "first20_loss_mean": sum(losses[:20]) / min(20, len(losses)),
              "last20_loss_mean": sum(losses[-20:]) / min(20, len(losses)),
              "train_seconds": time.perf_counter() - began,
              "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
              "checkpoint_bytes": checkpoint_path.stat().st_size,
              "checkpoint_sha256": sha256(checkpoint_path)}
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
