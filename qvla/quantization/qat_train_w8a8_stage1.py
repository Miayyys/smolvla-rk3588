#!/usr/bin/env python3
"""Stage-1 expert W8A8 QAT with FP master weights and frozen INT8 activation ranges.

The saved checkpoint contains trained floating weights, not a real quantized model.
RKNN conversion and an independent PTQ build are separate required steps.
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
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from safetensors.torch import save_file
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

from qvla.evaluation.probe_action_sensitivity import run_action, selected_frames, sha256


PREFIX = "model.vlm_with_expert.lm_expert"


class QATLinear(nn.Module):
    def __init__(self, linear: nn.Linear, low: float, high: float):
        super().__init__()
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError(f"Invalid activation range: {low}, {high}")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = nn.Parameter(linear.weight.detach().float().clone())
        self.bias = (nn.Parameter(linear.bias.detach().float().clone())
                     if linear.bias is not None else None)
        scale = max((high - low) / 255, 1e-12)
        zero = int(np.clip(np.rint(-128 - low / scale), -128, 127))
        self.register_buffer("activation_scale", torch.tensor(scale, dtype=torch.float32), persistent=False)
        self.register_buffer("activation_zero", torch.tensor(zero, dtype=torch.int32), persistent=False)
        self.fake_quant_enabled = True

    def forward(self, x):
        if self.fake_quant_enabled:
            xq = torch.fake_quantize_per_tensor_affine(
                x.float(), float(self.activation_scale), int(self.activation_zero), -128, 127)
            w = self.weight
            scales = w.detach().abs().amax(dim=1).clamp_min(1e-10) / 127
            zeros = torch.zeros(w.shape[0], device=w.device, dtype=torch.int32)
            wq = torch.fake_quantize_per_channel_affine(w, scales, zeros, 0, -127, 127)
        else:
            xq, wq = x, self.weight
        return F.linear(xq.to(x.dtype), wq.to(x.dtype),
                        None if self.bias is None else self.bias.to(x.dtype))


def load_policy(args):
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
    return policy, pre, post


def selected_linears(policy):
    return {name: mod for name, mod in policy.named_modules()
            if name.startswith(PREFIX) and isinstance(mod, nn.Linear)}


def calibrate(policy, pre, post, rows, names):
    stats = {name: {"min": float("inf"), "max": float("-inf"), "calls": 0} for name in names}
    hooks = []
    for name in names:
        def hook(_mod, inputs, name=name):
            values = inputs[0].detach()
            record = stats[name]
            record["min"] = min(record["min"], float(values.min()))
            record["max"] = max(record["max"], float(values.max()))
            record["calls"] += 1
        hooks.append(policy.get_submodule(name).register_forward_pre_hook(hook))
    try:
        for i, row in enumerate(rows):
            run_action(policy, pre, post, row)
            if (i + 1) % 10 == 0:
                print(f"calibration {i+1}/{len(rows)}", flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    if any(value["calls"] != len(rows) * 10 for value in stats.values()):
        raise ValueError("Unexpected expert invocation count during calibration")
    return stats


def actions(policy, pre, post, rows):
    policy.eval()
    return np.stack([run_action(policy, pre, post, row) for row in rows])


def compare(reference, candidate):
    difference = np.abs(reference - candidate)
    return {"mae": float(difference.mean()), "p95_element": float(np.percentile(difference, 95)),
            "per_task_mae": difference.mean(axis=(1, 2)).tolist()}


def training_data(args, split, development):
    episodes = [int(e) for e in split["qat_train"] if e not in development]
    dataset = LeRobotDataset("lerobot/libero", root=args.dataset_root, episodes=episodes,
                            delta_timestamps={"action": [i / 10 for i in range(50)]},
                            video_backend="pyav", return_uint8=True)
    by_task = defaultdict(list)
    offset = 0
    for episode in episodes:
        length = int(dataset.meta.episodes["length"][episode])
        raw = dataset.get_raw_item(offset)
        if int(raw["episode_index"]) != episode or int(raw["frame_index"]) != 0:
            raise ValueError("Unexpected training episode order")
        by_task[int(raw["task_index"])].append((offset, length, episode))
        offset += length
    if offset != len(dataset) or len(by_task) != 40:
        raise ValueError("Training dataset coverage mismatch")
    return dataset, by_task, episodes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "vlm-assets-dir", "dataset-root", "splits", "partition", "map", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--calibration-tasks", type=int, default=40)
    p.add_argument("--development-tasks", type=int, default=40)
    p.add_argument("--eval-every", type=int, default=100)
    args = p.parse_args()
    if (args.steps < 1 or args.eval_every < 1 or not 1 <= args.calibration_tasks <= 40
            or not 1 <= args.development_tasks <= 40):
        p.error("Invalid steps or task counts")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    began = time.perf_counter()
    partition = json.loads(args.partition.read_text())
    target = json.loads(args.map.read_text())
    split_doc = json.loads(args.splits.read_text())
    split = split_doc["splits"]
    if (partition["source_split_sha256"] != sha256(args.splits) or
            partition["source_weight_sha256"] != sha256(args.model_dir / "model.safetensors") or
            target["source_weight_sha256"] != partition["source_weight_sha256"] or
            target["evaluation_partition_sha256"] != sha256(args.partition)):
        raise ValueError("Frozen model/split/map hash mismatch")
    calibration = partition["calibration_episode_ids_from_ptq_calibration"]
    development = partition["development_episode_ids_from_qat_train"]
    if not (len(calibration) == len(development) == 40 and
            set(calibration) <= set(split["ptq_calibration"]) and
            set(development) <= set(split["qat_train"]) and
            not set(calibration) & set(development)):
        raise ValueError("Episode partition invalid")
    cal_rows = selected_frames(args.dataset_root, calibration, 1, 40)[:args.calibration_tasks]
    dev_rows = selected_frames(args.dataset_root, development, 1, 40)[:args.development_tasks]
    policy, pre, post = load_policy(args)
    names = selected_linears(policy)
    if len(names) != target["selected_linear_count"]:
        raise ValueError(f"Expected {target['selected_linear_count']} Linear, got {len(names)}")
    baseline = actions(policy, pre, post, dev_rows)
    stats = calibrate(policy, pre, post, cal_rows, names)
    for name, linear in names.items():
        parent, leaf = name.rsplit(".", 1)
        value = stats[name]
        setattr(policy.get_submodule(parent), leaf, QATLinear(linear, value["min"], value["max"]))
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for name in names:
        mod = policy.get_submodule(name)
        mod.weight.requires_grad_(True)
        trainable.append(mod.weight)
        if mod.bias is not None:
            mod.bias.requires_grad_(True)
            trainable.append(mod.bias)
    prepared = actions(policy, pre, post, dev_rows)
    dataset, by_task, episodes = training_data(args, split, set(development))
    rng = random.Random(args.seed)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0)
    policy.train()
    policy.model.vlm_with_expert.vlm.eval()
    torch.cuda.reset_peak_memory_stats()
    losses = []
    first_grad_norm = None
    development_history = []
    for step in range(args.steps):
        task = sorted(by_task)[step % len(by_task)]
        start, length, _ = rng.choice(by_task[task])
        frame = dataset[start + rng.randrange(length)]
        observation = {key: frame[key] for key in ("observation.state", "action", "task", "action_is_pad") if key in frame}
        for key in ("observation.images.image", "observation.images.image2"):
            observation[key] = frame[key].float() / 255
        batch = pre(observation)
        if batch["action"].ndim == 2:
            batch["action"] = batch["action"].unsqueeze(0)
        if "action_is_pad" in batch and batch["action_is_pad"].ndim == 1:
            batch["action_is_pad"] = batch["action_is_pad"].unsqueeze(0)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = policy.forward(batch)
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss at step {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if first_grad_norm is None:
            first_grad_norm = float(norm)
            if not np.isfinite(first_grad_norm) or first_grad_norm == 0:
                raise ValueError("QAT expert gradient invalid")
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 20 == 0 or step + 1 == args.steps:
            print(f"train {step+1}/{args.steps} loss20={np.mean(losses[-20:]):.6f} elapsed={time.perf_counter()-began:.1f}s", flush=True)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            candidate = actions(policy, pre, post, dev_rows)
            metric = compare(baseline, candidate)
            snapshot = args.output_dir / f"expert_master_step_{step+1}.safetensors"
            selected_weights = {}
            for name in names:
                mod = policy.get_submodule(name)
                selected_weights[f"{name}.weight"] = mod.weight.detach().cpu().contiguous()
                if mod.bias is not None:
                    selected_weights[f"{name}.bias"] = mod.bias.detach().cpu().contiguous()
            save_file(selected_weights, str(snapshot))
            development_history.append({"step": step + 1, "baseline_vs_fake": metric,
                                        "snapshot": str(snapshot), "snapshot_sha256": sha256(snapshot)})
            print(f"development step={step+1} action_mae={metric['mae']:.6f}", flush=True)
            policy.train()
            policy.model.vlm_with_expert.vlm.eval()
    trained_fake = actions(policy, pre, post, dev_rows)
    for name in names:
        policy.get_submodule(name).fake_quant_enabled = False
    trained_float = actions(policy, pre, post, dev_rows)
    weights = {key: value.detach().cpu().contiguous() for key, value in policy.state_dict().items()}
    checkpoint = args.output_dir / "qat_float_master.safetensors"
    save_file(weights, str(checkpoint))
    report = {
        "status": "trained_float_master_requires_real_rknn_w8a8_conversion",
        "source_weight_sha256": partition["source_weight_sha256"],
        "split_sha256": sha256(args.splits), "partition_sha256": sha256(args.partition),
        "map_sha256": sha256(args.map), "seed": args.seed, "steps": args.steps,
        "learning_rate": args.learning_rate, "selected_linear_count": len(names),
        "train_episode_count": len(episodes), "calibration_episode_ids": [r["episode_index"] for r in cal_rows],
        "development_episode_ids": [r["episode_index"] for r in dev_rows],
        "activation_scheme": "static_per_tensor_affine_int8_minmax", "weight_scheme": "dynamic_symmetric_per_output_channel_int8",
        "weight_qrange": [-127, 127], "activation_qrange": [-128, 127],
        "first_gradient_norm": first_grad_norm, "losses": losses,
        "baseline_vs_prepared": compare(baseline, prepared),
        "baseline_vs_trained_fake": compare(baseline, trained_fake),
        "baseline_vs_trained_float": compare(baseline, trained_float),
        "development_history": development_history,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": time.perf_counter() - began,
        "checkpoint_bytes": checkpoint.stat().st_size, "checkpoint_sha256": sha256(checkpoint),
        "torch_version": torch.__version__,
    }
    (args.output_dir / "activation_ranges.json").write_text(json.dumps(stats, indent=2) + "\n")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "losses"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
