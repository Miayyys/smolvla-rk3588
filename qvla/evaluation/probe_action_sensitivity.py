#!/usr/bin/env python3
"""Rank SmolVLA modules by paired full-action response to static INT8 output rounding.

This is a fake-quant diagnostic, not a packed checkpoint or RKNN execution.
Calibration episodes set ranges; disjoint QAT-train development episodes score
action changes. It cannot establish rollout quality or board performance.
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
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_frames(root: Path, episodes: list[int], frames_per_task: int,
                    max_tasks: int) -> list[dict]:
    dataset = LeRobotDataset("lerobot/libero", root=root, episodes=episodes,
                             video_backend="pyav", return_uint8=True)
    selected = []
    offset = 0
    for episode in episodes:
        length = int(dataset.meta.episodes["length"][episode])
        raw = dataset.get_raw_item(offset)
        if int(raw["episode_index"]) != episode or int(raw["frame_index"]) != 0:
            raise ValueError(f"Unexpected episode at dataset offset {offset}")
        task = int(raw["task_index"])
        positions = np.linspace(0, length - 1, frames_per_task, dtype=int)
        if len(set(positions)) != frames_per_task:
            raise ValueError(f"Episode {episode} is shorter than frames_per_task")
        for rank, frame_index in enumerate(positions):
            frame = dataset[offset + int(frame_index)]
            selected.append({"task_index": task, "episode_index": episode,
                             "frame_index": int(frame_index), "frame_rank": rank,
                             "observation": {
                                 "observation.state": frame["observation.state"],
                                 "observation.images.image":
                                     frame["observation.images.image"].float() / 255.0,
                                 "observation.images.image2":
                                     frame["observation.images.image2"].float() / 255.0,
                                 "task": frame["task"]},
                             "recorded_action": frame["action"].detach().cpu().numpy()})
        offset += length
    if offset != len(dataset):
        raise ValueError("Dataset episode lengths do not sum to frame count")
    if len({row["task_index"] for row in selected}) != max_tasks:
        raise ValueError("Selected episodes must cover one episode per requested task")
    return sorted(selected, key=lambda x: (x["task_index"], x["frame_rank"]))


def model_groups(policy: SmolVLAPolicy) -> dict[str, list[str]]:
    prefix = "model.vlm_with_expert"
    model = policy.model.vlm_with_expert
    groups = {
        "vision_0_5": [f"{prefix}.vlm.model.vision_model.encoder.layers.{i}.mlp"
                       for i in range(6)],
        "vision_6_11": [f"{prefix}.vlm.model.vision_model.encoder.layers.{i}.mlp"
                        for i in range(6, 12)],
        "language_0_7": [f"{prefix}.vlm.model.text_model.layers.{i}.mlp"
                         for i in range(8)],
        "language_8_15": [f"{prefix}.vlm.model.text_model.layers.{i}.mlp"
                          for i in range(8, 16)],
        "expert_0_7": [f"{prefix}.lm_expert.layers.{i}.mlp" for i in range(8)],
        "expert_8_15": [f"{prefix}.lm_expert.layers.{i}.mlp" for i in range(8, 16)],
        "expert_0_only": [f"{prefix}.lm_expert.layers.0.mlp"],
        "vision_patch_projection": [f"{prefix}.vlm.model.vision_model.embeddings.patch_embedding"],
        "action_interface_projections": [f"model.{name}" for name in
                                         ("state_proj", "action_in_proj", "action_out_proj")],
        "connector_projection": [f"{prefix}.vlm.model.connector.modality_projection.proj"],
        "language_token_embedding": [f"{prefix}.vlm.model.text_model.embed_tokens"],
        "action_time_projections": ["model.action_time_mlp_in", "model.action_time_mlp_out"],
    }
    for stage, root, spans, projections in (
        ("vision", f"{prefix}.vlm.model.vision_model.encoder.layers",
         ((0, 5), (6, 11)), ("q_proj", "k_proj", "v_proj", "out_proj")),
        ("language", f"{prefix}.vlm.model.text_model.layers",
         ((0, 7), (8, 15)), ("q_proj", "k_proj", "v_proj", "o_proj")),
        ("expert", f"{prefix}.lm_expert.layers",
         ((0, 7), (8, 15)), ("q_proj", "k_proj", "v_proj", "o_proj")),
    ):
        for start, end in spans:
            groups[f"{stage}_attention_{start}_{end}"] = [
                f"{root}.{layer}.self_attn.{projection}"
                for layer in range(start, end + 1) for projection in projections]
    for i in range(6, 12):
        groups[f"vision_{i}_only"] = [
            f"{prefix}.vlm.model.vision_model.encoder.layers.{i}.mlp"]
    for i in range(8):
        groups[f"language_{i}_only"] = [
            f"{prefix}.vlm.model.text_model.layers.{i}.mlp"]
    groups["vision_6_10"] = [
        f"{prefix}.vlm.model.vision_model.encoder.layers.{i}.mlp"
        for i in range(6, 11)]
    groups["language_0_2_4_7"] = [
        f"{prefix}.vlm.model.text_model.layers.{i}.mlp"
        for i in (0, 1, 2, 4, 5, 6, 7)]
    groups["vision_6_10_language_without_3"] = (
        groups["vision_6_10"] + groups["language_0_2_4_7"])
    groups["mixed_v0_active_outputs"] = (
        groups["vision_0_5"] + groups["vision_6_10"] +
        groups["language_0_2_4_7"] + groups["language_8_15"] +
        groups["expert_0_7"] + groups["expert_8_15"] +
        groups["vision_attention_0_5"] + groups["vision_attention_6_11"] +
        groups["language_attention_0_7"] + groups["language_attention_8_15"] +
        groups["expert_attention_0_7"] + groups["expert_attention_8_15"])
    groups["mixed_stage1_expert_outputs"] = (
        groups["expert_0_7"] + groups["expert_8_15"] +
        groups["expert_attention_0_7"] + groups["expert_attention_8_15"])
    groups["mixed_stage2_late_vlm_outputs"] = (
        groups["mixed_stage1_expert_outputs"] + groups["vision_6_10"] +
        groups["vision_attention_6_11"] + groups["language_8_15"] +
        groups["language_attention_8_15"])
    groups["mixed_stage3_early_vlm_outputs"] = (
        groups["mixed_stage2_late_vlm_outputs"] + groups["vision_0_5"] +
        groups["vision_attention_0_5"] + groups["language_0_2_4_7"] +
        groups["language_attention_0_7"])
    if (len(model.vlm.model.vision_model.encoder.layers) != 12 or
            len(model.vlm.model.text_model.layers) != 16 or
            len(model.lm_expert.layers) != 16):
        raise ValueError("Unexpected SmolVLA layer counts; review group definitions")
    for names in groups.values():
        for name in names:
            policy.get_submodule(name)
    return groups


def affine_int8(values: np.ndarray, low: float, high: float) -> tuple[float, int, np.ndarray]:
    scale = (high - low) / 255.0
    zero_point = int(np.clip(np.rint(-128 - low / scale), -128, 127))
    integer = np.clip(np.rint(values / scale + zero_point), -128, 127)
    reconstructed = (integer - zero_point) * scale
    return scale, zero_point, reconstructed


def run_action(policy: SmolVLAPolicy, preprocess, postprocess, row: dict) -> np.ndarray:
    seed = 1009 * row["episode_index"] + 9176 * row["frame_rank"] + 17 * row["task_index"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    policy.reset()
    with torch.inference_mode():
        action = postprocess(policy.predict_action_chunk(preprocess(row["observation"])))
    result = action.detach().float().cpu().numpy()
    if result.ndim != 3 or result.shape[0] != 1 or result.shape[2] != 7 or not np.isfinite(result).all():
        raise ValueError(f"Invalid action chunk shape or values: {result.shape}")
    return result[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--vlm-assets-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames-per-task", type=int, choices=(1, 2), default=2)
    parser.add_argument("--max-tasks", type=int, default=40)
    parser.add_argument("--groups", nargs="*", help="Subset of group names; default all")
    parser.add_argument("--calibration-method", choices=("minmax", "mse", "mse_exact"),
                        default="minmax")
    args = parser.parse_args()
    if not 1 <= args.max_tasks <= 40:
        parser.error("--max-tasks must be 1..40")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partition = json.loads(args.partition.read_text())
    if partition["source_split_sha256"] != sha256(args.splits):
        raise ValueError("Split hash differs from frozen partition")
    weights = args.model_dir / "model.safetensors"
    if partition["source_weight_sha256"] != sha256(weights):
        raise ValueError("Weight hash differs from frozen partition")
    splits = json.loads(args.splits.read_text())["splits"]
    calibration_episodes = partition["calibration_episode_ids_from_ptq_calibration"]
    development_episodes = partition["development_episode_ids_from_qat_train"]
    if (len(calibration_episodes) != 40 or len(development_episodes) != 40 or
            not set(calibration_episodes) <= set(splits["ptq_calibration"]) or
            not set(development_episodes) <= set(splits["qat_train"]) or
            set(calibration_episodes) & set(development_episodes)):
        raise ValueError("Invalid development/calibration episode partition")
    # Episode IDs in the frozen manifest cover one episode per task. Select the
    # first N tasks by task index after loading all 40 episode headers.
    calibration_rows = selected_frames(args.dataset_root, calibration_episodes,
                                       args.frames_per_task, 40)
    development_rows = selected_frames(args.dataset_root, development_episodes,
                                       args.frames_per_task, 40)
    calibration_rows = [row for row in calibration_rows if row["task_index"] < args.max_tasks]
    development_rows = [row for row in development_rows if row["task_index"] < args.max_tasks]
    if len(calibration_rows) != args.max_tasks * args.frames_per_task or len(development_rows) != len(calibration_rows):
        raise ValueError("Missing samples in requested task subset")

    config = SmolVLAConfig.from_pretrained(args.model_dir)
    config.device = "cuda"
    config.vlm_model_name = str(args.vlm_assets_dir.resolve())
    config.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(args.model_dir, config=config, strict=True).eval()
    preprocess, postprocess = make_pre_post_processors(
        config, str(args.model_dir),
        preprocessor_overrides={"tokenizer_processor": {
            "tokenizer_name": str(args.vlm_assets_dir.resolve())}},
    )
    groups = model_groups(policy)
    chosen = args.groups or ["vision_0_5", "vision_6_11", "language_0_7",
                             "language_8_15", "expert_0_7", "expert_8_15",
                             "vision_6_10", "language_0_2_4_7",
                             "vision_6_10_language_without_3"]
    if any(name not in groups for name in chosen):
        parser.error(f"Unknown group; choose from {list(groups)}")
    selected_modules = sorted({name for group in chosen for name in groups[group]})
    ranges = {name: {"min": float("inf"), "max": -float("inf"), "calls": 0}
              for name in selected_modules}
    calibration_samples = {name: [] for name in selected_modules}

    def collect(name):
        def hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"Expected tensor output for {name}")
            x = output.detach()
            ranges[name]["min"] = min(ranges[name]["min"], float(x.amin().item()))
            ranges[name]["max"] = max(ranges[name]["max"], float(x.amax().item()))
            ranges[name]["calls"] += 1
            if args.calibration_method in ("mse", "mse_exact"):
                flat = x.float().reshape(-1)
                stride = max(1, flat.numel() // 2048)
                calibration_samples[name].append(flat[::stride][:2048].cpu().numpy())
        return hook

    handles = [policy.get_submodule(name).register_forward_hook(collect(name))
               for name in selected_modules]
    try:
        for i, row in enumerate(calibration_rows, 1):
            run_action(policy, preprocess, postprocess, row)
            if i % 10 == 0:
                print(f"calibrated {i}/{len(calibration_rows)} frames", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    quantizers = {}
    for name, observed in ranges.items():
        low, high = observed["min"], observed["max"]
        if not np.isfinite([low, high]).all() or high <= low or observed["calls"] == 0:
            raise ValueError(f"Invalid calibration range for {name}: {observed}")
        chosen_percentile = 1.0
        candidates = []
        if args.calibration_method in ("mse", "mse_exact"):
            values = np.concatenate(calibration_samples[name]).astype(np.float64)
            for percentile in (0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999, 1.0):
                if percentile == 1.0:
                    candidate_low, candidate_high = low, high
                else:
                    tail = (1.0 - percentile) / 2.0
                    candidate_low, candidate_high = np.quantile(values, [tail, 1.0 - tail])
                scale_candidate, zp_candidate, reconstructed = affine_int8(
                    values, float(candidate_low), float(candidate_high))
                candidates.append({"retained_fraction": percentile,
                                   "min": float(candidate_low), "max": float(candidate_high),
                                   "scale": scale_candidate, "zero_point": zp_candidate,
                                   "sample_mse": float(np.mean((reconstructed - values) ** 2)),
                                   "sample_saturation_fraction": float(np.mean(
                                       (values < candidate_low) | (values > candidate_high)))})
            best = min(candidates, key=lambda row: row["sample_mse"])
            low, high = best["min"], best["max"]
            chosen_percentile = best["retained_fraction"]
        scale, zero_point, _ = affine_int8(np.asarray([low, high]), low, high)
        quantizers[name] = {**observed, "scale": scale, "zero_point": zero_point,
                            "dtype": "int8", "granularity": "per_tensor_output",
                            "calibration_rule": args.calibration_method,
                            "quant_min": low, "quant_max": high,
                            "selected_retained_fraction": chosen_percentile,
                            "calibration_candidate_scan": candidates}

    if args.calibration_method == "mse_exact":
        sums = {name: np.zeros(len(quantizers[name]["calibration_candidate_scan"]),
                               dtype=np.float64) for name in selected_modules}
        counts = {name: 0 for name in selected_modules}
        saturations = {name: np.zeros_like(sums[name]) for name in selected_modules}

        def validate(name):
            candidates = quantizers[name]["calibration_candidate_scan"]

            def hook(_module, _inputs, output):
                x = output.detach().float()
                counts[name] += x.numel()
                for i, candidate in enumerate(candidates):
                    scale = candidate["scale"]
                    zp = candidate["zero_point"]
                    integer = torch.clamp(torch.round(x / scale + zp), -128, 127)
                    reconstructed = (integer - zp) * scale
                    sums[name][i] += float(torch.sum((reconstructed - x) ** 2).item())
                    saturations[name][i] += float(torch.sum(
                        (x < candidate["min"]) | (x > candidate["max"])).item())
            return hook

        handles = [policy.get_submodule(name).register_forward_hook(validate(name))
                   for name in selected_modules]
        try:
            for i, row in enumerate(calibration_rows, 1):
                run_action(policy, preprocess, postprocess, row)
                if i % 10 == 0:
                    print(f"exact MSE checked {i}/{len(calibration_rows)} frames", flush=True)
        finally:
            for handle in handles:
                handle.remove()
        for name in selected_modules:
            if counts[name] == 0:
                raise ValueError(f"No exact MSE samples for {name}")
            candidates = quantizers[name]["calibration_candidate_scan"]
            full_scan = [{"retained_fraction": row["retained_fraction"],
                          "min": row["min"], "max": row["max"],
                          "scale": row["scale"], "zero_point": row["zero_point"],
                          "full_calibration_mse": float(sums[name][i] / counts[name]),
                          "full_calibration_saturation_fraction":
                              float(saturations[name][i] / counts[name])}
                         for i, row in enumerate(candidates)]
            best = min(full_scan, key=lambda row: row["full_calibration_mse"])
            quantizers[name].update(scale=best["scale"], zero_point=best["zero_point"],
                                    quant_min=best["min"], quant_max=best["max"],
                                    selected_retained_fraction=best["retained_fraction"],
                                    full_calibration_candidate_scan=full_scan,
                                    full_calibration_elements=counts[name])

    fp_chunks = []
    for i, row in enumerate(development_rows, 1):
        fp_chunks.append(run_action(policy, preprocess, postprocess, row))
        if i % 10 == 0:
            print(f"FP development {i}/{len(development_rows)} frames", flush=True)
    repeated = run_action(policy, preprocess, postprocess, development_rows[0])
    repeat_max_abs = float(np.max(np.abs(repeated - fp_chunks[0])))
    if repeat_max_abs > 1e-5:
        raise ValueError(f"FP inference is not reproducible under fixed seed: {repeat_max_abs}")
    np.savez_compressed(args.output_dir / "fp_action_chunks.npz",
                        chunks=np.stack(fp_chunks).astype(np.float32))

    group_reports = {}
    for group in chosen:
        names = groups[group]
        changed = {name: 0 for name in names}

        def quantize(name):
            scale = quantizers[name]["scale"]
            zero_point = quantizers[name]["zero_point"]

            def hook(_module, _inputs, output):
                if not isinstance(output, torch.Tensor):
                    raise TypeError(f"Expected tensor output for {name}")
                x = output.float()
                integer = torch.clamp(torch.round(x / scale + zero_point), -128, 127)
                reconstructed = ((integer - zero_point) * scale).to(output.dtype)
                changed[name] += 1
                return reconstructed
            return hook

        handles = [policy.get_submodule(name).register_forward_hook(quantize(name))
                   for name in names]
        records = []
        chunks = []
        try:
            for row, reference in zip(development_rows, fp_chunks):
                action = run_action(policy, preprocess, postprocess, row)
                if action.shape != reference.shape:
                    raise ValueError("Quantized and FP action chunk shapes differ")
                delta = action - reference
                target = row["recorded_action"]
                records.append({"task_index": row["task_index"],
                                "episode_index": row["episode_index"],
                                "frame_index": row["frame_index"],
                                "frame_rank": row["frame_rank"],
                                "full_chunk_mae_vs_fp": float(np.mean(np.abs(delta))),
                                "full_chunk_rmse_vs_fp": float(np.sqrt(np.mean(delta ** 2))),
                                "first_action_mae_vs_fp": float(np.mean(np.abs(delta[0]))),
                                "last_action_mae_vs_fp": float(np.mean(np.abs(delta[-1]))),
                                "max_abs_vs_fp": float(np.max(np.abs(delta))),
                                "fp_first_action_mae_vs_recorded":
                                    float(np.mean(np.abs(reference[0] - target))),
                                "quant_first_action_mae_vs_recorded":
                                    float(np.mean(np.abs(action[0] - target)))})
                chunks.append(action)
        finally:
            for handle in handles:
                handle.remove()
        if any(count == 0 for count in changed.values()):
            raise ValueError(f"A requested module was not used: {changed}")
        np.savez_compressed(args.output_dir / f"{group}_action_chunks.npz",
                            chunks=np.stack(chunks).astype(np.float32))
        group_reports[group] = {
            "modules": names,
            "module_parameter_count": sum(sum(p.numel() for p in policy.get_submodule(name).parameters())
                                          for name in names),
            "calls_per_module": changed,
            "mean_full_chunk_mae_vs_fp": float(np.mean([r["full_chunk_mae_vs_fp"] for r in records])),
            "mean_first_action_mae_vs_fp": float(np.mean([r["first_action_mae_vs_fp"] for r in records])),
            "max_sample_full_chunk_mae_vs_fp": max(r["full_chunk_mae_vs_fp"] for r in records),
            "mean_first_action_mae_vs_recorded_delta": float(np.mean([
                r["quant_first_action_mae_vs_recorded"] - r["fp_first_action_mae_vs_recorded"]
                for r in records])),
            "records": records}
        print(f"{group}: mean full-chunk MAE vs FP={group_reports[group]['mean_full_chunk_mae_vs_fp']:.6f}",
              flush=True)
        report_partial = args.output_dir / "partial_report.json"
        report_partial.write_text(json.dumps({"status": "running", "groups": group_reports},
                                             indent=2) + "\n")

    report = {"scope": "full SmolVLA action-chunk sensitivity to module-output fake INT8",
              "not_real_quantization": True, "not_rknn_execution": True,
              "not_closed_loop": True,
              "calibration": f"static per-tensor affine INT8 {args.calibration_method}",
              "mse_sampling": "deterministic stride, at most 2048 output values per module call" if args.calibration_method in ("mse", "mse_exact") else None,
              "quantization_formula": "scale=(max-min)/255; zp=clip(round(-128-min/scale),-128,127); q=clip(round(x/scale+zp),-128,127); xhat=(q-zp)*scale",
              "source_weight_sha256": sha256(weights), "split_sha256": sha256(args.splits),
              "partition_sha256": sha256(args.partition),
              "calibration_episodes": [r["episode_index"] for r in calibration_rows[::args.frames_per_task]],
              "development_episodes": [r["episode_index"] for r in development_rows[::args.frames_per_task]],
              "frames_per_task": args.frames_per_task, "tasks": args.max_tasks,
              "development_samples": len(development_rows),
              "fp_repeat_max_abs": repeat_max_abs,
              "fp_mean_first_action_mae_vs_recorded": float(np.mean([
                  np.mean(np.abs(action[0] - row["recorded_action"]))
                  for action, row in zip(fp_chunks, development_rows)])),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "quantizers": quantizers, "groups": group_reports}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "complete", "tasks": args.max_tasks,
                      "samples": len(development_rows),
                      "group_mae": {name: row["mean_full_chunk_mae_vs_fp"]
                                    for name, row in group_reports.items()}}), flush=True)


if __name__ == "__main__":
    main()
