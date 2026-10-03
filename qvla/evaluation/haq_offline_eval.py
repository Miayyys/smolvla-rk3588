#!/usr/bin/env python3
"""Build a portable FP-action cache or score a candidate against that cache.

cache/evaluate require the existing server LeRobot environment. replay needs
only NumPy and reuses archived real-model actions without running a model.
"""

from __future__ import annotations

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
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qvla.haq.offline_actions import (action_seed, file_sha256, frame_positions,
                                      load_cache, save_cache, score_actions,
                                      validate_partition)  # noqa: E402


def identity(args) -> dict:
    partition = json.loads(args.partition.read_text())
    splits = json.loads(args.splits.read_text())
    validate_partition(partition, splits)
    weight = args.model_dir / 'model.safetensors'
    if (file_sha256(args.splits) != partition['source_split_sha256']
            or file_sha256(weight) != partition['source_weight_sha256']):
        raise ValueError('Pinned split or model hash mismatch')
    model_files = sorted(path for path in args.model_dir.iterdir()
                         if path.is_file() and (path.suffix == '.json'
                         or path.name.startswith('policy_') and path.suffix == '.safetensors'))
    asset_files = sorted(path for path in args.vlm_assets_dir.iterdir()
                         if path.is_file() and path.suffix in ('.json', '.txt', '.model'))
    return {'checkpoint_sha256': partition['source_weight_sha256'],
            'source_model_file_bytes': weight.stat().st_size,
            'split_sha256': file_sha256(args.splits),
            'partition_sha256': file_sha256(args.partition),
            'model_and_processor_files': {path.name: file_sha256(path) for path in model_files},
            'vlm_asset_files': {path.name: file_sha256(path) for path in asset_files}}


def load_policy(args, pack: dict | None = None):
    # Lazy imports keep replay and cache/metric tests independent of LeRobot.
    import torch
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    config = SmolVLAConfig.from_pretrained(args.model_dir)
    config.device = args.device
    config.vlm_model_name = str(args.vlm_assets_dir.resolve())
    config.load_vlm_weights = False
    if pack is None:
        policy = SmolVLAPolicy.from_pretrained(args.model_dir, config=config, strict=True)
    else:
        if not args.device.startswith('cuda'):
            raise ValueError('Existing real CUDA INT8 candidate requires CUDA')
        from safetensors.torch import load_file
        from qvla.quantization.mixed_int8 import replace_from_manifest
        from qvla.quantization.real_int8_linear import replace_expert_linears
        policy = SmolVLAPolicy(config)
        names = (replace_from_manifest(policy, pack['manifest'])
                 if pack.get('format') == 'mixed-int8-v1' else replace_expert_linears(policy))
        if len(names) != pack['selected_linears']:
            raise ValueError('Packed module count differs from manifest')
        policy.load_state_dict(load_file(str(args.checkpoint), device='cpu'), strict=True)
    policy.to(args.device).eval()
    pre, post = make_pre_post_processors(
        config, str(args.model_dir), preprocessor_overrides={
            'tokenizer_processor': {'tokenizer_name': str(args.vlm_assets_dir.resolve())},
            'device_processor': {'device': args.device}})
    return policy, pre, post


def predict_cached(policy, pre, post, observations: dict, samples: list[dict], device: str):
    """Shared inference callable usable by a future full-action-space evaluator."""
    import torch
    device_obj = torch.device(device)
    cuda = device_obj.type == 'cuda'
    devices = [device_obj.index if device_obj.index is not None else torch.cuda.current_device()] if cuda else []
    actions, times = [], []
    for i, sample in enumerate(samples):
        obs = {
            'observation.images.image': torch.from_numpy(observations['image1'][i].copy()).float() / 255,
            'observation.images.image2': torch.from_numpy(observations['image2'][i].copy()).float() / 255,
            'observation.state': torch.from_numpy(observations['state'][i].copy()),
            'task': str(observations['task'][i]),
        }
        expected_seed = action_seed(sample['episode_index'], sample['task_index'], sample['frame_index'])
        if sample['action_seed'] != expected_seed:
            raise ValueError('Sample noise seed does not match its stable frame identity')
        if cuda:
            torch.cuda.synchronize(device_obj)
        started = time.perf_counter()
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.random.default_generator.manual_seed(expected_seed)
            if cuda:
                with torch.cuda.device(device_obj):
                    torch.cuda.manual_seed(expected_seed)
            policy.reset()
            result = post(policy.predict_action_chunk(pre(obs)))
        if cuda:
            torch.cuda.synchronize(device_obj)
        actions.append(result.detach().float().cpu().numpy()[0])
        times.append(time.perf_counter() - started)
        if (i + 1) % 10 == 0:
            print(f'action {i + 1}/{len(samples)}', flush=True)
    return np.stack(actions), {'per_observation_seconds': times,
                               'total_inference_seconds': float(sum(times)),
                               'p50_seconds': float(np.percentile(times, 50)),
                               'p95_seconds': float(np.percentile(times, 95)),
                               'scope': 'host_inference_including_processors_not_RK3588_cost'}


def build_cache(args):
    started = time.perf_counter()
    source_identity = identity(args)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    partition = json.loads(args.partition.read_text())
    episodes = sorted(partition['development_episode_ids_from_qat_train'])
    dataset = LeRobotDataset('lerobot/libero', root=args.dataset_root, episodes=episodes,
                             video_backend='pyav', return_uint8=True)
    arrays = {key: [] for key in ('image1', 'image2', 'state', 'task', 'recorded_action')}
    samples, tasks, offset = [], set(), 0
    for episode in episodes:
        length = int(dataset.meta.episodes['length'][episode])
        first = dataset.get_raw_item(offset)
        if int(first['episode_index']) != episode or int(first['frame_index']) != 0:
            raise ValueError('Dataset episode order or offsets differ from partition')
        task = int(first['task_index'])
        if task in tasks:
            raise ValueError('Expected exactly one development episode per task')
        tasks.add(task)
        for frame_index in frame_positions(length, task, args.phases):
            frame = dataset[offset + frame_index]
            if int(frame['episode_index']) != episode or int(frame['frame_index']) != frame_index:
                raise ValueError('Decoded frame identity mismatch')
            for key, source in [('image1', 'observation.images.image'),
                                ('image2', 'observation.images.image2')]:
                value = frame[source].detach().cpu().numpy()
                if value.dtype != np.uint8:
                    raise ValueError('Expected uint8 camera input from pinned dataset')
                arrays[key].append(value)
            arrays['state'].append(frame['observation.state'].detach().cpu().numpy())
            arrays['recorded_action'].append(frame['action'].detach().cpu().numpy())
            arrays['task'].append(frame['task'])
            samples.append({'episode_index': episode, 'task_index': task, 'frame_index': frame_index,
                            'action_seed': action_seed(episode, task, frame_index)})
        offset += length
    if offset != len(dataset) or len(tasks) != 40:
        raise ValueError('Expected complete 40-task development coverage')
    order = sorted(range(len(samples)), key=lambda i: (samples[i]['task_index'], samples[i]['frame_index']))
    samples = [samples[i] for i in order]
    observations = {key: np.stack([values[i] for i in order]) for key, values in arrays.items()}
    policy, pre, post = load_policy(args)
    fp, timing = predict_cached(policy, pre, post, observations, samples, args.device)
    import torch
    manifest = {'identity': source_identity, 'samples': samples, 'phases_per_task': args.phases,
                'selection': 'start/middle/90pct; quick panel uses task_index modulo 3',
                'torch_version': torch.__version__, 'device': args.device,
                'inference_steps': policy.config.num_steps,
                'fp_timing': timing, 'build_seconds_before_serialization': time.perf_counter() - started}
    save_cache(args.output, manifest, observations, fp)
    print(json.dumps({'status': 'fp_cache_saved', 'observations': len(samples),
                      'output': str(args.output), 'timing': timing}, ensure_ascii=False))


def evaluate(args):
    started = time.perf_counter()
    source_identity = identity(args)
    manifest, observations, fp = load_cache(args.cache, source_identity)
    pack, candidate = None, {'kind': 'fp_roundtrip_control'}
    if args.pack_report is not None:
        pack = json.loads(args.pack_report.read_text())
        if (pack['source_weight_sha256'] != source_identity['checkpoint_sha256']
                or pack['partition_sha256'] != source_identity['partition_sha256']):
            raise ValueError('Candidate source model or development partition mismatch')
        item = pack['outputs'][args.mode]
        args.checkpoint = args.checkpoint or Path(item['path'])
        if (file_sha256(args.checkpoint) != item['sha256']
                or args.checkpoint.stat().st_size != item['bytes']):
            raise ValueError('Candidate checkpoint hash/size mismatch')
        candidate = {'kind': 'real_cuda_int8', 'mode': args.mode,
                     'pack_report_sha256': file_sha256(args.pack_report),
                     'checkpoint_sha256': item['sha256'], 'actual_model_file_bytes': item['bytes'],
                     'compression_fraction': 1 - item['bytes'] / source_identity['source_model_file_bytes'],
                     'meets_40pct_file_compression': 5 * item['bytes'] <= 3 * source_identity['source_model_file_bytes']}
    elif args.checkpoint is not None:
        raise ValueError('--checkpoint requires --pack-report')
    policy, pre, post = load_policy(args, pack)
    if policy.config.num_steps != manifest['inference_steps']:
        raise ValueError('Candidate flow inference step count differs from FP cache')
    import torch
    if torch.__version__ != manifest['torch_version'] or args.device != manifest['device']:
        raise ValueError('FP cache must be regenerated when torch version or inference device changes')
    actions, timing = predict_cached(policy, pre, post, observations, manifest['samples'], args.device)
    metrics = score_actions(actions, fp, observations['recorded_action'],
                            np.array([sample['task_index'] for sample in manifest['samples']]))
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError('Use a new candidate output directory')
    np.savez_compressed(args.output / 'candidate_actions.npz', actions=actions)
    report = {'scope': 'offline_development_action_proxy', 'candidate': candidate,
              'cache_manifest_sha256': file_sha256(args.cache / 'manifest.json'),
              'candidate_actions_sha256': file_sha256(args.output / 'candidate_actions.npz'),
              'metrics': metrics, 'host_timing': timing,
              'total_seconds': time.perf_counter() - started,
              'RK3588_table_cost': 'not_connected', 'closed_loop_success_rate': 'not_measured'}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'output': str(args.output), 'task_macro': metrics['task_macro'],
                      'timing': timing}, ensure_ascii=False))


def replay(args):
    with np.load(args.actions, allow_pickle=False) as stored:
        metrics = score_actions(stored[args.candidate_key], stored['original'],
                                stored['recorded_first'], stored['task_index'])
    report = {'scope': 'archived_real_model_actions_reanalysis_no_new_inference',
              'source_actions_sha256': file_sha256(args.actions), 'candidate_key': args.candidate_key,
              'metrics': metrics, 'new_inference': False, 'closed_loop_success_rate': 'not_measured'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'output': str(args.output), 'task_macro': metrics['task_macro']}))


def main():
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('cache', 'evaluate'):
        p = sub.add_parser(command)
        p.add_argument('--model-dir', type=Path, required=True)
        p.add_argument('--vlm-assets-dir', type=Path, required=True)
        p.add_argument('--splits', type=Path, default=ROOT / 'data/libero_splits.json')
        p.add_argument('--partition', type=Path, default=ROOT / 'config/evaluation_partition_v2.json')
        p.add_argument('--device', default='cuda')
        p.add_argument('--output', type=Path, required=True)
        if command == 'cache':
            p.add_argument('--dataset-root', type=Path, required=True)
            p.add_argument('--phases', type=int, choices=(1, 3), default=1)
        else:
            p.add_argument('--cache', type=Path, required=True)
            p.add_argument('--pack-report', type=Path)
            p.add_argument('--mode', choices=('ptq', 'qat'), default='ptq')
            p.add_argument('--checkpoint', type=Path)
    p = sub.add_parser('replay')
    p.add_argument('--actions', type=Path, required=True)
    p.add_argument('--candidate-key', choices=('ptq', 'qat', 'original'), default='ptq')
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'cache': build_cache, 'evaluate': evaluate, 'replay': replay}[args.command](args)


if __name__ == '__main__':
    main()
