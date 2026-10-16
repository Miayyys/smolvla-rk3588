"""Diagnostic closed-loop SO101 rollouts with the unmodified floating base.

Task statistics are adapted from training-only SO100 demonstrations. They are
not claimed to recover the pretraining embodiment's normalization/calibration.
Simulation pauses during inference: this is not a real-time board benchmark.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/home/loser/Study/so101-nexus'))
    parser.add_argument('--dataset', type=Path, default=Path('artifacts/datasets/svla_so100_pickplace'))
    parser.add_argument('--partition', type=Path, default=Path('runs/so100_pickplace_readiness_v1/report.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, default=Path('artifacts/smolvla_base'))
    parser.add_argument('--normalization-file', type=Path)
    parser.add_argument('--max-steps', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path.insert(0, str(args.source.resolve() / 'src'))
    revision = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != '7bdb671aeffa74061993d628f646ae5ba777ae5a':
        raise ValueError('Unexpected simulator revision')
    import gymnasium as gym
    import pyarrow.parquet as pq
    import torch
    from PIL import Image
    import so101_nexus.mujoco  # noqa: F401
    from so101_nexus import (
        JointPositions, OverheadCamera, PhysicsConfig,
        dataset_row_to_sim_qpos, sim_qpos_to_dataset_row,
    )
    from lerobot.policies import make_pre_post_processors
    from qvla.evaluation.haq_offline_eval import load_policy

    torch.set_num_threads(4)
    torch.manual_seed(42)
    args.output.mkdir(parents=True, exist_ok=True)
    partition = json.loads(args.partition.read_text())
    table = pq.read_table(args.dataset / 'data/chunk-000/file-000.parquet')
    selected = np.isin(np.asarray(table['episode_index']), partition['splits']['train'])
    stats = {}
    for key in ['observation.state', 'action']:
        values = np.asarray(table[key].to_pylist(), dtype=np.float32)[selected]
        stats[key] = {'mean': values.mean(0).tolist(), 'std': np.maximum(values.std(0), 1e-6).tolist()}
    if args.normalization_file:
        supplied = json.loads(args.normalization_file.read_text())
        for key in stats:
            for statistic in ['mean', 'std']:
                if not np.allclose(supplied[key][statistic], stats[key][statistic], atol=1e-6):
                    raise ValueError('Supplied statistics differ from the fixed train-only partition')
        stats = supplied
    (args.output / 'normalization.json').write_text(json.dumps({
        'source': 'SO100 pickplace train split, not original base pretraining statistics',
        'train_episodes': partition['splits']['train'], 'training_frames': int(selected.sum()),
        'stats': stats,
    }, indent=2) + '\n')
    model_dir = args.model_dir.resolve()
    vlm_dir = Path('artifacts/transfer/smolvlm2_assets').resolve()
    source_sha = file_sha(model_dir / 'model.safetensors')
    if model_dir == Path('artifacts/smolvla_base').resolve() and source_sha != '7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb':
        raise ValueError('Unexpected model checkpoint')
    policy, _, _ = load_policy(SimpleNamespace(model_dir=model_dir, vlm_assets_dir=vlm_dir, device='cuda'))
    policy.float().eval()
    pre, post = make_pre_post_processors(
        policy.config, str(model_dir),
        preprocessor_overrides={
            'tokenizer_processor': {'tokenizer_name': str(vlm_dir)},
            'device_processor': {'device': 'cuda'},
            'normalizer_processor': {'stats': stats},
        },
        postprocessor_overrides={'unnormalizer_processor': {'stats': stats}},
    )
    report = {
        'scope': 'diagnostic real-to-sim under assumed SO100-to-SO101 adapter; not formal benchmark',
        'model_dir': str(model_dir),
        'model_sha256': source_sha, 'nexus_revision': revision,
        'normalization_sha256': file_sha(args.output / 'normalization.json'),
        'dataset_table_sha256': file_sha(args.dataset / 'data/chunk-000/file-000.parquet'),
        'partition_sha256': file_sha(args.partition),
        'camera': 'single overhead mapped to camera1',
        'model_resize': list(policy.config.resize_imgs_with_padding),
        'num_steps': policy.config.num_steps, 'chunk_size': 50, 'execute_actions_per_chunk': 50,
        'control_hz_simulated': 30, 'real_time_scheduled': False,
        'mapping_assumptions': 'first five joints degrees; gripper 0..100; upstream zero/sign conventions assumed, not physically calibrated',
        'episodes': [],
    }
    tasks = ['MuJoCoTouch-v1', 'MuJoCoPickLift-v1', 'MuJoCoPickAndPlace-v2']
    for env_id in tasks:
        import importlib
        module, name = gym.spec(env_id).entry_point.split(':')
        cls = getattr(importlib.import_module(module), name)
        config = cls.default_config_cls(
            obs_mode='visual', observations=[JointPositions(), OverheadCamera(width=640, height=480)],
            physics=PhysicsConfig(timestep_s=1 / 300, control_period_s=1 / 30),
        )
        env = gym.make(env_id, config=config, control_mode='pd_joint_pos', max_episode_steps=args.max_steps)
        try:
            obs, info = env.reset(seed=args.seed)
            policy.reset()
            task = env.unwrapped.task_description
            limits = (float(env.action_space.low[-1]), float(env.action_space.high[-1]))
            frames, states, actions, requested_actions, infos, times = [], [], [], [], [], []
            step = chunks = clipped_values = total_values = 0
            success = False
            started = time.perf_counter()
            while step < args.max_steps:
                state = sim_qpos_to_dataset_row(obs['state'], gripper_limits_rad=limits).astype('float32')
                inputs = {
                    'observation.images.camera1': torch.from_numpy(obs['overhead_camera'].copy()).permute(2, 0, 1).float() / 255,
                    'observation.state': torch.from_numpy(state), 'task': task,
                }
                torch.manual_seed(42000 + args.seed * 1000 + chunks)
                torch.cuda.synchronize()
                begin = time.perf_counter()
                with torch.inference_mode():
                    predicted = post(policy.predict_action_chunk(pre(inputs))).detach().float().cpu().numpy()[0]
                torch.cuda.synchronize()
                times.append(time.perf_counter() - begin)
                if predicted.shape != (50, 6) or not np.isfinite(predicted).all():
                    raise RuntimeError('Invalid policy output')
                targets = dataset_row_to_sim_qpos(predicted, gripper_limits_rad=limits)
                chunks += 1
                for target in targets:
                    if step % 10 == 0:
                        frames.append(Image.fromarray(obs['overhead_camera']).resize((320, 240)))
                    states.append(obs['state'].copy())
                    action = np.clip(target, env.action_space.low, env.action_space.high).astype('float32')
                    clipped_values += int(np.count_nonzero(action != target))
                    total_values += action.size
                    actions.append(action)
                    requested_actions.append(target.copy())
                    obs, reward, terminated, truncated, info = env.step(action)
                    if not np.isfinite(obs['state']).all():
                        raise RuntimeError('Invalid simulated state')
                    infos.append({k: float(v) for k, v in info.items() if isinstance(v, (float, int, np.number))})
                    step += 1
                    success = success or bool(info.get('success', False))
                    if terminated or truncated or step >= args.max_steps:
                        break
                if terminated or truncated:
                    break
            np.savez_compressed(args.output / f'{env_id}.npz', states=np.asarray(states), actions=np.asarray(actions),
                                requested_actions=np.asarray(requested_actions))
            frames[0].save(args.output / f'{env_id}.gif', save_all=True, append_images=frames[1:], duration=333, loop=0)
            result = {
                'env_id': env_id, 'seed': args.seed, 'task': task, 'success': success,
                'steps': step, 'chunks': chunks, 'simulated_seconds': step / 30,
                'wall_seconds': time.perf_counter() - started,
                'inference_seconds': times, 'clipped_value_fraction': clipped_values / total_values,
                'max_lift_height': max((v.get('lift_height', 0) for v in infos), default=0),
                'ever_grasped': any(v.get('is_grasped', 0) for v in infos), 'last_info': infos[-1],
            }
            report['episodes'].append(result)
            (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(result), flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
