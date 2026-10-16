"""Dual-camera SmolVLA fine-tuning and closed-loop evaluation on scripted SO101 data.

Training uses the trainer in ``finetune_so100_real`` (same flow-matching loss,
padding mask, optimizer and checkpoint rules); this module only supplies the
data reader, the prepare step and the simulator evaluation. Labels are the clean
expert actions; executed (noisy) commands are never used as targets.
"""

import argparse
import io
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from qvla.evaluation.finetune_so100_real import emit, sha


def _subset_rows(data, seeds):
    return np.flatnonzero(np.isin(data['episodes'], seeds)) if seeds is not None else np.arange(len(data['episodes']))


def _check_rows(data, episodes, per_episode):
    rows = []
    for ep in episodes:
        ep_rows = np.flatnonzero(data['episodes'] == ep)
        rows.extend(np.linspace(ep_rows[0], max(ep_rows[0], ep_rows[-1] - 49), per_episode, dtype=int).tolist())
    return rows


def prepare_scripted(cfg, out):
    source = Path(cfg['dataset_root'])
    manifest = json.loads((source / 'manifest.json').read_text())
    if sha(cfg['partition']) != sha(source / 'manifest.json'):
        raise ValueError('Partition must be the generated manifest')
    train = np.load(source / 'train.npz'); validation = np.load(source / 'validation.npz')
    train_rows = _subset_rows(train, cfg.get('train_seeds'))
    train_eps = np.unique(train['episodes'][train_rows])
    if cfg.get('offline_check') == 'train':
        # Overfit diagnostics check the fit on the training episodes themselves.
        check = _check_rows(train, train_eps, cfg['validation_frames_per_episode'])
        check_split = 'train'
    else:
        eps = np.unique(validation['episodes'])
        check = (np.asarray(_check_rows(validation, eps, cfg['validation_frames_per_episode'])) + len(train['episodes'])).tolist()
        check_split = 'validation'
    if cfg.get('train_seeds') is None:
        stats = json.loads((source / 'stats.json').read_text())
    else:
        stats = {key: {'mean': train[src][train_rows].mean(0).tolist(), 'std': np.maximum(train[src][train_rows].std(0), 1e-6).tolist()}
                 for key, src in (('observation.state', 'states'), ('action', 'actions'))}
    (out / 'stats.json').write_text(json.dumps(stats, indent=2) + '\n')
    np.savez(out / 'data.npz', train_rows=train_rows, validation_rows=np.asarray(check))
    prepared = {'config': cfg, 'partition_sha256': sha(cfg['partition']), 'stats_sha256': sha(out / 'stats.json'),
                'source_splits': {k: {kk: vv for kk, vv in v.items()} for k, v in manifest['splits'].items()},
                'train_episodes': train_eps.tolist(), 'train_frames': int(len(train_rows)),
                'offline_check_split': check_split, 'offline_check_rows': check,
                'scope': 'scripted-expert simulator demonstrations; offline check is not task success'}
    (out / 'prepared.json').write_text(json.dumps(prepared, indent=2) + '\n')
    emit(out, {'stage': 'prepared', 'train_frames': len(train_rows), 'train_episodes': len(train_eps),
               'offline_check_split': check_split, 'offline_check_rows': len(check)})


class ScriptedFrames:
    """Same interface as ``finetune_so100_real.Frames``: ``.data`` rows and ``.batch(rows)``."""

    def __init__(self, cfg, out):
        source = Path(cfg['dataset_root'])
        self.parts = [np.load(source / f'{split}.npz') for split in ('train', 'validation')]
        self.base = len(self.parts[0]['episodes'])
        self.streams = [{cam: (source / f'{split}_{cam}.jpeg.bin').open('rb') for cam in cfg['camera_map'].values()}
                        for split in ('train', 'validation')]
        index = np.load(out / 'data.npz')
        self.data = {'train_rows': index['train_rows'], 'validation_rows': index['validation_rows']}
        self.camera_map = cfg['camera_map']; self.task = cfg['task']

    def _image(self, part, cam, row):
        import torch
        from PIL import Image
        stream = self.streams[part][cam]; data = self.parts[part]
        stream.seek(int(data[f'{cam}_offsets'][row]))
        image = Image.open(io.BytesIO(stream.read(int(data[f'{cam}_lengths'][row])))).convert('RGB')
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)

    def batch(self, rows):
        import torch
        images = {key: [] for key in self.camera_map}; states, actions, masks = [], [], []
        for global_row in rows:
            part = int(global_row >= self.base); row = int(global_row - part * self.base); data = self.parts[part]
            for key, cam in self.camera_map.items():
                images[key].append(self._image(part, cam, row))
            indices = row + np.arange(50)
            end = data['end_rows'][row]
            masks.append(indices > end)
            actions.append(data['actions'][np.minimum(indices, end)])
            states.append(data['states'][row])
        batch = {key: torch.stack(v).float() / 255 for key, v in images.items()}
        batch.update({'observation.state': torch.from_numpy(np.stack(states)),
                      'action': torch.from_numpy(np.stack(actions)),
                      'action_is_pad': torch.from_numpy(np.stack(masks)), 'task': [self.task] * len(rows)})
        return batch


def evaluate(args):
    """Closed-loop rollouts from fixed environment seeds; policy sees only images and joint state."""
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import sys
    import torch
    from PIL import Image
    from lerobot.policies import make_pre_post_processors
    from qvla.evaluation.haq_offline_eval import load_policy
    from qvla.evaluation.so101_scripted_expert import NEXUS_SOURCE, make_env
    sys.path.insert(0, NEXUS_SOURCE + '/src')
    from so101_nexus import dataset_row_to_sim_qpos, sim_qpos_to_dataset_row
    cfg = json.loads(args.config.read_text()); run = Path(cfg['output'])
    data_manifest = json.loads((Path(cfg['dataset_root']) / 'manifest.json').read_text())
    stats = json.loads((run / 'stats.json').read_text())
    policy, _, _ = load_policy(SimpleNamespace(model_dir=args.model_dir.resolve(),
        vlm_assets_dir=Path(cfg['vlm_assets_dir']).resolve(), device='cuda'))
    policy.float().eval(); torch.set_num_threads(4)
    pre, post = make_pre_post_processors(policy.config, str(args.model_dir), preprocessor_overrides={
        'device_processor': {'device': 'cuda'}, 'tokenizer_processor': {'tokenizer_name': str(Path(cfg['vlm_assets_dir']).resolve())},
        'normalizer_processor': {'stats': stats}}, postprocessor_overrides={'unnormalizer_processor': {'stats': stats}})
    gen = data_manifest['config']
    env = make_env(image_size=tuple(gen['image_size']), cameras=True, max_episode_steps=args.max_steps + 10)
    limits = (float(env.action_space.low[-1]), float(env.action_space.high[-1]))
    expert = {(e['split'], e['seed']): e['success'] for e in data_manifest['episodes']}
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'scope': f'closed-loop {args.split} seeds; simulator only', 'model_sha256': sha(args.model_dir / 'model.safetensors'),
              'stats_sha256': sha(run / 'stats.json'), 'execute_per_chunk': args.execute, 'max_steps': args.max_steps,
              'num_flow_steps': 10, 'camera_map': cfg['camera_map'], 'episodes': []}
    start, count = gen['splits'][args.split]
    seeds = args.seeds if args.seeds else list(range(start, start + count))
    for seed in seeds:
        obs, _ = env.reset(seed=seed); policy.reset()
        success = grasp = False; steps = 0; chunks = 0; trace = []; frames = []; started = time.perf_counter()
        while steps < args.max_steps:
            state = sim_qpos_to_dataset_row(obs['state'], gripper_limits_rad=limits).astype('float32')
            batch = {key: torch.from_numpy(obs[cam].copy()).permute(2, 0, 1).float() / 255 for key, cam in cfg['camera_map'].items()}
            batch.update({'observation.state': torch.from_numpy(state), 'task': cfg['task']})
            torch.manual_seed(900000 + seed * 100 + chunks)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                actions = post(policy.predict_action_chunk(pre(batch))).float().cpu().numpy()[0]
            chunks += 1
            for row in actions[:args.execute]:
                if steps % 10 == 0 and args.gif:
                    frames.append(Image.fromarray(np.concatenate([obs[c] for c in cfg['camera_map'].values()], axis=1)))
                target = np.clip(dataset_row_to_sim_qpos(row, gripper_limits_rad=limits), env.action_space.low, env.action_space.high)
                obs, _, terminated, _, info = env.step(target.astype('float32')); steps += 1
                success |= bool(info['success']); grasp |= bool(info['is_grasped'] >= 0.5)
                trace.append([float(info['tcp_to_obj_dist']), float(info['obj_to_target_dist']), float(info['is_grasped'])])
                if success and args.stop_on_success:
                    break
            if success and args.stop_on_success:
                break
        result = {'seed': seed, 'success': success, 'ever_grasped': grasp, 'steps': steps, 'chunks': chunks,
                  'expert_success': expert.get((args.split, seed)), 'min_tcp_to_object_m': min(t[0] for t in trace),
                  'final_object_to_target_m': trace[-1][1], 'wall_seconds': time.perf_counter() - started}
        report['episodes'].append(result)
        if frames:
            frames[0].save(args.output / f'seed{seed}.gif', save_all=True, append_images=frames[1:], duration=333, loop=0)
        print(json.dumps(result), flush=True)
    eps = report['episodes']
    report['summary'] = {'success': sum(e['success'] for e in eps), 'grasped': sum(e['ever_grasped'] for e in eps), 'episodes': len(eps),
                         'success_on_expert_feasible': [sum(e['success'] for e in eps if e['expert_success']), sum(1 for e in eps if e['expert_success'])]}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary']), flush=True)
    env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=['train', 'validation', 'test'], default='validation')
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--execute', type=int, default=50)
    p.add_argument('--max-steps', type=int, default=900)
    p.add_argument('--stop-on-success', action='store_true')
    p.add_argument('--gif', action='store_true')
    evaluate(p.parse_args())


if __name__ == '__main__':
    main()
