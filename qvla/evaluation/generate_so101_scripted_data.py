"""Generate scripted-expert SO101-Nexus pick-and-place demonstrations.

Splits are fixed by environment seed before any success filtering. Training
episodes draw an execution-noise level per seed (DART-style: the recorded action
is the clean expert label for the measured state, the executed command adds
noise in coarse transport phases). Validation/test episodes are noiseless.
Only successful episodes are stored as demonstrations; every attempt, including
failures, is written to the report. Normalization statistics use successful
training frames only.
"""

import argparse
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np

PHASES = ['pregrasp', 'descend', 'close', 'lift', 'carry', 'lower', 'open', 'retreat', 'hold']


def worker(job):
    os.environ.setdefault('MUJOCO_GL', 'egl')
    from PIL import Image
    from qvla.evaluation.so101_scripted_expert import make_env, run_episode
    import sys
    from qvla.evaluation.so101_scripted_expert import NEXUS_SOURCE
    sys.path.insert(0, NEXUS_SOURCE + '/src')
    from so101_nexus import sim_qpos_to_dataset_row
    cfg, split, seeds, shard = job
    out = Path(cfg['output']) / 'shards'; out.mkdir(parents=True, exist_ok=True)
    w, h = cfg['image_size']
    env = make_env(image_size=(w, h), cameras=True, max_episode_steps=cfg['max_steps'] + 10)
    u = env.unwrapped
    limits = (float(env.action_space.low[-1]), float(env.action_space.high[-1]))
    results = []
    arrays = {k: [] for k in ('states', 'actions', 'executed', 'episodes', 'frame_index', 'phase')}
    streams = {cam: (out / f'{split}_{shard}_{cam}.jpeg.bin').open('wb') for cam in cfg['cameras']}
    offsets = {cam: [] for cam in cfg['cameras']}; lengths = {cam: [] for cam in cfg['cameras']}
    try:
        for seed in seeds:
            noise = cfg['noise_levels_deg'][seed % len(cfg['noise_levels_deg'])] if split == 'train' else 0.0
            buffer = []

            def record(obs, label, executed, phase):
                images = {}
                for cam in cfg['cameras']:
                    b = io.BytesIO(); Image.fromarray(obs[cam]).save(b, format='JPEG', quality=cfg['jpeg_quality'])
                    images[cam] = b.getvalue()
                buffer.append((sim_qpos_to_dataset_row(obs['state'], gripper_limits_rad=limits).astype('float32'),
                               sim_qpos_to_dataset_row(label, gripper_limits_rad=limits).astype('float32'),
                               sim_qpos_to_dataset_row(executed, gripper_limits_rad=limits).astype('float32'),
                               PHASES.index(phase), images))

            started = time.perf_counter()
            result = run_episode(env, seed, noise_std_deg=noise, max_steps=cfg['max_steps'], recorder=record)
            result.update({'split': split, 'frames': len(buffer), 'seconds': time.perf_counter() - started,
                           'task': u.task_description,
                           'object_xy': u._get_object_pose()[:2].tolist(), 'target_xy': u._get_target_pos()[:2].tolist()})
            result['stored'] = bool(result['success'])
            if result['stored']:
                for index, (state, action, executed, phase, images) in enumerate(buffer):
                    arrays['states'].append(state); arrays['actions'].append(action); arrays['executed'].append(executed)
                    arrays['episodes'].append(seed); arrays['frame_index'].append(index); arrays['phase'].append(phase)
                    for cam in cfg['cameras']:
                        offsets[cam].append(streams[cam].tell()); lengths[cam].append(len(images[cam]))
                        streams[cam].write(images[cam])
            results.append(result)
            print(json.dumps({k: result[k] for k in ('split', 'seed', 'success', 'frames', 'noise_std_deg', 'attempts')}), flush=True)
    finally:
        for s in streams.values():
            s.close()
        env.close()
    np.savez(out / f'{split}_{shard}.npz', **{k: np.asarray(v) for k, v in arrays.items()},
             **{f'{cam}_offsets': np.asarray(offsets[cam], dtype='int64') for cam in cfg['cameras']},
             **{f'{cam}_lengths': np.asarray(lengths[cam], dtype='int64') for cam in cfg['cameras']})
    return results


def merge(cfg, all_results):
    from qvla.evaluation.finetune_so100_real import sha
    out = Path(cfg['output']); shards = out / 'shards'
    manifest = {'config': cfg, 'phases': PHASES, 'splits': {}, 'episodes': all_results}
    for split in cfg['splits']:
        parts = sorted(shards.glob(f'{split}_*.npz'), key=lambda p: int(p.stem.split('_')[1]))
        merged = {k: [] for k in ('states', 'actions', 'executed', 'episodes', 'frame_index', 'phase')}
        cam_offsets = {cam: [] for cam in cfg['cameras']}; cam_lengths = {cam: [] for cam in cfg['cameras']}
        targets = {cam: (out / f'{split}_{cam}.jpeg.bin').open('wb') for cam in cfg['cameras']}
        try:
            for part in parts:
                data = np.load(part)
                if len(data['episodes']) == 0:
                    continue
                for k in merged:
                    merged[k].append(data[k])
                for cam in cfg['cameras']:
                    base = targets[cam].tell()
                    targets[cam].write((shards / f'{part.stem}_{cam}.jpeg.bin').read_bytes())
                    cam_offsets[cam].append(data[f'{cam}_offsets'] + base); cam_lengths[cam].append(data[f'{cam}_lengths'])
        finally:
            for t in targets.values():
                t.close()
        merged = {k: np.concatenate(v) for k, v in merged.items()}
        order = np.lexsort((merged['frame_index'], merged['episodes']))
        merged = {k: v[order] for k, v in merged.items()}
        for cam in cfg['cameras']:
            merged[f'{cam}_offsets'] = np.concatenate(cam_offsets[cam])[order]
            merged[f'{cam}_lengths'] = np.concatenate(cam_lengths[cam])[order]
        end_rows = np.empty(len(merged['episodes']), dtype='int64')
        for ep in np.unique(merged['episodes']):
            rows = np.flatnonzero(merged['episodes'] == ep); end_rows[rows] = rows[-1]
        merged['end_rows'] = end_rows
        np.savez(out / f'{split}.npz', **merged)
        attempted = [r for r in all_results if r['split'] == split]
        manifest['splits'][split] = {
            'seeds_attempted': len(attempted), 'episodes_stored': int(len(np.unique(merged['episodes']))),
            'frames': int(len(merged['episodes'])),
            'expert_success_rate': float(np.mean([r['success'] for r in attempted])),
            'data_sha256': sha(out / f'{split}.npz'),
            **{f'{cam}_sha256': sha(out / f'{split}_{cam}.jpeg.bin') for cam in cfg['cameras']},
            **{f'{cam}_bytes': (out / f'{split}_{cam}.jpeg.bin').stat().st_size for cam in cfg['cameras']}}
        if split == 'train':
            stats = {key: {'mean': merged[src].mean(0).tolist(), 'std': np.maximum(merged[src].std(0), 1e-6).tolist()}
                     for key, src in (('observation.state', 'states'), ('action', 'actions'))}
            (out / 'stats.json').write_text(json.dumps(stats, indent=2) + '\n')
            manifest['stats_sha256'] = sha(out / 'stats.json')
    for f in shards.iterdir():
        f.unlink()
    shards.rmdir()
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    out = Path(cfg['output'])
    if (out / 'manifest.json').exists():
        raise ValueError('Output already generated; preserve it')
    out.mkdir(parents=True, exist_ok=True)
    from qvla.evaluation.finetune_so100_real import sha
    cfg['expert_source_sha256'] = sha('qvla/evaluation/so101_scripted_expert.py')
    cfg['generator_source_sha256'] = sha('qvla/evaluation/generate_so101_scripted_data.py')
    jobs = []
    for split, (start, count) in cfg['splits'].items():
        seeds = list(range(start, start + count))
        for shard in range(cfg['workers']):
            part = seeds[shard::cfg['workers']]
            if part:
                jobs.append((cfg, split, part, shard))
    started = time.time()
    with mp.get_context('spawn').Pool(cfg['workers']) as pool:
        results = [r for chunk in pool.imap_unordered(worker, jobs) for r in chunk]
    results.sort(key=lambda r: (r['split'], r['seed']))
    manifest = merge(cfg, results)
    manifest['wall_seconds'] = time.time() - started
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'stage': 'generated', 'wall_seconds': manifest['wall_seconds'],
                      **{s: {k: v for k, v in m.items() if 'sha' not in k} for s, m in manifest['splits'].items()}}), flush=True)


if __name__ == '__main__':
    main()
