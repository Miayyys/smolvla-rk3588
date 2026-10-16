"""Independent checks on a generated scripted SO101 dataset.

1. Re-execute stored executed commands from the same seed and compare states/success.
2. Summaries by split, noise level and object distance from the manifest.
3. A contact sheet of decoded frames from both cameras.
"""

import argparse
import io
import json
import os
from pathlib import Path
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--replays', type=int, default=8)
    args = p.parse_args()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    from PIL import Image
    from qvla.evaluation.so101_scripted_expert import NEXUS_SOURCE, make_env
    sys.path.insert(0, NEXUS_SOURCE + '/src')
    from so101_nexus import dataset_row_to_sim_qpos, sim_qpos_to_dataset_row
    manifest = json.loads((args.data / 'manifest.json').read_text())
    cfg = manifest['config']
    report = {'replay': [], 'summary': {}}
    eps = manifest['episodes']
    for split in cfg['splits']:
        rows = [e for e in eps if e['split'] == split]
        by_noise = {}
        for e in rows:
            by_noise.setdefault(str(e['noise_std_deg']), []).append(e['success'])
        r = np.array([np.hypot(*e['object_xy']) for e in rows]); ok = np.array([e['success'] for e in rows])
        bins = [0, 0.25, 0.3, 0.35, 0.4, 0.6]
        report['summary'][split] = {
            'attempted': len(rows), 'success': int(ok.sum()),
            'by_noise_deg': {k: [int(sum(v)), len(v)] for k, v in by_noise.items()},
            'by_object_radius_m': {f'{bins[i]}-{bins[i+1]}': [int(ok[(r >= bins[i]) & (r < bins[i+1])].sum()), int(((r >= bins[i]) & (r < bins[i+1])).sum())] for i in range(len(bins) - 1)},
            'stored_frames_per_episode': [int(np.min([e['frames'] for e in rows if e['success']])), float(np.mean([e['frames'] for e in rows if e['success']])), int(np.max([e['frames'] for e in rows if e['success']]))],
            'failed_seeds': [e['seed'] for e in rows if not e['success']]}
    data = np.load(args.data / 'train.npz')
    # Same observation config as generation: camera randomization may consume reset RNG.
    env = make_env(image_size=tuple(cfg['image_size']), cameras=True, max_episode_steps=cfg['max_steps'] + 10)
    limits = (float(env.action_space.low[-1]), float(env.action_space.high[-1]))
    seeds = np.unique(data['episodes'])[:: max(1, len(np.unique(data['episodes'])) // args.replays)][: args.replays]
    for seed in seeds:
        rows = np.flatnonzero(data['episodes'] == seed)
        obs, _ = env.reset(seed=int(seed)); err = 0.0; success = False
        for row in rows:
            state = sim_qpos_to_dataset_row(obs['state'], gripper_limits_rad=limits)
            err = max(err, float(np.abs(state - data['states'][row]).max()))
            command = np.clip(dataset_row_to_sim_qpos(data['executed'][row], gripper_limits_rad=limits), env.action_space.low, env.action_space.high)
            obs, _, _, _, info = env.step(command.astype('float32'))
        report['replay'].append({'seed': int(seed), 'frames': len(rows), 'max_state_error_dataset_units': err,
                                 'final_success': bool(info['success'])})
    env.close()
    sheet = Image.new('RGB', (cfg['image_size'][0] * 4, cfg['image_size'][1] * 2))
    ep_rows = np.flatnonzero(data['episodes'] == seeds[0])
    for i, frac in enumerate([0.0, 0.3, 0.5, 0.95]):
        row = ep_rows[int(frac * (len(ep_rows) - 1))]
        for j, cam in enumerate(cfg['cameras']):
            with (args.data / f'train_{cam}.jpeg.bin').open('rb') as f:
                f.seek(int(data[f'{cam}_offsets'][row])); im = Image.open(io.BytesIO(f.read(int(data[f'{cam}_lengths'][row]))))
            sheet.paste(im, (i * cfg['image_size'][0], j * cfg['image_size'][1]))
    sheet.save(args.data / 'contact_sheet.jpg')
    (args.data / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
