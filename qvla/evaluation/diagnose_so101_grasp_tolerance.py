"""Replay a recorded matched-sim demonstration with controlled perturbations.

Measures how much constant joint bias or gripper timing shift the training
demonstration itself tolerates. No model is run; this is a scene tolerance
diagnostic, not a policy result.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from qvla.evaluation.so101_paths import NEXUS_SOURCE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--source-episode', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--trials', type=int, default=6)
    args = p.parse_args()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    sys.path.insert(0, NEXUS_SOURCE+'/src')
    from so101_nexus import dataset_row_to_sim_qpos
    from qvla.evaluation.prepare_so101_sim_training import make_scene
    from qvla.evaluation.finetune_so100_real import sha
    cfg = json.loads(args.config.read_text()); run = Path(cfg['output'])
    prepared = json.loads((run / 'prepared.json').read_text())
    scene = next(x for x in prepared['scene_initializations'] if x['episode'] == args.source_episode)
    data = np.load(run / 'data.npz'); demo = data['actions'][data['episodes'] == args.source_episode]

    def replay(actions):
        env, _ = make_scene(cfg['dataset_root'], scene['environment_state_names'], scene['initial_environment_state'], cfg, NEXUS_SOURCE)
        try:
            limits = (float(env.action_space.low[-1]), float(env.action_space.high[-1]))
            success = grasp = False; tcp = []; goal = []
            for row in actions:
                target = np.clip(dataset_row_to_sim_qpos(row, gripper_limits_rad=limits), env.action_space.low, env.action_space.high).astype('float32')
                _, _, terminated, truncated, info = env.step(target)
                success |= bool(info['success']); grasp |= bool(info['is_grasped'])
                tcp.append(float(info['tcp_to_obj_dist'])); goal.append(float(info['obj_to_target_dist']))
                if terminated or truncated: break
            return {'success': success, 'ever_grasped': grasp, 'min_tcp_to_object_m': min(tcp),
                    'min_tcp_step': int(np.argmin(tcp)), 'tcp_to_object_m': tcp, 'object_to_goal_m': goal}
        finally:
            env.close()

    report = {'scope': 'demonstration replay tolerance; no policy', 'source_episode': args.source_episode,
              'data_sha256': sha(run / 'data.npz'), 'demo': replay(demo), 'joint_bias': [], 'gripper_shift': []}
    rng = np.random.default_rng(0)
    for sigma in [0.5, 1.0, 2.0, 3.0]:
        for trial in range(args.trials):
            offset = np.zeros(6, dtype='float32'); offset[:5] = rng.normal(0, sigma, 5)
            r = replay(demo + offset)
            report['joint_bias'].append({'sigma_deg': sigma, 'trial': trial, 'offset_deg': offset[:5].tolist(),
                                         **{k: r[k] for k in ('success', 'ever_grasped', 'min_tcp_to_object_m')}})
    for shift in [-15, -8, -4, 4, 8, 15]:
        shifted = demo.copy(); g = demo[:, -1]
        shifted[:, -1] = np.roll(g, shift)
        if shift > 0: shifted[:shift, -1] = g[0]
        else: shifted[shift:, -1] = g[-1]
        r = replay(shifted)
        report['gripper_shift'].append({'shift_steps': shift, **{k: r[k] for k in ('success', 'ever_grasped', 'min_tcp_to_object_m')}})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    d = report['demo']
    print('demo', d['success'], d['ever_grasped'], round(d['min_tcp_to_object_m'] * 1000, 1), d['min_tcp_step'])
    for k in range(120, 215, 10):
        print(' step', k, 'tcp_mm', round(d['tcp_to_object_m'][k] * 1000, 1), 'obj2goal', round(d['object_to_goal_m'][k], 3), 'grip', round(float(demo[k, -1]), 1))
    for sigma in [0.5, 1.0, 2.0, 3.0]:
        rs = [r for r in report['joint_bias'] if r['sigma_deg'] == sigma]
        print('bias sigma', sigma, 'grasp', sum(r['ever_grasped'] for r in rs), '/', len(rs), 'success', sum(r['success'] for r in rs))
    for r in report['gripper_shift']:
        print('gripper shift', r['shift_steps'], 'grasp', r['ever_grasped'], 'success', r['success'])


if __name__ == '__main__':
    main()
