"""Check real-demo pose coverage and simulator tracking without a learned policy.

Forward kinematics assumes the current degree/gripper mapping. It does not
reconstruct the original real scene or validate its unknown calibration.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('/home/loser/Study/so101-nexus'))
    p.add_argument('--training-run', type=Path, default=Path('runs/so100_real_finetune_v1'))
    p.add_argument('--output', type=Path, default=Path('runs/so101_transfer_diagnosis_v1'))
    args = p.parse_args()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    sys.path.insert(0, str(args.source / 'src'))
    import gymnasium as gym
    import mujoco
    from PIL import Image
    import so101_nexus.mujoco  # noqa: F401
    from so101_nexus import (
        JointPositions, OverheadCamera, PhysicsConfig, PickAndPlaceV2Config,
        SO101_JOINT_NAMES, dataset_row_to_sim_qpos, sim_qpos_to_dataset_row,
    )
    revision = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != '7bdb671aeffa74061993d628f646ae5ba777ae5a':
        raise ValueError('Unexpected simulator revision')
    args.output.mkdir(exist_ok=True, parents=True)
    z = np.load(args.training_run / 'data.npz')
    stats = json.loads((args.training_run / 'stats.json').read_text())
    prepared = json.loads((args.training_run / 'prepared.json').read_text())
    config = PickAndPlaceV2Config(
        obs_mode='visual', observations=[JointPositions(), OverheadCamera(width=640, height=480)],
        physics=PhysicsConfig(timestep_s=1/300, control_period_s=1/30),
    )
    env = gym.make('MuJoCoPickAndPlace-v2', config=config, control_mode='pd_joint_pos', max_episode_steps=1024)
    try:
        obs, _ = env.reset(seed=0)
        u = env.unwrapped
        limits = (float(env.action_space.low[-1]), float(env.action_space.high[-1]))
        initial = sim_qpos_to_dataset_row(obs['state'], gripper_limits_rad=limits)
        reference = u._placement_reference_pos().copy()
        sid = mujoco.mj_name2id(u.model, mujoco.mjtObj.mjOBJ_SITE, 'gripperframe')
        train_rows = z['train_rows']
        points, valid = [], []
        for row in train_rows:
            q = dataset_row_to_sim_qpos(z['states'][row], gripper_limits_rad=limits)
            valid.append(bool(np.all((q >= env.action_space.low-1e-5) & (q <= env.action_space.high+1e-5))))
            u.data.qpos[u._qpos_addrs] = q
            mujoco.mj_forward(u.model, u.data)
            points.append(u.data.site_xpos[sid].copy())
        points = np.asarray(points)
        states = z['states'][train_rows]
        distance = np.linalg.norm(points-reference, axis=1)
        report = {
            'scope': 'real-to-sim mapping/coverage diagnostic; not policy success benchmark',
            'nexus_revision': revision, 'joint_names': list(SO101_JOINT_NAMES),
            'sim_initial_row': initial.tolist(),
            'training_state_mean': stats['observation.state']['mean'],
            'initial_abs_zscores': np.abs((initial-np.asarray(stats['observation.state']['mean'])) / np.asarray(stats['observation.state']['std'])).tolist(),
            'training_state_p01': np.percentile(states, 1, axis=0).tolist(),
            'training_state_p99': np.percentile(states, 99, axis=0).tolist(),
            'sim_object_placement_reference_xyz': reference.tolist(),
            'mapped_training_fk_xyz_min': points.min(0).tolist(),
            'mapped_training_fk_xyz_max': points.max(0).tolist(),
            'mapped_fk_min_distance_to_object_reference_m': float(distance.min()),
            'mapped_fk_frames_within_3cm': int(np.count_nonzero(distance < .03)),
            'fraction_within_sim_joint_limits': float(np.mean(valid)),
            'training_frames': len(train_rows), 'replays': [],
        }
        # Bypass policy/vision: execute original successful real action sequences.
        for episode in prepared['splits']['train'][:3]:
            rows = np.flatnonzero(z['episodes'] == episode)
            start = dataset_row_to_sim_qpos(z['states'][rows[0]], gripper_limits_rad=limits)
            obs, info = env.reset(seed=0, options={'init_qpos': start})
            errors, tracking, snapshots = [], [], []
            ever_grasped = success = False
            clipped = total = 0
            for index, row in enumerate(rows[:-1]):
                if index % 10 == 0 and episode == prepared['splits']['train'][0]:
                    snapshots.append(Image.fromarray(obs['overhead_camera']).resize((320, 240)))
                target = dataset_row_to_sim_qpos(z['actions'][row], gripper_limits_rad=limits)
                action = np.clip(target, env.action_space.low, env.action_space.high).astype('float32')
                clipped += int(np.count_nonzero(action != target)); total += action.size
                obs, _, terminated, truncated, info = env.step(action)
                physical = sim_qpos_to_dataset_row(obs['state'], gripper_limits_rad=limits)
                errors.append(np.abs(physical-z['states'][rows[index+1]]))
                tracking.append(physical)
                success |= bool(info['success']); ever_grasped |= bool(info['is_grasped'])
                if terminated or truncated: break
            record = {'episode': episode, 'steps': len(errors), 'success': success,
                      'ever_grasped': ever_grasped,
                      'state_mae_vs_next_real_state': np.asarray(errors).mean(0).tolist(),
                      'units': 'degrees for five arm joints, gripper percentage points',
                      'clipped_value_fraction': clipped/total,
                      'final_tcp_to_object_m': float(info['tcp_to_obj_dist'])}
            report['replays'].append(record)
            np.savez_compressed(args.output / f'gt_replay_{episode}.npz', states=np.asarray(tracking), errors=np.asarray(errors))
            if snapshots:
                snapshots[0].save(args.output/'gt_replay.gif', save_all=True, append_images=snapshots[1:], duration=333, loop=0)
            print(json.dumps(record), flush=True)
        np.savez_compressed(args.output/'fk.npz', points=points, train_rows=train_rows, object_reference=reference)
        (args.output/'geometry.json').write_text(json.dumps(report, indent=2)+'\n')
    finally:
        env.close()


if __name__ == '__main__':
    main()
