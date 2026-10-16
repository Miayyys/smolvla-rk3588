"""Replay author-provided SO101 demos with recorded initial scene geometry.

Privileged state is used only to restore replay conditions and measure errors;
it must never be supplied as a learned policy's observation.
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
    p.add_argument('--dataset', type=Path, default=Path('artifacts/datasets/so101_sim_pickplace'))
    p.add_argument('--output', type=Path, default=Path('runs/so101_sim_demo_replay_v1'))
    p.add_argument('--release-tail', action='store_true', help='Append a new slow-release/settle segment; preserve original score separately')
    args = p.parse_args()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    sys.path.insert(0, str(args.source / 'src'))
    import gymnasium as gym
    import mujoco
    import pyarrow.parquet as pq
    from PIL import Image
    import so101_nexus.mujoco  # noqa: F401
    from so101_nexus import JointPositions, OverheadCamera, PickAndPlaceConfig, dataset_row_to_sim_qpos
    from so101_nexus.teleop.config_customization import apply_config_overrides, load_profile_overrides
    from qvla.evaluation.finetune_so100_real import sha

    root = args.dataset
    args.output.mkdir(exist_ok=True, parents=True)
    info = json.loads((root/'meta/info.json').read_text())
    names = info['features']['observation.environment_state']['names']
    t = pq.read_table(root/'data/chunk-000/file-000.parquet')
    a = np.asarray(t['action'].to_pylist(), dtype='float32')
    es = np.asarray(t['observation.environment_state'].to_pylist())
    ids = np.asarray(t['episode_index'])
    config = PickAndPlaceConfig(obs_mode='visual', observations=[JointPositions(), OverheadCamera()])
    config = apply_config_overrides(config, load_profile_overrides(root/'meta/so101_nexus_env.json', 'MuJoCoPickAndPlace-v1', config))
    env = gym.make('MuJoCoPickAndPlace-v1', config=config, control_mode='pd_joint_pos', max_episode_steps=2048)
    report = {
        'scope': 'original simulator action replay, not learned policy evaluation',
        'nexus_revision': subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip(),
        'dataset_table_sha256': sha(root/'data/chunk-000/file-000.parquet'),
        'profile_sha256': sha(root/'meta/so101_nexus_env.json'),
        'dataset_fps': info['fps'], 'native_control_dt': env.unwrapped.control_dt,
        'native_physics_timestep': float(env.unwrapped.model.opt.timestep),
        'restore': 'recorded initial raw joint qpos/qvel, object pose and target position',
        'release_tail': args.release_tail,
        'episodes': [],
    }

    def component(row, prefix, count):
        return np.asarray([es[row, names.index(f'{prefix}_{i}')] for i in range(count)])

    try:
        u = env.unwrapped
        limits = (float(env.action_space.low[-1]), float(env.action_space.high[-1]))
        for ep in sorted(set(ids.tolist())):
            rows = np.flatnonzero(ids == ep)
            first = int(rows[0])
            obs, initial_info = env.reset(seed=int(ep), options={'init_qpos': component(first, 'joint_positions', 6)})
            u.data.qpos[u._qpos_addrs] = component(first, 'joint_positions', 6)
            u.data.qvel[u._qvel_addrs] = component(first, 'joint_velocities', 6)
            slot = u._slots[u._target_slot_idx]
            u.data.qpos[slot.qpos_addr:slot.qpos_addr+7] = component(first, 'object_pose', 7)
            u.data.qvel[slot.dof_addr:slot.dof_addr+6] = 0
            u.model.body_pos[u._target_body_id] = component(first, 'target_position', 3)
            mujoco.mj_forward(u.model, u.data)
            u._refresh_reset_reference_state()
            obs = u._get_obs()
            fk_initial_error = float(np.linalg.norm(u._get_tcp_pose()[:3]-component(first, 'end_effector_pose', 7)[:3]))
            snapshots, robot_errors, object_errors, trace = [], [], [], []
            success = grasp = False
            for i, row in enumerate(rows[:-1]):
                if ep == 0 and i % 10 == 0:
                    snapshots.append(Image.fromarray(obs['overhead_camera']).resize((320,240)))
                target = dataset_row_to_sim_qpos(a[row], gripper_limits_rad=limits)
                obs, reward, terminated, truncated, state = env.step(np.clip(target, env.action_space.low, env.action_space.high).astype('float32'))
                robot_errors.append(np.abs(obs['state']-component(rows[i+1], 'joint_positions', 6)))
                object_errors.append(float(np.linalg.norm(u._get_object_pose()[:3]-component(rows[i+1], 'object_pose', 7)[:3])))
                trace.append({'success': bool(state['success']), 'grasp': bool(state['is_grasped']),
                              'object_to_goal_m': float(state['obj_to_target_dist'])})
                success |= bool(state['success']); grasp |= bool(state['is_grasped'])
                if terminated or truncated: break
            original_success = success
            original_steps = len(trace)
            if args.release_tail and not success and not (terminated or truncated):
                # Use the demonstrator's existing open range rather than a new
                # arbitrary joint calibration. These are NEW generated actions.
                release_row = a[rows[-1]].copy()
                start_grip = float(release_row[-1])
                open_grip = float(np.max(a[:, -1]))
                for j in range(100):
                    release_row[-1] = start_grip + min(1.0, (j+1)/20) * (open_grip-start_grip)
                    target = dataset_row_to_sim_qpos(release_row, gripper_limits_rad=limits)
                    if ep == 0 and j % 10 == 0:
                        snapshots.append(Image.fromarray(obs['overhead_camera']).resize((320,240)))
                    obs, _, terminated, truncated, state = env.step(np.clip(target, env.action_space.low, env.action_space.high).astype('float32'))
                    trace.append({'success': bool(state['success']), 'grasp': bool(state['is_grasped']),
                                  'object_to_goal_m': float(state['obj_to_target_dist'])})
                    success |= bool(state['success'])
                    if terminated or truncated: break
            result = {'episode': int(ep), 'steps': len(trace), 'original_steps': original_steps,
                      'original_success': original_success, 'success': success, 'ever_grasped': grasp,
                      'dataset_reports_success': bool(np.max(np.asarray(t['success'].to_pylist())[rows])),
                      'initial_fk_position_error_m': fk_initial_error,
                      'robot_state_mae_radians': np.asarray(robot_errors).mean(0).tolist(),
                      'object_position_mean_error_m': float(np.mean(object_errors)),
                      'last': trace[-1]}
            report['episodes'].append(result)
            (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            if snapshots:
                snapshots[0].save(args.output/'episode0.gif',save_all=True,append_images=snapshots[1:],duration=333,loop=0)
            print(json.dumps(result),flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
