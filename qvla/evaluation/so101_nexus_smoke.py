"""Check a pinned SO101-Nexus checkout without installing its optional extras.

This exercises simulation and rendering, not policy quality or task success.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    revision = subprocess.check_output(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True
    ).strip()
    if revision != '7bdb671aeffa74061993d628f646ae5ba777ae5a':
        raise ValueError('Unexpected SO101-Nexus revision; review before changing the pin')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    sys.path.insert(0, str(source / 'src'))
    import gymnasium as gym
    import so101_nexus.mujoco  # noqa: F401: registers environments
    from so101_nexus import (
        JointPositions, OverheadCamera, PhysicsConfig,
        dataset_row_to_sim_qpos, sim_qpos_to_dataset_row,
    )
    from so101_nexus.env_ids import all_registered_env_ids

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        'scope': 'simulation smoke only; no policy inference or task quality evaluation',
        'source_revision': revision,
        'source_dirty': bool(subprocess.check_output(
            ['git', '-C', str(source), 'status', '--porcelain'], text=True
        ).strip()),
        'mujoco_gl': os.environ['MUJOCO_GL'], 'control_hz': 30,
        'action_mode': 'pd_joint_pos', 'action_units': 'radians',
        'tasks': [],
    }
    for env_id in all_registered_env_ids():
        # Construct the task-specific config while keeping a single RGB camera
        # and six joint positions as the only policy-visible observations.
        spec = gym.spec(env_id)
        module, name = spec.entry_point.split(':')
        import importlib
        cls = getattr(importlib.import_module(module), name)
        config = cls.default_config_cls(
            obs_mode='visual',
            observations=[JointPositions(), OverheadCamera(width=256, height=256)],
            physics=PhysicsConfig(timestep_s=1 / 300, control_period_s=1 / 30),
        )
        env = gym.make(env_id, config=config, control_mode='pd_joint_pos')
        try:
            obs, info = env.reset(seed=0)
            assert obs['state'].shape == (6,)
            image = obs['overhead_camera']
            assert image.shape == (256, 256, 3) and image.dtype == np.uint8
            limits = (float(env.action_space.low[-1]), float(env.action_space.high[-1]))
            row = sim_qpos_to_dataset_row(obs['state'], gripper_limits_rad=limits)
            restored = dataset_row_to_sim_qpos(row, gripper_limits_rad=limits)
            error = float(np.max(np.abs(restored - obs['state'])))
            assert error < 1e-5
            action = np.clip(restored, env.action_space.low, env.action_space.high).astype('float32')
            for _ in range(10):
                obs, reward, terminated, truncated, info = env.step(action)
                assert np.isfinite(obs['state']).all() and np.isfinite(reward)
                if terminated or truncated:
                    raise RuntimeError('Smoke episode ended unexpectedly')
            from PIL import Image
            Image.fromarray(image).save(args.output / f'{env_id}.png')
            record = {
                'env_id': env_id, 'status': 'passed', 'steps': 10,
                'action_shape': list(env.action_space.shape),
                'rgb_shape': list(image.shape), 'control_dt': env.unwrapped.control_dt,
                'task_description': getattr(env.unwrapped, 'task_description', None),
                'unit_roundtrip_max_error': error,
                'initial_image_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                'success_flag_present': 'success' in info,
            }
            report['tasks'].append(record)
            print(json.dumps(record), flush=True)
        finally:
            env.close()
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
