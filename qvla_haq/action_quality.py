"""Offline action proxy for search; does not estimate task success."""
import math
import numpy as np


def action_quality(candidate, reference, metrics, mode='chunk_mae'):
    scale = max(float(np.abs(reference.astype(np.float64)).mean()), 1e-8)
    chunk = metrics['task_macro']['chunk_mae_vs_fp']/scale
    if mode == 'chunk_mae':
        error = chunk
        velocity = gripper = None
    elif mode == 'trajectory_gripper':
        # Equal per-observation weighting: current cache has one observation/task.
        ref = reference.astype(np.float64)
        cand = candidate.astype(np.float64)
        delta_ref = np.diff(ref[:,:,:6], axis=1)
        delta = np.diff(cand[:,:,:6], axis=1)
        velocity_scale = max(float(np.abs(delta_ref).mean()), 1e-3)
        velocity = float(np.abs(delta-delta_ref).mean())/velocity_scale
        gripper = metrics['task_macro']['gripper_sign_disagreement_vs_fp']
        error = .8*chunk+.1*velocity+.1*gripper
    else:
        raise ValueError('Unknown offline quality mode')
    return {'quality': math.exp(-error), 'normalized_error': error,
            'chunk_normalized_mae': chunk, 'velocity_normalized_mae': velocity,
            'gripper_sign_disagreement': gripper, 'mode': mode,
            'scope': 'offline_proxy_not_task_success'}
