"""How do two renders of the same reset state differ? (pixel counts, magnitude, location, repeated renders)."""
import json, os, sys
import numpy as np
os.environ.setdefault('MUJOCO_GL', 'egl')
from qvla.evaluation.so101_scripted_expert import make_env
from PIL import Image

env = make_env(image_size=(320, 240), cameras=True, max_episode_steps=910)
u = env.unwrapped
shots = []
for i in range(4):
    obs, _ = env.reset(seed=100006)
    shots.append({c: obs[c].copy() for c in ('overhead_camera', 'wrist_camera')})
out = {}
for c in ('overhead_camera', 'wrist_camera'):
    rows = []
    for i in range(1, 4):
        d = np.abs(shots[0][c].astype(int) - shots[i][c].astype(int))
        ys, xs = np.nonzero(d.max(-1))
        rows.append({'vs_reset': i, 'pixels_differing': int((d.max(-1) > 0).sum()), 'of': int(d.shape[0] * d.shape[1]),
                     'max_abs': int(d.max()), 'mean_abs_over_differing': float(d.max(-1)[d.max(-1) > 0].mean()) if d.any() else 0.0,
                     'bbox': [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if d.any() else None})
    out[c] = rows
# Model state after reset: object pose, qpos, light/material properties identical?
poses = []
for i in range(2):
    env.reset(seed=100006)
    poses.append((u.data.qpos.copy(), u.model.light_pos.copy(), u.model.light_diffuse.copy(), u.model.mat_rgba.copy(), u.model.geom_rgba.copy(),
                  u.model.cam_pos.copy(), u.model.cam_quat.copy(), u.model.cam_fovy.copy()))
names = ['qpos', 'light_pos', 'light_diffuse', 'mat_rgba', 'geom_rgba', 'cam_pos', 'cam_quat', 'cam_fovy']
out['model_state_max_abs_diff'] = {n: float(np.abs(a - b).max()) if a.size else 0.0 for n, a, b in zip(names, poses[0], poses[1])}
# Re-render the same MjData twice without reset.
r1 = u._render_camera('overhead_camera') if hasattr(u, '_render_camera') else None
out['has_render_camera_fn'] = r1 is not None
Image.fromarray(np.concatenate([shots[0]['overhead_camera'], shots[1]['overhead_camera'],
    (np.abs(shots[0]['overhead_camera'].astype(int) - shots[1]['overhead_camera'].astype(int)) * 20).clip(0, 255).astype('uint8')], 1)).save(
    'runs/so101_render_diff.png')
print(json.dumps(out, indent=1))
