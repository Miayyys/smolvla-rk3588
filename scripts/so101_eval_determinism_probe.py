"""Locate run-to-run nondeterminism in the SO101 closed-loop evaluation (rendering, policy, physics)."""
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'; os.environ.setdefault('MUJOCO_GL', 'egl')
import torch
from lerobot.policies import make_pre_post_processors
from qvla.evaluation import so101_goal
from qvla.evaluation.so101_scripted_expert import NEXUS_SOURCE, make_env
sys.path.insert(0, NEXUS_SOURCE + '/src')
from so101_nexus import dataset_row_to_sim_qpos, sim_qpos_to_dataset_row

cfg = json.load(open('config/so101_goal_B.json')); run = Path(cfg['output'])
model = Path('runs/so101_goal_B/snapshot_step5000/best')
stats = json.loads((run / 'stats.json').read_text())
env = make_env(image_size=(320, 240), cameras=True, max_episode_steps=910)
limits = (float(env.action_space.low[-1]), float(env.action_space.high[-1]))
seed = 100006
out = {}

obs1, _ = env.reset(seed=seed); obs2, _ = env.reset(seed=seed)
out['render_reset_equal'] = {c: bool(np.array_equal(obs1[c], obs2[c])) for c in cfg['camera_map'].values()}

# Physics with a fixed open-loop command sequence.
rng = np.random.default_rng(0); commands = [env.action_space.sample() * 0 + env.unwrapped.data.qpos[env.unwrapped._qpos_addrs] + rng.normal(0, 0.02, 6) for _ in range(60)]
finals = []
for _ in range(2):
    env.reset(seed=seed)
    for c in commands:
        obs, *_ = env.step(np.clip(c, env.action_space.low, env.action_space.high).astype('float32'))
    finals.append((obs['state'].copy(), {k: obs[k].copy() for k in cfg['camera_map'].values()}))
out['physics_state_max_abs_diff'] = float(np.abs(finals[0][0] - finals[1][0]).max())
out['render_after_steps_equal'] = {k: bool(np.array_equal(finals[0][1][k], finals[1][1][k])) for k in finals[0][1]}

policy = so101_goal.load_goal_policy(model.resolve(), Path(cfg['vlm_assets_dir']), device='cuda'); policy.float().eval()
pre, post = make_pre_post_processors(policy.config, str(model), preprocessor_overrides={
    'device_processor': {'device': 'cuda'}, 'tokenizer_processor': {'tokenizer_name': str(Path(cfg['vlm_assets_dir']).resolve())},
    'normalizer_processor': {'stats': stats}}, postprocessor_overrides={'unnormalizer_processor': {'stats': stats}})
state = sim_qpos_to_dataset_row(obs1['state'], gripper_limits_rad=limits).astype('float32')
vector = torch.zeros(1, so101_goal.GOAL_DIM, device='cuda'); vector[0, 0] = 1
chunks = []
for autocast in (True, True, False, False):
    batch = {k: torch.from_numpy(obs1[c].copy()).permute(2, 0, 1).float() / 255 for k, c in cfg['camera_map'].items()}
    batch.update({'observation.state': torch.from_numpy(state), 'task': cfg['task']})
    policy.reset(); torch.manual_seed(1234)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
        so101_goal.set_goal(policy, vector)
        chunks.append(post(policy.predict_action_chunk(pre(batch))).float().cpu().numpy()[0])
        so101_goal.set_goal(policy, None)
out['policy_bf16_repeat_max_abs_diff'] = float(np.abs(chunks[0] - chunks[1]).max())
out['policy_fp32_repeat_max_abs_diff'] = float(np.abs(chunks[2] - chunks[3]).max())
out['policy_bf16_vs_fp32_max_abs_diff'] = float(np.abs(chunks[0] - chunks[2]).max())
out['torch'] = torch.__version__; out['cudnn_deterministic'] = torch.backends.cudnn.deterministic
print(json.dumps(out, indent=1))
