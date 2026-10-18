"""Closed-loop SO101 evaluation of a goal-conditioned checkpoint with simulator goals (experiments B and C).

The goal source is the privileged scripted expert run alongside the policy. Its phase machine advances on the
measured robot state every control step (reach / stall / timeout rules, grasp check after lift, bounded retry),
which is the rule gate of the design; at each chunk start its current phase and goal pose, with the TCP from
forward kinematics of the current joint positions, form the goal vector. The expert's own joint commands are
never executed. ``--goal-position-noise-m``/``--goal-rotation-noise-deg`` perturb that goal (experiment C) with
a generator separate from the flow-matching noise, so runs that differ only in goal noise share the same
initial action noise. ``--goal-source null`` uses the learned null-goal embedding instead.

This reads simulator ground truth: a diagnostic upper bound for the goal-conditioned Expert, not a deployable
policy. Execution, seeds and success criteria match ``finetune_so101_scripted.evaluate``.
"""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from qvla.evaluation.finetune_so100_real import sha


def evaluate(args):
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import sys
    import mujoco
    import torch
    from PIL import Image
    from lerobot.policies import make_pre_post_processors
    from qvla.evaluation import so101_goal
    from qvla.evaluation.so101_scripted_expert import NEXUS_SOURCE, Expert, ExpertParams, make_env
    sys.path.insert(0, NEXUS_SOURCE + '/src')
    from so101_nexus import dataset_row_to_sim_qpos, sim_qpos_to_dataset_row
    cfg = json.loads(args.config.read_text()); run = Path(cfg['output'])
    if not cfg.get('goal_conditioning'):
        raise ValueError('Config is not goal-conditioned; use finetune_so101_scripted for baseline checkpoints')
    data_manifest = json.loads((Path(cfg['dataset_root']) / 'manifest.json').read_text())
    stats = json.loads((run / 'stats.json').read_text())
    policy = so101_goal.load_goal_policy(args.model_dir.resolve(), Path(cfg['vlm_assets_dir']), device='cuda')
    policy.float().eval(); torch.set_num_threads(4)
    pre, post = make_pre_post_processors(policy.config, str(args.model_dir), preprocessor_overrides={
        'device_processor': {'device': 'cuda'}, 'tokenizer_processor': {'tokenizer_name': str(Path(cfg['vlm_assets_dir']).resolve())},
        'normalizer_processor': {'stats': stats}}, postprocessor_overrides={'unnormalizer_processor': {'stats': stats}})
    gen = data_manifest['config']
    env = make_env(image_size=tuple(gen['image_size']), cameras=True, max_episode_steps=args.max_steps + 10)
    u = env.unwrapped
    scratch = mujoco.MjData(u.model)
    limits = (float(env.action_space.low[-1]), float(env.action_space.high[-1]))
    expert_success = {(e['split'], e['seed']): e['success'] for e in data_manifest['episodes']}
    args.output.mkdir(parents=True, exist_ok=True)

    def tcp_now():
        # Same forward kinematics as the generator's tcp_pose label (u.data is left untouched).
        scratch.qpos[:] = u.data.qpos
        mujoco.mj_kinematics(u.model, scratch)
        return scratch.site_xpos[u._tcp_site_id].copy(), scratch.site_xmat[u._tcp_site_id].reshape(3, 3).copy()

    report = {'scope': f'closed-loop {args.split} seeds; simulator only; goal from the privileged scripted expert (not deployable)',
              'model_sha256': sha(args.model_dir / 'model.safetensors'), 'stats_sha256': sha(run / 'stats.json'),
              'goal_source': args.goal_source, 'goal_position_noise_m': args.goal_position_noise_m,
              'goal_rotation_noise_deg': args.goal_rotation_noise_deg, 'gate_max_attempts': args.gate_max_attempts,
              'execute_per_chunk': args.execute,
              'max_steps': args.max_steps, 'num_flow_steps': 10, 'camera_map': cfg['camera_map'], 'episodes': []}
    start, count = gen['splits'][args.split]
    seeds = args.seeds if args.seeds else list(range(start, start + count))
    for seed in seeds:
        obs, _ = env.reset(seed=seed); policy.reset()
        expert = Expert(env, ExpertParams(max_attempts=args.gate_max_attempts))
        goal_rng = np.random.default_rng(700000 + seed)
        info = {'is_grasped': 0.0, 'success': False}
        success = grasp = False; steps = chunks = 0; queue = []; trace = []; frames = []; phases = []
        started = time.perf_counter()
        while steps < args.max_steps:
            # Advance the gate on the current observation, as the generator did before recording frame t.
            expert.act(info)
            if not queue:
                state = sim_qpos_to_dataset_row(obs['state'], gripper_limits_rad=limits).astype('float32')
                batch = {key: torch.from_numpy(obs[cam].copy()).permute(2, 0, 1).float() / 255 for key, cam in cfg['camera_map'].items()}
                batch.update({'observation.state': torch.from_numpy(state), 'task': cfg['task']})
                goal_pos, goal_rot, _ = expert.last_goal
                tcp_pos, tcp_rot = tcp_now()
                as_tensor = lambda a: torch.as_tensor(np.asarray(a, dtype='float32'), device='cuda')[None]
                offset = rot_noise = None
                if args.goal_position_noise_m > 0 or args.goal_rotation_noise_deg > 0:
                    axis = goal_rng.standard_normal(3); axis /= np.linalg.norm(axis)
                    offset = as_tensor(goal_rng.standard_normal(3) * args.goal_position_noise_m / np.sqrt(3))
                    angle = goal_rng.standard_normal() * np.radians(args.goal_rotation_noise_deg)
                    rot_noise = so101_goal.axis_angle_matrix(as_tensor(axis), torch.tensor([angle], dtype=torch.float32, device='cuda'))
                vector = so101_goal.goal_vector(torch.tensor([so101_goal.PHASES.index(expert.phase)], device='cuda'),
                                                as_tensor(goal_pos), as_tensor(goal_rot), as_tensor(tcp_pos), as_tensor(tcp_rot),
                                                offset, rot_noise)
                null_mask = torch.tensor([args.goal_source == 'null'], device='cuda')
                torch.manual_seed(900000 + seed * 100 + chunks)
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    so101_goal.set_goal(policy, vector, null_mask)
                    try:
                        actions = post(policy.predict_action_chunk(pre(batch))).float().cpu().numpy()[0]
                    finally:
                        so101_goal.set_goal(policy, None)
                queue = list(actions[:args.execute]); chunks += 1
                phases.append(expert.phase)
            row = queue.pop(0)
            if steps % 10 == 0 and args.gif:
                frames.append(Image.fromarray(np.concatenate([obs[c] for c in cfg['camera_map'].values()], axis=1)))
            target = np.clip(dataset_row_to_sim_qpos(row, gripper_limits_rad=limits), env.action_space.low, env.action_space.high)
            obs, _, terminated, _, info = env.step(target.astype('float32')); steps += 1
            success |= bool(info['success']); grasp |= bool(info['is_grasped'] >= 0.5)
            trace.append([float(info['tcp_to_obj_dist']), float(info['obj_to_target_dist']), float(info['is_grasped'])])
            if success and args.stop_on_success:
                break
        reached = [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]]
        result = {'seed': seed, 'success': success, 'ever_grasped': grasp, 'steps': steps, 'chunks': chunks,
                  'expert_success': expert_success.get((args.split, seed)), 'min_tcp_to_object_m': min(t[0] for t in trace),
                  'final_object_to_target_m': trace[-1][1], 'gate_phase_sequence': reached, 'gate_attempts': expert.attempt,
                  'gate_gave_up': bool(expert.failed), 'wall_seconds': time.perf_counter() - started}
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
    p.add_argument('--execute', type=int, default=10)
    p.add_argument('--max-steps', type=int, default=900)
    p.add_argument('--goal-source', choices=['oracle', 'null'], default='oracle')
    p.add_argument('--goal-position-noise-m', type=float, default=0.0)
    p.add_argument('--goal-rotation-noise-deg', type=float, default=0.0)
    p.add_argument('--gate-max-attempts', type=int, default=2,
                   help='grasp attempts before the gate gives up (the scripted expert and its data use 2)')
    p.add_argument('--stop-on-success', action='store_true')
    p.add_argument('--gif', action='store_true')
    evaluate(p.parse_args())


if __name__ == '__main__':
    main()
