"""Chained (long-horizon) closed-loop SO101 evaluation: several pick-and-place tasks in one episode.

After each success the scene is re-laid out without resetting the robot: the environment's own task reset
(disc and cube positions, cube yaw; next draws of the seed's RNG stream), forward kinematics, the configured
settle frames and the reference-state refresh, i.e. the second half of ``reset()``. The robot keeps its current
pose and actuator targets, so every task after the first starts from wherever the previous one ended. Each task
has ``--max-steps``; the episode ends at the first task that runs out of steps. Per chunk the policy is queried as
in the other evaluations (execute ``--execute`` actions; flow noise seeded per seed and chunk).

Policies: a plain SmolVLA checkpoint (no goal input), or a goal-conditioned one with ``--goal-source`` planner
(deployable: the planner head's phase/goal and forward kinematics of the measured joints), takeover (simulator
expert goal, upper bound) or null. Training/validation/test seeds as in the data manifest; default validation.
"""

import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

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
    from qvla.evaluation.collect_so101_dagger import _snapshot, takeover_expert
    from qvla.evaluation.haq_offline_eval import load_policy
    from qvla.evaluation.so101_scripted_expert import NEXUS_SOURCE, Expert, ExpertParams, make_env
    sys.path.insert(0, NEXUS_SOURCE + '/src')
    from so101_nexus import dataset_row_to_sim_qpos, sim_qpos_to_dataset_row
    cfg = json.loads(args.config.read_text()); run = Path(cfg['output'])
    goal_model = bool(cfg.get('goal_conditioning'))
    if goal_model == (args.goal_source == 'none'):
        raise ValueError('--goal-source none is for plain checkpoints; goal-conditioned ones need planner/takeover/null')
    stats = json.loads((Path(args.stats_dir or run) / 'stats.json').read_text())
    if goal_model:
        policy = so101_goal.load_goal_policy(args.model_dir.resolve(), Path(cfg['vlm_assets_dir']), device='cuda')
    else:
        policy, _, _ = load_policy(SimpleNamespace(model_dir=args.model_dir.resolve(),
            vlm_assets_dir=Path(cfg['vlm_assets_dir']).resolve(), device='cuda'))
    policy.float().eval(); torch.set_num_threads(4)
    pre, post = make_pre_post_processors(policy.config, str(args.model_dir), preprocessor_overrides={
        'device_processor': {'device': 'cuda'}, 'tokenizer_processor': {'tokenizer_name': str(Path(cfg['vlm_assets_dir']).resolve())},
        'normalizer_processor': {'stats': stats}}, postprocessor_overrides={'unnormalizer_processor': {'stats': stats}})
    data_manifest = json.loads((Path(cfg['dataset_root']) / 'manifest.json').read_text()); gen = data_manifest['config']
    env = make_env(image_size=tuple(gen['image_size']), cameras=True, max_episode_steps=args.chain * args.max_steps + 10)
    u = env.unwrapped
    limits = (float(env.action_space.low[-1]), float(env.action_space.high[-1]))
    observed = mujoco.MjData(u.model); simulated = mujoco.MjData(u.model)

    def tcp_from_observed(state_row):
        q = dataset_row_to_sim_qpos(state_row, gripper_limits_rad=limits)
        observed.qpos[u._arm_qpos_addrs] = q[:5]; mujoco.mj_kinematics(u.model, observed)
        return observed.site_xpos[u._tcp_site_id].copy(), observed.site_xmat[u._tcp_site_id].reshape(3, 3).copy()

    def tcp_from_sim():
        simulated.qpos[:] = u.data.qpos; mujoco.mj_kinematics(u.model, simulated)
        return simulated.site_xpos[u._tcp_site_id].copy(), simulated.site_xmat[u._tcp_site_id].reshape(3, 3).copy()

    def relayout():
        """Second half of reset(): new disc/cube layout from the seed's RNG stream; the robot is not touched."""
        u._task_reset(); mujoco.mj_forward(u.model, u.data); u._settle_after_reset(); u._refresh_reset_reference_state()
        return u._observe()

    args.output.mkdir(parents=True, exist_ok=True)
    report = {'scope': f'chained closed loop ({args.chain} pick-and-place tasks per episode, robot not reset between tasks); '
                       f'{args.split} seeds; simulator only', 'model_sha256': sha(args.model_dir / 'model.safetensors'),
              'stats_sha256': sha(Path(args.stats_dir or run) / 'stats.json'), 'goal_source': args.goal_source,
              'chain': args.chain, 'execute_per_chunk': args.execute, 'max_steps_per_task': args.max_steps,
              'num_flow_steps': 10, 'episodes': []}
    start, count = gen['splits'][args.split]
    seeds = args.seeds if args.seeds else list(range(start, start + count))
    for seed in seeds:
        obs, _ = env.reset(seed=seed); policy.reset()
        info = {'is_grasped': 0.0, 'success': False}
        tasks = []; chunks = 0; frames = []; started = time.perf_counter()
        for task in range(args.chain):
            if task:
                obs, info = relayout(); policy.reset()
            layout = {'cube_xy': u._get_object_pose()[:2].round(4).tolist(), 'disc_xy': u._get_target_pos()[:2].round(4).tolist()}
            tracker = Expert(env, ExpertParams(max_attempts=10)) if goal_model else None
            queue = []; steps = 0; success = grasp = False; planner_log = []; min_tcp = np.inf
            while steps < args.max_steps:
                before = _snapshot(tracker) if tracker else None
                if tracker:
                    tracker.act(info)
                if not queue:
                    state = sim_qpos_to_dataset_row(obs['state'], gripper_limits_rad=limits).astype('float32')
                    batch = {key: torch.from_numpy(obs[cam].copy()).permute(2, 0, 1).float() / 255 for key, cam in cfg['camera_map'].items()}
                    batch.update({'observation.state': torch.from_numpy(state), 'task': cfg['task']})
                    vector = null_mask = None
                    if goal_model:
                        take, _ = takeover_expert(env, before, info['is_grasped'] >= 0.5, max_attempts=5)
                        take.act(info); goal_pos, goal_rot, _ = take.last_goal; goal_phase = take.phase
                        tcp_pos, tcp_rot = tcp_from_sim()
                        if args.goal_source == 'planner':
                            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                                prefix_out, pad, pstate = so101_goal.prefix_features(policy, pre(dict(batch)))
                                planned = policy.model.goal_planner(policy.model.goal_planner.slow_features(prefix_out, pad), pstate)
                            planned_phase = so101_goal.PHASES[int(planned['phase_logits'].argmax(-1)[0])]
                            planned_pos = planned['goal_pos'][0].float().cpu().numpy()
                            planner_log.append((planned_phase == goal_phase, float(np.linalg.norm(planned_pos - goal_pos))))
                            goal_pos, goal_rot, goal_phase = planned_pos, planned['goal_rot'][0].float().cpu().numpy(), planned_phase
                            tcp_pos, tcp_rot = tcp_from_observed(state)
                        t = lambda a: torch.as_tensor(np.asarray(a, dtype='float32'), device='cuda')[None]
                        vector = so101_goal.goal_vector(torch.tensor([so101_goal.PHASES.index(goal_phase)], device='cuda'),
                                                        t(goal_pos), t(goal_rot), t(tcp_pos), t(tcp_rot))
                        null_mask = torch.tensor([args.goal_source == 'null'], device='cuda')
                    torch.manual_seed(args.noise_seed_base + seed * 1000 + chunks)
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                        if goal_model:
                            so101_goal.set_goal(policy, vector, null_mask)
                        try:
                            actions = post(policy.predict_action_chunk(pre(batch))).float().cpu().numpy()[0]
                        finally:
                            if goal_model:
                                so101_goal.set_goal(policy, None)
                    queue = list(actions[:args.execute]); chunks += 1
                row = queue.pop(0)
                if args.gif and steps % 10 == 0:
                    frames.append(Image.fromarray(np.concatenate([obs[c] for c in cfg['camera_map'].values()], axis=1)))
                target = np.clip(dataset_row_to_sim_qpos(row, gripper_limits_rad=limits), env.action_space.low, env.action_space.high)
                obs, _, _, _, info = env.step(target.astype('float32')); steps += 1
                grasp |= bool(info['is_grasped'] >= 0.5); min_tcp = min(min_tcp, float(info['tcp_to_obj_dist']))
                if info['success']:
                    success = True
                    break
            record = {'task': task, 'success': success, 'grasped': grasp, 'steps': steps, 'min_tcp_to_object_m': min_tcp,
                      'final_object_to_target_m': float(info['obj_to_target_dist']), **layout}
            if planner_log:
                record['planner_phase_agreement'] = float(np.mean([a for a, _ in planner_log]))
                record['planner_goal_error_mm_median'] = float(np.median([e for _, e in planner_log]) * 1000)
            tasks.append(record)
            if not success:
                break
        result = {'seed': seed, 'completed': sum(r['success'] for r in tasks), 'tasks': tasks,
                  'total_steps': sum(r['steps'] for r in tasks), 'wall_seconds': time.perf_counter() - started}
        report['episodes'].append(result)
        if frames:
            frames[0].save(args.output / f'seed{seed}.gif', save_all=True, append_images=frames[1:], duration=333, loop=0)
        print(json.dumps({k: v for k, v in result.items() if k != 'tasks'} | {'per_task': [(r['success'], r['steps']) for r in tasks]}), flush=True)
    eps = report['episodes']
    report['summary'] = {'episodes': len(eps), 'mean_completed': float(np.mean([e['completed'] for e in eps])),
                         'completed_distribution': {str(k): sum(e['completed'] == k for e in eps) for k in range(args.chain + 1)},
                         'task_success': [sum(len(e['tasks']) > k and e['tasks'][k]['success'] for e in eps) for k in range(args.chain)],
                         'task_attempted': [sum(len(e['tasks']) > k for e in eps) for k in range(args.chain)]}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary']), flush=True)
    env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--stats-dir', type=Path, help='directory with stats.json (default: the config output)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=['train', 'validation', 'test'], default='validation')
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--chain', type=int, default=3)
    p.add_argument('--execute', type=int, default=10)
    p.add_argument('--max-steps', type=int, default=900, help='per task')
    p.add_argument('--goal-source', choices=['none', 'planner', 'takeover', 'null'], default='none')
    p.add_argument('--noise-seed-base', type=int, default=3000000)
    p.add_argument('--gif', action='store_true')
    evaluate(p.parse_args())


if __name__ == '__main__':
    main()
