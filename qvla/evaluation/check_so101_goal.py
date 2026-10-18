"""Pre-training checks for the goal-conditioned SO101 model (experiment B). No training, no test seeds.

1. Goal vectors from recorded labels: delta = goal - FK TCP, orthonormal relative rotation, phase/aperture table.
2. Evaluation goals equal training goals: the scripted expert drives validation seeds (noise-free, as recorded);
   the goal vector built the way ``so101_goal_eval`` builds it must match the vector from the recorded labels
   frame by frame.
3. Zero-init equivalence: with the goal modules attached, the flow loss (same noise/time) and the predicted
   action chunk (same seed) equal the unmodified SmolVLA's.
4. Saves the step-0 goal model (trainer layout) for a closed-loop pipeline smoke test.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def check_labels(data, so101_goal, torch):
    goal = {k: torch.from_numpy(data[k]) for k in ('phase', 'goal_pos', 'goal_rot', 'tcp_pose')}
    vector = so101_goal.goal_vector_from_labels(goal)
    R = so101_goal.matrix_from_6d(vector[:, 12:18])
    relative = so101_goal.quat_to_matrix(goal['tcp_pose'][:, 3:]).transpose(-1, -2) @ goal['goal_rot'].reshape(-1, 3, 3)
    delta = (data['goal_pos'] - data['tcp_pose'][:, :3])
    phase = data['phase']
    per_phase = {p: {'frames': int((phase == i).sum()),
                     'delta_mm_median': float(np.median(np.linalg.norm(delta[phase == i], axis=1)) * 1000),
                     'aperture': float(vector[phase == i, 18].mean() * 100)}
                 for i, p in enumerate(so101_goal.PHASES) if (phase == i).any()}
    return {'frames': len(phase), 'dim': int(vector.shape[1]), 'finite': bool(torch.isfinite(vector).all()),
            'one_hot_ok': bool((vector[:, :9].sum(1) == 1).all() and (vector[:, :9].argmax(1).numpy() == phase).all()),
            'delta_max_abs_error_m': float(np.abs(vector[:, 9:12].numpy() * so101_goal.POS_SCALE - delta).max()),
            'distance_max_abs_error_m': float(np.abs(vector[:, 19].numpy() * so101_goal.POS_SCALE - np.linalg.norm(delta, axis=1)).max()),
            'relative_rotation_6d_roundtrip_max_error': float((R - relative).abs().max()),
            'per_phase': per_phase}


def check_eval_goals(cfg, data, seeds, so101_goal, torch):
    import sys
    import mujoco
    from qvla.evaluation.so101_scripted_expert import NEXUS_SOURCE, Expert, make_env
    sys.path.insert(0, NEXUS_SOURCE + '/src')
    gen = json.loads((Path(cfg['dataset_root']) / 'manifest.json').read_text())['config']
    env = make_env(image_size=tuple(gen['image_size']), cameras=False, max_episode_steps=gen['max_steps'] + 10)
    u = env.unwrapped; scratch = mujoco.MjData(u.model)
    rows = []
    for seed in seeds:
        recorded = np.flatnonzero(data['episodes'] == seed)
        labels = so101_goal.goal_vector_from_labels({k: torch.from_numpy(data[k][recorded]) for k in ('phase', 'goal_pos', 'goal_rot', 'tcp_pose')})
        env.reset(seed=seed); expert = Expert(env); info = {'is_grasped': 0.0, 'success': False}
        live = []
        for _ in range(len(recorded)):
            label = expert.act(info)
            scratch.qpos[:] = u.data.qpos; mujoco.mj_kinematics(u.model, scratch)
            goal_pos, goal_rot, _ = expert.last_goal
            t = lambda a: torch.as_tensor(np.asarray(a, dtype='float32'))[None]
            live.append(so101_goal.goal_vector(torch.tensor([so101_goal.PHASES.index(expert.phase)]), t(goal_pos), t(goal_rot),
                                               t(scratch.site_xpos[u._tcp_site_id]), t(scratch.site_xmat[u._tcp_site_id].reshape(3, 3)))[0])
            # Validation seeds are noise-free, so the executed command is the clipped label, as in the generator.
            _, _, _, _, info = env.step(np.clip(label, u._target_low, u._target_high).astype('float32'))
        live = torch.stack(live)
        rows.append({'seed': seed, 'frames': len(recorded), 'phase_equal': bool((live[:, :9].argmax(1) == labels[:, :9].argmax(1)).all()),
                     'max_abs_diff': float((live - labels).abs().max())})
    env.close()
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('config/so101_goal_B.json'))
    p.add_argument('--output', type=Path, default=Path('runs/so101_goal_B_check'))
    p.add_argument('--eval-seeds', type=int, nargs='+', default=[100000, 100003, 100005],
                   help='stored (expert-successful) validation seeds; 100003 includes a second grasp attempt')
    p.add_argument('--rows', type=int, default=4)
    args = p.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import torch
    from lerobot.policies import make_pre_post_processors
    from safetensors.torch import save_file
    from qvla.evaluation import so101_goal
    from qvla.evaluation.finetune_so100_real import sha
    from qvla.evaluation.finetune_so101_scripted import ScriptedFrames
    from qvla.evaluation.haq_offline_eval import load_policy
    cfg = json.loads(args.config.read_text()); run = Path(cfg['output'])
    args.output.mkdir(parents=True, exist_ok=True)
    validation = dict(np.load(Path(cfg['dataset_root']) / 'validation.npz'))
    report = {'config': str(args.config), 'scope': 'pre-training checks; validation split only; no training'}
    report['labels_validation'] = check_labels(validation, so101_goal, torch)
    print(json.dumps({'labels_validation': {k: v for k, v in report['labels_validation'].items() if k != 'per_phase'}}), flush=True)
    report['eval_goal_vs_recorded'] = check_eval_goals(cfg, validation, args.eval_seeds, so101_goal, torch)
    print(json.dumps({'eval_goal_vs_recorded': report['eval_goal_vs_recorded']}), flush=True)

    # Model: same construction as the trainer.
    torch.manual_seed(cfg['seed'])
    policy, _, _ = load_policy(SimpleNamespace(model_dir=Path(cfg['model_dir']).resolve(),
        vlm_assets_dir=Path(cfg['vlm_assets_dir']).resolve(), device='cuda'))
    policy.float()
    policy.config.resize_imgs_with_padding = (cfg['image_size'], cfg['image_size'])
    policy.config.input_features = {k: v for k, v in policy.config.input_features.items() if k in ['observation.state', *cfg['camera_map']]}
    policy.config.empty_cameras = 0
    stats = json.loads((run / 'stats.json').read_text())
    pre, post = make_pre_post_processors(policy.config, cfg['model_dir'], preprocessor_overrides={
        'tokenizer_processor': {'tokenizer_name': str(Path(cfg['vlm_assets_dir']).resolve())},
        'device_processor': {'device': 'cuda'}, 'normalizer_processor': {'stats': stats}},
        postprocessor_overrides={'unnormalizer_processor': {'stats': stats}})
    frames = ScriptedFrames(cfg, run)
    rows = [int(r) for r in frames.data['validation_rows'][:args.rows]]
    raw = frames.batch(rows); goal = so101_goal.labels_to_device(raw.pop('goal'), 'cuda')
    single = frames.batch(rows[:1]); single_goal = so101_goal.labels_to_device(single.pop('goal'), 'cuda')
    policy.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        batch = pre(raw)
        generator = torch.Generator(device='cuda').manual_seed(0)
        noise = torch.randn((len(rows), policy.config.chunk_size, policy.config.max_action_dim), device='cuda', generator=generator)
        time = torch.rand(len(rows), device='cuda', generator=generator) * 0.999 + 0.001
        original_loss, _ = policy.forward(batch, noise=noise, time=time)
        policy.reset(); torch.manual_seed(1); original_chunk = post(policy.predict_action_chunk(pre(dict(single)))).float().cpu()
    so101_goal.attach(policy)
    policy.eval()
    goal_cfg = cfg['goal_conditioning']
    weights = torch.ones(len(so101_goal.PHASES), device='cuda')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        flow, planner, terms, joint_prefix = so101_goal.training_loss(policy, batch, goal, goal_cfg, weights, noise=noise,
                                                                      time=time, return_prefix=True)
        alone_prefix, alone_pad, _ = so101_goal.prefix_features(policy, batch)
        valid = alone_pad[:, :, None]
        so101_goal.set_goal(policy, so101_goal.goal_vector_from_labels(single_goal))
        policy.reset(); torch.manual_seed(1); goal_chunk = post(policy.predict_action_chunk(pre(dict(single)))).float().cpu()
        so101_goal.set_goal(policy, None)
    report['zero_init'] = {'rows': rows, 'original_flow_loss': float(original_loss), 'goal_flow_loss': float(flow),
                           'flow_loss_abs_diff': abs(float(original_loss) - float(flow)),
                           'chunk_max_abs_diff': float((original_chunk - goal_chunk).abs().max()),
                           # Planner input: joint forward (training) vs prefix-only forward (evaluation), valid tokens.
                           'prefix_features_max_abs_diff': float(((joint_prefix - alone_prefix).float() * valid).abs().max()),
                           'prefix_features_max_abs': float((alone_prefix.float() * valid).abs().max()),
                           'planner_loss_at_init': float(planner), 'planner_terms_at_init': terms,
                           'goal_embedding_parameters': sum(p.numel() for p in policy.model.goal_embedding.parameters()),
                           'planner_parameters': sum(p.numel() for p in policy.model.goal_planner.parameters())}
    print(json.dumps({'zero_init': report['zero_init']}), flush=True)

    folder = args.output / 'init_model'; folder.mkdir(exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in policy.state_dict().items()}, str(folder / 'model.safetensors'))
    policy.config.save_pretrained(folder); pre.save_pretrained(folder); post.save_pretrained(folder)
    report['init_model'] = {'path': str(folder), 'sha256': sha(folder / 'model.safetensors'),
                            'scope': 'step-0 goal model (equals SmolVLA base + zero-initialised goal modules); pipeline smoke only'}
    (args.output / 'check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'saved': report['init_model']}), flush=True)


if __name__ == '__main__':
    main()
