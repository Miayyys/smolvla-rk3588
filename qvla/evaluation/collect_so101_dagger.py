"""DAgger-style data for SO101 with expert takeover labels (policy drives, scripted expert labels).

Stage ``rollout`` (GPU): the policy runs closed loop on training seeds exactly as in evaluation (chunks of
``--execute`` actions). The scripted expert runs alongside only to track the task phase. Every ``--label-every``
steps a sample is stored: both camera images (JPEG, as the v2 data), the joint state, the full MuJoCo integration
state, the phase-tracking expert's internal state and the grasp flag.

Stage ``label`` (CPU workers): each sample's simulator state is restored in a separate environment instance and a
scripted expert *takes over* from it for 50 steps without noise; its 50 commands are the sample's action chunk.
This is the chunk the expert itself would execute from that state, the same convention as the v2 labels (clean
expert targets along the expert's own trajectory). Per-step labels along the policy trajectory are not used: the
expert's moving reference ignores the robot, so they can sit tens of degrees from the measured state.

Takeover phase (the expert's state machine needs one): grasped cube -> keep the tracker's transport state if it is
lifting/carrying/lowering/opening, else start ``lift`` here; tracker in ``descend``/``close`` with the TCP within
2 cm of its grasp point -> continue it; tracker releasing/retreating -> continue; otherwise re-plan from
``pregrasp`` (fresh grasp frame from the current object pose). The reference point and previous command start at
the measured robot state, as at the start of every generated episode.

Output per shard: ``train.npz`` (states, action_chunks (N,50,6), chunk_pad (N,50), actions = first chunk action,
phase = tracker phase, takeover_phase, episodes, frame_index, end_rows, JPEG offsets/lengths), the JPEG streams and
``manifest.json``.
"""

import argparse
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import time

import numpy as np

from qvla.evaluation.finetune_so100_real import sha

SNAPSHOT_KEYS = ('phase', 'attempt', 'counter', 'gripper', 'ref', 'place_rot', 'lift_from', 'failed', 'carry_offset',
                 'retreat_point', 'grasp_rot', 'tilt', 'grasp_raise', 'open_point', 'prev_label', 'prev_tcp', 'stall',
                 'roll_hold', 'done')


def _snapshot(expert):
    return {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in expert.__dict__.items() if k in SNAPSHOT_KEYS}


def rollout(args):
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import sys
    from types import SimpleNamespace
    import mujoco
    import torch
    from PIL import Image
    from lerobot.policies import make_pre_post_processors
    from qvla.evaluation.generate_so101_scripted_data import PHASES
    from qvla.evaluation.haq_offline_eval import load_policy
    from qvla.evaluation.so101_scripted_expert import NEXUS_SOURCE, Expert, ExpertParams, make_env
    sys.path.insert(0, NEXUS_SOURCE + '/src')
    from so101_nexus import dataset_row_to_sim_qpos, sim_qpos_to_dataset_row
    cfg = json.loads(args.config.read_text())
    gen = json.loads((Path(cfg['dataset_root']) / 'manifest.json').read_text())['config']
    start, count = gen['splits']['train']
    seeds = args.seeds[args.shard::args.num_shards]
    if any(not start <= s < start + count for s in seeds):
        raise ValueError('DAgger collection only uses training-split seeds')
    stats = json.loads((Path(args.stats_dir) / 'stats.json').read_text())
    policy, _, _ = load_policy(SimpleNamespace(model_dir=args.model_dir.resolve(),
        vlm_assets_dir=Path(cfg['vlm_assets_dir']).resolve(), device='cuda'))
    policy.float().eval(); torch.set_num_threads(4)
    pre, post = make_pre_post_processors(policy.config, str(args.model_dir), preprocessor_overrides={
        'device_processor': {'device': 'cuda'}, 'tokenizer_processor': {'tokenizer_name': str(Path(cfg['vlm_assets_dir']).resolve())},
        'normalizer_processor': {'stats': stats}}, postprocessor_overrides={'unnormalizer_processor': {'stats': stats}})
    env = make_env(image_size=tuple(gen['image_size']), cameras=True, max_episode_steps=args.max_steps + 10)
    u = env.unwrapped
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state_size = mujoco.mj_stateSize(u.model, spec)
    limits = (float(env.action_space.low[-1]), float(env.action_space.high[-1]))
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    if (out / 'rollout.json').exists():
        raise ValueError('Shard already collected; preserve it')
    samples = {k: [] for k in ('states', 'phase', 'episodes', 'frame_index', 'is_grasped', 'sim_state')}
    snapshots = []
    streams = {cam: (out / f'train_{cam}.jpeg.bin').open('wb') for cam in gen['cameras']}
    offsets = {cam: [] for cam in gen['cameras']}; lengths = {cam: [] for cam in gen['cameras']}
    results = []; started_all = time.perf_counter()
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed); policy.reset()
            tracker = Expert(env, ExpertParams(max_attempts=args.tracker_max_attempts))
            info = {'is_grasped': 0.0, 'success': False}
            queue = []; chunks = steps = stored = 0; success = grasp = False; first_success = None; started = time.perf_counter()
            while steps < args.max_steps:
                before = _snapshot(tracker)  # the takeover re-runs act() on this state, so snapshot before it
                tracker.act(info)
                if steps % args.label_every == 0:
                    for cam in gen['cameras']:
                        b = io.BytesIO(); Image.fromarray(obs[cam]).save(b, format='JPEG', quality=gen['jpeg_quality'])
                        offsets[cam].append(streams[cam].tell()); lengths[cam].append(len(b.getvalue())); streams[cam].write(b.getvalue())
                    sim_state = np.empty(state_size); mujoco.mj_getState(u.model, u.data, sim_state, spec)
                    samples['states'].append(sim_qpos_to_dataset_row(obs['state'], gripper_limits_rad=limits).astype('float32'))
                    samples['phase'].append(PHASES.index(tracker.phase)); samples['episodes'].append(seed)
                    samples['frame_index'].append(steps); samples['is_grasped'].append(float(info['is_grasped']))
                    samples['sim_state'].append(sim_state); snapshots.append(before); stored += 1
                if not queue:
                    state = sim_qpos_to_dataset_row(obs['state'], gripper_limits_rad=limits).astype('float32')
                    batch = {key: torch.from_numpy(obs[c].copy()).permute(2, 0, 1).float() / 255 for key, c in cfg['camera_map'].items()}
                    batch.update({'observation.state': torch.from_numpy(state), 'task': cfg['task']})
                    torch.manual_seed(args.noise_seed_base + seed * 100 + chunks)
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                        actions = post(policy.predict_action_chunk(pre(batch))).float().cpu().numpy()[0]
                    queue = list(actions[:args.execute]); chunks += 1
                row = queue.pop(0)
                executed = np.clip(dataset_row_to_sim_qpos(row, gripper_limits_rad=limits), env.action_space.low, env.action_space.high)
                obs, _, _, truncated, info = env.step(executed.astype('float32')); steps += 1
                grasp |= bool(info['is_grasped'] >= 0.5)
                if info['success'] and first_success is None:
                    first_success = steps
                success |= bool(info['success'])
                # Keep a short tail after success (release/retreat states), then stop.
                if truncated or (first_success is not None and steps >= first_success + args.after_success_steps):
                    break
            result = {'seed': seed, 'steps': steps, 'samples': stored, 'policy_success': success, 'policy_grasped': grasp,
                      'first_success_step': first_success, 'tracker_phase': tracker.phase, 'tracker_attempts': tracker.attempt,
                      'seconds': time.perf_counter() - started}
            results.append(result); print(json.dumps(result), flush=True)
    finally:
        for s in streams.values():
            s.close()
    arrays = {k: np.asarray(v) for k, v in samples.items()}
    for cam in gen['cameras']:
        arrays[f'{cam}_offsets'] = np.asarray(offsets[cam], dtype='int64'); arrays[f'{cam}_lengths'] = np.asarray(lengths[cam], dtype='int64')
    np.savez(out / 'rollout.npz', **arrays)
    (out / 'tracker_snapshots.pkl').write_bytes(pickle.dumps(snapshots))
    meta = {'scope': 'DAgger rollout: policy-visited states (unlabelled); training seeds only', 'config': str(args.config),
            'model_dir': str(args.model_dir), 'model_sha256': sha(args.model_dir / 'model.safetensors'),
            'stats_sha256': sha(Path(args.stats_dir) / 'stats.json'), 'execute_per_chunk': args.execute, 'max_steps': args.max_steps,
            'label_every': args.label_every, 'after_success_steps': args.after_success_steps,
            'tracker_max_attempts': args.tracker_max_attempts, 'noise_seed_base': args.noise_seed_base,
            'shard': args.shard, 'num_shards': args.num_shards, 'seeds': seeds, 'samples': int(len(arrays['episodes'])),
            'mujoco_state_spec': 'mjSTATE_INTEGRATION', 'wall_seconds': time.perf_counter() - started_all, 'episodes': results}
    (out / 'rollout.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps({'stage': 'rolled_out', 'episodes': len(results), 'samples': meta['samples'],
                      'policy_success': sum(r['policy_success'] for r in results)}), flush=True)
    env.close()


def takeover_expert(env, snapshot, is_grasped, max_attempts):
    """A scripted expert continuing from the restored simulator state (see module docstring)."""
    from qvla.evaluation.so101_scripted_expert import GRIPPER_CLOSED, Expert, ExpertParams
    u = env.unwrapped
    expert = Expert(env, ExpertParams(max_attempts=max_attempts))  # reset(): fresh plan from the current state
    tcp = u._get_tcp_pose()[:3].copy(); q = u.data.qpos[u._arm_qpos_addrs].copy()
    phase = snapshot['phase']

    def adopt():
        for key in ('phase', 'attempt', 'counter', 'gripper', 'place_rot', 'lift_from', 'carry_offset', 'retreat_point',
                    'grasp_rot', 'tilt', 'grasp_raise', 'open_point'):
            if key in snapshot:
                setattr(expert, key, np.array(snapshot[key], copy=True) if isinstance(snapshot[key], np.ndarray) else snapshot[key])

    obj = u._get_object_pose()[:3]; target = u._get_target_pos()
    placed = (np.linalg.norm(obj[:2] - target[:2]) <= u.config.goal_thresh and obj[2] < u._initial_obj_z + 0.01)
    mode = 'replan'
    if is_grasped:
        if (phase in ('carry', 'lower') and snapshot.get('place_rot') is not None) or phase in ('close', 'lift', 'open'):
            adopt(); mode = 'continue'
        else:
            adopt(); expert.phase, expert.counter, expert.lift_from, expert.gripper = 'lift', 0, tcp.copy(), GRIPPER_CLOSED
            mode = 'lift_here'
    elif phase in ('open', 'retreat', 'hold'):
        adopt(); mode = 'continue'
    elif placed:
        # The policy released the cube on the disc before the tracker got there: back off, do not re-grasp.
        adopt(); expert.phase, expert.counter = 'retreat', 0
        rot = snapshot.get('place_rot') if snapshot.get('place_rot') is not None else expert.grasp_rot
        expert.place_rot = rot; expert.retreat_point = tcp - rot[:, 2] * 0.05 + np.array([0, 0, 0.03])
        mode = 'retreat_here'
    elif phase in ('descend', 'close'):
        # Descend runs from the pregrasp point (pregrasp_back behind the grasp point) along the approach axis.
        fresh_raise = expert.grasp_raise
        expert.grasp_raise = snapshot.get('grasp_raise', 0.0)
        approach = np.asarray(snapshot['grasp_rot'])[:, 2]
        offset = tcp - expert._grasp_point(obj, np.asarray(snapshot['grasp_rot']))
        back = -float(offset @ approach); lateral = float(np.linalg.norm(offset + back * approach))
        on_line = lateral <= 0.02 and -0.01 <= back <= (expert.params.pregrasp_back + 0.02 if phase == 'descend' else 0.02)
        if on_line:
            adopt(); mode = 'continue'
        else:
            expert.grasp_raise = fresh_raise
    if mode == 'replan':
        expert.attempt = min(int(snapshot.get('attempt', 1)), max_attempts - 1)
    # Keep the tracker's reference and last command when the robot is where they expect it (then the takeover is
    # exactly the original expert); otherwise start them at the measured robot, as at the start of an episode.
    ref, prev = np.asarray(snapshot['ref']), np.asarray(snapshot['prev_label'])
    keep = mode == 'continue' and np.linalg.norm(ref - tcp) <= 0.03 and np.abs(prev[:5] - q[:5]).max() <= np.radians(8)
    expert.ref = ref.copy() if keep else tcp
    expert.prev_label = prev.copy() if keep else q
    expert.prev_tcp = np.asarray(snapshot['prev_tcp']).copy() if keep else tcp.copy()
    expert.stall = snapshot['stall'] if keep else 0
    expert.roll_hold = snapshot['roll_hold'] if keep else None
    expert.done = False; expert.failed = False
    return expert, mode + ('' if keep or mode != 'continue' else '_anchored')


def _label_worker(job):
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import sys
    import mujoco
    from qvla.evaluation.generate_so101_scripted_data import PHASES
    from qvla.evaluation.so101_scripted_expert import NEXUS_SOURCE, make_env
    sys.path.insert(0, NEXUS_SOURCE + '/src')
    from so101_nexus import sim_qpos_to_dataset_row
    directory, indices, horizon, max_attempts = job
    data = np.load(directory / 'rollout.npz'); snapshots = pickle.loads((directory / 'tracker_snapshots.pkl').read_bytes())
    env = make_env(cameras=False, max_episode_steps=100000); u = env.unwrapped
    limits = (float(env.action_space.low[-1]), float(env.action_space.high[-1]))
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    out = []; current_seed = None
    for i in indices:
        seed = int(data['episodes'][i])
        if seed != current_seed:
            env.reset(seed=seed); current_seed = seed  # scene layout (target disc, initial object height) for this seed
        mujoco.mj_setState(u.model, u.data, data['sim_state'][i], spec); mujoco.mj_forward(u.model, u.data)
        info = {'is_grasped': float(data['is_grasped'][i]), 'success': False}
        expert, mode = takeover_expert(env, snapshots[i], info['is_grasped'] >= 0.5, max_attempts)
        start_phase = expert.phase
        chunk = np.zeros((horizon, 6), dtype='float32'); pad = np.ones(horizon, dtype=bool)
        for t in range(horizon):
            label = expert.act(info)
            chunk[t] = sim_qpos_to_dataset_row(label, gripper_limits_rad=limits); pad[t] = False
            _, _, _, _, info = env.step(np.clip(label, u._target_low, u._target_high).astype('float32'))
            if expert.done:
                break
        out.append((i, chunk, pad, PHASES.index(start_phase), mode))
    env.close()
    return out


def label(args):
    directory = args.output
    data = np.load(directory / 'rollout.npz'); meta = json.loads((directory / 'rollout.json').read_text())
    n = len(data['episodes'])
    # Contiguous blocks keep one environment reset per seed and worker.
    blocks = np.array_split(np.arange(n), max(1, args.workers * 4))
    jobs = [(directory, block.tolist(), args.horizon, args.takeover_max_attempts) for block in blocks if len(block)]
    started = time.perf_counter()
    with mp.get_context('spawn').Pool(args.workers) as pool:
        results = [r for part in pool.imap(_label_worker, jobs) for r in part]
    results.sort(key=lambda r: r[0])
    assert [r[0] for r in results] == list(range(n))
    chunks = np.stack([r[1] for r in results]); pads = np.stack([r[2] for r in results])
    arrays = {'states': data['states'], 'action_chunks': chunks, 'chunk_pad': pads, 'actions': chunks[:, 0],
              'phase': data['phase'], 'takeover_phase': np.asarray([r[3] for r in results]),
              'episodes': data['episodes'], 'frame_index': data['frame_index'], 'end_rows': np.arange(n),
              **{k: data[k] for k in data.files if k.endswith('_offsets') or k.endswith('_lengths')}}
    np.savez(directory / 'train.npz', **arrays)
    modes = {}
    for r in results:
        modes[r[4]] = modes.get(r[4], 0) + 1
    manifest = dict(meta, scope='DAgger round data: policy-visited states, action chunks from scripted-expert takeover; training seeds only',
                    horizon=args.horizon, takeover_max_attempts=args.takeover_max_attempts, takeover_modes=modes,
                    label_seconds=time.perf_counter() - started, label_workers=args.workers,
                    expert_source_sha256=sha('qvla/evaluation/so101_scripted_expert.py'),
                    collector_source_sha256=sha('qvla/evaluation/collect_so101_dagger.py'),
                    rollout_sha256=sha(directory / 'rollout.npz'), data_sha256=sha(directory / 'train.npz'))
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'stage': 'labelled', 'samples': n, 'modes': modes, 'padded_steps_mean': float(pads.sum(1).mean()),
                      'seconds': manifest['label_seconds']}), flush=True)


def _goal_worker(job):
    """Goal labels for the goal-conditioned model: phase and goal pose of the takeover expert at the sample's state
    (after its first act(), as the generator records them) and the TCP pose from forward kinematics."""
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import sys
    import mujoco
    from qvla.evaluation.generate_so101_scripted_data import PHASES
    from qvla.evaluation.so101_scripted_expert import NEXUS_SOURCE, make_env
    sys.path.insert(0, NEXUS_SOURCE + '/src')
    from so101_nexus import sim_qpos_to_dataset_row
    directory, indices, max_attempts = job
    data = np.load(directory / 'rollout.npz'); labelled = np.load(directory / 'train.npz')
    snapshots = pickle.loads((directory / 'tracker_snapshots.pkl').read_bytes())
    env = make_env(cameras=False, max_episode_steps=100000); u = env.unwrapped; scratch = mujoco.MjData(u.model)
    limits = (float(env.action_space.low[-1]), float(env.action_space.high[-1]))
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    out = []; current_seed = None
    for i in indices:
        seed = int(data['episodes'][i])
        if seed != current_seed:
            env.reset(seed=seed); current_seed = seed
        mujoco.mj_setState(u.model, u.data, data['sim_state'][i], spec); mujoco.mj_forward(u.model, u.data)
        scratch.qpos[:] = u.data.qpos; mujoco.mj_kinematics(u.model, scratch)
        quat = np.zeros(4); mujoco.mju_mat2Quat(quat, scratch.site_xmat[u._tcp_site_id])
        tcp = np.concatenate([scratch.site_xpos[u._tcp_site_id], quat]).astype('float32')
        info = {'is_grasped': float(data['is_grasped'][i]), 'success': False}
        expert, _ = takeover_expert(env, snapshots[i], info['is_grasped'] >= 0.5, max_attempts)
        first = sim_qpos_to_dataset_row(expert.act(info), gripper_limits_rad=limits).astype('float32')
        goal, rot, _ = expert.last_goal
        same = bool(np.array_equal(first, labelled['action_chunks'][i, 0]))
        out.append((i, PHASES.index(expert.phase), goal.astype('float32'), np.asarray(rot, dtype='float32').reshape(9), tcp, same))
    env.close()
    return out


def goal_labels(args):
    source, out = args.source, args.output
    labelled = np.load(source / 'train.npz'); n = len(labelled['episodes'])
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise ValueError('Goal labels already written; preserve them')
    blocks = np.array_split(np.arange(n), max(1, args.workers * 4))
    jobs = [(source, block.tolist(), args.takeover_max_attempts) for block in blocks if len(block)]
    started = time.perf_counter()
    with mp.get_context('spawn').Pool(args.workers) as pool:
        results = sorted((r for part in pool.imap(_goal_worker, jobs) for r in part), key=lambda r: r[0])
    assert [r[0] for r in results] == list(range(n))
    arrays = {k: labelled[k] for k in labelled.files}
    arrays['tracker_phase'] = labelled['phase']
    arrays['phase'] = np.asarray([r[1] for r in results])  # phase of the expert that produced the chunk
    arrays['goal_pos'] = np.stack([r[2] for r in results]); arrays['goal_rot'] = np.stack([r[3] for r in results])
    arrays['tcp_pose'] = np.stack([r[4] for r in results])
    first_action_equal = int(sum(r[5] for r in results))
    if first_action_equal != n:
        raise RuntimeError(f'Takeover first action differs from the stored chunk in {n - first_action_equal} samples')
    np.savez(out / 'train.npz', **arrays)
    for cam_file in source.glob('train_*.jpeg.bin'):
        (out / cam_file.name).symlink_to(cam_file.resolve())
    manifest = {'scope': 'goal labels (takeover expert phase/goal pose after its first act; FK TCP) for a labelled DAgger shard',
                'source': str(source), 'source_data_sha256': sha(source / 'train.npz'), 'samples': n,
                'first_action_equal_to_stored_chunk': first_action_equal, 'takeover_max_attempts': args.takeover_max_attempts,
                'seconds': time.perf_counter() - started, 'data_sha256': sha(out / 'train.npz')}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'stage': 'goal_labelled', **{k: manifest[k] for k in ('samples', 'first_action_equal_to_stored_chunk', 'seconds')}}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='stage', required=True)
    r = sub.add_parser('rollout')
    r.add_argument('--config', type=Path, required=True, help='training config of the driving policy (data root, cameras, task)')
    r.add_argument('--model-dir', type=Path, required=True)
    r.add_argument('--stats-dir', type=Path, required=True, help='directory with the stats.json the policy was trained with')
    r.add_argument('--output', type=Path, required=True)
    r.add_argument('--seeds', type=int, nargs='+', required=True)
    r.add_argument('--shard', type=int, default=0)
    r.add_argument('--num-shards', type=int, default=1)
    r.add_argument('--execute', type=int, default=10)
    r.add_argument('--max-steps', type=int, default=900)
    r.add_argument('--label-every', type=int, default=5)
    r.add_argument('--after-success-steps', type=int, default=30)
    r.add_argument('--tracker-max-attempts', type=int, default=10, help='phase tracker only; takeover decides labels')
    r.add_argument('--noise-seed-base', type=int, default=500000, help='flow-noise seeds, distinct from evaluation (900000)')
    lab = sub.add_parser('label')
    lab.add_argument('--output', type=Path, required=True, help='shard directory produced by rollout')
    lab.add_argument('--workers', type=int, default=8)
    lab.add_argument('--horizon', type=int, default=50)
    lab.add_argument('--takeover-max-attempts', type=int, default=5)
    gl = sub.add_parser('goal-labels')
    gl.add_argument('--source', type=Path, required=True, help='labelled shard directory')
    gl.add_argument('--output', type=Path, required=True)
    gl.add_argument('--workers', type=int, default=8)
    gl.add_argument('--takeover-max-attempts', type=int, default=5)
    args = p.parse_args()
    {'rollout': rollout, 'label': label, 'goal-labels': goal_labels}[args.stage](args)


if __name__ == '__main__':
    main()
