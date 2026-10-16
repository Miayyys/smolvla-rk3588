"""Checks for the per-frame privileged labels of a scripted SO101 dataset (v2 generator).

1. Optional reference comparison: for seeds present in both datasets with the same
   noise level, success, frame count, states, actions, executed commands and phases
   must match (the label recording must not change the expert's behaviour).
2. FK consistency: TCP position recomputed from the stored dataset-unit state must
   match the recorded ``tcp_pose`` (validates the state -> TCP chain used for goal deltas).
3. Goal sanity: goal rotations orthonormal; pregrasp goals sit ``pregrasp_back`` behind
   the grasp point along the approach axis; per-phase goal-to-TCP distances.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

PRIVILEGED_KEYS = ('tcp_pose', 'object_pose', 'target_pos', 'goal_pos', 'goal_rot', 'attempt')


def compare_reference(data_dir, ref_dir, split):
    new, ref = np.load(data_dir / f'{split}.npz'), np.load(ref_dir / f'{split}.npz')
    man_new = {e['seed']: e for e in json.loads((data_dir / 'manifest.json').read_text())['episodes'] if e['split'] == split}
    man_ref = {e['seed']: e for e in json.loads((ref_dir / 'manifest.json').read_text())['episodes'] if e['split'] == split}
    rows = []
    for seed, e in sorted(man_new.items()):
        r = man_ref.get(seed)
        if r is None or r['noise_std_deg'] != e['noise_std_deg']:
            continue
        item = {'seed': seed, 'success_equal': e['success'] == r['success'], 'frames': [e['frames'], r['frames']]}
        if e['success'] and r['success']:
            a, b = np.flatnonzero(new['episodes'] == seed), np.flatnonzero(ref['episodes'] == seed)
            if len(a) == len(b):
                for k in ('states', 'actions', 'executed'):
                    item[f'{k}_max_abs_diff'] = float(np.abs(new[k][a] - ref[k][b]).max())
                item['phase_equal'] = bool((new['phase'][a] == ref['phase'][b]).all())
        rows.append(item)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--reference', type=Path, help='earlier dataset generated with the same seeds/noise levels')
    p.add_argument('--fk-frames', type=int, default=2000, help='random frames per split for the FK check')
    args = p.parse_args()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    from qvla.evaluation.so101_scripted_expert import NEXUS_SOURCE, ExpertParams, make_env
    sys.path.insert(0, NEXUS_SOURCE + '/src')
    import mujoco
    from so101_nexus import dataset_row_to_sim_qpos
    manifest = json.loads((args.data / 'manifest.json').read_text())
    phases = manifest['phases']
    report = {'reference': {}, 'fk': {}, 'goal': {}}
    if args.reference:
        for split in manifest['splits']:
            if (args.reference / f'{split}.npz').exists():
                rows = compare_reference(args.data, args.reference, split)
                report['reference'][split] = {
                    'seeds_compared': len(rows),
                    'all_success_equal': all(r['success_equal'] for r in rows),
                    'all_frames_equal': all(r['frames'][0] == r['frames'][1] for r in rows),
                    'all_phase_equal': all(r.get('phase_equal', True) for r in rows),
                    'max_abs_diff': {k: max((r.get(f'{k}_max_abs_diff', 0.0) for r in rows), default=None)
                                     for k in ('states', 'actions', 'executed')},
                    'rows': rows}
    env = make_env(image_size=tuple(manifest['config']['image_size']), cameras=False)
    u = env.unwrapped
    limits = (float(env.action_space.low[-1]), float(env.action_space.high[-1]))
    scratch = mujoco.MjData(u.model)
    back = ExpertParams().pregrasp_back
    rng = np.random.default_rng(0)
    for split in manifest['splits']:
        d = np.load(args.data / f'{split}.npz')
        missing = [k for k in PRIVILEGED_KEYS if k not in d.files]
        if missing:
            report['fk'][split] = {'missing_keys': missing}
            continue
        n = len(d['states'])
        idx = rng.choice(n, size=min(args.fk_frames, n), replace=False)
        err = []
        for i in idx:
            q = dataset_row_to_sim_qpos(d['states'][i], gripper_limits_rad=limits)
            scratch.qpos[:] = u.data.qpos
            scratch.qpos[u._qpos_addrs[:6]] = q[:6]
            mujoco.mj_kinematics(u.model, scratch)
            err.append(np.linalg.norm(scratch.site_xpos[u._tcp_site_id] - d['tcp_pose'][i, :3]))
        err = np.array(err) * 1000
        report['fk'][split] = {'frames': int(len(idx)), 'tcp_error_mm': {'mean': float(err.mean()), 'p99': float(np.percentile(err, 99)),
                                                                          'max': float(err.max())}}
        R = d['goal_rot'].reshape(-1, 3, 3).astype(np.float64)
        ortho = np.abs(np.einsum('nij,nkj->nik', R, R) - np.eye(3)).max()
        gap = np.linalg.norm(d['goal_pos'] - d['tcp_pose'][:, :3], axis=1) * 1000
        per_phase = {}
        for k, name in enumerate(phases):
            m = d['phase'] == k
            if m.any():
                per_phase[name] = {'frames': int(m.sum()), 'goal_to_tcp_mm_median': float(np.median(gap[m])),
                                   'goal_to_tcp_mm_p95': float(np.percentile(gap[m], 95))}
        # pregrasp goal = grasp point - approach * back; grasp point lies near the object.
        pre = d['phase'] == phases.index('pregrasp')
        approach = R[pre][:, :, 2]
        grasp_from_goal = d['goal_pos'][pre] + approach * back
        obj_gap = np.linalg.norm(grasp_from_goal - d['object_pose'][pre, :3], axis=1) * 1000
        report['goal'][split] = {'rotation_orthonormal_max_err': float(ortho), 'per_phase': per_phase,
                                 'pregrasp_grasp_point_to_object_mm': {'median': float(np.median(obj_gap)), 'max': float(obj_gap.max())},
                                 'attempts_seen': sorted(int(a) for a in np.unique(d['attempt']))}
    env.close()
    out = args.data / 'privileged_label_check.json'
    out.write_text(json.dumps(report, indent=2) + '\n')
    summary = {k: ({s: {kk: vv for kk, vv in v.items() if kk != 'rows'} for s, v in report[k].items()}) for k in report}
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
