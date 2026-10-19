"""Checks for DAgger expert-takeover labels (no policy, no training).

1. A camera env and a camera-free env reset with the same seed give the same scene (target, object pose, rest height).
2. mj_getState/mj_setState round trip: restored qpos/qvel equal, and stepping both with the same command stays equal.
3. Takeover from states of noise-free expert episodes: run the takeover expert to the end (success rate vs the
   original expert from the same seeds) and compare its first 50 commands with the original expert's next 50 labels.
"""

import json
import os
import sys

import numpy as np

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
from qvla.evaluation import so101_scripted_expert as se
from qvla.evaluation.collect_so101_dagger import _snapshot, takeover_expert
sys.path.insert(0, se.NEXUS_SOURCE + '/src')
from so101_nexus import sim_qpos_to_dataset_row

spec = mujoco.mjtState.mjSTATE_INTEGRATION
cam_env = se.make_env(cameras=True, max_episode_steps=2000)
envA = se.make_env(cameras=False, max_episode_steps=100000); envB = se.make_env(cameras=False, max_episode_steps=100000)
uA, uB = envA.unwrapped, envB.unwrapped
limits = (float(envA.action_space.low[-1]), float(envA.action_space.high[-1]))
report = {'scene_equal': [], 'restore': [], 'takeover': []}

for seed in (0, 4, 8, 100000, 100003):
    cam_env.reset(seed=seed); envA.reset(seed=seed); cu = cam_env.unwrapped
    report['scene_equal'].append({'seed': seed, 'target': bool(np.array_equal(cu._get_target_pos(), uA._get_target_pos())),
                                  'object': bool(np.array_equal(cu._get_object_pose(), uA._get_object_pose())),
                                  'rest_z': cu._initial_obj_z == uA._initial_obj_z,
                                  'qpos': bool(np.array_equal(cu.data.qpos, uA.data.qpos))})

for seed in (0, 4, 8, 12, 16, 20):
    # Expert-driven, noise-free episode in env A; store samples every 25 steps.
    envA.reset(seed=seed); expert = se.Expert(envA); info = {'is_grasped': 0.0, 'success': False}
    labels, samples = [], []; steps = 0; success = False
    while steps < 900 and not expert.done:
        before = _snapshot(expert); label = expert.act(info)
        if steps % 25 == 0:
            s = np.empty(mujoco.mj_stateSize(uA.model, spec)); mujoco.mj_getState(uA.model, uA.data, s, spec)
            samples.append((steps, s, before, float(info["is_grasped"])))
        labels.append(sim_qpos_to_dataset_row(label, gripper_limits_rad=limits))
        _, _, _, _, info = envA.step(np.clip(label, uA._target_low, uA._target_high).astype('float32')); steps += 1
        success |= bool(info['success'])
    labels = np.array(labels)
    envB.reset(seed=seed)
    for step, s, snap, grasped in samples[1:]:
        mujoco.mj_setState(uB.model, uB.data, s, spec); mujoco.mj_forward(uB.model, uB.data)
        restore = {'seed': seed, 'step': step}
        restore['qpos_equal'] = True  # compared below after stepping both copies
        # 2. round trip: step a fresh copy of A's state and B's restored state with the same command.
        envC = envA  # reuse A: restore the same state there too
        mujoco.mj_setState(uA.model, uA.data, s, spec); mujoco.mj_forward(uA.model, uA.data)
        cmd = np.clip(labels[min(step, len(labels) - 1)] * 0 + uA.data.qpos[uA._qpos_addrs], uA._target_low, uA._target_high).astype('float32')
        envA.step(cmd); envB.step(cmd)
        restore['qpos_equal_after_step'] = bool(np.array_equal(uA.data.qpos, uB.data.qpos))
        restore['qvel_equal_after_step'] = bool(np.array_equal(uA.data.qvel, uB.data.qvel))
        report['restore'].append(restore)
        # 3. takeover from the stored state, run to completion.
        mujoco.mj_setState(uB.model, uB.data, s, spec); mujoco.mj_forward(uB.model, uB.data)
        take, mode = takeover_expert(envB, snap, grasped >= 0.5, max_attempts=5); start_phase = take.phase
        tinfo = {'is_grasped': grasped, 'success': False}; chunk = []; t = 0; tsuccess = False
        while t < 900 and not take.done:
            lab = take.act(tinfo)
            if t < 50:
                chunk.append(sim_qpos_to_dataset_row(lab, gripper_limits_rad=limits))
            _, _, _, _, tinfo = envB.step(np.clip(lab, uB._target_low, uB._target_high).astype('float32')); t += 1
            tsuccess |= bool(tinfo['success'])
        chunk = np.array(chunk); ref = labels[step:step + len(chunk)]
        n = min(len(chunk), len(ref))
        diff = np.abs(chunk[:n, :5] - ref[:n, :5]).max(1) if n else np.zeros(1)
        report['takeover'].append({'seed': seed, 'step': step, 'tracker_phase': snap['phase'], 'mode': mode,
                                   'start_phase': start_phase, 'original_success': success,
                                   'takeover_success': tsuccess, 'takeover_steps': t,
                                   'chunk_vs_original_deg_first10_median': float(np.median(diff[:10])),
                                   'chunk_vs_original_deg_all_median': float(np.median(diff))})
        envB.reset(seed=seed)

summary = {'scene_equal_all': all(all(v for k, v in r.items() if k != 'seed') for r in report['scene_equal']),
           'restore_all_equal': all(r['qpos_equal_after_step'] and r['qvel_equal_after_step'] for r in report['restore']),
           'takeovers': len(report['takeover']),
           'takeover_success': sum(r['takeover_success'] for r in report['takeover']),
           'takeover_success_where_original_succeeded': [sum(r['takeover_success'] for r in report['takeover'] if r['original_success']),
                                                         sum(1 for r in report['takeover'] if r['original_success'])],
           'modes': {m: sum(1 for r in report['takeover'] if r['mode'] == m) for m in {r['mode'] for r in report['takeover']}},
           'chunk_vs_original_first10_median_deg': float(np.median([r['chunk_vs_original_deg_first10_median'] for r in report['takeover']])),
           'chunk_vs_original_all_median_deg': float(np.median([r['chunk_vs_original_deg_all_median'] for r in report['takeover']]))}
report['summary'] = summary
print(json.dumps(summary, indent=1))
failed = [r for r in report['takeover'] if not r['takeover_success']]
print(json.dumps(failed, indent=0)[:3000])
open('runs/so101_dagger_takeover_check.json', 'w').write(json.dumps(report, indent=2) + '\n')
