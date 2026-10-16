"""Generate a matched 30 Hz training cache by replaying simulator demonstrations.

Original native-50-Hz targets are interpolated, then a release tail is added.
Successful training trajectories alone enter normalization and the optimizer.
Original episode IDs determine disjoint splits before success filtering.
"""

import io
import json
import os
from pathlib import Path
import sys

import numpy as np


def make_scene(root, names, row, config, source):
    sys.path.insert(0, str(Path(source)/'src'))
    import gymnasium as gym
    import mujoco
    import so101_nexus.mujoco  # noqa: F401
    from so101_nexus import JointPositions, OverheadCamera, PickAndPlaceConfig, PhysicsConfig
    from so101_nexus.teleop.config_customization import apply_config_overrides, load_profile_overrides
    cfg = PickAndPlaceConfig(obs_mode='visual', observations=[JointPositions(), OverheadCamera()],
        physics=PhysicsConfig(timestep_s=1/300, control_period_s=1/30,
            integrator='implicitfast', noslip_iterations=3, tolerance=1e-8))
    cfg = apply_config_overrides(cfg, load_profile_overrides(Path(root)/'meta/so101_nexus_env.json','MuJoCoPickAndPlace-v1',cfg))
    env = gym.make('MuJoCoPickAndPlace-v1',config=cfg,control_mode='pd_joint_pos',max_episode_steps=1024)
    u = env.unwrapped
    def value(prefix,n): return np.asarray([row[names.index(f'{prefix}_{i}')] for i in range(n)])
    env.reset(seed=0,options={'init_qpos':value('joint_positions',6)})
    u.data.qpos[u._qpos_addrs] = value('joint_positions',6)
    u.data.qvel[u._qvel_addrs] = value('joint_velocities',6)
    slot = u._slots[u._target_slot_idx]
    u.data.qpos[slot.qpos_addr:slot.qpos_addr+7] = value('object_pose',7)
    u.data.qvel[slot.dof_addr:slot.dof_addr+6] = 0
    u.model.body_pos[u._target_body_id] = value('target_position',3)
    mujoco.mj_forward(u.model,u.data)
    u._refresh_reset_reference_state()
    return env,u._get_obs()


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('config/so101_sim_finetune_v1.json'))
    args=p.parse_args()
    os.environ.setdefault('MUJOCO_GL','egl')
    cfg=json.loads(args.config.read_text());out=Path(cfg['output']);out.mkdir(exist_ok=True,parents=True)
    sys.path.insert(0,'/home/loser/Study/so101-nexus/src')
    import pyarrow.parquet as pq
    from PIL import Image
    from so101_nexus import dataset_row_to_sim_qpos,sim_qpos_to_dataset_row
    from qvla.evaluation.finetune_so100_real import emit,sha
    root=Path(cfg['dataset_root']);info=json.loads((root/'meta/info.json').read_text())
    table=pq.read_table(root/'data/chunk-000/file-000.parquet')
    ids=np.asarray(table['episode_index']);original=np.asarray(table['action'].to_pylist(),dtype='float32')
    envstates=np.asarray(table['observation.environment_state'].to_pylist())
    names=info['features']['observation.environment_state']['names']
    # Source identity, not generated-frame identity, fixes train/val/test boundaries.
    order=np.random.default_rng(42).permutation(sorted(set(ids.tolist()))).tolist()
    splits={'train':order[:8],'validation':order[8:9],'test':order[9:]}
    Path(cfg['partition']).write_text(json.dumps({'splits':splits,'seed':42,'scope':'source-episode split before success filtering'},indent=2)+'\n')
    offsets=[];lengths=[];states=[];actions=[];eps=[];episode_reports=[]
    records=[]
    with (out/'top_frames.jpeg.bin').open('wb') as stream:
        for ep in sorted(set(ids.tolist())):
            rows=np.flatnonzero(ids==ep);sequence=original[rows]
            env,obs=make_scene(root,names,envstates[rows[0]],cfg,'/home/loser/Study/so101-nexus')
            start=len(eps);success=False
            try:
                limits=(float(env.action_space.low[-1]),float(env.action_space.high[-1]))
                # One original action advanced the native simulator by .02 s;
                # the original MP4 fps=30 was a recording clock, not this clock.
                duration=(len(sequence)-1)*.02
                slow_retime = cfg.get('replay_retiming') == 'one_source_action_per_30Hz_step'
                play_steps = len(sequence)-1 if slow_retime else int(np.ceil(duration*30))
                tail_start=float(sequence[-1,-1]);tail_open=float(original[:,-1].max())
                for step in range(play_steps+90):
                    if step<play_steps:
                        index = float(step) if slow_retime else min(step/30/.02,len(sequence)-1)
                        left=int(np.floor(index));right=min(left+1,len(sequence)-1)
                        target_row=sequence[left]*(1-(index-left))+sequence[right]*(index-left)
                    else:
                        target_row=sequence[-1].copy()
                        fraction=min(1.0,(step-play_steps+1)/12)
                        target_row[-1]=tail_start+fraction*(tail_open-tail_start)
                    target=dataset_row_to_sim_qpos(target_row,gripper_limits_rad=limits)
                    command=np.clip(target,env.action_space.low,env.action_space.high).astype('float32')
                    # Save the actual executed command after bounds handling.
                    actions.append(sim_qpos_to_dataset_row(command,gripper_limits_rad=limits).astype('float32'))
                    states.append(sim_qpos_to_dataset_row(obs['state'],gripper_limits_rad=limits).astype('float32'))
                    eps.append(int(ep))
                    b=io.BytesIO();Image.fromarray(obs['overhead_camera']).save(b,format='JPEG',quality=95,subsampling=0)
                    encoded=b.getvalue();offsets.append(stream.tell());lengths.append(len(encoded));stream.write(encoded)
                    obs,_,terminated,truncated,result=env.step(command)
                    success|=bool(result['success'])
                    if terminated or truncated:break
                episode_reports.append({'episode':int(ep),'frames':len(eps)-start,'success':success,
                    'ever_grasped_final':bool(result['is_grasped']),'last_object_to_goal_m':float(result['obj_to_target_dist'])})
                records.append({'episode':int(ep),'initial_environment_state':envstates[rows[0]].tolist(),
                    'environment_state_names':names})
                emit(out,{'stage':'generated_episode',**episode_reports[-1]})
            finally:env.close()
    states=np.asarray(states);actions=np.asarray(actions);eps=np.asarray(eps)
    passed={e['episode'] for e in episode_reports if e['success']}
    training_eps=set(splits['train'])&passed
    if len(training_eps)<4:raise RuntimeError('Insufficient successful training sources; do not start training')
    train_rows=np.flatnonzero(np.isin(eps,list(training_eps)))
    end_rows=np.empty(len(eps),dtype='int64');validation=[]
    for ep in sorted(set(eps.tolist())):
        rows=np.flatnonzero(eps==ep);end_rows[rows]=rows[-1]
        if ep in splits['validation']:
            if ep not in passed:raise RuntimeError('Fixed validation source failed generation; do not select a replacement')
            validation.extend(np.linspace(rows[0],max(rows[0],rows[-1]-49),cfg['validation_frames_per_episode'],dtype=int).tolist())
    stats={k:{'mean':v[train_rows].mean(0).tolist(),'std':np.maximum(v[train_rows].std(0),1e-6).tolist()}
        for k,v in [('observation.state',states),('action',actions)]}
    np.savez(out/'data.npz',states=states,actions=actions,episodes=eps,end_rows=end_rows,
        train_rows=train_rows,validation_rows=validation,offsets=offsets,lengths=lengths)
    (out/'stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    report={'config':cfg,'splits':splits,'train_frames':len(train_rows),'training_successful_sources':sorted(training_eps),
        'failed_source_episodes':[e['episode'] for e in episode_reports if not e['success']],
        'generation':episode_reports,'scene_initializations':records,'control_dt':1/30,'physics_timestep':1/300,
        'source_native_control_dt':.02,'source_dataset_table_sha256':sha(root/'data/chunk-000/file-000.parquet'),
        'partition_sha256':sha(cfg['partition']),'stats_sha256':sha(out/'stats.json'),
        'cache_sha256':sha(out/'top_frames.jpeg.bin'),'cache_bytes':(out/'top_frames.jpeg.bin').stat().st_size,
        'scope':'new 30Hz simulated trajectories with explicit release tail; not original unchanged dataset'}
    (out/'prepared.json').write_text(json.dumps(report,indent=2)+'\n')
    emit(out,{'stage':'prepared','successes':len(passed),'episodes':len(episode_reports),'train_frames':len(train_rows),'splits':splits})


if __name__=='__main__':main()
