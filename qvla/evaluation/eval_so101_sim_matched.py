"""Evaluate an explicit checkpoint in held-out recorded SO101 initial scenes."""

import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--partitions', nargs='+', choices=['train','validation','test'], default=['validation','test'])
    p.add_argument('--source-episodes', nargs='+', type=int)
    p.add_argument('--execute-actions', nargs='+', type=int, default=[50])
    p.add_argument('--max-steps', type=int, default=1024)
    args=p.parse_args();cfg=json.loads(args.config.read_text());run=Path(cfg['output'])
    if args.max_steps <= 0 or any(n < 1 or n > 50 for n in args.execute_actions):
        p.error('max steps must be positive; execution horizons must be 1..50')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ.setdefault('MUJOCO_GL','egl')
    import torch
    from PIL import Image
    from lerobot.policies import make_pre_post_processors
    from qvla.evaluation.haq_offline_eval import load_policy
    from qvla.evaluation.prepare_so101_sim_training import make_scene
    from qvla.evaluation.finetune_so100_real import sha
    # make_scene imports the pinned simulator source before these helpers.
    import sys
    sys.path.insert(0,'/home/loser/Study/so101-nexus/src')
    from so101_nexus import sim_qpos_to_dataset_row,dataset_row_to_sim_qpos

    prepared=json.loads((run/'prepared.json').read_text())
    stats=json.loads((run/'stats.json').read_text())
    policy,_,_=load_policy(SimpleNamespace(model_dir=args.model_dir.resolve(),vlm_assets_dir=Path(cfg['vlm_assets_dir']).resolve(),device='cuda'))
    policy.float().eval();torch.set_num_threads(4)
    pre,post=make_pre_post_processors(policy.config,str(args.model_dir),preprocessor_overrides={
        'device_processor':{'device':'cuda'},'tokenizer_processor':{'tokenizer_name':str(Path(cfg['vlm_assets_dir']).resolve())},
        'normalizer_processor':{'stats':stats}},postprocessor_overrides={'unnormalizer_processor':{'stats':stats}})
    args.output.mkdir(parents=True,exist_ok=True)
    report={'scope':('training-source diagnostic, not held-out task performance' if 'train' in args.partitions else
                     'held-out recorded scene diagnostic, not real-world success'),
        'model_sha256':sha(args.model_dir/'model.safetensors'),'stats_sha256':sha(run/'stats.json'),
        'control_hz':30,'chunk_size':50,'execution_horizons':args.execute_actions,'num_flow_steps':10,
        'real_time_scheduled':False,'episodes':[]}
    for partition in args.partitions:
        episodes = prepared['training_successful_sources'] if partition == 'train' else prepared['splits'][partition]
        if args.source_episodes:
            episodes = [ep for ep in episodes if ep in args.source_episodes]
        for ep, execute_count in [(ep,n) for ep in episodes for n in args.execute_actions]:
            scene=next(x for x in prepared['scene_initializations'] if x['episode']==ep)
            env,obs=make_scene(cfg['dataset_root'],scene['environment_state_names'],scene['initial_environment_state'],cfg,'/home/loser/Study/so101-nexus')
            try:
                policy.reset();limits=(float(env.action_space.low[-1]),float(env.action_space.high[-1]))
                snapshots=[];trace=[];times=[];success=grasp=False;chunks=0;started=time.perf_counter()
                terminated=truncated=False
                while len(trace)<args.max_steps and not (terminated or truncated):
                    state=sim_qpos_to_dataset_row(obs['state'],gripper_limits_rad=limits).astype('float32')
                    batch={'observation.images.camera1':torch.from_numpy(obs['overhead_camera'].copy()).permute(2,0,1).float()/255,
                        'observation.state':torch.from_numpy(state),'task':cfg['task']}
                    torch.manual_seed(800000+ep*1000+chunks)
                    torch.cuda.synchronize();begin=time.perf_counter()
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        actions=post(policy.predict_action_chunk(pre(batch))).float().cpu().numpy()[0]
                    torch.cuda.synchronize();times.append(time.perf_counter()-begin);chunks+=1
                    assert actions.shape==(50,6) and np.isfinite(actions).all()
                    for row in actions[:execute_count]:
                        if len(trace)%10==0:snapshots.append(Image.fromarray(obs['overhead_camera']).resize((320,240)))
                        target=dataset_row_to_sim_qpos(row,gripper_limits_rad=limits)
                        before = obs['state'].copy()
                        applied = np.clip(target,env.action_space.low,env.action_space.high).astype('float32')
                        obs,_,terminated,truncated,info=env.step(applied)
                        success|=bool(info['success']);grasp|=bool(info['is_grasped'])
                        trace.append({'success':bool(info['success']),'grasp':bool(info['is_grasped']),
                            'tcp_to_object_m':float(info['tcp_to_obj_dist']),'object_to_goal_m':float(info['obj_to_target_dist']),
                            'state_before_rad':before.tolist(),'target_rad':applied.tolist(),
                            'state_after_rad':obs['state'].tolist()})
                        if terminated or truncated or len(trace)>=args.max_steps:break
                result={'env_id':f'MuJoCoPickAndPlace-v1/source{ep}','partition':partition,'source_episode':ep,
                    'execute_per_chunk':execute_count,
                    'success':success,'ever_grasped':grasp,'steps':len(trace),'chunks':chunks,
                    'wall_seconds':time.perf_counter()-started,'inference_seconds':times,
                    'min_tcp_to_object_m':min(x['tcp_to_object_m'] for x in trace),'last':trace[-1]}
                report['episodes'].append(result)
                (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
                suffix = f'_execute{execute_count}' if len(args.execute_actions)>1 else ''
                (args.output/f'episode{ep}{suffix}.json').write_text(json.dumps(trace)+'\n')
                snapshots[0].save(args.output/f'episode{ep}{suffix}.gif',save_all=True,append_images=snapshots[1:],duration=333,loop=0)
                print(json.dumps(result),flush=True)
            finally:env.close()


if __name__=='__main__':main()
