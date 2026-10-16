"""Oracle gripper intervention on TRAIN trajectories; not policy task scores.

Replay already saved model targets to its closest approach, then compare holding
the current gripper against slow closure while keeping identical arm targets.
The intervention uses simulator truth and must not be reported as learned grasp.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('config/so101_sim_finetune_matched_v1.json'))
    p.add_argument('--diagnostic',type=Path,default=Path('runs/so101_sim_training_diagnostic_v1'))
    args=p.parse_args();os.environ.setdefault('MUJOCO_GL','egl')
    from qvla.evaluation.prepare_so101_sim_training import make_scene
    cfg=json.loads(args.config.read_text());prep=json.loads((Path(cfg['output'])/'prepared.json').read_text())
    results=[]
    for ep in [3,4]:
        if ep not in prep['training_successful_sources']:raise ValueError('Only training sources allowed')
        trace=json.loads((args.diagnostic/f'episode{ep}_execute50.json').read_text())
        nearest=int(np.argmin([r['tcp_to_object_m'] for r in trace]))
        scene=next(s for s in prep['scene_initializations'] if s['episode']==ep)
        for intervention in ['hold','close']:
            env,obs=make_scene(cfg['dataset_root'],scene['environment_state_names'],scene['initial_environment_state'],cfg,'/home/loser/Study/so101-nexus')
            try:
                for row in trace[:nearest+1]:
                    obs,_,terminated,truncated,info=env.step(np.array(row['target_rad'],dtype='float32'))
                    if terminated or truncated:raise RuntimeError('Unexpected replay end')
                q=obs['state'].copy();start=q.copy()
                replay_error=float(np.max(np.abs(q-np.asarray(trace[nearest]['state_after_rad']))))
                initial_grasp=bool(info['is_grasped']);ever=initial_grasp
                for j in range(60):
                    target=start.copy()
                    if intervention=='close':
                        target[-1]=start[-1]+min(1.0,(j+1)/20)*(env.action_space.low[-1]-start[-1])
                    obs,_,terminated,truncated,info=env.step(np.clip(target,env.action_space.low,env.action_space.high).astype('float32'))
                    ever|=bool(info['is_grasped'])
                    if terminated or truncated:break
                record={'source_episode':ep,'intervention':intervention,'approach_step':nearest+1,
                    'closest_tcp_to_object_m':trace[nearest]['tcp_to_object_m'],
                    'replay_state_max_error_rad':replay_error,'initial_grasp':initial_grasp,
                    'ever_grasped_after_intervention':ever,'final_grasp':bool(info['is_grasped']),
                    'final_tcp_to_object_m':float(info['tcp_to_obj_dist'])}
                results.append(record);print(json.dumps(record),flush=True)
            finally:env.close()
    (args.diagnostic/'gripper_intervention.json').write_text(json.dumps({
        'scope':'oracle intervention on training trajectories, not model or task success',
        'results':results},indent=2)+'\n')


if __name__=='__main__':main()
