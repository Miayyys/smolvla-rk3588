"""Fixed development action chunks for periodic training diagnostics, not success."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import random
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from qvla.haq.offline_actions import action_seed,frame_positions,validate_partition
from qvla.haq.training_control import capture_random_state,restore_random_state


class DevelopmentChunks:
    def __init__(self,args,partition,split,train_episodes):
        episodes=validate_partition(partition,split)
        if set(episodes)&set(train_episodes):raise ValueError('Development/training overlap')
        data=LeRobotDataset('lerobot/libero',root=args.dataset_root,episodes=episodes,
            video_backend='pyav',return_uint8=True,delta_timestamps={'action':[i/10 for i in range(50)]})
        self.rows=[];self.observations=[];offset=0
        for episode in episodes:
            length=int(data.meta.episodes['length'][episode]);task=int(data.get_raw_item(offset)['task_index'])
            rank=frame_positions(length,task,1)[0];frame=data[offset+rank]
            if (int(frame['episode_index']),int(frame['frame_index']))!=(episode,rank):raise ValueError('Development frame changed')
            self.rows.append(dict(episode_index=episode,task_index=task,frame_index=rank,action_seed=action_seed(episode,task,rank)))
            self.observations.append({k:frame[k] for k in ('observation.images.image','observation.images.image2','observation.state','task','action','action_is_pad')})
            offset+=length
        if len(self.rows)!=40 or len({r['task_index'] for r in self.rows})!=40:raise ValueError('Expected one isolated observation per task')
        self.reference=None

    def evaluate(self,policy,pre,post):
        rng=random.Random(0);saved=capture_random_state(rng);training=policy.training;values=[]
        try:
            policy.eval()
            with torch.no_grad():
                for row,obs in zip(self.rows,self.observations):
                    batch=pre({k:v.float()/255 if k.startswith('observation.images.') else v
                               for k,v in obs.items() if k not in ('action','action_is_pad')})
                    generator=torch.Generator(device='cuda').manual_seed(row['action_seed'])
                    noise=torch.randn((1,policy.config.chunk_size,policy.config.max_action_dim),device='cuda',generator=generator)
                    pred=post(policy.predict_action_chunk(batch,noise=noise)).float().cpu().numpy()[0]
                    if not np.isfinite(pred).all():raise ValueError('Nonfinite development actions')
                    values.append(pred)
        finally:
            policy.train(training);restore_random_state(saved,rng)
        predicted=np.stack(values);targets=np.stack([o['action'].numpy() for o in self.observations])
        valid=np.stack([(~o['action_is_pad']).numpy() for o in self.observations])
        def per_row(errors):return (errors*valid[:,:,None]).sum((1,2))/(valid.sum(1)*errors.shape[2])
        errors=per_row(np.abs(predicted-targets));continuous=per_row(np.abs(predicted[:,:,:6]-targets[:,:,:6]))
        grips=per_row(((predicted[:,:,6]>0)!=(targets[:,:,6]>0)).astype(float)[:,:,None])
        tasks=np.array([r['task_index'] for r in self.rows]);reference=self.reference
        result={'scope':'full_valid_action_chunk_development_proxy_not_task_success_or_RKNN',
                'valid_chunk_mae':float(errors.mean()),'continuous6_mae':float(continuous.mean()),
                'gripper_sign_disagreement':float(grips.mean()),'long10_valid_chunk_mae':float(errors[tasks<10].mean()),
                'other30_valid_chunk_mae':float(errors[tasks>=10].mean()),'valid_timesteps':int(valid.sum()),
                'per_task_mae':errors.tolist(),'rows':self.rows}
        if reference is not None:result['mae_vs_training_start']=float(per_row(np.abs(predicted-reference)).mean())
        else:self.reference=predicted.copy()
        return result,predicted,targets,valid
