"""Validated offline teacher action labels and masked flow-matching distillation."""
import json
from pathlib import Path

import numpy as np
import torch

from qvla_haq.offline_actions import file_sha256


class TeacherCache:
    def __init__(self,path,*,partition_sha256,split_sha256,allowed_episodes,
                 allow_synthetic=False,review=None):
        self.root=Path(path)
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        m=self.manifest
        if m['partition_sha256']!=partition_sha256 or m['split_sha256']!=split_sha256:
            raise ValueError('Teacher cache belongs to another partition')
        synthetic=m.get('source_kind')=='synthetic_test_only'
        if synthetic and not allow_synthetic:raise ValueError('Synthetic labels prohibited in real training')
        if not synthetic and (m.get('source_kind')!='openvla_oft_real_inference' or not m.get('teacher_execution_verified')):
            raise ValueError('Teacher execution not verified')
        if m.get('action_space')!='libero_simulator_7d' or m.get('fps')!=10:
            raise ValueError('Teacher action convention or dataset timestep mismatch')
        if file_sha256(self.root/'actions.npy')!=m['actions_sha256']:raise ValueError('Teacher action hash changed')
        self.actions=np.load(self.root/'actions.npy',allow_pickle=False)
        self.rows=m['rows'];self.by_task={}
        if self.actions.ndim!=3 or self.actions.shape[0]!=len(self.rows) or self.actions.shape[-1]!=7:
            raise ValueError('Teacher action shape mismatch')
        if not np.isfinite(self.actions).all() or self.actions.shape[1]>50:raise ValueError('Invalid teacher action values')
        self.accepted_masks=np.ones(self.actions.shape[:2],dtype=bool)
        if review is not None:
            reviewed=json.loads(Path(review).read_text())
            if reviewed.get('teacher_actions_sha256')!=m['actions_sha256'] or reviewed.get('rows')!=self.rows:
                raise ValueError('Teacher review belongs to another label cache')
            raw=np.asarray(reviewed['accepted_timestep_mask'])
            if raw.dtype!=np.bool_ or raw.shape!=self.actions.shape[:2]:raise ValueError('Invalid teacher review mask')
            self.accepted_masks=raw
        seen=set()
        allowed=set(allowed_episodes)
        for i,row in enumerate(self.rows):
            identity=(int(row['episode_index']),int(row['frame_index']))
            if identity in seen or identity[0] not in allowed:raise ValueError('Repeated or excluded teacher episode/frame')
            if row.get('suite')!='libero_10':raise ValueError('LIBERO-10 teacher used outside its declared suite')
            if not 1<=row['valid_length']<=self.actions.shape[1]:raise ValueError('Invalid teacher action horizon')
            self.accepted_masks[i,row['valid_length']:]=False
            seen.add(identity)
            if self.accepted_masks[i].any():self.by_task.setdefault(int(row['task_index']),[]).append(i)

    def choose(self,task,rng):
        candidates=self.by_task.get(int(task))
        return None if not candidates else rng.choice(candidates)


def teacher_batch(observation,preprocessor,actions,valid_length,accepted_mask=None):
    """Normalize only after conversion to simulator action coordinates.

    Unsupervised suffix stays ground truth context; no repeated teacher chunk.
    Only matched, non-padded prefix timesteps contribute to teacher loss.
    """
    obs=dict(observation)
    target=observation['action'].clone()
    values=torch.as_tensor(actions,dtype=target.dtype,device=target.device)
    if target.ndim!=2 or target.shape[1]!=7 or values.ndim!=2 or values.shape[1]!=7:
        raise ValueError('Expected unnormalized action arrays [T,7]')
    length=min(int(valid_length),len(values),len(target))
    target[:length]=values[:length]
    padded=observation.get('action_is_pad',torch.zeros(len(target),dtype=torch.bool,device=target.device)).clone()
    padded[length:]=True
    if accepted_mask is not None:
        accepted=torch.as_tensor(accepted_mask,dtype=torch.bool,device=padded.device)
        if len(accepted)!=len(values):raise ValueError('Teacher review horizon mismatch')
        padded[:length]|=~accepted[:length]
        target[:length]=torch.where(accepted[:length,None],values[:length],observation['action'][:length])
    if bool(padded[:length].all()):raise ValueError('No matched teacher timestep')
    obs['action']=target;obs['action_is_pad']=padded
    batch=preprocessor(obs)
    if batch['action'].ndim==2:batch['action']=batch['action'].unsqueeze(0)
    if batch['action_is_pad'].ndim==1:batch['action_is_pad']=batch['action_is_pad'].unsqueeze(0)
    return batch


def combined_flow_loss(policy,batch,teacher=None,teacher_weight=0.2):
    if teacher_weight<0:raise ValueError('Negative teacher loss weight')
    padded=policy.prepare_action(batch).shape
    noise=policy.model.sample_noise(padded,batch['action'].device)
    time=policy.model.sample_time(padded[0],batch['action'].device)
    gt_loss,_=policy.forward(batch,noise=noise,time=time)
    kd_loss=None
    loss=gt_loss
    if teacher is not None and teacher_weight>0:
        kd_loss,_=policy.forward(teacher,noise=noise,time=time)
        loss=gt_loss+teacher_weight*kd_loss
    return loss,{'ground_truth_loss':float(gt_loss.detach()),
                 'teacher_loss':None if kd_loss is None else float(kd_loss.detach()),
                 'teacher_weight':teacher_weight if kd_loss is not None else 0.}
