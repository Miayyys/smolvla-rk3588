"""Training schedules, reproducible recovery and stratified observation sampling."""
import math
import os
import random
from pathlib import Path

import numpy as np
import torch


def learning_rate_at(step,total,peak,minimum,warmup,schedule):
    if not 0<=step<total or total<1 or not 0<minimum<=peak or not 0<=warmup<total:
        raise ValueError('Invalid learning-rate schedule')
    if schedule=='constant':return peak
    if schedule!='warmup_cosine':raise ValueError('Unknown learning-rate schedule')
    if warmup and step<warmup:
        return minimum+(peak-minimum)*(step+1)/warmup
    length=total-warmup
    progress=(step-warmup)/max(1,length-1)
    return minimum+.5*(peak-minimum)*(1+math.cos(math.pi*progress))


def capture_random_state(rng):
    return dict(sampler=rng.getstate(),python=random.getstate(),numpy=np.random.get_state(),
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_random_state(state,rng):
    rng.setstate(state['sampler']);random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path,model,optimizer,step,identity,rng,history):
    path=Path(path);temporary=path.with_suffix('.tmp');path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                 optimizer=optimizer.state_dict(),completed_steps=step,identity=identity,
                 random_state=capture_random_state(rng),history=history)
    torch.save(payload,temporary);os.replace(temporary,path)


def restore_checkpoint(path,model,optimizer,identity,rng):
    # Only use this loader for locally generated, hash-verified training checkpoints.
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload['identity']!=identity:raise ValueError('Resume source/data/map/training settings changed')
    model.load_state_dict(payload['model'],strict=True);optimizer.load_state_dict(payload['optimizer'])
    restore_random_state(payload['random_state'],rng)
    return payload['completed_steps'],payload['history']


def stratified_frames(episodes,count,rng):
    """Unique frames, balanced across episodes and temporal thirds; >=8 actions."""
    pools=[[],[],[]]
    for entry in episodes:
        start,length,episode,*_=entry
        for rank in range(max(0,length-7)):
            phase=min(2,rank*3//length)
            pools[phase].append((entry,rank))
    if sum(map(len,pools))<count:raise ValueError('Not enough unique valid teacher frames')
    # Shuffle episodes first, then interleave them within each temporal third.
    for phase in range(3):
        grouped={}
        for item in pools[phase]:grouped.setdefault(item[0][2],[]).append(item)
        groups=list(grouped.values());rng.shuffle(groups)
        for group in groups:rng.shuffle(group)
        ordered=[]
        while groups:
            next_groups=[]
            for group in groups:
                ordered.append(group.pop())
                if group:next_groups.append(group)
            groups=next_groups
        pools[phase]=ordered
    positions=[0,0,0];selected=[]
    while len(selected)<count:
        for phase in range(3):
            if positions[phase]<len(pools[phase]) and len(selected)<count:
                entry,rank=pools[phase][positions[phase]];positions[phase]+=1
                selected.append((entry,rank,phase))
    return selected
