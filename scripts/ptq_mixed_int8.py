#!/usr/bin/env python3
"""Calibrate and pack the 40%-compression candidate; strict reload and action smoke.
This is GPU PTQ, not RKNN deployment or a completed quality evaluation.
"""
import argparse
import gc
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file
from qat_train_w8a8_stage1 import load_policy
from probe_action_sensitivity import selected_frames, run_action, sha256
from mixed_int8 import select, replace_from_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('model-dir','vlm-assets-dir','dataset-root','splits','partition','map','output-dir'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--calibration-tasks',type=int,default=40)
    p.add_argument('--smoke-tasks',type=int,default=2)
    args=p.parse_args()
    if not 1<=args.calibration_tasks<=40 or not 1<=args.smoke_tasks<=40:
        p.error('Task counts must be 1..40')
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    source=args.model_dir/'model.safetensors'
    part=json.loads(args.partition.read_text());cfg=json.loads(args.map.read_text())
    splits=json.loads(args.splits.read_text())['splits']
    if sha256(source)!=part['source_weight_sha256'] or sha256(args.splits)!=part['source_split_sha256'] or sha256(args.partition)!=cfg['evaluation_partition_sha256'] or sha256(source)!=cfg['checkpoint_weight_sha256']:
        raise ValueError('Frozen source/split/map mismatch')
    cal=part['calibration_episode_ids_from_ptq_calibration']
    dev=part['development_episode_ids_from_qat_train']
    if set(cal)&set(dev) or not set(cal)<=set(splits['ptq_calibration']) or not set(dev)<=set(splits['qat_train']):
        raise ValueError('Invalid calibration/development partition')
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    started=time.perf_counter()
    policy,pre,post=load_policy(args)
    selected,embedding=select(policy,cfg)
    stats={name:{'min':float('inf'),'max':float('-inf'),'calls':0,'input_dtypes':[], 'shapes':[]} for name in selected}
    handles=[]
    for name,mod in selected.items():
        def hook(module,inputs,name=name):
            x=inputs[0].detach();s=stats[name]
            lo,hi=float(x.min()),float(x.max())
            if not np.isfinite([lo,hi]).all(): raise ValueError(f'Nonfinite calibration: {name}')
            s['min']=min(s['min'],lo);s['max']=max(s['max'],hi);s['calls']+=1
            if str(x.dtype) not in s['input_dtypes']:s['input_dtypes'].append(str(x.dtype))
            if list(x.shape) not in s['shapes']:s['shapes'].append(list(x.shape))
        handles.append(mod.register_forward_pre_hook(hook))
    rows=selected_frames(args.dataset_root,cal,1,40)[:args.calibration_tasks]
    try:
        for i,row in enumerate(rows):
            run_action(policy,pre,post,row)
            print(f'calibration {i+1}/{len(rows)}',flush=True)
    finally:
        for h in handles:h.remove()
    if any(s['calls']==0 for s in stats.values()):raise ValueError('Unvisited quantized Linear')
    (out/'calibration.json').write_text(json.dumps({'episode_ids':cal[:args.calibration_tasks],'ranges':stats},indent=2)+'\n')
    manifest=[{'name':name,'kind':'linear','shape':list(m.weight.shape),'compute_dtype':str(m.weight.dtype).split('.')[-1]} for name,m in selected.items()]
    em=policy.get_submodule(embedding)
    manifest.append({'name':embedding,'kind':'embedding','shape':list(em.weight.shape),'compute_dtype':str(em.weight.dtype).split('.')[-1]})
    devrows=selected_frames(args.dataset_root,dev,1,40)[:args.smoke_tasks]
    baseline=np.stack([run_action(policy,pre,post,r) for r in devrows])
    del selected,em,policy;gc.collect();torch.cuda.empty_cache()
    excluded={s['name']+suffix for s in manifest for suffix in ('.weight','.bias')}
    with safe_open(str(source),framework='pt',device='cpu') as src:
        state={k:src.get_tensor(k).contiguous() for k in src.keys() if k not in excluded}
        for spec in manifest:
            name=spec['name'];w=src.get_tensor(name+'.weight').float()
            scale=w.abs().amax(1).clamp_min(1e-10)/127
            q=torch.round(w/scale[:,None]).clamp(-127,127).to(torch.int8)
            state[name+'.weight_scale']=scale
            if spec['kind']=='embedding':state[name+'.weight_q']=q.contiguous()
            else:
                if w.shape[0]%8 or w.shape[1]%8:raise ValueError(f'Unaligned GEMM weights: {name}')
                s=stats[name];lo=min(s['min'],0.);hi=max(s['max'],0.)
                a=max((hi-lo)/255,1e-12);z=int(np.clip(np.rint(-128-lo/a),-128,127))
                state[name+'.weight_q_t']=q.t().contiguous()
                state[name+'.weight_sum']=q.sum(1,dtype=torch.int32)
                state[name+'.activation_scale']=torch.tensor(a,dtype=torch.float32)
                state[name+'.activation_zero']=torch.tensor(z,dtype=torch.int32)
                if name+'.bias' in src.keys():state[name+'.bias']=src.get_tensor(name+'.bias').float()
    packed=out/'mixed_ptq.safetensors';save_file(state,str(packed));del state;gc.collect()
    report={'format':'mixed-int8-v1','scope':'GPU PTQ candidate; not RKNN; quality unverified',
            'source_weight_sha256':sha256(source),'source_weight_bytes':source.stat().st_size,
            'partition_sha256':sha256(args.partition),'map_sha256':sha256(args.map),
            'calibration_sha256':sha256(out/'calibration.json'),'selected_linears':len(manifest)-1,
            'manifest':manifest,'outputs':{'ptq':{'path':str(packed),'bytes':packed.stat().st_size,'sha256':sha256(packed)}},
            'reduction_fraction':1-packed.stat().st_size/source.stat().st_size,'reload_smoke':'pending',
            'environment':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name()},
            'script_sha256':sha256(Path(__file__))}
    report['meets_40_percent']=report['reduction_fraction']>=.4
    def write(): (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    write()
    policy,pre,post=load_policy(args)
    replace_from_manifest(policy,manifest)
    policy.load_state_dict(load_file(str(packed)),strict=True);policy.to('cuda').eval()
    quant=np.stack([run_action(policy,pre,post,r) for r in devrows])
    if not np.isfinite(quant).all():raise ValueError('Nonfinite PTQ actions')
    np.savez(out/'smoke_actions.npz',fp=baseline,ptq=quant)
    report.update(reload_smoke='passed',smoke_tasks=len(devrows),smoke_action_mae=float(np.abs(quant-baseline).mean()),elapsed_seconds=time.perf_counter()-started)
    write();print(json.dumps({k:v for k,v in report.items() if k!='manifest'}),flush=True)
    if not report['meets_40_percent']:raise RuntimeError('Packed checkpoint misses 40% compression gate')


if __name__=='__main__':main()
