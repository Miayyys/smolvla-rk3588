#!/usr/bin/env python3
"""Paired offline development actions for FP, original v2, distilled FP and QAT.

This runs observations, not LIBERO task episodes; it cannot measure success rate.
"""
import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qat_train_w8a8_stage1 import load_policy
from qvla_haq.qat import prepare_mixed_qat
from qvla_haq.runtime import apply_assignment
from qvla_haq.offline_actions import action_seed,file_sha256,frame_positions,validate_partition,score_actions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir','vlm-assets-dir','dataset-root','splits','partition','candidate','fp-run','qat-run','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(8)
    candidate=json.loads(args.candidate.read_text());partition=json.loads(args.partition.read_text());split=json.loads(args.splits.read_text())
    episodes=validate_partition(partition,split)
    if file_sha256(args.model_dir/'model.safetensors')!=candidate['source_checkpoint_sha256'] or file_sha256(args.splits)!=partition['source_split_sha256']:
        raise ValueError('Source identity changed')
    artifact=args.candidate.parent.parent/candidate['artifact_directory']
    for name in ('space.json','calibration.json','assignment.json'):
        if file_sha256(artifact/name)!=candidate['files_sha256'][name]:raise ValueError('Frozen map changed')
    space=json.loads((artifact/'space.json').read_text());assignment=candidate['assignment'];stats=json.loads((artifact/'calibration.json').read_text())['ranges']
    fp_report=json.loads((args.fp_run/'report.json').read_text());qat_report=json.loads((args.qat_run/'report.json').read_text())
    fp_path=args.fp_run/'distilled_float_master.safetensors';qat_path=args.qat_run/'qat_local_packed.safetensors'
    if file_sha256(fp_path)!=fp_report['master_sha256'] or file_sha256(qat_path)!=qat_report['packed_sha256']:raise ValueError('Trained weight hash changed')
    for report in (fp_report,qat_report):
        if report['partition_sha256']!=file_sha256(args.partition) or report['teacher_source_kind']!='openvla_oft_real_inference':raise ValueError('Not real teacher training on this partition')
        if set(episodes)&set(report['train_episode_ids']):raise ValueError('Evaluation episode used in training')
    data=LeRobotDataset('lerobot/libero',root=args.dataset_root,episodes=episodes,video_backend='pyav',return_uint8=True)
    rows=[];observations=[];offset=0
    for episode in episodes:
        length=int(data.meta.episodes['length'][episode]);first=data.get_raw_item(offset);task=int(first['task_index'])
        rank=frame_positions(length,task,1)[0];frame=data[offset+rank]
        if (int(frame['episode_index']),int(frame['frame_index']))!=(episode,rank):raise ValueError('Decoded frame mismatch')
        rows.append({'episode_index':episode,'task_index':task,'frame_index':rank,'action_seed':action_seed(episode,task,rank)})
        observations.append({key:frame[key] for key in ('observation.images.image','observation.images.image2','observation.state','task','action')})
        offset+=length
    if len(rows)!=40 or len({r['task_index'] for r in rows})!=40:raise ValueError('Expected forty isolated development tasks')
    args.output.mkdir(parents=True,exist_ok=False)
    actions={};timings={};started=time.perf_counter()
    for mode in ('original_fp','original_v2','distilled_fp','distilled_v2_qat'):
        policy,_,_=load_policy(args)
        pre,post=make_pre_post_processors(policy.config,str(args.model_dir),preprocessor_overrides={
            'tokenizer_processor':{'tokenizer_name':str(args.vlm_assets_dir.resolve())},
            'rename_observations_processor':{'rename_map':{'observation.images.image':'observation.images.camera1','observation.images.image2':'observation.images.camera2'}}})
        if mode=='distilled_fp':
            names=prepare_mixed_qat(policy,space,assignment,stats)
            state=load_file(str(fp_path))
            for name in names:
                policy.get_submodule(name).fake_quant_enabled=False
                state[name+'.master_weight']=state.pop(name+'.weight')
            policy.load_state_dict(state,strict=True)
            del state
        elif mode in ('original_v2','distilled_v2_qat'):
            originals={s['module']:copy.deepcopy(policy.get_submodule(s['module'])).cpu() for s in space['action_sites']}
            apply_assignment(policy,space,assignment,stats,originals)
            del originals
            if mode=='distilled_v2_qat':
                policy.load_state_dict(load_file(str(qat_path)),strict=True)
                for name in assignment:policy.get_submodule(name).refresh()
        policy.to('cuda').eval();values=[];times=[]
        for i,(obs,row) in enumerate(zip(observations,rows)):
            batch=pre({**{k:v for k,v in obs.items() if not k.startswith('observation.images.') and k!='action'},
                **{k:v.float()/255 for k,v in obs.items() if k.startswith('observation.images.')}})
            generator=torch.Generator(device='cuda').manual_seed(row['action_seed'])
            noise=torch.randn((1,policy.config.chunk_size,policy.config.max_action_dim),device='cuda',generator=generator)
            torch.cuda.synchronize();tick=time.perf_counter()
            with torch.inference_mode():value=post(policy.predict_action_chunk(batch,noise=noise)).float().cpu().numpy()[0]
            torch.cuda.synchronize();times.append(time.perf_counter()-tick)
            if not np.isfinite(value).all():raise ValueError('Nonfinite development action')
            values.append(value)
            if (i+1)%10==0:print(json.dumps({'mode':mode,'observation':i+1,'total':len(rows)}),flush=True)
        actions[mode]=np.stack(values);timings[mode]={'seconds':sum(times),'per_observation_seconds':times,'scope':'A10_inference_not_RK3588'}
        del policy,pre,post;gc.collect();torch.cuda.empty_cache()
    task_ids=np.asarray([r['task_index'] for r in rows]);recorded=np.stack([o['action'].numpy() for o in observations])
    metrics={mode:score_actions(value,actions['original_fp'],recorded,task_ids) for mode,value in actions.items()}
    for mode,value in actions.items():
        per=np.abs(value[:,0]-recorded).mean(1)
        metrics[mode]['teacher_suite_10_first_action_mae']=float(per[task_ids<10].mean())
        metrics[mode]['other30_first_action_mae']=float(per[task_ids>=10].mean())
    np.savez_compressed(args.output/'actions.npz',**actions,recorded_first_action=recorded)
    report={'scope':'forty_single_observation_development_proxies_not_closed_loop_success',
        'candidate_config_sha256':file_sha256(args.candidate),'partition_sha256':file_sha256(args.partition),
        'teacher_manifest_sha256':fp_report['teacher_manifest_sha256'],
        'fp_master_sha256':fp_report['master_sha256'],'qat_pack_sha256':qat_report['packed_sha256'],
        'rows':rows,'metrics':metrics,'timings':timings,'elapsed_seconds':time.perf_counter()-started,
        'actions_sha256':file_sha256(args.output/'actions.npz'),'closed_loop_quality':'not_measured',
        'teacher_suite_mapping':'dataset tasks0to9 verified against benchmark descriptions by teacher exporter'}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({mode:m['task_macro'] for mode,m in metrics.items()}),flush=True)

if __name__=='__main__':main()
