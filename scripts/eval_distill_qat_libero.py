#!/usr/bin/env python3
"""Small closed-loop paired rollout with verified initial states and noise."""
import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.scripts import lerobot_eval
from lerobot.envs.libero import LiberoEnv

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qvla_haq.qat import prepare_mixed_qat
from qvla_haq.runtime import apply_assignment
from qvla_haq.offline_actions import file_sha256
from qvla_haq.module_recovery import GROUPS, recovery_group, recover_state
from qvla_haq.fp_checkpoint import reconstruct_fp_state


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--qvla-mode',choices=('original_fp','original_v2','distilled_fp','distilled_v2_qat','v2_ptq'),required=True)
    p.add_argument('--qvla-run',type=Path)
    p.add_argument('--qvla-candidate',type=Path,required=True)
    p.add_argument('--qvla-vlm-assets-dir',type=Path,required=True)
    p.add_argument('--qvla-reset-audit',type=Path,required=True)
    p.add_argument('--qvla-init-state',type=int,default=0)
    p.add_argument('--qvla-panel',choices=('five_case','long10','other9','custom'),default='five_case')
    p.add_argument('--qvla-task-ids',type=int,nargs='+')
    p.add_argument('--qvla-runtime-metrics',type=Path)
    p.add_argument('--qvla-restore-group', choices=GROUPS)
    p.add_argument('--qvla-recovery-audit', type=Path)
    p.add_argument('--qvla-action-trace', type=Path)
    args,rest=p.parse_known_args();sys.argv=[sys.argv[0],*rest]
    if args.qvla_panel=='custom' and (not args.qvla_task_ids or len(set(args.qvla_task_ids))!=len(args.qvla_task_ids) or any(t<0 or t>9 for t in args.qvla_task_ids)):
        p.error('Custom panel needs unique task IDs in 0..9')
    if args.qvla_restore_group and (args.qvla_mode!='distilled_fp' or not args.qvla_recovery_audit):
        p.error('Module recovery requires distilled_fp and a recovery audit path')
    args.qvla_reset_audit.parent.mkdir(parents=True,exist_ok=True)
    candidate=json.loads(args.qvla_candidate.read_text())
    artifact=args.qvla_candidate.parent.parent/candidate['artifact_directory']
    for name in ('space.json','calibration.json','assignment.json'):
        if file_sha256(artifact/name)!=candidate['files_sha256'][name]:raise ValueError('Frozen config changed')
    space=json.loads((artifact/'space.json').read_text());stats=json.loads((artifact/'calibration.json').read_text())['ranges'];assignment=candidate['assignment']
    report=None;path=None
    runtime=[]
    if args.qvla_run:
        report=json.loads((args.qvla_run/'report.json').read_text())
        if args.qvla_mode!='v2_ptq' and report['teacher_source_kind']!='openvla_oft_real_inference':raise ValueError('Real teacher labels required')
        if args.qvla_mode=='distilled_fp':path=args.qvla_run/'distilled_float_master.safetensors';expected=report['master_sha256']
        elif args.qvla_mode=='distilled_v2_qat':path=args.qvla_run/'qat_local_packed.safetensors';expected=report['packed_sha256']
        elif args.qvla_mode=='v2_ptq':
            path=args.qvla_run/'ptq_local_packed.safetensors';expected=report['packed_sha256']
            if report['candidate_config_sha256']!=file_sha256(args.qvla_candidate) or report['source_weight_sha256']!=candidate['source_checkpoint_sha256']:
                raise ValueError('PTQ source or map mismatch')
            if not report['strict_pack_reload'] or report['weight_training_performed']:raise ValueError('Invalid PTQ contract')
        else:raise ValueError('Unexpected run for original model')
        if file_sha256(path)!=expected:raise ValueError('Trained weight hash changed')
        if args.qvla_mode=='distilled_v2_qat' and (report['candidate_config_sha256']!=file_sha256(args.qvla_candidate) or
                report['source_weight_sha256']!=candidate['source_checkpoint_sha256']):
            raise ValueError('QAT result source or candidate map mismatch')
        if report.get('master_storage_format')=='trainable_fp_overlay' and report['source_weight_sha256']!=candidate['source_checkpoint_sha256']:
            raise ValueError('Overlay source checkpoint mismatch')
    elif args.qvla_mode.startswith('distilled') or args.qvla_mode=='v2_ptq':raise ValueError('Missing saved model run')
    if args.qvla_mode=='original_v2':
        path=artifact/'model.safetensors'
        if file_sha256(path)!=candidate['files_sha256']['model.safetensors']:
            raise ValueError('Frozen original PTQ pack changed')
    def make_policy(*,cfg,env_cfg,rename_map):
        if cfg.type!='smolvla' or file_sha256(Path(cfg.pretrained_path)/'model.safetensors')!=candidate['source_checkpoint_sha256']:raise ValueError('FP source changed')
        cfg.vlm_model_name=str(args.qvla_vlm_assets_dir.resolve());cfg.load_vlm_weights=False;cfg.device='cuda'
        policy=SmolVLAPolicy.from_pretrained(cfg.pretrained_path,config=cfg,strict=True)
        if args.qvla_mode=='distilled_fp':
            source={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()
                    if args.qvla_restore_group and recovery_group(k)==args.qvla_restore_group}
            if args.qvla_restore_group and not source:raise ValueError('Empty recovery group')
            names=prepare_mixed_qat(policy,space,assignment,stats);state=load_file(str(path))
            for name in names:
                policy.get_submodule(name).fake_quant_enabled=False
            state=reconstruct_fp_state(state,policy.state_dict(),set(names),report)
            if args.qvla_restore_group:
                recovered=recover_state(state, source, set(names))
                recovered.update(group=args.qvla_restore_group,
                    source_checkpoint_sha256=candidate['source_checkpoint_sha256'],
                    trained_master_sha256=expected, scope='FP_source_weight_intervention_not_quantization')
                args.qvla_recovery_audit.parent.mkdir(parents=True,exist_ok=True)
                args.qvla_recovery_audit.write_text(json.dumps(recovered,indent=2)+'\n')
            policy.load_state_dict(state,strict=True)
        elif args.qvla_mode in ('original_v2','distilled_v2_qat','v2_ptq'):
            originals={s['module']:copy.deepcopy(policy.get_submodule(s['module'])).cpu() for s in space['action_sites']}
            apply_assignment(policy,space,assignment,stats,originals)
            if path:
                policy.load_state_dict(load_file(str(path)),strict=True)
                for name in assignment:policy.get_submodule(name).refresh()
        policy=policy.to('cuda').eval()
        if args.qvla_runtime_metrics:
            original_chunk=policy._get_action_chunk
            def measured_chunk(*a,**kw):
                torch.cuda.synchronize()
                tick=time.perf_counter()
                value=original_chunk(*a,**kw)
                torch.cuda.synchronize()
                runtime.append({**context,'chunk_ms':(time.perf_counter()-tick)*1000})
                return value
            policy._get_action_chunk=measured_chunk
            torch.cuda.reset_peak_memory_stats()
        return policy
    context={};resets=[];original_reset=LiberoEnv.reset
    traces=[];original_step=LiberoEnv.step
    def traced_step(env,action):
        result=original_step(env,action)
        if args.qvla_action_trace and getattr(env,'_qvla_reset_count',0)==1:
            a=np.asarray(action,dtype=float).reshape(-1)
            if not np.isfinite(a).all():raise ValueError('Nonfinite rollout action')
            traces.append({**context,'step':getattr(env,'_qvla_trace_step',0),
                'action':a.tolist(),'reward':float(np.asarray(result[1]).item()),
                'terminated':bool(result[2]),'truncated':bool(result[3]) if len(result)==5 else False})
            env._qvla_trace_step=getattr(env,'_qvla_trace_step',0)+1
        return result
    def audited_reset(env,seed=None,**kw):
        env._ensure_env()
        count=getattr(env,'_qvla_reset_count',0)
        if count==0:env.init_state_id=args.qvla_init_state
        index=env.init_state_id%len(env._init_states)
        state=np.asarray(env._init_states[index])
        actual_seed=0 if args.qvla_panel in ('long10','other9') else seed
        observation,info=original_reset(env,seed=actual_seed,**kw)
        env._qvla_reset_count=count+1
        row={**context,'env_seed':actual_seed,'task_name':env.task,'reset_number':count+1,'actual_init_state_index':index,
             'init_state_sha256':hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest(),
             'initial_camera_sha256':{k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in observation['pixels'].items()}}
        resets.append(row);args.qvla_reset_audit.write_text(json.dumps(resets,indent=2)+'\n')
        return observation,info
    original_run_one=lerobot_eval.run_one
    def paired_run(*a,**kw):
        group=kw.get('task_group',a[0] if a else None);task=kw.get('task_id',a[1] if len(a)>1 else None)
        groups=('libero_spatial','libero_object','libero_goal','libero_10')
        seed=int(kw.get('start_seed') or 0)+100000*(groups.index(group)+1)+int(task)
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        context.clear();context.update(suite=group,task_id=int(task),policy_noise_seed=seed)
        return original_run_one(*a,**kw)
    original_processors=lerobot_eval.make_pre_post_processors
    def processors(*a,**kw):
        overrides=dict(kw.get('preprocessor_overrides') or {})
        overrides.update(tokenizer_processor={'tokenizer_name':str(args.qvla_vlm_assets_dir.resolve())},
            rename_observations_processor={'rename_map':{'observation.images.image':'observation.images.camera1','observation.images.image2':'observation.images.camera2'}})
        kw['preprocessor_overrides']=overrides;return original_processors(*a,**kw)
    original_all=lerobot_eval.eval_policy_all
    def small_panel(envs,*a,**kw):
        selected={}
        for suite,tasks in envs.items():
            selected[suite]={}
            for task,env in tasks.items():
                keep=(suite=='libero_10') if args.qvla_panel=='long10' else (int(task)==0 or suite=='libero_10' and int(task)==3)
                if args.qvla_panel=='other9':keep=suite in ('libero_spatial','libero_object','libero_goal') and int(task) in (0,4,8)
                if args.qvla_panel=='custom':keep=int(task) in args.qvla_task_ids
                if keep:selected[suite][task]=env
                else:env.close()
        kw.update(max_episodes_rendered=0,videos_dir=None,max_parallel_tasks=1)
        return original_all(selected,*a,**kw)
    LiberoEnv.reset=audited_reset;lerobot_eval.make_policy=make_policy;lerobot_eval.run_one=paired_run
    lerobot_eval.make_pre_post_processors=processors;lerobot_eval.eval_policy_all=small_panel
    if args.qvla_action_trace:LiberoEnv.step=traced_step
    lerobot_eval.main()
    if args.qvla_action_trace:
        args.qvla_action_trace.parent.mkdir(parents=True,exist_ok=True)
        args.qvla_action_trace.write_text(json.dumps(traces,indent=2)+'\n')
    if args.qvla_runtime_metrics:
        args.qvla_runtime_metrics.write_text(json.dumps(dict(
            scope='A10_local_runtime_observed_rollout_inputs_not_RK3588_or_fixed_input_benchmark',
            rows=runtime,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            timing='synchronized_full_action_chunk_excludes_environment_processors_and_cached_actions'),indent=2)+'\n')

if __name__=='__main__':main()
