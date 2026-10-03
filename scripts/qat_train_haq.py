#!/usr/bin/env python3
"""Full frozen HAQ-map QAT diagnostic; save FP masters and a real local pack.

This is not an RKNN conversion or a completed hardware-aware QAT deployment.
Supports isolated teacher caches for FP distillation and mixed-map QAT.
Synthetic labels require an explicit two-step-only plumbing smoke flag.
"""
import argparse
import copy
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file,save_file

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qvla_haq.distillation import TeacherCache, teacher_batch, combined_flow_loss
from qvla_haq.training_control import learning_rate_at,save_checkpoint,restore_checkpoint
from qvla_haq.training_development import DevelopmentChunks
from qvla_haq.training_scope import apply_training_scope, frozen_parameter_digest
from qvla_haq.fp_checkpoint import reconstruct_fp_state, source_key
from qvla_haq.qat import prepare_mixed_qat
from qvla_haq.runtime import apply_assignment,ConfiguredOperator
from qat_train_w8a8_stage1 import load_policy,training_data
from probe_action_sensitivity import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir','vlm-assets-dir','dataset-root','splits','partition','candidate','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mode',choices=('qat','fp-distill'),default='qat')
    p.add_argument('--teacher-cache',type=Path)
    p.add_argument('--teacher-weight',type=float,default=0.2)
    p.add_argument('--initial-master',type=Path)
    p.add_argument('--allow-synthetic-teacher-smoke',action='store_true')
    p.add_argument('--steps',type=int,default=2)
    p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--seed',type=int,default=29)
    p.add_argument('--lr-schedule',choices=('constant','warmup_cosine'),default='constant')
    p.add_argument('--minimum-learning-rate',type=float,default=1e-7)
    p.add_argument('--warmup-ratio',type=float,default=0.05)
    p.add_argument('--gradient-accumulation',type=int,default=1)
    p.add_argument('--eval-every',type=int,default=0)
    p.add_argument('--save-every',type=int,default=0)
    p.add_argument('--keep-snapshots',type=int,default=2)
    p.add_argument('--resume',type=Path)
    p.add_argument('--teacher-review',type=Path)
    p.add_argument('--teacher-task-fraction',type=float)
    p.add_argument('--require-expanded-teacher',action='store_true')
    p.add_argument('--train-scope', choices=('all_sites','expert_first8','expert_and_interface'), default='all_sites')
    p.add_argument('--save-trainable-only',action='store_true')
    p.add_argument('--allow-unverified-backend-diagnostic',action='store_true')
    p.add_argument('--operator-audit',action='store_true')
    args=p.parse_args()
    if args.mode=='qat' and not args.allow_unverified_backend_diagnostic:
        p.error('Mixed RKNN equivalence is pending; explicitly opt into a local diagnostic')
    if args.steps<1 or args.learning_rate<=0:p.error('Invalid training budget')
    if args.save_trainable_only and (args.mode!='fp-distill' or args.save_every):
        p.error('FP overlays are only supported for FP diagnostics without periodic full snapshots')
    if args.mode=='fp-distill' and args.teacher_cache is None:p.error('FP distillation requires teacher labels')
    if args.teacher_weight<0:p.error('Teacher weight must be nonnegative; zero is a GT-only control with identical cached-frame sampling')
    if args.allow_synthetic_teacher_smoke and args.steps>2:p.error('Synthetic plumbing smoke is limited to two steps')
    if args.gradient_accumulation<1 or min(args.eval_every,args.save_every)<0 or args.keep_snapshots<1:p.error('Invalid training intervals')
    if not 0<=args.warmup_ratio<1 or not 0<args.minimum_learning_rate<=args.learning_rate:p.error('Invalid learning-rate range')
    if args.resume and args.initial_master and args.resume.resolve()==args.initial_master.resolve():p.error('Resume is training state, not a float master')
    if args.teacher_task_fraction is not None and not 0<args.teacher_task_fraction<1:p.error('Teacher task fraction must be between zero and one')
    if args.require_expanded_teacher and not args.teacher_review:p.error('Expanded training requires an explicit teacher label review')
    args.output_dir.mkdir(parents=True,exist_ok=bool(args.resume))
    torch.set_num_threads(8);random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    started=time.perf_counter()
    candidate=json.loads(args.candidate.read_text())
    split_doc=json.loads(args.splits.read_text());partition=json.loads(args.partition.read_text())
    if (sha256(args.model_dir/'model.safetensors')!=candidate['source_checkpoint_sha256'] or
        partition['source_weight_sha256']!=candidate['source_checkpoint_sha256'] or
        sha256(args.splits)!=partition['source_split_sha256']):raise ValueError('Source/split identity mismatch')
    asset=args.candidate.parent.parent/candidate['artifact_directory']
    for name in ('space.json','calibration.json','assignment.json'):
        if sha256(asset/name)!=candidate['files_sha256'][name]:raise ValueError('Frozen assignment changed')
    space=json.loads((asset/'space.json').read_text())
    assignment=json.loads((asset/'assignment.json').read_text())
    if assignment!=candidate['assignment']:raise ValueError('Candidate config differs from frozen map')
    cal=json.loads((asset/'calibration.json').read_text());stats=cal['ranges']
    if set(cal['episode_ids'])!=set(partition['calibration_episode_ids_from_ptq_calibration']):raise ValueError('Calibration partition mismatch')
    policy,pre,post=load_policy(args)
    # Match the established inference path and the two raw camera fields.
    from lerobot.policies import make_pre_post_processors
    pre,post=make_pre_post_processors(policy.config,str(args.model_dir),preprocessor_overrides={
        'tokenizer_processor':{'tokenizer_name':str(args.vlm_assets_dir.resolve())},
        'rename_observations_processor':{'rename_map':{
            'observation.images.image':'observation.images.camera1',
            'observation.images.image2':'observation.images.camera2'}}})
    originals={s['module']:copy.deepcopy(policy.get_submodule(s['module'])).cpu() for s in space['action_sites']}
    original_dtypes={name:module.weight.dtype for name,module in originals.items()}
    names=prepare_mixed_qat(policy,space,assignment,stats)
    if args.initial_master:
        state=load_file(str(args.initial_master))
        initial_report_path=args.initial_master.parent/'report.json'
        initial_report=json.loads(initial_report_path.read_text()) if initial_report_path.exists() else {}
        if initial_report.get('master_storage_format')=='trainable_fp_overlay':
            if initial_report['source_weight_sha256']!=candidate['source_checkpoint_sha256'] or sha256(args.initial_master)!=initial_report['master_sha256']:
                raise ValueError('Initial overlay source or weight hash mismatch')
        state=reconstruct_fp_state(state,policy.state_dict(),set(names),initial_report)
        policy.load_state_dict(state,strict=True)
    if args.mode=='fp-distill':
        for name in names:policy.get_submodule(name).fake_quant_enabled=False
    scope_audit=apply_training_scope(policy,args.train_scope)
    frozen_initial_digest=frozen_parameter_digest(policy)
    scope_audit['frozen_initial_sha256']=frozen_initial_digest
    (args.output_dir/'training_scope.json').write_text(json.dumps(scope_audit,indent=2)+'\n')
    print(json.dumps({'training_scope':scope_audit['scope'],'trainable_parameter_elements':scope_audit['trainable_parameter_elements'],
                      'frozen_parameter_elements':scope_audit['frozen_parameter_elements']}),flush=True)
    policy.to('cuda')
    trainable=[v for v in policy.parameters() if v.requires_grad]
    if args.save_every:
        master_bytes=sum(v.numel()*v.element_size() for v in policy.state_dict().values())
        optimizer_bytes=2*sum(v.numel()*v.element_size() for v in trainable)
        # Rolling state + temporary replacement, retained masters and one new snapshot.
        required=2*(master_bytes+optimizer_bytes)+(args.keep_snapshots+2)*master_bytes
        free=shutil.disk_usage(args.output_dir).free
        if free<required:raise ValueError(f'Checkpoint space insufficient: need about {required} bytes free, have {free}')
    development=set(partition['development_episode_ids_from_qat_train'])
    dataset,by_task,episodes=training_data(args,split_doc['splits'],development)
    if set(episodes)&(set(split_doc['splits']['test'])|set(cal['episode_ids'])|development):raise ValueError('Training/evaluation overlap')
    teacher_cache=None
    if args.teacher_cache:
        teacher_cache=TeacherCache(args.teacher_cache,partition_sha256=sha256(args.partition),
            split_sha256=sha256(args.splits),allowed_episodes=episodes,
            allow_synthetic=args.allow_synthetic_teacher_smoke,review=args.teacher_review)
    teacher_tasks=sorted(teacher_cache.by_task) if teacher_cache is not None else []
    other_tasks=sorted(set(by_task)-set(teacher_tasks))
    if args.teacher_task_fraction is not None and (not teacher_tasks or not other_tasks):raise ValueError('Need teacher and GT-only task pools')
    if args.require_expanded_teacher and (len(teacher_tasks)!=10 or any(len(teacher_cache.by_task[t])<64 for t in teacher_tasks)):
        raise ValueError('Expanded training needs at least 64 accepted observations for each of ten long tasks')
    episode_offsets={episode:(start,length,task) for task,entries in by_task.items() for start,length,episode in entries}
    optimizer=torch.optim.AdamW(trainable,lr=args.learning_rate,weight_decay=0)
    rng=random.Random(args.seed);losses=[];gradient_by_stage={};sample_rows=[]
    stage_names={'vision':'.vision_model.','connector':'.connector.','prefix':'.text_model.',
                 'expert':'.lm_expert.','embedding':'.embed_tokens'}
    warmup=min(args.steps-1,int(args.steps*args.warmup_ratio))
    identity={'mode':args.mode,'steps':args.steps,'seed':args.seed,'learning_rate':args.learning_rate,
        'lr_schedule':args.lr_schedule,'minimum_learning_rate':args.minimum_learning_rate,'warmup_steps':warmup,
        'gradient_accumulation':args.gradient_accumulation,'teacher_weight':args.teacher_weight,'teacher_task_fraction':args.teacher_task_fraction,
        'source':candidate['source_checkpoint_sha256'],'candidate':sha256(args.candidate),
        'partition':sha256(args.partition),'splits':sha256(args.splits),
        'teacher_manifest':None if teacher_cache is None else sha256(args.teacher_cache/'manifest.json'),
        'teacher_review':None if args.teacher_review is None else sha256(args.teacher_review),
        'initial_master':None if args.initial_master is None else sha256(args.initial_master),
        'training_script':sha256(Path(__file__)),'training_control':sha256(ROOT/'qvla_haq/training_control.py'),
        'training_scope':scope_audit,'training_scope_implementation':sha256(ROOT/'qvla_haq/training_scope.py'),
        'master_storage_format':'trainable_fp_overlay' if args.save_trainable_only else 'full_fp_master',
        'fp_checkpoint_implementation':sha256(ROOT/'qvla_haq/fp_checkpoint.py'),
        'qat_implementation':sha256(ROOT/'qvla_haq/qat.py')}
    (args.output_dir/'training_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    development_panel=DevelopmentChunks(args,partition,split_doc,episodes) if args.eval_every else None
    if development_panel:
        metric,pred,target,valid=development_panel.evaluate(policy,pre,post)
        np.savez_compressed(args.output_dir/'development_initial.npz',predictions=pred,targets=target,valid=valid)
        (args.output_dir/'development_initial.json').write_text(json.dumps(metric,indent=2)+'\n')
    components=[];teacher_steps=0;learning_rates=[];development_history=[];start_step=0
    if args.resume:
        sidecar=json.loads(args.resume.with_suffix('.json').read_text())
        if sha256(args.resume)!=sidecar['checkpoint_sha256']:raise ValueError('Resume checkpoint hash mismatch')
        start_step,h=restore_checkpoint(args.resume,policy,optimizer,identity,rng)
        losses=h['losses'];components=h['components'];sample_rows=h['sample_rows'];teacher_steps=h['teacher_steps']
        learning_rates=h['learning_rates'];development_history=h['development_history'];gradient_by_stage=h['gradient_by_stage']
        if not 0<=start_step<args.steps:raise ValueError('Checkpoint already completed this training budget')
    policy.train();torch.cuda.reset_peak_memory_stats()
    for step in range(start_step,args.steps):
        lr=learning_rate_at(step,args.steps,args.learning_rate,args.minimum_learning_rate,warmup,args.lr_schedule)
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True);update_loss=0.
        for micro in range(args.gradient_accumulation):
            if args.teacher_task_fraction is None:
                task=sorted(by_task)[(step*args.gradient_accumulation+micro)%len(by_task)]
            else:
                pool=teacher_tasks if rng.random()<args.teacher_task_fraction else other_tasks
                task=rng.choice(pool)
            start,length,episode=rng.choice(by_task[task]);rank=rng.randrange(length)
            teacher_index=None if teacher_cache is None else teacher_cache.choose(task,rng)
            if teacher_index is not None:
                row=teacher_cache.rows[teacher_index];episode=int(row['episode_index']);rank=int(row['frame_index'])
                start,length,actual_task=episode_offsets[episode]
                if actual_task!=task or not 0<=rank<length:raise ValueError('Teacher row does not match dataset task/frame')
            frame=dataset[start+rank]
            if int(frame['episode_index'])!=episode or int(frame['frame_index'])!=rank:raise ValueError('Dataset frame alignment changed')
            obs={key:frame[key] for key in ('observation.state','action','task','action_is_pad') if key in frame}
            for key in ('observation.images.image','observation.images.image2'):obs[key]=frame[key].float()/255
            batch=pre(obs)
            if batch['action'].ndim==2:batch['action']=batch['action'].unsqueeze(0)
            if 'action_is_pad' in batch and batch['action_is_pad'].ndim==1:batch['action_is_pad']=batch['action_is_pad'].unsqueeze(0)
            teacher=None
            if teacher_index is not None:
                if frame['task']!=row['task']:raise ValueError('Teacher task description mismatch')
                teacher=teacher_batch(obs,pre,teacher_cache.actions[teacher_index],row['valid_length'],teacher_cache.accepted_masks[teacher_index])
                if args.teacher_weight>0:teacher_steps+=1
            loss,component=combined_flow_loss(policy,batch,teacher,args.teacher_weight)
            components.append(component)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            (loss/args.gradient_accumulation).backward();update_loss+=float(loss.detach())/args.gradient_accumulation
            sample_rows.append({'step':step+1,'microbatch':micro,'task':task,'episode':episode,'frame':rank})
        norm=torch.nn.utils.clip_grad_norm_(trainable,1.)
        if not torch.isfinite(norm) or float(norm)==0:raise ValueError('Invalid training gradients')
        if step==0:
            for stage,pattern in stage_names.items():
                gradient_by_stage[stage]=sum(float(policy.get_submodule(n).master_weight.grad.abs().sum())
                    for n in names if pattern in n and policy.get_submodule(n).master_weight.grad is not None)
            expected=[stage for stage,pattern in stage_names.items() if any(pattern in n for n in scope_audit['trainable_names'])]
            if any(gradient_by_stage[stage]<=0 or not np.isfinite(gradient_by_stage[stage]) for stage in expected):raise ValueError('Missing trainable stage gradients')
            if any(p.grad is not None for p in policy.parameters() if not p.requires_grad):raise ValueError('Frozen parameter has gradient')
        optimizer.step();losses.append(update_loss);learning_rates.append(lr)
        progress={'step':step+1,'steps':args.steps,'loss':losses[-1],'gradient_norm':float(norm),'learning_rate':lr,
                  'teacher_supervised_microbatches':teacher_steps,'gradient_accumulation':args.gradient_accumulation,
                  'elapsed_seconds':time.perf_counter()-started,'peak_cuda_memory_bytes':torch.cuda.max_memory_allocated()}
        (args.output_dir/'progress.json').write_text(json.dumps(progress,indent=2)+'\n');print(json.dumps(progress),flush=True)
        if development_panel and ((step+1)%args.eval_every==0 or step+1==args.steps):
            metric,pred,target,valid=development_panel.evaluate(policy,pre,post);metric['completed_steps']=step+1
            development_history.append(metric)
            evaldir=args.output_dir/'development'/f'step_{step+1:06d}';evaldir.mkdir(parents=True)
            np.savez_compressed(evaldir/'actions.npz',predictions=pred,targets=target,valid=valid)
            (evaldir/'report.json').write_text(json.dumps(metric,indent=2)+'\n')
            print(json.dumps({'development_step':step+1,'metrics':{k:v for k,v in metric.items() if k not in ('rows','per_task_mae')}}),flush=True)
        if args.save_every and ((step+1)%args.save_every==0 or step+1==args.steps):
            history=dict(losses=losses,components=components,sample_rows=sample_rows,teacher_steps=teacher_steps,
                         learning_rates=learning_rates,development_history=development_history,gradient_by_stage=gradient_by_stage)
            checkpoint=args.output_dir/'training_state.pt'
            save_checkpoint(checkpoint,policy,optimizer,step+1,identity,rng,history)
            checkpoint.with_suffix('.json').write_text(json.dumps({'completed_steps':step+1,'checkpoint_sha256':sha256(checkpoint),'identity':identity},indent=2)+'\n')
            snapshot=args.output_dir/'checkpoints'/f'step_{step+1:06d}';snapshot.mkdir(parents=True)
            state={k.removesuffix('.master_weight')+'.weight' if k.endswith('.master_weight') else k:
                   v.detach().cpu().contiguous() for k,v in policy.state_dict().items()}
            masterfile=snapshot/('distilled_float_master.safetensors' if args.mode=='fp-distill' else 'qat_float_master.safetensors')
            save_file(state,str(masterfile));del state
            snapshot_report={'status':'periodic_float_master_not_converted_quantized_model','steps':step+1,
                'mode':args.mode,'master_sha256':sha256(masterfile),'teacher_source_kind':None if teacher_cache is None else teacher_cache.manifest['source_kind'],
                'teacher_manifest_sha256':identity['teacher_manifest'],'partition_sha256':identity['partition'],
                'source_weight_sha256':identity['source'],'train_episode_ids':episodes,'learning_rate':lr,
                'closed_loop_quality':'not_measured','RKNN_conversion_verified':False}
            (snapshot/'report.json').write_text(json.dumps(snapshot_report,indent=2)+'\n')
            for old in sorted((args.output_dir/'checkpoints').glob('step_*'))[:-args.keep_snapshots]:shutil.rmtree(old)
    policy.eval()
    operator_errors={};audit_handles=[]
    if args.operator_audit and args.mode=='qat':
        for name in names:
            mod=policy.get_submodule(name)
            ordinary=copy.deepcopy(originals[name])
            ordinary.weight=torch.nn.Parameter(mod.master_weight.detach().cpu().clone())
            if mod.bias is not None:ordinary.bias=torch.nn.Parameter(mod.bias.detach().cpu().clone())
            reference=ConfiguredOperator(ordinary,assignment[name],stats.get(name)).cuda()
            reference.original_dtype=mod.original_dtype
            reference._dtype_marker=reference._dtype_marker.to(mod.original_dtype)
            def compare_operator(module,inputs,output,name=name,reference=reference):
                if name in operator_errors:return
                expected=reference(inputs[0]);difference=(output-expected).abs()
                operator_errors[name]={'mae':float(difference.mean()),'max_abs':float(difference.max()),
                                     'format':assignment[name],'original_dtype':str(originals[name].weight.dtype)}
            audit_handles.append(mod.register_forward_hook(compare_operator))
    probe_noise=torch.randn((batch['action'].shape[0],policy.config.chunk_size,policy.config.max_action_dim),device='cuda')
    with torch.no_grad():
        fake_actions=policy.predict_action_chunk(batch,noise=probe_noise).cpu().numpy()
    for handle in audit_handles:handle.remove()
    scope_audit['frozen_final_sha256']=frozen_parameter_digest(policy)
    scope_audit['frozen_parameters_unchanged']=scope_audit['frozen_final_sha256']==frozen_initial_digest
    if not scope_audit['frozen_parameters_unchanged']:raise ValueError('Frozen parameters changed during training')
    (args.output_dir/'training_scope.json').write_text(json.dumps(scope_audit,indent=2)+'\n')
    if operator_errors:
        (args.output_dir/'operator_parity.json').write_text(json.dumps(operator_errors,indent=2)+'\n')
        print('Largest operator errors '+json.dumps(sorted(operator_errors.items(),key=lambda x:x[1]['mae'],reverse=True)[:8]),flush=True)
    retained=set(scope_audit['trainable_names']) if args.save_trainable_only else set(policy.state_dict())
    master={source_key(k):v.detach().cpu().contiguous() for k,v in policy.state_dict().items() if k in retained}
    master_path=args.output_dir/('distilled_float_master.safetensors' if args.mode=='fp-distill' else 'qat_float_master.safetensors');save_file(master,str(master_path));del master
    del optimizer
    teacher_report={'teacher_loss_used':teacher_steps>0,'teacher_supervised_steps':teacher_steps,
        'teacher_source_kind':None if teacher_cache is None else teacher_cache.manifest['source_kind'],
        'teacher_manifest_sha256':None if teacher_cache is None else sha256(args.teacher_cache/'manifest.json'),
        'lr_schedule':args.lr_schedule,'warmup_steps':warmup,'minimum_learning_rate':args.minimum_learning_rate,
        'learning_rate_history':learning_rates,'development_history':development_history,
        'gradient_accumulation':args.gradient_accumulation,'teacher_task_fraction':args.teacher_task_fraction,'resumed_from_step':start_step,
        'teacher_review_sha256':identity['teacher_review'],
        'loss_components':components,'initial_master_sha256':None if args.initial_master is None else sha256(args.initial_master),
        'training_scope':scope_audit,'master_storage_format':identity['master_storage_format'],
        'fp_checkpoint_source_required':args.save_trainable_only}
    if args.mode=='fp-distill':
        restored=load_file(str(master_path))
        restored=reconstruct_fp_state(restored,policy.state_dict(),set(names),teacher_report)
        policy.load_state_dict(restored,strict=True)
        with torch.no_grad():reloaded=policy.predict_action_chunk(batch,noise=probe_noise).cpu().numpy()
        error=np.abs(fake_actions-reloaded)
        parity={'mae':float(error.mean()),'max_abs':float(error.max())}
        if parity['max_abs']!=0:raise ValueError('FP master strict reload parity failed')
        report={'status':'FP_student_distillation_training_path_not_quality_validated','steps':args.steps,
            'seed':args.seed,'learning_rate':args.learning_rate,'losses':losses,'sample_rows':sample_rows,
            'train_episode_ids':episodes,'gradient_l1_by_stage':gradient_by_stage,
            'partition_sha256':sha256(args.partition),'split_sha256':sha256(args.splits),
            'source_weight_sha256':candidate['source_checkpoint_sha256'],'master_sha256':sha256(master_path),
            'strict_master_reload':True,'reload_action_parity':parity,'fake_quant_enabled':False,
            'closed_loop_quality':'not_measured','elapsed_seconds':time.perf_counter()-started,**teacher_report}
        (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('sample_rows','train_episode_ids')}),flush=True)
        return
    # Restore original ordinary modules and strictly reload the trained FP state.
    for name,original in originals.items():
        original.weight=torch.nn.Parameter(original.weight.detach().float())
        if getattr(original,'bias',None) is not None:original.bias=torch.nn.Parameter(original.bias.detach().float())
        parent,leaf=name.rsplit('.',1);setattr(policy.get_submodule(parent),leaf,original)
    policy.load_state_dict(load_file(str(master_path)),strict=True)
    trained={n:copy.deepcopy(policy.get_submodule(n)).cpu() for n in names}
    apply_assignment(policy,space,assignment,stats,trained)
    # Masters loaded into float modules must not change the external dispatch dtype.
    for name in names:
        module=policy.get_submodule(name)
        module.original_dtype=original_dtypes[name]
        module._dtype_marker=module._dtype_marker.to(original_dtypes[name])
    packed_path=args.output_dir/'qat_local_packed.safetensors'
    packed_state={k:v.detach().cpu().contiguous() for k,v in policy.state_dict().items()}
    # Preserve the source dtype of frozen, unconfigured tensors, as RL packing does.
    with safe_open(args.model_dir/'model.safetensors',framework='pt') as source:
        trained_bias_keys={n+'.bias' for n in names}
        for key in source.keys():
            if key in packed_state and key not in trained_bias_keys and source.get_slice(key).get_dtype()=='BF16':
                packed_state[key]=packed_state[key].to(torch.bfloat16)
    save_file(packed_state,str(packed_path));del packed_state
    policy.load_state_dict(load_file(str(packed_path)),strict=True)
    for n in names:policy.get_submodule(n).refresh()
    policy.to('cuda').eval()
    with torch.no_grad():
        packed_loss,_=policy.forward(batch)
        packed_actions=policy.predict_action_chunk(batch,noise=probe_noise).cpu().numpy()
    if not torch.isfinite(packed_loss):raise ValueError('Reloaded packed policy failed')
    error=np.abs(fake_actions-packed_actions)
    parity={'mae':float(error.mean()),'max_abs':float(error.max()),'mae_limit':0.0001,'max_abs_limit':0.002,
            'scope':'same_training_observation_and_noise_full_action_chunk_local_pack_parity'}
    np.savez_compressed(args.output_dir/'reload_parity.npz',fake_actions=fake_actions,
                        packed_actions=packed_actions,noise=probe_noise.cpu().numpy())
    (args.output_dir/'reload_parity.json').write_text(json.dumps(parity,indent=2)+'\n')
    if parity['mae']>parity['mae_limit'] or parity['max_abs']>parity['max_abs_limit']:
        raise ValueError('Fake/packed full action parity failed; inspect reload_parity.json')
    report={'status':'QAT_float_master_and_strictly_reloaded_local_pack_not_RKNN',
            'candidate_version':candidate['version'],'steps':args.steps,'seed':args.seed,'learning_rate':args.learning_rate,
            'candidate_config_sha256':sha256(args.candidate),'source_weight_sha256':candidate['source_checkpoint_sha256'],
            'partition_sha256':sha256(args.partition),'split_sha256':sha256(args.splits),
            'train_episode_ids':episodes,'calibration_episode_ids':cal['episode_ids'],'sample_rows':sample_rows,
            'selected_modules':names,'trainable_parameters':sum(v.numel() for v in trainable),
            'gradient_l1_by_stage':gradient_by_stage,'losses':losses,'packed_training_forward_loss':float(packed_loss),
            'master_sha256':sha256(master_path),'packed_sha256':sha256(packed_path),
            'packed_file_bytes':packed_path.stat().st_size,
            'packed_compression_fraction':1-packed_path.stat().st_size/(args.model_dir/'model.safetensors').stat().st_size,
            'strict_master_reload':True,'strict_pack_reload':True,**teacher_report,
            'reload_action_parity':parity,
            'peak_cuda_memory_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.perf_counter()-started,
            'torch_version':torch.__version__,'RKNN_conversion_verified':False,
            'closed_loop_quality':'not_measured','hardware_resource_benefits':'not_measured',
            'limitation':'INT16/DFP/Conv numeric references need RKNN parity; QAT is diagnostic only'}
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('train_episode_ids','selected_modules','sample_rows') }),flush=True)


if __name__=='__main__':main()
