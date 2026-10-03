#!/usr/bin/env python3
"""Diagnostic RL loop with real full-policy actions and signature cost lookup.

Not formal HAQ deployment search: INT16/Conv numeric references are not RKNN
replicas, lookup is isolated costs, and offline quality is not success rate.
"""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import argparse
import copy
import gc
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from qvla.evaluation.haq_offline_eval import identity, load_policy, predict_cached
from qvla.evaluation.probe_action_sensitivity import selected_frames, run_action
from qvla.haq.offline_actions import file_sha256, load_cache, score_actions
from qvla.haq.policy import RecurrentPolicyGradient
from qvla.haq.feasibility_reward import score_feasibility
from qvla.haq.action_quality import action_quality
from qvla.haq.runtime import apply_assignment, standalone_table_cost
from qvla.haq.search_space import build_search_space


def write(path, obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, default=ROOT/'artifacts/transfer/model')
    p.add_argument('--vlm-assets-dir', type=Path, default=ROOT/'artifacts/transfer/smolvlm2_assets')
    p.add_argument('--dataset-root', type=Path, default=ROOT/'artifacts/transfer/libero')
    p.add_argument('--cache',type=Path,default=ROOT/'runs/haq_offline_local_v1/fp_cache40')
    p.add_argument('--splits',type=Path,default=ROOT/'data/libero_splits.json')
    p.add_argument('--partition',type=Path,default=ROOT/'config/evaluation_partition_v2.json')
    p.add_argument('--tables',type=Path,default=ROOT/'config/hardware/tables.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--seed',type=int,default=29)
    p.add_argument('--int8-logit-prior',type=float,default=0.,
                   help='Finite initial logit bias based only on storage budget; no option masked')
    p.add_argument('--entropy-weight',type=float,default=0.,
                   help='Default zero verifies updates driven by measured feedback alone')
    p.add_argument('--entropy-final',type=float)
    p.add_argument('--quality-mode',choices=('chunk_mae','trajectory_gripper'),default='chunk_mae')
    p.add_argument('--early-stop',action='store_true')
    p.add_argument('--min-rounds',type=int,default=100)
    p.add_argument('--patience',type=int,default=60)
    p.add_argument('--check-every',type=int,default=20)
    p.add_argument('--improvement-threshold',type=float,default=1e-4)
    p.add_argument('--validation-task-indices',type=int,nargs='+')
    p.add_argument('--score-task-indices',type=int,nargs='+',
                   help='Fixed small development panel for longer search; other tasks remain held out')
    p.add_argument('--random-control',action='store_true',
                   help='Same initial policy, sampled without parameter updates')
    p.add_argument('--checkpoint-retention',choices=('all','best'),default='all')
    p.add_argument('--device',default='cuda')
    args = p.parse_args()
    if args.rounds < 1 or args.batch_size < 2:
        p.error('Need positive rounds and batch size >=2')
    if min(args.min_rounds,args.patience,args.check_every)<1 or args.improvement_threshold<=0:
        p.error('Stop settings must be positive')
    if args.entropy_weight<0 or (args.entropy_final is not None and args.entropy_final<0):
        p.error('Entropy weights must be nonnegative')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    started = time.perf_counter()
    ident = identity(args)
    manifest, obs, fp = load_cache(args.cache,ident)
    if torch.__version__ != manifest['torch_version'] or args.device != manifest['device']:
        raise ValueError('Cache runtime mismatch')
    space = build_search_space(args.tables)
    tables = json.loads(args.tables.read_text())
    write(args.output/'tables_snapshot.json',tables)
    write(args.output/'space.json',space)
    policy,pre,post = load_policy(args)
    if policy.config.num_steps != manifest['inference_steps']:
        raise ValueError('Flow inference step mismatch')
    names = [s['module'] for s in space['action_sites']]
    stats = {name: {'min':float('inf'),'max':float('-inf'),'calls':0,'shapes':[]}
             for name in names}
    handles=[]
    for name in names:
        def hook(module, inputs, name=name):
            x=inputs[0].detach(); r=stats[name]
            lo,hi=float(x.min()),float(x.max())
            if not math.isfinite(lo+hi):
                raise ValueError('Nonfinite calibration')
            r['min']=min(r['min'],lo);r['max']=max(r['max'],hi);r['calls']+=1
            if list(x.shape) not in r['shapes']:r['shapes'].append(list(x.shape))
        handles.append(policy.get_submodule(name).register_forward_pre_hook(hook))
    part=json.loads(args.partition.read_text())
    episodes=part['calibration_episode_ids_from_ptq_calibration']
    rows=selected_frames(args.dataset_root,episodes,1,40)
    if len(rows)!=40 or set(r['episode_index'] for r in rows)!=set(episodes):
        raise ValueError('Incomplete isolated calibration episodes')
    try:
        for i,row in enumerate(rows):
            run_action(policy,pre,post,row)
            if (i+1)%10==0: print(f'calibration {i+1}/40',flush=True)
    finally:
        for h in handles:h.remove()
    if any(r['calls']==0 for r in stats.values()):
        raise ValueError('Selectable sites missing from actual action execution')
    write(args.output/'calibration.json',{'episode_ids':episodes,'ranges':stats,
          'samples':[ {k:r[k] for k in ['episode_index','task_index','frame_rank']} for r in rows],
          'scope':'40 isolated calibration episodes; one selected frame each; existing seed rule'})
    del rows
    calls={name:{'calls':0,'shapes':[]} for name in names}
    handles=[]
    for name in names:
        def trace(module,inputs,name=name):
            r=calls[name];r['calls']+=1
            if list(inputs[0].shape) not in r['shapes']:r['shapes'].append(list(inputs[0].shape))
        handles.append(policy.get_submodule(name).register_forward_pre_hook(trace))
    try:
        check,_=predict_cached(policy,pre,post,obs,manifest['samples'],args.device)
        np.testing.assert_array_equal(check,fp)
    finally:
        for h in handles:h.remove()
    for r in calls.values():r['calls_per_observation']=r['calls']/len(fp)
    write(args.output/'execution_calls.json',calls)
    full_obs=obs
    full_fp=fp
    full_samples=manifest['samples']
    holdout_positions=[]
    if args.score_task_indices is not None:
        chosen=set(args.score_task_indices)
        available={r['task_index'] for r in manifest['samples']}
        if len(chosen)!=len(args.score_task_indices) or not chosen or not chosen<=available:
            raise ValueError('Score task indices must be unique, nonempty development tasks')
        positions=[i for i,r in enumerate(manifest['samples']) if r['task_index'] in chosen]
        holdout_positions=[i for i,r in enumerate(manifest['samples']) if r['task_index'] not in chosen]
        obs={k:v[positions] for k,v in obs.items()}
        fp=fp[positions]
        score_samples=[manifest['samples'][i] for i in positions]
    else:
        score_samples=manifest['samples']
    validation_positions=[]
    if args.validation_task_indices:
        ids=set(args.validation_task_indices)
        if len(ids)!=len(args.validation_task_indices) or ids & {s['task_index'] for s in score_samples}:
            raise ValueError('Validation tasks must be unique and separate from search')
        validation_positions=[i for i,s in enumerate(full_samples) if s['task_index'] in ids]
        if {full_samples[i]['task_index'] for i in validation_positions}!=ids:
            raise ValueError('Missing validation tasks')
        holdout_positions=[i for i in holdout_positions if i not in validation_positions]
    if args.early_stop and (not validation_positions or args.random_control):
        raise ValueError('Early stop requires a separate validation panel and learning arm')
    # Keep source operators on CPU; each candidate is rebuilt from FP, never
    # successively quantize an already quantized candidate.
    originals={name:copy.deepcopy(policy.get_submodule(name)).cpu() for name in names}
    ref_assignment={s['module']:('native_bf16_row_lookup' if s['kind']=='embedding' else 'bfloat16')
                    for s in space['action_sites']}
    ref_cost=standalone_table_cost(space,tables,ref_assignment,calls)
    write(args.output/'reference_cost.json',ref_cost)
    controller=RecurrentPolicyGradient(space,hidden_size=16,seed=args.seed)
    if not math.isfinite(args.int8_logit_prior):
        raise ValueError('Initial logit prior must be finite')
    for fmt in ['w8a8','cpu_int8_row_lookup']:
        controller.params['bo'][space['format_vocab'].index(fmt)]+=args.int8_logit_prior
    controller.save(args.output/'controller_initial.npz')
    initial={k:v.copy() for k,v in controller.params.items()}
    action_scale=max(float(np.abs(fp.astype(np.float64)).mean()),1e-8)
    source_dtypes={}
    with safe_open(args.model_dir/'model.safetensors',framework='pt') as source:
        source_dtypes={k:source.get_slice(k).get_dtype() for k in source.keys()}
    results=[];update_rows=[]
    retained_checkpoint=None
    retained_report=None
    retained_reward=-float('inf')
    source_bytes=ident['source_model_file_bytes']
    validation_history=[]
    meaningful_best=-float('inf');last_reward_improvement=0
    best_validation_error=float('inf');last_validation_improvement=0
    stop_reason='max_rounds'
    for round_id in range(args.rounds):
        samples=[controller.sample() for _ in range(args.batch_size)]
        rewards=[]
        for index,(assignment,trajectory) in enumerate(samples):
            candidate_dir=args.output/f'round{round_id+1:03d}_candidate{index+1:02d}'
            candidate_dir.mkdir()
            tick=time.perf_counter()
            write(candidate_dir/'assignment.json',assignment)
            apply_assignment(policy,space,assignment,stats,originals)
            state={k:v.detach().cpu().contiguous() for k,v in policy.state_dict().items()}
            # Unmodified BF16 source tensors remain BF16 in the file, not loader-upcast F32.
            for k,v in state.items():
                if source_dtypes.get(k)=='BF16' and v.dtype==torch.float32:
                    state[k]=v.to(torch.bfloat16)
            checkpoint=candidate_dir/'model.safetensors'
            save_file(state,str(checkpoint));del state
            # Full state is strictly reloaded before its complete-policy quality evaluation.
            policy.load_state_dict(load_file(str(checkpoint)),strict=True)
            for name in names:policy.get_submodule(name).refresh()
            policy.to(args.device).eval()
            cost=standalone_table_cost(space,tables,assignment,calls)
            actions,timing=predict_cached(policy,pre,post,obs,score_samples,args.device)
            metrics=score_actions(actions,fp,obs['recorded_action'],
                                  np.array([r['task_index'] for r in score_samples]))
            mae=metrics['task_macro']['chunk_mae_vs_fp']
            quality_report=action_quality(actions,fp,metrics,args.quality_mode)
            quality=quality_report['quality']
            gain=math.sqrt(quality*ref_cost['total_ms']/cost['total_ms'])
            size=checkpoint.stat().st_size
            feasible=5*size<=3*source_bytes
            # Feasibility is still a hard final selection condition. Infeasible
            # candidates get a negative distance signal instead of identical -1.
            compression=1-size/source_bytes
            reward_result=score_feasibility(model_bytes=size,reference_bytes=source_bytes,
                                            gain=gain,backend_status='not_attempted',allow_proxy=True)
            reward=reward_result['reward']
            rewards.append(reward)
            np.savez_compressed(candidate_dir/'actions.npz',actions=actions)
            record={'round':round_id+1,'candidate':index+1,'reward':reward,
                    'feasible_40pct':feasible,'actual_model_bytes':size,
                    'compression_fraction':1-size/source_bytes,'checkpoint_sha256':file_sha256(checkpoint),
                    'strict_state_reload':True,'format_counts':dict(Counter(assignment.values())),
                    'metrics':metrics,'quality_proxy_exp_normalized_mae':quality,'G_proxy':gain,
                    'quality_proxy':quality_report,
                    'cost':cost,'host_timing':timing,'elapsed_seconds':time.perf_counter()-tick,
                    'scope':'complete_policy_local_numeric_reference_plus_isolated_cost_lookup',
                    'RKNN_full_graph_execution':False,'closed_loop_success_rate':'not_measured',
                    'backend_feasibility':reward_result,
                    'quantized_formats_backend_equivalence':'INT16/DFP/Conv unverified against RKNN',
                    'embedding_execution':'GPU reference lookup; table uses measured CPU lookup',
                    'score_task_indices':args.score_task_indices}
            write(candidate_dir/'report.json',record)
            results.append({k:v for k,v in record.items() if k not in ['metrics','cost','host_timing']})
            if args.checkpoint_retention=='best':
                if feasible and reward>retained_reward:
                    if retained_checkpoint is not None:
                        retained_checkpoint.unlink()
                        previous=json.loads(retained_report.read_text())
                        previous['checkpoint_retained']=False
                        write(retained_report,previous)
                    retained_checkpoint=checkpoint
                    retained_report=candidate_dir/'report.json'
                    retained_reward=reward
                    record['checkpoint_retained']=True
                else:
                    checkpoint.unlink()
                    record['checkpoint_retained']=False
                write(candidate_dir/'report.json',record)
            print(json.dumps({'round':round_id+1,'candidate':index+1,'reward':reward,
                              'mae':mae,'table_cost_ms':cost['total_ms'],'compression':record['compression_fraction']}),flush=True)
            gc.collect()
        if args.random_control:
            update={'reward_mean':float(np.mean(rewards)),
                    'reward_std':float(np.std(rewards)),
                    'gradient_norm_before_clip':None,'random_no_update':True}
        else:
            entropy=(args.entropy_weight if args.entropy_final is None else
                     args.entropy_weight+(args.entropy_final-args.entropy_weight)*round_id/max(args.rounds-1,1))
            update=controller.update_batch([t for _,t in samples],rewards,
                                           entropy_weight=entropy)
            update['entropy_weight_used']=entropy
            assert all(np.isfinite(v).all() for v in controller.params.values())
        controller.save(args.output/f'controller_round{round_id+1:03d}.npz')
        update_rows.append({'round':round_id+1,'rewards':rewards,**update})
        write(args.output/'updates.json',update_rows)
        write(args.output/'progress.json',{'rounds_completed':round_id+1,
                                          'candidates_completed':len(results),
                                          'best_feasible_reward':max((r['reward'] for r in results if r['feasible_40pct']),default=None),
                                          'random_control':args.random_control})
        print(f"{'Random control' if args.random_control else 'RL update'} {round_id+1}/{args.rounds}: {update}",flush=True)
        current_best=max((r['reward'] for r in results if r['feasible_40pct']),default=-float('inf'))
        if current_best>meaningful_best+args.improvement_threshold:
            meaningful_best=current_best;last_reward_improvement=round_id+1
        if args.early_stop and (round_id+1)%args.check_every==0:
            if retained_checkpoint is None:
                last_validation_improvement=round_id+1
            else:
                assignment=json.loads((retained_checkpoint.parent/'assignment.json').read_text())
                apply_assignment(policy,space,assignment,stats,originals)
                policy.load_state_dict(load_file(str(retained_checkpoint)),strict=True)
                for name in names:policy.get_submodule(name).refresh()
                policy.to(args.device).eval()
                vobs={k:v[validation_positions] for k,v in full_obs.items()}
                vsamples=[full_samples[i] for i in validation_positions]
                va,vt=predict_cached(policy,pre,post,vobs,vsamples,args.device)
                vm=score_actions(va,full_fp[validation_positions],vobs['recorded_action'],np.array([s['task_index'] for s in vsamples]))
                vq=action_quality(va,full_fp[validation_positions],vm,args.quality_mode)
                ve=vq['normalized_error']
                if ve<best_validation_error-args.improvement_threshold:
                    best_validation_error=ve;last_validation_improvement=round_id+1
                np.savez_compressed(args.output/f'validation_round{round_id+1:03d}.npz',actions=va)
                validation_history.append({'round':round_id+1,'checkpoint_sha256':file_sha256(retained_checkpoint),
                                           'quality_proxy':vq,'metrics':vm,'host_timing':vt})
                write(args.output/'validation_history.json',validation_history)
                if (round_id+1>=args.min_rounds and
                    round_id+1-last_reward_improvement>=args.patience and
                    round_id+1-last_validation_improvement>=args.patience):
                    stop_reason='reward_and_validation_plateau';break
    feasible_results=[r for r in results if r['feasible_40pct']]
    holdout_result=None
    if args.score_task_indices is not None and retained_checkpoint is not None and holdout_positions:
        best_assignment=json.loads((retained_checkpoint.parent/'assignment.json').read_text())
        apply_assignment(policy,space,best_assignment,stats,originals)
        policy.load_state_dict(load_file(str(retained_checkpoint)),strict=True)
        for name in names:policy.get_submodule(name).refresh()
        policy.to(args.device).eval()
        held_obs={k:v[holdout_positions] for k,v in full_obs.items()}
        held_fp=full_fp[holdout_positions]
        held_samples=[full_samples[i] for i in holdout_positions]
        held_actions,held_timing=predict_cached(policy,pre,post,held_obs,held_samples,args.device)
        held_metrics=score_actions(held_actions,held_fp,held_obs['recorded_action'],
                                   np.array([r['task_index'] for r in held_samples]))
        holdout_result={'observations':len(holdout_positions),'task_indices':sorted({r['task_index'] for r in held_samples}),
                        'metrics':held_metrics,'host_timing':held_timing,
                        'checkpoint_sha256':file_sha256(retained_checkpoint),
                        'scope':'held_out_cached_actions_not_closed_loop_success'}
        write(args.output/'heldout_best.json',holdout_result)
    summary={'scope':'local_real_action_feedback_RL_diagnostic_not_formal_HAQ_search',
             'formal_search_ready':False,'rounds':len(update_rows),'max_rounds':args.rounds,'batch_size':args.batch_size,
             'seed':args.seed,'real_candidate_evaluations':len(results),'results':results,'updates':update_rows,
             'int8_logit_prior':args.int8_logit_prior,
             'entropy_weight':args.entropy_weight,
             'entropy_final':args.entropy_final,'quality_mode':args.quality_mode,
             'validation_task_indices':args.validation_task_indices,
             'validation_history':validation_history,
             'score_task_indices':args.score_task_indices,
             'random_control':args.random_control,
             'checkpoint_retention':args.checkpoint_retention,
             'heldout_best':holdout_result,
             'identity':ident,'cache_manifest_sha256':file_sha256(args.cache/'manifest.json'),
             'tables_sha256':space['source']['tables_sha256'],
             'best_by_proxy':max(feasible_results,key=lambda r:r['G_proxy']) if feasible_results else None,
             'reward_definition':{'A':('exp(-chunk_MAE_vs_FP / mean_abs_FP_action)' if args.quality_mode=='chunk_mae'
                                     else 'exp(-(.8*normalized_chunk_MAE+.1*normalized_action_delta_MAE+.1*gripper_sign_disagreement))'),
                 'mean_abs_FP_action':action_scale,'Tref_ms':ref_cost['total_ms'],
                 'G':'sqrt(A*Tref/T_lookup)',
                 'reward':'G/(1+G) if file <=60% source; else -d/(1+d), d=B/(0.6*B_FP)-1',
                 'backend_status':'not_attempted; proxy feedback only; conversion penalty not exercised',
                 'quality_scope':'offline proxy, not success rate; provisional diagnostic mapping'},
             'parameter_change_l2':float(np.sqrt(sum(np.sum((controller.params[k]-initial[k])**2) for k in initial))),
             'stop_rule':{'reason':stop_reason,'minimum_rounds':args.min_rounds,'patience':args.patience,
                          'check_every':args.check_every,'improvement_threshold':args.improvement_threshold},
             'elapsed_seconds':time.perf_counter()-started,
             'scripts_sha256':{str(f.relative_to(ROOT)):file_sha256(f) for f in
                [Path(__file__),ROOT/'qvla.haq/runtime.py',ROOT/'qvla.haq/policy.py',
                 ROOT/'qvla.haq/feasibility_reward.py',ROOT/'qvla.haq/action_quality.py']}}
    write(args.output/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['scope','rounds','real_candidate_evaluations','parameter_change_l2','elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
