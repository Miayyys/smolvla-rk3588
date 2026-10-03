#!/usr/bin/env python3
"""Predeclared paired development screening of six frozen candidates plus FP."""
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/six_candidate_screen_init1_seed2_v1'
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')
TASKS=[1,2,5]
ARMS=[('original_fp','original_fp','v1',None)]
for v in ('v1','v2'):
    ARMS.extend([(v+'_qat_teacher','distilled_v2_qat',v,f'runs/qat_distilled_full_{v}_v1'),
                 (v+'_qat_no_teacher','distilled_v2_qat',v,f'runs/qat_distilled_{v}_no_teacher_v1')])
ARMS.extend([(v+'_haq_ptq','original_v2',v,None) for v in ('v1','v2')])
def read(p): return json.loads(p.read_text())
def write(p,v): p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')


def main():
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.time()
        protocol=dict(scope='new_fixed12_development_screen_not_final_test',task_ids=TASKS,suites=SUITES,
                      seed=2,initial_state=1,episodes_per_task=1,arms=ARMS,
                      metrics=['success','steps_for_shared_successes','observed_A10_chunk_latency','peak_cuda_memory'],
                      selection='Report all results; no post-hoc task selection or stopping at a favorable candidate')
        if (BASE/'protocol.json').exists() and read(BASE/'protocol.json')!=json.loads(json.dumps(protocol)):
            raise ValueError('Existing protocol differs')
        write(BASE/'protocol.json',protocol)
        def status(stage,state,**extra):
            row=dict(stage=stage,state=state,pid=os.getpid(),elapsed_seconds=time.time()-started,**extra)
            write(BASE/'status.json',row);print(json.dumps(row),flush=True)
        env=dict(os.environ,MUJOCO_GL='egl',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                 OMP_NUM_THREADS='1',LIBERO_CONFIG_PATH=str(Path.home()/'.libero'))
        results={};reference=None
        for label,mode,version,source in ARMS:
            out=BASE/label
            if mode=='original_v2':
                pack=ROOT/f'artifacts/candidates/{version}/model.safetensors'
                while not pack.exists() or pack.stat().st_size!=read(ROOT/f'config/haq_candidate_{version}.json')['actual_model_bytes']:
                    status(label,'waiting_for_local_pack_upload',path=str(pack));time.sleep(30)
            if not (out/'eval_info.json').exists():
                if out.exists():raise ValueError('Partial evaluation requires inspection: '+str(out))
                out.mkdir()
                cmd=[sys.executable,'scripts/eval_distill_qat_libero.py',
                     '--qvla-panel','custom','--qvla-task-ids',*[str(t) for t in TASKS],
                     '--qvla-mode',mode,'--qvla-candidate',f'config/haq_candidate_{version}.json',
                     '--qvla-vlm-assets-dir','artifacts/smolvlm2_assets',
                     '--qvla-reset-audit',str(out/'reset_audit.json'),
                     '--qvla-action-trace',str(out/'action_trace.json'),
                     '--qvla-runtime-metrics',str(out/'runtime_metrics.json'),
                     '--qvla-init-state','1','--policy.path','artifacts/model',
                     '--env.type','libero','--env.task',','.join(SUITES),'--env.task_ids',json.dumps(TASKS),
                     '--env.observation_height','256','--env.observation_width','256',
                     '--eval.n_episodes','1','--eval.batch_size','1','--eval.use_async_envs','false',
                     '--seed','2','--output_dir',str(out)]
                if source:cmd.extend(['--qvla-run',source])
                write(out/'command.json',cmd);status(label,'running')
                with (out/'rollout.log').open('w') as log:
                    done=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                if done.returncode:status(label,'failed',returncode=done.returncode);raise RuntimeError('Inspect '+str(out/'rollout.log'))
            audit={(r['suite'],r['task_id']):r for r in read(out/'reset_audit.json') if r['reset_number']==1}
            keys={(s,t) for s in SUITES for t in TASKS}
            if set(audit)!=keys:raise ValueError('Unexpected reset task set')
            if reference is None:reference=audit
            for k in keys:
                for f in ('env_seed','actual_init_state_index','init_state_sha256','initial_camera_sha256','policy_noise_seed'):
                    if audit[k][f]!=reference[k][f]:raise ValueError('Unpaired '+label+' '+f)
            traces=read(out/'action_trace.json');runtime=read(out/'runtime_metrics.json')
            rows=[]
            for r in read(out/'eval_info.json')['per_task']:
                k=(r['task_group'],int(r['task_id']))
                actions=[x for x in traces if (x['suite'],x['task_id'])==k]
                timing=[x['chunk_ms'] for x in runtime['rows'] if (x['suite'],x['task_id'])==k][2:]
                if not actions:raise ValueError('Missing action trace')
                rows.append(dict(suite=k[0],task_id=k[1],success=bool(r['metrics']['successes'][0]),
                    rollout_steps=len(actions),chunk_samples_after_two_warmups=len(timing),
                    chunk_p50_ms=statistics.median(timing) if timing else None))
            if {(r['suite'],r['task_id']) for r in rows}!=keys:raise ValueError('Incomplete results')
            results[label]=dict(rows=rows,successes=sum(x['success'] for x in rows),
                per_suite={s:sum(x['success'] for x in rows if x['suite']==s) for s in SUITES},
                peak_cuda_allocated_bytes=runtime['peak_cuda_allocated_bytes'],
                peak_cuda_reserved_bytes=runtime['peak_cuda_reserved_bytes'])
            fp={(x['suite'],x['task_id']):x for x in results['original_fp']['rows']}
            for x in rows:
                b=fp[(x['suite'],x['task_id'])];x['FP_success']=b['success'];x['FP_steps']=b['rollout_steps']
            results[label]['gained']=[{'suite':x['suite'],'task_id':x['task_id']} for x in rows if x['success'] and not x['FP_success']]
            results[label]['lost']=[{'suite':x['suite'],'task_id':x['task_id']} for x in rows if not x['success'] and x['FP_success']]
            shared=[x for x in rows if x['success'] and x['FP_success']]
            results[label]['shared_success_step_change_vs_FP']=dict(count=len(shared),
                mean_steps_change=statistics.mean(x['rollout_steps']-x['FP_steps'] for x in shared) if shared else None)
            write(BASE/'summary.json',dict(protocol=protocol,completed_arms=list(results),results=results,
                limitations='One new state and seed; exploratory screening, report all tasks. Latencies use different observed rollout inputs and are A10 local reference performance, not RK3588 benefit. Fewer steps only compared when both succeed.'))
            status(label,'completed',successes=results[label]['successes'],completed_arms=len(results))
        status('pipeline','completed',successes={k:r['successes'] for k,r in results.items()})

if __name__=='__main__':main()
