#!/usr/bin/env python3
"""Original/failed-FP controls plus six module recoveries on Long tasks 3,6."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qvla_haq.module_recovery import GROUPS


def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def main():
    out=ROOT/'runs/module_recovery_v1';out.mkdir(exist_ok=True)
    with (out/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=dict(os.environ,MUJOCO_GL='egl',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                 OMP_NUM_THREADS='1',LIBERO_CONFIG_PATH=str(Path.home()/'.libero'))
        started=time.time();results={};reference=None;failed=None
        old=read(ROOT/'runs/student_long10_init0_v1/summary.json')
        previous={r['task_id']:r for r in old['rows']['original_fp']}
        def status(stage,state,**extra):
            row=dict(stage=stage,state=state,pid=os.getpid(),elapsed_seconds=time.time()-started,
                     training_started=False,QAT_started=False,**extra)
            write(out/'status.json',row);print(json.dumps(row),flush=True)
        arms=[('original_fp','original_fp',None),('failed_distilled_fp','distilled_fp',None)]
        arms.extend(('restore_'+g,'distilled_fp',g) for g in GROUPS)
        for label,mode,group in arms:
            dest=out/label
            if not (dest/'eval_info.json').exists():
                if dest.exists():raise ValueError('Preserve partial output before retry: '+str(dest))
                dest.mkdir()
                cmd=[sys.executable,'scripts/eval_distill_qat_libero.py',
                    '--qvla-mode',mode,'--qvla-panel','long10',
                    '--qvla-candidate','config/haq_candidate_v2.json',
                    '--qvla-vlm-assets-dir','artifacts/smolvlm2_assets',
                    '--qvla-reset-audit',str(dest/'reset_audit.json'),
                    '--qvla-action-trace',str(dest/'action_trace.json'),'--qvla-init-state','0',
                    '--policy.path','artifacts/model','--env.type','libero','--env.task','libero_10',
                    '--env.task_ids','[3,6]','--env.observation_height','256','--env.observation_width','256',
                    '--eval.n_episodes','1','--eval.batch_size','1','--eval.use_async_envs','false',
                    '--seed','0','--output_dir',str(dest)]
                if mode=='distilled_fp':cmd.extend(['--qvla-run','runs/distill_fp_expanded_v3'])
                if group:cmd.extend(['--qvla-restore-group',group,'--qvla-recovery-audit',str(dest/'recovery.json')])
                write(dest/'command.json',cmd);status(label,'running')
                with (dest/'rollout.log').open('w') as log:
                    done=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                if done.returncode:
                    status(label,'failed',returncode=done.returncode);raise RuntimeError('Inspect '+label+'/rollout.log')
            audit={r['task_id']:r for r in read(dest/'reset_audit.json') if r['reset_number']==1}
            raw=read(dest/'eval_info.json');trace=read(dest/'action_trace.json');rows=[]
            for task in raw['per_task']:
                i=int(task['task_id']);base=previous[i]
                for key in ('init_state_sha256','initial_camera_sha256','policy_noise_seed','env_seed'):
                    if audit[i][key]!=base[key]:raise ValueError('Pairing failed: '+key)
                points=[r for r in trace if r['task_id']==i]
                rows.append(dict(task_id=i,success=bool(task['metrics']['successes'][0]),
                                 recorded_env_steps=len(points),reward_sum=sum(r['reward'] for r in points)))
            if sorted(r['task_id'] for r in rows)!=[3,6]:raise ValueError('Missing diagnostic tasks')
            by_id={r['task_id']:r for r in rows}
            if label=='original_fp':
                if not all(r['success'] for r in rows):raise ValueError('Original successful baseline no longer reproduces')
                reference=by_id
            elif label=='failed_distilled_fp':
                if any(r['success'] for r in rows):raise ValueError('Failed FP baseline no longer reproduces')
                failed=by_id
            result=dict(rows=rows,successes=sum(r['success'] for r in rows),restored_group=group,
                        rescued_task_ids=[r['task_id'] for r in rows if group and r['success'] and not failed[r['task_id']]['success']])
            if group:
                result['recovery']=read(dest/'recovery.json')
                if result['recovery']['changed_tensors']==0:raise ValueError('Recovery has no changed weights')
            results[label]=result
            write(out/'summary.json',dict(scope='module_source_weight_recovery_on_reused_two_task_diagnostic_not_formal_quality',
                results=results,total_completed_episodes=len(results)*2,paired_state_images_noise_verified=True,
                elapsed_seconds=time.time()-started,training_started=False,QAT_started=False,
                limitations='Single initial state; recovery shows intervention impact, not sole cause or optimal trainable layers. '
                            'Action traces after trajectories diverge are not paired state action errors.'))
            status(label,'completed',successes=result['successes'],rescued_task_ids=result['rescued_task_ids'])
        status('pipeline','completed',successes={k:v['successes'] for k,v in results.items()})


if __name__=='__main__':main()
