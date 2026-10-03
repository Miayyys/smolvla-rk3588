#!/usr/bin/env python3
"""Nine preselected other-suite tasks across five fixed FP checkpoints."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
SUITES=('libero_spatial','libero_object','libero_goal')
ARMS=[('original_fp',None),
      ('first8_gt','runs/distillation_expert_first8_v1/gt_lr1e6'),
      ('first8_kd','runs/distillation_expert_first8_v1/kd_lr1e6'),
      ('expert_interface_gt','runs/distillation_expert_interface_v1/gt_lr1e6'),
      ('expert_interface_kd','runs/distillation_expert_interface_v1/kd_lr1e6')]


def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')


def main():
    out=ROOT/'runs/freezing_other9_v1';out.mkdir(exist_ok=True)
    with (out/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=dict(os.environ,MUJOCO_GL='egl',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                 LIBERO_CONFIG_PATH=str(Path.home()/'.libero'),OMP_NUM_THREADS='1')
        started=time.time();results={};reference=None
        expected={(s,i) for s in SUITES for i in (0,4,8)}
        def status(stage,state,**extra):
            row=dict(stage=stage,state=state,pid=os.getpid(),elapsed_seconds=time.time()-started,
                     training_started=False,QAT_started=False,**extra)
            write(out/'status.json',row);print(json.dumps(row),flush=True)
        write(out/'protocol.json',dict(suites=SUITES,task_ids=[0,4,8],initial_state_index=0,
             policy_seed_rule='100000*(suite_index+1)+task_id',env_seed=0,
             selection='fixed task IDs spanning suite order before seeing these checkpoint outcomes; not random representative sample',
             arms=ARMS,episodes_per_arm=9,training_started=False))
        for label,run in ARMS:
            dest=out/label
            if not (dest/'eval_info.json').exists():
                if dest.exists():raise ValueError('Partial output must be preserved before retry: '+str(dest))
                dest.mkdir()
                command=[sys.executable,'qvla/evaluation/eval_distill_qat_libero.py','--qvla-panel','other9',
                    '--qvla-mode','distilled_fp' if run else 'original_fp',
                    '--qvla-candidate','config/haq_candidate_v2.json',
                    '--qvla-vlm-assets-dir','artifacts/smolvlm2_assets',
                    '--qvla-reset-audit',str(dest/'reset_audit.json'),
                    '--qvla-action-trace',str(dest/'action_trace.json'),'--qvla-init-state','0',
                    '--policy.path','artifacts/model','--env.type','libero',
                    '--env.task',','.join(SUITES),'--env.task_ids','[0,4,8]',
                    '--env.observation_height','256','--env.observation_width','256',
                    '--eval.n_episodes','1','--eval.batch_size','1','--eval.use_async_envs','false',
                    '--seed','0','--output_dir',str(dest)]
                if run:command.extend(['--qvla-run',run])
                write(dest/'command.json',command);status(label,'running')
                with (dest/'rollout.log').open('w') as log:
                    done=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                if done.returncode:
                    status(label,'failed',returncode=done.returncode);raise RuntimeError('Inspect '+label+'/rollout.log')
            audits={(r['suite'],r['task_id']):r for r in read(dest/'reset_audit.json') if r['reset_number']==1}
            traces=read(dest/'action_trace.json');rows=[]
            for task in read(dest/'eval_info.json')['per_task']:
                key=(task['task_group'],int(task['task_id']))
                a=audits[key];points=[r for r in traces if (r['suite'],r['task_id'])==key]
                rows.append({**a,'success':bool(task['metrics']['successes'][0]),
                             'recorded_env_steps':len(points)})
            if {(r['suite'],r['task_id']) for r in rows}!=expected:raise ValueError('Unexpected task panel')
            if reference is None:reference={(r['suite'],r['task_id']):r for r in rows}
            for row in rows:
                base=reference[(row['suite'],row['task_id'])]
                for key in ('init_state_sha256','initial_camera_sha256','policy_noise_seed','env_seed'):
                    if row[key]!=base[key]:raise ValueError('Unpaired '+key)
            result=dict(rows=rows,successes=sum(r['success'] for r in rows),
                by_suite={s:sum(r['success'] for r in rows if r['suite']==s) for s in SUITES},
                lost=[dict(suite=r['suite'],task_id=r['task_id']) for r in rows
                      if reference[(r['suite'],r['task_id'])]['success'] and not r['success']],
                gained=[dict(suite=r['suite'],task_id=r['task_id']) for r in rows
                        if not reference[(r['suite'],r['task_id'])]['success'] and r['success']])
            if run:
                train=read(ROOT/run/'report.json')
                if not train['training_scope']['frozen_parameters_unchanged']:raise ValueError('Frozen integrity failed')
                result['master_sha256']=train['master_sha256']
                result['train_scope']=train['training_scope']['scope']
            results[label]=result
            combined={}
            for arm,source in ARMS:
                if arm not in results:continue
                if source:
                    parent=Path(source).parent
                    lr_label=Path(source).name
                    long=read(ROOT/parent/'results.json')['arms'][lr_label]
                    long_successes=long['successes']
                else:long_successes=7
                combined[arm]=dict(other9_successes=results[arm]['successes'],
                    long10_successes=long_successes,total19_successes=results[arm]['successes']+long_successes)
            write(out/'summary.json',dict(scope='other9_fixed_panel_FP_freezing_comparison_not_formal_all_suite_benchmark',
                results=results,combined_with_previous_long10=combined,
                total_completed_episodes=9*len(results),paired_state_images_noise_verified=True,
                elapsed_seconds=time.time()-started,QAT_started=False,
                limitation='One initial state per task, nonrandom subset. Combined19 combines two runs; '
                           'not the full40, not evidence of general statistical superiority.'))
            status(label,'completed',successes=result['successes'],by_suite=result['by_suite'])
        status('pipeline','completed',successes={k:v['successes'] for k,v in results.items()})


if __name__=='__main__':main()
