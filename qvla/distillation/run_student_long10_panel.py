#!/usr/bin/env python3
"""Evaluate original and distilled FP on ten fixed long tasks; no training."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]

def main():
    root=ROOT/'runs/student_long10_init0_v1';root.mkdir(exist_ok=False)
    env=dict(os.environ,MUJOCO_GL='egl',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',LIBERO_CONFIG_PATH=str(Path.home()/'.libero'))
    jobs=[];started=time.time()
    for label,mode,training in [('original_fp','original_fp',None),('distilled_fp_lr1e7','distilled_fp','runs/distill_fp_real40_lr1e7_v1')]:
        out=root/label;out.mkdir();log=(out/'rollout.log').open('w')
        cmd=[sys.executable,'qvla/evaluation/eval_distill_qat_libero.py','--qvla-panel','long10','--qvla-mode',mode,
             '--qvla-candidate','config/haq_candidate_v2.json','--qvla-vlm-assets-dir','artifacts/smolvlm2_assets',
             '--qvla-reset-audit',str(out/'reset_audit.json'),'--qvla-init-state','0',
             '--policy.path','artifacts/model','--env.type','libero','--env.task','libero_10',
             '--env.task_ids','[0,1,2,3,4,5,6,7,8,9]','--env.observation_height','256','--env.observation_width','256',
             '--eval.n_episodes','1','--eval.batch_size','1','--eval.use_async_envs','false','--seed','0','--output_dir',str(out)]
        if training:cmd.extend(['--qvla-run',training])
        jobs.append((label,out,subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT),log))
    results={}
    for label,out,job,log in jobs:
        code=job.wait();log.close()
        if code:raise RuntimeError(f'{label} evaluation failed; inspect rollout.log')
        data=json.loads((out/'eval_info.json').read_text());audit=json.loads((out/'reset_audit.json').read_text())
        first={r['task_id']:r for r in audit if r['reset_number']==1}
        rows=[]
        for task in data['per_task']:
            task_id=int(task['task_id']);r=first[task_id]
            rows.append({**r,'success':bool(task['metrics']['successes'][0])})
        if sorted(r['task_id'] for r in rows)!=list(range(10)):raise ValueError('Missing long tasks')
        results[label]=rows
    ref={r['task_id']:r for r in results['original_fp']}
    for r in results['distilled_fp_lr1e7']:
        for key in ('init_state_sha256','initial_camera_sha256','policy_noise_seed','env_seed'):
            if r[key]!=ref[r['task_id']][key]:raise ValueError('Student pairing failed: '+key)
    teacher=json.loads((ROOT/'runs/teacher_long10_init0_v1/summary.json').read_text())
    teacher_ref={r['task_id']:r for r in teacher['rows']}
    teacher_pair={str(i):all(ref[i][k]==teacher_ref[i][k] for k in ('init_state_sha256','initial_camera_sha256')) for i in range(10)}
    report={'scope':'long10_single_initial_state_original_vs_short_distilled_FP_paired_screen',
            'rows':results,'successes':{name:sum(r['success'] for r in rows) for name,rows in results.items()},
            'student_pairing_verified':True,'teacher_state_and_camera_pairing_by_task':teacher_pair,
            'teacher_successes':teacher['successes'],'elapsed_seconds':time.time()-started,
            'next_training_or_data_changes':'requires_user_decision; not_started'}
    (root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
