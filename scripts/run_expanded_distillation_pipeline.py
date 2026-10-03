#!/usr/bin/env python3
"""Background prepare -> teacher -> contract checks -> FP distillation -> long10.

Never starts QAT. Writes stage status and forwards live subprocess logs.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    status_path=ROOT/'runs/distillation_expanded_v3_status.json'
    with (ROOT/'runs/distillation_expanded_v3.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=dict(os.environ,MUJOCO_GL='egl',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                 LIBERO_CONFIG_PATH=str(Path.home()/'.libero'),OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        started=time.time();history=[]
        def status(stage,state,**extra):
            record=dict(stage=stage,state=state,pid=os.getpid(),elapsed_seconds=time.time()-started,**extra)
            status_path.write_text(json.dumps({**record,'history':history},indent=2)+'\n')
            print(json.dumps(record),flush=True)
        def run(stage,cmd):
            status(stage,'running')
            done=subprocess.run(cmd,cwd=ROOT,env=env)
            history.append(dict(stage=stage,returncode=done.returncode))
            if done.returncode:
                status(stage,'failed',returncode=done.returncode);raise RuntimeError('Stage failed: '+stage)
            status(stage,'completed')
        cfg=json.loads((ROOT/'config/distillation_expanded_v3.json').read_text())
        completed={'prepare':ROOT/cfg['teacher_inputs']/'manifest.json',
                   'label':ROOT/cfg['teacher_cache']/'manifest.json',
                   'review':ROOT/cfg['teacher_review'],
                   'audit':ROOT/cfg['teacher_cache']/'contract_audit.json',
                   'fp':ROOT/'runs/distill_fp_expanded_v3/report.json'}
        for stage in ('prepare','label','review','audit','fp'):
            if completed[stage].exists():
                status(stage,'existing_artifact_next_stage_will_verify');continue
            cmd=[sys.executable,'scripts/run_expanded_distillation.py','--stage',stage]
            if stage=='label' and (ROOT/cfg['teacher_cache']/'teacher_progress.json').exists():cmd.append('--resume')
            if stage=='fp' and (ROOT/'runs/distill_fp_expanded_v3/training_state.pt').exists():cmd.append('--resume')
            run(stage,cmd)
        output=ROOT/'runs/distill_fp_expanded_v3/long10'
        if not (output/'eval_info.json').exists():
            if output.exists():raise ValueError('Partial long10 output exists; preserve it before restarting evaluation')
            run('long10',[sys.executable,'scripts/eval_distill_qat_libero.py','--qvla-panel','long10',
                '--qvla-mode','distilled_fp','--qvla-run','runs/distill_fp_expanded_v3',
                '--qvla-candidate','config/haq_candidate_v2.json','--qvla-vlm-assets-dir','artifacts/smolvlm2_assets',
                '--qvla-reset-audit',str(output/'reset_audit.json'),'--qvla-init-state','0',
                '--policy.path','artifacts/model','--env.type','libero','--env.task','libero_10',
                '--env.task_ids','[0,1,2,3,4,5,6,7,8,9]','--env.observation_height','256','--env.observation_width','256',
                '--eval.n_episodes','1','--eval.batch_size','1','--eval.use_async_envs','false','--seed','0','--output_dir',str(output)])
        measured=json.loads((output/'eval_info.json').read_text())
        audit=json.loads((output/'reset_audit.json').read_text())
        actual={r['task_id']:r for r in audit if r['reset_number']==1}
        old=json.loads((ROOT/'runs/student_long10_init0_v1/summary.json').read_text())
        reference={r['task_id']:r for r in old['rows']['original_fp']}
        rows=[]
        for task in measured['per_task']:
            i=int(task['task_id']);before=reference[i]
            for key in ('init_state_sha256','initial_camera_sha256','policy_noise_seed','env_seed'):
                if actual[i][key]!=before[key]:raise ValueError('Final FP/reference pairing failed: '+key)
            rows.append(dict(task_id=i,success=bool(task['metrics']['successes'][0]),original_fp_success=before['success']))
        if sorted(r['task_id'] for r in rows)!=list(range(10)):raise ValueError('Missing long tasks')
        result=dict(scope='single_initial_state_long10_paired_screen_not_final_all_suite_quality',rows=rows,
            successes=sum(r['success'] for r in rows),original_fp_successes=sum(r['original_fp_success'] for r in rows),
            paired_state_images_noise_verified=True,other_suite_closed_loop_regression='not_measured',QAT_started=False)
        (output/'paired_summary.json').write_text(json.dumps(result,indent=2)+'\n')
        status('pipeline','completed',long10_successes=result['successes'],QAT_started=False)

if __name__=='__main__':main()
