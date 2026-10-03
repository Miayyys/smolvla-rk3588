#!/usr/bin/env python3
"""Two independently packed FP PTQ controls followed by paired 19-task panels."""

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
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')


def main():
    out=ROOT/'runs/v2_ptq_controls_v1';out.mkdir(exist_ok=True)
    with (out/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=dict(os.environ,MUJOCO_GL='egl',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                 OMP_NUM_THREADS='1',LIBERO_CONFIG_PATH=str(Path.home()/'.libero'))
        started=time.time();results={}
        arms=[('student_ptq','runs/distillation_expert_first8_v1/kd_lr1e6'),('original_ptq',None)]
        shared=['--model-dir','artifacts/model','--vlm-assets-dir','artifacts/smolvlm2_assets',
            '--dataset-root','data/libero','--splits','data/libero_splits.json',
            '--partition','config/evaluation_partition_v2.json','--candidate','config/haq_candidate_v2.json']
        reference=read(ROOT/'runs/student_long10_init0_v1/summary.json')['rows']['original_fp']
        reference+=read(ROOT/'runs/freezing_other9_v1/summary.json')['results']['original_fp']['rows']
        refs={(r['suite'],r['task_id']):r for r in reference}
        def status(stage,state,**extra):
            v=dict(stage=stage,state=state,pid=os.getpid(),elapsed_seconds=time.time()-started,
                   weight_training_performed=False,**extra)
            write(out/'status.json',v);print(json.dumps(v),flush=True)
        def run(stage,command):
            status(stage,'running');write(out/(stage+'_command.json'),command)
            with (out/(stage+'.log')).open('w') as log:
                done=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            if done.returncode:
                status(stage,'failed',returncode=done.returncode);raise RuntimeError('Inspect '+stage+'.log')
            status(stage,'completed')
        for label,student in arms:
            dest=out/label
            if not (dest/'report.json').exists():
                if dest.exists():raise ValueError('Preserve partial packing output before retry')
                cmd=[sys.executable,'qvla/quantization/pack_v2_ptq_control.py',*shared,'--output-dir',str(dest)]
                if student:cmd.extend(['--student-run',student])
                run(label+'_pack',cmd)
            report=read(dest/'report.json');panels={}
            for panel in ('long10','other9'):
                evaldir=dest/panel
                if not (evaldir/'eval_info.json').exists():
                    if evaldir.exists():raise ValueError('Preserve partial eval output before retry')
                    suites='libero_10' if panel=='long10' else 'libero_spatial,libero_object,libero_goal'
                    tasks='[0,1,2,3,4,5,6,7,8,9]' if panel=='long10' else '[0,4,8]'
                    cmd=[sys.executable,'qvla/evaluation/eval_distill_qat_libero.py','--qvla-mode','v2_ptq',
                        '--qvla-run',str(dest),'--qvla-panel',panel,
                        '--qvla-candidate','config/haq_candidate_v2.json',
                        '--qvla-vlm-assets-dir','artifacts/smolvlm2_assets',
                        '--qvla-reset-audit',str(evaldir/'reset_audit.json'),
                        '--qvla-action-trace',str(evaldir/'action_trace.json'),'--qvla-init-state','0',
                        '--policy.path','artifacts/model','--env.type','libero','--env.task',suites,
                        '--env.task_ids',tasks,'--env.observation_height','256','--env.observation_width','256',
                        '--eval.n_episodes','1','--eval.batch_size','1','--eval.use_async_envs','false',
                        '--seed','0','--output_dir',str(evaldir)]
                    run(label+'_'+panel,cmd)
                audit={(r['suite'],r['task_id']):r for r in read(evaldir/'reset_audit.json') if r['reset_number']==1}
                rows=[]
                for task in read(evaldir/'eval_info.json')['per_task']:
                    key=(task['task_group'],int(task['task_id']));base=refs[key]
                    for field in ('init_state_sha256','initial_camera_sha256','policy_noise_seed','env_seed'):
                        if audit[key][field]!=base[field]:raise ValueError('PTQ pairing failed: '+field)
                    rows.append(dict(suite=key[0],task_id=key[1],success=bool(task['metrics']['successes'][0]),
                                     original_FP_success=base['success']))
                if len(rows)!=(10 if panel=='long10' else 9):raise ValueError('Missing tasks')
                panels[panel]=dict(successes=sum(r['success'] for r in rows),rows=rows)
                results[label]=dict(panels=panels,packed_file_bytes=report['packed_file_bytes'],
                    compression=report['packed_compression_fraction'],reload_action_parity=report['reload_action_parity'],
                    source_kind=report['source_kind'],packed_sha256=report['packed_sha256'])
                qat=read(ROOT/'runs/qat_distilled_full_v2_v1_pipeline/summary.json')
                write(out/'summary.json',dict(scope='two_FP_PTQ_local_pack_fixed19_diagnostic_not_RKNN',
                    results=results,student_QAT_successes={k:v['successes'] for k,v in qat['results'].items()},
                    original_FP_and_student_FP_successes=dict(long10=7,other9=9),
                    paired_state_images_noise_verified=True,weight_training_performed=False,
                    RKNN_full_policy_verified=False,elapsed_seconds=time.time()-started))
                status(label+'_'+panel,'evaluated',successes=panels[panel]['successes'])
        status('pipeline','completed',successes={k:{p:v['successes'] for p,v in r['panels'].items()} for k,r in results.items()})


if __name__=='__main__':main()
