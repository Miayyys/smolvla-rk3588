#!/usr/bin/env python3
"""Fixed distilled student -> configured HAQ-map QAT -> local pack -> 19 tasks."""
import fcntl
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/qat_first8_v2_v1.json')
    args=parser.parse_args()
    cfg=read(args.config);out=ROOT/cfg['output']
    control=ROOT/(cfg['output']+'_pipeline');control.mkdir(exist_ok=True)
    with (control/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=dict(os.environ,MUJOCO_GL='egl',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            OMP_NUM_THREADS='1',LIBERO_CONFIG_PATH=str(Path.home()/'.libero'))
        started=time.time()
        def status(stage,state,**extra):
            value=dict(stage=stage,state=state,pid=os.getpid(),elapsed_seconds=time.time()-started,**extra)
            write(control/'status.json',value);print(json.dumps(value),flush=True)
        def run(stage,cmd):
            status(stage,'running')
            write(control/(stage+'_command.json'),cmd)
            with (control/(stage+'.log')).open('w') as log:
                done=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            if done.returncode:
                status(stage,'failed',returncode=done.returncode);raise RuntimeError('Inspect '+stage+'.log')
            status(stage,'completed')
        student=ROOT/cfg['student_run'];master=student/'distilled_float_master.safetensors'
        fp=read(student/'report.json')
        if sha(master)!=cfg['student_master_sha256'] or fp['master_sha256']!=cfg['student_master_sha256']:
            raise ValueError('Selected FP student changed')
        if not fp['training_scope']['frozen_parameters_unchanged']:raise ValueError('FP scope integrity failed')
        cache=ROOT/cfg['teacher_cache'];audit=read(cache/'contract_audit.json')
        if not audit['teacher_execution_verified'] or audit['teacher_manifest_sha256']!=sha(cache/'manifest.json'):
            raise ValueError('Exact teacher cache not audited')
        write(control/'protocol.json',cfg)
        if not (out/'report.json').exists():
            if out.exists():raise ValueError('Partial QAT exists; explicit resume required to preserve it')
            cmd=[sys.executable,'scripts/qat_train_haq.py','--mode','qat',
                '--model-dir','artifacts/model','--vlm-assets-dir','artifacts/smolvlm2_assets',
                '--dataset-root','data/libero','--splits','data/libero_splits.json',
                '--partition','config/evaluation_partition_v2.json','--candidate',cfg['candidate'],
                '--initial-master',str(master),'--output-dir',str(out),
                '--teacher-cache',cfg['teacher_cache'],'--teacher-review',cfg['teacher_review'],
                '--require-expanded-teacher','--allow-unverified-backend-diagnostic']
            for key in ('train_scope','steps','gradient_accumulation','learning_rate','minimum_learning_rate',
                        'lr_schedule','warmup_ratio','teacher_weight','teacher_task_fraction','seed',
                        'eval_every','save_every','keep_snapshots'):
                cmd.extend(['--'+key.replace('_','-'),str(cfg[key])])
            run('qat',cmd)
        trained=read(out/'report.json')
        if trained['candidate_config_sha256']!=sha(ROOT/cfg['candidate']):
            raise ValueError('QAT result uses a different candidate map')
        if not trained['strict_pack_reload'] or not trained['training_scope']['frozen_parameters_unchanged']:
            raise ValueError('QAT pack or frozen parameter verification failed')
        old_long=read(ROOT/'runs/student_long10_init0_v1/summary.json')['rows']['original_fp']
        old_other=read(ROOT/'runs/freezing_other9_v1/summary.json')['results']['original_fp']['rows']
        refs={(r['suite'],r['task_id']):r for r in old_long+old_other}
        student_long=read(student.parent/'results.json')['arms'][student.name]['rows']
        student_other=read(ROOT/'runs/freezing_other9_v1/summary.json')['results']['first8_kd']['rows']
        student_refs={('libero_10',r['task_id']):r for r in student_long}
        student_refs.update({(r['suite'],r['task_id']):r for r in student_other})
        results={}
        for panel in cfg['closed_loop_panels']:
            dest=out/panel
            if not (dest/'eval_info.json').exists():
                if dest.exists():raise ValueError('Partial eval exists: '+str(dest))
                suites='libero_10' if panel=='long10' else 'libero_spatial,libero_object,libero_goal'
                tasks='[0,1,2,3,4,5,6,7,8,9]' if panel=='long10' else '[0,4,8]'
                cmd=[sys.executable,'scripts/eval_distill_qat_libero.py','--qvla-panel',panel,
                    '--qvla-mode','distilled_v2_qat','--qvla-run',str(out),'--qvla-candidate',cfg['candidate'],
                    '--qvla-vlm-assets-dir','artifacts/smolvlm2_assets',
                    '--qvla-reset-audit',str(dest/'reset_audit.json'),
                    '--qvla-action-trace',str(dest/'action_trace.json'),'--qvla-init-state','0',
                    '--policy.path','artifacts/model','--env.type','libero','--env.task',suites,
                    '--env.task_ids',tasks,'--env.observation_height','256','--env.observation_width','256',
                    '--eval.n_episodes','1','--eval.batch_size','1','--eval.use_async_envs','false',
                    '--seed','0','--output_dir',str(dest)]
                run(panel,cmd)
            audits={(r['suite'],r['task_id']):r for r in read(dest/'reset_audit.json') if r['reset_number']==1}
            rows=[]
            for task in read(dest/'eval_info.json')['per_task']:
                key=(task['task_group'],int(task['task_id']));a=audits[key];base=refs[key]
                for field in ('init_state_sha256','initial_camera_sha256','policy_noise_seed','env_seed'):
                    if a[field]!=base[field]:raise ValueError('QAT/FP pairing failed: '+field)
                rows.append(dict(suite=key[0],task_id=key[1],success=bool(task['metrics']['successes'][0]),
                                 original_FP_success=base['success'],student_FP_success=student_refs[key]['success']))
            expected=10 if panel=='long10' else 9
            if len(rows)!=expected:raise ValueError('Incomplete panel')
            results[panel]=dict(successes=sum(r['success'] for r in rows),rows=rows)
            write(control/'summary.json',dict(scope='real_local_QAT_pack_19_task_diagnostic_not_RKNN',
                candidate_version=trained['candidate_version'],candidate=cfg['candidate'],
                student_master_sha256=cfg['student_master_sha256'],
                results=results,packed_file_bytes=trained['packed_file_bytes'],
                compression=trained['packed_compression_fraction'],reload_parity=trained['reload_action_parity'],
                RKNN_full_policy_verified=False,independent_original_FP_PTQ='pending',
                elapsed_seconds=time.time()-started))
        status('pipeline','completed',successes={k:v['successes'] for k,v in results.items()})


if __name__=='__main__':main()
