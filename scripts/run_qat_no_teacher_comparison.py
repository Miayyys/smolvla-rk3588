#!/usr/bin/env python3
"""Only two new no-teacher QAT runs, then compare all four fixed candidates."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')


def main():
    out=ROOT/'runs/qat_four_way_v1';out.mkdir(exist_ok=True)
    with (out/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.time();env=dict(os.environ,PYTHONUNBUFFERED='1')
        def status(stage,state,**extra):
            v=dict(stage=stage,state=state,pid=os.getpid(),elapsed_seconds=time.time()-started,**extra)
            write(out/'status.json',v);print(json.dumps(v),flush=True)
        for version in ('v1','v2'):
            run=ROOT/f'runs/qat_distilled_{version}_no_teacher_v1'
            control=Path(str(run)+'_pipeline')
            if not (control/'summary.json').exists() or read(control/'status.json')['state']!='completed':
                status(version+'_no_teacher','running')
                cmd=[sys.executable,'-u','scripts/run_first8_v2_qat.py','--config',
                     f'config/qat_distilled_{version}_no_teacher_v1.json']
                with (out/(version+'_no_teacher.log')).open('w') as log:
                    done=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                if done.returncode:
                    status(version+'_no_teacher','failed',returncode=done.returncode)
                    raise RuntimeError('Inspect inner pipeline status/log')
            status(version+'_no_teacher','completed')
        rows=[];reference_sample_hash=None
        for version in ('v1','v2'):
            for teacher in (True,False):
                path=ROOT/(f'runs/qat_distilled_full_{version}_v1' if teacher else
                           f'runs/qat_distilled_{version}_no_teacher_v1')
                r=read(path/'report.json');measured=read(Path(str(path)+'_pipeline')/'summary.json')
                cfg=read(ROOT/(f'config/qat_distilled_full_{version}_v1.json' if teacher else
                               f'config/qat_distilled_{version}_no_teacher_v1.json'))
                if r['teacher_loss_used']!=teacher or r['teacher_source_kind']!='openvla_oft_real_inference':
                    raise ValueError('Incorrect teacher loss or distilled student provenance')
                if r['initial_master_sha256']!=cfg['student_master_sha256'] or not r['strict_pack_reload']:
                    raise ValueError('Student source or pack check failed')
                digest=hashlib.sha256(json.dumps(r['sample_rows'],sort_keys=True).encode()).hexdigest()
                if reference_sample_hash is not None and digest!=reference_sample_hash:raise ValueError('Frame sampling differs')
                reference_sample_hash=digest
                outcomes=[x for panel in measured['results'].values() for x in panel['rows']]
                if len(outcomes)!=19:raise ValueError('Missing common tasks')
                rows.append(dict(label=version+('_with_teacher' if teacher else '_without_teacher'),
                    version=version,teacher_loss=teacher,student_run=cfg['student_run'],
                    long10_successes=measured['results']['long10']['successes'],
                    other9_successes=measured['results']['other9']['successes'],
                    total19_successes=sum(x['success'] for x in outcomes),
                    lost_from_student=[dict(suite=x['suite'],task_id=x['task_id']) for x in outcomes if x['student_FP_success'] and not x['success']],
                    gained_from_student=[dict(suite=x['suite'],task_id=x['task_id']) for x in outcomes if not x['student_FP_success'] and x['success']],
                    packed_file_bytes=r['packed_file_bytes'],compression=r['packed_compression_fraction'],
                    size_constraint_met=r['packed_compression_fraction']>=.4,
                    reload_parity=r['reload_action_parity'],packed_sha256=r['packed_sha256'],
                    elapsed_training_and_pack_seconds=r['elapsed_seconds'],
                    weight_path=str(path/'qat_local_packed.safetensors'),sample_rows_sha256=digest,
                    outcomes=outcomes))
        eligible=[r for r in rows if r['size_constraint_met']]
        ordered=sorted(eligible,key=lambda r:(-r['total19_successes'],-r['long10_successes'],r['packed_file_bytes']))
        top=ordered[0] if ordered else None
        tied=[r['label'] for r in eligible if top and (r['total19_successes'],r['long10_successes'],r['packed_file_bytes'])==
              (top['total19_successes'],top['long10_successes'],top['packed_file_bytes'])]
        report=dict(scope='four_QAT_variants_fixed19_diagnostic_not_final_heldout_or_RKNN',
            candidates=rows,ranking_rule='size decrease>=40%; more task successes; then more Long10 successes; then smaller file',
            provisional_preferred_labels=tied,original_FP_and_student_FP_total19=16,
            all_frame_sampling_identical=True,elapsed_seconds=time.time()-started,
            speed_benefits='not_measured; training_duration_is_not_inference_latency',
            limitations='One training seed and one initial state per task; previously reused diagnostic panel. '
                        'Four-way winner does not establish general superiority or distillation improvement.')
        write(out/'comparison.json',report)
        status('pipeline','completed',successes={r['label']:r['total19_successes'] for r in rows},
               provisional_preferred_labels=tied)


if __name__=='__main__':main()
