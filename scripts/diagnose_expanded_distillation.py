#!/usr/bin/env python3
"""Bounded real FP controls: GT vs teacher, and lower LR. Never starts QAT.

Existing long10 cases are reused for diagnosis, not a fresh held-out evaluation.
"""
import fcntl
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/distillation_ablation_v4'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--train-scope',choices=('all_sites','expert_first8','expert_and_interface'),default='all_sites')
    parser.add_argument('--save-trainable-only',action='store_true')
    parser.add_argument('--two-arm-controls',action='store_true')
    args=parser.parse_args();OUT=args.output.resolve()
    OUT.mkdir(exist_ok=True)
    with (OUT / 'pipeline.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = dict(os.environ, MUJOCO_GL='egl', HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', LIBERO_CONFIG_PATH=str(Path.home()/'.libero'),
                   OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1')
        started = time.time()

        def status(stage, state, **extra):
            value = dict(stage=stage, state=state, elapsed_seconds=time.time()-started,
                         pid=os.getpid(), QAT_started=False, **extra)
            write(OUT/'status.json', value)
            print(json.dumps(value), flush=True)

        def run(stage, command):
            status(stage, 'running')
            with (OUT/(stage+'.log')).open('w') as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT)
            if result.returncode:
                status(stage, 'failed', returncode=result.returncode)
                raise RuntimeError('Inspect ' + stage + '.log')
            status(stage, 'completed')

        previous = ROOT/'runs/distill_fp_expanded_v3'
        report = read(previous/'report.json')
        audit = read(ROOT/'runs/teacher_actions_expanded_v3/contract_audit.json')
        metrics = [dict(read(previous/'development_initial.json'), completed_steps=0)] + report['development_history']
        keys = ('completed_steps', 'valid_chunk_mae', 'long10_valid_chunk_mae',
                'other30_valid_chunk_mae', 'gripper_sign_disagreement', 'mae_vs_training_start')
        groups = {}
        for row in audit['rows']:
            groups.setdefault(str(row['task']), []).append(row)
        diagnostic = dict(
            scope='existing_training_and_reused_long10_diagnosis_not_new_heldout_quality',
            prior_development=[{k:m.get(k) for k in keys} for m in metrics],
            prior_long10=read(previous/'long10/paired_summary.json'),
            teacher_contract={k:v for k,v in audit.items() if k!='rows'},
            teacher_by_task={k:dict(samples=len(rows),
                continuous6_mae=sum(r['first6_mae'] for r in rows)/len(rows),
                gripper_agreement=sum(r['gripper_agreement'] for r in rows)/len(rows))
                for k,rows in groups.items()},
            controls='All start original FP; same seed, teacher-frame sampling and task proportions. '
                     '250 updates, accumulation2, constant LR; compare lambda0 vs0.2 at1e-6 '
                     'and lambda0.2 at1e-7. This does not exactly reproduce the v3 cosine run.',
            teacher_contract_filter_is_semantic_validation=False)
        write(OUT/'prior_diagnosis.json', diagnostic)
        reference = {r['task_id']:r for r in read(ROOT/'runs/student_long10_init0_v1/summary.json')['rows']['original_fp']}
        arms = [('gt_lr1e6', 0., 1e-6), ('kd_lr1e6', .2, 1e-6), ('kd_lr1e7', .2, 1e-7)]
        if args.two_arm_controls:arms=arms[:2]
        write(OUT/'protocol.json',dict(training_scope=args.train_scope,arms=arms,steps=250,
                                     gradient_accumulation=2,seed=29,original_FP_start=True,QAT_started=False))
        results = {}; sample_digest = None
        for label, weight, lr in arms:
            train = OUT/label
            if not (train/'report.json').exists():
                if train.exists():
                    raise ValueError('Partial training exists; preserve before restarting: '+str(train))
                command = [sys.executable, 'scripts/qat_train_haq.py',
                    '--model-dir', 'artifacts/model', '--vlm-assets-dir', 'artifacts/smolvlm2_assets',
                    '--dataset-root', 'data/libero', '--splits', 'data/libero_splits.json',
                    '--partition', 'config/evaluation_partition_v2.json',
                    '--candidate', 'config/haq_candidate_v2.json', '--output-dir', str(train),
                    '--mode', 'fp-distill', '--teacher-cache', 'runs/teacher_actions_expanded_v3',
                    '--teacher-review', 'runs/teacher_actions_expanded_v3/review.json',
                    '--require-expanded-teacher', '--teacher-weight', str(weight),
                    '--teacher-task-fraction', '.5', '--steps', '250', '--seed', '29',
                    '--gradient-accumulation', '2', '--learning-rate', str(lr),
                    '--minimum-learning-rate', str(lr), '--lr-schedule', 'constant',
                    '--eval-every', '250', '--train-scope', args.train_scope]
                if args.save_trainable_only:command.append('--save-trainable-only')
                run(label+'_train', command)
            measured = read(train/'report.json')
            if measured['training_scope']['scope']!=args.train_scope or not measured['training_scope']['frozen_parameters_unchanged']:
                raise ValueError('Training scope or frozen parameter integrity failed')
            digest = hashlib.sha256(json.dumps(measured['sample_rows'], sort_keys=True).encode()).hexdigest()
            if sample_digest is not None and digest != sample_digest:
                raise ValueError('Control arms have different frame sampling')
            sample_digest = digest
            if measured['fake_quant_enabled'] or not measured['strict_master_reload']:
                raise ValueError('Expected strictly reloaded FP controls')
            rollout = train/'long10'
            if not (rollout/'eval_info.json').exists():
                if rollout.exists():raise ValueError('Partial rollout exists: '+str(rollout))
                run(label+'_long10', [sys.executable, 'scripts/eval_distill_qat_libero.py',
                    '--qvla-panel', 'long10', '--qvla-mode', 'distilled_fp', '--qvla-run', str(train),
                    '--qvla-candidate', 'config/haq_candidate_v2.json',
                    '--qvla-vlm-assets-dir', 'artifacts/smolvlm2_assets',
                    '--qvla-reset-audit', str(rollout/'reset_audit.json'), '--qvla-init-state', '0',
                    '--policy.path', 'artifacts/model', '--env.type', 'libero', '--env.task', 'libero_10',
                    '--env.task_ids', '[0,1,2,3,4,5,6,7,8,9]', '--env.observation_height', '256',
                    '--env.observation_width', '256', '--eval.n_episodes', '1', '--eval.batch_size', '1',
                    '--eval.use_async_envs', 'false', '--seed', '0', '--output_dir', str(rollout)])
            reset = {r['task_id']:r for r in read(rollout/'reset_audit.json') if r['reset_number']==1}
            rows = []
            for task in read(rollout/'eval_info.json')['per_task']:
                i = int(task['task_id'])
                for key in ('init_state_sha256', 'initial_camera_sha256', 'policy_noise_seed', 'env_seed'):
                    if reset[i][key] != reference[i][key]:raise ValueError('Pairing failed: '+key)
                rows.append(dict(task_id=i, success=bool(task['metrics']['successes'][0]),
                                 original_fp_success=reference[i]['success']))
            if sorted(r['task_id'] for r in rows)!=list(range(10)):raise ValueError('Missing tasks')
            results[label] = dict(teacher_weight=weight, learning_rate=lr, steps=250,
                successes=sum(r['success'] for r in rows), rows=rows,
                sample_rows_sha256=digest, teacher_supervised_microbatches=measured['teacher_supervised_steps'],
                training_scope=measured['training_scope'],
                development={k:v for k,v in measured['development_history'][-1].items()
                             if k not in ('rows', 'per_task_mae')}, master_sha256=measured['master_sha256'])
            write(OUT/'results.json', dict(scope=diagnostic['scope'], arms=results,
                original_fp_successes=7, paired_state_images_noise_verified=True,
                other_suite_closed_loop='not_measured', QAT_started=False))
            status(label, 'evaluated', successes=results[label]['successes'])
        status('pipeline', 'completed', successes={k:v['successes'] for k,v in results.items()})


if __name__=='__main__':
    main()
