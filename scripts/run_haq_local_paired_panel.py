#!/usr/bin/env python3
"""Run and summarize a small, fixed, paired LIBERO development panel."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'runs/haq_real_feedback_local_v3')
    p.add_argument('--output',type=Path,default=ROOT/'runs/haq_libero_panel_v1')
    p.add_argument('--model-dir',type=Path,default=ROOT/'artifacts/transfer/model')
    p.add_argument('--vlm-assets-dir',type=Path,default=ROOT/'artifacts/transfer/smolvlm2_assets')
    p.add_argument('--fp-reference-output',type=Path,
                   help='Completed paired FP panel with the same task IDs, seed, image size and environment')
    p.add_argument('--task-ids',type=int,nargs='+',default=[0,3,6])
    p.add_argument('--seed',type=int,default=0)
    args=p.parse_args()
    if not args.task_ids or len(set(args.task_ids))!=len(args.task_ids) or any(t<0 or t>9 for t in args.task_ids):
        p.error('Provide unique LIBERO task IDs in 0..9')
    if args.fp_reference_output is not None:
        reference=json.loads((args.fp_reference_output/'summary.json').read_text())
        if reference['seed']!=args.seed or reference['task_ids']!=args.task_ids:
            p.error('FP reference must use identical seed and task IDs')
    if (args.output/'summary.json').exists():
        existing=json.loads((args.output/'summary.json').read_text())
        if existing['seed']!=args.seed or existing['task_ids']!=args.task_ids or existing['source_run']!=str(args.run):
            p.error('Existing output belongs to another experiment')
    args.output.mkdir(parents=True,exist_ok=True)
    record_file=args.output/'progress.json'
    progress=json.loads(record_file.read_text()) if record_file.exists() else []
    env=dict(os.environ,LIBERO_CONFIG_PATH=str((ROOT/'runs/libero_local/config').resolve()),
             MPLCONFIGDIR=str((ROOT/'runs/libero_local/mpl').resolve()),
             MUJOCO_GL='egl',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1')
    for suite in SUITES:
        for mode in ('fp','best'):
            if mode=='fp' and args.fp_reference_output is not None:
                result=args.fp_reference_output/f'fp_{suite}'/'eval_info.json'
                if not result.is_file():raise FileNotFoundError(result)
                print(f'Using paired FP result {result}',flush=True)
                continue
            out=args.output/f'{mode}_{suite}'
            out.mkdir(exist_ok=True)
            result=out/'eval_info.json'
            if result.is_file():
                print(f'{mode}/{suite} resumed',flush=True)
                continue
            cmd=[sys.executable,str(ROOT/'scripts/eval_haq_local_libero.py'),
                 '--qvla-run',str(args.run),'--qvla-mode',mode,
                 '--qvla-model-dir',str(args.model_dir),'--qvla-vlm-assets-dir',str(args.vlm_assets_dir),
                 '--policy.path',str(args.model_dir),'--env.type','libero',
                 '--env.task',suite,'--env.task_ids',json.dumps(args.task_ids),
                 '--env.observation_height','256','--env.observation_width','256',
                 '--eval.n_episodes','1','--eval.batch_size','1','--seed',str(args.seed),
                 '--output_dir',str(out)]
            start=time.time()
            print(f'Running {mode}/{suite}: {args.task_ids}',flush=True)
            with (out/'rollout.log').open('w') as log:
                done=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
            row={'mode':mode,'suite':suite,'task_ids':args.task_ids,
                 'returncode':done.returncode,'seconds':time.time()-start,
                 'result_present':result.is_file(),'command':cmd}
            progress.append(row)
            record_file.write_text(json.dumps(progress,indent=2)+'\n')
            print(f"{mode}/{suite}: code={done.returncode} result={row['result_present']} time={row['seconds']:.1f}s",flush=True)
            if done.returncode or not result.is_file():
                raise RuntimeError(f'Rollout failed; see {out/"rollout.log"}')
    pairs=[]
    for suite in SUITES:
        by_mode={}
        for mode in ('fp','best'):
            source=(args.fp_reference_output if mode=='fp' and args.fp_reference_output is not None
                    else args.output)
            raw=json.loads((source/f'{mode}_{suite}'/'eval_info.json').read_text())
            rows=raw['per_task']
            by_mode[mode]={int(r['task_id']):bool(r['metrics']['successes'][0])
                           for r in rows}
            if set(by_mode[mode])!=set(args.task_ids):raise ValueError('Unexpected task IDs in result')
        for task in args.task_ids:
            pairs.append({'suite':suite,'task_id':task,
                          'fp':by_mode['fp'][task],'best':by_mode['best'][task]})
    summary={'scope':'single_seed_local_libero_development_panel_not_final_quality',
             'seed':args.seed,'task_ids':args.task_ids,'pairs':pairs,
             'successes':{m:sum(r[m] for r in pairs) for m in ('fp','best')},
             'improved':sum(r['best'] and not r['fp'] for r in pairs),
             'worsened':sum(r['fp'] and not r['best'] for r in pairs),
             'source_run':str(args.run),'success_denominator':len(pairs),
             'fp_reference_output':str(args.fp_reference_output) if args.fp_reference_output else None,
             'warning':'one initial state per task; descriptive validation of offline proxy only'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
