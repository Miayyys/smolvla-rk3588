#!/usr/bin/env python3
"""Five predefined LIBERO cases across six student variants (30 total episodes)."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    arms=[('original_fp','original_fp',None),('original_v2','original_v2',None),
        ('fp_lr1e5','distilled_fp','runs/distill_fp_real40_v1'),('qat_lr1e5','distilled_v2_qat','runs/distill_qat_real40_v1'),
        ('fp_lr1e7','distilled_fp','runs/distill_fp_real40_lr1e7_v1'),('qat_lr1e7','distilled_v2_qat','runs/distill_qat_real40_lr1e7_v1')]
    env=dict(os.environ,MUJOCO_GL='egl',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',
        LIBERO_CONFIG_PATH=str(Path.home()/'.libero'),MPLCONFIGDIR='/tmp/qvla-mpl')
    results={};started=time.time();progress=[]
    for label,mode,run in arms:
        out=args.output/label;out.mkdir()
        cmd=[sys.executable,'qvla/evaluation/eval_distill_qat_libero.py','--qvla-mode',mode,
            '--qvla-candidate','config/haq_candidate_v2.json','--qvla-vlm-assets-dir','artifacts/smolvlm2_assets',
            '--qvla-reset-audit',str(out/'reset_audit.json'),'--qvla-init-state','0',
            '--policy.path','artifacts/model','--env.type','libero',
            '--env.task','libero_spatial,libero_object,libero_goal,libero_10','--env.task_ids','[0,3]',
            '--env.observation_height','256','--env.observation_width','256','--eval.n_episodes','1',
            '--eval.batch_size','1','--eval.use_async_envs','false','--seed','0','--output_dir',str(out)]
        if run:cmd.extend(['--qvla-run',run])
        tick=time.time();print('Starting '+label,flush=True)
        with (out/'rollout.log').open('w') as log:done=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        progress.append({'arm':label,'returncode':done.returncode,'seconds':time.time()-tick,'command':cmd})
        (args.output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
        if done.returncode:raise RuntimeError('Rollout failed: '+str(out/'rollout.log'))
        raw=json.loads((out/'eval_info.json').read_text());audit=json.loads((out/'reset_audit.json').read_text())
        first={(r['suite'],r['task_id']):r for r in audit if r['reset_number']==1}
        rows=[]
        for task in raw['per_task']:
            key=(task['task_group'],int(task['task_id']))
            if key not in first:raise ValueError('No reset audit for rollout task')
            rows.append({'suite':key[0],'task_id':key[1],'success':bool(task['metrics']['successes'][0]),
                         'init_state_sha256':first[key]['init_state_sha256'],
                         'initial_camera_sha256':first[key]['initial_camera_sha256'],
                         'policy_noise_seed':first[key]['policy_noise_seed']})
        if len(rows)!=5:raise ValueError('Expected five predefined cases')
        results[label]=rows;print(json.dumps({'arm':label,'successes':sum(r['success'] for r in rows),'episodes':len(rows)}),flush=True)
    reference={(r['suite'],r['task_id']):r for r in results['original_fp']}
    for rows in results.values():
        for row in rows:
            base=reference[(row['suite'],row['task_id'])]
            for k in ('init_state_sha256','initial_camera_sha256','policy_noise_seed'):
                if row[k]!=base[k]:raise ValueError('Unpaired initial state/image/noise: '+k)
    report={'scope':'five_case_single_initial_state_paired_closed_loop_diagnostic_not_final_quality',
        'total_episodes':30,'paired_initial_state_images_and_noise_verified':True,
        'successes':{arm:sum(r['success'] for r in rows) for arm,rows in results.items()},
        'rows':results,'elapsed_seconds':time.time()-started,
        'limitation':'single initial state and small predefined panel; no final hyperparameter selection'}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
