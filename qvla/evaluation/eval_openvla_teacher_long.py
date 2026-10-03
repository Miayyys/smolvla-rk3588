#!/usr/bin/env python3
"""Real OFT teacher: ten LIBERO-Long tasks, one predefined initial state each."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher-repo',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--simulation-site',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    for name in ('teacher_repo','checkpoint','simulation_site','output'):
        setattr(args,name,getattr(args,name).resolve())
    # Teacher packages have priority; only missing simulator packages fall back.
    sys.path.append(str(args.simulation_site));sys.path.insert(0,str(args.teacher_repo))
    direct=importlib.metadata.distribution('transformers').read_text('direct_url.json') or ''
    if 'transformers-openvla-oft' not in direct:raise ValueError('Official teacher fork required')
    args.output.mkdir(parents=True,exist_ok=False)
    work=args.output/'checkpoint_work';work.mkdir()
    hashes={}
    for entry in args.checkpoint.iterdir():
        target=work/entry.name
        if entry.is_dir() or entry.suffix in ('.safetensors','.pt'):
            target.symlink_to(entry.resolve(),target_is_directory=entry.is_dir())
        else:shutil.copy2(entry,target)
        if entry.is_file():hashes[entry.name]=sha(entry)
    os.chdir(args.teacher_repo)
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    import numpy as np
    import torch
    from libero.libero import benchmark
    from experiments.robot import openvla_utils as utils
    from experiments.robot.libero import run_libero_eval as official
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK,PROPRIO_DIM,ACTION_DIM
    if (NUM_ACTIONS_CHUNK,PROPRIO_DIM,ACTION_DIM)!=(8,8,7):raise ValueError('Wrong teacher constants')
    utils.model_is_on_hf_hub=lambda path:False
    cfg=official.GenerateConfig(pretrained_checkpoint=str(work),task_suite_name='libero_10',
        num_trials_per_task=1,seed=0,use_wandb=False)
    official.set_seed_everywhere(0)
    started=time.time()
    model,head,proprio,noisy,processor=official.initialize_model(cfg)
    resize=official.get_image_resize_size(cfg)
    suite=benchmark.get_benchmark_dict()['libero_10']();rows=[]
    for task_id in range(suite.n_tasks):
        official.set_seed_everywhere(400000+task_id)
        task=suite.get_task(task_id);state=np.asarray(suite.get_task_init_states(task_id)[0])
        env,description=official.get_libero_env(task,cfg.model_family,resolution=256)
        step=env.step;counter=[0];audit={}
        def audited_step(action):
            obs,reward,done,info=step(action);counter[0]+=1
            if counter[0]==cfg.num_steps_wait:
                audit.update(initial_camera_sha256={
                    'image':hashlib.sha256(np.ascontiguousarray(obs['agentview_image']).tobytes()).hexdigest(),
                    'image2':hashlib.sha256(np.ascontiguousarray(obs['robot0_eye_in_hand_image']).tobytes()).hexdigest()})
            return obs,reward,done,info
        env.step=audited_step;tick=time.time()
        with (args.output/f'task_{task_id}.log').open('w+') as log:
            with torch.inference_mode():
                success,_=official.run_episode(cfg,env,description,model,resize,processor,head,proprio,noisy,state,log)
            log.seek(0);error='Episode error:' in log.read()
        env.close()
        row={'suite':'libero_10','task_id':task_id,'task':description,'success':bool(success),
             'runtime_error':error,'initial_state_index':0,'env_seed':0,'policy_seed':400000+task_id,
             'init_state_sha256':hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest(),
             **audit,'steps':max(0,counter[0]-cfg.num_steps_wait),'seconds':time.time()-tick}
        rows.append(row);(args.output/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps(row),flush=True)
        if error:raise RuntimeError('Teacher episode runtime error; not a quality failure')
    report={'scope':'teacher_LIBERO_Long_10tasks_single_initial_state_screen_not_paper_success_rate',
            'episodes':len(rows),'successes':sum(r['success'] for r in rows),'rows':rows,
            'elapsed_seconds':time.time()-started,'teacher_checkpoint_root_sha256':hashes,
            'teacher_source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'teacher_source_diff':subprocess.check_output(['git','diff'],text=True),
            'transformers_direct_url':json.loads(direct),'torch_version':torch.__version__,
            'peak_cuda_memory_bytes':torch.cuda.max_memory_allocated(),
            'configuration':vars(cfg),'whole_long_task_improvement_from_distillation':'not_established'}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'completed':len(rows),'successes':report['successes']}),flush=True)

if __name__=='__main__':main()
