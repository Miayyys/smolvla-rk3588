#!/usr/bin/env python3
"""Generate real OFT teacher chunks in its isolated official runtime environment."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlparse

import numpy as np


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher-repo',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    for field in ('teacher_repo','checkpoint','inputs','output'):
        setattr(args,field,getattr(args,field).resolve())
    if not (args.teacher_repo/'prismatic').is_dir():p.error('Clone the official moojink/openvla-oft source first')
    sys.path.insert(0,str(args.teacher_repo.resolve()))
    missing=[m for m in ('torch','transformers','timm','peft','sentencepiece','tensorflow','json_numpy','diffusers')
             if importlib.util.find_spec(m) is None]
    if missing:raise RuntimeError('Teacher environment is missing: '+', '.join(missing))
    direct=importlib.metadata.distribution('transformers').read_text('direct_url.json') or ''
    fork_verified='transformers-openvla-oft' in direct
    if not fork_verified:
        raise RuntimeError('Use the official bidirectional Transformers fork in an isolated teacher environment')
    os.chdir(args.teacher_repo)  # Official model-code synchronization searches ./prismatic.
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')  # TensorFlow preprocessing must not reserve the teacher VRAM.
    import torch
    from experiments.robot import openvla_utils as utils
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK,PROPRIO_DIM,ACTION_DIM
    if (NUM_ACTIONS_CHUNK,PROPRIO_DIM,ACTION_DIM)!=(8,8,7):raise ValueError('Wrong OFT robot-platform constants')
    print('Teacher dependency/fork/import/constants preflight passed',flush=True)
    if args.preflight_only:return
    meta=json.loads((args.inputs/'manifest.json').read_text())
    if sha(args.inputs/'observations.npz')!=meta['observations_sha256']:raise ValueError('Teacher observations changed')
    if meta.get('source_kind')!='isolated_training_observations' or meta.get('fps')!=10:
        raise ValueError('Not validated isolated 10Hz training inputs')
    if meta.get('image_convention')!='LIBERO_RLDS_training_orientation; no_extra_rotation_then_official_resize_and_crop':
        raise ValueError('Unknown image orientation contract; regenerate teacher inputs')
    if any(r['suite']!='libero_10' for r in meta['rows']):raise ValueError('Teacher checkpoint scope mismatch')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    if args.resume and (args.output/'manifest.json').exists():raise ValueError('Teacher cache already complete; preserve it')
    checkpoint={e.name:sha(e) for e in args.checkpoint.iterdir() if e.is_file()}
    label_identity={'inputs_sha256':sha(args.inputs/'manifest.json'),'teacher_files_sha256':checkpoint,
                    'teacher_source_revision':subprocess.check_output(['git','-C',str(args.teacher_repo),'rev-parse','HEAD'],text=True).strip(),
                    'worker_sha256':sha(Path(__file__).resolve()),
                    'teacher_source_diff':subprocess.check_output(['git','-C',str(args.teacher_repo),'diff'],text=True),
                    'transformers_direct_url':direct}
    actions=[]
    if args.resume:
        progress=json.loads((args.output/'teacher_progress.json').read_text())
        if progress['identity']!=label_identity:raise ValueError('Resume teacher/input/source identity changed')
        partial=np.load(args.output/'actions_partial.npy',allow_pickle=False)
        completed=progress['completed_samples']
        if len(partial)<completed or partial.shape[1:]!=(8,7):raise ValueError('Partial teacher cache is corrupt')
        actions=list(partial[:completed])
    # Official helper rewrites configuration/model-code files. Keep those changes
    # in a private working copy and keep the verified weight files read-only inputs.
    work=args.output/'checkpoint_work';work.mkdir(exist_ok=args.resume)
    for entry in args.checkpoint.iterdir():
        target=work/entry.name
        if target.exists() or target.is_symlink():continue
        if entry.is_dir() or entry.suffix in ('.safetensors','.pt'):target.symlink_to(entry.resolve(),target_is_directory=entry.is_dir())
        else:shutil.copy2(entry,target)
    utils.model_is_on_hf_hub=lambda path:False
    cfg=SimpleNamespace(pretrained_checkpoint=str(work.resolve()),use_l1_regression=True,use_diffusion=False,
                        use_film=False,num_images_in_input=2,use_proprio=True,load_in_8bit=False,load_in_4bit=False,
                        center_crop=True,num_open_loop_steps=8,unnorm_key='libero_10_no_noops',lora_rank=32)
    started=time.time();torch.manual_seed(meta['seed'])
    model=utils.get_vla(cfg);processor=utils.get_processor(cfg)
    head=utils.get_action_head(cfg,llm_dim=model.llm_dim)
    proprio=utils.get_proprio_projector(cfg,llm_dim=model.llm_dim,proprio_dim=8)
    with np.load(args.inputs/'observations.npz',allow_pickle=False) as observations:
        arrays={key:observations[key] for key in ('image','wrist','state')}
        for i,row in enumerate(meta['rows']):
            if i<len(actions):continue
            # Recorded LeRobot/RLDS images already have the training orientation.
            # Only live raw simulator images need the evaluator's 180-degree rotation.
            obs={'full_image':arrays['image'][i].copy(),
                 'wrist_image':arrays['wrist'][i].copy(),'state':arrays['state'][i].copy()}
            with torch.inference_mode():
                values=np.asarray(utils.get_vla_action(cfg,model,processor,obs,row['task'],head,proprio),dtype=np.float32)
            if values.shape!=(8,7) or not np.isfinite(values).all():raise ValueError('Invalid real teacher chunk')
            # Official process_action: [0,1] -> [-1,1], binarize, invert OpenVLA gripper.
            values[:,6]=-np.sign(2*values[:,6]-1)
            actions.append(values)
            temporary=args.output/'actions_partial.tmp.npy'
            np.save(temporary,np.stack(actions));os.replace(temporary,args.output/'actions_partial.npy')
            state={'identity':label_identity,'completed_samples':len(actions),'total_samples':len(meta['rows'])}
            temporary=args.output/'teacher_progress.tmp.json';temporary.write_text(json.dumps(state,indent=2)+'\n')
            os.replace(temporary,args.output/'teacher_progress.json')
            print(json.dumps({'sample':i+1,'total':len(meta['rows']),'seconds':time.time()-started}),flush=True)
    np.save(args.output/'actions.npy',np.stack(actions))
    revision=subprocess.check_output(['git','-C',str(args.teacher_repo),'rev-parse','HEAD'],text=True).strip()
    patch=args.teacher_repo/'qvla_inference_patch.json'
    source_diff=subprocess.check_output(['git','-C',str(args.teacher_repo),'diff'],text=True)
    patch_doc=json.loads(patch.read_text()) if patch.exists() else None
    fork_doc=json.loads(direct);fork_revision=fork_doc.get('vcs_info',{}).get('commit_id')
    fork_url=urlparse(fork_doc['url'])
    if fork_url.scheme=='file':
        fork_root=Path(unquote(fork_url.path))
        fork_revision=subprocess.check_output(['git','-C',str(fork_root),'rev-parse','HEAD'],text=True).strip()
    result={**meta,'source_kind':'openvla_oft_real_inference','teacher_execution_verified':True,
            'teacher_closed_loop_quality_verified':False,'action_space':'libero_simulator_7d',
            'actions_sha256':sha(args.output/'actions.npy'),'teacher_source_revision':revision,
            'teacher_source_diff':source_diff,'inference_import_patch':patch_doc,
            'teacher_utils_sha256':sha(args.teacher_repo/'experiments/robot/openvla_utils.py'),
            'teacher_files_sha256':checkpoint,'transformers_direct_url':fork_doc,'transformers_revision':fork_revision,
            'transformers_version':importlib.metadata.version('transformers'),
            'torch_version':torch.__version__,'tensorflow_version':tf.__version__,
            'teacher_parameter_dtype':str(next(model.parameters()).dtype),
            'peak_cuda_memory_bytes':torch.cuda.max_memory_allocated(),
            'teacher_chunk':8,'teacher_proprio_dim':8,'elapsed_seconds':time.time()-started,
            'alignment_scope':'corresponding_dataset_step_prefix; physical_robot_rate_not_validated'}
    (args.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Real teacher labels saved',flush=True)


if __name__=='__main__':main()
