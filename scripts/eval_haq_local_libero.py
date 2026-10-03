#!/usr/bin/env python3
"""Small paired LIBERO rollout for a locally packed HAQ diagnostic candidate."""
import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.scripts import lerobot_eval

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qvla_haq.offline_actions import file_sha256
from qvla_haq.runtime import apply_assignment


def local_args():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--qvla-run',type=Path,required=True)
    parser.add_argument('--qvla-mode',choices=('fp','best'),required=True)
    parser.add_argument('--qvla-model-dir',type=Path,required=True)
    parser.add_argument('--qvla-vlm-assets-dir',type=Path,required=True)
    ours,rest=parser.parse_known_args()
    sys.argv=[sys.argv[0]]+rest
    for i,token in enumerate(rest):
        if token=='--output_dir' and i+1<len(rest):Path(rest[i+1]).mkdir(parents=True,exist_ok=True)
        elif token.startswith('--output_dir='):Path(token.split('=',1)[1]).mkdir(parents=True,exist_ok=True)
    return ours


def main():
    args=local_args()
    summary=json.loads((args.qvla_run/'summary.json').read_text())
    source=args.qvla_model_dir/'model.safetensors'
    if file_sha256(source)!=summary['identity']['checkpoint_sha256']:
        raise ValueError('Original FP checkpoint changed')
    space=json.loads((args.qvla_run/'space.json').read_text())
    stats=json.loads((args.qvla_run/'calibration.json').read_text())['ranges']
    selected=None
    if args.qvla_mode=='best':
        best=summary['best_by_proxy']
        if best is None or not best['feasible_40pct']:
            raise ValueError('No feasible diagnostic candidate')
        selected=args.qvla_run/f"round{best['round']:03d}_candidate{best['candidate']:02d}"
        checkpoint=selected/'model.safetensors'
        if file_sha256(checkpoint)!=best['checkpoint_sha256'] or checkpoint.stat().st_size!=best['actual_model_bytes']:
            raise ValueError('Selected quantized checkpoint hash/size mismatch')

    def make_policy(*,cfg,env_cfg,rename_map):
        if cfg.type!='smolvla' or Path(cfg.pretrained_path).resolve()!=args.qvla_model_dir.resolve():
            raise ValueError('Unexpected evaluation model')
        cfg.vlm_model_name=str(args.qvla_vlm_assets_dir.resolve())
        cfg.load_vlm_weights=False
        cfg.device='cuda'
        policy=SmolVLAPolicy.from_pretrained(cfg.pretrained_path,config=cfg,strict=True)
        if selected is not None:
            assignment=json.loads((selected/'assignment.json').read_text())
            originals={s['module']:copy.deepcopy(policy.get_submodule(s['module'])).cpu()
                       for s in space['action_sites']}
            apply_assignment(policy,space,assignment,stats,originals)
            policy.load_state_dict(load_file(str(checkpoint)),strict=True)
            for name in assignment:policy.get_submodule(name).refresh()
            print(f"Strictly loaded {len(assignment)} precision choices from {checkpoint}",flush=True)
        return policy.to('cuda').eval()

    original_make_processors=lerobot_eval.make_pre_post_processors
    original_eval_all=lerobot_eval.eval_policy_all
    original_run_one=lerobot_eval.run_one
    asset_path=str(args.qvla_vlm_assets_dir.resolve())

    def processors(*a,**kw):
        overrides=dict(kw.get('preprocessor_overrides') or {})
        tokenizer=dict(overrides.get('tokenizer_processor') or {})
        tokenizer['tokenizer_name']=asset_path
        overrides['tokenizer_processor']=tokenizer
        overrides['rename_observations_processor']={'rename_map':{
            'observation.images.image':'observation.images.camera1',
            'observation.images.image2':'observation.images.camera2'}}
        kw['preprocessor_overrides']=overrides
        return original_make_processors(*a,**kw)

    def no_video(*a,**kw):
        kw['max_episodes_rendered']=0
        kw['videos_dir']=None
        return original_eval_all(*a,**kw)

    def paired_noise(*a,**kw):
        group=kw.get('task_group',a[0] if a else None)
        task=kw.get('task_id',a[1] if len(a)>1 else None)
        groups=('libero_spatial','libero_object','libero_goal','libero_10')
        if group not in groups or task is None:raise ValueError('Unexpected LIBERO task')
        seed=int(kw.get('start_seed') or 0)+100000*(groups.index(group)+1)+int(task)
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        print(f'Paired task noise {group}/{task}: {seed}',flush=True)
        return original_run_one(*a,**kw)

    lerobot_eval.make_policy=make_policy
    lerobot_eval.make_pre_post_processors=processors
    lerobot_eval.eval_policy_all=no_video
    lerobot_eval.run_one=paired_noise
    lerobot_eval.main()


if __name__=='__main__':main()
