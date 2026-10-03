#!/usr/bin/env python3
"""Inject two real RKNN vision outputs into one fixed original-FP action."""

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
from pathlib import Path
import numpy as np
import torch
from qvla.evaluation.haq_offline_eval import identity, load_policy, predict_cached
from qvla.haq.offline_actions import file_sha256, load_cache
from qvla.evaluation.verify_smolvla_vision_float import metrics

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,default=ROOT/'artifacts/transfer/model')
    p.add_argument('--vlm-assets-dir',type=Path,default=ROOT/'artifacts/transfer/smolvlm2_assets')
    p.add_argument('--splits',type=Path,default=ROOT/'data/libero_splits.json')
    p.add_argument('--partition',type=Path,default=ROOT/'config/evaluation_partition_v2.json')
    p.add_argument('--cache',type=Path,default=ROOT/'runs/haq_offline_local_v1/fp_cache40')
    p.add_argument('--vision-root',type=Path,default=ROOT/'runs/smolvla_vision_split_v1')
    p.add_argument('--output-dir',type=Path,default=ROOT/'runs/smolvla_board_vision_action_v1')
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    torch.set_num_threads(4)
    ident=identity(args)
    manifest, observations, reference=load_cache(args.cache,ident)
    contract=json.loads((ROOT/'runs/smolvla_split_contract_v1.json').read_text())
    if manifest['samples'][0] != contract['sample'] or ident['checkpoint_sha256'] != contract['checkpoint_sha256']:
        raise ValueError('Pinned sample identity differs')
    if torch.__version__ != manifest['torch_version'] or args.device != manifest['device']:
        raise ValueError('FP cache runtime mismatch')
    chosen={k:v[:1] for k,v in observations.items()}
    samples=manifest['samples'][:1]
    policy,pre,post=load_policy(args)
    original,timing=predict_cached(policy,pre,post,chosen,samples,args.device)
    np.testing.assert_array_equal(original,reference[:1])
    outputs=[np.load(args.vision_root/name,allow_pickle=False) for name in
             ('board_output_nhwc.npy','board_output_camera1.npy')]
    pixels=[np.load(args.vision_root/f'vision_input_{i}.npy',allow_pickle=False) for i in (0,1)]
    for i in (0,1):
        board_report=json.loads((args.vision_root/('board_report_nhwc.json' if i==0 else 'board_report_camera1.json')).read_text())
        if board_report['status']!='success' or list(outputs[i].shape)!=board_report['output_shape']:
            raise ValueError('Board result is not a successful matching output')
    count=0
    def board_embed(image):
        nonlocal count
        if count>=2:
            raise ValueError('Unexpected extra image call')
        np.testing.assert_array_equal(image.detach().float().cpu().numpy(),pixels[count])
        out=torch.from_numpy(outputs[count]).to(device=image.device,dtype=torch.float32)
        count+=1
        return out
    module=policy.model.vlm_with_expert
    old=module.embed_image
    module.embed_image=board_embed
    try:
        candidate,host_timing=predict_cached(policy,pre,post,chosen,samples,args.device)
        if count!=2 or not np.isfinite(candidate).all():
            raise ValueError('Wrong image call count or non-finite action')
    finally:
        module.embed_image=old
    args.output_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output_dir/'actions.npz',original=original,candidate=candidate)
    report={'scope':'one fixed action, real RKNN vision features + original GPU prefix/expert',
            'sample':samples[0],'checkpoint_sha256':ident['checkpoint_sha256'],
            'fp_cache_exact_match':True,'board_camera_calls':count,
            'vision_output_sha256':[file_sha256(args.vision_root/name) for name in
                                   ('board_output_nhwc.npy','board_output_camera1.npy')],
            'action_shape':list(candidate.shape),'action_vs_fp':metrics(candidate,original),
            'per_action_dimension_mae':np.abs(candidate-original).mean(axis=(0,1)).tolist(),
            'closed_loop_quality':'not_measured','full_board_execution':'not_measured',
            'host_timing':host_timing}
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
