#!/usr/bin/env python3
"""Capture/evaluate real RKNN vision features on the pinned 40-observation panel."""

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
from qvla.haq.offline_actions import file_sha256, load_cache, score_actions

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('capture','evaluate'))
    p.add_argument('--model-dir',type=Path,default=ROOT/'artifacts/transfer/model')
    p.add_argument('--vlm-assets-dir',type=Path,default=ROOT/'artifacts/transfer/smolvlm2_assets')
    p.add_argument('--splits',type=Path,default=ROOT/'data/libero_splits.json')
    p.add_argument('--partition',type=Path,default=ROOT/'config/evaluation_partition_v2.json')
    p.add_argument('--cache',type=Path,default=ROOT/'runs/haq_offline_local_v1/fp_cache40')
    p.add_argument('--root',type=Path,default=ROOT/'runs/smolvla_board_vision_panel_v1')
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    torch.set_num_threads(4)
    ident=identity(args)
    manifest,observations,fp=load_cache(args.cache,ident)
    if torch.__version__!=manifest['torch_version'] or args.device!=manifest['device']:
        raise ValueError('FP cache runtime mismatch')
    if len(manifest['samples'])!=40:
        raise ValueError('Expected exactly 40 pinned development observations')
    args.root.mkdir(parents=True,exist_ok=True)
    policy,pre,post=load_policy(args)
    module=policy.model.vlm_with_expert
    if args.mode=='capture':
        pixels=[];features=[]
        def image_hook(_,inputs,kwargs,output):
            image=inputs[0] if inputs else kwargs['pixel_values']
            pixels.append(image.detach().float().cpu().numpy().copy())
        def feature_hook(_,inputs,output):
            features.append(output.detach().float().cpu().numpy().copy())
        handles=[module.get_vlm_model().vision_model.register_forward_hook(image_hook,with_kwargs=True),
                 module.get_vlm_model().connector.register_forward_hook(feature_hook)]
        try:
            actions,timing=predict_cached(policy,pre,post,observations,manifest['samples'],args.device)
            np.testing.assert_array_equal(actions,fp)
        finally:
            for h in handles:h.remove()
        if len(pixels)!=80 or len(features)!=80:
            raise ValueError('Expected exactly two camera calls per observation')
        panel=args.root/'vision_panel.npz'
        np.savez_compressed(panel,pixels=np.concatenate(pixels),reference=np.concatenate(features))
        info={'scope':'40 isolated development observations, FP boundary capture',
              'identity':ident,'cache_manifest_sha256':file_sha256(args.cache/'manifest.json'),
              'panel_sha256':file_sha256(panel),'samples':manifest['samples'],
              'camera_count':80,'fp_actions_exact_match':True,'host_timing':timing}
        (args.root/'capture.json').write_text(json.dumps(info,indent=2)+'\n')
        print(json.dumps({'panel_bytes':panel.stat().st_size,'camera_count':80,'fp_exact_match':True}))
        return
    capture=json.loads((args.root/'capture.json').read_text())
    board=json.loads((args.root/'board_report.json').read_text())
    if (capture['identity']!=ident or capture['cache_manifest_sha256']!=file_sha256(args.cache/'manifest.json')
        or capture['samples']!=manifest['samples'] or board['status']!='success'
        or board['panel_sha256']!=file_sha256(args.root/'vision_panel.npz')
        or board['output_sha256']!=file_sha256(args.root/'board_features.npz')):
        raise ValueError('Pinned board panel identity mismatch')
    expected_model=ROOT/'runs/smolvla_vision_split_v1/vision_connector_fp16.rknn'
    if board['model_sha256']!=file_sha256(expected_model):
        raise ValueError('Board RKNN model differs')
    with np.load(args.root/'vision_panel.npz',allow_pickle=False) as z:
        pixels=z['pixels'].copy()
    with np.load(args.root/'board_features.npz',allow_pickle=False) as z:
        features=z['features'].copy()
    if features.shape!=(80,64,960):
        raise ValueError('Board features shape differs')
    count=0
    def embed(image):
        nonlocal count
        if count>=80:raise ValueError('Unexpected camera call')
        np.testing.assert_array_equal(image.detach().float().cpu().numpy(),pixels[count:count+1])
        out=torch.from_numpy(features[count:count+1]).to(device=image.device,dtype=torch.float32)
        count+=1
        return out
    old=module.embed_image;module.embed_image=embed
    try:
        actions,timing=predict_cached(policy,pre,post,observations,manifest['samples'],args.device)
        if count!=80 or not np.isfinite(actions).all():raise ValueError('Invalid candidate inference')
    finally:module.embed_image=old
    delta=actions.astype(np.float64)-fp.astype(np.float64)
    task_mae=np.abs(delta).mean(axis=(1,2))
    metrics=score_actions(actions,fp,observations['recorded_action'],
                          np.array([s['task_index'] for s in manifest['samples']]))
    report={'scope':'RK3588 real vision features + original GPU prefix/expert, offline development only',
            'observations':40,'camera_calls':count,'closed_loop_success_rate':'not_measured',
            'full_board_execution':'not_measured','checkpoint_sha256':ident['checkpoint_sha256'],
            'panel_sha256':capture['panel_sha256'],'board_report_sha256':file_sha256(args.root/'board_report.json'),
            'action_mae':float(np.abs(delta).mean()),'action_rmse':float(np.sqrt(np.mean(delta**2))),
            'action_max_abs_error':float(np.abs(delta).max()),
            'per_observation_action_mae':task_mae.tolist(),'metrics':metrics,'host_timing':timing}
    report['per_dimension_action_mae']=np.abs(delta).mean(axis=(0,1)).tolist()
    report['per_dimension_max_abs_error']=np.abs(delta).max(axis=(0,1)).tolist()
    flips=np.argwhere((actions[:,:,6]>0)!=(fp[:,:,6]>0))
    report['gripper_sign_disagreements']=[{
        'observation_index':int(i),'task_index':manifest['samples'][i]['task_index'],
        'chunk_offset':int(j),'fp':float(fp[i,j,6]),'candidate':float(actions[i,j,6])}
        for i,j in flips]
    np.savez_compressed(args.root/'actions.npz',candidate=actions,original=fp)
    (args.root/'action_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('metrics','host_timing','per_observation_action_mae')},indent=2))


if __name__=='__main__':main()
