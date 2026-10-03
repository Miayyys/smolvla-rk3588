#!/usr/bin/env python3
"""Feed one actual RKNN prefix cache into the original GPU action expert."""

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
from transformers import DynamicCache
from qvla.evaluation.haq_offline_eval import identity, load_policy, predict_cached
from qvla.haq.offline_actions import file_sha256,load_cache
from qvla.evaluation.verify_smolvla_vision_float import metrics
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,default=ROOT/'artifacts/transfer/model')
    p.add_argument('--vlm-assets-dir',type=Path,default=ROOT/'artifacts/transfer/smolvlm2_assets')
    p.add_argument('--splits',type=Path,default=ROOT/'data/libero_splits.json')
    p.add_argument('--partition',type=Path,default=ROOT/'config/evaluation_partition_v2.json')
    p.add_argument('--cache',type=Path,default=ROOT/'runs/haq_offline_local_v1/fp_cache40')
    p.add_argument('--root',type=Path,default=ROOT/'runs/smolvla_prefix_split_v1')
    p.add_argument('--device',default='cuda')
    args=p.parse_args();torch.set_num_threads(4)
    ident=identity(args);manifest,obs,fp=load_cache(args.cache,ident)
    if torch.__version__!=manifest['torch_version'] or args.device!=manifest['device']:
        raise ValueError('FP runtime mismatch')
    report=json.loads((args.root/'prefix_board_report.json').read_text())
    export=json.loads((args.root/'export.json').read_text())
    if (report['status']!='success' or export['sample']!=manifest['samples'][0]
        or export['checkpoint_sha256']!=ident['checkpoint_sha256']
        or report['model_sha256']!=file_sha256(args.root/'prefix_with_kv_fp16.rknn')
        or report['boundary_sha256']!=file_sha256(args.root/'prefix_boundary.npz')
        or report['output_sha256']!=file_sha256(args.root/'prefix_board_report.npz')):
        raise ValueError('Pinned RKNN cache identity mismatch')
    with np.load(args.root/'prefix_boundary.npz',allow_pickle=False) as z:
        boundary={n:z[n].copy() for n in ('prefix','attention_mask','position_ids')}
    with np.load(args.root/'prefix_board_report.npz',allow_pickle=False) as z:
        outputs=[z[f'output_{i}'].copy() for i in range(33)]
    policy,pre,post=load_policy(args);module=policy.model.vlm_with_expert
    chosen={k:v[:1] for k,v in obs.items()};samples=manifest['samples'][:1]
    original,_=predict_cached(policy,pre,post,chosen,samples,args.device)
    np.testing.assert_array_equal(original,fp[:1])
    old=module.forward;calls=0;expert_calls=0
    def injected(*a,**kw):
        nonlocal calls,expert_calls
        if kw.get('past_key_values') is None and kw['inputs_embeds'][0] is not None:
            calls+=1;prefix=kw['inputs_embeds'][0]
            for n,t in [('prefix',prefix),('attention_mask',kw['attention_mask']),('position_ids',kw['position_ids'])]:
                np.testing.assert_array_equal(t.detach().cpu().numpy(),boundary[n])
            cache=DynamicCache()
            layers=module.get_vlm_model().text_model.layers
            for i in range(16):
                key=torch.from_numpy(outputs[1+2*i]).to(device=prefix.device,dtype=layers[i].self_attn.k_proj.weight.dtype)
                value=torch.from_numpy(outputs[2+2*i]).to(device=prefix.device,dtype=layers[i].self_attn.v_proj.weight.dtype)
                cache.update(key,value,i)
            hidden=torch.from_numpy(outputs[0]).to(device=prefix.device,dtype=layers[0].self_attn.q_proj.weight.dtype)
            return [hidden,None],cache
        expert_calls+=1
        return old(*a,**kw)
    module.forward=injected
    try:
        candidate,timing=predict_cached(policy,pre,post,chosen,samples,args.device)
        if calls!=1 or expert_calls!=10 or not np.isfinite(candidate).all():
            raise ValueError('Unexpected split call pattern')
    finally:module.forward=old
    np.savez_compressed(args.root/'prefix_action.npz',original=original,candidate=candidate)
    result={'scope':'one real RKNN prefix K/V cache + original GPU vision and expert',
            'fp_action_exact_match':True,'prefix_calls':calls,'expert_calls':expert_calls,
            'cache_bridge_dtype':str(module.get_vlm_model().text_model.layers[0].self_attn.k_proj.weight.dtype),
            'sample':samples[0],'action_vs_fp':metrics(candidate,original),
            'gripper_sign_disagreements':int(np.count_nonzero((candidate[:,:,6]>0)!=(original[:,:,6]>0))),
            'board_report_sha256':file_sha256(args.root/'prefix_board_report.json'),
            'full_board_execution':'not_measured','closed_loop_success_rate':'not_measured',
            'host_timing':timing}
    (args.root/'prefix_action_report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
