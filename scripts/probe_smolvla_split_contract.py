#!/usr/bin/env python3
"""Record exact SmolVLA deployment boundary tensors on one pinned FP action."""
import argparse
import json
import os
from pathlib import Path

os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'

import numpy as np
import torch

from haq_offline_eval import identity, load_policy, predict_cached
from qvla_haq.offline_actions import file_sha256, load_cache


ROOT=Path(__file__).resolve().parents[1]


def tensor(t):
    return {'shape':list(t.shape),'dtype':str(t.dtype),'elements':t.numel(),
            'bytes_at_current_dtype':t.numel()*t.element_size()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,default=ROOT/'artifacts/transfer/model')
    p.add_argument('--vlm-assets-dir',type=Path,default=ROOT/'artifacts/transfer/smolvlm2_assets')
    p.add_argument('--splits',type=Path,default=ROOT/'data/libero_splits.json')
    p.add_argument('--partition',type=Path,default=ROOT/'config/evaluation_partition_v2.json')
    p.add_argument('--cache',type=Path,default=ROOT/'runs/haq_offline_local_v1/fp_cache40')
    p.add_argument('--output',type=Path,default=ROOT/'runs/smolvla_split_contract_v1.json')
    p.add_argument('--index',type=int,default=0)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    torch.set_num_threads(4)
    ident=identity(args)
    manifest,obs,fp=load_cache(args.cache,ident)
    if not 0<=args.index<len(manifest['samples']):p.error('index out of range')
    if torch.__version__!=manifest['torch_version'] or args.device!=manifest['device']:
        raise ValueError('Cached FP runtime mismatch')
    policy,pre,post=load_policy(args)
    module=policy.model.vlm_with_expert
    result={'scope':'one cached FP action; local GPU tensor contract, no board execution',
            'sample':manifest['samples'][args.index],
            'checkpoint_sha256':ident['checkpoint_sha256'],
            'cache_manifest_sha256':file_sha256(args.cache/'manifest.json'),
            'model_config':{'num_vlm_layers':module.num_vlm_layers,
                            'num_expert_layers':module.num_expert_layers,
                            'attention_mode':module.attention_mode,
                            'self_attn_every_n_layers':module.self_attn_every_n_layers,
                            'flow_steps':policy.config.num_steps,
                            'action_chunk_size':policy.config.chunk_size},
            'calls':[]}
    boundary_arrays={}
    handles=[]

    def vision_hook(_,inp,kwargs,out):
        pixels=inp[0] if inp else kwargs['pixel_values']
        index=len(result.get('vision_calls',[]))
        boundary_arrays[f'vision_input_{index}']=pixels.detach().float().cpu().numpy()
        boundary_arrays[f'vision_output_{index}']=out.last_hidden_state.detach().float().cpu().numpy()
        result.setdefault('vision_calls',[]).append({'input':tensor(pixels),
          'last_hidden_state':tensor(out.last_hidden_state)})

    def connector_hook(_,inp,out):
        index=len(result.get('connector_calls',[]))
        boundary_arrays[f'connector_output_{index}']=out.detach().float().cpu().numpy()
        result.setdefault('connector_calls',[]).append({'input':tensor(inp[0]),'output':tensor(out)})

    def vlm_hook(_,inp,out):
        embeds,cache=out
        row={'stage':'prefix_prefill' if not result['calls'] else 'expert_denoise',
             'vlm_output':tensor(embeds[0]) if embeds[0] is not None else None,
             'expert_output':tensor(embeds[1]) if embeds[1] is not None else None}
        if cache is not None:
            row['cache_layers']=len(cache.layers)
            row['cache_shapes']=[{'layer':i,'keys':tensor(layer.keys),'values':tensor(layer.values)}
                                 for i,layer in enumerate(cache.layers)]
            row['cache_total_bytes_at_current_dtype']=sum(
                z['keys']['bytes_at_current_dtype']+z['values']['bytes_at_current_dtype']
                for z in row['cache_shapes'])
        result['calls'].append(row)

    handles.append(module.get_vlm_model().vision_model.register_forward_hook(vision_hook,with_kwargs=True))
    handles.append(module.get_vlm_model().connector.register_forward_hook(connector_hook))
    # SmolVLA calls this method as `.forward(...)`, bypassing Module.__call__ hooks.
    original_forward=module.forward
    def traced_forward(*forward_args,**forward_kwargs):
        out=original_forward(*forward_args,**forward_kwargs)
        vlm_hook(module,forward_args,out)
        return out
    module.forward=traced_forward
    try:
        chosen={k:v[args.index:args.index+1] for k,v in obs.items()}
        sample=[manifest['samples'][args.index]]
        action,timing=predict_cached(policy,pre,post,chosen,sample,args.device)
        np.testing.assert_array_equal(action,fp[args.index:args.index+1])
    finally:
        for handle in handles:handle.remove()
        module.forward=original_forward
    result['action_output']=tensor(torch.from_numpy(action))
    result['fp_action_exact_match']=True
    result['host_timing']=timing
    npz_path=args.output.with_suffix('.npz')
    np.savez_compressed(npz_path,**boundary_arrays)
    result['boundary_npz']=str(npz_path)
    result['script_sha256']=file_sha256(Path(__file__))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'prefix_calls':sum(x['stage']=='prefix_prefill' for x in result['calls']),
                      'denoise_calls':sum(x['stage']=='expert_denoise' for x in result['calls']),
                      'prefix_cache_bytes':result['calls'][0].get('cache_total_bytes_at_current_dtype'),
                      'fp_action_exact_match':result['fp_action_exact_match'],
                      'output':str(args.output)}))


if __name__=='__main__':main()
