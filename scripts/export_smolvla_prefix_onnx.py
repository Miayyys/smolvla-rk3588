#!/usr/bin/env python3
"""Export fixed-shape SmolVLA prefix prefill with all per-layer K/V outputs."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from haq_offline_eval import identity, load_policy, predict_cached
from qvla_haq.offline_actions import file_sha256, load_cache
from verify_smolvla_vision_float import metrics

ROOT=Path(__file__).resolve().parents[1]


def export_rope(x,positions,max_wavelength=10_000):
    dtype=x.dtype;half=x.shape[-1]//2
    v=x.float()
    exponents=(2.0/x.shape[-1])*torch.arange(half,dtype=torch.float32,device=x.device)
    radians=positions[...,None].float()/(max_wavelength**exponents)[None,None,:]
    sine=torch.sin(radians)[:,:,None,:];cosine=torch.cos(radians)[:,:,None,:]
    a,b=v.split(half,dim=-1)
    return torch.cat((a*cosine-b*sine,b*cosine+a*sine),dim=-1).to(dtype)


class PrefixWithKV(torch.nn.Module):
    def __init__(self,module):
        super().__init__();self.module=module

    def forward(self,prefix,attention_mask,position_ids):
        outputs,cache=self.module.forward(attention_mask=attention_mask,position_ids=position_ids,
            past_key_values=None,inputs_embeds=[prefix,None],use_cache=True)
        values=[outputs[0].float()]
        for layer in cache.layers:
            values.extend((layer.keys.float(),layer.values.float()))
        return tuple(values)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,default=ROOT/'artifacts/transfer/model')
    p.add_argument('--vlm-assets-dir',type=Path,default=ROOT/'artifacts/transfer/smolvlm2_assets')
    p.add_argument('--splits',type=Path,default=ROOT/'data/libero_splits.json')
    p.add_argument('--partition',type=Path,default=ROOT/'config/evaluation_partition_v2.json')
    p.add_argument('--cache',type=Path,default=ROOT/'runs/haq_offline_local_v1/fp_cache40')
    p.add_argument('--root',type=Path,default=ROOT/'runs/smolvla_prefix_split_v1')
    p.add_argument('--device',default='cuda')
    p.add_argument('--onnx-site-packages',type=Path)
    args=p.parse_args()
    if args.onnx_site_packages:sys.path.append(str(args.onnx_site_packages))
    import onnx
    import lerobot.policies.smolvla.smolvlm_with_expert as source
    torch.set_num_threads(4)
    ident=identity(args);manifest,obs,fp=load_cache(args.cache,ident)
    if torch.__version__!=manifest['torch_version'] or args.device!=manifest['device']:
        raise ValueError('FP cache runtime mismatch')
    policy,pre,post=load_policy(args)
    module=policy.model.vlm_with_expert
    old_forward=module.forward
    captured={}
    def trace(*a,**kw):
        outputs,cache=old_forward(*a,**kw)
        if not captured:
            for name,t in [('prefix',kw['inputs_embeds'][0]),('attention_mask',kw['attention_mask']),('position_ids',kw['position_ids'])]:
                captured[name]=t.detach().cpu().clone()
            captured['reference']=[outputs[0].detach().float().cpu().clone()]
            for layer in cache.layers:
                captured['reference'].extend((layer.keys.detach().float().cpu().clone(),layer.values.detach().float().cpu().clone()))
        return outputs,cache
    module.forward=trace
    try:
        chosen={k:v[:1] for k,v in obs.items()}
        actions,_=predict_cached(policy,pre,post,chosen,manifest['samples'][:1],args.device)
        np.testing.assert_array_equal(actions,fp[:1])
    finally:module.forward=old_forward
    args.root.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.root/'prefix_boundary.npz',
        **{k:captured[k].numpy() for k in ('prefix','attention_mask','position_ids')},
        **{f'output_{i}':x.numpy() for i,x in enumerate(captured['reference'])})
    wrapper=PrefixWithKV(module).eval().requires_grad_(False).to('cpu',dtype=torch.float32)
    values=tuple(x.clone() for x in (captured['prefix'].float(),captured['attention_mask'],captured['position_ids']))
    # Avoid in-place slice assignment lowering to ScatterND; verify identical FP32 algebra first.
    with torch.inference_mode():original_fp32=wrapper(*values)
    old_rope=source.apply_rope;source.apply_rope=export_rope
    try:
        with torch.inference_mode():actual=wrapper(*values)
        for a,b in zip(actual,original_fp32):torch.testing.assert_close(a,b,rtol=0,atol=0)
        output_names=['prefix_hidden']+[f'{kind}_{i}' for i in range(module.num_vlm_layers) for kind in ('key','value')]
        path=args.root/'prefix_with_kv.onnx'
        with torch.inference_mode():
            torch.onnx.export(wrapper,values,str(path),dynamo=False,opset_version=17,
                input_names=['prefix','attention_mask','position_ids'],output_names=output_names,
                do_constant_folding=True)
    finally:source.apply_rope=old_rope
    model=onnx.load(str(path));onnx.checker.check_model(model)
    np.savez_compressed(args.root/'prefix_fp32_reference.npz',
                        **{f'output_{i}':x.numpy() for i,x in enumerate(actual)})
    report={'scope':'fixed-shape prefix ONNX; all 16 layer K/V explicitly exported, not board executed',
        'checkpoint_sha256':ident['checkpoint_sha256'],'sample':manifest['samples'][0],
        'fp_action_exact_match':True,'rope_rewrite_fp32_exact':True,
        'onnx_sha256':file_sha256(path),'onnx_bytes':path.stat().st_size,
        'inputs':[{'name':name,'shape':list(x.shape),'dtype':str(x.dtype)}
                  for name,x in zip(('prefix','attention_mask','position_ids'),values)],
        'outputs':[{'index':i,'name':name,'shape':list(x.shape),
                    'fp32_vs_original_loaded':metrics(x.numpy(),captured['reference'][i].numpy())}
                   for i,(name,x) in enumerate(zip(output_names,actual))],
        'scatternd_nodes':sum(n.op_type=='ScatterND' for n in model.graph.node)}
    (args.root/'export.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='outputs'},indent=2))


if __name__=='__main__':main()
