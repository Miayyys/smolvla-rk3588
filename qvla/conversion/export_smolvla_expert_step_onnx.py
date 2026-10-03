#!/usr/bin/env python3
"""Export one complete expert denoising step, including time/action projections."""

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
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import DynamicCache
from qvla.evaluation.haq_offline_eval import identity, load_policy, predict_cached
from qvla.haq.offline_actions import file_sha256,load_cache
from qvla.conversion.export_smolvla_prefix_onnx import export_rope
from qvla.evaluation.verify_smolvla_vision_float import metrics
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.policies.common.vla_utils import create_sinusoidal_pos_embedding
ROOT=Path(__file__).resolve().parents[2]


class ExpertStep(torch.nn.Module):
    def __init__(self,core):
        super().__init__();self.core=core

    def forward(self,noisy_actions,time_embedding,prefix_pad_masks,*kv):
        cache=DynamicCache()
        for i in range(16):cache.update(kv[2*i],kv[2*i+1],i)
        action_embedding=self.core.action_in_proj(noisy_actions)
        joined=torch.cat((action_embedding,time_embedding[:,None,:].expand_as(action_embedding)),dim=2)
        suffix=self.core.action_time_mlp_out(torch.nn.functional.silu(self.core.action_time_mlp_in(joined)))
        pad_masks=torch.ones(suffix.shape[:2],dtype=torch.bool,device=suffix.device)
        att_masks=torch.ones(suffix.shape[:2],dtype=suffix.dtype,device=suffix.device)
        prefix_mask=prefix_pad_masks[:,None,:].expand(suffix.shape[0],suffix.shape[1],prefix_pad_masks.shape[1])
        mask=torch.cat((prefix_mask,make_att_2d_masks(pad_masks,att_masks)),dim=2)
        # ONNX CumSum / ReduceSum do not accept boolean input tensors.
        positions=prefix_pad_masks.long().sum(-1)[:,None]+pad_masks.long().cumsum(-1)-1
        outputs,_=self.core.vlm_with_expert.forward(attention_mask=mask,position_ids=positions,
            past_key_values=cache,inputs_embeds=[None,suffix],use_cache=self.core.config.use_cache)
        hidden=outputs[1][:,-self.core.config.chunk_size:].float()
        return self.core.action_out_proj(hidden)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,default=ROOT/'artifacts/transfer/model')
    p.add_argument('--vlm-assets-dir',type=Path,default=ROOT/'artifacts/transfer/smolvlm2_assets')
    p.add_argument('--splits',type=Path,default=ROOT/'data/libero_splits.json')
    p.add_argument('--partition',type=Path,default=ROOT/'config/evaluation_partition_v2.json')
    p.add_argument('--cache',type=Path,default=ROOT/'runs/haq_offline_local_v1/fp_cache40')
    p.add_argument('--root',type=Path,default=ROOT/'runs/smolvla_expert_step_split_v1')
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
    policy,pre,post=load_policy(args);core=policy.model;old=core.denoise_step;capture={}
    def step(*a,**kw):
        first=not capture
        if first:
            for name,key in [('noisy_actions','x_t'),('timestep','timestep'),('prefix_pad_masks','prefix_pad_masks')]:
                capture[name]=kw[key].detach().cpu().clone()
            capture['kv']=[]
            for layer in kw['past_key_values'].layers:
                capture['kv'].extend((layer.keys.detach().float().cpu().clone(),layer.values.detach().float().cpu().clone()))
        out=old(*a,**kw)
        if first:capture['original_velocity']=out.detach().float().cpu().clone()
        return out
    core.denoise_step=step
    try:
        chosen={k:v[:1] for k,v in obs.items()}
        actions,_=predict_cached(policy,pre,post,chosen,manifest['samples'][:1],args.device)
        np.testing.assert_array_equal(actions,fp[:1])
    finally:core.denoise_step=old
    args.root.mkdir(parents=True,exist_ok=True)
    def time_embedding(t):
        return create_sinusoidal_pos_embedding(t.float(),core.vlm_with_expert.expert_hidden_size,
            core.config.min_period,core.config.max_period,device=torch.device('cpu')).float()
    names=['noisy_actions','time_embedding','prefix_pad_masks']+[
        f'{kind}_{i}' for i in range(16) for kind in ('key','value')]
    values=tuple(x.clone() for x in [capture['noisy_actions'].float(),time_embedding(capture['timestep']),
                                   capture['prefix_pad_masks'],*capture['kv']])
    np.savez_compressed(args.root/'expert_boundary.npz',
        **{name:x.numpy() for name,x in zip(names,values)},original_velocity=capture['original_velocity'].numpy(),
        timestep=capture['timestep'].numpy())
    wrapper=ExpertStep(core).eval().requires_grad_(False).to('cpu',dtype=torch.float32)
    with torch.inference_mode():
        reference_cache=DynamicCache()
        for i in range(16):reference_cache.update(values[3+2*i],values[4+2*i],i)
        original_fp32=core.denoise_step(prefix_pad_masks=values[2],past_key_values=reference_cache,
                                       x_t=values[0],timestep=capture['timestep'].float())
        torch.testing.assert_close(wrapper(*values),original_fp32,rtol=0,atol=0)
    old_rope=source.apply_rope;source.apply_rope=export_rope
    try:
        with torch.inference_mode():
            actual=wrapper(*values);torch.testing.assert_close(actual,original_fp32,rtol=0,atol=0)
            path=args.root/'expert_step.onnx'
            torch.onnx.export(wrapper,values,str(path),dynamo=False,opset_version=17,
                input_names=names,output_names=['velocity'],do_constant_folding=True)
            alternate=(values[0]*0.7,time_embedding(capture['timestep']*0.5),*values[2:])
            alternate_output=wrapper(*alternate)
    finally:source.apply_rope=old_rope
    model=onnx.load(str(path))
    # The suffix positions are prefix_valid_count + arange(50), so their minimum
    # is always the first entry. Avoid INT64 ReduceMin, which compiled but failed
    # in the RK3588 runtime CPU fallback. Gather preserves the integer exactly.
    reductions=[n for n in model.graph.node if n.op_type=='ReduceMin']
    if len(reductions)!=1 or reductions[0].name!='/ReduceMin':
        raise ValueError('Unexpected position reduction graph')
    reduction=reductions[0]
    attrs={a.name:onnx.helper.get_attribute_value(a) for a in reduction.attribute}
    if attrs!={'axes':[1],'keepdims':1}:raise ValueError('Unexpected minimum axes')
    index_name='qvla_first_suffix_position'
    model.graph.initializer.append(onnx.numpy_helper.from_array(np.array([0],dtype=np.int64),index_name))
    replacement=onnx.helper.make_node('Gather',[reduction.input[0],index_name],list(reduction.output),
                                     name='QVLAMonotonePositionMinimum',axis=1)
    reduction.CopyFrom(replacement)
    onnx.checker.check_model(model);onnx.save(model,str(path))
    np.save(args.root/'expert_fp32_reference.npy',actual.numpy(),allow_pickle=False)
    np.savez_compressed(args.root/'expert_alternate.npz',
        **{name:x.numpy() for name,x in zip(names,alternate)},velocity=alternate_output.numpy())
    report={'scope':'full expert step ONNX, 16 layers and time/action interfaces; no board execution',
        'checkpoint_sha256':ident['checkpoint_sha256'],'sample':manifest['samples'][0],
        'fp_action_exact_match':True,'rope_rewrite_fp32_exact':True,'explicit_integer_masks_fp32_exact':True,
        'cpu_time_embedding_interface':True,'monotone_position_minimum_replaced_by_first_entry':True,
        'onnx_sha256':file_sha256(path),'onnx_bytes':path.stat().st_size,
        'inputs':[{'name':name,'shape':list(x.shape),'dtype':str(x.dtype)} for name,x in zip(names,values)],
        'output_shape':list(actual.shape),'fp32_velocity_vs_original_loaded':metrics(actual.numpy(),capture['original_velocity'].numpy()),
        'scatternd_nodes':sum(n.op_type=='ScatterND' for n in model.graph.node)}
    (args.root/'export.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='inputs'},indent=2))


if __name__=='__main__':main()
