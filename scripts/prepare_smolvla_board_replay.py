#!/usr/bin/env python3
"""Capture one pinned observation and export actual CPU glue parameters."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from haq_offline_eval import identity,load_policy,predict_cached
from qvla_haq.offline_actions import file_sha256,load_cache
from smolvla_numpy_glue import assemble_prefix,time_embedding,postprocess
from lerobot.policies.common.vla_utils import create_sinusoidal_pos_embedding
from verify_smolvla_vision_float import metrics
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name,default in [('model-dir','artifacts/transfer/model'),('vlm-assets-dir','artifacts/transfer/smolvlm2_assets'),
                         ('splits','data/libero_splits.json'),('partition','config/evaluation_partition_v2.json'),
                         ('cache','runs/haq_offline_local_v1/fp_cache40'),('root','runs/smolvla_full_board_v1')]:
        p.add_argument('--'+name,type=Path,default=ROOT/default)
    p.add_argument('--device',default='cuda');args=p.parse_args();torch.set_num_threads(4)
    ident=identity(args);manifest,obs,fp=load_cache(args.cache,ident)
    if torch.__version__!=manifest['torch_version'] or args.device!=manifest['device']:
        raise ValueError('Pinned runtime mismatch')
    policy,pre,post=load_policy(args);core=policy.model;module=core.vlm_with_expert
    if core.add_image_special_tokens or core.config.adapt_to_pi_aloha or core._rtc_enabled():
        raise ValueError('Unsupported CPU glue variant')
    captured={};features=[];steps=[]
    def array(x):return x.detach().float().cpu().numpy().copy()
    old_embed=core.embed_prefix;old_image=module.embed_image;old_step=core.denoise_step;old_sample=core.sample_actions
    def embed(images,img_masks,lang_tokens,lang_masks,state=None):
        captured.update(images=np.stack([array(x) for x in images]),
            image_masks=np.stack([x.detach().cpu().numpy().copy() for x in img_masks]),
            lang_tokens=lang_tokens.detach().cpu().numpy().copy(),lang_masks=lang_masks.detach().cpu().numpy().copy(),state=array(state))
        result=old_embed(images,img_masks,lang_tokens,lang_masks,state=state)
        captured['prefix']=array(result[0]);return result
    def image(x):
        y=old_image(x);features.append(array(y));return y
    def step(*a,**kw):
        if not steps:captured['noise']=array(kw['x_t'])
        steps.append(kw['timestep'].detach().cpu().numpy().copy())
        return old_step(*a,**kw)
    def sample(*a,**kw):
        y=old_sample(*a,**kw);captured['raw_actions']=array(y);return y
    core.embed_prefix=embed;module.embed_image=image;core.denoise_step=step;core.sample_actions=sample
    try:
        actions,_=predict_cached(policy,pre,post,{k:v[:1] for k,v in obs.items()},manifest['samples'][:1],args.device)
        np.testing.assert_array_equal(actions,fp[:1])
    finally:core.embed_prefix=old_embed;module.embed_image=old_image;core.denoise_step=old_step;core.sample_actions=old_sample
    if len(features)!=2 or len(steps)!=10:raise ValueError('Unexpected execution pattern')
    stats=load_file(str(args.model_dir/'policy_postprocessor_step_0_unnormalizer_processor.safetensors'))
    weights={'token_embedding':array(module.get_vlm_model().text_model.get_input_embeddings().weight),
             'state_weight':array(core.state_proj.weight),'state_bias':array(core.state_proj.bias),
             'action_mean':stats['action.mean'].numpy(),'action_std':stats['action.std'].numpy()}
    inputs={k:captured[k] for k in ('images','image_masks','lang_tokens','lang_masks','state','noise')}
    prefix,pad,mask,pos=assemble_prefix(features,inputs,weights)
    np.testing.assert_allclose(prefix,captured['prefix'],rtol=1e-6,atol=1e-5)
    with np.load(ROOT/'runs/smolvla_prefix_split_v1/prefix_boundary.npz') as z:
        np.testing.assert_array_equal(mask,z['attention_mask']);np.testing.assert_array_equal(pos,z['position_ids'])
    cpu_actions=postprocess(captured['raw_actions'],weights)[0]
    np.testing.assert_allclose(cpu_actions,actions[0],rtol=0,atol=1e-7)
    time_errors=[]
    for t in steps:
        actual=time_embedding(t,module.expert_hidden_size,core.config.min_period,core.config.max_period)
        reference=create_sinusoidal_pos_embedding(torch.from_numpy(t),module.expert_hidden_size,
            core.config.min_period,core.config.max_period,device=torch.device('cpu')).float().numpy()
        np.testing.assert_allclose(actual,reference,rtol=0,atol=1e-7);time_errors.append(metrics(actual,reference))
    args.root.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.root/'replay_inputs.npz',**inputs)
    # Uncompressed parameters make loading measurable and avoid repeated decompression on the board.
    np.savez(args.root/'cpu_weights.npz',**weights)
    np.savez_compressed(args.root/'fp_reference.npz',actions=actions,raw_actions=captured['raw_actions'],prefix=captured['prefix'],features=np.stack(features))
    report={'scope':'one processed-observation replay, CPU glue verified; tokenizer/image preprocessing captured on host',
            'checkpoint_sha256':ident['checkpoint_sha256'],'sample':manifest['samples'][0],
            'fp_action_exact_match':True,'cpu_prefix_vs_original':metrics(prefix,captured['prefix']),
            'cpu_postprocessor_vs_original':metrics(cpu_actions,actions[0]),'cpu_time_embedding_errors':time_errors,
            'num_steps':len(steps),'min_period':core.config.min_period,'max_period':core.config.max_period,
            'expert_hidden_size':module.expert_hidden_size,'original_storage_dtype':'BF16 majority, FP32 interfaces',
            'hashes':{name:file_sha256(args.root/name) for name in ('replay_inputs.npz','cpu_weights.npz','fp_reference.npz')}}
    (args.root/'replay.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
