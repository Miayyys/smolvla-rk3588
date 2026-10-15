"""Export all learned assets from the downloaded six-action SmolVLA base for speed tests."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse,hashlib,json,shutil,sys,pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from transformers import LlamaConfig
from qvla.evaluation.haq_offline_eval import load_policy
from qvla.conversion.export_smolvla_expert_step_onnx import ExpertStep
from qvla.conversion.export_smolvla_prefix_onnx import export_rope
from qvla.runtime.smolvla_numpy_glue import time_embedding


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--onnx-site-packages',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path.append(str(a.onnx_site_packages))
    import onnx
    import lerobot.policies.smolvla.smolvlm_with_expert as source
    root=Path.cwd();model_dir=root/'artifacts/smolvla_base';assets=root/'artifacts/transfer/smolvlm2_assets'
    checkpoint_sha=sha(model_dir/'model.safetensors')
    if checkpoint_sha!='7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb':
        raise ValueError('Unexpected new-base checkpoint')
    torch.set_num_threads(4);torch.manual_seed(42)
    args=SimpleNamespace(model_dir=model_dir,vlm_assets_dir=assets,device='cuda')
    policy,pre,post=load_policy(args);policy.float().eval()
    module=policy.model.vlm_with_expert;core=policy.model
    module.vlm.set_attn_implementation('eager')
    capture={};oldforward=module.forward;oldstep=core.denoise_step
    def capture_forward(*args,**kw):
        if kw['inputs_embeds'][0] is not None:
            capture['prefix']=kw['inputs_embeds'][0].detach().float().cpu()
            capture['mask']=kw['attention_mask'].detach().cpu()
            capture['positions']=kw['position_ids'].detach().cpu()
        return oldforward(*args,**kw)
    def capture_step(*args,**kw):
        if 'expert' not in capture:
            values=[kw['x_t'].detach().float().cpu(),torch.from_numpy(time_embedding(kw['timestep'].detach().cpu().numpy(),720,.004,4.)),kw['prefix_pad_masks'].detach().cpu()]
            for layer in kw['past_key_values'].layers:values.extend((layer.keys.detach().float().cpu().clone(),layer.values.detach().float().cpu().clone()))
            capture['expert']=tuple(values)
        return oldstep(*args,**kw)
    module.forward=capture_forward;core.denoise_step=capture_step
    with np.load(root/'runs/smolvla_base_smoke_v1/inputs.npz') as z:
        obs={'observation.images.camera1':torch.from_numpy(z['observation.images.camera1'].copy()),'observation.state':torch.from_numpy(z['observation.state'].copy()),'task':str(z['task'].item())}
        noise=torch.from_numpy(z['noise'].copy()).cuda()
    with torch.inference_mode():
        batch=pre(obs.copy())
        images,_=policy.prepare_images(batch)
        np.save(out/'vision_input.npy',images[0].cpu().numpy())
        y=post(policy.predict_action_chunk(batch,noise=noise)).cpu().numpy()
    module.forward=oldforward;core.denoise_step=oldstep
    assert y.shape==(1,50,6) and capture['prefix'].shape==(1,113,960)
    np.save(out/'gpu_reference.npy',y)
    print('CAPTURED new base single camera, six actions',flush=True)
    names=['noisy_actions','time_embedding','prefix_pad_masks']+[f'{k}_{i}' for i in range(16) for k in ['key','value']]
    np.savez_compressed(out/'expert_boundary.npz',**{k:v.numpy() for k,v in zip(names,capture['expert'])})
    core=core.cpu();wrapper=ExpertStep(core).eval().requires_grad_(False)
    oldrope=source.apply_rope;source.apply_rope=export_rope
    try:
        with torch.inference_mode():
            torch.onnx.export(wrapper,capture['expert'],str(out/'expert_full.onnx'),dynamo=False,opset_version=17,input_names=names,output_names=['velocity'])
    finally:source.apply_rope=oldrope
    m=onnx.load(str(out/'expert_full.onnx'))
    for node in m.graph.node:
        if node.op_type=='ReduceMin':
            ix='base_first_suffix_position';m.graph.initializer.append(onnx.numpy_helper.from_array(np.array([0],dtype=np.int64),ix))
            node.CopyFrom(onnx.helper.make_node('Gather',[node.input[0],ix],list(node.output),name='BaseMonotonePositionMinimum',axis=1))
    onnx.checker.check_model(m);onnx.save(m,str(out/'expert_full.onnx'));del m
    folder=out/'expert_full';folder.mkdir(exist_ok=True)
    if not (folder/'model.onnx').exists():(folder/'model.onnx').symlink_to(out/'expert_full.onnx')
    for name,value in zip(names,capture['expert']):np.save(folder/(name+'.npy'),value.numpy())
    (folder/'dataset.txt').write_text(' '.join(str(folder/(n+'.npy')) for n in names)+'\n')
    (folder/'export.json').write_text(json.dumps({'source_checkpoint_sha256':checkpoint_sha,'layers':16,'prefix_tokens':113,'action_dim':6}))
    print('EXPORTED new base expert',flush=True)
    # Every learned CPU parameter derives from the new checkpoint, including embedding quantization.
    state=policy.state_dict();prefix='model.vlm_with_expert.vlm.model.text_model.'
    embedding=state[prefix+'embed_tokens.weight'].detach().float().cpu().numpy()
    scale=np.maximum(np.max(np.abs(embedding),axis=1)/127,np.float32(1e-12)).astype('f4')
    quant=np.clip(np.rint(embedding/scale[:,None]),-127,127).astype('i1')
    normal=load_file(str(model_dir/'policy_preprocessor_step_5_normalizer_processor.safetensors'))
    unnormal=load_file(str(model_dir/'policy_postprocessor_step_0_unnormalizer_processor.safetensors'))
    dep=out/'deployment';dep.mkdir(exist_ok=True)
    np.savez(dep/'cpu_weights.npz',token_stored_weight=quant,token_weight_scale=scale,
        state_weight=state['model.state_proj.weight'].detach().cpu().numpy(),state_bias=state['model.state_proj.bias'].detach().cpu().numpy(),
        action_mean=np.zeros(6,dtype='f4'),action_std=np.ones(6,dtype='f4'))
    np.savez(dep/'state_stats.npz',mean=np.zeros(6,dtype='f4'),std=np.ones(6,dtype='f4'))
    (dep/'normalization_note.json').write_text(json.dumps({'scope':'speed-only identity normalization matches current LeRobot missing canonical-feature stats behavior; not physical calibration', 'source_preprocessor_keys':list(normal), 'source_postprocessor_keys':list(unnormal), 'canonical_state_stats_present':'observation.state.mean' in normal, 'canonical_action_stats_present':'action.mean' in unnormal},indent=2))
    for name in ['config.json','policy_preprocessor.json','policy_postprocessor.json']:
        shutil.copy2(model_dir/name,dep/name)
    for name in ['tokenizer.json','tokenizer_config.json']:
        shutil.copy2(assets/name,dep/name)
    cfg_hashes={n:sha(dep/n) for n in ['config.json','policy_preprocessor.json','tokenizer.json','tokenizer_config.json']}
    (dep/'preprocess_export.json').write_text(json.dumps({'config_hashes':cfg_hashes,'hashes':{'state_stats.npz':sha(dep/'state_stats.npz')},'source_checkpoint_sha256':checkpoint_sha},indent=2))
    (dep/'replay.json').write_text(json.dumps({'scope':'new smolvla_base native single-camera six-action speed probe','checkpoint_sha256':checkpoint_sha,
        'num_steps':10,'min_period':.004,'max_period':4.,'expert_hidden_size':720,'hashes':{'cpu_weights.npz':sha(dep/'cpu_weights.npz')}},indent=2))
    raw_image=np.clip(np.rint(obs['observation.images.camera1'].numpy()*255),0,255).astype('u1')
    np.savez_compressed(out/'inputs.npz',image1=raw_image[None],image2=raw_image[None],state=obs['observation.state'].numpy()[None],task=np.array([obs['task']]),noise=noise.cpu().numpy()[None])
    # Rebuild RKLLM's container from new base language weights, never an old QAT language artifact.
    language=out/'language';hf=language/'hf_model';hf.mkdir(parents=True,exist_ok=True)
    weights={'model.'+k[len(prefix):]:v.detach().float().cpu().contiguous() for k,v in state.items() if k.startswith(prefix)}
    cfg=LlamaConfig(vocab_size=49280,hidden_size=960,intermediate_size=2560,num_hidden_layers=16,num_attention_heads=15,num_key_value_heads=5,head_dim=64,rms_norm_eps=1e-5,rope_theta=10000,max_position_embeddings=8192,tie_word_embeddings=True,pad_token_id=2,bos_token_id=1,eos_token_id=2)
    cfg.architectures=['LlamaForCausalLM'];cfg.save_pretrained(hf);save_file(weights,str(hf/'model.safetensors'),metadata={'format':'pt'})
    for f in assets.iterdir():
        if f.is_file() and ('token' in f.name or f.name in ['vocab.json','merges.txt']):shutil.copy2(f,hf/f.name)
    (hf/'extraction.json').write_text(json.dumps({'checkpoint_sha256':checkpoint_sha,'source':'lerobot/smolvla_base','layers':16,'rope_theta':10000,'scope':'new base weights in RKLLM causal container with separately patched runtime'},indent=2))
    valid=np.flatnonzero(capture['mask'][0].numpy().any(axis=0));n=len(valid)
    compact=capture['prefix'][:,valid].contiguous();mask=capture['mask'][:,valid][:,:,valid]
    sample={'inputs_embeds':compact,'attention_mask':torch.where(mask[:,None],torch.tensor(0.,dtype=torch.float32),torch.tensor(torch.finfo(torch.float32).min)), 'position_ids':capture['positions'][:,valid].clone()}
    with (language/'sample.pkl').open('wb') as f:pickle.dump(sample,f)
    (language/'dataset.json').write_text(json.dumps([{'sample':'sample.pkl','token_nums':n}]))
    report={'source_model':'lerobot/smolvla_base','checkpoint_sha256':checkpoint_sha,'output_shape':[1,50,6],'single_camera':True,'prefix_tokens':113,'valid_language_tokens':n,
        'expert_onnx_sha256':sha(out/'expert_full.onnx'),'language_weights_sha256':sha(hf/'model.safetensors'),'cpu_weights_sha256':sha(dep/'cpu_weights.npz'),'quality_validated':False}
    (out/'source.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
