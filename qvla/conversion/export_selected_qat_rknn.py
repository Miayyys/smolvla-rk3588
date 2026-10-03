#!/usr/bin/env python3
"""Export selected QAT masters into three fixed RKNN graphs and isolated calibration."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
import torch
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from qvla.evaluation.haq_offline_eval import load_policy
from qvla.haq.offline_actions import file_sha256
from qvla.conversion.export_smolvla_vision_onnx import VisionAndConnector
from qvla.conversion.export_smolvla_prefix_onnx import PrefixWithKV, export_rope
from qvla.conversion.export_smolvla_expert_step_onnx import ExpertStep
from qvla.runtime.smolvla_numpy_glue import time_embedding


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-dir',type=Path,default=ROOT/'artifacts/model')
    p.add_argument('--vlm-assets-dir',type=Path,default=ROOT/'artifacts/smolvlm2_assets')
    p.add_argument('--run',type=Path,default=ROOT/'runs/qat_distilled_v1_no_teacher_v1')
    p.add_argument('--candidate',type=Path,default=ROOT/'config/haq_candidate_v1.json')
    p.add_argument('--replay-inputs',type=Path,required=True)
    p.add_argument('--dataset-root',type=Path,default=ROOT/'data/libero')
    p.add_argument('--output',type=Path,default=ROOT/'runs/qat_v1_rknn_deploy_v1')
    p.add_argument('--device',default='cuda')
    p.add_argument('--calibration-episodes',type=int,default=40)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    import onnx
    import lerobot.policies.smolvla.smolvlm_with_expert as source
    r=json.loads((a.run/'report.json').read_text());cfg=json.loads(a.candidate.read_text())
    master=a.run/'qat_float_master.safetensors';pack_path=a.run/'qat_local_packed.safetensors'
    if r['teacher_loss_used'] or r['candidate_version']!='v1':raise ValueError('Wrong chosen QAT')
    for path,h in [(master,r['master_sha256']),(pack_path,r['packed_sha256']),
                   (a.candidate,r['candidate_config_sha256']),(a.model_dir/'model.safetensors',r['source_weight_sha256'])]:
        if file_sha256(path)!=h:raise ValueError('Changed source '+str(path))
    report=dict(scope='selected_QAT_FP_master_export_for_RKNN_post_training_calibration_not_identical_local_pack',
                master_sha256=r['master_sha256'],packed_sha256=r['packed_sha256'],
                source_checkpoint_sha256=r['source_weight_sha256'],candidate_sha256=r['candidate_config_sha256'],
                calibration_episode_ids=r['calibration_episode_ids'][:a.calibration_episodes],
                RKNN_conversion_verified=False,board_execution_verified=False)
    (a.output/'source.json').write_text(json.dumps(report,indent=2)+'\n')
    policy,pre,post=load_policy(a);policy=policy.float().eval();state=load_file(str(master))
    if set(state)!=set(policy.state_dict()):raise ValueError('Master keys differ from native policy')
    policy.load_state_dict(state,strict=True);del state
    policy.model.vlm_with_expert.vlm.set_attn_implementation('eager')
    core=policy.model;module=core.vlm_with_expert
    with np.load(a.replay_inputs) as z:inp={k:z[k].copy() for k in z.files}
    capture={};oldforward=module.forward;oldstep=core.denoise_step
    def prefix_hook(*args,**kw):
        out=oldforward(*args,**kw)
        if kw['inputs_embeds'][0] is not None and not capture:
            capture['prefix']=tuple(x.detach().float().cpu() if x.is_floating_point() else x.detach().cpu()
                                    for x in (kw['inputs_embeds'][0],kw['attention_mask'],kw['position_ids']))
        return out
    def step_hook(*args,**kw):
        if 'expert' not in capture:
            t=kw['timestep'].detach().float().cpu().numpy()
            values=[kw['x_t'].detach().float().cpu(),torch.from_numpy(time_embedding(t,module.expert_hidden_size,core.config.min_period,core.config.max_period)),kw['prefix_pad_masks'].detach().cpu()]
            for layer in kw['past_key_values'].layers:values.extend((layer.keys.detach().float().cpu(),layer.values.detach().float().cpu()))
            capture['expert']=tuple(values)
        return oldstep(*args,**kw)
    module.forward=prefix_hook;core.denoise_step=step_hook
    try:
        with torch.inference_mode():
            out=core.sample_actions(images=[torch.from_numpy(x).to(a.device) for x in inp['images']],
                img_masks=[torch.from_numpy(x).to(a.device) for x in inp['image_masks']],
                lang_tokens=torch.from_numpy(inp['lang_tokens']).to(a.device),
                lang_masks=torch.from_numpy(inp['lang_masks']).to(a.device),
                state=torch.from_numpy(inp['state']).to(a.device),noise=torch.from_numpy(inp['noise']).to(a.device))
    finally:module.forward=oldforward;core.denoise_step=oldstep
    np.savez_compressed(a.output/'master_reference.npz',raw_actions=out.detach().float().cpu().numpy())
    policy=policy.cpu();oldrope=source.apply_rope;source.apply_rope=export_rope
    try:
        for kind,wrapper,values,inputs,outputs in [
          ('vision',VisionAndConnector(policy),(torch.from_numpy(inp['images'][0]),),['pixel'],['features']),
          ('prefix',PrefixWithKV(module),capture['prefix'],['prefix','attention_mask','position_ids'],['prefix_hidden']+[f'{k}_{i}' for i in range(16) for k in ('key','value')]),
          ('expert',ExpertStep(core),capture['expert'],['noisy_actions','time_embedding','prefix_pad_masks']+[f'{k}_{i}' for i in range(16) for k in ('key','value')],['velocity'])]:
            path=a.output/(kind+'.onnx');print('export',kind,flush=True)
            wrapper.eval().requires_grad_(False)
            with torch.inference_mode():
                actual=wrapper(*values);actuals=actual if isinstance(actual,tuple) else (actual,)
                torch.onnx.export(wrapper,values,str(path),dynamo=False,opset_version=17,input_names=inputs,output_names=outputs,do_constant_folding=True)
            m=onnx.load(str(path))
            if kind=='expert':
                reductions=[n for n in m.graph.node if n.op_type=='ReduceMin']
                if len(reductions)!=1:raise ValueError('Unexpected position reduction')
                n=reductions[0]
                if {x.name:onnx.helper.get_attribute_value(x) for x in n.attribute}!={'axes':[1],'keepdims':1}:raise ValueError('Unexpected minimum axes')
                ix='qvla_first_suffix_position';m.graph.initializer.append(onnx.numpy_helper.from_array(np.array([0],dtype=np.int64),ix))
                n.CopyFrom(onnx.helper.make_node('Gather',[n.input[0],ix],list(n.output),name='QVLAMonotonePositionMinimum',axis=1));onnx.save(m,str(path))
            onnx.checker.check_model(m)
            special={}
            for name,fmt in cfg['assignment'].items():
                if fmt in ('w8a8','cpu_int8_row_lookup'):continue
                # Match weights, not fuzzy layer numbers: exported initializer names
                # have the wrapper's root path in place of the original model prefix.
                if kind=='expert' and name.startswith('model.vlm_with_expert.lm_expert.'):
                    weight='core.'+name.removeprefix('model.')+'.weight'
                elif kind=='prefix' and name.startswith('model.vlm_with_expert.vlm.model.text_model.'):
                    weight='module.'+name.removeprefix('model.vlm_with_expert.')+'.weight'
                else:continue
                initial={x.name:x for x in m.graph.initializer};matched=[]
                for node in m.graph.node:
                    for input_name in node.input:
                        if input_name in initial and onnx.numpy_helper.to_array(initial[input_name]).shape==tuple(policy.get_submodule(name).weight.shape[::-1]) and input_name.startswith('onnx::MatMul'):
                            w=onnx.numpy_helper.to_array(initial[input_name]);expected=policy.get_submodule(name).weight.detach().numpy().T
                            if np.array_equal(w,expected):matched.append(node.output[0])
                        elif input_name==weight:matched.append(node.output[0])
                if len(set(matched))!=1:raise ValueError(f'Ambiguous precision node {kind} {name}: {matched}')
                special[name]=dict(format=fmt,onnx_output=matched[0])
            np.savez_compressed(a.output/(kind+'_boundary.npz'),**{k:v.numpy() for k,v in zip(inputs,values)})
            np.savez_compressed(a.output/(kind+'_reference.npz'),**{k:v.detach().numpy() for k,v in zip(outputs,actuals)})
            metadata=dict(onnx_sha256=file_sha256(path),onnx_bytes=path.stat().st_size,input_names=inputs,output_names=outputs,special=special)
            (a.output/(kind+'_export.json')).write_text(json.dumps(metadata,indent=2)+'\n');print(metadata,flush=True)
            del m,wrapper
    finally:source.apply_rope=oldrope
    # Preserve actual selected CPU embedding and state quantization metadata.
    pack=load_file(str(pack_path));stats=load_file(str(a.model_dir/'policy_postprocessor_step_0_unnormalizer_processor.safetensors'))
    weights={}
    for alias,name in [('token','model.vlm_with_expert.vlm.model.text_model.embed_tokens'),('state','model.state_proj')]:
        for leaf in ('stored_weight','weight_scale','activation_scale','activation_zero','bias'):
            key=name+'.'+leaf
            if key in pack:weights[alias+'_'+leaf]=pack[key].numpy()
    weights.update(action_mean=stats['action.mean'].numpy(),action_std=stats['action.std'].numpy())
    np.savez(a.output/'cpu_weights.npz',**weights);del pack
    # Calibration is from the original isolated calibration episodes, never the replay/development inputs.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    episodes=report['calibration_episode_ids'];dataset=LeRobotDataset('lerobot/libero',root=a.dataset_root,episodes=episodes,video_backend='pyav',return_uint8=True)
    policy=policy.to(a.device);offset=0;files={k:[] for k in ('vision','prefix','expert')};rows=[]
    current={};oldimage=module.embed_image
    def save_case(kind,values):
        folder=a.output/'calibration'/kind;folder.mkdir(parents=True,exist_ok=True)
        index=len(files[kind]);paths=[]
        for j,t in enumerate(values):
            path=folder/f'{index:04d}_{j:02d}.npy';np.save(path,t.detach().cpu().numpy());paths.append(str(path.resolve()))
        files[kind].append(' '.join(paths))
    def image_hook(x):save_case('vision',[x]);return oldimage(x)
    def prefix_cal(*args,**kw):
        if kw['inputs_embeds'][0] is not None:save_case('prefix',[kw['inputs_embeds'][0],kw['attention_mask'],kw['position_ids']])
        return oldforward(*args,**kw)
    def expert_cal(*args,**kw):
        i=current.get('step',0);current['step']=i+1
        if i in (0,5,9):
            values=[kw['x_t'],torch.from_numpy(time_embedding(kw['timestep'].detach().cpu().numpy(),module.expert_hidden_size,core.config.min_period,core.config.max_period)),kw['prefix_pad_masks']]
            for layer in kw['past_key_values'].layers:values.extend([layer.keys,layer.values])
            save_case('expert',values)
        return oldstep(*args,**kw)
    module.embed_image=image_hook;module.forward=prefix_cal;core.denoise_step=expert_cal
    try:
        for ep in episodes:
            frame=dataset[offset];length=int(dataset.meta.episodes['length'][ep]);offset+=length
            if int(frame['episode_index'])!=ep or int(frame['frame_index'])!=0:raise ValueError('Calibration ordering changed')
            current['step']=0
            obs={k:frame[k].unsqueeze(0) for k in ('observation.images.image','observation.images.image2','observation.state')}
            for k in ('observation.images.image','observation.images.image2'):obs[k]=obs[k].float()/255
            obs['task']=[frame['task']];policy.reset();torch.manual_seed(29+ep);torch.cuda.manual_seed_all(29+ep)
            with torch.inference_mode():policy.predict_action_chunk(pre(obs))
            rows.append(dict(episode_index=ep,task_index=int(frame['task_index']),frame_index=0,noise_seed=29+ep));print('calibration',len(rows),len(episodes),flush=True)
    finally:module.embed_image=oldimage;module.forward=oldforward;core.denoise_step=oldstep
    for k,lines in files.items():(a.output/(k+'_dataset.txt')).write_text('\n'.join(lines)+'\n')
    report.update(status='exported',calibration_rows=rows,calibration_counts={k:len(v) for k,v in files.items()},
                  cpu_weights_sha256=file_sha256(a.output/'cpu_weights.npz'))
    (a.output/'export_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
