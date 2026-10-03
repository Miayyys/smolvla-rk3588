#!/usr/bin/env python3
"""Recompute exact selected local-pack action reference for RKNN conversion replay."""

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
from qvla.haq.runtime import apply_assignment
from qvla.runtime.smolvla_numpy_glue import assemble_prefix, postprocess


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-dir',type=Path,default=ROOT/'artifacts/model');p.add_argument('--vlm-assets-dir',type=Path,default=ROOT/'artifacts/smolvlm2_assets')
    p.add_argument('--run',type=Path,default=ROOT/'runs/qat_distilled_v1_no_teacher_v1');p.add_argument('--root',type=Path,default=ROOT/'runs/qat_v1_rknn_deploy_v1');p.add_argument('--device',default='cuda')
    a=p.parse_args();torch.set_num_threads(4);cfg=json.loads((ROOT/'config/haq_candidate_v1.json').read_text());artifact=ROOT/cfg['artifact_directory']
    r=json.loads((a.run/'report.json').read_text());pack=a.run/'qat_local_packed.safetensors'
    if file_sha256(pack)!=r['packed_sha256']:raise ValueError('Selected pack changed')
    policy,_,_=load_policy(a);space=json.loads((artifact/'space.json').read_text());ranges=json.loads((artifact/'calibration.json').read_text())['ranges']
    originals={s['module']:copy.deepcopy(policy.get_submodule(s['module'])).cpu() for s in space['action_sites']}
    apply_assignment(policy,space,cfg['assignment'],ranges,originals);policy.load_state_dict(load_file(str(pack)),strict=True)
    for name in cfg['assignment']:policy.get_submodule(name).refresh()
    policy.to(a.device).eval();core=policy.model;module=core.vlm_with_expert;features=[];capture={}
    oldimage=module.embed_image;oldprefix=core.embed_prefix
    def image(x):
        y=oldimage(x);features.append(y.detach().float().cpu().numpy());return y
    def prefix(*args,**kw):
        y=oldprefix(*args,**kw);capture['prefix']=y[0].detach().float().cpu().numpy();return y
    module.embed_image=image;core.embed_prefix=prefix
    with np.load(a.root/'replay_inputs.npz') as z:inp={k:z[k] for k in z.files}
    try:
        with torch.inference_mode():
            raw=core.sample_actions(images=[torch.from_numpy(x).to(a.device) for x in inp['images']],
                img_masks=[torch.from_numpy(x).to(a.device) for x in inp['image_masks']],
                lang_tokens=torch.from_numpy(inp['lang_tokens']).to(a.device),lang_masks=torch.from_numpy(inp['lang_masks']).to(a.device),
                state=torch.from_numpy(inp['state']).to(a.device),noise=torch.from_numpy(inp['noise']).to(a.device)).detach().float().cpu().numpy()
    finally:module.embed_image=oldimage;core.embed_prefix=oldprefix
    with np.load(a.root/'cpu_weights.npz') as z:weights={k:z[k] for k in z.files}
    cpu_prefix=assemble_prefix(features,inp,weights)[0];delta=np.abs(cpu_prefix-capture['prefix'])
    if not np.allclose(cpu_prefix,capture['prefix'],rtol=1e-6,atol=1e-5):raise ValueError('Selected quantized CPU prefix mismatch')
    actions=postprocess(raw,weights);np.savez_compressed(a.root/'fp_reference.npz',actions=actions,raw_actions=raw,prefix=capture['prefix'],features=np.stack(features))
    report=dict(scope='selected_local_QAT_pack_reference_not_original_FP',checkpoint_sha256=r['source_weight_sha256'],selected_pack_sha256=r['packed_sha256'],
        sample=dict(episode_index=18,task_index=0,frame_index=0,action_seed=2416662958),num_steps=core.config.num_steps,
        min_period=core.config.min_period,max_period=core.config.max_period,expert_hidden_size=module.expert_hidden_size,
        cpu_prefix_parity_max_abs=float(delta.max()),reference_kind='selected_local_QAT_pack',
        hashes={name:file_sha256(a.root/name) for name in ('replay_inputs.npz','cpu_weights.npz','fp_reference.npz')})
    (a.root/'replay.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
