#!/usr/bin/env python3
"""Losslessly restore source BF16 storage and verify unchanged real model actions."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file,save_file

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qat_train_w8a8_stage1 import load_policy
from qvla_haq.runtime import apply_assignment
from qvla_haq.training_development import DevelopmentChunks


def main():
    torch.set_num_threads(8)
    root=ROOT/'runs/v2_ptq_controls_v1';summary=json.loads((root/'summary.json').read_text())
    args=SimpleNamespace(model_dir=ROOT/'artifacts/model',vlm_assets_dir=ROOT/'artifacts/smolvlm2_assets',
        dataset_root=ROOT/'data/libero')
    candidate=json.loads((ROOT/'config/haq_candidate_v2.json').read_text())
    artifact=ROOT/candidate['artifact_directory'];space=json.loads((artifact/'space.json').read_text())
    cal=json.loads((artifact/'calibration.json').read_text())
    names=[s['module'] for s in space['action_sites']];biases={n+'.bias' for n in names}
    for label in ('student_ptq','original_ptq'):
        out=root/label;path=out/'ptq_local_packed.safetensors';report=json.loads((out/'report.json').read_text())
        old_hash=hashlib.sha256(path.read_bytes()).hexdigest()
        if old_hash!=report['packed_sha256']:raise ValueError('PTQ file identity changed')
        state=load_file(str(path));before_bytes=path.stat().st_size;changed=[]
        with safe_open(str(args.model_dir/'model.safetensors'),framework='pt') as source:
            for key in source.keys():
                if key in state and key not in biases and source.get_slice(key).get_dtype()=='BF16' and state[key].dtype!=torch.bfloat16:
                    restored=state[key].to(torch.bfloat16)
                    if not torch.equal(restored.float(),state[key].float()):raise ValueError('Lossy change prohibited')
                    state[key]=restored;changed.append(key)
        temp=out/'ptq_normalized.tmp.safetensors';save_file(state,str(temp))
        policy,_,_=load_policy(args);policy.cpu()
        originals={n:copy.deepcopy(policy.get_submodule(n)).cpu() for n in names}
        apply_assignment(policy,space,candidate['assignment'],cal['ranges'],originals);del originals
        policy.load_state_dict(load_file(str(temp)),strict=True)
        for n in names:policy.get_submodule(n).refresh()
        policy.cuda().eval()
        from lerobot.policies import make_pre_post_processors
        pre,post=make_pre_post_processors(policy.config,str(args.model_dir),preprocessor_overrides={
            'tokenizer_processor':{'tokenizer_name':str(args.vlm_assets_dir)},
            'rename_observations_processor':{'rename_map':{'observation.images.image':'observation.images.camera1',
                                                          'observation.images.image2':'observation.images.camera2'}}})
        partition=json.loads((ROOT/'config/evaluation_partition_v2.json').read_text())
        split=json.loads((ROOT/'data/libero_splits.json').read_text())
        panel=DevelopmentChunks(args,partition,split,[])
        _,after,_,_=panel.evaluate(policy,pre,post)
        with np.load(out/'reload_actions.npz') as saved:old=saved['after']
        diff=np.abs(old-after)
        if diff.max()!=0:raise ValueError('Storage repair changed real actions')
        audit=dict(old_packed_sha256=old_hash,old_bytes=before_bytes,changed_storage_keys=changed,
            all_tensor_values_exactly_preserved=True,real_40_observation_action_mae=float(diff.mean()),
            real_40_observation_action_max_abs=float(diff.max()),new_bytes=temp.stat().st_size)
        temp.replace(path)
        report.update(packed_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),packed_file_bytes=path.stat().st_size,
            packed_compression_fraction=1-path.stat().st_size/(args.model_dir/'model.safetensors').stat().st_size,
            storage_normalization=audit)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        for key in ('packed_file_bytes','packed_sha256'):summary['results'][label][key]=report[key]
        summary['results'][label]['compression']=report['packed_compression_fraction']
        summary['results'][label]['storage_normalization']=audit
        (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(dict(label=label,**audit)),flush=True)
        del policy,state,panel;torch.cuda.empty_cache()


if __name__=='__main__':main()
