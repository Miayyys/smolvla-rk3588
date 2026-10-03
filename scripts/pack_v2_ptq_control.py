#!/usr/bin/env python3
"""Independent original/student FP PTQ under frozen v2 calibration. No optimizer."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from safetensors import safe_open

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qvla_haq.offline_actions import file_sha256,validate_partition
from qvla_haq.runtime import apply_assignment
from qvla_haq.training_development import DevelopmentChunks
from qat_train_w8a8_stage1 import load_policy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir','vlm-assets-dir','dataset-root','splits','partition','candidate','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--student-run',type=Path)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8);torch.manual_seed(29);started=time.time()
    read=lambda x:json.loads(x.read_text())
    candidate=read(args.candidate);artifact=args.candidate.parent.parent/candidate['artifact_directory']
    if file_sha256(args.model_dir/'model.safetensors')!=candidate['source_checkpoint_sha256']:raise ValueError('FP source changed')
    for name in ('space.json','assignment.json','calibration.json'):
        if file_sha256(artifact/name)!=candidate['files_sha256'][name]:raise ValueError('Map/calibration changed')
    space=read(artifact/'space.json');cal=read(artifact/'calibration.json');assignment=candidate['assignment']
    if read(artifact/'assignment.json')!=assignment:raise ValueError('Assignment differs')
    split=read(args.splits);partition=read(args.partition)
    dev=set(validate_partition(partition,split))
    if set(cal['episode_ids'])!=set(partition['calibration_episode_ids_from_ptq_calibration']):raise ValueError('Calibration selection differs')
    if dev & set(cal['episode_ids']):raise ValueError('Calibration/development overlap')
    policy,pre,post=load_policy(args);policy.cpu()
    from lerobot.policies import make_pre_post_processors
    pre,post=make_pre_post_processors(policy.config,str(args.model_dir),preprocessor_overrides={
        'tokenizer_processor':{'tokenizer_name':str(args.vlm_assets_dir.resolve())},
        'rename_observations_processor':{'rename_map':{'observation.images.image':'observation.images.camera1',
                                                       'observation.images.image2':'observation.images.camera2'}}})
    names=[s['module'] for s in space['action_sites']]
    dtypes={n:policy.get_submodule(n).weight.dtype for n in names}
    student_hash=None
    if args.student_run:
        sr=read(args.student_run/'report.json');master=args.student_run/'distilled_float_master.safetensors'
        student_hash=file_sha256(master)
        if sr['master_sha256']!=student_hash or sr['source_weight_sha256']!=candidate['source_checkpoint_sha256']:raise ValueError('Student FP identity changed')
        if sr['fake_quant_enabled'] or not sr['strict_master_reload'] or sr.get('master_storage_format','full_fp_master')!='full_fp_master':raise ValueError('Expected complete verified FP student')
        state=load_file(str(master));base=policy.state_dict()
        if set(state)!=set(base):raise ValueError('FP student state key mismatch')
        parameters=dict(policy.named_parameters());replace={n+'.weight' for n in names}|{n+'.bias' for n in names}
        for key,value in state.items():
            if base[key].shape!=value.shape:raise ValueError('FP shape mismatch')
            if key in replace:
                parent,leaf=key.rsplit('.',1)
                setattr(policy.get_submodule(parent),leaf,torch.nn.Parameter(value,requires_grad=False))
            elif not torch.equal(value.float(),base[key].float()):
                raise ValueError('Unexpected updated unconfigured tensor: '+key)
        policy.load_state_dict(state,strict=True);del state,base
    originals={n:copy.deepcopy(policy.get_submodule(n)).cpu() for n in names}
    apply_assignment(policy,space,assignment,cal['ranges'],originals);del originals
    for n in names:
        m=policy.get_submodule(n);m.original_dtype=dtypes[n];m._dtype_marker=m._dtype_marker.to(dtypes[n])
    policy.cuda().eval()
    panel=DevelopmentChunks(args,partition,split,[])
    metric,before,targets,valid=panel.evaluate(policy,pre,post)
    packed=args.output_dir/'ptq_local_packed.safetensors'
    packed_state={k:v.detach().cpu().contiguous() for k,v in policy.state_dict().items()}
    configured_biases={n+'.bias' for n in names}
    with safe_open(str(args.model_dir/'model.safetensors'),framework='pt') as source:
        for key in source.keys():
            if key in packed_state and key not in configured_biases and source.get_slice(key).get_dtype()=='BF16':
                restored=packed_state[key].to(torch.bfloat16)
                if not torch.equal(restored.float(),packed_state[key].float()):
                    raise ValueError('Frozen BF16 storage change would alter values: '+key)
                packed_state[key]=restored
    save_file(packed_state,str(packed));del packed_state
    policy.load_state_dict(load_file(str(packed)),strict=True)
    for n in names:policy.get_submodule(n).refresh()
    reloaded,after,_,_=panel.evaluate(policy,pre,post)
    diff=np.abs(before-after);parity=dict(mae=float(diff.mean()),max_abs=float(diff.max()),
        scope='40_fixed_development_observations_full_action_chunks_before_after_actual_pack_reload')
    if parity['max_abs']!=0:raise ValueError('PTQ strict reloaded action mismatch')
    np.savez_compressed(args.output_dir/'reload_actions.npz',before=before,after=after,targets=targets,valid=valid)
    report=dict(status='independently_created_FP_PTQ_real_local_pack_not_RKNN',weight_training_performed=False,
        source_kind='distilled_FP_student' if args.student_run else 'original_FP',student_master_sha256=student_hash,
        source_weight_sha256=candidate['source_checkpoint_sha256'],candidate_config_sha256=file_sha256(args.candidate),
        calibration_sha256=file_sha256(artifact/'calibration.json'),calibration_episode_ids=cal['episode_ids'],
        partition_sha256=file_sha256(args.partition),split_sha256=file_sha256(args.splits),
        packed_sha256=file_sha256(packed),packed_file_bytes=packed.stat().st_size,
        packed_compression_fraction=1-packed.stat().st_size/(args.model_dir/'model.safetensors').stat().st_size,
        strict_pack_reload=True,reload_action_parity=parity,development=metric,
        RKNN_conversion_verified=False,hardware_resource_benefits='not_measured',elapsed_seconds=time.time()-started,
        calibration_note='Fixed v2 FP calibration reused for matched QAT/PTQ control; no new student-specific calibration',
        closed_loop_quality='pending')
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='development'}),flush=True)


if __name__=='__main__':main()
