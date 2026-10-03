#!/usr/bin/env python3
"""Compare NumPy/tokenizers preprocessing with the pinned original processor."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from haq_offline_eval import identity,load_policy
from qvla_haq.offline_actions import file_sha256,load_cache
from smolvla_board_preprocess import SmolVLAPreprocessor
from verify_smolvla_vision_float import metrics
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name,default in [('model-dir','artifacts/transfer/model'),('vlm-assets-dir','artifacts/transfer/smolvlm2_assets'),
            ('splits','data/libero_splits.json'),('partition','config/evaluation_partition_v2.json'),
            ('cache','runs/haq_offline_local_v1/fp_cache40'),('root','runs/smolvla_raw_board_v1')]:
        p.add_argument('--'+name,type=Path,default=ROOT/default)
    p.add_argument('--device',default='cuda');args=p.parse_args();torch.set_num_threads(4)
    ident=identity(args);manifest,obs,fp=load_cache(args.cache,ident)
    args.root.mkdir(parents=True,exist_ok=True)
    stats=load_file(str(args.model_dir/'policy_preprocessor_step_5_normalizer_processor.safetensors'))
    np.savez(args.root/'state_stats.npz',mean=stats['observation.state.mean'].numpy(),std=stats['observation.state.std'].numpy())
    processor=SmolVLAPreprocessor(args.vlm_assets_dir/'tokenizer.json',args.vlm_assets_dir/'tokenizer_config.json',
        args.model_dir/'policy_preprocessor.json',args.model_dir/'config.json',args.root/'state_stats.npz')
    policy,pre,_=load_policy(args)
    with np.load(ROOT/'runs/smolvla_full_board_v1/replay_inputs.npz') as z:noise=z['noise'].copy();old_inputs={k:z[k].copy() for k in z.files}
    refs={'lang_tokens':[],'lang_masks':[],'state':[]};rows=[]
    for i in range(len(manifest['samples'])):
        raw={'observation.images.image':torch.from_numpy(obs['image1'][i].copy()).float()/255,
             'observation.images.image2':torch.from_numpy(obs['image2'][i].copy()).float()/255,
             'observation.state':torch.from_numpy(obs['state'][i].copy()),'task':str(obs['task'][i])}
        with torch.inference_mode():
            original=pre(raw);images,masks=policy.prepare_images(original);state=policy.prepare_state(original)
        ours=processor(obs['image1'][i],obs['image2'][i],obs['state'][i],str(obs['task'][i]),noise)
        reference_images=np.stack([x.detach().cpu().numpy() for x in images])
        tokens=original['observation.language.tokens'].cpu().numpy();langmask=original['observation.language.attention_mask'].cpu().numpy()
        ref_state=state.cpu().numpy()
        np.testing.assert_array_equal(ours['lang_tokens'],tokens);np.testing.assert_array_equal(ours['lang_masks'],langmask)
        np.testing.assert_allclose(ours['images'],reference_images,rtol=0,atol=5e-7)
        np.testing.assert_allclose(ours['state'],ref_state,rtol=0,atol=1e-6)
        rows.append({'sample':manifest['samples'][i],'image_error':metrics(ours['images'],reference_images),
                     'state_error':metrics(ours['state'],ref_state),'token_ids_exact':True,'token_mask_exact':True})
        for key,value in [('lang_tokens',tokens[0]),('lang_masks',langmask[0]),('state',ref_state[0])]:refs[key].append(value)
        if i==0:
            for key in old_inputs:np.testing.assert_allclose(ours[key],old_inputs[key],rtol=0,atol=5e-7)
    # Exercise actual padding/truncation and newline behavior against the same original tokenizer.
    text_cases=['','hello\n','Pick up the cup!','打开抽屉，然后拿起杯子。',str(obs['task'][0])*12]
    ref_tokenizer=next(step.input_tokenizer for step in pre.steps if hasattr(step,'input_tokenizer'))
    texts=[x if x.endswith('\n') else x+'\n' for x in text_cases]
    expected=ref_tokenizer(texts,max_length=48,truncation=True,padding='max_length',padding_side='right',return_tensors='np')
    tokens,masks=processor.tokenize(text_cases)
    np.testing.assert_array_equal(tokens,expected['input_ids']);np.testing.assert_array_equal(masks,expected['attention_mask'])
    np.savez_compressed(args.root/'raw_panel.npz',image1=obs['image1'],image2=obs['image2'],state=obs['state'],task=obs['task'])
    np.savez_compressed(args.root/'preprocess_reference.npz',**{k:np.stack(v) for k,v in refs.items()},
                        extra_token_ids=expected['input_ids'],extra_token_masks=expected['attention_mask'])
    np.savez_compressed(args.root/'raw_inputs.npz',image1=obs['image1'][0],image2=obs['image2'][0],
                        state=obs['state'][0],task=obs['task'][0],noise=noise)
    report={'scope':'40 raw development observations compared with original GPU processor; no network/closed-loop evaluation',
        'checkpoint_sha256':ident['checkpoint_sha256'],'sample':manifest['samples'][0],
        'tokenizers_version':__import__('tokenizers').__version__,'rows':rows,'extra_text_cases':text_cases,
        'extra_text_ids_masks_exact':True,'input_convention':'RGB uint8 CHW, raw state of size8, literal task string',
        'vision_panel_sha256':file_sha256(ROOT/'runs/smolvla_board_vision_panel_v1/vision_panel.npz'),
        'config_hashes':{name:file_sha256(path) for name,path in [('tokenizer.json',args.vlm_assets_dir/'tokenizer.json'),
         ('tokenizer_config.json',args.vlm_assets_dir/'tokenizer_config.json'),('policy_preprocessor.json',args.model_dir/'policy_preprocessor.json'),('config.json',args.model_dir/'config.json')]},
        'hashes':{n:file_sha256(args.root/n) for n in ('raw_panel.npz','preprocess_reference.npz','raw_inputs.npz','state_stats.npz')}}
    (args.root/'preprocess_export.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'samples':len(rows),'max_image_error':max(r['image_error']['max_abs_error'] for r in rows),
        'max_state_error':max(r['state_error']['max_abs_error'] for r in rows),'all_token_ids_masks_exact':True},indent=2))


if __name__=='__main__':main()
