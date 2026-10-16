"""Separate training fit from validation fit using fixed recorded action chunks.

No test sources, optimizer updates, teacher models, or quantization are used.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('config/so101_sim_finetune_matched_v1.json'))
    parser.add_argument('--output',type=Path,default=Path('runs/so101_sim_training_diagnostic_v1/offline'))
    args=parser.parse_args();cfg=json.loads(args.config.read_text());run=Path(cfg['output'])
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from lerobot.policies import make_pre_post_processors
    from qvla.evaluation.finetune_so100_real import Frames,sha
    from qvla.evaluation.haq_offline_eval import load_policy
    prepared=json.loads((run/'prepared.json').read_text());stats=json.loads((run/'stats.json').read_text())
    frames=Frames(run,cfg['task']);data=frames.data
    torch.set_num_threads(4)
    model=run/'best'
    policy,_,_=load_policy(SimpleNamespace(model_dir=model.resolve(),vlm_assets_dir=Path(cfg['vlm_assets_dir']).resolve(),device='cuda'))
    policy.float().eval()
    pre,post=make_pre_post_processors(policy.config,str(model),preprocessor_overrides={
        'device_processor':{'device':'cuda'},'tokenizer_processor':{'tokenizer_name':str(Path(cfg['vlm_assets_dir']).resolve())},
        'normalizer_processor':{'stats':stats}},postprocessor_overrides={'unnormalizer_processor':{'stats':stats}})
    args.output.mkdir(exist_ok=True,parents=True)
    report={'scope':'fixed train and validation observation diagnostics; excludes test sources',
        'model_sha256':sha(model/'model.safetensors'),'stats_sha256':sha(run/'stats.json'),'results':{}}
    for partition,episodes in [('train',prepared['training_successful_sources']),('validation',prepared['splits']['validation'])]:
        prediction_errors=[];hold_errors=[];first_errors=[];rows_report=[];predictions=[];targets=[]
        for ep in episodes:
            selected=np.flatnonzero(data['episodes']==ep)
            rows=np.linspace(selected[0],max(selected[0],selected[-1]-49),5,dtype=int)
            for row in rows:
                raw=frames.batch([int(row)])
                target=raw['action'].numpy()[0].copy();valid=~raw['action_is_pad'].numpy()[0]
                hold=np.broadcast_to(raw['observation.state'].numpy()[0],target.shape).copy()
                torch.manual_seed(100000+int(row))
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    pred=post(policy.predict_action_chunk(pre(raw))).float().cpu().numpy()[0]
                error=np.abs(pred[valid]-target[valid]);prediction_errors.append(error)
                hold_errors.append(np.abs(hold[valid]-target[valid]));first_errors.append(np.abs(pred[0]-target[0]))
                predictions.append(pred);targets.append(target)
                rows_report.append({'episode':int(ep),'row':int(row),'first_action_mae':first_errors[-1].tolist(),
                    'chunk_per_joint_mae':error.mean(0).tolist()})
        mae=np.concatenate(prediction_errors).mean(0)
        hold_mae=np.concatenate(hold_errors).mean(0)
        result={'observations':len(prediction_errors),'per_joint_mae':mae.tolist(),
            'first_action_per_joint_mae':np.stack(first_errors).mean(0).tolist(),
            'normalized_action_mae':float(np.mean(mae/np.asarray(stats['action']['std']))),
            'hold_current_state_per_joint_mae':hold_mae.tolist(),
            'hold_current_state_normalized_mae':float(np.mean(hold_mae/np.asarray(stats['action']['std']))),
            'units':'five arm degrees, gripper percentage points','rows':rows_report}
        report['results'][partition]=result
        np.savez_compressed(args.output/f'{partition}.npz',predictions=predictions,targets=targets)
        print(json.dumps({'partition':partition,**{k:v for k,v in result.items() if k!='rows'}}),flush=True)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
