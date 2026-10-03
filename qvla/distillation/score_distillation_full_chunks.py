#!/usr/bin/env python3
"""Score all valid predicted actions against development demonstration chunks.

Uses saved real-model actions; no new inference and no task-success claim.
"""

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
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from qvla.haq.offline_actions import file_sha256,validate_partition


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation','dataset-root','splits','partition','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();report=json.loads((args.evaluation/'report.json').read_text())
    partition=json.loads(args.partition.read_text());split=json.loads(args.splits.read_text())
    episodes=validate_partition(partition,split)
    if report['partition_sha256']!=file_sha256(args.partition) or report['actions_sha256']!=file_sha256(args.evaluation/'actions.npz'):raise ValueError('Evaluation identity changed')
    if {r['episode_index'] for r in report['rows']}!=set(episodes):raise ValueError('Evaluation is not the isolated development panel')
    data=LeRobotDataset('lerobot/libero',root=args.dataset_root,episodes=episodes,video_backend='pyav',return_uint8=True,delta_timestamps={'action':[i/10 for i in range(50)]})
    offsets={};offset=0
    for e in episodes:offsets[e]=offset;offset+=int(data.meta.episodes['length'][e])
    targets=[];masks=[]
    for row in report['rows']:
        frame=data[offsets[row['episode_index']]+row['frame_index']]
        if int(frame['task_index'])!=row['task_index']:raise ValueError('Task mismatch')
        targets.append(frame['action'].numpy());masks.append((~frame['action_is_pad']).numpy())
    target=np.stack(targets);valid=np.stack(masks);metrics={}
    with np.load(args.evaluation/'actions.npz',allow_pickle=False) as saved:
        if not np.array_equal(saved['recorded_first_action'],target[:,0]):raise ValueError('Recorded action identity mismatch')
        for mode in ('original_fp','original_v2','distilled_fp','distilled_v2_qat'):
            pred=saved[mode];err=np.abs(pred-target)
            if pred.shape!=target.shape:raise ValueError('Chunk shape mismatch')
            def macro(values,mask):
                counts=mask.sum(1)*values.shape[2];chosen=counts>0
                return float(((values*mask[:,:,None]).sum((1,2))[chosen]/counts[chosen]).mean())
            prefix=valid.copy();prefix[:,8:]=False
            suffix=valid.copy();suffix[:,:8]=False
            delta_mask=valid[:,1:]&valid[:,:-1]
            delta=np.abs(np.diff(pred[:,:,:6],axis=1)-np.diff(target[:,:,:6],axis=1))
            grips=((pred[:,:,6]>0)!=(target[:,:,6]>0)).astype(float)[:,:,None]
            metrics[mode]={
                'valid_chunk_mae_vs_demonstration':macro(err,valid),
                'valid_continuous_chunk_mae_vs_demonstration':macro(err[:,:,:6],valid),
                'prefix8_mae_vs_demonstration':macro(err,prefix),
                'remaining42_mae_vs_demonstration':macro(err,suffix),
                'temporal_delta_mae_vs_demonstration':macro(delta,delta_mask),
                'gripper_sign_disagreement_vs_demonstration':macro(grips,valid),
                'per_timestep_mae_vs_demonstration':[float(err[:,t][valid[:,t]].mean()) if valid[:,t].any() else None for t in range(50)],
                'per_timestep_valid_samples':valid.sum(0).tolist()}
    result={'scope':'full_valid_development_action_chunks_and_temporal_gripper_proxies_not_success_rate',
        'source_evaluation_sha256':file_sha256(args.evaluation/'report.json'),'rows':report['rows'],
        'metrics':metrics,'valid_action_timesteps':int(valid.sum()),'closed_loop_quality':'not_measured'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(args.output.with_suffix('.npz'),recorded_action_chunks=target,valid_timestep_mask=valid)
    print(json.dumps({m:{k:v for k,v in d.items() if not k.startswith('per_timestep')} for m,d in metrics.items()}),flush=True)

if __name__=='__main__':main()
