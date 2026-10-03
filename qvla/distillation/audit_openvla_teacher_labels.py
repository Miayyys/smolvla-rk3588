#!/usr/bin/env python3
"""Audit real teacher labels against their exact training observations and actions.

Agreement with recorded actions is a convention diagnostic, not task success.
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

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from qvla.haq.distillation import TeacherCache
from qvla.haq.offline_actions import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','inputs','dataset-root','splits','partition','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    split=json.loads(args.splits.read_text());partition=json.loads(args.partition.read_text())
    reserved=set(partition['development_episode_ids_from_qat_train'])|set(split['splits']['test'])|set(split['splits']['ptq_calibration'])
    allowed=set(split['splits']['qat_train'])-reserved
    cache=TeacherCache(args.cache,partition_sha256=file_sha256(args.partition),split_sha256=file_sha256(args.splits),allowed_episodes=allowed)
    source=json.loads((args.inputs/'manifest.json').read_text())
    if cache.rows!=source['rows'] or cache.manifest['observations_sha256']!=file_sha256(args.inputs/'observations.npz'):
        raise ValueError('Teacher observations/row identities changed')
    episodes=sorted({r['episode_index'] for r in cache.rows})
    data=LeRobotDataset('lerobot/libero',root=args.dataset_root,episodes=episodes,video_backend='pyav',return_uint8=True,
                       delta_timestamps={'action':[i/10 for i in range(8)]})
    offsets={};start=0
    for e in episodes:offsets[e]=start;start+=int(data.meta.episodes['length'][e])
    teachers=[];targets=[];records=[]
    with np.load(args.inputs/'observations.npz',allow_pickle=False) as obs:
        arrays={key:obs[key] for key in ('image','wrist','state')}
        for i,row in enumerate(cache.rows):
            frame=data[offsets[row['episode_index']]+row['frame_index']]
            if (int(frame['episode_index']),int(frame['frame_index']),int(frame['task_index']),frame['task'])!=(row['episode_index'],row['frame_index'],row['task_index'],row['task']):
                raise ValueError('Training frame/task identity differs')
            if not np.array_equal(frame['observation.state'].numpy(),arrays['state'][i]):raise ValueError('Teacher proprio differs from student frame')
            for key,field in [('observation.images.image','image'),('observation.images.image2','wrist')]:
                if not np.array_equal(frame[key].numpy().transpose(1,2,0),arrays[field][i]):raise ValueError('Teacher camera differs from student frame')
            valid=(~frame['action_is_pad']).numpy();valid[row['valid_length']:]=False
            t=cache.actions[i][valid];g=frame['action'].numpy()[valid]
            teachers.append(t);targets.append(g)
            records.append({'task':row['task_index'],'episode':row['episode_index'],'frame':row['frame_index'],
                            'matched_steps':int(valid.sum()),'first6_mae':float(np.abs(t[:,:6]-g[:,:6]).mean()),
                            'gripper_agreement':float((t[:,6]==g[:,6]).mean())})
            if (i+1)%100==0:print(json.dumps({'stage':'audit','observations':i+1,'total':len(cache.rows)}),flush=True)
    t=np.concatenate(teachers);g=np.concatenate(targets)
    if not np.isfinite(t).all() or not np.isin(t[:,6],[-1,0,1]).all():raise ValueError('Invalid simulator action output')
    report={'status':'real_teacher_observation_and_action_contract_audit_passed',
            'teacher_execution_verified':True,'closed_loop_quality_verified':False,
            'teacher_manifest_sha256':file_sha256(args.cache/'manifest.json'),
            'samples':len(records),'matched_action_steps':len(t),'teacher_min_per_dimension':t.min(0).tolist(),
            'teacher_max_per_dimension':t.max(0).tolist(),'teacher_std_per_dimension':t.std(0).tolist(),
            'first6_mae_to_demonstration':float(np.abs(t[:,:6]-g[:,:6]).mean()),
            'gripper_agreement_to_demonstration':float((t[:,6]==g[:,6]).mean()),'rows':records,
            'scope':'same_training_frame_contract_only; expert_behavior_can_differ_from_demonstration'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
