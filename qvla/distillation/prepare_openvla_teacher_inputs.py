#!/usr/bin/env python3
"""Export isolated training observations for an independent OpenVLA-OFT worker."""

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
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from qvla.haq.offline_actions import file_sha256
from qvla.haq.training_control import stratified_frames


def canonical(text):
    return ' '.join(str(text).lower().replace('_',' ').split()).rstrip('.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--splits',type=Path,required=True)
    p.add_argument('--partition',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples-per-task',type=int,default=32)
    p.add_argument('--seed',type=int,default=29)
    args=p.parse_args()
    if args.samples_per_task<1:p.error('Need positive samples per task')
    config=ROOT/'runs/libero_local/config'
    if not (config/'config.yaml').exists():config=Path.home()/'.libero'
    if 'LIBERO_CONFIG_PATH' not in os.environ and not (config/'config.yaml').exists():
        p.error('Configure LIBERO_CONFIG_PATH first; refusing an interactive first-run prompt')
    os.environ.setdefault('LIBERO_CONFIG_PATH',str(config))
    from libero.libero import benchmark
    suite=benchmark.get_benchmark_dict()['libero_10']()
    known={canonical(suite.get_task(i).language):i for i in range(suite.n_tasks)}
    partition=json.loads(args.partition.read_text());split=json.loads(args.splits.read_text())
    if file_sha256(args.splits)!=partition['source_split_sha256']:raise ValueError('Split identity changed')
    train=sorted(set(split['splits']['qat_train'])-set(partition['development_episode_ids_from_qat_train']))
    if set(train)&(set(split['splits']['test'])|set(split['splits']['ptq_calibration'])):raise ValueError('Training partition overlaps')
    data=LeRobotDataset('lerobot/libero',root=args.dataset_root,episodes=train,video_backend='pyav',return_uint8=True)
    if data.fps!=10:raise ValueError('Dataset FPS changed; recheck chunk alignment')
    candidates={};offset=0
    for episode in train:
        length=int(data.meta.episodes['length'][episode]);frame=data.get_raw_item(offset)
        description=str(data.meta.tasks.index[int(frame['task_index'])])
        if canonical(description) in known:
            candidates.setdefault(int(frame['task_index']),[]).append((offset,length,episode,description,known[canonical(description)]))
        offset+=length
    if len(candidates)!=10:raise ValueError('Did not match exactly ten LIBERO-10 task descriptions')
    started=time.perf_counter()
    rng=random.Random(args.seed);rows=[];images=[];wrists=[];states=[]
    for task,episodes in sorted(candidates.items()):
        for entry,rank,phase in stratified_frames(episodes,args.samples_per_task,rng):
            start,length,episode,description,benchmark_id=entry
            frame=data[start+rank]
            state=np.asarray(frame['observation.state'],dtype=np.float32)
            if state.shape!=(8,):raise ValueError('Teacher needs 8D proprio; do not fabricate gripper state')
            def image(key):
                a=frame[key].numpy()
                if a.shape[0]!=3 or a.dtype!=np.uint8:raise ValueError('Unexpected video frame layout')
                return np.transpose(a,(1,2,0)).copy()
            images.append(image('observation.images.image'));wrists.append(image('observation.images.image2'));states.append(state)
            rows.append({'task_index':task,'benchmark_task_id':benchmark_id,'suite':'libero_10',
                         'episode_index':episode,'frame_index':rank,'temporal_third':phase,'task':description,'valid_length':min(8,length-rank)})
            if len(rows)%100==0:
                print(json.dumps({'stage':'prepare','observations':len(rows),'total':10*args.samples_per_task,'task_index':task,'elapsed_seconds':time.perf_counter()-started}),flush=True)
    args.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(args.output/'observations.npz',image=np.stack(images),wrist=np.stack(wrists),state=np.stack(states))
    manifest={'source_kind':'isolated_training_observations','rows':rows,'fps':10,'seed':args.seed,
              'partition_sha256':file_sha256(args.partition),'split_sha256':file_sha256(args.splits),
              'sampling':'unique_episode_balanced_temporal_thirds_with_8valid_actions',
              'samples_per_task':args.samples_per_task,'dataset_revision':split['dataset_revision'],'observations_sha256':file_sha256(args.output/'observations.npz'),
              'image_convention':'LIBERO_RLDS_training_orientation; no_extra_rotation_then_official_resize_and_crop',
              'dataset_task_index_to_benchmark_id':{str(t):v[0][4] for t,v in candidates.items()}}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'observations':len(rows),'tasks':sorted(candidates),'output':str(args.output)}),flush=True)


if __name__=='__main__':main()
