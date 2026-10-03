#!/usr/bin/env python3
"""Reject malformed/bounded-action violations and explicit reviewed exclusions.

This is a contract filter, not an automatic classifier of task-quality errors.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qvla_haq.offline_actions import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--exclusions',type=Path,help='JSON with teacher_actions_sha256 and exclusions [{row,timesteps,reason}]')
    args=p.parse_args();meta=json.loads((args.cache/'manifest.json').read_text())
    path=args.cache/'actions.npy'
    if meta['source_kind']!='openvla_oft_real_inference' or file_sha256(path)!=meta['actions_sha256']:raise ValueError('Real teacher label identity required')
    a=np.load(path,allow_pickle=False)
    if a.shape!=(len(meta['rows']),8,7):raise ValueError('Unexpected teacher shape')
    mask=np.isfinite(a).all(2)&(np.abs(a[:,:,:6])<=1+1e-6).all(2)&np.isin(a[:,:,6],[-1,0,1])
    for i,row in enumerate(meta['rows']):mask[i,row['valid_length']:]=False
    exclusions=[]
    if args.exclusions:
        review=json.loads(args.exclusions.read_text())
        if review['teacher_actions_sha256']!=meta['actions_sha256']:raise ValueError('Exclusions belong to another cache')
        exclusions=review['exclusions']
        for row in exclusions:
            i=row['row'];timesteps=row['timesteps']
            if not row['reason'] or not 0<=i<len(a) or not timesteps or any(not 0<=t<8 for t in timesteps):raise ValueError('Invalid explicit rejection')
            mask[i,timesteps]=False
    result={'teacher_actions_sha256':meta['actions_sha256'],'rows':meta['rows'],
            'accepted_timestep_mask':mask.tolist(),'accepted_observations':int(mask.any(1).sum()),
            'accepted_action_steps':int(mask.sum()),'rejected_action_steps':int(mask.size-mask.sum()),
            'explicit_exclusions':exclusions,'scope':'finite_simulator_bounds_and_explicit_review_not_semantic_correctness',
            'task_success_or_failure_was_not_used_to_reject_entire_tasks':True}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','accepted_timestep_mask')}))

if __name__=='__main__':main()
