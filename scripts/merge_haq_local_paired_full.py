#!/usr/bin/env python3
"""Join disjoint local LIBERO panels into a paired 40-task development report."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qvla_haq.offline_actions import file_sha256

SUITES=('libero_spatial','libero_object','libero_goal','libero_10')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first',type=Path,required=True)
    p.add_argument('--second',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    summaries=[]
    for folder in [args.first,args.second]:
        path=folder/'summary.json'
        raw=json.loads(path.read_text())
        if raw['seed']!=0 or raw['scope']!='single_seed_local_libero_development_panel_not_final_quality':
            raise ValueError('Only identical seed-0 paired development panels can be joined')
        summaries.append((raw,file_sha256(path)))
    run=json.loads((args.run/'summary.json').read_text())
    selected=run['best_by_proxy']
    if selected is None or not selected['feasible_40pct']:
        raise ValueError('No feasible selected checkpoint')
    source_run=str(args.run.resolve())
    for raw,_ in summaries:
        if Path(raw['source_run']).resolve()!=Path(source_run):
            raise ValueError('Panels refer to different quantized runs')
    rows=summaries[0][0]['pairs']+summaries[1][0]['pairs']
    unique={(r['suite'],r['task_id']) for r in rows}
    if len(rows)!=40 or unique!={(suite,task) for suite in SUITES for task in range(10)}:
        raise ValueError('Panels do not cover each of the 40 tasks exactly once')
    rows.sort(key=lambda r:(SUITES.index(r['suite']),r['task_id']))
    by_suite=defaultdict(list)
    for r in rows:
        if type(r['fp']) is not bool or type(r['best']) is not bool:
            raise ValueError('Expected Boolean episode outcomes')
        by_suite[r['suite']].append(r)
    output={
        'scope':'local_libero_40task_single_seed_paired_development_not_final_quality',
        'seed':0,'one_episode_per_task':True,
        'partition_note':'LIBERO benchmark task rollouts; data calibration/frozen episodes not used',
        'model_checkpoint_sha256':run['identity']['checkpoint_sha256'],
        'quantized_checkpoint_sha256':selected['checkpoint_sha256'],
        'quantized_file_bytes':selected['actual_model_bytes'],
        'compression_fraction':selected['compression_fraction'],
        'source_panels':[{'path':str(folder/'summary.json'),'sha256':digest}
                         for folder,(_,digest) in zip([args.first,args.second],summaries)],
        'per_suite':{suite:{'fp':sum(r['fp'] for r in rs),'candidate':sum(r['best'] for r in rs)}
                     for suite,rs in by_suite.items()},
        'fp_successes':sum(r['fp'] for r in rows),
        'candidate_successes':sum(r['best'] for r in rows),
        'improved_tasks':sum(r['best'] and not r['fp'] for r in rows),
        'worsened_tasks':sum(r['fp'] and not r['best'] for r in rows),
        'per_task':rows,
        'conclusion_limit':'single matched initial state/seed per task; different local Torch/NumPy from prior A10 run; no RK3588 full-policy execution'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='per_task'},ensure_ascii=False))


if __name__=='__main__':main()
