#!/usr/bin/env python3
"""Compare equal-budget real-model RL and no-update random search."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_run(path):
    run=json.loads((path/'summary.json').read_text())
    rows=run['results']
    if len(rows)!=run['rounds']*run['batch_size']:
        raise ValueError(f'Incomplete run: {path}')
    return run,rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rl',type=Path,required=True)
    p.add_argument('--random',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    rl,rl_rows=load_run(args.rl)
    rnd,rnd_rows=load_run(args.random)
    for field in ('seed','rounds','batch_size','int8_logit_prior','entropy_weight',
                  'score_task_indices','cache_manifest_sha256','tables_sha256'):
        if rl[field]!=rnd[field]:raise ValueError(f'Unequal comparison: {field}')
    for field in ('quality_mode','entropy_final','validation_task_indices'):
        if rl.get(field)!=rnd.get(field):raise ValueError(f'Unequal comparison: {field}')
    if rl['random_control'] or not rnd['random_control']:
        raise ValueError('Expected RL updates versus no-update random control')
    if len(rl_rows)!=len(rnd_rows):raise ValueError('Unequal candidate budget')
    args.output.mkdir(parents=True,exist_ok=True)
    curves={}
    summary={'candidate_budget_per_arm':len(rl_rows),'rounds':rl['rounds'],
             'batch_size':rl['batch_size'],'score_task_indices':rl['score_task_indices'],
             'scope':'local offline action proxy; one controller seed; not task success or RK3588 whole-policy speed',
             'arms':{}}
    with (args.output/'candidates.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['arm','evaluation','round','candidate','reward',
                              'best_so_far_reward','compression_fraction','chunk_mae_vs_fp',
                              'signature_sum_cost_ms','feasible_40pct'])
        writer.writeheader()
        for label,run,rows in (('RL',rl,rl_rows),('Random',rnd,rnd_rows)):
            run_path=args.rl if label=='RL' else args.random
            best=float('-inf');curve=[]
            for i,row in enumerate(rows,1):
                report=json.loads((run_path/f"round{row['round']:03d}_candidate{row['candidate']:02d}"/'report.json').read_text())
                if row['feasible_40pct']:best=max(best,row['reward'])
                curve.append(best)
                writer.writerow({'arm':label,'evaluation':i,'round':row['round'],
                                 'candidate':row['candidate'],'reward':row['reward'],
                                 'best_so_far_reward':best,'compression_fraction':row['compression_fraction'],
                                 'chunk_mae_vs_fp':report['metrics']['task_macro']['chunk_mae_vs_fp'],
                                 'signature_sum_cost_ms':report['cost']['total_ms'],
                                 'feasible_40pct':row['feasible_40pct']})
            curves[label]=curve
            best_row=max((r for r in rows if r['feasible_40pct']),key=lambda r:r['reward'])
            best_report=json.loads((run_path/f"round{best_row['round']:03d}_candidate{best_row['candidate']:02d}"/'report.json').read_text())
            holdout=run['heldout_best']
            summary['arms'][label]={
                'best_reward':best_row['reward'],'best_evaluation':rows.index(best_row)+1,
                'best_compression_fraction':best_row['compression_fraction'],
                'best_model_bytes':best_row['actual_model_bytes'],
                'best_proxy_cost_ms':best_report['cost']['total_ms'],
                'best_scoring_chunk_mae_vs_fp':best_report['metrics']['task_macro']['chunk_mae_vs_fp'],
                'mean_reward_first_20':sum(r['reward'] for r in rows[:20])/20,
                'mean_reward_last_20':sum(r['reward'] for r in rows[-20:])/20,
                'feasible_count':sum(r['feasible_40pct'] for r in rows),
                'controller_parameter_change_l2':run['parameter_change_l2'],
                'elapsed_seconds':run['elapsed_seconds'],
                'heldout_action_count':holdout['observations'] if holdout else None,
                'heldout_chunk_mae_vs_fp':holdout['metrics']['task_macro']['chunk_mae_vs_fp'] if holdout else None,
                'heldout_checkpoint_sha256':holdout['checkpoint_sha256'] if holdout else None,
                'best_checkpoint_path':str((args.rl if label=='RL' else args.random)/
                    f"round{best_row['round']:03d}_candidate{best_row['candidate']:02d}"/'model.safetensors')}
    fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True)
    for label,run,rows in (('RL',rl,rl_rows),('Random',rnd,rnd_rows)):
        x=list(range(1,len(rows)+1))
        axes[0].plot(x,curves[label],label=label)
        window=10
        means=[sum(r['reward'] for r in rows[max(0,i-window+1):i+1])/
               len(rows[max(0,i-window+1):i+1]) for i in range(len(rows))]
        axes[1].plot(x,means,label=label)
    axes[0].set_ylabel('Best feasible proxy reward')
    axes[1].set_ylabel('Reward moving mean (10 candidates)')
    axes[1].set_xlabel('Real quantized candidates evaluated per arm')
    for ax in axes:
        ax.grid(alpha=.3);ax.legend()
    fig.suptitle('SmolVLA RL vs random, same 40% file-compression rule; offline proxy only')
    fig.tight_layout()
    fig.savefig(args.output/'comparison.png',dpi=160)
    fig.savefig(args.output/'comparison.svg')
    plt.close(fig)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
