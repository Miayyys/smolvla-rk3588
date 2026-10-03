#!/usr/bin/env python3
"""Audit actual candidate files, action feedback, proposals, and optimizer steps."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qvla_haq.offline_actions import file_sha256, load_cache, score_actions
from qvla_haq.policy import RecurrentPolicyGradient
from qvla_haq.search_space import validate_assignment
from qvla_haq.feasibility_reward import score_feasibility
from qvla_haq.action_quality import action_quality


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--cache',type=Path,default=ROOT/'runs/haq_offline_local_v1/fp_cache40')
    args=p.parse_args()
    summary=json.loads((args.run/'summary.json').read_text())
    space=json.loads((args.run/'space.json').read_text())
    _,obs,fp=load_cache(args.cache,summary['identity'])
    manifest=json.loads((args.cache/'manifest.json').read_text())
    if summary.get('score_task_indices') is not None:
        positions=[i for i,s in enumerate(manifest['samples'])
                   if s['task_index'] in set(summary['score_task_indices'])]
        obs={k:v[positions] for k,v in obs.items()}
        fp=fp[positions]
        samples_manifest=[manifest['samples'][i] for i in positions]
    else:
        samples_manifest=manifest['samples']
    assert file_sha256(args.cache/'manifest.json')==summary['cache_manifest_sha256']
    table_path=args.run/'tables_snapshot.json'
    if not table_path.exists():table_path=ROOT/'docs/hardware/tables.json'
    tables=json.loads(table_path.read_text())
    assert file_sha256(table_path)==summary['tables_sha256']
    costs={r['case_id']:r for r in tables['measured_costs']+tables['supplemental_costs']}
    policy=RecurrentPolicyGradient(space,hidden_size=16,seed=summary['seed'])
    policy.load(args.run/'controller_initial.npz')
    for update in summary['updates']:
        samples=[policy.sample() for _ in range(summary['batch_size'])]
        rewards=[]
        for i,(assignment,trajectory) in enumerate(samples):
            d=args.run/f"round{update['round']:03d}_candidate{i+1:02d}"
            saved=json.loads((d/'assignment.json').read_text())
            validate_assignment(space,saved)
            assert assignment==saved, 'Candidate did not come from current RL policy'
            r=json.loads((d/'report.json').read_text())
            checkpoint=d/'model.safetensors'
            if checkpoint.exists():
                assert file_sha256(checkpoint)==r['checkpoint_sha256']
                assert checkpoint.stat().st_size==r['actual_model_bytes']
            else:
                assert summary.get('checkpoint_retention')=='best' and r['checkpoint_retained'] is False
            with np.load(d/'actions.npz',allow_pickle=False) as a:
                scores=score_actions(a['actions'],fp,obs['recorded_action'],
                                     np.array([s['task_index'] for s in samples_manifest]))
            assert scores==r['metrics'], 'Quality feedback does not match saved actions'
            total=sum(costs[row['case_id']]['p50_ms']*row['calls_per_observation']
                      for row in r['cost']['rows'])
            assert math.isclose(total,r['cost']['total_ms'],rel_tol=1e-12)
            scale=summary['reward_definition']['mean_abs_FP_action']
            assert math.isclose(scale,float(np.abs(fp.astype(np.float64)).mean()),rel_tol=1e-12)
            A=math.exp(-scores['task_macro']['chunk_mae_vs_fp']/scale)
            if 'quality_proxy' in r:
                with np.load(d/'actions.npz',allow_pickle=False) as saved_actions:
                    qp=action_quality(saved_actions['actions'],fp,scores,summary.get('quality_mode','chunk_mae'))
                assert qp==r['quality_proxy']
                A=qp['quality']
            G=math.sqrt(A*summary['reward_definition']['Tref_ms']/total)
            size=r['actual_model_bytes'];original=summary['identity']['source_model_file_bytes']
            feasible=5*size<=3*original
            if 'backend_feasibility' in r:
                result=score_feasibility(model_bytes=size,reference_bytes=original,
                                         gain=G,backend_status='not_attempted',allow_proxy=True)
                assert result==r['backend_feasibility']
                expected=result['reward']
            else:
                expected=G/(1+G) if feasible else -(0.4-(1-size/original))/0.4-1e-6
            assert feasible==r['feasible_40pct']
            assert math.isclose(expected,r['reward'],rel_tol=1e-12)
            rewards.append(expected)
        if not summary.get('random_control'):
            policy.update_batch([t for _,t in samples],rewards,
                                entropy_weight=update.get('entropy_weight_used',summary['entropy_weight']))
        restored=RecurrentPolicyGradient(space,hidden_size=16)
        restored.load(args.run/f"controller_round{update['round']:03d}.npz")
        assert restored.optimizer_step==(0 if summary.get('random_control') else update['round'])
        for k in policy.params:
            np.testing.assert_array_equal(policy.params[k],restored.params[k])
            np.testing.assert_array_equal(policy.m[k],restored.m[k])
            np.testing.assert_array_equal(policy.v[k],restored.v[k])
    report={'status':'passed','scope':'saved_real_action_and_table_feedback_to_exact_RL_updates',
            'candidates_verified':summary['real_candidate_evaluations'],
            'updates_verified':summary['rounds'],
            'does_not_prove':'closed-loop quality or RKNN full-policy deployment'}
    (args.run/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
