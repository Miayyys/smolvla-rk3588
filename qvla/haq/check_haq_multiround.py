#!/usr/bin/env python3
"""Multi-round controller regression; synthetic rewards, no model search."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from qvla.haq.policy import RecurrentPolicyGradient
from qvla.haq.search_space import build_search_space, validate_assignment
from qvla.haq.offline_actions import file_sha256, score_actions


def run_controller(space, seed, output, rounds=40, batch_size=8):
    policy = RecurrentPolicyGradient(space, hidden_size=8, seed=seed)
    # Deliberately known synthetic optimum, unrelated to real quality/hardware.
    targets = {s['module']: s['options'][0]['format'] for s in space['action_sites']}
    def reward(a):
        return sum(a[k] == v for k, v in targets.items()) / len(targets)
    def probe():
        state = copy.deepcopy(policy.rng.bit_generator.state)
        values = [reward(policy.sample()[0]) for _ in range(64)]
        policy.rng.bit_generator.state = state
        return float(np.mean(values))
    before = probe()
    original = {k: v.copy() for k, v in policy.params.items()}
    rows, seen = [], {s['module']: set() for s in space['action_sites']}
    resume_verified = False
    for iteration in range(rounds):
        samples = [policy.sample() for _ in range(batch_size)]
        values = []
        for assignment, _ in samples:
            validate_assignment(space, assignment)
            values.append(reward(assignment))
            for key, fmt in assignment.items():
                seen[key].add(fmt)
        statistics = policy.update_batch([t for _, t in samples], values,
                                         learning_rate=0.005)
        assert all(np.isfinite(p).all() for p in policy.params.values())
        assert policy.optimizer_step == iteration + 1
        rows.append({'round': iteration + 1, 'rewards': values,
                     'assignments': [a for a, _ in samples], **statistics})
        if iteration + 1 == rounds // 2:
            path = output / f'controller_seed{seed}_midpoint.npz'
            policy.save(path)
            restored = RecurrentPolicyGradient(space, hidden_size=8, seed=999)
            restored.load(path)
            state = copy.deepcopy(policy.rng.bit_generator.state)
            a = [policy.sample() for _ in range(2)]
            b = [restored.sample() for _ in range(2)]
            assert [x[0] for x in a] == [x[0] for x in b]
            ra = [reward(x[0]) for x in a]
            policy.update_batch([x[1] for x in a], ra, learning_rate=0.005)
            restored.update_batch([x[1] for x in b], ra, learning_rate=0.005)
            for key in policy.params:
                np.testing.assert_array_equal(policy.params[key], restored.params[key])
                np.testing.assert_array_equal(policy.m[key], restored.m[key])
                np.testing.assert_array_equal(policy.v[key], restored.v[key])
            # Resume the actual run from the saved midpoint, without probe updates.
            policy.load(path)
            assert policy.rng.bit_generator.state == state
            resume_verified = True
    after = probe()
    assert after > before, (seed, before, after)
    policy.save(output / f'controller_seed{seed}_final.npz')
    delta = float(np.sqrt(sum(np.sum((policy.params[k] - original[k]) ** 2)
                              for k in original)))
    return {'seed': seed, 'rounds': rounds, 'batch_size': batch_size,
            'candidate_evaluations': rounds * batch_size,
            'before_synthetic_reward': before, 'after_synthetic_reward': after,
            'parameter_change_l2': delta, 'resume_sample_and_update_exact': resume_verified,
            'all_options_sampled_sites': sum(len(seen[s['module']]) == len(s['options'])
                                             for s in space['action_sites']),
            'rows': rows}


def proxy_check():
    actions = ROOT / 'runs/expert_real_w8a8_action_v1/actions.npz'
    source_report = ROOT / 'runs/expert_real_w8a8_action_v1/report.json'
    report = json.loads(source_report.read_text())
    assert file_sha256(actions) == report['actions_sha256']
    paired = ROOT / 'runs/paired_noise_seed0_v2/paired_report.json'
    closed = json.loads(paired.read_text())
    with np.load(actions, allow_pickle=False) as a:
        scores = {mode: score_actions(a[array], a['original'], a['recorded_first'],
                                     a['task_index'])['task_macro']
                  for mode, array in [('fp', 'original'), ('ptq', 'ptq'), ('qat', 'qat')]}
    return {'scope': 'historical_model_level_comparison_not_paired_observation_rollout',
            'sources': {str(p.relative_to(ROOT)): file_sha256(p)
                        for p in [actions, source_report, paired]},
            'models': {m: {'offline': scores[m], 'closed_loop_successes_out_of_40': n}
                       for m, n in closed['totals'].items()},
            'counterexample_recorded_mae': all(
                scores[m]['first_action_mae_vs_recorded'] < scores['fp']['first_action_mae_vs_recorded']
                and closed['totals'][m] < closed['totals']['fp'] for m in ['ptq', 'qat']),
            'conclusion': 'Useful action-deviation diagnostic; not validated as success-ranking reward.',
            'limits': ['Only two quantized candidates and one rollout per task.',
                       'Offline demonstration observations differ from on-policy rollout states/noise.',
                       'Historical models compress only 11.04%; not feasible 40%-compression candidates.',
                       'No fresh closed-loop rollout and no formal proxy correlation claim.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/haq_multiround_local_v1')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    space = build_search_space(ROOT / 'config/hardware/tables.json')
    proxy = proxy_check()
    (args.output / 'offline_usefulness.json').write_text(json.dumps(proxy, indent=2) + '\n')
    summaries = []
    for seed in [7, 29, 83]:
        result = run_controller(space, seed, args.output)
        (args.output / f'seed{seed}.json').write_text(json.dumps(result, indent=2) + '\n')
        summaries.append({k: v for k, v in result.items() if k != 'rows'})
        print(json.dumps(summaries[-1]), flush=True)
    summary = {'status': 'passed', 'scope': 'synthetic_controller_regression_only',
               'real_model_RL_evaluated': False, 'formal_search_ready': space['search_ready'],
               'stopping_rule': 'Fixed 40 optimizer updates per seed; stop immediately on failed assertions.',
               'reward': 'Fraction of sites selecting their first listed option; synthetic oracle only.',
               'space_counts': space['counts'], 'space_source': space['source'],
               'runs': summaries, 'elapsed_seconds': time.perf_counter() - start,
               'script_sha256': file_sha256(Path(__file__))}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
