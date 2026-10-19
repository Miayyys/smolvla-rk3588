"""Why do grasped cubes get dropped? Classify every grasp-loss event in closed-loop traces (``--trace`` output).

A grasp-loss event is a step where ``is_grasped`` goes 1 -> 0 after at least 3 grasped steps. At that step the
gripper command and measured gripper (dataset units 0-100; scripted expert: open 55.97, closed 0.24, release
37.74) tell whether the policy opened the jaw or the cube slipped from a closed jaw; the object height and
distance to the target tell whether it was a placement release. Reference: gripper state while the scripted
expert carries the cube (v2 training frames in phases lift/carry).

Usage: so101_grasp_drop_analysis.py <eval_dir> [<eval_dir> ...] [--data runs/so101_scripted_data_v2]
"""

import argparse
import json
from pathlib import Path

import numpy as np

PHASES = ['pregrasp', 'descend', 'close', 'lift', 'carry', 'lower', 'open', 'retreat', 'hold']


def events(values, columns):
    c = {name: i for i, name in enumerate(columns)}
    grasped = values[:, c['is_grasped']] >= 0.5
    out = []; run = 0; start = None
    for t, g in enumerate(grasped):
        if g:
            if run == 0:
                start = t
            run += 1
            continue
        if run >= 3:
            row = values[t]
            command = float(row[c['gripper_command']]); state = float(row[c['gripper_state']])
            to_target = float(row[c['object_to_target_m']])
            kind = ('placement_release' if to_target < 0.03 else
                    'policy_opened' if command > 15 else 'slip_with_closed_command')
            event = {'step': t, 'grasp_start': start, 'grasped_steps': run, 'kind': kind,
                     'gripper_command': command, 'gripper_state': state,
                     'command_mean_while_grasped': float(values[start:t, c['gripper_command']].mean()),
                     'state_mean_while_grasped': float(values[start:t, c['gripper_state']].mean()),
                     'object_z_m': float(row[c['object_z_m']]), 'max_object_z_m': float(values[start:t, c['object_z_m']].max()),
                     'object_to_target_m': to_target}
            if 'gate_phase' in c:
                event['gate_phase'] = PHASES[int(row[c['gate_phase']])]
            out.append(event)
        run = 0
    return out, int(grasped.sum())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('eval_dirs', type=Path, nargs='+')
    p.add_argument('--data', type=Path, default=Path('runs/so101_scripted_data_v2'))
    args = p.parse_args()
    report = {}
    train = np.load(args.data / 'train.npz')
    carrying = np.isin(train['phase'], [PHASES.index('lift'), PHASES.index('carry')])
    report['expert_reference'] = {'gripper_state_lift_carry_percentiles_5_50_95': np.percentile(train['states'][carrying, 5], [5, 50, 95]).round(2).tolist(),
                                  'gripper_command_lift_carry_percentiles_5_50_95': np.percentile(train['actions'][carrying, 5], [5, 50, 95]).round(2).tolist()}
    for directory in args.eval_dirs:
        episodes = {e['seed']: e for e in json.loads((directory / 'report.json').read_text())['episodes']}
        rows = []
        for path in sorted(directory.glob('trace_seed*.npz')):
            seed = int(path.stem.replace('trace_seed', ''))
            data = np.load(path); found, grasped_steps = events(data['values'], [str(x) for x in data['columns']])
            rows.append({'seed': seed, 'success': episodes[seed]['success'], 'grasped_steps': grasped_steps, 'events': found})
        all_events = [e for r in rows for e in r['events']]
        kinds = {}
        for e in all_events:
            kinds[e['kind']] = kinds.get(e['kind'], 0) + 1
        summary = {'episodes': len(rows), 'episodes_grasped': sum(r['grasped_steps'] > 0 for r in rows),
                   'success': sum(r['success'] for r in rows), 'grasp_loss_events': len(all_events), 'by_kind': kinds}
        for kind in kinds:
            sel = [e for e in all_events if e['kind'] == kind]
            summary[kind] = {k: float(np.median([e[k] for e in sel])) for k in
                             ('grasped_steps', 'gripper_command', 'gripper_state', 'command_mean_while_grasped',
                              'state_mean_while_grasped', 'object_z_m', 'max_object_z_m', 'object_to_target_m')}
            if 'gate_phase' in sel[0]:
                phases = {}
                for e in sel:
                    phases[e['gate_phase']] = phases.get(e['gate_phase'], 0) + 1
                summary[kind]['gate_phase_counts'] = phases
        report[str(directory)] = {'summary': summary, 'episodes': rows}
        print(json.dumps({str(directory): summary}, indent=1))
    print(json.dumps({'expert_reference': report['expert_reference']}))
    out = args.eval_dirs[0] / 'grasp_drop_analysis.json'
    out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
