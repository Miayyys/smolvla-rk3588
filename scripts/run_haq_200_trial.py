#!/usr/bin/env python3
"""Run the authorized 200-round local diagnostic, audit, and paired panel."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/haq_rl_trial_200x4_rl'
STATUS = ROOT/'runs/haq_rl_trial_200x4_status.json'


def main():
    started = time.time()
    env = dict(os.environ, MPLCONFIGDIR='/tmp/qvla-mpl',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    steps = [
        ('search', [sys.executable, 'scripts/run_haq_local_loop.py', '--output', str(RUN),
                    '--rounds', '200', '--batch-size', '4', '--seed', '29',
                    '--int8-logit-prior', '5', '--entropy-weight', '0',
                    '--score-task-indices', '0', '5', '10', '15', '20', '25', '30', '35',
                    '--checkpoint-retention', 'best']),
        ('audit', [sys.executable, 'scripts/audit_haq_local_loop.py', str(RUN)]),
        ('paired_panel', [sys.executable, 'scripts/run_haq_local_paired_panel.py',
                          '--run', str(RUN), '--output', str(RUN)+'_panel',
                          '--fp-reference-output', 'runs/haq_libero_panel_v1']),
        ('summarize', [sys.executable, 'scripts/summarize_haq_200_trial.py']),
    ]
    state = {'status': 'running', 'pid': os.getpid(), 'started_unix': started,
             'scope': 'local_proxy_search; RKNN_conversion_not_attempted', 'steps': []}
    for name, command in steps:
        state['stage'] = name
        STATUS.write_text(json.dumps(state, indent=2)+'\n')
        tick = time.time()
        print('Starting ' + name, flush=True)
        result = subprocess.run(command, cwd=ROOT, env=env, check=False)
        state['steps'].append({'stage': name, 'command': command,
                               'returncode': result.returncode, 'seconds': time.time()-tick})
        if result.returncode:
            state.update(status='failed', elapsed_seconds=time.time()-started)
            STATUS.write_text(json.dumps(state, indent=2)+'\n')
            raise SystemExit(result.returncode)
    state.update(status='complete', elapsed_seconds=time.time()-started)
    STATUS.write_text(json.dumps(state, indent=2)+'\n')
    print('200-round diagnostic and follow-up checks complete.', flush=True)


if __name__ == '__main__':
    main()
