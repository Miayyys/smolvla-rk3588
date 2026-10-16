"""Watch an existing fine-tuning job, then run matched real-to-sim diagnostics."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch-pid', type=int, required=True)
    parser.add_argument('--config', type=Path, default=Path('config/so100_real_finetune_v1.json'))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    out = Path(cfg['output'])

    def status(stage, **extra):
        report = {'stage': stage, 'time_unix': time.time(), **extra}
        (out / 'pipeline_status.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)

    status('waiting_for_training', training_pid=args.watch_pid)
    while True:
        try:
            command = Path(f'/proc/{args.watch_pid}/cmdline').read_bytes()
        except FileNotFoundError:
            break
        if b'qvla.evaluation.finetune_so100_real' not in command:
            break
        time.sleep(5)
    events = [json.loads(line) for line in (out / 'events.jsonl').read_text().splitlines()]
    if not any(e.get('stage') == 'completed' for e in events):
        status('training_failed', log=str(out / 'train.log'))
        return
    candidate = out / 'best'
    if not (candidate / 'model.safetensors').is_file():
        status('no_validation_improvement', note='Original baseline was not beaten; last saved model is diagnostic only.')
        candidate = out / 'last'
    reports = {}
    matched = cfg.get('evaluation') == 'matched_sim_dataset'
    models = [('real_finetuned_fp' if matched else 'original_fp', Path(cfg['model_dir'])),
              ('sim_adapted_fp' if matched else 'finetuned_fp', candidate)]
    for label, model in models:
        folder = out / 'simulation' / label
        status('simulation_running', label=label, model=str(model))
        with (out / f'simulation_{label}.log').open('wb') as log:
            command = [
                sys.executable, '-u', '-m', 'qvla.evaluation.eval_so101_base',
                '--model-dir', str(model), '--normalization-file', str(out / 'stats.json'),
                '--output', str(folder), '--max-steps', '1024', '--seed', '0',
            ]
            if matched:
                command = [sys.executable, '-u', '-m', 'qvla.evaluation.eval_so101_sim_matched',
                           '--config', str(args.config), '--model-dir', str(model), '--output', str(folder)]
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        if result.returncode:
            status('simulation_failed', label=label, returncode=result.returncode)
            return
        report = json.loads((folder / 'report.json').read_text())
        reports[label] = [{'task': e['env_id'], 'success': e['success'], 'steps': e['steps'],
                           'ever_grasped': e['ever_grasped'], 'clipped_value_fraction': e.get('clipped_value_fraction')}
                          for e in report['episodes']]
    status('completed', simulation=reports,
           scope=('two matched-scene episodes per model, not physical success' if matched else
                  'three paired seed-0 real-to-sim diagnostic episodes per model, not physical success'),
           quantization_started=False)


if __name__ == '__main__':
    main()
