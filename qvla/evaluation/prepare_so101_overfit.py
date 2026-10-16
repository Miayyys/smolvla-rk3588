"""Derive a single/few-source overfit run from an existing matched SO101 cache.

Only train_rows and the offline check rows change. The JPEG cache, actions and
normalization statistics are reused unchanged so that the data subset and the
training length are the only variables. Offline rows come from training
sources: this is a capacity diagnostic, not a held-out metric.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from qvla.evaluation.finetune_so100_real import emit, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    source, out = Path(cfg['source_run']), Path(cfg['output'])
    if (out / 'progress.json').exists():
        raise ValueError('Run already trained; preserve it')
    out.mkdir(parents=True, exist_ok=True)
    prepared = json.loads((source / 'prepared.json').read_text())
    if prepared['partition_sha256'] != sha(cfg['partition']):
        raise ValueError('Partition differs from the source run')
    if prepared['stats_sha256'] != sha(source / 'stats.json'):
        raise ValueError('Source statistics changed')
    if not set(cfg['overfit_sources']) <= set(prepared['training_successful_sources']):
        raise ValueError('Overfit sources must be successful training sources')
    data = dict(np.load(source / 'data.npz'))
    eps = data['episodes']
    data['train_rows'] = np.flatnonzero(np.isin(eps, cfg['overfit_sources']))
    check = []
    for ep in cfg['overfit_sources']:
        rows = np.flatnonzero(eps == ep)
        check.extend(np.linspace(rows[0], max(rows[0], rows[-1] - 49), cfg['validation_frames_per_episode'], dtype=int).tolist())
    data['validation_rows'] = np.asarray(check)
    np.savez(out / 'data.npz', **data)
    (out / 'stats.json').write_bytes((source / 'stats.json').read_bytes())
    cache = out / 'top_frames.jpeg.bin'
    if not cache.exists():
        cache.symlink_to((source / 'top_frames.jpeg.bin').resolve())
    if sha(cache) != prepared['cache_sha256']:
        raise ValueError('Cache hash differs from source run')
    manifest = {**prepared, 'config': cfg, 'derived_from': str(source),
                'derived_from_prepared_sha256': sha(source / 'prepared.json'),
                'training_successful_sources': cfg['overfit_sources'],
                'train_frames': int(len(data['train_rows'])), 'offline_check_rows': check,
                'stats_sha256': sha(out / 'stats.json'), 'data_sha256': sha(out / 'data.npz'),
                'scope': 'overfit capacity diagnostic; offline check rows are training frames; stats reuse the 5-source train statistics'}
    (out / 'prepared.json').write_text(json.dumps(manifest, indent=2) + '\n')
    emit(out, {'stage': 'prepared_overfit', 'sources': cfg['overfit_sources'],
               'train_frames': manifest['train_frames'], 'offline_check_rows': check})


if __name__ == '__main__':
    main()
