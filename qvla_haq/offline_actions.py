"""Portable development observation/FP-action cache and diagnostic metrics.

No model dependencies. These metrics are offline proxies, not task success.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_partition(partition: dict, splits: dict) -> list[int]:
    dev = partition['development_episode_ids_from_qat_train']
    cal = partition['calibration_episode_ids_from_ptq_calibration']
    frozen = partition['remaining_frozen_test_episode_ids']
    explored = partition['exploratory_test_episode_ids_to_exclude']
    sets = [set(items) for items in (dev, cal, frozen, explored)]
    if any(len(items) != len(set(items)) for items in (dev, cal, frozen, explored)):
        raise ValueError('Duplicate episode ID in partition')
    if any(sets[i] & sets[j] for i in range(4) for j in range(i + 1, 4)):
        raise ValueError('Development/calibration/frozen/exploratory overlap')
    source = splits['splits']
    if not sets[0] <= set(source['qat_train']) or not sets[1] <= set(source['ptq_calibration']):
        raise ValueError('Development/calibration does not belong to its source split')
    if not (sets[2] | sets[3]) <= set(source['test']):
        raise ValueError('Test partition does not belong to source test split')
    return sorted(dev)


def frame_positions(length: int, task: int, phases: int) -> list[int]:
    """Quick panel spreads phases across tasks; extended panel takes all three."""
    if length < 3 or phases not in (1, 3):
        raise ValueError('Need an episode of at least three frames and 1 or 3 phases')
    positions = [0, (length - 1) // 2, int((length - 1) * 0.9)]
    if len(set(positions)) != 3:
        positions = [0, (length - 1) // 2, length - 1]
    return [positions[task % 3]] if phases == 1 else positions


def action_seed(episode: int, task: int, frame: int) -> int:
    # Actual frame index, not panel rank: a sample keeps its noise in both panels.
    key = f'qvla-offline-v1:{episode}:{task}:{frame}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], 'little')


def validate_actions(actions: np.ndarray, count: int) -> None:
    if count < 1 or actions.ndim != 3 or actions.shape[0] != count or actions.shape[1] < 1 or actions.shape[2] != 7:
        raise ValueError(f'Expected [N, chunk_length, 7], got {actions.shape}')
    if not np.isfinite(actions).all():
        raise ValueError('Non-finite action value')


def save_cache(directory: Path, manifest: dict, observations: dict,
               fp_actions: np.ndarray) -> None:
    validate_actions(fp_actions, len(manifest['samples']))
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError(f'Refusing to replace an existing cache: {directory}')
    for filename, arrays in [('observations.npz', observations),
                             ('fp_actions.npz', {'actions': fp_actions})]:
        with (directory / filename).open('wb') as stream:
            np.savez_compressed(stream, **arrays)
    record = dict(manifest)
    record.update(schema_version=1, scope='offline_development_action_proxy',
                  closed_loop_success_rate='not_measured',
                  files={name: file_sha256(directory / name)
                         for name in ('observations.npz', 'fp_actions.npz')})
    (directory / 'manifest.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')


def load_cache(directory: Path, expected_identity: dict | None = None) -> tuple[dict, dict, np.ndarray]:
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('schema_version') != 1 or manifest.get('scope') != 'offline_development_action_proxy':
        raise ValueError('Unknown cache schema or evaluation scope')
    if expected_identity is not None and manifest['identity'] != expected_identity:
        raise ValueError('Model, processor, VLM assets, split or partition identity mismatch')
    if set(manifest['files']) != {'observations.npz', 'fp_actions.npz'}:
        raise ValueError('Cache file manifest mismatch')
    for name, digest in manifest['files'].items():
        if file_sha256(directory / name) != digest:
            raise ValueError(f'Cache hash mismatch: {name}')
    with np.load(directory / 'observations.npz', allow_pickle=False) as stored:
        observations = {name: stored[name] for name in stored.files}
    with np.load(directory / 'fp_actions.npz', allow_pickle=False) as stored:
        actions = stored['actions']
    count = len(manifest['samples'])
    validate_actions(actions, count)
    expected_keys = {'image1', 'image2', 'state', 'task', 'recorded_action'}
    if set(observations) != expected_keys or any(len(value) != count for value in observations.values()):
        raise ValueError('Observation cache fields/count mismatch')
    if any(observations[key].dtype != np.uint8 for key in ('image1', 'image2')):
        raise ValueError('Camera cache must contain uint8 pixels')
    return manifest, observations, actions


def score_actions(candidate: np.ndarray, reference: np.ndarray, recorded: np.ndarray,
                  task_ids: np.ndarray) -> dict:
    """Equal task weighting; continuous controls and gripper are also reported."""
    task_ids = np.asarray(task_ids)
    count = len(task_ids)
    validate_actions(candidate, count)
    validate_actions(reference, count)
    if candidate.shape != reference.shape:
        raise ValueError('Candidate and FP action chunk shapes differ')
    if recorded.shape != (count, 7) or not np.isfinite(recorded).all():
        raise ValueError('Expected finite recorded first actions [N, 7]')
    difference = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    per_sample = {
        'chunk_mae_vs_fp': difference.mean(axis=(1, 2)),
        'continuous_chunk_mae_vs_fp': difference[:, :, :6].mean(axis=(1, 2)),
        'gripper_chunk_mae_vs_fp': difference[:, :, 6].mean(axis=1),
        # LIBERO command convention: >0 is one gripper direction, <=0 the other.
        'gripper_sign_disagreement_vs_fp':
            ((candidate[:, :, 6] > 0) != (reference[:, :, 6] > 0)).mean(axis=1),
        'first_action_mae_vs_recorded': np.abs(candidate[:, 0].astype(np.float64) - recorded).mean(axis=1),
        'fp_first_action_mae_vs_recorded': np.abs(reference[:, 0].astype(np.float64) - recorded).mean(axis=1),
    }
    per_task = []
    for task in sorted(set(task_ids.tolist())):
        chosen = task_ids == task
        per_task.append({'task_index': int(task), 'observations': int(chosen.sum()),
                         **{key: float(values[chosen].mean()) for key, values in per_sample.items()}})
    macro = {key: float(np.mean([row[key] for row in per_task])) for key in per_sample}
    macro['first_action_mae_change_vs_fp'] = (
        macro['first_action_mae_vs_recorded'] - macro['fp_first_action_mae_vs_recorded'])
    return {'scope': 'offline_action_proxy_not_success_rate', 'observations': count,
            'tasks': len(per_task), 'chunk_length': candidate.shape[1],
            'task_macro': macro, 'per_task': per_task,
            'per_observation': {key: values.tolist() for key, values in per_sample.items()}}
