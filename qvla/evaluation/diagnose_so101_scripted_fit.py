"""Training fit versus validation fit of a SO101 scripted-data checkpoint on recorded action chunks.

Same inference path as the trainer's offline check (bf16 autocast, fixed per-row noise seed). Errors are
split by chunk horizon (first action, first 10 = what closed loop executes, all 50) and by a grasp window
(the 20 frames before the gripper command first crosses half-closed). Test episodes are not read.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _close_rows(data, episodes, before=20):
    rows = []
    for ep in episodes:
        ep_rows = np.flatnonzero(data['episodes'] == ep)
        grip = data['actions'][ep_rows, -1]
        half = (grip.max() + grip.min()) / 2
        crossed = np.flatnonzero((grip[:-1] >= half) & (grip[1:] < half))
        if len(crossed):
            first = crossed[0]
            rows.extend(ep_rows[max(0, first - before):first + 1:5].tolist())
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=92, help='training episodes sampled (validation has 92)')
    args = p.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from lerobot.policies import make_pre_post_processors
    from qvla.evaluation.finetune_so100_real import sha
    from qvla.evaluation.finetune_so101_scripted import ScriptedFrames, _check_rows
    from qvla.evaluation.haq_offline_eval import load_policy
    cfg = json.loads(args.config.read_text()); run = Path(cfg['output'])
    stats = json.loads((run / 'stats.json').read_text())
    frames = ScriptedFrames(cfg, run)
    train, val = frames.parts[:2]
    train_eps = np.unique(train['episodes'][frames.data['train_rows']])
    picked = np.sort(np.random.default_rng(0).choice(train_eps, args.episodes, replace=False))
    val_eps = np.unique(val['episodes'])
    per = cfg['validation_frames_per_episode']
    sets = {'train': _check_rows(train, picked, per),
            'validation': [r + int(frames.bases[1]) for r in _check_rows(val, val_eps, per)],
            'train_grasp_window': _close_rows(train, picked),
            'validation_grasp_window': [r + int(frames.bases[1]) for r in _close_rows(val, val_eps)]}
    policy, _, _ = load_policy(SimpleNamespace(model_dir=args.model_dir.resolve(),
        vlm_assets_dir=Path(cfg['vlm_assets_dir']).resolve(), device='cuda'))
    policy.float().eval(); torch.set_num_threads(4)
    pre, post = make_pre_post_processors(policy.config, str(args.model_dir), preprocessor_overrides={
        'device_processor': {'device': 'cuda'}, 'tokenizer_processor': {'tokenizer_name': str(Path(cfg['vlm_assets_dir']).resolve())},
        'normalizer_processor': {'stats': stats}}, postprocessor_overrides={'unnormalizer_processor': {'stats': stats}})
    std = np.asarray(stats['action']['std'])
    report = {'scope': 'offline recorded chunks; training vs validation fit; not task success',
              'model_sha256': sha(args.model_dir / 'model.safetensors'), 'stats_sha256': sha(run / 'stats.json'),
              'train_episodes': picked.tolist(), 'units': 'degrees for 5 arm joints; gripper percentage points', 'results': {}}
    for name, rows in sets.items():
        errors = []
        for row in rows:
            raw = frames.batch([int(row)])
            target = raw['action'].numpy()[0]; valid = ~raw['action_is_pad'].numpy()[0]
            torch.manual_seed(100000 + int(row))
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = post(policy.predict_action_chunk(pre(raw))).float().cpu().numpy()[0]
            error = np.abs(prediction - target); error[~valid] = np.nan
            errors.append(error)
        errors = np.stack(errors)
        result = {'rows': len(rows)}
        for label, horizon in (('first', 1), ('first10', 10), ('all50', 50)):
            mae = np.nanmean(errors[:, :horizon].reshape(-1, errors.shape[-1]), axis=0)
            result[label] = {'per_joint_mae': np.round(mae, 3).tolist(), 'normalized_mae': round(float(np.mean(mae / std)), 4)}
        report['results'][name] = result
        print(json.dumps({name: {k: v['normalized_mae'] if isinstance(v, dict) else v for k, v in result.items()}}), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'fit.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
