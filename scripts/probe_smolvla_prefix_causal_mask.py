#!/usr/bin/env python3
"""Measure changing only SmolVLA prefix attention to a causal mask.

This is a PyTorch compatibility diagnostic, not an RKLLM inference test.
"""
import argparse
import json
import os
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
import torch

from haq_offline_eval import identity, load_policy, predict_cached
from qvla_haq.offline_actions import file_sha256, load_cache

ROOT = Path(__file__).resolve().parents[1]


def metrics(actual, reference):
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    return {'mae': float(np.abs(delta).mean()),
            'rmse': float(np.sqrt(np.square(delta).mean())),
            'max_abs_error': float(np.abs(delta).max())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, default=ROOT/'artifacts/transfer/model')
    p.add_argument('--vlm-assets-dir', type=Path, default=ROOT/'artifacts/transfer/smolvlm2_assets')
    p.add_argument('--splits', type=Path, default=ROOT/'data/libero_splits.json')
    p.add_argument('--partition', type=Path, default=ROOT/'config/evaluation_partition_v2.json')
    p.add_argument('--cache', type=Path, default=ROOT/'runs/haq_offline_local_v1/fp_cache40')
    p.add_argument('--output-dir', type=Path, default=ROOT/'runs/rkllm_prefix_feasibility_v1')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    torch.set_num_threads(4)
    ident = identity(args)
    manifest, obs, fp = load_cache(args.cache, ident)
    if torch.__version__ != manifest['torch_version'] or args.device != manifest['device']:
        raise ValueError('FP cache runtime mismatch')
    policy, pre, post = load_policy(args)
    module = policy.model.vlm_with_expert
    forward = module.forward
    chosen = {k: v[:1] for k, v in obs.items()}
    sample = manifest['samples'][:1]
    arrays, timings = {}, {}
    for mode in ('original', 'causal'):
        def traced(*a, **kw):
            prefix_call = kw['inputs_embeds'][1] is None and kw.get('past_key_values') is None
            if prefix_call:
                mask = kw['attention_mask']
                if mode == 'causal':
                    n = mask.shape[-1]
                    mask = mask & torch.ones((n, n), device=mask.device, dtype=torch.bool).tril()[None]
                    kw['attention_mask'] = mask
                arrays[f'{mode}_mask'] = mask.detach().cpu().numpy().copy()
                arrays[f'{mode}_positions'] = kw['position_ids'].detach().cpu().numpy().copy()
            out, cache = forward(*a, **kw)
            if prefix_call:
                arrays[f'{mode}_hidden'] = out[0].detach().float().cpu().numpy().copy()
                for i, layer in enumerate(cache.layers):
                    for kind, t in [('key', layer.keys), ('value', layer.values)]:
                        arrays[f'{mode}_{kind}_{i}'] = t.detach().float().cpu().numpy().copy()
            return out, cache
        module.forward = traced
        try:
            arrays[f'{mode}_actions'], timings[mode] = predict_cached(
                policy, pre, post, chosen, sample, args.device)
        finally:
            module.forward = forward
    np.testing.assert_array_equal(arrays['original_actions'], fp[:1])
    original, causal = arrays['original_mask'], arrays['causal_mask']
    n = original.shape[-1]
    report = {'scope': 'one development observation; PyTorch mask ablation, NOT RKLLM execution',
              'identity': ident, 'sample': sample[0], 'torch_version': torch.__version__,
              'device': args.device, 'original_actions_match_fp_cache_exactly': True,
              'prefix_shape': list(arrays['original_hidden'].shape),
              'mask_shape': list(original.shape),
              'original_allowed_pairs': int(original.sum()),
              'original_future_allowed_pairs': int((original & np.triu(np.ones((n,n), bool), 1)[None]).sum()),
              'removed_allowed_pairs': int((original & ~causal).sum()),
              'positions_unchanged': bool(np.array_equal(arrays['original_positions'], arrays['causal_positions'])),
              'last_hidden': metrics(arrays['causal_hidden'], arrays['original_hidden']),
              'per_layer_kv': [{'layer': i, **{kind: metrics(arrays[f'causal_{kind}_{i}'], arrays[f'original_{kind}_{i}'])
                                  for kind in ('key', 'value')}} for i in range(module.num_vlm_layers)],
              'actions': metrics(arrays['causal_actions'], arrays['original_actions']),
              'gripper_sign_changes': int(((arrays['causal_actions'][...,-1] > 0) !=
                                           (arrays['original_actions'][...,-1] > 0)).sum()),
              'timings': timings, 'script_sha256': file_sha256(Path(__file__))}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir/'mask_ablation.npz'
    np.savez_compressed(path, **arrays)
    report['raw_sha256'] = file_sha256(path)
    (args.output_dir/'mask_ablation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('identity','per_layer_kv')}, indent=2))


if __name__ == '__main__':
    main()
