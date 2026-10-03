"""Compare saved isolated native caches with the matching FP32 language algebra."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaForCausalLM


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--lengths', type=int, nargs='+', default=[64, 96, 128, 160])
    a = p.parse_args()
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LlamaForCausalLM.from_pretrained(
        a.model, dtype=torch.float32, attn_implementation='eager').model.to(device).eval()
    report = {'scope': 'valid-token native KV vs FP32; not task quality',
              'device': device, 'weights_sha256': sha256(a.model / 'model.safetensors'),
              'lengths': {}}
    for n in a.lengths:
        valid = min(n, 151)
        input_path, cache_path = a.inputs / f'input{n}.bin', a.inputs / f'cache{n}.npz'
        x = torch.from_numpy(np.fromfile(input_path, '<f4').reshape(1, n, 960)).to(device)
        allowed = torch.zeros((n, n), dtype=torch.bool, device=device)
        allowed[:, :valid-1] = True
        allowed[valid-1, valid-1] = True
        mask = torch.where(allowed, 0., torch.finfo(torch.float32).min)[None, None]
        with torch.inference_mode():
            y = model(inputs_embeds=x, attention_mask=mask,
                      position_ids=torch.arange(n, device=device)[None], use_cache=True)
        native = np.load(cache_path)['values']
        rows = []
        for repeat in range(len(native)):
            layers = []
            for i, layer in enumerate(y.past_key_values.layers):
                key = native[repeat, i].reshape(n, 5, 32, 2).transpose(1, 0, 3, 2).reshape(1, 5, n, 64)
                value = native[repeat, 16+i].reshape(5, 64, n).transpose(0, 2, 1)[None]
                row = {'layer': i}
                for name, board, ref in [('key', key, layer.keys), ('value', value, layer.values)]:
                    delta = np.abs(board[:, :, :valid] - ref.cpu().numpy()[:, :, :valid])
                    row[name + '_mae'] = float(delta.mean())
                    row[name + '_max'] = float(delta.max())
                layers.append(row)
            rows.append({'repeat': repeat, 'layers': layers})
        report['lengths'][str(n)] = {'actual_tokens': valid, 'input_sha256': sha256(input_path),
                                    'cache_sha256': sha256(cache_path), 'comparisons': rows}
        print(json.dumps({'tokens': n, 'first_run': rows[0]['layers']}), flush=True)
    (a.inputs / 'fp32_comparison.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
