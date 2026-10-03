"""Check extracted language algebra with original mask and prepare native causal reference."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transformers import LlamaForCausalLM

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--boundary-root', type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(4)
model = LlamaForCausalLM.from_pretrained(a.root / 'hf_model', dtype=torch.float32,
                                        attn_implementation='eager').eval()
b = np.load(a.boundary_root / 'prefix_boundary.npz')
original = np.load(a.boundary_root / 'prefix_reference.npz')
x = torch.from_numpy(b['prefix']).float()
x.numpy().astype('<f4').tofile(a.root / 'input_embeds.bin')
mask = torch.where(torch.from_numpy(b['attention_mask'])[:, None], 0., torch.finfo(torch.float32).min)
report = {'scope': 'selected language FP32 recontainerization and causal mismatch, not board performance'}
for mode in ('original_mask', 'native_causal'):
    kw = {'attention_mask': mask, 'position_ids': torch.from_numpy(b['position_ids'])} if mode == 'original_mask' else {}
    with torch.inference_mode():
        y = model.model(inputs_embeds=x, use_cache=True, **kw)
    arrays = {'prefix_hidden': y.last_hidden_state.numpy()}
    for i, layer in enumerate(y.past_key_values.layers):
        arrays[f'key_{i}'] = layer.keys.numpy()
        arrays[f'value_{i}'] = layer.values.numpy()
    np.savez_compressed(a.root / f'{mode}_reference.npz', **arrays)
    rows = []
    for k, v in arrays.items():
        d = v.astype('f8') - original[k].astype('f8')
        rows.append(dict(name=k, mae=float(np.abs(d).mean()), max_abs=float(np.abs(d).max())))
    report[mode] = rows
(a.root / 'reference_report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v[:3] if isinstance(v, list) else v for k, v in report.items()}))
