"""Validate SDK 1.3.1 FP16 cache records for the selected 16-layer language probe."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import argparse
import hashlib
import json
import struct
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--tokens', type=int, default=177)
a = p.parse_args()
b = (a.root / 'prompt_cache.bin').read_bytes()
layers, tokens, heads, dim = 16, a.tokens, 5, 64
assert struct.unpack('<III', b[:12]) == (0, 8, tokens)
assert np.array_equal(np.frombuffer(b, '<i4', tokens, 12), np.arange(100, 100 + tokens))
assert b[-16:] == bytes.fromhex('6e736567010000000000000000000000')
size = tokens * heads * dim * 2
refs = {mode: np.load(a.root / f'{mode}_reference.npz') for mode in ('original_mask', 'native_causal')}
rows, arrays = [], {}
for j in range(layers * 2):
    key = j < layers
    name = f'{"key" if key else "value"}_{j % layers}'
    offset = len(b) - 16 - (layers * 2 - j) * (size + 12) + 12
    metadata = struct.unpack('<III', b[offset-12:offset])
    assert metadata == ((1, heads * dim * 2, 0) if key else (1, 2, heads * dim)), metadata
    v = np.frombuffer(b, '<f2', tokens * heads * dim, offset).astype('f4')
    if key:
        v = v.reshape(tokens, heads, dim//2, 2).transpose(1, 0, 3, 2).reshape(1, heads, tokens, dim)
    else:
        v = v.reshape(heads, dim, tokens).transpose(0, 2, 1)[None]
    assert np.isfinite(v).all()
    arrays[name] = v
    row = {'name': name}
    for mode, ref in refs.items():
        d = v.astype('f8') - ref[name].astype('f8')
        row[mode] = dict(mae=float(np.abs(d).mean()), max_abs=float(np.abs(d).max()))
    rows.append(row)
np.savez_compressed(a.root / 'native_cache_kv.npz', **arrays)
report = dict(scope='real 16-layer FP16 native RKLLM K/V; fixed SDK layout with record checks',
              cache_sha256=hashlib.sha256(b).hexdigest(), outputs=rows,
              summary={mode: dict(mean_mae=float(np.mean([r[mode]['mae'] for r in rows])),
                                  max_abs=max(r[mode]['max_abs'] for r in rows)) for mode in refs})
(a.root / 'native_cache_report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report['summary']))
