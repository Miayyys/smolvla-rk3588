"""Deterministic tiny ONNX MatMul: target-specific RKNN conversion probe."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import time
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from rknn.api import RKNN

out = Path('runs/tiny_w4a16_probe').resolve()
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20260928)
w = (rng.standard_normal((128, 64)) / np.sqrt(128)).astype(np.float32)
graph = helper.make_graph([helper.make_node('MatMul', ['input', 'weight'], ['output'])],
    'tiny_linear', [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 16, 128])],
    [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 16, 64])], [numpy_helper.from_array(w, 'weight')])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=8)
onnx.checker.check_model(model)
onnx.save(model, out / 'tiny.onnx')
cal = []
for i in range(8):
    path = out / f'cal_{i}.npy'
    np.save(path, rng.standard_normal((1, 16, 128)).astype(np.float32))
    cal.append(str(path))
(out / 'dataset.txt').write_text('\n'.join(cal) + '\n')
x = rng.standard_normal((1, 16, 128)).astype(np.float32)
np.save(out / 'input.npy', x)
np.save(out / 'reference.npy', x @ w)
report = {'python': platform.python_version(), 'versions': {p: metadata.version(p) for p in ['rknn-toolkit2', 'torch', 'onnx', 'numpy']},
          'seed': 20260928, 'shape': [1,16,128], 'weight_shape': [128,64], 'calibration_count': 8, 'evaluation_count': 1,
          'scope': 'synthetic conversion probe; not VLA quality or board execution', 'attempts': []}
for target, dtype in [('rk3588','w4a16'), ('rk3588','w8a8'), ('rk3576','w4a16')]:
    row = {'target': target, 'dtype': dtype}
    r = RKNN(verbose=False)
    start = time.perf_counter()
    try:
        for stage, fn in [
            ('config', lambda: r.config(target_platform=target, quantized_dtype=dtype)),
            ('load_onnx', lambda: r.load_onnx(model=str(out / 'tiny.onnx'))),
            ('build', lambda: r.build(do_quantization=True, dataset=str(out / 'dataset.txt'))),
            ('export', lambda: r.export_rknn(str(out / f'{target}_{dtype}.rknn')))]:
            row['stage'] = stage
            ret = fn()
            row[stage + '_return'] = ret
            if ret != 0:
                raise RuntimeError(f'{stage} returned {ret}')
        row['bytes'] = (out / f'{target}_{dtype}.rknn').stat().st_size
        row['success'] = True
    except Exception as exc:
        row['success'] = False
        row['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        r.release()
    row['conversion_seconds'] = time.perf_counter() - start
    report['attempts'].append(row)
    report['hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.suffix in ['.onnx','.npy','.rknn','.txt']}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(row), flush=True)
