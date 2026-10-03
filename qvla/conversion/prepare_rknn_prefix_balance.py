#!/usr/bin/env python3
"""Reversible prefix input scaling fitted only on isolated calibration frames."""

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
import shlex
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper
import onnxruntime as ort


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.source=a.source.resolve();a.output=a.output.resolve();a.output.mkdir(exist_ok=True)
    lines=[shlex.split(x) for x in (a.source/'prefix_dataset.txt').read_text().splitlines() if x.strip()]
    if len(lines)!=40 or any(len(x)!=3 for x in lines):raise ValueError('Expected pinned 40-prefix calibration')
    maxima=None
    for row in lines:
        x=np.load(row[0],allow_pickle=False)
        maxima=np.abs(x) if maxima is None else np.maximum(maxima,np.abs(x))
    scale=np.maximum(maxima,1.0).astype(np.float32)
    np.save(a.output/'prefix_input_scale.npy',scale,allow_pickle=False)
    model=onnx.load(str(a.source/'prefix.onnx'))
    if model.graph.input[0].name!='prefix':raise ValueError('Unexpected graph contract')
    model.graph.input[0].name='balanced_prefix'
    model.graph.initializer.append(numpy_helper.from_array(scale,name='QVLAPrefixScale'))
    model.graph.node.insert(0,helper.make_node('Mul',['balanced_prefix','QVLAPrefixScale'],['prefix'],name='QVLAPrefixInputRestore'))
    onnx.checker.check_model(model);onnx.save(model,str(a.output/'prefix.onnx'))
    dataset=[]
    for i,row in enumerate(lines):
        path=a.output/f'balanced_calibration_{i:02d}.npy'
        np.save(path,np.load(row[0],allow_pickle=False)/scale,allow_pickle=False)
        dataset.append(shlex.join([str(path),*row[1:]]))
    (a.output/'prefix_dataset.txt').write_text('\n'.join(dataset)+'\n')
    meta=json.loads((a.source/'prefix_export.json').read_text())
    meta.update(onnx_sha256=sha(a.output/'prefix.onnx'),onnx_bytes=(a.output/'prefix.onnx').stat().st_size,
                input_names=['balanced_prefix',*meta['input_names'][1:]],
                quantized_input_bounds={'balanced_prefix':[-2.0,2.0]})
    (a.output/'prefix_export.json').write_text(json.dumps(meta,indent=2)+'\n')
    with np.load(a.source/'prefix_boundary.npz') as z:inputs={k:z[k].copy() for k in z.files}
    balanced={'balanced_prefix':inputs['prefix']/scale,**{k:v for k,v in inputs.items() if k!='prefix'}}
    np.savez_compressed(a.output/'prefix_boundary.npz',**balanced)
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(a.output/'prefix.onnx'),options,providers=['CPUExecutionProvider'])
    values=session.run(None,balanced)
    with np.load(a.source/'prefix_reference.npz') as z:refs={k:z[k].copy() for k in z.files}
    metrics=[]
    for (name,ref),value in zip(refs.items(),values):
        d=value.astype(np.float64)-ref.astype(np.float64)
        passed=bool(np.allclose(value,ref,atol=2e-4,rtol=2e-4))
        metrics.append({'name':name,'max_abs':float(np.abs(d).max()),'mae':float(np.abs(d).mean()),'passed':passed})
    np.savez_compressed(a.output/'prefix_reference.npz',**refs)
    report={'scope':'calibration-only reversible per-position/channel input balance; FP parity before quantization',
            'source_onnx_sha256':sha(a.source/'prefix.onnx'),'balanced_onnx_sha256':sha(a.output/'prefix.onnx'),
            'scale_sha256':sha(a.output/'prefix_input_scale.npy'),'calibration_input_hashes':[sha(Path(x[0])) for x in lines],
            'formula':'scale=max(max_abs_over_40_calibration_prefixes,1); board_input=prefix/scale; graph_input_restore=board_input*scale',
            'scale_shape':list(scale.shape),'balanced_replay_abs_max':float(np.abs(balanced['balanced_prefix']).max()),
            'passed':all(x['passed'] for x in metrics),'outputs':metrics}
    (a.output/'balance_export_report.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:raise ValueError('Reversible balance FP parity failed')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
