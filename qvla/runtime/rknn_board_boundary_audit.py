#!/usr/bin/env python3
"""Compare an RKNN subgraph against FP master with pinned boundary inputs."""

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
import time
from pathlib import Path
import numpy as np
from rknnlite.api import RKNNLite


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'inputs', 'reference', 'output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    with np.load(a.inputs) as z:
        inputs=[np.ascontiguousarray(z[k]) for k in z.files]
        input_names=z.files
    with np.load(a.reference) as z:
        refs={k:z[k].copy() for k in z.files}
    report={'scope':'isolated subgraph with FP master boundary inputs, not closed-loop quality',
            'hashes':{k:digest(getattr(a,k)) for k in ('model','inputs','reference')},
            'input_names':input_names,'status':'running'}
    m=RKNNLite()
    try:
        if m.load_rknn(str(a.model)) or m.init_runtime(core_mask=RKNNLite.NPU_CORE_0):
            raise RuntimeError('RKNN init failed')
        start=time.perf_counter()
        outputs=m.inference(inputs=inputs,data_format=['nchw']*len(inputs))
        report['inference_ms']=(time.perf_counter()-start)*1000
        if outputs is None or len(outputs)!=len(refs):
            raise ValueError('Output count differs')
        arrays={};rows=[]
        for (name,ref),value in zip(refs.items(),outputs):
            if value.shape!=ref.shape or not np.isfinite(value).all():
                raise ValueError('Invalid output '+name)
            d=value.astype(np.float64)-ref.astype(np.float64)
            rms=float(np.sqrt(np.mean(ref.astype(np.float64)**2)))
            rmse=float(np.sqrt(np.mean(d*d)))
            rows.append({'name':name,'shape':list(value.shape),'mae':float(np.abs(d).mean()),
                         'rmse':rmse,'relative_rmse':rmse/max(rms,1e-12),'max_abs':float(np.abs(d).max())})
            arrays[name]=value
        np.savez_compressed(a.output.with_suffix('.npz'),**arrays)
        report.update(status='executed',outputs=rows)
    except BaseException as e:
        report.update(status='failed',error=repr(e))
        raise
    finally:
        m.release()
        a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
