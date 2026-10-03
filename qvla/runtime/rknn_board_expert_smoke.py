#!/usr/bin/env python3
"""Validate all expert RKNN inputs on RK3588 with explicit KV layout."""

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
import resource
import time
from pathlib import Path
import numpy as np
from rknnlite.api import RKNNLite


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','boundary','reference','export','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    spec=json.loads(args.export.read_text())
    with np.load(args.boundary,allow_pickle=False) as z:
        inputs=[np.ascontiguousarray(z[row['name']]) for row in spec['inputs']]
        original=z['original_velocity'].copy()
    reference=np.load(args.reference,allow_pickle=False)
    report={'scope':'one complete expert step on RK3588, not a full policy',
            'model_sha256':digest(args.model),'boundary_sha256':digest(args.boundary),
            'model_bytes':args.model.stat().st_size,'input_count':len(inputs),
            'data_format':'explicit nchw, including every KV tensor','status':'running',
            'npu_core_mask':'NPU_CORE_0','warmup_runs':1,'timed_runs':2}
    model=RKNNLite()
    try:
        if model.load_rknn(str(args.model)) or model.init_runtime(core_mask=RKNNLite.NPU_CORE_0):
            raise RuntimeError('RKNN initialization failed')
        durations=[]
        for i in range(3):
            started=time.perf_counter()
            values=model.inference(inputs=[x.copy() for x in inputs],data_format=['nchw']*len(inputs))
            elapsed=(time.perf_counter()-started)*1000
            if values is None or len(values)!=1:raise RuntimeError('Expected one velocity output')
            if i:durations.append(elapsed)
        value=values[0]
        if value.shape!=reference.shape or not np.isfinite(value).all():
            raise ValueError('Invalid velocity shape or values')
        def errors(ref):
            d=value.astype(np.float64)-ref.astype(np.float64)
            return {'mae':float(np.abs(d).mean()),'rmse':float(np.sqrt(np.mean(d*d))),
                    'max_abs_error':float(np.abs(d).max())}
        path=args.output.with_suffix('.npy');np.save(path,value,allow_pickle=False)
        report.update(status='success',shape=list(value.shape),vs_fp32=errors(reference),
                      vs_original_loaded=errors(original),durations_ms=durations,
                      latency_p50_ms=float(np.percentile(durations,50)),output_sha256=digest(path))
    except Exception as e:
        report.update(status='failed',error=repr(e));raise
    finally:
        report['rusage_maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        model.release();args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
