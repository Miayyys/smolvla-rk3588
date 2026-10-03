#!/usr/bin/env python3
"""Run the SmolVLA prefix RKNN and verify all 16 per-layer K/V outputs."""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
import numpy as np
from rknnlite.api import RKNNLite


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--boundary',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    with np.load(args.boundary,allow_pickle=False) as z:
        inputs=[np.ascontiguousarray(z[n]) for n in ('prefix','attention_mask','position_ids')]
    report={'scope':'one fixed prefix RKNN on RK3588; full policy not executed',
            'model_sha256':digest(args.model),'model_bytes':args.model.stat().st_size,
            'boundary_sha256':digest(args.boundary),'status':'running',
            'inputs':[{'shape':list(x.shape),'dtype':str(x.dtype)} for x in inputs],
            'npu_core_mask':'NPU_CORE_0','warmup_runs':1,'timed_runs':2}
    model=RKNNLite()
    try:
        if model.load_rknn(str(args.model))!=0 or model.init_runtime(core_mask=RKNNLite.NPU_CORE_0)!=0:
            raise RuntimeError('RKNN init failed')
        durations=[]
        for i in range(3):
            start=time.perf_counter()
            values=model.inference(inputs=[x.copy() for x in inputs])
            elapsed=(time.perf_counter()-start)*1000
            if values is None or len(values)!=33:
                raise RuntimeError('Expected 33 non-empty prefix outputs')
            if i:durations.append(elapsed)
        rows=[]
        with np.load(args.reference,allow_pickle=False) as refs:
            for i,v in enumerate(values):
                ref=refs[f'output_{i}']
                if v.shape!=ref.shape:
                    raise ValueError(f'Output {i} shape {v.shape} differs from {ref.shape}')
                if not np.isfinite(v).all():raise ValueError(f'Non-finite output {i}')
                d=v.astype(np.float64)-ref.astype(np.float64)
                rows.append({'index':i,'shape':list(v.shape),'mae':float(np.abs(d).mean()),
                    'rmse':float(np.sqrt(np.mean(d*d))),'max_abs_error':float(np.abs(d).max())})
        path=args.output.with_suffix('.npz')
        np.savez_compressed(path,**{f'output_{i}':v for i,v in enumerate(values)})
        report.update(status='success',outputs=rows,output_sha256=digest(path),
                      durations_ms=durations,latency_p50_ms=float(np.percentile(durations,50)))
    except Exception as e:
        report.update(status='failed',error=repr(e));raise
    finally:
        report['rusage_maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        model.release()
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='outputs'},indent=2))


if __name__=='__main__':main()
