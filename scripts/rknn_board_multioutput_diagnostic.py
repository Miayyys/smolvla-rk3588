#!/usr/bin/env python3
"""Compare instrumented RKNN stage outputs on one pinned input."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from rknnlite.api import RKNNLite


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--stages',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    pixel=np.ascontiguousarray(np.load(args.input,allow_pickle=False).transpose(0,2,3,1))
    spec=json.loads(args.stages.read_text())
    report={'scope':'instrumented vision graph; one board inference, not a latency benchmark',
            'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),'outputs':[]}
    model=RKNNLite()
    try:
        if model.load_rknn(str(args.model)) != 0 or model.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
            raise RuntimeError('RKNN init failed')
        start=time.perf_counter()
        values=model.inference(inputs=[pixel],data_format=['nhwc'])
        report['elapsed_ms']=(time.perf_counter()-start)*1000
        if values is None or len(values) != len(spec['outputs']):
            raise RuntimeError('Wrong output count')
        np.savez_compressed(args.output.with_suffix('.npz'),
                            **{f'output_{i}':v for i,v in enumerate(values)})
        with np.load(args.reference,allow_pickle=False) as ref:
            for row,v in zip(spec['outputs'],values):
                expected=ref[f'output_{row["index"]}']
                if v.shape != expected.shape:
                    raise ValueError(f'Output {row["index"]} shape mismatch')
                d=v.astype(np.float64)-expected.astype(np.float64)
                rmse=float(np.sqrt(np.mean(d*d)))
                report['outputs'].append({**row,'mae':float(np.abs(d).mean()),
                    'rmse':rmse,'max_abs_error':float(np.abs(d).max()),
                    'relative_rmse':rmse/max(float(np.sqrt(np.mean(expected.astype(np.float64)**2))),1e-12)})
        report['status']='success'
    finally:
        model.release()
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
