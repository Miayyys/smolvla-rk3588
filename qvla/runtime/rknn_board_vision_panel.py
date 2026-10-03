#!/usr/bin/env python3
"""Run 80 fixed real camera inputs through one loaded RKNN model on RK3588."""

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
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args()
    with np.load(args.panel,allow_pickle=False) as z:
        pixels=z['pixels'].copy();reference=z['reference'].copy()
    if pixels.shape!=(80,3,512,512) or reference.shape!=(80,64,960):
        raise ValueError('Unexpected pinned input shape')
    args.root.mkdir(parents=True,exist_ok=True)
    report={'scope':'80 real camera forwards through one loaded vision graph, core0',
            'model_sha256':digest(args.model),'panel_sha256':digest(args.panel),'status':'running',
            'input_layout':'explicit NHWC converted from pinned NCHW','durations_ms':[]}
    model=RKNNLite()
    try:
        if model.load_rknn(str(args.model))!=0 or model.init_runtime(core_mask=RKNNLite.NPU_CORE_0)!=0:
            raise RuntimeError('RKNN init failed')
        values=[]
        for i,pixel in enumerate(pixels):
            x=np.ascontiguousarray(pixel[None].transpose(0,2,3,1))
            if i==0:
                for _ in range(2):
                    warm=model.inference(inputs=[x],data_format=['nhwc'])
                    if warm is None:raise RuntimeError('Warmup failed')
            start=time.perf_counter()
            output=model.inference(inputs=[x],data_format=['nhwc'])
            report['durations_ms'].append((time.perf_counter()-start)*1000)
            if output is None or len(output)!=1 or output[0].shape!=(1,64,960):
                raise RuntimeError(f'Inference {i} failed')
            values.append(output[0].astype(np.float32))
            if (i+1)%10==0:print(f'camera {i+1}/80',flush=True)
        features=np.concatenate(values)
        if not np.isfinite(features).all():raise ValueError('Non-finite output')
        path=args.root/'board_features.npz'
        np.savez_compressed(path,features=features)
        delta=features.astype(np.float64)-reference.astype(np.float64)
        report.update(status='success',output_sha256=digest(path),
                      features_mae=float(np.abs(delta).mean()),
                      features_rmse=float(np.sqrt(np.mean(delta**2))),
                      per_camera_mae=np.abs(delta).mean(axis=(1,2)).tolist(),
                      latency_p50_ms=float(np.percentile(report['durations_ms'],50)),
                      latency_p95_ms=float(np.percentile(report['durations_ms'],95)))
    finally:
        model.release()
        (args.root/'board_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('durations_ms','per_camera_mae')},indent=2))


if __name__=='__main__':main()
