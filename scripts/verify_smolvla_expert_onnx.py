#!/usr/bin/env python3
"""Validate expert ONNX at the captured and a different time/noise input."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('runs/smolvla_expert_step_split_v1'))
    args=p.parse_args();r=args.root
    options=ort.SessionOptions();options.intra_op_num_threads=4
    model=ort.InferenceSession(str(r/'expert_step.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    rows=[]
    for case,path in [('captured','expert_boundary.npz'),('different_time_and_noise','expert_alternate.npz')]:
        with np.load(r/path,allow_pickle=False) as z:
            feed={item.name:z[item.name] for item in model.get_inputs()}
            ref=np.load(r/'expert_fp32_reference.npy') if case=='captured' else z['velocity']
            out=model.run(None,feed)[0]
        if out.shape!=ref.shape:raise ValueError('Expert output shape mismatch')
        delta=out.astype(np.float64)-ref.astype(np.float64)
        rows.append({'case':case,'mae':float(np.abs(delta).mean()),
                     'max_abs_error':float(np.abs(delta).max()),
                     'allclose_1e_4':bool(np.allclose(out,ref,rtol=1e-4,atol=1e-4))})
        np.save(r/f'onnx_{case}_velocity.npy',out,allow_pickle=False)
    report={'scope':'expert ONNX export validation; two time/noise inputs, not task evaluation',
        'onnxruntime':ort.__version__,'input_count':len(model.get_inputs()),
        'onnx_sha256':hashlib.sha256((r/'expert_step.onnx').read_bytes()).hexdigest(),
        'input_names':[x.name for x in model.get_inputs()],
        'rows':rows,'passed':all(x['allclose_1e_4'] for x in rows)}
    (r/'onnx_parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise RuntimeError('Expert ONNX mismatch')


if __name__=='__main__':main()
