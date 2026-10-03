"""Diagnostic K/V reconstruction from RKLLM norm dumps and explicit FP weights.

This does not extract RKLLM's quantized cache or implement SmolVLA's mask.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dump','weights','reference','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    with np.load(args.weights,allow_pickle=False) as z:
        weights={k:z[k].copy() for k in z.files}
    with np.load(args.reference,allow_pickle=False) as z:
        refs={k:z[k].copy() for k in z.files}
    cos,sin=weights['cos'],weights['sin']
    tokens,dim=cos.shape
    layers=len([k for k in weights if k.startswith('key_weight_')])
    arrays={};rows=[];hashes={};started=time.perf_counter()
    for i in range(layers):
        path=args.dump/f'0-attn_norm-{i}'
        hidden=weights[f'key_weight_{i}'].shape[1]
        if path.stat().st_size!=tokens*hidden*4:raise ValueError('Norm dump size mismatch')
        n=np.fromfile(path,dtype='<f4').reshape(tokens,hidden)
        if not np.isfinite(n).all():raise ValueError('Non-finite norm')
        hashes[path.name]=sha(path)
        heads=weights[f'key_weight_{i}'].shape[0]//dim
        for name in ('key','value'):
            projected=n@weights[f'{name}_weight_{i}'].T
            a=projected.reshape(tokens,heads,dim).transpose(1,0,2)[None]
            if name=='key':
                rotate=np.concatenate((-a[...,dim//2:],a[...,:dim//2]),axis=-1)
                a=a*cos[None,None]+rotate*sin[None,None]
            a=np.ascontiguousarray(a,dtype=np.float32)
            ref=refs[f'{name}_{i}']
            if a.shape!=ref.shape or not np.isfinite(a).all():raise ValueError('Invalid K/V output')
            d=a.astype(np.float64)-ref.astype(np.float64)
            rows.append({'name':f'{name}_{i}','shape':list(a.shape),'mae':float(np.abs(d).mean()),
                         'rmse':float(np.sqrt(np.mean(d*d))),
                         'relative_rmse':float(np.sqrt(np.mean(d*d))/max(np.sqrt(np.mean(ref.astype(np.float64)**2)),1e-12))})
            arrays[f'{name}_{i}']=a
    elapsed=(time.perf_counter()-started)*1000
    np.savez_compressed(args.output.with_suffix('.npz'),**arrays)
    report={'scope':'norm dump plus explicit FP CPU projection; not native quantized K/V extraction',
            'weights_sha256':sha(args.weights),'reference_sha256':sha(args.reference),
            'norm_hashes':hashes,'outputs':rows,'reconstruction_ms':elapsed,'status':'executed'}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
