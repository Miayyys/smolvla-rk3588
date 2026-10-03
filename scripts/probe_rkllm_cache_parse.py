"""Validate a fixed tiny FP16 RKLLM cache fixture; NOT a general cache parser."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('cache','reference','output'):p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args();b=args.cache.read_bytes()
    # Only this SDK 1.3.1 / 2-layer / 8-token / 2-KV-head / D64 fixture.
    if len(b)<8244 or np.frombuffer(b[:12],dtype='<u4').tolist()!=[0,8,8]:
        raise ValueError('Not the verified eight-token cache fixture')
    if np.frombuffer(b[12:44],dtype='<i4').tolist()!=list(range(100,108)):
        raise ValueError('Unexpected fixture token IDs')
    arrays={};rows=[]
    with np.load(args.reference,allow_pickle=False) as refs:
        for name,tail in [('key_0',8244),('key_1',6184),('value_0',4124),('value_1',2064)]:
            v=np.frombuffer(b,dtype='<f2',count=1024,offset=len(b)-tail).astype('f4')
            if name.startswith('key'):
                v=v.reshape(8,2,32,2).transpose(1,0,3,2).reshape(1,2,8,64)
            else:v=v.reshape(2,64,8).transpose(0,2,1)[None]
            ref=refs[name]
            if ref.shape!=v.shape or not np.isfinite(v).all():raise ValueError('Invalid K/V')
            d=v.astype('f8')-ref.astype('f8')
            rows.append({'name':name,'mae':float(np.abs(d).mean()),'max_abs':float(np.abs(d).max())})
            arrays[name]=v
    np.savez_compressed(args.output.with_suffix('.npz'),**arrays)
    report={'scope':'fixed tiny FP16 native cache parsed; no dump or recomputation',
            'general_parser':False,'cache_sha256':hashlib.sha256(b).hexdigest(),
            'reference_sha256':hashlib.sha256(args.reference.read_bytes()).hexdigest(),
            'layout':'fixed from first input, independently checked on second input',
            'outputs':rows,'parity_within_0_001_max_abs':all(x['max_abs']<.001 for x in rows)}
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
