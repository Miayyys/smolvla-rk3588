"""Audit captured native MatMul boundaries; no task-quality interpretation."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();r=a.root
    log=(r/'worker.log').read_text()
    shapes={int(j):(name,int(m),int(k),int(n)) for j,name,m,k,n in re.findall(
        r'MATMUL_SYNC round=0 index=(\d+) input=0 tensor=(\S+) M=(\d+) K=(\d+) N=(\d+)',log)}
    rounds=sorted({int(re.match(r'round(\d+)_',p.name)[1]) for p in r.glob('round*_op*_output_*.bin')})
    report={'scope':'native MatMul diagnostic snapshots; not task quality or performance',
            'first_changed_output':[], 'qk_reference':[], 'files':{}}
    def read(path,dtype):
        raw=path.read_bytes();report['files'][path.name]=hashlib.sha256(raw).hexdigest()
        return np.frombuffer(raw,dtype=dtype).astype('f4')
    for repeat in rounds:
        first=False
        for j in sorted(shapes):
            name,m,k,n=shapes[j]
            base=r/f'round0_op{j:03}_output_{name}.bin'
            path=r/f'round{repeat}_op{j:03}_output_{name}.bin'
            if not path.exists():continue
            current=read(path,'<f4');original=read(base,'<f4')
            d=np.abs(current-original)
            if repeat and not first and np.any(d):
                inputs=[]
                for inp in sorted(r.glob(f'round0_op{j:03}_input_*.bin')):
                    dd=np.abs(read(inp,'<f2')-read(r/inp.name.replace('round0_',f'round{repeat}_'),'<f2'))
                    inputs.append({'tensor':inp.name,'max_abs':float(dd.max()),'changed_elements':int(np.count_nonzero(dd))})
                report['first_changed_output'].append({'repeat':repeat,'index':j,'tensor':name,
                    'max_abs':float(d.max()),'changed_elements':int(np.count_nonzero(d)),'inputs':inputs});first=True
            if name=='matmul_qk_C':
                left=read(r/f'round{repeat}_op{j:03}_input_matmul_qk_A.bin','<f2').reshape(m,k)
                right=read(r/f'round{repeat}_op{j:03}_input_matmul_qk_feature_B.bin','<f2').reshape(n,k)
                # This boundary's normal row-major interpretation was checked
                # against healthy native results and the public MatMul API.
                delta=np.abs(current.reshape(m,n)-left@right.T)
                report['qk_reference'].append({'repeat':repeat,'index':j,'mae':float(delta.mean()),
                    'max_abs':float(delta.max()),'columns_above_0p001_diagnostic':np.where(delta.max(0)>1e-3)[0].tolist()})
    (r/'matmul_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'first_changed_output':report['first_changed_output'],
                     'qk_above_0p001': [x for x in report['qk_reference'] if x['max_abs']>1e-3]}),flush=True)


if __name__=='__main__':
    main()
