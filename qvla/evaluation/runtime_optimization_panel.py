"""Short isolated correctness or final latency/resource panel for runtime variants."""
import argparse
import json
import os
import re
import threading
import time
from pathlib import Path
import numpy as np

def text(p):
    try:return Path(p).read_text()
    except OSError:return ''

def resource_sample():
    parents={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        m=re.search(r'^PPid:\s+(\d+)',text(p/'status'),re.M)
        if m:parents[int(p.name)]=int(m[1])
    ids=[os.getpid()]
    for pid in ids:
        ids.extend(p for p,parent in parents.items() if parent==pid and p not in ids)
    pss=[];rss=[]
    for pid in ids:
        s=text(f'/proc/{pid}/smaps_rollup')
        for out,name in ((pss,'Pss'),(rss,'Rss')):
            m=re.search('^'+name+r':\s+(\d+)',s,re.M);out.append(int(m[1]) if m else None)
    return {'time':time.monotonic(),'pids':ids,'pss_kib':sum(v or 0 for v in pss),'rss_kib':sum(v or 0 for v in rss),
            'complete':all(v is not None for v in pss),'npu_load':text('/sys/kernel/debug/rknpu/load').strip(),
            'thermal':[text(p).strip() for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')],
            'npu_freq':text('/sys/kernel/debug/rknpu/freq').strip()}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--options',default='{}')
    p.add_argument('--sequence',default='0,1,2,0');p.add_argument('--warmup',type=int,default=0)
    p.add_argument('--repeats',type=int,default=1);p.add_argument('--resources',action='store_true')
    a=p.parse_args()
    from qvla.runtime.smolvla_board_runtime import BoardSmolVLA
    a.output.mkdir(parents=True,exist_ok=True);options=json.loads(a.options)
    with np.load(a.inputs,allow_pickle=False) as z:panel={k:z[k].copy() for k in z.files}
    samples=[];stop=threading.Event();phase='initialization'
    def monitor():
        while not stop.is_set():
            r=resource_sample();r['phase']=phase;samples.append(r);stop.wait(.5)
    thread=threading.Thread(target=monitor,daemon=True)
    if a.resources:thread.start()
    model=None;actions=[];timings=[];begin=time.perf_counter()
    try:
        model=BoardSmolVLA(a.root,options=options) if options else BoardSmolVLA(a.root)
        init=time.perf_counter()-begin;sequence=[int(i) for i in a.sequence.split(',')]
        for i in range(a.warmup+len(sequence)*a.repeats):
            index=sequence[0] if i<a.warmup else sequence[(i-a.warmup)%len(sequence)]
            raw={k:v[index].copy() for k,v in panel.items()};phase='warmup' if i<a.warmup else 'measured'
            y,t=model.predict(raw)
            if i>=a.warmup:actions.append(y.copy());timings.append(dict(sample=index,**t))
            print(json.dumps({'iteration':i,'phase':phase,**t}),flush=True)
        phase='loaded';stop.wait(.6)
    finally:
        stop.set()
        if a.resources:thread.join(timeout=5)
        if model:model.close()
    values=[t['inference_ms'] for t in timings]
    report={'options':options,'init_seconds':init,'timings':timings,'p50_ms':float(np.median(values)),
            'p95_ms':float(np.percentile(values,95)),'mean_ms':float(np.mean(values)),
            'sample_interval_seconds':.5,'peak_pss_kib':max((s['pss_kib'] for s in samples),default=None),
            'peak_rss_kib':max((s['rss_kib'] for s in samples),default=None),
            'resource_processes_max':max((len(s['pids']) for s in samples),default=None),
            'resource_complete':all(s['complete'] for s in samples),
            'loaded_hashes':model.hashes,'input_shape':[list(v.shape) for v in actions]}
    np.savez_compressed(a.output/'actions.npz',actions=np.stack(actions))
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output/'resources.json').write_text(json.dumps(samples)+'\n')
    print(json.dumps({'stage':'completed','p50_ms':report['p50_ms'],'pss_mib':(report['peak_pss_kib'] or 0)/1024}),flush=True)

if __name__=='__main__':main()
