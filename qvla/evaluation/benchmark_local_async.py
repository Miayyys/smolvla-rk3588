"""Original SmolVLA GPU latency and 30-Hz shadow action consumption, no robot I/O."""
import argparse
import hashlib
import json
import os
import queue
import threading
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from qvla.evaluation.haq_offline_eval import load_policy

ROOT=Path(__file__).resolve().parents[2]

def stats(values):
    a=np.asarray(values,dtype=np.float64)
    return {'count':len(a),'mean':float(a.mean()),'p50':float(np.percentile(a,50)),
            'p95':float(np.percentile(a,95)),'p99':float(np.percentile(a,99)),
            'min':float(a.min()),'max':float(a.max())} if len(a) else {}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'runs/local_async30_v1')
    p.add_argument('--repeats',type=int,default=60)
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--seconds',type=float,default=30)
    p.add_argument('--hz',type=float,default=30)
    p.add_argument('--request-every',type=int,default=10)
    args=p.parse_args()
    if args.repeats<1 or args.warmup<0 or args.seconds<=0 or args.hz<=0 or args.request_every<1:
        p.error('Use positive repeats/seconds/hz/request-every and nonnegative warmup')
    if not torch.cuda.is_available():p.error('CUDA GPU required for this benchmark')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    cfg=SimpleNamespace(model_dir=ROOT/'artifacts/transfer/model',vlm_assets_dir=ROOT/'artifacts/transfer/smolvlm2_assets',device='cuda')
    with np.load(ROOT/'runs/haq_offline_local_v1/fp_cache40/observations.npz',allow_pickle=False) as z:
        observations={name:z[name].copy() for name in ('image1','image2','state','task')}
    noises=np.random.default_rng(42).standard_normal((len(observations['task']),1,50,32)).astype('float32')
    started=time.perf_counter();policy,pre,post=load_policy(cfg)
    load_seconds=time.perf_counter()-started
    print(json.dumps({'stage':'loaded','seconds':load_seconds,'dtypes':dict(Counter(str(x.dtype) for x in policy.parameters()))}),flush=True)
    def infer(index):
        index%=len(observations['task'])
        torch.cuda.synchronize();t=time.perf_counter()
        obs={'observation.images.image':torch.from_numpy(observations['image1'][index].copy()).float()/255,
             'observation.images.image2':torch.from_numpy(observations['image2'][index].copy()).float()/255,
             'observation.state':torch.from_numpy(observations['state'][index].copy()),
             'task':str(observations['task'][index])}
        with torch.inference_mode():
            policy.reset()
            batch=pre(obs)
            noise=torch.from_numpy(noises[index]).to('cuda')
            result=post(policy.predict_action_chunk(batch,noise=noise)).detach().float().cpu().numpy()[0]
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-t
        if result.shape!=(50,7) or not np.isfinite(result).all():raise ValueError('Bad complete action chunk')
        return result,elapsed
    for i in range(args.warmup):infer(i)
    torch.cuda.reset_peak_memory_stats();times=[]
    for i in range(args.repeats):
        actions,elapsed=infer(i);times.append(elapsed)
        if (i+1)%10==0:print(json.dumps({'stage':'latency','done':i+1,'seconds':elapsed}),flush=True)
    serial=stats(times)
    # Initial chunk is deliberately ready before the control clock starts.
    initial,initial_seconds=infer(0)
    pending=queue.Queue(maxsize=1);stop=threading.Event();lock=threading.Lock()
    timeline={i:row.copy() for i,row in enumerate(initial)}
    worker_times=[];worker_errors=[];requests=[]
    def worker():
        while not stop.is_set():
            try:tick,index=pending.get(timeout=.05)
            except queue.Empty:continue
            try:
                chunk,elapsed=infer(index)
                now_tick=int(np.ceil((time.perf_counter()-epoch)*args.hz))
                skipped=max(0,now_tick-tick)
                with lock:
                    for offset in range(skipped,len(chunk)):timeline[tick+offset]=chunk[offset].copy()
                worker_times.append(elapsed)
                requests.append({'observation_tick':tick,'ready_tick':now_tick,'discarded_late_actions':min(skipped,len(chunk)),'seconds':elapsed})
            except Exception as exc:
                worker_errors.append(repr(exc));stop.set()
    # Shadow control: consume a real predicted vector every tick, no serial writes.
    epoch=time.perf_counter();thread=threading.Thread(target=worker,daemon=True);thread.start()
    ticks=int(args.seconds*args.hz);actual=[];lateness=[];durations=[];underruns=[];request_drops=0
    try:
        for tick in range(ticks):
            target=epoch+tick/args.hz
            remaining=target-time.perf_counter()
            if remaining>0:time.sleep(remaining)
            start=time.perf_counter();actual.append(start);lateness.append(max(0,start-target))
            with lock:
                action=timeline.pop(tick,None)
                # Bound retained timeline; only future actions are useful.
                for stale in [n for n in timeline if n<tick]:timeline.pop(stale)
            if action is None:underruns.append(tick)
            else:assert action.shape==(7,) and np.isfinite(action).all()
            if tick%args.request_every==0:
                try:pending.put_nowait((tick,tick//args.request_every))
                except queue.Full:request_drops+=1
            durations.append(time.perf_counter()-start)
    finally:
        stop.set();thread.join(timeout=20)
    if thread.is_alive():raise RuntimeError('Inference worker failed to stop')
    if worker_errors:raise RuntimeError(worker_errors)
    hz=args.hz;chunk=policy.config.chunk_size
    report={'scope':'original checkpoint GPU + recorded LIBERO observations + timed shadow consumer; no cameras, serial bus or robot connected',
        'checkpoint_sha256':hashlib.sha256((cfg.model_dir/'model.safetensors').read_bytes()).hexdigest(),
        'hardware':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'observation_cache_sha256':hashlib.sha256((ROOT/'runs/haq_offline_local_v1/fp_cache40/observations.npz').read_bytes()).hexdigest(),
        'weight_dtypes':dict(Counter(str(x.dtype) for x in policy.parameters())),
        'use_amp':policy.config.use_amp,'num_steps':policy.config.num_steps,'chunk_size':chunk,'output_action_dim':7,
        'observation_shapes':{k:list(v.shape[1:]) for k,v in observations.items() if k!='task'},
        'load_seconds':load_seconds,'warmup':args.warmup,'sequential_chunk_seconds':serial,
        'hz':hz,'tick_budget_seconds':1/hz,'chunk_horizon_seconds':chunk/hz,
        'action_supply_hz_at_p95':chunk/serial['p95'],
        'minimum_lookahead_actions_at_p95':int(np.ceil(serial['p95']*hz)),
        'peak_allocated_cuda_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_cuda_bytes':torch.cuda.max_memory_reserved(),
        'shadow_async':{'requested_seconds':args.seconds,'ticks':ticks,'request_every_ticks':args.request_every,
            'completed_inferences':len(worker_times),'chunk_seconds':stats(worker_times),'underruns':underruns,
            'dropped_requests':request_drops,'late_ticks_over_one_period':sum(x>1/hz for x in lateness),
            'tick_start_lateness_seconds':stats(lateness),'tick_body_seconds':stats(durations),
            'tick_interval_seconds':stats(np.diff(actual)),'requests':requests,
            'first_chunk_ready_before_clock':True,'initial_chunk_seconds':initial_seconds,
            'algorithm':'latest timestamp-aligned chunk overwrites future actions; skip elapsed chunk prefix; no RTC/prefix conditioning or continuity guarantee'},
        'limitations':['LIBERO checkpoint actions are not calibrated SO101 joint targets; needs embodiment/task fine-tuning',
            'No image acquisition, USB/serial motor writes, network latency or physical motion measured',
            'Recorded frames are not a closed-loop SO101 task; action continuity and task quality untested',
            'No FP16/autocast/compile or reduced denoising steps enabled; use original inference path',
            'Host OS and desktop GPU load not isolated; shadow timing is not hard-real-time certification']}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(args.output/'timings.npz',chunk_seconds=times,tick_lateness=lateness,tick_body=durations,tick_starts=actual)
    print(json.dumps({k:v for k,v in report.items() if k!='shadow_async'}),flush=True)
    print(json.dumps({'stage':'shadow_done','ticks':ticks,'underruns':len(underruns),'completed_inferences':len(worker_times),'worker_p95':report['shadow_async']['chunk_seconds'].get('p95'),'output':str(args.output)}),flush=True)

if __name__=='__main__':main()
