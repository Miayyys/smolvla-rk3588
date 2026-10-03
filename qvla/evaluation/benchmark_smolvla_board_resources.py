"""Complete policy benchmark with sampled parent/worker PSS, RSS and board telemetry."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import argparse,json,os,re,threading,time
from pathlib import Path
import numpy as np
from qvla.runtime.smolvla_board_runtime import BoardSmolVLA

def read(path):
    try:return Path(path).read_text()
    except OSError:return ''

def tree(pid):
    # /proc/PID/task/TID/children needs CONFIG_CHECKPOINT_RESTORE,
    # absent on this board. Scan PPid instead; include all descendants.
    parents={}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        status=read(path/'status')
        match=re.search(r'^PPid:\s+(\d+)',status,re.M)
        if match:parents[int(path.name)]=int(match[1])
    found=[pid]
    for parent in found:
        for child,ppid in parents.items():
            if ppid==parent and child not in found:found.append(child)
    return found

def sample(pid):
    processes=[]
    for p in tree(pid):
        status=read(f'/proc/{p}/status');smaps=read(f'/proc/{p}/smaps_rollup')
        def val(text,name):
            m=re.search(r'^'+name+r':\s+(\d+)',text,re.M);return int(m[1]) if m else None
        stat=read(f'/proc/{p}/stat');ticks=None
        if stat:
            fields=stat[stat.rfind(')')+2:].split();ticks=int(fields[11])+int(fields[12])
        processes.append({'pid':p,'rss_kib':val(status,'VmRSS'),'pss_kib':val(smaps,'Pss'),'swap_kib':val(smaps,'Swap'),'cpu_ticks':ticks})
    loads=[int(x) for x in re.findall(r'Core\d+:\s*(\d+)%',read('/sys/kernel/debug/rknpu/load'))]
    mem={k:int(v) for k,v in re.findall(r'^(MemAvailable|MemFree|Cached|Shmem|SwapFree):\s+(\d+)',read('/proc/meminfo'),re.M)}
    thermals={str(p):read(p).strip() for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')}
    cpu_freq={str(p):read(p).strip() for p in Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_cur_freq')}
    return {'monotonic':time.monotonic(),'processes':processes,'aggregate_rss_kib':sum(x['rss_kib'] or 0 for x in processes),
      'aggregate_pss_kib':sum(x['pss_kib'] or 0 for x in processes),'pss_complete':all(x['pss_kib'] is not None for x in processes),
      'meminfo_kib':mem,'npu_load_percent':loads,'npu_freq_hz':read('/sys/kernel/debug/rknpu/freq').strip(),'thermal_millic':thermals,'cpu_freq_khz':cpu_freq}

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--warmup',type=int,default=3);p.add_argument('--repeats',type=int,default=20)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
if a.repeats<10:raise ValueError('At least 10 measured repetitions')
raw=dict(np.load(a.root/'raw_inputs.npz'));samples=[];stop=threading.Event();phase='initialization'
idle=sample(os.getpid());interval=.5

def monitor():
    previous={};previous_time=None;hz=os.sysconf('SC_CLK_TCK')
    with (a.output/'resource_samples.jsonl').open('w') as f:
        while not stop.is_set():
            row=sample(os.getpid());row['phase']=phase;now=row['monotonic'];delta=0
            for proc in row['processes']:
                ticks=proc['cpu_ticks'];pid=proc['pid']
                if ticks is not None and pid in previous:delta+=max(0,ticks-previous[pid])
                if ticks is not None:previous[pid]=ticks
            row['cpu_percent_one_core_100']=100*delta/hz/(now-previous_time) if previous_time is not None else None
            previous_time=now;samples.append(row);f.write(json.dumps(row)+'\n');f.flush();stop.wait(interval)
thread=threading.Thread(target=monitor,daemon=True);thread.start();model=None;values=[];timings=[];begin=time.monotonic()
try:
    model=BoardSmolVLA(a.root);init_seconds=time.monotonic()-begin
    for i in range(a.warmup+a.repeats):
        phase='warmup' if i<a.warmup else 'measured'
        actions,timing=model.predict(raw)
        if i>=a.warmup:values.append(actions);timings.append(timing)
        print(json.dumps({'iteration':i,'phase':phase,**timing}),flush=True)
    phase='idle_loaded';time.sleep(1)
finally:
    stop.set();thread.join(timeout=5)
    if model:model.close()
if not values:raise RuntimeError('No benchmark results')
if model.language and not any(model.language.process.pid in [p['pid'] for p in row['processes']] for row in samples):raise RuntimeError('RKLLM worker missing from resource samples')
measured=[x for x in samples if x['phase']=='measured'];t=[x['inference_ms'] for x in timings]
cpu=[x['cpu_percent_one_core_100'] for x in measured if x['cpu_percent_one_core_100'] is not None]
loads=[x['npu_load_percent'] for x in measured if len(x['npu_load_percent'])==3]
from qvla.runtime.rknn_board_full_replay import digest
report={'scope':'fixed raw-input complete board policy including preprocessing; monitored parent and all descendants',
 'root':str(a.root),'loaded_graphs':model.hashes,'language_backend':'rkllm' if model.language else 'rknn',
 'raw_input_sha256':digest(a.root/'raw_inputs.npz'),'manifest_sha256':digest(a.root/'deployment_manifest.json') if (a.root/'deployment_manifest.json').exists() else None,
 'warmup':a.warmup,'repeats':a.repeats,'sample_interval_seconds':interval,'initialization_seconds':init_seconds,
 'inference_ms':t,'p50_ms':float(np.median(t)),'p95_ms':float(np.percentile(t,95)),'mean_ms':float(np.mean(t)),
 'preprocess_p50_ms':float(np.median([x['preprocess_ms'] for x in timings])),
 'repeat_max_abs':max(float(np.max(np.abs(x-values[0]))) for x in values),
 'peak_aggregate_pss_kib':max(x['aggregate_pss_kib'] for x in samples),'peak_measured_pss_kib':max(x['aggregate_pss_kib'] for x in measured),
 'peak_aggregate_rss_kib':max(x['aggregate_rss_kib'] for x in samples),'pss_complete':all(x['pss_complete'] for x in samples),
 'process_tree_method':'proc_status_PPid_scan','max_process_count':max(len(x['processes']) for x in samples),'mean_cpu_percent_one_core_100':float(np.mean(cpu)) if cpu else None,
 'mean_npu_core_load_percent':np.mean(loads,axis=0).tolist() if loads else None,'idle_meminfo_kib':idle['meminfo_kib'],
 'minimum_memavailable_kib':min(x['meminfo_kib']['MemAvailable'] for x in samples),'timings':timings,
 'memory_scope':'Sampled process-tree PSS/RSS including RKLLM worker; not all unmapped driver DMA memory. RSS double counts shared pages. Sampling may miss short peaks.',
 'telemetry_scope':'NPU load is sampled global per-core occupancy, not a per-model speed metric; CPU 100% means one full core.'}
(a.output/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed(a.output/'actions.npz',actions=values[0])
print(json.dumps({k:v for k,v in report.items() if k not in ('timings','loaded_graphs','inference_ms')},indent=2),flush=True)
