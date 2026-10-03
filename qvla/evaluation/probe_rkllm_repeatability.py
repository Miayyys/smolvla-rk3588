"""Board-only fixed-input language repetition; excludes RKNN frontend and expert."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import argparse,os,subprocess,json,struct,hashlib,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True);p.add_argument('--tokens',type=int,required=True)
p.add_argument('--runs',type=int,default=3);p.add_argument('--output',type=Path,required=True)
p.add_argument('--perturb',action='store_true',help='A-B-A, B from recorded changed frontend boundary')
p.add_argument('--stock',action='store_true');a=p.parse_args();r=a.root.resolve();a.output.mkdir(parents=True,exist_ok=True)
m=json.loads((r/'deployment_manifest.json').read_text())['language_backend']
def file_hash(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
assets={name:file_hash(r/name) for name in (m['model'],'rkllm_worker','patched_lib/librkllmrt.so')}
z=np.load(r/'native_live_debug.npz');prefix=z['prefixes'][0,0]
# The first test has 150 valid prefix tokens and state at original slot 176.
assert 2<=a.tokens<=160
x=np.zeros((a.tokens,960),dtype='f4');n=min(a.tokens,151)
x[:n-1]=prefix[:n-1];x[n-1]=prefix[176]
x.tofile(a.output/'embeds.bin')
env=dict(os.environ,LD_LIBRARY_PATH=str(r/'patched_lib'),RKLLM_DUMP_LEVEL='0',QVLA_RKLLM_TOKENS=str(a.tokens))
if a.stock:env['QVLA_RKLLM_STOCK']='1'
proc=subprocess.Popen([str(r/'rkllm_worker'),str(r/m['model']),str(a.output.resolve()/'embeds.bin'),str(a.output.resolve()/'cache.bin'),str(m.get('cpu_threads',1))],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
log=(a.output/'worker.log').open('w')
def read(expected):
 for line in proc.stdout:
  log.write(line);log.flush()
  if 'E RKNN' in line or 'E rkllm' in line:raise RuntimeError(line.strip())
  if line.startswith('RKLLM_REPLY '):
   assert line.strip()=='RKLLM_REPLY '+expected,line
   return
 raise RuntimeError('worker exited')
def cache():
 b=(a.output/'cache.bin').read_bytes();t=a.tokens;size=t*5*64*2;arrays=[]
 assert struct.unpack('<III',b[:12])==(0,8,t)
 for j in range(32):
  off=len(b)-16-(32-j)*(size+12)+12
  assert struct.unpack('<III',b[off-12:off])==((1,640,0) if j<16 else (1,2,320))
  arrays.append(np.frombuffer(b,'<f2',t*5*64,off).astype('f4'))
 return np.stack(arrays)
values=[];times=[]
try:
 read('ready')
 for i in range(a.runs):
  if a.perturb:
   current=x.copy()
   if i==1:
    current[:n-1]=z['prefixes'][1,0,:n-1];current[n-1]=z['prefixes'][1,0,176]
   current.tofile(a.output/'embeds.bin')
  start=time.perf_counter();proc.stdin.write(str(n)+'\n');proc.stdin.flush();read('done '+str(a.tokens));times.append((time.perf_counter()-start)*1000);values.append(cache())
 rows=[]
 for i in range(1,len(values)):
  d=np.abs(values[i]-values[0]);rows.append({'run':i,'mae':float(d.mean()),'max_abs':float(d.max()),'layers':[{'layer':j,'key_max':float(d[j].max()),'value_max':float(d[16+j].max())} for j in range(16)]})
 report={'scope':'isolated native language exact repeated inputs; no expert/frontend','tokens':a.tokens,'actual_tokens':n,'stock':a.stock,'perturb':a.perturb,'assets':assets,'manifest_assets_match':all(m['assets'].get(k)==v for k,v in assets.items()),'cpu_threads':m.get('cpu_threads',1),'diagnostic_environment':{k:v for k,v in env.items() if k.startswith('QVLA_RKLLM_') or k in ('OMP_NUM_THREADS','OMP_DYNAMIC','RKLLM_DUMP_LEVEL')},'input_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'comparisons':rows}
 report['diagnostic_environment'].update({k:v for k,v in env.items() if k.startswith('QVLA_RKNPU_') or k=='LD_PRELOAD'})
 if env.get('LD_PRELOAD'):report['preload_sha256']=file_hash(Path(env['LD_PRELOAD']))
 report['run_and_cache_ms']=times
 report['timing_scope']='run plus prompt-cache save and worker stdio; excludes initialization and Python cache parsing; first call excluded from percentiles'
 if len(times)>1:report['warm_latency_ms']={f'p{p}':float(np.percentile(times[1:],p)) for p in (50,90,95)}
 np.savez_compressed(a.output/'repeated_cache.npz',values=np.stack(values));(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
finally:
 if proc.poll() is None:
  proc.stdin.close()
  try:proc.wait(timeout=10)
  except subprocess.TimeoutExpired:proc.kill();proc.wait()
 log.close()
