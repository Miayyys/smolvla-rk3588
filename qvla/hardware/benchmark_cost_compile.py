"""Compile deduplicated Linear cost cases using real checkpoint weights, synthetic inputs.
Synthetic calibration is for hardware costing ONLY, never policy quality selection.
"""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import json,struct,hashlib,sys,subprocess,os
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto

def worker(case):
 from rknn.api import RKNN
 root=Path('runs/hardware_cost_v1').resolve();dst=root/case['case_id'];dst.mkdir(parents=True,exist_ok=True)
 src=Path('artifacts/transfer/model/model.safetensors')
 with src.open('rb') as f:
  length=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(length));base=8+length
  def tensor(key):
   t=h[key];f.seek(base+t['data_offsets'][0]);buf=f.read(t['data_offsets'][1]-t['data_offsets'][0]);a=np.frombuffer(buf,dtype='<u2' if t['dtype']=='BF16' else '<f4')
   if t['dtype']=='BF16':a=(a.astype(np.uint32)<<16).view(np.float32)
   return a.reshape(t['shape']).copy()
  name=case['modules'][0];w=tensor(name+'.weight');b=tensor(name+'.bias') if case['has_bias'] else None
 rng=np.random.default_rng(20260929)
 shape=case['input_shape'];x=rng.standard_normal(shape).astype(np.float32)
 np.save(dst/'input.npy',x)
 cal=[]
 for i in range(2):
  p=dst/f'cal{i}.npy';np.save(p,rng.standard_normal(shape).astype(np.float32));cal.append(str(p))
 (dst/'dataset.txt').write_text('\n'.join(cal)+'\n')
 if case['kind']=='linear':
  nodes=[helper.make_node('MatMul',['input','weight'],['mm' if b is not None else 'output'])];weights=[numpy_helper.from_array(w.T.copy(),'weight')]
  outshape=shape[:-1]+[w.shape[0]]
  if b is not None:nodes.append(helper.make_node('Add',['mm','bias'],['output']));weights.append(numpy_helper.from_array(b,'bias'))
 elif case['kind']=='conv2d':
  nodes=[helper.make_node('Conv',['input','weight']+(['bias'] if b is not None else []),['output'],kernel_shape=[16,16],strides=[16,16])]
  weights=[numpy_helper.from_array(w,'weight')]
  if b is not None:weights.append(numpy_helper.from_array(b,'bias'))
  outshape=[shape[0],w.shape[0],shape[2]//16,shape[3]//16]
 else:raise ValueError(f"Unsupported cost case kind: {case['kind']}")
 graph=helper.make_graph(nodes,f'{case["kind"]}_cost',[helper.make_tensor_value_info('input',TensorProto.FLOAT,shape)],[helper.make_tensor_value_info('output',TensorProto.FLOAT,outshape)],weights)
 m=helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)],ir_version=8);onnx.checker.check_model(m);onnx.save(m,dst/'model.onnx')
 if case['kind']=='linear':
  reference=x@w.T+(b if b is not None else 0)
 else:
  import onnxruntime as ort
  ref_session=ort.InferenceSession(str(dst/'model.onnx'),providers=['CPUExecutionProvider'])
  reference=ref_session.run(None,{'input':x})[0]
 np.save(dst/'reference.npy',reference)
 r=RKNN(verbose=False);record={'case':case,'representative':name,'input_scope':'synthetic_normal_cost_only','seed':20260929,'status':'failed'}
 try:
  quant=case['format'].startswith('w');field='quantized_dtype' if quant else 'float_dtype'
  for stage,fn in [('config',lambda:r.config(target_platform='rk3588',**{field:case['format']})),('load',lambda:r.load_onnx(model=str(dst/'model.onnx'))),('build',lambda:r.build(do_quantization=quant,dataset=str(dst/'dataset.txt') if quant else None)),('export',lambda:r.export_rknn(str(dst/'model.rknn')))]:
   record['stage']=stage;ret=fn()
   if ret!=0:raise RuntimeError(f'{stage} returned {ret}')
  record['status']='compiled'
 except Exception as e:record['error']=str(e)
 finally:
  r.release();record['hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dst.iterdir() if p.suffix in ('.onnx','.npy','.rknn')};(dst/'compile.json').write_text(json.dumps(record,indent=2))
if __name__=='__main__':
 cases=json.load(open('config/hardware/tables.json'))['pending_cases']
 if len(sys.argv)>1:worker(next(c for c in cases if c['case_id']==sys.argv[1]))
 else:
  from concurrent.futures import ThreadPoolExecutor
  def run(c):
   d=Path('runs/hardware_cost_v1')/c['case_id'];d.mkdir(parents=True,exist_ok=True)
   if (d/'compile.json').exists():return
   with (d/'compile.log').open('w') as log:
    try:ret=subprocess.run([sys.executable,__file__,c['case_id']],stdout=log,stderr=subprocess.STDOUT,timeout=240,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')).returncode
    except subprocess.TimeoutExpired:ret=124
   if not (d/'compile.json').exists():(d/'compile.json').write_text(json.dumps({'status':'failed','returncode':ret,'case':c}))
   print(c['case_id'],json.load(open(d/'compile.json'))['status'],flush=True)
  with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,cases))
