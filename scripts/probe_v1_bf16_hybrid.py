"""Check whether an RKNN mixed graph really retains one BF16 projection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import yaml
from rknn.api import RKNN

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(29)
weights=rng.standard_normal((128,64)).astype('f4')*.05
nodes=[helper.make_node('MatMul',['input','weight_gate'],['gate'],name='gate'),
       helper.make_node('MatMul',['input','weight_low'],['low'],name='low')]
graph=helper.make_graph(nodes,'mixed_precision_gate_probe',
 [helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,128])],
 [helper.make_tensor_value_info(n,TensorProto.FLOAT,[1,64]) for n in ['gate','low']],
 [numpy_helper.from_array(weights,'weight_gate'),numpy_helper.from_array(weights*1.01,'weight_low')])
model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)]);model.ir_version=9
onnx.checker.check_model(model);onnx.save(model,root/'probe.onnx')
paths=[]
for i in range(4):
 path=root/f'input{i}.npy';np.save(path,rng.standard_normal((1,128)).astype('f4'));paths.append(str(path))
(root/'dataset.txt').write_text('\n'.join(paths)+'\n')
results=[]
for name,global_dtype,custom_dtype in [('explicit_bf16','float16','bfloat16'),('bf16_default_float16_override','bfloat16','float16'),('bf16_param_float32','bfloat16','param_float32'),('bf16_param_bfloat16','bfloat16','param_bfloat16'),('bf16_native_dtype','bfloat16',None)]:
 out=root/name;out.mkdir(exist_ok=True);previous=Path.cwd();os.chdir(out)
 row={'variant':name,'global_float_dtype':global_dtype,'requested_custom_dtype':custom_dtype,'scope':'tiny graph format support, not full expert quality'}
 m=RKNN(verbose=True,verbose_file=str(out/'build.log'));start=time.monotonic()
 try:
  for stage,fn in [('config',lambda:m.config(target_platform='rk3588',float_dtype=global_dtype,quantized_dtype='w8a8',quantized_method='channel')),('load',lambda:m.load_onnx(model=str(root/'probe.onnx')))]:
   row['stage']=stage;rc=fn()
   if rc:raise RuntimeError(f'{stage}: {rc}')
  if custom_dtype:
   row['stage']='step1';rc=m.hybrid_quantization_step1(dataset=str(root/'dataset.txt'),proposal=False)
   if rc:raise RuntimeError(f'step1: {rc}')
   cfg_path=next(out.glob('*.quantization.cfg'));cfg=yaml.safe_load(cfg_path.read_text())
   intermediate=onnx.load(str(next(out.glob('*.model'))))
   gate_nodes=[n for n in intermediate.graph.node if 'weight_gate' in n.input]
   if len(gate_nodes)!=1:raise RuntimeError('Cannot uniquely locate gate')
   q=cfg['quantize_parameters'];targets=[*gate_nodes[0].output,*gate_nodes[0].input]
   if custom_dtype.startswith('param_'):
    custom={}
    for n in targets:
     if n in q:q[n].update(dtype=custom_dtype.removeprefix('param_'),scale=[],zero_point=[])
   else:custom={n:custom_dtype for n in targets if n in q}
   row['kernel_precision_targets']=targets
   cfg['custom_quantize_layers']=custom;chosen=out/'selected.cfg';chosen.write_text(yaml.safe_dump(cfg,sort_keys=False))
   row['custom_quantize_layers']=custom;m.release();m=RKNN(verbose=True,verbose_file=str(out/'step2.log'))
   row['stage']='step2';rc=m.hybrid_quantization_step2(model_input=str(next(out.glob('*.model'))),data_input=str(next(out.glob('*.data'))),model_quantization_cfg=str(chosen))
  else:
   row['stage']='build_unquantized';rc=m.build(do_quantization=False)
  if rc:raise RuntimeError(f"{row['stage']}: {rc}")
  row['stage']='export';rc=m.export_rknn(str(out/'probe.rknn'))
  if rc:raise RuntimeError(f'export: {rc}')
  row['status']='compiled';row['bytes']=(out/'probe.rknn').stat().st_size
  row['rknn_sha256']=hashlib.sha256((out/'probe.rknn').read_bytes()).hexdigest()
 except Exception as e:row.update(status='failed',error=repr(e))
 finally:
  m.release();os.chdir(previous);row['elapsed_seconds']=time.monotonic()-start
  results.append(row);(root/'report.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(row),flush=True)
