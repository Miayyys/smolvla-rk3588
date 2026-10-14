"""Build small QK/attention controls for RK3588 multicore numerical diagnosis."""
import argparse
from pathlib import Path
import numpy as np,onnx,json
from onnx import helper as h,TensorProto as T
from rknn.api import RKNN
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--heads',type=int,default=5)
parser.add_argument('--output',type=Path,default=Path('runs/expert_multicore_diagnostic_v1/attention_control'))
parser.add_argument('--precisions',nargs='+',choices=['fp16','int8'],default=['fp16','int8'])
args=parser.parse_args()
out=args.output;out.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(7);q=rng.standard_normal((1,args.heads,50,64)).astype('float32');k=rng.standard_normal((1,args.heads,227,64)).astype('float32');v=rng.standard_normal((1,args.heads,227,64)).astype('float32')
np.savez(out/'inputs.npz',input_0=q,input_1=k,input_2=v)
np.savez(out/'qk_inputs.npz',input_0=q,input_1=k)
for i in range(4):
 for j,x in enumerate([q,k,v]):np.save(out/f'sample{i}_{j}.npy',x if i==0 else rng.standard_normal(x.shape).astype('float32'))
for kind in ['qk','attention']:
 inputs=[h.make_tensor_value_info('q',T.FLOAT,list(q.shape)),h.make_tensor_value_info('k',T.FLOAT,list(k.shape))]
 nodes=[h.make_node('Transpose',['k'],['kt'],perm=[0,1,3,2]),h.make_node('MatMul',['q','kt'],['logits'])]
 initial=[]
 if kind=='attention':
  inputs.append(h.make_tensor_value_info('v',T.FLOAT,list(v.shape)))
  initial=[onnx.numpy_helper.from_array(np.array(.125,dtype='float32'),'scale')]
  nodes.extend([h.make_node('Mul',['logits','scale'],['scaled']),h.make_node('Softmax',['scaled'],['prob'],axis=-1),h.make_node('MatMul',['prob','v'],['out'])]);shape=[1,args.heads,50,64];output='out'
 else:shape=[1,args.heads,50,227];output='logits'
 model=h.make_model(h.make_graph(nodes,kind,inputs,[h.make_tensor_value_info(output,T.FLOAT,shape)],initializer=initial),opset_imports=[h.make_opsetid('',18)]);model.ir_version=9
 path=out/(kind+'.onnx');onnx.checker.check_model(model);onnx.save(model,path)
 dataset=out/(kind+'_dataset.txt');dataset.write_text('\n'.join(' '.join(str((out/f'sample{i}_{j}.npy').resolve()) for j in range(len(inputs))) for i in range(4))+'\n')
 for precision in args.precisions:
  name=kind+'_'+precision;rk=RKNN(verbose=False)
  try:
   assert rk.config(target_platform='rk3588',float_dtype='float16',quantized_dtype='w8a8',quantized_method='channel',optimization_level=3)==0
   assert rk.load_onnx(model=str(path))==0
   assert rk.build(do_quantization=precision=='int8',dataset=str(dataset) if precision=='int8' else None)==0
   assert rk.export_rknn(str(out/(name+'.rknn')))==0
   print(json.dumps({'model':name,'status':'built'}),flush=True)
  finally:rk.release()
