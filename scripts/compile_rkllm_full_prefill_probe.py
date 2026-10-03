"""Compile only observed missing full-prefill NPU shapes for a fixed diagnostic input."""
import argparse
import hashlib,json,re,time
from pathlib import Path
from rkllm.api import RKLLM
from rkllm.base.opfbs import get_op_cmd
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,default=Path('runs/rkllm_native_patch_v1'))
p.add_argument('--language-root',type=Path,default=Path('runs/selected_language_rkllm_v1'))
p.add_argument('--aligned',action='store_true',help='Add fully aligned 160-token diagnostic shapes')
p.add_argument('--diagnostic-lengths',type=int,nargs='+',default=[160])
p.add_argument('--prepend-shapes',action='store_true',help='Keep original final tensor layout when adding command variants')
p.add_argument('--output',type=Path)
p.add_argument('--cores',type=int,default=3,choices=[1,3])
p.add_argument('--quantized-dtype',choices=['fp16','w8a8','w8a8_g128'],default='fp16')
p.add_argument('--dataset',type=Path)
a=p.parse_args();root=a.root;required={}
if a.quantized_dtype!='fp16' and a.dataset is None:raise ValueError('Explicit isolated calibration required')
logs=[root/'real_compact/board.log',root/'real_compact/board_shapes_fixed.log']
text='\n'.join(x.read_text() for x in logs if x.exists())
for name,shape in re.findall(r'op name: ([^,]+), shape: ([^\n]+)',text):
 value=[int(x.strip()) for x in shape.split(',')]
 if value not in required.setdefault(name,[]):required[name].append(value)
for name,values in required.items():
 if a.aligned and all(v[0]==152 for v in values):
  for v in list(values):
   extra=[160,*v[1:]]
   if extra not in values:values.append(extra)
old=get_op_cmd.add_matmul_info;added=[]
def hook(self,opinfo,dims=[[1,64,32],[4,64,32]]):
 dims=[list(x) for x in dims]
 extra=required.get(opinfo.name,[])
 if a.cores==1:
  if '.weight' in opinfo.name:extra=[[(t+7)//8*8,*dims[0][1:]] for t in a.diagnostic_lengths]
  elif 'matmul_qkv' in opinfo.name or 'attn_kqv' in opinfo.name:extra=[[m,(t+31)//32*32,64] for t in a.diagnostic_lengths for m in (((t+31)//32*32),2*((t+31)//32*32),3*((t+31)//32*32))]
  elif 'matmul_qk' in opinfo.name or 'attn_kq' in opinfo.name:extra=[[m,64,(t+31)//32*32] for t in a.diagnostic_lengths for m in (((t+31)//32*32),2*((t+31)//32*32),3*((t+31)//32*32))]
 for value in extra:
  if value not in dims:
   (dims.insert(0,value) if a.prepend_shapes else dims.append(value));added.append({'name':opinfo.name,'shape':value})
 return old(self,opinfo,dims)
get_op_cmd.add_matmul_info=hook
m=RKLLM();report={'scope':'full prefill diagnostic NPU shapes; not a general length support claim','num_npu_core':a.cores,'aligned_160':a.aligned or a.cores==1,'prepend_shapes':a.prepend_shapes,'added':added,'quantized_dtype':a.quantized_dtype,'optimization_level':1,'quantized_algorithm':'normal','dataset_sha256':hashlib.sha256(a.dataset.read_bytes()).hexdigest() if a.dataset else None}
report['source_extraction']=json.loads((a.language_root/'hf_model/extraction.json').read_text())
report['source_language_weight_sha256']=hashlib.sha256((a.language_root/'hf_model/model.safetensors').read_bytes()).hexdigest()
started=time.monotonic()
try:
 assert m.load_huggingface(str(a.language_root/'hf_model'),device='cpu',dtype='float32')==0
 assert m.build(do_quantization=a.quantized_dtype!='fp16',quantized_dtype='w8a8' if a.quantized_dtype=='fp16' else a.quantized_dtype,dataset=str(a.dataset.resolve()) if a.dataset else None,target_platform='rk3588',num_npu_core=a.cores,max_context=256,optimization_level=1)==0
 assert m.export_rkllm(str(a.output or root/'language_full_prefill_fp16.rkllm'),export_embedding=False)==0
 if a.cores==3:assert set(x['name'] for x in added)==set(required),set(required)-set(x['name'] for x in added)
 report['status']='compiled'
 out=a.output or root/'language_full_prefill_fp16.rkllm'
 report.update(bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest())
except BaseException as e:
 report.update(status='failed',error=repr(e));raise
finally:
 report['elapsed_seconds']=time.monotonic()-started
 ((a.output.with_suffix('.compile.json')) if a.output else root/'full_prefill_compile_report.json').write_text(json.dumps(report,indent=2)+'\n')
