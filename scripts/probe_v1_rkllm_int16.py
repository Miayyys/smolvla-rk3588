"""Test V1's INT16 DFP request against the installed RKLLM SDK on a tiny fixture."""
import argparse,hashlib,json,time
from pathlib import Path
from rkllm.api import RKLLM
from rkllm.llmFbs.MatMulType import MatMulType
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--fixture',type=Path,default=Path('runs/rkllm_dump_probe_v1'))
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
report={'scope':'SDK format feasibility on tiny fixture; not selected V1 conversion',
 'matmul_type_enum':{k:v for k,v in vars(MatMulType).items() if not k.startswith('_')},'tests':[]}
for dtype in ['w16a16i_dfp','w16a16i']:
 m=RKLLM();row={'quantized_dtype':dtype};start=time.monotonic()
 try:
  row['stage']='load';rc=m.load_huggingface(str(a.fixture/'hf_model'),device='cpu',dtype='float32')
  row['load_return']=rc
  if rc:raise RuntimeError(f'load: {rc}')
  row['stage']='build';rc=m.build(do_quantization=True,quantized_dtype=dtype,target_platform='rk3588',num_npu_core=1,max_context=128,dataset=str(a.fixture/'dataset.json'))
  row['build_return']=rc;row['status']='accepted' if rc==0 else 'rejected'
 except (Exception,SystemExit) as e:row.update(status='rejected',error=repr(e))
 finally:
  row['elapsed_seconds']=time.monotonic()-start;report['tests'].append(row)
  (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)
  del m
