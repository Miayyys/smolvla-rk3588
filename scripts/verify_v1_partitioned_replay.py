"""Real complete board replay against the selected V1 GPU pack, not the original FP."""
import argparse,hashlib,json,resource
from pathlib import Path
import numpy as np
from smolvla_board_runtime import BoardSmolVLA
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
p.add_argument('--warmup',type=int,default=1);p.add_argument('--repeats',type=int,default=3)
a=p.parse_args()
if a.repeats<2 or a.warmup<0:raise ValueError('At least two measured repeats required')
with np.load(a.root/'raw_inputs.npz',allow_pickle=False) as z:raw={n:z[n].copy() for n in z.files}
with np.load(a.root/'fp_reference.npz',allow_pickle=False) as z:ref=z['actions'].copy()
runtime=BoardSmolVLA(a.root);values=[];times=[]
try:
 for i in range(a.warmup+a.repeats):
  result,timing=runtime.predict(raw)
  if i>=a.warmup:values.append(result);times.append(timing)
  print(json.dumps({'iteration':i,'warmup':i<a.warmup,**timing}),flush=True)
 d=np.abs(values[0].astype('f8')-ref.astype('f8'))
 repeat=max(float(np.max(np.abs(x-values[0]))) for x in values[1:])
 report={'scope':'complete raw-input V1 deployment replay compared to selected GPU pack',
 'language_backend':'rkllm' if runtime.language else 'rknn',
 'reference_sha256':hashlib.sha256((a.root/'fp_reference.npz').read_bytes()).hexdigest(),
 'raw_input_sha256':hashlib.sha256((a.root/'raw_inputs.npz').read_bytes()).hexdigest(),
 'manifest_sha256':hashlib.sha256((a.root/'deployment_manifest.json').read_bytes()).hexdigest(),
 'loaded_model_hashes':runtime.hashes,'action_mae':float(d.mean()),'max_abs':float(d.max()),
 'gripper_sign_disagreements':int(((values[0][...,6]>0)!=(ref[...,6]>0)).sum()),
 'repeat_max_abs':repeat,'inference_ms':[x['inference_ms'] for x in times],
 'p50_ms':float(np.median([x['inference_ms'] for x in times])),
 'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'closed_loop_tested':False}
 np.savez_compressed(a.root/'partitioned_replay.npz',actions=np.stack(values),reference=ref)
 (a.root/'partitioned_replay.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
finally:runtime.close()
