"""A-B-A live frontend test to detect stale native language cache."""
import argparse,json
from pathlib import Path
import numpy as np
from smolvla_board_runtime import BoardSmolVLA
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
p.add_argument('--task',help='Diagnostic validation of another instruction length; does not change deployment manifest')
a=p.parse_args()
raw=dict(np.load(a.root/'raw_inputs.npz'));model=BoardSmolVLA(a.root)
try:
 if a.task:
  raw['task']=np.array(a.task)
  _,text_mask=model.pre.tokenize([a.task])
  actual_tokens=129+int(text_mask.sum())
  if not 2<=actual_tokens<=160:raise ValueError('Instruction exceeds compiled prefill')
  model.language.allowed=[actual_tokens]
 changed={k:v.copy() for k,v in raw.items()}
 changed['image1']=np.zeros_like(changed['image1'])
 changed['state']=changed['state']+np.float32(0.1)
 values=[];timings=[];prefixes=[];caches=[]
 original=model.language.infer
 def capture(prefix,pad):
  out=original(prefix,pad);prefixes.append(prefix.copy());caches.append([v.copy() for v in out]);return out
 model.language.infer=capture
 for x in (raw,changed,raw):
  y,t=model.predict(x);values.append(y);timings.append(t);print(json.dumps(t),flush=True)
 repeat=float(np.max(np.abs(values[0]-values[2])))
 prefix_repeat=float(np.max(np.abs(prefixes[0]-prefixes[2])))
 kv_repeat=float(np.max(np.abs(np.stack(caches[0])-np.stack(caches[2]))))
 difference=float(np.mean(np.abs(values[0]-values[1])))
 np.savez_compressed(a.root/'native_live_debug.npz',actions=np.stack(values),prefixes=np.stack(prefixes),caches=np.stack(caches))
 print(json.dumps({'prefix_repeat':float(np.max(np.abs(prefixes[0]-prefixes[2]))),'kv_repeat':float(np.max(np.abs(np.stack(caches[0])-np.stack(caches[2]))))}),flush=True)
 np.savez_compressed(a.root/'native_live_replay.npz',actions=values[0])
 report={'scope':'live RKNN vision plus persistent native RKLLM plus RKNN expert A-B-A',
  'task':str(raw['task'].item()),'diagnostic_repeat_tolerance':1e-5,'status':'passed' if max(repeat,prefix_repeat,kv_repeat)<1e-5 and difference>1e-5 else 'failed',
  'prefix_repeat_max_abs':float(np.max(np.abs(prefixes[0]-prefixes[2]))),
  'kv_repeat_max_abs':float(np.max(np.abs(np.stack(caches[0])-np.stack(caches[2])))),
  'repeat_max_abs':repeat,'changed_input_action_mae':difference,'timings':timings,'closed_loop_tested':False}
 (a.root/'native_live_replay.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
 assert report['status']=='passed',report
finally:model.close()
