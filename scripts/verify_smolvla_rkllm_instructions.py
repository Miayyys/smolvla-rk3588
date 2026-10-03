"""Verify additional real instruction lengths at the compiled 160-token shape."""
import argparse,json
from pathlib import Path
import numpy as np
from smolvla_board_runtime import BoardSmolVLA
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--instructions',type=Path,required=True)
a=p.parse_args();raw=dict(np.load(a.root/'raw_inputs.npz'));model=BoardSmolVLA(a.root)
rows=[];values=[];prefixes=[];caches=[]
original=model.language.infer
try:
 instructions=json.loads(a.instructions.read_text())
 def capture(prefix,pad):
  out=original(prefix,pad);prefixes.append(prefix.copy());caches.append(np.stack(out));return out
 model.language.infer=capture
 for label,text in [('reference_before',str(raw['task'].item())),*instructions.items(),('reference_after',str(raw['task'].item()))]:
  _,mask=model.pre.tokenize([text]);n=129+int(mask.sum())
  if not 129<n<=160:raise ValueError('Instruction exceeds compiled shape: '+str(n))
  model.language.allowed=list(set(model.language.allowed+[n]))
  current={k:v.copy() for k,v in raw.items()};current['task']=np.array(text)
  action,timing=model.predict(current);values.append(action)
  rows.append(dict(label=label,instruction=text,valid_tokens=n,**timing));print(json.dumps(rows[-1]),flush=True)
 report={'scope':'instruction-length execution and cross-instruction repeatability, not task quality','rows':rows,
 'action_repeat_max_abs':float(np.max(np.abs(values[0]-values[-1]))),
 'prefix_repeat_max_abs':float(np.max(np.abs(prefixes[0]-prefixes[-1]))),
 'kv_repeat_max_abs':float(np.max(np.abs(caches[0]-caches[-1])))}
 report['passed']=max(report[k] for k in ('action_repeat_max_abs','prefix_repeat_max_abs','kv_repeat_max_abs'))<=1e-5
 (a.root/'instruction_length_validation.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report),flush=True)
 if not report['passed']:raise ValueError('Repeated reference changed across instructions')
finally:model.close()
