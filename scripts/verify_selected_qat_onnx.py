#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import onnxruntime as ort
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
rows=[];options=ort.SessionOptions();options.intra_op_num_threads=4
for kind in ('vision','prefix','expert'):
 model=ort.InferenceSession(str(a.root/(kind+'.onnx')),sess_options=options,providers=['CPUExecutionProvider'])
 with np.load(a.root/(kind+'_boundary.npz')) as z:feed={v.name:z[v.name] for v in model.get_inputs()}
 out=model.run(None,feed)
 with np.load(a.root/(kind+'_reference.npz')) as z:
  for info,v in zip(model.get_outputs(),out):
   ref=z[info.name];d=v.astype(np.float64)-ref.astype(np.float64)
   rows.append(dict(graph=kind,output=info.name,mae=float(np.abs(d).mean()),max_abs=float(np.abs(d).max()),passed=bool(np.allclose(v,ref,rtol=1e-4,atol=1e-4))))
 del model
report=dict(scope='exported_QAT_FP_master_ONNX_vs_Torch_FP32_not_local_integer_pack',passed=all(x['passed'] for x in rows),rows=rows)
(a.root/'onnx_parity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if not report['passed']:raise SystemExit(1)
