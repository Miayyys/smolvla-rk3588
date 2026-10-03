"""Fixed native RKLLM K/V plus selected RKNN expert; cached frontend, not full policy."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import argparse
import json
import time
from pathlib import Path
import numpy as np
from rknnlite.api import RKNNLite
from qvla.runtime.smolvla_numpy_glue import time_embedding, postprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input',type=Path,required=True)
p.add_argument('--model',type=Path,required=True)
p.add_argument('--weights',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
z=np.load(a.input);weights=dict(np.load(a.weights));m=RKNNLite()
try:
    assert m.load_rknn(str(a.model))==0
    assert m.init_runtime(core_mask=RKNNLite.NPU_CORE_0)==0
    action=z['noise'].copy();durations=[]
    for step in range(10):
        emb=time_embedding(np.array([1-step/10],dtype='f4'),720,0.004,4.0)
        inputs=[action,emb,z['pad']]+[z[f'{k}_{i}'] for i in range(16) for k in ('key','value')]
        start=time.perf_counter()
        out=m.inference(inputs=[np.ascontiguousarray(x) for x in inputs],data_format=['nchw']*len(inputs))
        durations.append((time.perf_counter()-start)*1000)
        assert out is not None and len(out)==1 and np.isfinite(out[0]).all()
        action=action-np.float32(0.1)*out[0]
    final=postprocess(action,weights);ref=postprocess(z['reference'],weights)
    diff=final.astype('f8')-ref.astype('f8')
    report=dict(scope='one cached frontend, native RKLLM K/V plus actual RKNN expert ten steps',
                action_mae=float(np.abs(diff).mean()),action_rmse=float(np.sqrt((diff*diff).mean())),
                max_abs=float(np.abs(diff).max()),
                gripper_sign_disagreements=int(((final[...,6]>0)!=(ref[...,6]>0)).sum()),
                expert_step_ms=durations, closed_loop_tested=False)
    np.savez_compressed(a.output.with_suffix('.npz'),actions=final,reference=ref)
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
finally:
    m.release()
