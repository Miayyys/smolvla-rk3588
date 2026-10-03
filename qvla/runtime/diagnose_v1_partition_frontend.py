"""Controlled input replacements for attribution, never a deployment fallback."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import argparse,json
from pathlib import Path
import numpy as np
from qvla.runtime.smolvla_board_runtime import BoardSmolVLA

class CaptureVision:
    def __init__(self,model,reference=None):self.model=model;self.reference=reference;self.outputs=[]
    def inference(self,**kwargs):
        result=self.model.inference(**kwargs)
        self.outputs.append(result[0].copy())
        return result if self.reference is None else [self.reference[len(self.outputs)-1].copy()]
    def release(self):self.model.release()

class CapturePrefix:
    def __init__(self,model,reference=None):self.model=model;self.reference=reference;self.input=None
    def infer(self,inputs,*args):
        if isinstance(inputs,np.ndarray):
            self.input=inputs.copy()
            return self.model.infer(inputs if self.reference is None else self.reference.copy(),*args)
        self.input=inputs[0].copy()
        if self.reference is not None:inputs=[self.reference.copy(),*inputs[1:]]
        return self.model.infer(inputs)
    @property
    def models(self):return self.model.models
    def close(self):self.model.close()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    with np.load(a.root/'raw_inputs.npz') as z:raw={n:z[n].copy() for n in z.files}
    with np.load(a.root/'fp_reference.npz') as z:ref={n:z[n].copy() for n in z.files}
    rows=[]
    for label in ['native_v1','GPU_features_as_input','GPU_prefix_as_input']:
        runtime=BoardSmolVLA(a.root)
        vision=CaptureVision(runtime.models[0],ref['features'] if label=='GPU_features_as_input' else None)
        prefix=CapturePrefix(runtime.language or runtime.partition_models['prefix'],ref['prefix'] if label=='GPU_prefix_as_input' else None)
        runtime.models[0]=vision
        if runtime.language:runtime.language=prefix
        else:runtime.partition_models['prefix']=prefix
        try:
            actions,timing=runtime.predict(raw)
            d=np.abs(actions.astype('f8')-ref['actions'].astype('f8'))
            features=np.stack(vision.outputs)
            row={'case':label,'action_mae':float(d.mean()),'max_abs':float(d.max()),
                'gripper_sign_disagreements':int(((actions[...,6]>0)!=(ref['actions'][...,6]>0)).sum()),
                'native_vision_feature_mae':float(np.abs(features-ref['features']).mean()),
                'assembled_prefix_mae':float(np.abs(prefix.input-ref['prefix']).mean()),**timing}
            rows.append(row);print(json.dumps(row),flush=True)
        finally:runtime.close()
    (a.root/'frontend_attribution.json').write_text(json.dumps({'scope':'one fixed raw observation, diagnostic input replacements only; no quantizer or weights changed; not rollout/deployment scores','cases':rows},indent=2)+'\n')

if __name__=='__main__':main()
