"""Execute hash-pinned RKNN partitions with explicit live tensor routing."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

from pathlib import Path
import os
import sys
import time
import numpy as np
from rknnlite.api import RKNNLite

class RKNNPartitions:
    def __init__(self,root,manifest,digest,core_mask=RKNNLite.NPU_CORE_0):
        self.input_names=manifest['input_names'];self.output_names=manifest['output_names']
        self.models=[];self.hashes={};self.static_inputs={};self.static_arrays={}
        try:
            for part in manifest['parts']:
                name=part['filename']
                if Path(name).name!=name:raise ValueError('Invalid partition filename')
                actual=digest(root/name)
                if actual!=part['sha256']:raise ValueError('Partition hash mismatch: '+name)
                if part.get('execution_backend')=='native_bf16_projection':
                    from qvla.runtime.smolvla_bf16_projection import NativeBF16Projection
                    m=NativeBF16Projection(root/name)
                    self.models.append((m,part));self.hashes[name]=actual
                    continue
                m=RKNNLite();self.models.append((m,part));self.hashes[name]=actual
                if m.load_rknn(str(root/name)) or m.init_runtime(core_mask=core_mask):raise RuntimeError('Partition init failed: '+name)
        except Exception:self.close();raise

    @staticmethod
    def prepare_input(x,dtype,name):
        if dtype=='BOOL':
            if x.dtype!=np.bool_:
                if not np.isin(x,[0,1]).all():raise ValueError('Nonbinary mask at '+name)
                x=x.astype(np.bool_)
        elif dtype=='INT64':
            if x.dtype!=np.int64:raise ValueError('Index dtype changed at '+name)
        elif dtype!='FLOAT':raise ValueError('Unsupported partition input type '+dtype)
        return np.ascontiguousarray(x)

    def bind_static(self,inputs):
        """Inputs must remain immutable until the ten-step chunk finishes."""
        self.static_inputs=dict(inputs);self.static_arrays={}
        for _,part in self.models:
            for name in part['input_names']:
                if name in inputs:
                    key=(name,part['input_dtypes'][name])
                    if key not in self.static_arrays:
                        self.static_arrays[key]=self.prepare_input(inputs[name],key[1],name)

    def infer(self,inputs):
        if len(inputs)!=len(self.input_names):raise ValueError('Partition input count mismatch')
        values=dict(zip(self.input_names,inputs));self.last_timings=[]
        for model,part in self.models:
            started=time.perf_counter();arrays=[]
            for n in part['input_names']:
                x=values[n];dtype=part['input_dtypes'][n]
                if n in self.static_inputs and x is self.static_inputs[n]:
                    arrays.append(self.static_arrays[(n,dtype)])
                else:arrays.append(self.prepare_input(x,dtype,n))
            if os.environ.get('QVLA_TRACE_PARTITIONS'):
                print(part['filename'], [(n,str(x.dtype),x.shape) for n,x in zip(part['input_names'],arrays)],file=sys.stderr,flush=True)
            prepared=time.perf_counter()
            result=model.inference(inputs=arrays,data_format=['nchw']*len(part['input_names']))
            if result is None or len(result)!=len(part['output_names']):raise RuntimeError('Partition output count mismatch')
            if any(not np.isfinite(x).all() for x in result):raise RuntimeError('Nonfinite partition result')
            self.last_timings.append({'name':part['filename'],'prepare_ms':(prepared-started)*1000,
                'run_validate_ms':(time.perf_counter()-prepared)*1000})
            values.update(zip(part['output_names'],result))
        return [values[n] for n in self.output_names]

    def close(self):
        for model,_ in reversed(self.models):model.release()
        self.models=[]
