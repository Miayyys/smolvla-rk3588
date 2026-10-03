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
import numpy as np
from rknnlite.api import RKNNLite

class RKNNPartitions:
    def __init__(self,root,manifest,digest):
        self.input_names=manifest['input_names'];self.output_names=manifest['output_names']
        self.models=[];self.hashes={}
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
                if m.load_rknn(str(root/name)) or m.init_runtime(core_mask=RKNNLite.NPU_CORE_0):raise RuntimeError('Partition init failed: '+name)
        except Exception:self.close();raise

    def infer(self,inputs):
        if len(inputs)!=len(self.input_names):raise ValueError('Partition input count mismatch')
        values=dict(zip(self.input_names,inputs))
        for model,part in self.models:
            arrays=[]
            for n in part['input_names']:
                x=values[n];dtype=part['input_dtypes'][n]
                if dtype=='BOOL':
                    if not np.isin(x,[0,1]).all():raise ValueError('Nonbinary mask at '+n)
                    x=x.astype(np.bool_)
                elif dtype=='INT64':
                    if x.dtype!=np.int64:raise ValueError('Index dtype changed at '+n)
                elif dtype!='FLOAT':raise ValueError('Unsupported partition input type '+dtype)
                arrays.append(np.ascontiguousarray(x))
            if os.environ.get('QVLA_TRACE_PARTITIONS'):
                print(part['filename'], [(n,str(x.dtype),x.shape) for n,x in zip(part['input_names'],arrays)],file=sys.stderr,flush=True)
            result=model.inference(inputs=arrays,data_format=['nchw']*len(part['input_names']))
            if result is None or len(result)!=len(part['output_names']):raise RuntimeError('Partition output count mismatch')
            if any(not np.isfinite(x).all() for x in result):raise RuntimeError('Nonfinite partition result')
            values.update(zip(part['output_names'],result))
        return [values[n] for n in self.output_names]

    def close(self):
        for model,_ in reversed(self.models):model.release()
        self.models=[]
