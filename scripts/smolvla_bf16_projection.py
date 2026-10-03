"""Strict native BF16 adapter, limited to the validated V1 gate projection."""
import ctypes
from pathlib import Path
import numpy as np

class NativeBF16Projection:
    def __init__(self,path):
        self.lib=ctypes.CDLL(str(Path(__file__).resolve().parent/'librknn_bf16_projection.so'))
        self.lib.qvla_bf16_open.argtypes=[ctypes.c_char_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.POINTER(ctypes.c_int)]
        self.lib.qvla_bf16_open.restype=ctypes.c_void_p
        self.lib.qvla_bf16_close.argtypes=[ctypes.c_void_p]
        self.lib.qvla_bf16_run.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
        self.lib.qvla_bf16_run.restype=ctypes.c_int
        rc=ctypes.c_int()
        self.handle=self.lib.qvla_bf16_open(str(path).encode(),50*720,50*2048,ctypes.byref(rc))
        if not self.handle:raise RuntimeError(f'Native BF16 init failed: {rc.value}')

    def inference(self,inputs,**kwargs):
        if len(inputs)!=1 or inputs[0].shape!=(1,50,720):raise ValueError('Unsupported BF16 projection input')
        x=np.ascontiguousarray(inputs[0],dtype=np.float32)
        if not np.isfinite(x).all():raise ValueError('Nonfinite BF16 input')
        y=np.empty((1,50,2048),dtype=np.float32)
        rc=self.lib.qvla_bf16_run(self.handle,x.ctypes.data,y.ctypes.data)
        if rc:raise RuntimeError(f'Native BF16 execution failed: {rc}')
        if not np.isfinite(y).all():raise RuntimeError('Nonfinite BF16 result')
        return [y]

    def release(self):
        if self.handle:self.lib.qvla_bf16_close(self.handle);self.handle=None
