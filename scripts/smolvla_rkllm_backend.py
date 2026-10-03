"""Native compact-prefix backend, experimental RKLLM 1.3.1 cache ABI."""
import os
import struct
import subprocess
import tempfile
from pathlib import Path
import numpy as np

class RKLLMLanguage:
    def __init__(self, root, manifest, digest):
        self.root = root.resolve();root = self.root
        self.temp = tempfile.TemporaryDirectory(prefix='qvla_language_', dir='/dev/shm')
        self.path = Path(self.temp.name)
        for name, sha in manifest['assets'].items():
            if digest(root/name) != sha:
                raise ValueError('RKLLM asset changed: '+name)
        env = dict(os.environ, LD_LIBRARY_PATH=str(root/'patched_lib'), RKLLM_DUMP_LEVEL='0')
        control=manifest.get('allocation_control')
        if control:
            if control['mode'] not in ('wc','no_sram'):
                raise ValueError('Unknown allocation control')
            library=control['library']
            if library not in manifest['assets']:
                raise ValueError('Allocation library must have a pinned hash')
            # Apply only to the isolated language worker, not the RKNN frontend.
            env['LD_PRELOAD']=str(root/library)
            env['QVLA_RKNPU_ALLOC_MODE']=control['mode']
            if control.get('pool_bytes'):
                env['QVLA_RKNPU_WC_SIZE']=str(control['pool_bytes'])
        if manifest.get('control_stock'):env['QVLA_RKLLM_STOCK']='1'
        self.process = subprocess.Popen([str(root/'rkllm_worker'), str(root/manifest['model']),
            str(self.path/'embeds.bin'), str(self.path/'cache.bin'),str(manifest.get('cpu_threads',3))], env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        self.log=(root/'rkllm_worker.log').open('w')
        self.allowed = manifest['supported_tokens']
        try:self._read('ready')
        except Exception:
            self.close();raise

    def _read(self, expected):
        for line in self.process.stdout:
            self.log.write(line);self.log.flush()
            if 'E RKNN' in line or 'E rkllm' in line or 'CALLBACK_ERROR' in line:
                raise RuntimeError('Native language error: '+line.strip())
            if line.startswith('RKLLM_REPLY '):
                value=line.strip().removeprefix('RKLLM_REPLY ')
                if value != expected: raise RuntimeError('Unexpected language reply: '+value)
                return
        raise RuntimeError('Native language worker exited')

    def infer(self, prefix, pad):
        valid=np.flatnonzero(pad[0]);n=len(valid)
        if n not in self.allowed: raise ValueError('Uncompiled full-prefill length: '+str(n))
        np.ascontiguousarray(prefix[0,valid],dtype='<f4').tofile(self.path/'embeds.bin')
        (self.path/'cache.bin').unlink(missing_ok=True)
        self.process.stdin.write(str(n)+'\n');self.process.stdin.flush();self._read('done 160')
        actual=n;n=160
        b=(self.path/'cache.bin').read_bytes()
        if struct.unpack('<III',b[:12]) != (0,8,n):raise ValueError('Cache header mismatch')
        if not np.array_equal(np.frombuffer(b,'<i4',n,12),np.arange(100,100+n)):raise ValueError('Cache IDs mismatch')
        if b[-16:]!=bytes.fromhex('6e736567010000000000000000000000'):raise ValueError('Cache footer mismatch')
        outputs=[];size=n*5*64*2
        for i in range(16):
            pair=[]
            for key,j in ((True,i),(False,16+i)):
                off=len(b)-16-(32-j)*(size+12)+12
                if struct.unpack('<III',b[off-12:off])!=((1,640,0) if key else (1,2,320)):
                    raise ValueError('Cache record mismatch')
                v=np.frombuffer(b,'<f2',n*5*64,off).astype('f4')
                if key:v=v.reshape(n,5,32,2).transpose(1,0,3,2).reshape(1,5,n,64)
                else:v=v.reshape(5,64,n).transpose(0,2,1)[None]
                if not np.isfinite(v).all():raise ValueError('Invalid native K/V')
                full=np.zeros((1,5,prefix.shape[1],64),dtype='f4');full[:,:,valid,:]=v[:,:,:actual,:];pair.append(full)
            outputs.extend(pair)
        return outputs

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        self.log.close()
        self.temp.cleanup()
