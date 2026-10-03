"""Add a pinned output-sync diagnostic callback to the isolated mask library."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True)
a=p.parse_args();b=bytearray(a.source.read_bytes())
assert hashlib.sha256(b).hexdigest()=='0030eb3ea743b48a56286bea77ca723cfe33311c0508c59325302fb0bd309072'
assert struct.unpack_from('<I',b,0x332124)[0]==0x9400396c
assert struct.unpack_from('<I',b,0x331ccc)[0]==0x94003ac1
assert not any(b[0x713800:0x713a00])
a.output_dir.mkdir(parents=True,exist_ok=True);scripts=Path(__file__).resolve().parent
for cmd in [
 ['aarch64-linux-gnu-as',str(scripts/'rkllm_native_matmul_trace.s'),'-o',str(a.output_dir/'trace.o')],
 ['aarch64-linux-gnu-ld','-T',str(scripts/'rkllm_native_matmul_trace.ld'),str(a.output_dir/'trace.o'),'-o',str(a.output_dir/'trace.elf')],
 ['aarch64-linux-gnu-objcopy','-O','binary',str(a.output_dir/'trace.elf'),str(a.output_dir/'trace.bin')]]:
 subprocess.run(cmd,check=True)
cave=(a.output_dir/'trace.bin').read_bytes();assert len(cave)<=512
b[0x713800:0x713800+len(cave)]=cave
struct.pack_into('<I',b,0x332124,0x14000000|(((0x713800-0x332124)//4)&0x3ffffff))
struct.pack_into('<I',b,0x331ccc,0x14000000|(((0x713900-0x331ccc)//4)&0x3ffffff))
assert struct.unpack_from('<QQ',b,96)==(0x713800,0x713800)
struct.pack_into('<QQ',b,96,0x713a00,0x713a00)
out=a.output_dir/'librkllmrt.so';assert out.resolve()!=a.source.resolve();out.write_bytes(b)
(a.output_dir/'trace_patch.json').write_text(json.dumps({'scope':'post native input/output sync callbacks, diagnostic only','sha256':hashlib.sha256(b).hexdigest(),'callback_va':'0x7138f0'},indent=2)+'\n')
