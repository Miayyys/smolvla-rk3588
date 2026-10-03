"""Build an isolated native mask patch for a compact prefix followed by one state token."""

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
import struct
import subprocess
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output-dir', type=Path, required=True)
p.add_argument('--aligned', action='store_true', help='Mask trailing alignment tokens; live count at VA 0x7137f0')
p.add_argument('--causal-control',action='store_true',help='Diagnostic: keep causal attention to isolate full-prefill execution')
a = p.parse_args()
b = bytearray(a.source.read_bytes())
digest = hashlib.sha256(b).hexdigest()
assert digest == 'f25e9b099db08aaacd0a3ac62b4697d3951d6ae61ae41ea09f6702cfa89eb32c'
a.output_dir.mkdir(parents=True, exist_ok=True)
scripts = Path(__file__).resolve().parent
stem='rkllm_native_aligned_mask' if a.aligned else 'rkllm_native_block_mask'
obj, elf, binary = [a.output_dir / ('cave.'+x) for x in ('o','elf','bin')]
assembly=scripts/(stem+'.s')
if a.causal_control:
    assembly=a.output_dir/'causal_control.s'
    assembly.write_text((scripts/(stem+'.s')).read_text().replace('strb wzr,[x24,#57]', 'mov w0,#1\nstrb w0,[x24,#57]'))
for cmd in [
    ['aarch64-linux-gnu-as', str(assembly), '-o', str(obj)],
    ['aarch64-linux-gnu-ld', '-T', str(scripts/(stem+'.ld')), str(obj), '-o', str(elf)],
    ['aarch64-linux-gnu-objcopy', '-O', 'binary', str(elf), str(binary)]]:
    subprocess.run(cmd, check=True)
cave = binary.read_bytes()
assert len(cave) <= 0x140 and not any(b[0x7136c0:0x713800])
for addr, expected, target in [(0x1cdbd8,0xbc2578e0,0x7136c0), (0x1e36c8,0x3900e703,0x713740)]:
    assert struct.unpack_from('<I', b, addr)[0] == expected
    delta = target - addr
    assert delta % 4 == 0 and -(1<<27) <= delta < (1<<27)
    struct.pack_into('<I', b, addr, 0x14000000 | ((delta//4) & 0x3ffffff))
b[0x7136c0:0x7136c0+len(cave)] = cave
# This exact ELF maps its executable first segment at VA/file offset zero.
# Extend that segment only through zero padding, before next file segment 0x714138.
assert struct.unpack_from('<Q',b,32)[0] == 64
assert struct.unpack_from('<IIQQ',b,64) == (1,5,0,0)
assert struct.unpack_from('<QQ',b,96) == (0x7136b3,0x7136b3)
struct.pack_into('<QQ',b,96,0x713800,0x713800)
out = a.output_dir/'librkllmrt.so'
assert out.resolve() != a.source.resolve()
out.write_bytes(b)
report = dict(source_sha256=digest, patched_sha256=hashlib.sha256(b).hexdigest(),
              scope='experimental fixed native block-mask implementation, not official API',
              mask='all prefix bidirectional; prefix excludes final state; state sees all',
              precondition='single full prefill, compact valid tokens, one final state token',
              causal_disabled=not a.causal_control, alignment_tail_mask=a.aligned, diagnostic_causal_control=a.causal_control, ubatch_set_to_batch=True, original_library_untouched=True)
(a.output_dir/'patch_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
