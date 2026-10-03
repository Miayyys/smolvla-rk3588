"""Prepare an isolated SDK 1.3.1 runtime hypothesis; board validation is mandatory."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output-dir', type=Path, required=True)
a = p.parse_args()
b = bytearray(a.source.read_bytes())
source_hash = hashlib.sha256(b).hexdigest()
expected = 'f25e9b099db08aaacd0a3ac62b4697d3951d6ae61ae41ea09f6702cfa89eb32c'
if source_hash != expected:
    raise ValueError('This probe only supports the audited official ARM64 runtime hash')
# Locate the instruction through ELF64 PT_LOAD rather than assuming VA == file offset.
phoff = struct.unpack_from('<Q', b, 32)[0]
entsize, count = struct.unpack_from('<HH', b, 54)
va = 0x1e36c8
offset = None
for i in range(count):
    typ, flags, off, addr, _, filesz, _, _ = struct.unpack_from('<IIQQQQQQ', b, phoff+i*entsize)
    if typ == 1 and flags & 1 and addr <= va and va+4 <= addr+filesz:
        offset = off + va - addr
if offset is None:
    raise ValueError('Instruction is not in an executable file-backed segment')
old = struct.unpack_from('<I', b, offset)[0]
if old != 0x3900e703:
    raise ValueError('Audited instruction mismatch')
# strb w3,[x24,#57] -> strb wzr,[x24,#57]. This field is inferred from
# the cparams.causal_attn assertion and related mask/context code, not public ABI.
new = 0x3900e71f
struct.pack_into('<I', b, offset, new)
a.output_dir.mkdir(parents=True, exist_ok=True)
out = a.output_dir / 'librkllmrt.so'
if out.resolve() == a.source.resolve():
    raise ValueError('Never overwrite the source runtime')
out.write_bytes(b)
report = dict(status='prepared_not_board_validated', source_sha256=source_hash,
              patched_sha256=hashlib.sha256(b).hexdigest(), instruction_va=hex(va),
              file_offset=hex(offset), old_instruction=hex(old), new_instruction=hex(new),
              hypothesis='force context causal_attn=false; not yet verified',
              supports_smolvla_block_mask=False, performance_measured=False,
              source_runtime_overwritten=False)
(a.output_dir / 'patch_report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
