"""Run the isolated native-runtime hypothesis on the board, then compare native K/V."""

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
import shlex
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--board', default='root@10.42.0.252')
p.add_argument('--root', type=Path, default=Path('runs/rkllm_native_patch_v1'))
a = p.parse_args()
remote = '/root/qvla_board_test/rkllm_native_patch_v1'
ssh = ['ssh', '-o', 'ProxyCommand=none', '-o', 'ConnectTimeout=8', a.board]
scp = ['scp', '-O', '-o', 'ProxyCommand=none', '-o', 'ConnectTimeout=8']

def run(cmd):
    print(shlex.join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)

# Connect before copying or changing board files. The installed runtime is untouched.
run(ssh + ['mkdir -p ' + remote + '/patched_lib'])
run(scp + [a.root/'patched_lib/librkllmrt.so', a.board+':'+remote+'/patched_lib/'])
run(scp + ['runs/rkllm_config_recheck_v1/tiny_noembed_fp16.rkllm',
           'runs/rkllm_dump_probe_v1/input_embeds.bin', 'qvla/runtime/probe_rkllm_dump_run.cpp',
           'runs/rkllm_prefix_feasibility_v1/rkllm.h', a.board+':'+remote+'/'])
run(ssh + ['cd '+remote+' && g++ -O2 probe_rkllm_dump_run.cpp -I. '
           '-L/usr/local/lib -lrkllmrt -Wl,-rpath,/usr/local/lib -o probe'])
cmd = ('cd '+remote+' && LD_LIBRARY_PATH='+remote+'/patched_lib ldd ./probe')
resolved = subprocess.check_output(ssh+[cmd], text=True)
(a.root/'library_resolution.log').write_text(resolved)
if remote+'/patched_lib/librkllmrt.so' not in resolved:
    raise RuntimeError('Isolated runtime was not selected by dynamic loader')
with (a.root/'board.log').open('w') as log:
    subprocess.run(ssh + ['cd '+remote+' && LD_LIBRARY_PATH='+remote+'/patched_lib '
                          'RKLLM_DUMP_LEVEL=0 timeout 60 ./probe tiny_noembed_fp16.rkllm '
                          'input_embeds.bin 0 1 callback'], stdout=log, stderr=subprocess.STDOUT, check=True)
run(scp + [a.board+':'+remote+'/prompt_cache.bin', a.root/'prompt_cache.bin'])
for mode, ref in [('causal','fp_kv_reference.npz'), ('noncausal','noncausal_fp_kv_reference.npz')]:
    run([sys.executable, 'qvla/evaluation/probe_rkllm_cache_parse.py', '--cache', a.root/'prompt_cache.bin',
         '--reference', 'runs/rkllm_dump_probe_v1/'+ref, '--output', a.root/(mode+'_report.json')])
report = {mode: json.loads((a.root/(mode+'_report.json')).read_text()) for mode in ('causal','noncausal')}
(a.root/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
