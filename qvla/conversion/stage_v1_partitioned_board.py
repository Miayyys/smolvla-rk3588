"""Validate downloaded V1 graphs and upload with resumable rsync to RK3588."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))
from qvla.paths import source_path

import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--models',type=Path,default=Path('runs/v1_exact_precision_v1'))
p.add_argument('--board',default='root@10.42.0.252')
a=p.parse_args();project=Path(__file__).resolve().parents[2];models=a.models.resolve()
manifest=json.loads((models/'deployment_manifest.json').read_text());files=[]
for group in manifest['partitioned_graphs'].values():
 for part in group['parts']:
  path=models/part['filename'];h=hashlib.sha256()
  with path.open('rb') as source:
   for b in iter(lambda:source.read(1048576),b''):h.update(b)
  if h.hexdigest()!=part['sha256']:raise ValueError('Downloaded model hash mismatch: '+str(path))
  files.append(path)
root='/dev/shm/qvla_v1_exact_precision_v1';base='/root/qvla_board_test/qat_v1_no_teacher_v1'
remote='''from pathlib import Path
import hashlib,json
r=Path(ROOT);base=Path(BASE);r.mkdir(exist_ok=True)
pm=json.loads((base/'preprocess_export.json').read_text())
files=['cpu_weights.npz','state_stats.npz','raw_inputs.npz','fp_reference.npz','replay.json','preprocess_export.json','smolvla_numpy_glue.py','smolvla_board_preprocess.py','rknn_board_full_replay.py','vision_selected.rknn',*pm['config_hashes']]
for name in dict.fromkeys(files):
 p=base/name
 if not p.is_file():raise FileNotFoundError(p)
 link=r/name
 if not link.exists():link.symlink_to(p)
 if link.resolve()!=p.resolve():raise ValueError('Unexpected existing asset '+name)
print(json.dumps({'root':str(r),'state':'base_assets_ready'}))
'''.replace('ROOT',repr(root)).replace('BASE',repr(base))
ssh=['ssh','-F','/dev/null','-o','ProxyCommand=none','-o','ConnectTimeout=8']
subprocess.run([*ssh,a.board,'python3 -'],input=remote,text=True,check=True)
files.append(models/'deployment_manifest.json')
files.extend(source_path(n) for n in ['smolvla_board_runtime.py','smolvla_rknn_partitions.py','smolvla_bf16_projection.py','rknn_bf16_projection.c','serve_smolvla_board_stdio.py','verify_v1_partitioned_replay.py'])
files.append(project/'runs/precision_support/rknn_api.h')
subprocess.run(['rsync','-a','--checksum','--partial','--info=progress2','-e','ssh -F /dev/null -o ProxyCommand=none -o ConnectTimeout=8',*[str(x) for x in files],a.board+':'+root+'/'],check=True)
subprocess.run([*ssh,a.board,'cd '+root+' && gcc -O2 -Wall -shared -fPIC rknn_bf16_projection.c -I. -L/usr/lib -lrknnrt -o librknn_bf16_projection.so'],check=True)
from qvla.runtime.deployment import stage_source
stage_source(a.board,root)
print(json.dumps({'root':root,'uploaded':True,'board_execution':'not_measured'}))
