"""Verify and stage the selected board release without training or conversion."""
import argparse
import hashlib
import json
import shlex
import subprocess
import tarfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT / 'models/final'


def digest(path):
    sha=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''): sha.update(block)
    return sha.hexdigest()


def verify(root, runtime=None):
    root=Path(root)
    m=json.loads((root/'deployment_manifest.json').read_text())
    pm=json.loads((root/'preprocess_export.json').read_text())
    pinned={g['filename']:g['sha256'] for g in m['graphs']}
    for partition in m.get('partitioned_graphs',{}).values():
        pinned.update({part['filename']:part['sha256'] for part in partition['parts']})
    pinned.update(pm['config_hashes'])
    pinned['cpu_weights.npz']=m['cpu_weights_sha256']
    pinned['state_stats.npz']=pm['hashes']['state_stats.npz']
    for name,sha in pinned.items():
        target=root/name
        if not target.is_file():raise FileNotFoundError(target)
        if target.is_symlink():raise ValueError('Release must contain real files: '+name)
        if digest(target)!=sha:raise ValueError('Release hash mismatch (or unpulled LFS pointer): '+name)
    # replay.json configures the Euler steps and needs its own frozen release identity.
    metadata=root/'release.json'
    if metadata.exists():
        for name,sha in json.loads(metadata.read_text())['file_hashes'].items():
            if digest(root/name)!=sha:raise ValueError('Release metadata mismatch: '+name)
    if runtime is not None:
        runtime=Path(runtime)
        for name,sha in m.get('language_backend',{}).get('assets',{}).items():
            target=root/name if name in pinned else runtime/name
            if digest(target)!=sha:raise ValueError('Runtime hash mismatch: '+name)
        if not (runtime/'librknn_bf16_projection.so').is_file():raise FileNotFoundError('Missing native BF16 wrapper')
        if metadata.exists():
            for name,sha in json.loads(metadata.read_text()).get('native_runtime_hashes',{}).items():
                if name.endswith('.h'):continue
                if digest(runtime/name)!=sha:raise ValueError('Release native runtime mismatch: '+name)
    return dict(status='verified',model_root=str(root.resolve()),pinned_files=len(pinned),native_runtime_checked=runtime is not None)


def stage(args):
    print(json.dumps(verify(args.root,args.runtime_dir)),flush=True)
    target=args.destination.rstrip('/')
    if not target.startswith('/') or target=='/':raise ValueError('Use an explicit absolute board directory')
    ssh=['ssh','-F','/dev/null','-o','ProxyCommand=none','-o','ConnectTimeout=8',args.board]
    archive=PROJECT/'runs/deployment_stage.tar'
    archive.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'w') as bundle:
        def source_filter(item):
            if '__pycache__' in Path(item.name).parts:return None
            item.mtime=0
            return item
        bundle.add(PROJECT/'qvla',arcname='qvla',filter=source_filter)
        bundle.add(PROJECT/'scripts',arcname='scripts',filter=source_filter)
        if not args.code_only:
            bundle.add(args.root,arcname='models/final')
            for file in args.runtime_dir.rglob('*'):
                if file.is_file() and file.suffix!='.h':
                    bundle.add(file,arcname='models/final/'+str(file.relative_to(args.runtime_dir)))
    command='mkdir -p '+shlex.quote(target)+' && tar -xf - -C '+shlex.quote(target)
    with archive.open('rb') as stream:
        subprocess.run([*ssh,command],stdin=stream,check=True)
    archive.unlink()
    print(json.dumps(dict(status='staged',board=args.board,destination=target,code_only=args.code_only)))


def stage_source(board, destination):
    """Copy the package for historical flat launchers after source migration."""
    import tempfile
    with tempfile.TemporaryFile() as stream:
        with tarfile.open(fileobj=stream,mode='w') as bundle:
            def source_filter(item):
                if '__pycache__' in Path(item.name).parts:return None
                item.mtime=0
                return item
            bundle.add(PROJECT/'qvla',arcname='qvla',filter=source_filter)
        stream.seek(0)
        subprocess.run(['ssh','-F','/dev/null','-o','ProxyCommand=none',
                        '-o','ConnectTimeout=8',board,
                        'tar -xf - -C '+shlex.quote(destination)],stdin=stream,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest='stage',required=True)
    v=commands.add_parser('verify',help='Verify final model hashes (no NPU needed)')
    v.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    v.add_argument('--runtime-dir',type=Path)
    s=commands.add_parser('stage',help='Stage final model, package and prepared native runtime to a board')
    s.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    s.add_argument('--runtime-dir',type=Path,default=PROJECT/'artifacts/runtime/final')
    s.add_argument('--board',default='root@10.42.0.252')
    s.add_argument('--destination',default='/root/qvla')
    s.add_argument('--code-only',action='store_true')
    serve=commands.add_parser('serve',help='Serve inference over stdio on RK3588')
    serve.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    infer=commands.add_parser('infer',help='Run one raw observation NPZ on RK3588')
    infer.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    infer.add_argument('--input',type=Path,required=True)
    infer.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.stage=='verify':print(json.dumps(verify(args.root,args.runtime_dir)))
    elif args.stage=='stage':stage(args)
    elif args.stage=='serve':
        import sys
        from qvla.runtime.serve_smolvla_board_stdio import main as serve_main
        sys.argv=[sys.argv[0],'--root',str(args.root)];serve_main()
    else:
        import numpy as np
        from qvla.runtime.smolvla_board_runtime import BoardSmolVLA
        with np.load(args.input,allow_pickle=False) as values:raw={name:values[name].copy() for name in values.files}
        model=BoardSmolVLA(args.root)
        try:
            actions,timing=model.predict(raw)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(args.output,actions=actions)
            print(json.dumps(timing))
        finally:model.close()
