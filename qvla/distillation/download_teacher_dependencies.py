#!/usr/bin/env python3
"""Download only the missing CPython 3.12 teacher wheels from official PyPI."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import argparse
import sys
from pathlib import Path

from qvla.data.download_and_upload import ROOT, run, upload, write_manifest, verify_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--skip-upload',action='store_true')
    p.add_argument('--upload-only',action='store_true')
    args=p.parse_args()
    if args.skip_upload and args.upload_only:p.error('Choose one transfer phase')
    dest=ROOT/'artifacts/teacher_wheelhouse';dest.mkdir(parents=True,exist_ok=True)
    if not args.upload_only:
        run([sys.executable,'-m','pip','--isolated','download','--no-deps',
             '--index-url','https://pypi.org/simple/','--dest',str(dest),
             '--python-version','3.12','--implementation','cp',
             '--abi','cp312','--abi','abi3','--abi','none',
             '--platform','manylinux_2_28_x86_64','--platform','manylinux2014_x86_64',
             '--platform','manylinux2010_x86_64',
             '--only-binary=:all:','--timeout','120','--retries','5',
             '-r',str(ROOT/'config/teacher_missing_wheels.txt')])
        write_manifest(dest,[f.name for f in dest.glob('*.whl')])
    verify_manifest(dest)
    if not args.skip_upload:
        upload(dest,'/root/qvla/artifacts/teacher_wheelhouse','GPUServer',Path.home()/'.ssh/config')
    print('教师依赖阶段完成；未下载模型、数据集或PyTorch/CUDA。',flush=True)

if __name__=='__main__':main()
