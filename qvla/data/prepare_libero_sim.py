#!/usr/bin/env python3
"""Download the LeRobot 0.6.1 LIBERO simulator extra locally and upload only missing files.

The laptop has Python 3.14 but GPUServer has Python 3.12. uv supplies a local
Python 3.12 resolver, so compiled wheels match the server's interpreter.
"""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))


import argparse
import shlex
import subprocess
from pathlib import Path

from qvla.data.download_and_upload import ROOT, upload, verify_manifest, write_manifest


WHEELHOUSE = ROOT / "artifacts" / "transfer" / "wheelhouse"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--upload-only", action="store_true")
    parser.add_argument("--ssh-alias", default="GPUServer")
    parser.add_argument("--ssh-config", type=Path, default=Path.home() / ".ssh" / "config")
    args = parser.parse_args()
    if args.skip_upload and args.upload_only:
        parser.error("--skip-upload 与 --upload-only 不能同时使用")
    if not args.upload_only:
        WHEELHOUSE.mkdir(parents=True, exist_ok=True)
        cmd = [
            "uv", "run", "--no-project", "--python", "3.12", "--with", "pip", "python",
            "-m", "pip", "--isolated", "download", "--dest", str(WHEELHOUSE),
            "--find-links", str(WHEELHOUSE),
            "--index-url", "https://pypi.org/simple/",
            "--extra-index-url", "https://download.pytorch.org/whl/cu118",
            "torch==2.7.1+cu118", "torchvision==0.22.1+cu118",
            "lerobot[libero]==0.6.1",
        ]
        print("+", shlex.join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        packages = [p.name for p in WHEELHOUSE.iterdir()
                    if p.is_file() and (p.name.endswith(".whl") or p.name.endswith(".tar.gz"))]
        write_manifest(WHEELHOUSE, packages)
        print(f"已校验并登记 {len(packages)} 个 wheel/源码包。", flush=True)
    else:
        verify_manifest(WHEELHOUSE)
    if not args.skip_upload:
        upload(WHEELHOUSE, "/root/qvla/artifacts/wheelhouse", args.ssh_alias,
               args.ssh_config.expanduser())


if __name__ == "__main__":
    main()
