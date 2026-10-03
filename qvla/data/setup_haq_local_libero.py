#!/usr/bin/env python3
"""Point the isolated LIBERO installation at already downloaded local assets."""

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))

import argparse
import importlib.util
import json
from pathlib import Path

import yaml

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,default=ROOT/'artifacts/transfer/libero_assets')
    p.add_argument('--dataset',type=Path,default=ROOT/'artifacts/transfer/libero')
    p.add_argument('--config-dir',type=Path,default=ROOT/'runs/libero_local/config')
    args=p.parse_args()
    spec=importlib.util.find_spec('libero')
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError('Install hf-libero 0.1.4 in the isolated project environment first')
    package=Path(next(iter(spec.submodule_search_locations)))/'libero'
    if not package.is_dir():raise FileNotFoundError(package)
    for root in [args.assets,args.dataset,package/'bddl_files',package/'init_files']:
        if not root.is_dir():raise FileNotFoundError(root)
    package_assets=package/'assets'
    if package_assets.exists() or package_assets.is_symlink():
        if package_assets.resolve()!=args.assets.resolve():
            raise ValueError(f'LIBERO package already uses different assets: {package_assets}')
    else:
        package_assets.symlink_to(args.assets.resolve(),target_is_directory=True)
    args.config_dir.mkdir(parents=True,exist_ok=True)
    mapping={'benchmark_root':str(package.resolve()),
             'bddl_files':str((package/'bddl_files').resolve()),
             'init_states':str((package/'init_files').resolve()),
             'datasets':str(args.dataset.resolve()),
             'assets':str(args.assets.resolve())}
    config=args.config_dir/'config.yaml'
    if config.exists():
        if yaml.safe_load(config.read_text())!=mapping:raise ValueError(f'Different existing config: {config}')
    else:
        config.write_text(yaml.safe_dump(mapping))
    (args.config_dir.parent/'mpl').mkdir(exist_ok=True)
    print(json.dumps({'LIBERO_CONFIG_PATH':str(args.config_dir.resolve()),
                      'MUJOCO_GL':'egl','config':str(config),'assets':str(args.assets.resolve())}))


if __name__=='__main__':main()
