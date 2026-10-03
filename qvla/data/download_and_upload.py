#!/usr/bin/env python3
"""Download the pinned QVLA baseline bundle locally, then upload it to GPUServer.

Requires: python3 -m pip install --user 'huggingface_hub>=1.6,<2' requests
The local machine may use any Python version; wheels target Linux x86_64/Python 3.12.
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
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests
from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile


ROOT = Path(__file__).resolve().parents[2]
MODEL_SEED_HASHES = {
    "config.json": "5c9f3ba9f5f37ea7024c9501b0b20b1941f232989c2167853cb46d9071a70dd7",
    "model.safetensors": "9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8",
    "policy_preprocessor.json": "122ec5106602b1bf129f49690d05ab2f49748a0ac6119de55ea0677d4e90d248",
    "policy_postprocessor.json": "3b51f092c70c710ce0213ee1b63bf51b4878ec67828dd2d67daf9ef51081a41a",
    "policy_preprocessor_step_5_normalizer_processor.safetensors": "b0cdde6e8a6f49a8e19eefb376728e47c09d3b3cc20ce3a97c45619fe7a732d9",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors": "b0cdde6e8a6f49a8e19eefb376728e47c09d3b3cc20ce3a97c45619fe7a732d9",
}
SMOKE_DATA_FILES = {
    "meta/info.json",
    "meta/stats.json",
    "meta/tasks.parquet",
    "meta/episodes/chunk-000/file-000.parquet",
    "data/chunk-000/file-000.parquet",
    "videos/observation.images.image/chunk-000/file-000.mp4",
    "videos/observation.images.image2/chunk-000/file-000.mp4",
}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def repo_files(api: HfApi, repo: str, revision: str, repo_type: str | None) -> list[RepoFile]:
    return [entry for entry in api.list_repo_tree(
        repo, revision=revision, repo_type=repo_type, recursive=True, expand=True
    ) if isinstance(entry, RepoFile)]


def verified(path: Path, entry: RepoFile) -> bool:
    if not path.is_file() or path.stat().st_size != entry.size:
        return False
    if entry.lfs:
        return digest(path) == entry.lfs.sha256
    sha = hashlib.sha1()
    sha.update(f"blob {entry.size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest() == entry.blob_id


def download_file(
    endpoint: str, repo: str, revision: str, repo_type: str | None,
    destination: Path, entry: RepoFile,
) -> None:
    target = destination / entry.path
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    prefix = "datasets/" if repo_type == "dataset" else ""
    url = f"{endpoint.rstrip('/')}/{prefix}{repo}/resolve/{revision}/{quote(entry.path, safe='/')}"
    for attempt in range(1, 7):
        try:
            offset = part.stat().st_size if part.exists() else 0
            if offset > entry.size:
                part.unlink()
                offset = 0
            headers = {"Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with requests.get(url, headers=headers, stream=True, timeout=(20, 120)) as response:
                if response.status_code == 416 and offset == entry.size:
                    if verified(part, entry):
                        os.replace(part, target)
                        return
                    part.unlink()
                    raise ValueError(f"已下载内容校验失败，重新下载：{entry.path}")
                response.raise_for_status()
                if offset and response.status_code == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise ValueError(f"续传范围不匹配：{entry.path}: {content_range}")
                    mode = "ab"
                else:
                    mode = "wb"
                with part.open(mode) as stream:
                    for block in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if block:
                            stream.write(block)
            if not verified(part, entry):
                if part.stat().st_size == entry.size:
                    part.unlink()
                raise ValueError(f"文件大小或仓库哈希不匹配：{entry.path}")
            os.replace(part, target)
            return
        except (requests.RequestException, OSError, ValueError) as error:
            if attempt == 6:
                raise RuntimeError(f"下载失败：{entry.path}") from error
            print(f"重试 {entry.path} ({attempt}/6): {error}", flush=True)
            time.sleep(min(attempt * 2, 10))


def seed_model(downloads: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected in MODEL_SEED_HASHES.items():
        source = downloads / name
        if not source.is_file():
            continue
        if digest(source) != expected:
            raise ValueError(f"本地已下载文件校验失败：{source}")
        target = destination / name
        if target.is_file() and digest(target) == expected:
            continue
        shutil.copy2(source, target)
        print(f"复用已下载文件：{source}", flush=True)


def download_repo(
    api: HfApi, repo: str, revision: str, destination: Path,
    *, repo_type: str | None = None, selected: set[str] | None = None,
) -> list[RepoFile]:
    entries = repo_files(api, repo, revision, repo_type)
    entries = [entry for entry in entries if selected is None or entry.path in selected]
    if selected is not None:
        absent = selected - {entry.path for entry in entries}
        if absent:
            raise ValueError(f"仓库 {repo} 缺少预期文件：{sorted(absent)}")
    destination.mkdir(parents=True, exist_ok=True)
    pending = [entry for entry in entries if not verified(destination / entry.path, entry)]
    total = sum(entry.size or 0 for entry in entries)
    pending_bytes = sum(entry.size or 0 for entry in pending)
    print(f"{repo}@{revision}: {len(entries)} 个文件，共 {total / 1e9:.2f} GB；"
          f"待下载 {len(pending)} 个、约 {pending_bytes / 1e9:.2f} GB", flush=True)
    if pending:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(download_file, api.endpoint, repo, revision,
                                       repo_type, destination, entry) for entry in pending]
            for count, future in enumerate(as_completed(futures), start=1):
                future.result()
                if count % 10 == 0 or count == len(pending):
                    print(f"  已完成 {count}/{len(pending)} 个文件", flush=True)
    for entry in entries:
        path = destination / entry.path
        if not verified(path, entry):
            raise ValueError(f"下载不完整或仓库哈希不匹配：{path}")
    return entries


def write_manifest(destination: Path, paths: list[str]) -> None:
    lines = []
    for relative in sorted(paths):
        if "\n" in relative or "\r" in relative or relative.startswith("-"):
            raise ValueError(f"不支持的文件名：{relative!r}")
        lines.append(f"{digest(destination / relative)}  {relative}\n")
    (destination / ".transfer-manifest.sha256").write_text("".join(lines))


def verify_manifest(destination: Path) -> None:
    manifest = destination / ".transfer-manifest.sha256"
    if not manifest.is_file():
        raise FileNotFoundError(f"缺少下载完成清单：{manifest}")
    lines = manifest.read_text().splitlines()
    if not lines:
        raise ValueError(f"空清单：{manifest}")
    for line in lines:
        expected, separator, relative = line.partition("  ")
        if not separator or len(expected) != 64 or not relative:
            raise ValueError(f"无效清单行：{line!r}")
        path = destination / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"上传前校验失败：{path}")


WHEEL_REQUIREMENTS = (
    "torch==2.7.1+cu118", "torchvision==0.22.1+cu118",
    "lerobot[smolvla,training]==0.6.1", "torchao==0.11.0",
)


def wheel_options(destination: Path) -> list[str]:
    return [
        "--index-url", "https://pypi.org/simple/",
        "--extra-index-url", "https://download.pytorch.org/whl/cu118",
        "--find-links", str(destination),
        "--python-version", "3.12", "--implementation", "cp",
        "--abi", "cp312", "--abi", "abi3", "--abi", "none",
        "--platform", "manylinux_2_28_x86_64",
        "--platform", "manylinux2014_x86_64",
        "--only-binary=:all:", "--timeout", "120", "--retries", "5",
    ]


def ensure_docopt_wheel(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    # num2words depends on docopt 0.6.2, which PyPI publishes only as an sdist.
    # Build its universal pure-Python wheel locally before cross-platform resolve.
    if not any(destination.glob("docopt-0.6.2-py2.py3-none-any.whl")):
        run([
            sys.executable, "-m", "pip", "--isolated", "wheel", "--no-cache-dir",
            "--no-deps", "--wheel-dir", str(destination),
            "--index-url", "https://pypi.org/simple/", "docopt==0.6.2",
        ])


def download_wheels(destination: Path) -> None:
    ensure_docopt_wheel(destination)
    # Target the server's Python 3.12 / Linux x86_64, even if the laptop uses
    # another Python version. +cu118 matches the A10 server's existing driver.
    run([
        sys.executable, "-m", "pip", "--isolated", "download",
        "--dest", str(destination),
        *wheel_options(destination), *WHEEL_REQUIREMENTS,
    ])
    wheel_paths = [p.name for p in destination.iterdir() if p.is_file() and p.suffix == ".whl"]
    if not wheel_paths:
        raise ValueError("wheelhouse 为空")
    write_manifest(destination, wheel_paths)


def upload(local: Path, remote: str, ssh_alias: str, ssh_config: Path) -> None:
    ssh = ["ssh", "-F", str(ssh_config), ssh_alias]
    run(ssh + [f"mkdir -p {shlex.quote(remote)}"])
    has_rsync = subprocess.run(
        ssh + ["command -v rsync >/dev/null 2>&1"], check=False,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0
    if has_rsync:
        run([
            "rsync", "-a", "--checksum", "--partial", "--append-verify",
            "--exclude", ".cache/", "-e", shlex.join(ssh[:-1]),
            str(local) + "/", f"{ssh_alias}:{remote}/",
        ])
    else:
        print("服务器没有 rsync，改用 scp 上传。", flush=True)
        run(["scp", "-O", "-F", str(ssh_config), str(local / ".transfer-manifest.sha256"),
             f"{ssh_alias}:{remote}/"])
        checked = subprocess.run(
            ssh + [f"cd {shlex.quote(remote)} && LC_ALL=C sha256sum -c .transfer-manifest.sha256 2>/dev/null"],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        present = {line[:-4] for line in checked.stdout.splitlines() if line.endswith(": OK")}
        paths = [local / line.partition("  ")[2]
                 for line in (local / ".transfer-manifest.sha256").read_text().splitlines()
                 if line.partition("  ")[2] not in present]
        print(f"服务器已有 {len(present)} 个校验通过的文件，待上传 {len(paths)} 个。", flush=True)
        # scp accepts regular files; upload per directory so paths are preserved.
        for directory in sorted({path.parent for path in paths}, key=lambda p: str(p)):
            relative = directory.relative_to(local)
            destination = remote if relative == Path(".") else f"{remote}/{relative.as_posix()}"
            run(ssh + [f"mkdir -p {shlex.quote(destination)}"])
            files = [str(path) for path in paths if path.parent == directory]
            if files:
                run(["scp", "-O", "-F", str(ssh_config), *files, f"{ssh_alias}:{destination}/"])
    run(ssh + [f"cd {shlex.quote(remote)} && sha256sum -c .transfer-manifest.sha256 --quiet"])


def without_server_packages(wheelhouse: Path, destination: Path) -> None:
    """Stage packages absent from the DSW image; preserve source hashes."""
    destination.mkdir(parents=True, exist_ok=True)
    excluded = ("torch-", "torchvision-", "triton-", "nvidia_")
    selected = []
    for source in wheelhouse.iterdir():
        if not (source.name.endswith(".whl") or source.name.endswith(".tar.gz")):
            continue
        if source.name.startswith(excluded):
            continue
        target = destination / source.name
        if not target.exists():
            os.link(source, target)
        selected.append(source.name)
    write_manifest(destination, selected)
    print(f"复用服务器 PyTorch/CUDA：上传 {len(selected)} 个依赖包，"
          f"约 {sum((destination / name).stat().st_size for name in selected) / 1e9:.2f} GB", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "transfer")
    parser.add_argument("--endpoint", default="https://huggingface.co",
                        help="Hugging Face 官方地址，默认 https://huggingface.co")
    parser.add_argument("--dataset", choices=("full", "smoke"), default="full")
    parser.add_argument("--skip-wheels", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--upload-only", action="store_true",
                        help="服务器重新启动后，仅校验本地文件并上传，不访问下载源")
    parser.add_argument("--reuse-server-torch", action="store_true",
                        help="上传时复用 DSW 镜像的 PyTorch/CUDA，跳过对应离线包")
    parser.add_argument("--ssh-alias", default="GPUServer")
    parser.add_argument("--ssh-config", type=Path, default=Path.home() / ".ssh" / "config")
    parser.add_argument("--remote-root", default="/root/qvla")
    args = parser.parse_args()
    if args.skip_upload and args.upload_only:
        parser.error("--skip-upload 与 --upload-only 不能同时使用")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.ssh_alias):
        parser.error("--ssh-alias 只能包含字母、数字、下划线、点和连字符")
    if not args.remote_root.startswith("/") or "'" in args.remote_root or "\n" in args.remote_root:
        parser.error("--remote-root 必须是简单的绝对路径")

    lock = json.loads((ROOT / "config" / "step1.lock.json").read_text())
    output = args.output.resolve()
    model_dir = output / "model"
    vlm_dir = output / "smolvlm2_assets"
    dataset_dir = output / "libero"
    libero_assets_dir = output / "libero_assets"
    upload_wheel_dir = output / "wheelhouse"
    if args.upload_only:
        for directory in (model_dir, vlm_dir, dataset_dir, libero_assets_dir):
            verify_manifest(directory)
        if not args.skip_wheels:
            verify_manifest(output / "wheelhouse")
    else:
        api = HfApi(endpoint=args.endpoint)
        seed_model(args.downloads.expanduser(), model_dir)
        model_entries = download_repo(api, lock["model_repo"], lock["model_revision"], model_dir)
        write_manifest(model_dir, [entry.path for entry in model_entries])

        vlm_entries = repo_files(api, lock["vlm_assets_repo"], lock["vlm_assets_revision"], None)
        assets = {entry.path for entry in vlm_entries
                  if "/" not in entry.path and entry.path != "model.safetensors"}
        vlm_entries = download_repo(api, lock["vlm_assets_repo"], lock["vlm_assets_revision"],
                                    vlm_dir, selected=assets)
        write_manifest(vlm_dir, [entry.path for entry in vlm_entries])

        selected = None if args.dataset == "full" else SMOKE_DATA_FILES
        data_entries = download_repo(api, lock["dataset_repo"], lock["dataset_revision"],
                                     dataset_dir, repo_type="dataset", selected=selected)
        write_manifest(dataset_dir, [entry.path for entry in data_entries])

        asset_entries = download_repo(api, lock["libero_assets_repo"],
                                      lock["libero_assets_revision"], libero_assets_dir,
                                      repo_type="dataset")
        write_manifest(libero_assets_dir, [entry.path for entry in asset_entries])

        if not args.skip_wheels:
            download_wheels(output / "wheelhouse")

    if args.reuse_server_torch and not args.skip_wheels:
        upload_wheel_dir = output / "wheelhouse_server_torch"
        without_server_packages(output / "wheelhouse", upload_wheel_dir)
        verify_manifest(upload_wheel_dir)

    if not args.skip_upload:
        remote_root = args.remote_root.rstrip("/")
        upload(model_dir, f"{remote_root}/artifacts/model", args.ssh_alias, args.ssh_config)
        upload(vlm_dir, f"{remote_root}/artifacts/smolvlm2_assets", args.ssh_alias, args.ssh_config)
        upload(dataset_dir, f"{remote_root}/data/libero", args.ssh_alias, args.ssh_config)
        upload(libero_assets_dir, f"{remote_root}/artifacts/libero_assets",
               args.ssh_alias, args.ssh_config)
        if not args.skip_wheels:
            upload(upload_wheel_dir, f"{remote_root}/artifacts/wheelhouse", args.ssh_alias,
                   args.ssh_config)
    print("所选阶段完成。", flush=True)


if __name__ == "__main__":
    main()
