# 官方源下载与 GPUServer 上传

本页记录数据准备与两种服务器环境路径。新服务器优先复用其预装 PyTorch/CUDA，并在实验前核对版本；`2.7.1+cu118` wheelhouse 命令用于隔离复现旧环境。切换到预装环境后，需重跑 FP 基线，不能直接比较不同软件栈的性能数据。

本脚本只覆盖当前锁定的 `smolvla_libero` + `lerobot/libero` 路线。将来换模型或数据集时，先更新 `config/step1.lock.json` 和脚本清单。完整 LIBERO 数据集约 1.94 GB、457 个文件，后续校准/测试 episode 可从中隔离划分。闭环模拟另需 `lerobot/libero-assets` 场景资产；它和 episode 数据是两个仓库。

## 在本机运行

```bash
cd /home/loser/Study/QVLA
python3 -m pip --isolated install --user --index-url https://pypi.org/simple/ 'huggingface_hub>=1.6,<2' requests
python3 scripts/download_and_upload.py --skip-upload
```

服务器可以在本机下载期间保持关闭。重新启动后运行：

新 DSW 镜像已提供可用的 PyTorch/CUDA 时，可省去约 2.9 GB 重复包：

```bash
python3 scripts/download_and_upload.py --upload-only --reuse-server-torch
```

此命令只校验并上传，**不会访问下载站**。它仍上传模型、processor、完整 LIBERO episode 数据、场景资产及其余 Python 依赖。服务器预装的 NVIDIA PyTorch `2.7.0a0+nv25.03` 与旧实验的 `2.7.1+cu118` 版本不同；采用它时需在新环境重新跑 FP、QAT 和 PTQ 对照，不能混用旧性能结果。若不复用服务器 PyTorch，去掉 `--reuse-server-torch` 会上传完整 wheelhouse。

默认使用 `https://huggingface.co`、`https://pypi.org/simple/` 和 `https://download.pytorch.org/whl/cu118`。若下载中断，原命令重跑；已有文件按仓库哈希检查，未完成的 `.part` 文件使用 HTTP Range 续传。`--upload-only` 会先按本地 SHA-256 清单校验，不访问下载源。`rsync` 上传也可续传。脚本只从 `~/Downloads/` 读取六个指定模型文件，不会扫描并上传其他文件。

`num2words` 需要的 `docopt==0.6.2` 在 PyPI 只有源码包，脚本会先从官方源码构建一个通用 wheel，再解析 Python 3.12/Linux x86_64 的其余依赖。离线包包含 LeRobot 的 `training` extra 和 torchao，供先 QAT、后真实量化及独立 PTQ 对照使用。

下载并上传的内容：

| 本地 `artifacts/transfer/` | GPUServer | 用途 |
| --- | --- | --- |
| `model/` | `/root/qvla/artifacts/model/` | SmolVLA checkpoint、processor、训练配置 |
| `smolvlm2_assets/` | `/root/qvla/artifacts/smolvlm2_assets/` | 基础 VLM 配置与分词器；不重复下载其 2 GB 权重 |
| `libero/` | `/root/qvla/data/libero/` | 固定修订号的完整 LIBERO 数据集 |
| `libero_assets/` | `/root/qvla/artifacts/libero_assets/` | LIBERO 闭环模拟场景、物体和纹理 |
| `wheelhouse/` | `/root/qvla/artifacts/wheelhouse/` | Python 3.12/Linux x86_64 的 PyTorch CUDA 11.8、LeRobot 训练/数据处理依赖及 torchao，供 PTQ 和 QAT 使用 |

只准备基线首条样本：

```bash
python3 scripts/download_and_upload.py --dataset smoke --skip-wheels
```

仅在本机下载、暂不上传：

```bash
python3 scripts/download_and_upload.py --skip-upload
```

## 上传后在服务器使用

脚本只下载与上传，不会自动替换服务器当前 Python 环境。服务器已有 Python 3.12 虚拟环境后，可从 wheelhouse 离线安装：

```bash
ssh -F ~/.ssh/config GPUServer
/root/qvla/.venv/bin/python -m pip install --no-index --find-links /root/qvla/artifacts/wheelhouse 'torch==2.7.1+cu118' 'torchvision==0.22.1+cu118' 'lerobot[smolvla,training]==0.6.1' 'torchao==0.11.0'
```

如虚拟环境没有 `pip`，使用服务器已安装的 `uv`：

```bash
/root/.local/bin/uv pip install --python /root/qvla/.venv/bin/python --no-index --find-links /root/qvla/artifacts/wheelhouse 'torch==2.7.1+cu118' 'torchvision==0.22.1+cu118' 'lerobot[smolvla,training]==0.6.1' 'torchao==0.11.0'
```

运行 FP 基线时传入 `--vlm-assets-dir /root/qvla/artifacts/smolvlm2_assets`，模型初始化会使用本地配置和分词器。RKNN-Toolkit2 等板端工具需等导出子图和目标系统版本确定后单独锁定；脚本目前没有下载它们。
