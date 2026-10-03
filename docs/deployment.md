# 安装、模型与板端运行

本文描述整理后的入口。模型结果见[results.md](results.md)，转换技术和历史精确配置见[project-route.md](project-route.md)。不需要重新训练就能运行终版。

## 1. 已验证的环境

| 环节 | 环境 |
| --- | --- |
| 本地GPU测试/自检 | Python3.12、Torch2.7.1+cu118、LeRobot0.6.1、NumPy2.2.6、Transformers5.5.4 |
| RKNN转换 | 独立`esp-ml`环境，RKNN-Toolkit2 2.3.2 |
| RKLLM转换 | 独立Toolkit1.3.1环境 |
| 板端 | RK3588 4GB、RKNN runtime/lite2 2.3.2、driver0.9.8、RKLLM1.3.1及本项目固定版本补丁 |
| 板端Python | NumPy、tokenizers0.22.2、RKNN-Toolkit-Lite2；已有环境位于`/root/qvla_board_test/python_site` |

训练、教师、两个转换Toolkit与板端环境分开；`pyproject.toml`提供功能依赖范围，不是完整硬件环境lock。原模型版本/修订号在`config/step1.lock.json`，教师环境见`config/teacher_inference_lock.txt`。沿用已验证环境时可用`python -m pip install --no-deps -e .`注册源码包，直接执行scripts入口则无需先安装本包。

## 2. 终版模型文件

只有`models/final/`提交模型，权重通过Git LFS管理。

```text
models/final/
├── vision_connector_fp16.rknn
├── language_w8a8.rkllm
├── expert_before_projection.rknn
├── expert_projection.rknn
├── expert_after_projection.rknn
├── cpu_weights.npz
├── state_stats.npz
├── tokenizer.json
├── tokenizer_config.json
├── policy_preprocessor.json
├── config.json
├── replay.json
├── preprocess_export.json
├── deployment_manifest.json
└── release.json
```

核心参数合计588,044,896字节，另有processor与清单。视觉来自原checkpoint FP16，语言来自V1无教师QAT master的RKLLM W8A8转换，专家使用混合RKNN，CPU参数包含INT8 embedding和前后处理参数。`cpu_weights.npz`不是教师或训练断点。图与CPU权重hash固定，不能随意替换组件。

```bash
git lfs install
git lfs pull
python3 scripts/deploy.py verify
```

校验无需GPU/NPU，会拒绝缺失文件、错误hash及未拉取的LFS指针。

## 3. Native运行组件

模型之外仍需要以下运行组件：

- `rkllm_worker`：由`qvla/runtime/smolvla_rkllm_worker.cpp`构建的AArch64常驻语言进程。
- `libqvla_rknpu_alloc.so`：由`qvla/runtime/probe_rknpu_allocation.c`构建，限定工作池大小的WC映射适配。
- `librknn_bf16_projection.so`：由`qvla/runtime/rknn_bf16_projection.c`构建的BF16接口。
- `patched_lib/librkllmrt.so`：固定RKLLM1.3.1原始库及本项目native mask补丁，官方二进制不放Git。

这台机器的已验证组件已从板端收集到Git忽略的`artifacts/runtime/final/`。worker、补丁库和allocation库身份在部署清单中；全套已收集文件hash在`release.json`。首次换机器需要从相同SDK构建/准备组件，不能只凭语言模型文件就运行；二进制ABI与内部偏移绑定版本。native mask构建入口是`qvla/runtime/patch_rkllm_block_mask_probe.py --aligned`，先校验源库SHA再修改；原理与定位记录见路线文档。

## 4. 传到板子

本地准备好的运行组件和模型都已验证后：

```bash
python3 scripts/deploy.py verify --runtime-dir artifacts/runtime/final
python3 scripts/deploy.py stage --board root@10.42.0.252 --destination /root/qvla
```

命令传输源码包、部署入口、终版模型及native组件。目标目录需要足够空间；当前板子根分区空间紧张，使用前确认容量，也可指定持久数据分区。`/dev/shm`只适合临时验证，重启会丢失。

已有模型且只更新代码可使用`--code-only`；它不会复制模型/native组件。代码需要与模型一起按下节路径运行。

## 5. 在板子上运行

以下在板子的项目根目录执行；按实际安装位置设置Python依赖路径：

```bash
set -lx OPENBLAS_NUM_THREADS 1
set -lx PYTHONPATH /root/qvla_board_test/python_site
python3 scripts/deploy.py infer --root models/final --input observation.npz --output actions.npz
```

上面环境设置是fish语法；bash使用`export OPENBLAS_NUM_THREADS=1`、`export PYTHONPATH=/root/qvla_board_test/python_site`。

`observation.npz`包含：

| 键 | 格式 |
| --- | --- |
| `image1`、`image2` | 原始RGB uint8 CHW图像 |
| `state` | 8维原始机器人状态 |
| `task` | 标量任务字符串 |
| `noise` | 固定初始噪声，float32 `[1,50,32]` |

输出`actions.npz`包含`actions`，shape `[1,50,7]`。一次前向生成动作块，不等于真实机器人已经执行任务。

长驻服务：

```bash
python3 scripts/deploy.py serve --root models/final
```

服务使用`QVLA_REPLY` JSON行协议，通过stdin传入包含上述NPZ的base64请求；GPU端LIBERO评测负责环境推进，板端服务只生成动作。

## 6. 搜索、训练、转换与评价入口

使用相应GPU/Toolkit环境，在项目根目录查看所需参数：

```bash
python scripts/search.py run --help
python scripts/distill.py train --help
python scripts/quantize.py qat --help
python scripts/quantize.py ptq --help
python scripts/convert.py language --help
python scripts/evaluate.py board --help
python scripts/selfcheck.py
```

QAT默认配置为`config/qat_distilled_v1_no_teacher_v1.json`，可用`--config`选择V2/教师对照；默认流程包含训练、打包及19任务开发评价，需要已有数据、学生和相应缓存。`quantize.py ptq`使用传入候选配置从原FP/学生FP独立量化，无优化器。底层模块名含v2或first8是历史命名，实际范围与精度由配置决定。运行时间、默认旧run路径及缓存要求见路线记录。

资源测量在板端执行`evaluate.py resources`对应实现，质量评价在GPU环境执行`evaluate.py board`，并明确`--board-root`。整策略延迟与算子查表成本分开。

## 7. 本次整理验证

- 终版板端模型清单与本地收集文件hash匹配。
- 12项训练契约自检通过：冻结范围、checkpoint恢复、GT/教师mask、学习率与flow监督等。
- 迁移后的完整板端前向成功，执行2次视觉、1次RKLLM语言、30次专家子图。
- 同一原始NPZ输入，旧平铺runtime与新功能包输出逐元素完全一致，最大差0；不是重新40任务评测。

历史source hash对应整理前版本，Git提交`22fac9d`保存旧结构。测试期间未改模型权重，也未覆盖板端已有部署。
