# QVLA：SmolVLA硬件感知混合精度量化与RK3588部署

从原始FP基线、HAQ风格强化学习搜索、OpenVLA-OFT动作蒸馏，到混合精度QAT、独立PTQ与真实RK3588整策略部署。语言使用RKLLM，视觉与动作专家使用RKNN，CPU执行embedding及前后处理。

## 实测结果

| 指标 | 原模型/板端基线 | 终版 |
| --- | ---: | ---: |
| 核心参数文件 | 原checkpoint906.71MB | 588.04MB（减少35.15%） |
| 同板进程树峰值PSS | 1824.4MiB | 1129.9MiB（减少38.07%） |
| 同板完整50步动作块P50 | 7.5199s | 7.2626s |
| LIBERO 40任务成功数 | 原GPU30/40 | 板端25/40 |
| Object / Goal成功数 | 8/10、7/10 | 8/10、7/10 |

固定40任务各一个初始状态，部分任务曾用于开发/选择，不是独立多种子统计。总体质量下降，不能称无损量化。HAQ搜索采用REINFORCE、真实离线动作反馈与板测查表；不是原论文DDPG复现，也没有逐候选整策略板测闭环。终版采用原FP16视觉和后端精度适配，不是V1搜索图的严格等价转换。

## 文档

- [技术路线与完整实验记录](docs/project-route.md)：原理、配置、决策及有效/失败实验。
- [结果](docs/results.md)：终版对比、任务明细、历史候选与审计结论。
- [部署](docs/deployment.md)：环境、模型校验、板端运行和复现入口。

## 目录

```text
qvla/           # data / hardware / haq / distillation / quantization /
                # conversion / runtime / evaluation
scripts/        # 9个主命令入口
config/         # 精度图、训练/数据配置、硬件成本表
models/final/   # 唯一提交的模型：终版RKNN/RKLLM/CPU参数和processor
docs/          # 路线、结果、部署说明及images
```

`artifacts/`、`data/`、`runs/`和第三方SDK/环境仍保留在本地，但不提交。辅助探针、诊断、绘图和自检归入对应功能包；不设archive、tools、results文件夹或独立tests目录。

## 获取模型与校验

模型权重使用Git LFS。已克隆仓库时：

```bash
git lfs install
git lfs pull
python3 scripts/deploy.py verify
```

`models/final/`包含真实文件，不依赖原板端`/dev/shm`符号链接。`deployment_manifest.json`记录图及运行接口；`release.json`记录文件hash和运行版本。官方runtime及自写native组件按部署说明准备。

## 主命令

在项目根目录执行；`--help`查看阶段列表，`阶段 --help`查看参数。

| 入口 | 用途 |
| --- | --- |
| `scripts/prepare_data.py` | 下载、划分、仿真准备 |
| `scripts/benchmark_hardware.py` | 支持格式、硬件表和板测 |
| `scripts/search.py` | HAQ风格RL搜索与随机对照 |
| `scripts/distill.py` | 教师输入/标签、审查与FP蒸馏 |
| `scripts/quantize.py` | 按配置QAT、独立FP PTQ |
| `scripts/convert.py` | 视觉/专家RKNN及语言RKLLM转换 |
| `scripts/deploy.py` | 模型hash校验、传输和板端推理 |
| `scripts/evaluate.py` | GPU/板端质量、资源和结果生成 |
| `scripts/selfcheck.py` | 冻结、教师监督与checkpoint契约自检 |

历史实现可通过`python -m qvla.<功能>.<模块>`运行；迁移后的源码hash不同于实验时冻结的源码hash，原版本保存在Git历史。
