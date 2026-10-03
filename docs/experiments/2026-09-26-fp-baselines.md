# FP 原始模型基线：离线首帧与 LIBERO 闭环

**性质**：已完成的开发基线；两个测试运行于不同服务器环境，数值不作同硬件延迟比较。冻结测试 episode 不用于选择量化参数。

## 模型、数据与测法

- checkpoint：`lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de`；模型权重 SHA-256 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`。新服务器 processor SHA-256 `122ec5106602b1bf129f49690d05ab2f49748a0ac6119de55ea0677d4e90d248`，模型/数据修订号见 [`runs/fp_libero_40x1_new/environment.json`](../../runs/fp_libero_40x1_new/environment.json)。
- 离线：冻结 `test` 划分，40 个任务各一个 episode 的首帧，共 40 条；每样本 seed 0，对数据集记录动作计算单步 MAE；运行脚本 [`scripts/fp_baseline.py`](../../scripts/fp_baseline.py)。旧服务器 PyTorch `2.7.1+cu118`、CUDA 11.8，原始报告 [`runs/fp_baseline_40tasks/report.json`](../../runs/fp_baseline_40tasks/report.json)。
- 闭环：新服务器 NVIDIA A10，PyTorch `2.7.0a0+nv25.03` / CUDA 12.8，LeRobot 0.6.1；LIBERO 四套件各 10 任务、每任务 1 回合、seed 0、固定初始状态，图像 256×256、双相机映射、relative control。原始逐任务报告 [`runs/fp_libero_40x1_new/report.json`](../../runs/fp_libero_40x1_new/report.json)。

## 结果与图

| 测试 | 结果 | 可解释范围 |
| --- | ---: | --- |
| 闭环 Spatial / Object / Goal / LIBERO-10 | 9/10 · 9/10 · 7/10 · 6/10 | 每任务只测 1 回合 |
| 闭环合计 | **31/40 = 77.5%** | 单 seed、单初始状态开发探针 |
| 离线单步动作 MAE vs 记录动作 | **0.03610** | 40 条首帧平均；不是成功率 |
| 旧服务器动作推理 p50 / p95 | **230.4 / 233.1 ms** | 与新服务器闭环耗时不可直接比较 |
| 旧服务器峰值 CUDA allocated | **1,264,298,496 B** | PyTorch allocated，非整机显存峰值 |
| 原始模型权重文件 | **906,712,520 B** | 未计 processor、运行依赖 |

- [闭环套件成功数与 40 个任务结果图](../../figures/fp_libero_40x1.png)；绘图脚本 [`scripts/plot_fp_libero_baseline.py`](../../scripts/plot_fp_libero_baseline.py)，直接读取逐任务 JSON 并检查汇总一致性。
- [离线逐任务动作 MAE 与延迟图](../../figures/fp_offline_40.png)；绘图脚本 [`scripts/plot_fp_offline_baseline.py`](../../scripts/plot_fp_offline_baseline.py)。第一条推理为 456.4 ms；图显示了该点，不将其删除。原始报告的 p50/p95 按全部 40 条计算。

## 结论边界与后续

FP 模型在当前链路能完成闭环任务，已有可用于量化配对比较的输入和初始状态；**31/40 不是正式多种子成功率**。正式比较仍需按[固定评测协议](../evaluation-protocol.md)扩充相同初始状态、回合数和种子，并在与量化模型相同环境重新测推理延迟。

后续为真实 W8A8 比较重新运行了 FP，并在每个任务固定相同的策略随机种子，得到 **32/40**（LIBERO-10 为 7/10）；同期 PTQ/QAT 配对结果见[新记录](2026-09-27-paired-w8a8-rollouts.md)。相较本页原 31/40 基线，单回合探针有一个任务结果不同；两次运行不合并、不以此估计种子波动。新配对批次内部以 32/40 的 FP 为对照，仍须扩展多回合、多种子。
