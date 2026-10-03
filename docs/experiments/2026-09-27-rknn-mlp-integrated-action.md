# RKNN W8A8 专家 MLP 接入完整动作推理：QAT 与 PTQ 配对

**状态**：已完成 40 个冻结开发任务的离线完整动作推理，但只有专家第 0 层的一个 MLP 使用 RKNN 主机模拟器。其余模块在 PTQ 路径为原始模型浮点运行，在 QAT 路径为训练后的浮点主权重运行。不是完整专家 INT8、RK3588 板端执行，也不是闭环成功率。

## 问题、预定判据和原理

前一实验发现同一 MLP 的 QAT 子图输出 MAE 略低于独立 PTQ，但局部误差不等于动作质量。本实验把第 0 层 `model.vlm_with_expert.lm_expert.layers.0.mlp` 的每次前向真正改为 RKNN Toolkit2 主机模拟器的 W8A8 计算，测同任务首步动作及 50×7 动作块。以原始加载 BF16 模型、同前后处理和同初始噪声为基线；模型质量优先，QAT 必须在配对任务质量上优于独立 PTQ 才能据此选用。只看均值的小差异、子图 MAE 或 MSE 不作最终选择。

QAT 来源为学习率 $10^{-7}$、第 50 步的 FP 主权重，再经独立校准和 RKNN W8A8 转换；PTQ 从原始 FP checkpoint 独立转换。两者都采用 `target_platform=rk3588`、`quantized_dtype=w8a8`、`quantized_method=channel`、`quantized_algorithm=mmse`，使用**同一** 240 条校准激活（40 个 `ptq_calibration` episode，2 帧×3 步）。量化公式、MMSE 工具配置与 INT8 权重证据见[前一子图实验](2026-09-27-stage1-rknn-qat-ptq.md)。本轮不选新的裁剪阈值，也未扫描新位宽。

主机模拟器不能重新加载已导出的 `.rknn`，故[`rknn_mlp_bridge_server.py`](../../scripts/rknn_mlp_bridge_server.py)分别从两个已固定 ONNX 和同一校准文件，按原导出配方重新构建模拟器，并核对已导出文件的 hash/字节。独立 LeRobot 进程通过本机 Unix socket 调用；[`eval_rknn_mlp_integrated_action.py`](../../scripts/eval_rknn_mlp_integrated_action.py)仅替换这个 MLP，输入先从 BF16 转 FP32，模拟器输出再转回原输入 dtype。这包含真实量化模拟器计算及 FP/INT8 边界数据转换，**没有证实导出的 `.rknn` 可在板端执行**。

## 冻结配置与复现

- 模型 `lerobot/smolvla_libero` 权重 SHA-256：`9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`；checkpoint 原处理器与本地 SmolVLM2 资产同此前 FP 基线。评估 partition SHA-256 `f755546a6b074d2fe248333fc42c3dbf30f9b54b9f0fa0b58a16506a54d942c2`；40 个 `qat_train` 开发 episode，各任务取冻结首帧。校准、开发 episode 不重叠；冻结测试 episode 未使用。完整 episode ID 与任务原始值见[动作报告](../../runs/rknn_mlp_integrated_action_v1/report.json)。
- QAT FP 主权重 SHA-256：`521e5f1968c1e74c48ad8fe5ceedeb59085baf92e441e1a7e61cfaf9fad88680`；[原动作代理](../../runs/stage1_action_proxy_v1/report.json)的 `actions.npz` 提供原始 FP 和 QAT FP 主权重基线。本轮重算各一任务的动作块，逐元素最大差均为 **0**，然后复用冻结数组。前后处理、每任务 seed 公式和动作预测均通过相同 `run_action`。
- 校准文件列表 SHA-256：`562a1ce53eb82e7689d65d6d09510ccb8eb1712e9b31caebedf7f34d1bedbfc8`。两个 ONNX SHA-256：PTQ `fb9e669995abff913834d0e0e061d787ed1442c4f640bccddf5d95d74057474b`、QAT `443c526cd4d339c7c97c282a20e28312fd610850af5ac7b366852cc5704e3648`。输入/输出为 `1×50×720`；每任务 10 个去噪步骤，故每路径 400 次模拟器调用。
- 环境：阿里云 NVIDIA A10 23 GB；LeRobot 推理用服务器 `.venv` 的 PyTorch `2.7.0a0+7c8ec84dab.nv25.03`；RKNN 侧 `.rknn-probe` 的 RKNN-Toolkit2 `2.3.2`。主机模拟器版本不能当作板端驱动/NPU 版本。

在服务器 `/root/qvla` 下先运行服务端（配置与路径是本次实际命令）：

```bash
.rknn-probe/bin/python scripts/rknn_mlp_bridge_server.py --ptq-onnx runs/qat_ptq_expert0_original/expert_layer0_mlp_fp32.onnx --ptq-rknn runs/qat_ptq_expert0_original/expert_layer0_mlp_int8_mmse_rk3588.rknn --qat-onnx runs/qat_ptq_expert0_qat50/expert_layer0_mlp_fp32.onnx --qat-rknn runs/qat_ptq_expert0_qat50/expert_layer0_mlp_int8_mmse_rk3588.rknn --dataset runs/rknn_expert_mlp_calibration_v2/rknn_dataset.txt --socket runs/rknn_mlp_bridge_v1.sock --report runs/rknn_mlp_bridge_v1_server.json
```

服务端显示 `READY_SOCKET` 后，另开终端运行：

```bash
.venv/bin/python scripts/eval_rknn_mlp_integrated_action.py --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets --dataset-root data/libero --splits data/libero_splits.json --partition config/evaluation_partition_v2.json --map config/quantization_map_qat_stage1.json --calibration-ranges runs/qat_w8a8_stage1_lr1e7/activation_ranges.json --qat-snapshot runs/qat_w8a8_stage1_lr1e7/expert_master_step_50.safetensors --proxy-report runs/stage1_action_proxy_v1/report.json --proxy-arrays runs/stage1_action_proxy_v1/actions.npz --socket runs/rknn_mlp_bridge_v1.sock --output-dir runs/rknn_mlp_integrated_action_v1 --stop-server
```

绘图/配对计算：`python3 scripts/plot_rknn_mlp_integrated_action.py --report runs/rknn_mlp_integrated_action_v1/report.json --output figures/rknn_mlp_integrated_action_v1.png --summary runs/rknn_mlp_integrated_action_v1/paired_summary.json`。

## 原始结果与计算

对任务 $i$，首步记录动作误差 $E_i=\frac17\sum_{j=1}^7|a_{i,0,j}-a^{\mathrm{recorded}}_{i,j}|$；动作块对 FP 偏差为 $D=\frac1{40\cdot50\cdot7}\sum|A-A^{\mathrm{original}}|$。配对差 $\Delta_i=E_i^{\mathrm{QAT}}-E_i^{\mathrm{PTQ}}$；负值偏向 QAT。40 任务以 seed 0、有放回任务重采样 10,000 次，取均值的百分位 95% 区间。数据为开发集描述性区间，QAT 第 50 步已用这批开发任务选择，**不是独立测试显著性结论**。

| 路径 | 首步 MAE vs 记录动作 ↓ | 动作块 MAE vs 原 FP ↓ | 动作块 MAE vs 各自浮点主权重 ↓ | 被量化 MLP 调用数 |
| --- | ---: | ---: | ---: | ---: |
| 原始加载模型 FP | 0.0309860 | 0 | 0 | 0 |
| 原始 FP → PTQ W8A8 MLP | 0.0310391 | 0.0007833 | 0.0007833 | 400 |
| QAT 第 50 步 → W8A8 MLP | 0.0309569 | 0.0008545 | 0.0006344 | 400 |

QAT−PTQ 的首步 MAE 配对均值 **−0.00008218**，40 个任务 QAT/PTQ 各占 20 个更低；任务 bootstrap 95% 区间 **[−0.00021885, 0.00004882]**，跨 0。QAT 比自身浮点主权重的单层量化动作偏差较小，但相对原始 FP 的动作块偏差比 PTQ 略大。两项指标的方向不同，不能宣称 QAT 已带来确定的动作质量提升。

[原始动作数组](../../runs/rknn_mlp_integrated_action_v1/actions.npz) SHA-256 `44cc035760331d8bd6eb16892f0245c3b57ba25ccca25a14e0fdb517d240aa97`，含原 FP、QAT 浮点、两条模拟器量化路径以及记录动作、任务/episode 索引。[逐任务报告](../../runs/rknn_mlp_integrated_action_v1/report.json)、[配对抽样结果](../../runs/rknn_mlp_integrated_action_v1/paired_summary.json)、[任务误差图](../../figures/rknn_mlp_integrated_action_v1.png)保留原始数据和全部任务；图横轴为冻结开发任务索引，上图为首步记录动作 MAE，下图为 QAT−PTQ 配对差。

两条已导出 `.rknn` 各 **4,524,893 B**，PTQ/QAT SHA-256 分别为 `9b8a2dd8aedd928512d525848415eee48d267b62551dde9ae25b765669c26839` 和 `2cdfef74ce6d632af581106e14fcee9fa794e4acb17273543d47b216d74ba659`；权重 INT8 证据见前一实验。[服务端原始报告](../../runs/rknn_mlp_integrated_action_v1/rknn_mlp_bridge_v1_server.json)记录每条配方构建约 173 秒、400 次主机模拟器推理总耗时 PTQ 2.744 秒/QAT 2.736 秒。单个 MLP 的这段模拟器耗时不含 GPU 其余模块、板端搬运和 NPU 调度。含 socket、CPU/GPU 复制及完整动作的 40 任务主机墙钟 PTQ 12.45 秒、QAT 11.86 秒，**不能作为 RK3588 延迟或 QAT 与 PTQ 的速度比较**。整模型字节、峰值 RAM、板端 p50/p95 延迟、功耗和完整收益函数 $G$ 均未测量。

## 判断与下一步

已经验证同一真实量化 MLP 可以在完整 SmolVLA 动作推理中逐去噪步参与计算，且两条配方分别跑完 40 任务；局部 QAT 数值收益未稳定转化为首步动作收益。此轮仍只有 1 个 MLP，不能定下专家其他 109 个 Linear 的位宽或宣称 stage 1 量化完成。下一步优先扩大实际量化覆盖并处理 FP/NPU 边界，按同任务动作质量筛除退化配置；完整图可执行后再使用冻结测试和多种子闭环确认，最后在 RK3588 上加载导出文件测资源与延迟。若扩大覆盖使质量下降，应保留相应模块的高精度或重设 QAT 训练约束，仍与原始 FP 的独立 PTQ 配对。
