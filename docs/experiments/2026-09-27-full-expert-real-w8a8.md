# 112 个动作专家 Linear：真实 GPU W8A8 QAT/PTQ 对照

**范围**：真实 SmolVLA checkpoint 的 112 个动作专家 Linear；QAT 后 FP 主权重与原始 FP checkpoint 分别保存为含 INT8 权重的整模型文件，严格重新加载，在 A10 的 CUDA 整数矩阵乘上完成 40 个冻结开发任务的动作推理。视觉语言主干仍为原精度；本结果不是 RK3588 NPU 全图，也不是闭环任务质量。

## 问题和方案

此前只将一个专家 MLP 的 RKNN 模拟器接入动作。为在服务器有效期内尽早验证**整个专家量化**，本轮按已锁定的 [stage 1 图](../../config/quantization_map_qat_stage1.json)，对专家 112 个 Linear 分别做两条 W8A8 路径：

1. QAT：使用学习率 $10^{-7}$、第 50 步的 FP 主权重，训练过程为静态激活/逐输出通道权重 fake quant、STE；这里只把训练后的浮点主权重转换成真正的整数权重。
2. PTQ：独立从原始 checkpoint 权重出发，用相同模块范围、相同位宽和同一组已冻结的激活范围转换。没有把 QAT 低比特产物再次 PTQ。

`pack_real_w8a8_expert.py` 对每个输出通道 $j$ 取 $s_{w,j}=\max_i|W_{j,i}|/127$，保存 $W^q_{i,j}=\operatorname{clip}(\operatorname{round}(W_{j,i}/s_{w,j}),-127,127)$ 为 **INT8**，按列存储以供整数 GEMM。激活沿用同一[校准范围](../../runs/qat_w8a8_stage1_lr1e7/activation_ranges.json)：$s_x=(x_{max}-x_{min})/255$，$z_x=\operatorname{clip}(\operatorname{round}(-128-x_{min}/s_x),-128,127)$，推理时 $x^q=\operatorname{clip}(\operatorname{round}(x/s_x)+z_x,-128,127)$。实际前向调用 `torch._int_mm(x^q,W^q)`，得到 INT32 累积，再用 $z_x\sum_iW^q_{i,j}$ 修正零点、乘 $s_xs_{w,j}$、加浮点 bias 并转换到原计算 dtype；见[`RealInt8Linear`](../../scripts/real_int8_linear.py)。这是**真整数矩阵乘**，不是 fake quant；其运行后端是 A10 CUDA，不能当 RKNN NPU 实测。

## 来源、环境、复现

- 模型 `lerobot/smolvla_libero` 原权重 SHA-256 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`；processor 与此前 FP 基线相同。QAT 主权重 SHA-256 `521e5f1968c1e74c48ad8fe5ceedeb59085baf92e441e1a7e61cfaf9fad88680`。冻结 partition SHA-256 `f755546a6b074d2fe248333fc42c3dbf30f9b54b9f0fa0b58a16506a54d942c2`。
- 40 个 `ptq_calibration` episode 的静态输入范围来自前述 QAT 校准；训练排除了 40 个开发 episode。评估用另外 40 个 `qat_train` 开发 episode，每任务首帧、相同 processor、相同由 episode/任务确定的随机噪声。冻结测试 episode 未使用；该第 50 步候选已在开发集选择，配对区间仅是描述性结果。
- 环境：服务器 NVIDIA A10 23 GB，PyTorch `2.7.0a0+7c8ec84dab.nv25.03`，LeRobot `0.6.1`。`torch._int_mm` 已在此 A10 上验证返回 CUDA INT32 矩阵。量化权重作为 `safetensors` 持久化，重新构造 112 个 `RealInt8Linear` 后 `load_state_dict(strict=True)`；这样既检查整模型键/shape，也避免只加载文件即宣称完成量化。

服务器 `/root/qvla` 下运行的命令：

```bash
.venv/bin/python scripts/pack_real_w8a8_expert.py --model-dir artifacts/model --qat-snapshot runs/qat_w8a8_stage1_lr1e7/expert_master_step_50.safetensors --calibration-ranges runs/qat_w8a8_stage1_lr1e7/activation_ranges.json --partition config/evaluation_partition_v2.json --map config/quantization_map_qat_stage1.json --output-dir runs/expert_real_w8a8_v1
.venv/bin/python scripts/eval_real_w8a8_expert.py --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets --dataset-root data/libero --splits data/libero_splits.json --partition config/evaluation_partition_v2.json --map config/quantization_map_qat_stage1.json --pack-report runs/expert_real_w8a8_v1/report.json --output-dir runs/expert_real_w8a8_action_v1 --max-tasks 40
```

## 文件与动作原始结果

| 路径 | 模型文件字节 | 相对原始文件 | 首步动作 MAE 对记录动作 ↓ | 50×7 动作块 MAE 对原 FP ↓ | A10 动作 p50 / p95 秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原始 FP | 906,712,520 | 1.000 | 0.0309860 | 0 | 0.2403 / 0.2860 |
| 原始 FP → PTQ W8A8 | 806,645,032 | 0.8896 | 0.0309246 | 0.0019563 | 0.3626 / 0.4808 |
| QAT 第 50 步 → W8A8 | 806,645,032 | 0.8896 | 0.0306452 | 0.0024068 | 0.3623 / 0.3790 |

文件仅缩小 **11.04%**，因为未量化的视觉语言主干仍占多数。两份 checkpoint 的 SHA-256 分别为 PTQ `e440470ff7285478eedad040d116b5ec7c0b4701b9efe6f99189658c2140e522`、QAT `95ee4b6242ccf4c1124c4e50219f24f859030281a697056e3f04ffb150459b0b`；精确量化规则、112 个模块名、文件与元数据大小见[打包报告](../../runs/expert_real_w8a8_v1/report.json)。文件体积是真实整模型文件，不含 processor/依赖，因此完整部署包 $B$ 尚未测量。

QAT−PTQ 的首步记录动作 MAE 按 40 任务配对均值 **−0.0002794**，QAT 在 24 个任务更低、PTQ 在 16 个任务更低；seed 0 的 10,000 次任务 bootstrap 百分位 95% 区间 **[−0.0005802, +0.0000124]** 跨 0。首步离线记录动作误差略低不等于闭环成功率提高；QAT 的动作块**对原 FP 更偏离**，因此当前不能判定 QAT 质量优于 PTQ，也不能据此确认此 stage 1 图。逐任务动作、时间和来源见[评估报告](../../runs/expert_real_w8a8_action_v1/report.json)及[原始动作数组](../../runs/expert_real_w8a8_action_v1/actions.npz)，数组 SHA-256 `775f9a879d30216cb2a20068eb2cec961d0b682a53003a5bba738fd0455fcc78`。[40 任务误差图](../../figures/expert_real_w8a8_action_v1.png)及[配对差原始数据](../../runs/expert_real_w8a8_action_v1/paired_summary.json)由[`plot_real_w8a8_expert_action.py`](../../scripts/plot_real_w8a8_expert_action.py)生成；图横轴为冻结开发任务，上图是首步记录动作 MAE，下图是 QAT−PTQ 的逐任务差。

A10 上本实现逐层做激活量化、INT8 GEMM、反量化，所以 p50 相比 FP 更慢约 51%；这只说明**此 CUDA 实现**没有加速，不能外推 RK3588。QAT 与 PTQ 的 p95 差异来自单轮 40 任务顺序运行，未做重复时延试验，不当作二者的速度结论。峰值 RAM、板端 p50/p95、能耗、正式 FP 多种子波动及收益函数 $G$ 均**未测量**。

## 边界和下一步

已完成 112 个专家 Linear 的真实 GPU W8A8 推理与独立 PTQ 对照；RKNN 侧仍须验证更多专家子图、完整注意力/缓存/残差及 CPU/NPU 边界。下一步按同任务质量保留或提高敏感模块精度，做开发闭环淘汰；只有完整可执行候选确定后才使用冻结测试和 RK3588 板端反馈决定最终位宽。当前质量优先规则下，**不能把较低离线首步 MAE 当作已达标**。

之后已完成三种模型的 LIBERO 配对闭环开发筛查：见[配对 rollout 记录](2026-09-27-paired-w8a8-rollouts.md)。在四 suite × 10 任务 × 1 回合下，FP 为 32/40，PTQ 与 QAT 各 31/40；每个量化模式相对本轮配对 FP 各有一个不同任务失败。此结果每任务只有一个初始状态，不构成质量通过，亦不改变上面对完整 RKNN 与板端资源仍待测的边界。
