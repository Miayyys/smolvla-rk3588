# Stage 1 全动作配对诊断：原权重与 QAT 假量化

**状态**：40 个开发任务、同输入同随机噪声的完整 50×7 动作及首步数据集动作误差已测。这里运行的是训练用 fake quant 和浮点主权重，**不是完整 RKNN 真实量化模型或闭环质量评测**。

## 问题与固定条件

[第 0 层专家 MLP](2026-09-27-stage1-rknn-qat-ptq.md)的 QAT 子图数值略有收益，但[Q/K/V 联合子图](2026-09-27-expert-attention-hybrid.md)的 QAT 更差。局部误差是否与完整动作质量方向一致？在原始 FP、原权重 fake W8A8、QAT 第 50 步 fake W8A8 与关闭 fake quant 的 QAT 浮点主权重之间做配对比较。原权重 fake W8A8 是**诊断对照**，不是已转换的 RKNN PTQ。QAT 快照是用这 40 个开发任务上的动作偏差选择的，因此结果受选择影响，不能视为独立确认。

模型原始权重 SHA-256 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`，QAT 第 50 步专家主权重 SHA-256 `521e5f1968c1e74c48ad8fe5ceedeb59085baf92e441e1a7e61cfaf9fad88680`。stage 1 图 SHA-256 `9efc34d153629a19b178bc27834554603a8b42e3d3f7141ea98d1e7b309debdb`；分区 SHA-256 `f755546a6b074d2fe248333fc42c3dbf30f9b54b9f0fa0b58a16506a54d942c2`。使用训练时同一份 40 个独立 `ptq_calibration` episode 测出的 112 个 Linear 输入 min/max；[训练报告](../../runs/qat_w8a8_stage1_lr1e7/report.json)和同目录的 `activation_ranges.json` 固定范围。40 个 `qat_train` 开发 episode 每任务首帧，均未用于 QAT 参数更新；固定 checkpoint processor、seed 规则和 10 个 flow 步骤。冻结测试未使用。

[`eval_stage1_qat_ptq_action_proxy.py`](../../scripts/eval_stage1_qat_ptq_action_proxy.py)从**同一个原始模型实例**依次运行原始 FP、插入 112 个原权重 fake-quant Linear、载入 QAT 第 50 步主权重后的 fake quant，再关闭 fake quant；每个 episode 由[`run_action`](../../scripts/probe_action_sensitivity.py)重设动作噪声。A10 / PyTorch `2.7.0a0+7c8ec84dab.nv25.03`。完整命令在服务器 `/root/qvla` 下为：

```bash
.venv/bin/python scripts/eval_stage1_qat_ptq_action_proxy.py --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets --dataset-root data/libero --splits data/libero_splits.json --partition config/evaluation_partition_v2.json --map config/quantization_map_qat_stage1.json --calibration-ranges runs/qat_w8a8_stage1_lr1e7/activation_ranges.json --qat-snapshot runs/qat_w8a8_stage1_lr1e7/expert_master_step_50.safetensors --output-dir runs/stage1_action_proxy_v1
```

运行在写入完整数据后才遇到终端摘要打印参数错误，脚本已修正。保存的[原始报告](../../runs/stage1_action_proxy_v1/report.json)和[四种完整动作数组](../../runs/stage1_action_proxy_v1/actions.npz)存在且 SHA-256 在报告中相互校验；错误**没有影响模型前向或保存的指标**。修正后的脚本通过静态编译检查；未为终端打印重新消耗服务器进行相同的 40×4 前向。

## 计算与结果

每个候选对同一原始 FP 动作 chunk 计算 $\operatorname{mean}_{t,d}|a^{candidate}_{t,d}-a^{FP}_{t,d}|$；首步数据集动作误差为 $\operatorname{mean}_{d}|a^{candidate}_{0,d}-a^{recorded}_{d}|$，每任务各有 1 个开发观测。首步记录动作只是离线代理，**不能表示闭环成功率**。[逐任务原始数据、固定 seed 0 的 10,000 次任务 bootstrap、图数据](../../figures/stage1_action_proxy_v1.json)由[`summarize_stage1_action_proxy.py`](../../scripts/summarize_stage1_action_proxy.py)从保存动作重算；[逐任务图](../../figures/stage1_action_proxy_v1.png)、[SVG](../../figures/stage1_action_proxy_v1.svg)。

| 完整动作状态 | 相对原始 FP 的 chunk MAE | 首步对记录动作 MAE |
| --- | ---: | ---: |
| 原始 FP | 0 | 0.030986 |
| 原权重 fake W8A8 | 0.001942 | 0.031081 |
| QAT 第 50 步 fake W8A8 | **0.002364** | **0.030831** |
| QAT 第 50 步浮点主权重，无 fake quant | 0.000560 | 0.030937 |

QAT fake W8A8 的首步误差相对原权重 fake W8A8 的任务平均差为 `−0.0002504`，26/40 个任务更低；任务 bootstrap 描述性 95% 区间 `[−0.0005422,+0.0000418]` **跨 0**。相对原始 FP 的均值差为 `−0.0001552`，24/40 个任务更低，区间 `[−0.0004773,+0.0001839]` 同样跨 0。故此数据不支持宣称 QAT 已改善动作质量；它说明仅按“与 FP 的 chunk MAE 最小”选择会漏掉另一种质量代理，后续仍应以闭环质量为准。

## 结论边界与下一步

QAT 浮点主权重和 fake quant 的作用可分辨：训练后未量化权重使完整动作相对 FP 仅偏 0.000560，而训练假量化后的动作偏差 0.002364；训练得到的变更没有明确提高首步数据集动作质量。由于 RKNN MMSE 内部阈值与训练 min/max fake quant 不完全一致，**不能把这些完整动作数据当作真实 RKNN QAT/PTQ 的输出**。选择性开发评估、每任务仅一帧、记录动作与闭环目标不等价，也限制了统计结论。下一步是处理完整动作图的 RKNN 执行边界并做真实量化后的动作/闭环配对；若模型质量下降，应调整或缩小量化范围而非为体积强行接受。当前完整模型部署包、板端 RAM、p50/p95 延迟和收益函数 $G$ 均**未测量**。
