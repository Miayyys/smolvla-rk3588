# 视觉/语言 MLP 输出截断：抽样 MSE 与逐元素复核

**结论范围**：在两个高动作敏感层组上，按抽样激活 MSE 选阈值会漏掉极端值；用全部校准输出逐元素复核时，14 个模块均选择 min/max。此结论只针对当前候选阈值、per-tensor 输出 fake INT8 和采样方式；不是 RKNN 内部算法、真实低比特模型或闭环成功率结果。

## 假设、公式与配置

[层组敏感度实验](2026-09-27-action-sensitivity.md)发现视觉层 6–11 和语言层 0–7 的 per-tensor min/max 范围很大，怀疑少数离群值把 INT8 步长拉大。本轮沿用完全相同的原始模型、checkpoint processor、40 个独立校准 episode、40 个开发 episode、每 episode 首末两帧和固定噪声；只更改这 14 个 MLP 输出的截断范围。原始 FP 动作 chunk 在 min/max 和 MSE 两轮**逐值完全相同**（最大绝对差 0）。未使用 213 个冻结测试 episode。实现见[`probe_action_sensitivity.py`](../../scripts/probe_action_sensitivity.py)。

对每个模块输出先在校准观测中记录真实 min/max。候选保留比例为 `p∈{98%,99%,99.5%,99.9%,99.95%,99.99%,100%}`；`p<100%` 的上下阈值由抽样分布的 `(1-p)/2` 与 `1-(1-p)/2` 分位数给出，`p=100%` 使用真实 min/max。每个候选按 `s=(high-low)/255`、`z=clip(round(-128-low/s),-128,127)`、`q=clip(round(x/s+z),-128,127)`、`x̂=(q-z)s` 计算重建 MSE。两个实验共享**同一候选阈值表**，区别仅在选择依据：

1. **抽样 MSE**：每次 MLP 调用从展平的输出以固定步长取最多 2048 个值，汇总后以抽样 MSE 选阈值；原始[报告](../../runs/action_sensitivity_mse_v1/report.json)。
2. **逐元素 MSE**：重新运行同一 80 个校准观测，直接遍历每次调用的所有输出值，对每个候选累加平方误差、实际饱和数和元素数，选完整校准集 MSE 最小者；原始[报告](../../runs/action_sensitivity_mse_exact_v1/report.json)。

两轮都在同一 80 个开发观测上比较 postprocessor 后完整 50×7 动作 chunk 与 FP，动作输出保存在各自 Git 忽略的 `runs/` 目录。候选逐层曲线及组级配对动作结果见[复核图](../../figures/action_clipping_validation_v1.png)（[SVG](../../figures/action_clipping_validation_v1.svg)）、[绘图数据](../../figures/action_clipping_validation_v1.json)和[`绘图脚本`](../../scripts/plot_action_clipping_validation.py)；首次抽样曲线另见[探索图](../../figures/action_clipping_compare_v1.png)。

## 阈值选取和效果

| 被扰动层组 | min/max 动作 chunk MAE vs FP | 抽样 MSE 所选动作 MAE | 逐元素 MSE 所选动作 MAE | 逐元素 MSE 最优阈值 |
| --- | ---: | ---: | ---: | --- |
| 视觉 6–11 | 0.030160 | 0.028246（80 帧中 48 帧更接近 FP） | **0.030160** | 6/6 模块都选 100% min/max |
| 语言 0–7 | **0.018173** | 0.024458（仅 8/80 帧更接近 FP） | **0.018173** | 8/8 模块都选 100% min/max |

抽样选择让视觉组动作相对 FP 的平均 MAE 降低约 **6.35%**，但首动作对示范动作的平均 MAE 增量从 min/max 的 **+0.013678** 变为 **+0.015837**，无法称作质量改进；语言组则让完整动作 MAE 增加约 **34.59%**。逐元素 MSE 选回 min/max，动作结果与初始实验相同。这一结果说明，单看抽样激活重建误差或单看动作对 FP 的接近程度，都不足以宣布闭环质量达标。

一个可核对的反例是**视觉第 11 层 MLP 输出**：抽样最小 MSE 选 `p=99.99%`，阈值 `[-5.64649, 3.47608]`，`s=0.03577481`、`z=30`。抽样重建 MSE 仅 `0.00011648`，而 min/max 阈值 `[-676.87152,569.12146]` 的抽样 MSE 为 `1.28005`；但对**全部**校准元素重算，前者 MSE 为 **117.21888**、真实饱和比例 **4.3748%**，后者 MSE 仅 **1.38026**、饱和为 0。`99.99%` 是从抽样值求得的分位范围，**不代表**完整激活中只有 0.01% 被截断。语言第 0 层也出现同样逆转：抽样选择 `p=99.95%` 的 MSE 为 `0.001393`，全量 MSE 却为 `1.31215`；其 min/max 的全量 MSE 为 `0.013156`。

运行命令（服务器 `/root/qvla`）：

```bash
.venv/bin/python scripts/probe_action_sensitivity.py --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets --dataset-root data/libero --splits data/libero_splits.json --partition config/evaluation_partition_v2.json --output-dir runs/action_sensitivity_mse_v1 --frames-per-task 2 --max-tasks 40 --groups vision_6_11 language_0_7 --calibration-method mse
.venv/bin/python scripts/probe_action_sensitivity.py --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets --dataset-root data/libero --splits data/libero_splits.json --partition config/evaluation_partition_v2.json --output-dir runs/action_sensitivity_mse_exact_v1 --frames-per-task 2 --max-tasks 40 --groups vision_6_11 language_0_7 --calibration-method mse_exact
```

## 下一步和边界

**本轮不选用抽样 MSE 阈值。** 下一次分布校准需使用全量或能保证覆盖离群值的统计方式，探索后端支持的更细粒度、可折叠 outlier 平滑或保留敏感模块 FP16，然后在完整动作与板端重新比较。逐元素 MSE 只在当前 7 个分位候选中选择 min/max，不能推断所有可能阈值都无效。这里的 fake quant 未改权重，不能称作已完成模型量化；实际 INT8/FP16 编译、闭环成功率和 RK3588 资源均未测量。
