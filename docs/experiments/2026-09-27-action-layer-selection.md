# 视觉与语言 MLP 的逐层定位及组合复核

**实验性质**：原始 FP SmolVLA 上的静态、逐张量、输出激活 INT8 舍入诊断。权重仍为 FP，尚无真实量化模型、RKNN 执行或闭环结果。目标是把[层组敏感度](2026-09-27-action-sensitivity.md)中视觉后段、语言前段的动作偏差定位到具体层，再验证排除高敏感层后的组合效应。

## 原理、配置和判据

每个 MLP 输出的校准范围来自 40 个 `ptq_calibration` episode、每个首末两帧；在与之隔离的 40 个 `qat_train` 开发 episode、每个首末两帧上比较完整 50×7 动作 chunk。所有实验使用同一输入、processor、模型、10 个 flow-matching 步骤和固定初始噪声。每次只对所列模块输出做 `s=(max-min)/255`、`z=clip(round(-128-min/s),-128,127)`、`q=clip(round(x/s+z),-128,127)`、`x̂=(q-z)s`。排序指标为开发集 `mean(|A_probe-A_FP|)`；另记录首动作对示范动作的 MAE 增量。组合验证要看排除敏感层后该指标是否下降；不以该代理指标判定最终 FP16/INT8。

固定模型 `lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de`，权重 SHA-256 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`；数据划分 SHA-256 `ca851a1bdc8fd60ad1e5b8d08dc7c405f971a8d4f999c4f0c2ecef154272d55f`，分区 SHA-256 `f755546a6b074d2fe248333fc42c3dbf30f9b54b9f0fa0b58a16506a54d942c2`。episode 明细、每层校准上下界、scale、zero point、逐观测动作偏差在原始报告中。GPUServer 为 NVIDIA A10；运行时 PyTorch `2.7.0a0+7c8ec84dab.nv25.03`、CUDA `12.8`。脚本为[`probe_action_sensitivity.py`](../../scripts/probe_action_sensitivity.py)；两次报告的 FP 重复执行最大差均为 0.0。

## 逐层结果

各行只扰动一个 MLP 输出，数值是 80 个开发观测的平均完整动作 chunk MAE 对 FP；图见[逐层图](../../figures/action_layer_sensitivity_v1.png)（[SVG](../../figures/action_layer_sensitivity_v1.svg)），[绘图数据](../../figures/action_layer_sensitivity_v1.json)与[脚本](../../scripts/plot_action_layer_sensitivity.py)。原始逐样本记录在[`runs/action_sensitivity_layer_v1/report.json`](../../runs/action_sensitivity_layer_v1/report.json)。

| 视觉 MLP 层 | 动作 chunk MAE | 语言 MLP 层 | 动作 chunk MAE |
| ---: | ---: | ---: | ---: |
| 6 | 0.001519 | 0 | 0.003188 |
| 7 | 0.000834 | 1 | 0.001687 |
| 8 | 0.000747 | 2 | 0.001337 |
| 9 | 0.000645 | **3** | **0.017307** |
| 10 | 0.000676 | 4 | 0.000768 |
| **11** | **0.029900** | 5 | 0.000714 |
|  |  | 6 | 0.000828 |
|  |  | 7 | 0.000723 |

视觉 11 与语言 3 分别贡献该实验设置下的最大单层偏差。它们是优先保留精度、检查校准与真实 RKNN 误差的候选层；单层影响不能简单相加预测组合影响。

## 组合复核

原层组与排除高敏感层的组合在相同 80 个开发观测上配对比较。每行的参数量仅表示所扰动 MLP 的 FP 参数数量，**不是**量化节省的字节数。图见[组合复核](../../figures/action_rescue_v1.png)（[SVG](../../figures/action_rescue_v1.svg)），[绘图数据](../../figures/action_rescue_v1.json)、[脚本](../../scripts/plot_action_rescue.py)，逐观测数据在[`runs/action_sensitivity_rescue_v1/report.json`](../../runs/action_sensitivity_rescue_v1/report.json)及[原层组报告](../../runs/action_sensitivity_v1/report.json)。

| 输出舍入范围 | MLP 参数量 | 动作 chunk MAE vs FP | 首动作对示范 MAE 增量 |
| --- | ---: | ---: | ---: |
| 视觉 6–11 | 28,334,592 | 0.030160 | +0.013678 |
| 视觉 6–10，排除 11 | 23,612,160 | **0.001706** | -0.000043 |
| 语言 0–7 | 58,982,400 | 0.018173 | +0.000877 |
| 语言 0–2、4–7，排除 3 | 51,609,600 | **0.004159** | +0.000445 |
| 上述视觉与语言 12 层同时扰动 | 75,221,760 | **0.004317** | +0.000537 |

排除视觉 11 后动作偏差减少约 94.3%；排除语言 3 后减少约 77.1%。在逐任务配对汇总中，这两项各自均为 40/40 任务下降。组合 12 层的偏差 0.004317 仍高于零；首动作对示范 MAE 的细小变化不等价于成功率变化，也不能据此承诺组合可用。

## 复现与下一步

服务器运行命令：

```bash
cd /root/qvla
.venv/bin/python scripts/probe_action_sensitivity.py --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets --dataset-root data/libero --splits data/libero_splits.json --partition config/evaluation_partition_v2.json --output-dir runs/action_sensitivity_layer_v1 --frames-per-task 2 --max-tasks 40 --groups vision_6_only vision_7_only vision_8_only vision_9_only vision_10_only vision_11_only language_0_only language_1_only language_2_only language_3_only language_4_only language_5_only language_6_only language_7_only
.venv/bin/python scripts/probe_action_sensitivity.py --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets --dataset-root data/libero --splits data/libero_splits.json --partition config/evaluation_partition_v2.json --output-dir runs/action_sensitivity_rescue_v1 --frames-per-task 2 --max-tasks 40 --groups vision_6_10 language_0_2_4_7 vision_6_10_language_without_3
```

优先把视觉 11、语言 3 与相邻低敏感层分别导出，检查 RKNN FP16/INT8 编译和独立开发激活的实际输出误差。然后在模型里验证**真实权重与激活量化**的完整动作与闭环质量，再确定混合精度方案。当前完整模型的实际低比特文件字节、峰值内存、板端延迟、功耗和闭环成功率均**未测量**。
