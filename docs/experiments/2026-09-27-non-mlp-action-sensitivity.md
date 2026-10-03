# 非 MLP 模块的完整动作敏感度

**实验性质**：原始 FP SmolVLA 上的静态输出激活 INT8 舍入诊断；权重保持原精度。它不产生真实量化文件，也不测闭环任务质量或 RK3588 板端性能。

## 问题与方法

原来的逐层诊断只覆盖 MLP。这里检查视觉 patch 卷积、视觉/语言/动作专家注意力的四个投影，以及状态、动作输入和动作输出投影，找出需要优先做真实量化验证的区域。事前仅以动作变化排序，不设动作 MAE 或激活 MSE 硬阈值。是否继续降低精度，要由真实量化后的闭环质量、资源与硬件执行结果判断。

固定 `lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de`，checkpoint SHA-256 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`；数据划分 SHA-256 `ca851a1bdc8fd60ad1e5b8d08dc7c405f971a8d4f999c4f0c2ecef154272d55f`，隔离分区 SHA-256 `f755546a6b074d2fe248333fc42c3dbf30f9b54b9f0fa0b58a16506a54d942c2`。processor 使用 checkpoint 随附配置及固定本地 SmolVLM2 tokenizer。40 个 `ptq_calibration` episode 各取首末两帧确定输出 min/max；另 40 个 `qat_train` 开发 episode 各取首末两帧比较 50×7 动作 chunk。具体 episode、帧、模块名、逐模块范围、scale、zero point、调用次数及逐观测数据见[原始报告](../../runs/action_non_mlp_sensitivity_v1/report.json)。冻结测试 episode 未使用。

每个模块独立采用静态逐张量仿射 INT8：`s=(max-min)/255`，`z=clip(round(-128-min/s),-128,127)`，`q=clip(round(x/s+z),-128,127)`，`x̂=(q-z)s`。只替换对应模块的输出；所有组使用相同预处理、后处理、观测和由 episode/帧/任务决定的初始噪声。开发指标是 `mean(|A_probe-A_FP|)`，在 80 个观测及动作 chunk 元素上平均；p95 是 80 个逐观测 chunk MAE 的第 95 百分位。FP 重复执行最大绝对差为 0。运行环境 NVIDIA A10，PyTorch `2.7.0a0+7c8ec84dab.nv25.03`，CUDA `12.8`。

## 结果

[实测图](../../figures/non_mlp_action_sensitivity_v1.png)（[SVG](../../figures/non_mlp_action_sensitivity_v1.svg)）、[绘图数据](../../figures/non_mlp_action_sensitivity_v1.json)和[绘图脚本](../../scripts/plot_non_mlp_action_sensitivity.py)。以下参数量只是参与扰动的原模型 FP 参数数目，**不是**量化后的大小。

| 输出舍入组 | 模块数 | FP 参数数 | 平均 chunk MAE vs FP | 逐观测 p95 | 逐观测最大值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 视觉 patch 投影 | 1 | 590,592 | 0.002098 | 0.004221 | 0.008904 |
| 视觉注意力 0–5 | 24 | 14,174,208 | 0.001304 | 0.002465 | 0.003349 |
| 视觉注意力 6–11 | 24 | 14,174,208 | 0.001010 | 0.001778 | 0.007109 |
| 语言注意力 0–7 | 32 | 19,660,800 | 0.001717 | 0.003750 | 0.009759 |
| 语言注意力 8–15 | 32 | 19,660,800 | 0.001085 | 0.001690 | 0.006442 |
| 动作专家注意力 0–7 | 32 | 13,721,600 | 0.000625 | 0.000799 | 0.001513 |
| 动作专家注意力 8–15 | 32 | 13,721,600 | 0.000659 | 0.000935 | 0.001232 |
| 状态/动作输入/动作输出投影 | 3 | 78,512 | 0.001345 | 0.001846 | 0.008701 |

此探针下专家注意力两组的平均动作变化较小，但输出投影仅有少量参数，量化它的潜在体积收益也小。语言前段注意力和视觉 patch 投影的平均动作变化稍大；少数观测有更大的尾部偏差。与先前 MLP 的视觉 11、语言 3 高敏感结果相比，这些组的平均偏差较低，但**不能**据此宣布 INT8 安全或给它们设最终位宽。组内同时扰动多层，不能直接推断单层贡献；输出激活 fake quant 也不包含权重量化及 RKNN 的真实校准行为。

## 选择规则与下一步

局部激活 MSE、离线动作 MAE、p95 和异常任务用于发现风险与安排实验；最终压缩深度取决于完整模型真实量化后的闭环成功率、安全相关动作、4 GB 内存约束及板端时延。在达到资源约束的候选中，继续试更低精度，即使局部 MSE 变大；只在实际任务质量退化到预定容忍范围外时放弃。容忍范围须由 FP 多种子波动和任务要求预先确定，当前**未测量**。真实模型字节、峰值内存、板端时延和闭环质量也**未测量**。

下一步先对高风险 MLP 与注意力/输入投影做同格式真实权重与激活量化，检查完整动作和尾部任务，再开展闭环配对。该诊断不为 W4 或 MX 格式提供 RK3588 支持证据。

复现命令：

```bash
cd /root/qvla
.venv/bin/python scripts/probe_action_sensitivity.py --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets --dataset-root data/libero --splits data/libero_splits.json --partition config/evaluation_partition_v2.json --output-dir runs/action_non_mlp_sensitivity_v1 --frames-per-task 2 --max-tasks 40 --groups vision_patch_projection action_interface_projections vision_attention_0_5 vision_attention_6_11 language_attention_0_7 language_attention_8_15 expert_attention_0_7 expert_attention_8_15
```
