# SmolVLA 完整动作的 MLP 层组敏感度：静态 INT8 输出舍入

**性质**：质量优先的离线定位实验。只在原始 FP 模型运行时，对指定 MLP 层组的**输出激活**做静态仿射 INT8 舍入再反量化；模型权重没有量化，未保存低比特模型，没有 RKNN 执行、闭环成功率或板端数据。结果只能帮助选择下一批实际量化候选，不能判定最终高低精度配置。

本实验各方案共享原始 FP 权重文件，实际低比特模型体积 **未测量/不存在**；推理开销包含 Python hook 的诊断扰动，不能代表量化内核速度，因此模型/板端延迟和内存收益均记为 **未测量**。后续真实转换实验必须补齐这些指标。

## 问题和原理

此前仅验证过一个动作专家 MLP 子图的数值误差，尚不知道该误差对**完整动作 chunk**的影响。本轮从视觉、语言、动作专家各选两个连续层组，并单列动作专家第 0 层，观察同一 FP checkpoint 在局部激活舍入后产生的动作变化。以每个模块校准集输出的 min/max 计算 `s=(max-min)/255`、`z=clip(round(-128-min/s),-128,127)`，执行 `q=clip(round(x/s+z),-128,127)`、`x̂=(q-z)s`，再把 `x̂` 送给模型的下一层。这是 **per-tensor 输出激活 fake quant**：没有权重量化，也不保证与 RKNN 编译器的实际中间格式一致。

选择依据是开发集与 FP 的配对**完整动作 chunk MAE**，不是仅凭局部重建误差；同时记录首个动作与示范动作的 MAE 变化，但该指标也不能替代闭环成功率。视觉/语言/专家的参数量只用于了解层组规模，不能把参数量乘以 8 bit 当作实际存储收益。

## 固定配置和隔离

| 项目 | 配置 |
| --- | --- |
| 模型 | `lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de`；权重 SHA-256 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`；checkpoint 自带 processor |
| 数据划分 | [`evaluation_partition_v2.json`](../../config/evaluation_partition_v2.json)：40 个 `ptq_calibration` episode 校准；40 个互不重叠的 `qat_train` episode 开发，且从后续 QAT 训练中排除；不使用剩余 213 个冻结测试 episode |
| 采样 | 每任务各 1 个 episode，每个取首帧和末帧：80 校准观测、80 开发观测；每次动作生成 10 个流匹配步骤，完整动作 chunk 50×7 |
| 配对 | 同一图像、状态、任务、checkpoint 前后处理；每个 episode/帧以 `1009×episode_index + 9176×frame_rank + 17×task_index` 重置 PyTorch 与 CUDA 随机种子；每次 `policy.reset()` |
| 环境 | GPUServer NVIDIA A10；PyTorch/CUDA 实际版本在原始报告；[`探针脚本`](../../scripts/probe_action_sensitivity.py) |

同一开发样本重复执行 FP 的动作 chunk 最大绝对差为 **0.0**，确认本轮配对随机性的可重复性。脚本校准了 44 个 MLP 输出，其中动作专家 MLP 每帧调用 10 次。原始报告、逐样本结果、每个模块的校准范围/scale/zero point 和完整动作 chunk 保存于 Git 忽略的 [`runs/action_sensitivity_v1/`](../../runs/action_sensitivity_v1/report.json)；[图](../../figures/action_sensitivity_v1.png)（[SVG](../../figures/action_sensitivity_v1.svg)）、[逐任务绘图数据](../../figures/action_sensitivity_v1_summary.json)与[`绘图脚本`](../../scripts/plot_action_sensitivity.py)可复核均值及任务分布。

## 开发集结果

每次仅扰动表中指定层组，其他模块保持 FP。95% 区间是以**任务**为单位、有放回抽样 10,000 次得到的均值区间（固定随机种子 20260927）；它描述这一批任务的差异，不是闭环成功率置信区间。

| 被扰动的 MLP 层组 | 模块参数量 | 完整动作 chunk MAE vs FP | 任务 bootstrap 95% 区间 | 首动作 MAE vs FP | 首动作对示范 MAE 增量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 视觉 0–5 | 28,334,592 | 0.010713 | 0.009714–0.011787 | 0.007780 | +0.001478 |
| 视觉 6–11 | 28,334,592 | **0.030160** | 0.027706–0.032784 | 0.026129 | +0.013678 |
| 语言 0–7 | 58,982,400 | **0.018173** | 0.014778–0.022546 | 0.009920 | +0.000877 |
| 语言 8–15 | 58,982,400 | 0.003662 | 0.003393–0.003960 | 0.003315 | +0.000272 |
| 动作专家 0–7 | 35,389,440 | 0.000471 | 0.000463–0.000478 | 0.000459 | -0.000001 |
| 动作专家 8–15 | 35,389,440 | 0.000420 | 0.000412–0.000428 | 0.000431 | +0.000012 |
| 动作专家第 0 层 | 4,423,680 | 0.000420 | 0.000413–0.000427 | 0.000444 | -0.000022 |

本开发集 FP 首动作对示范动作 MAE 为 **0.022987**。该值与之前测试 episode 的首帧基线采用不同样本，不作跨实验直接比较。上述“示范 MAE 增量”有正有负，数值接近零时尤其不能当作成功率提升或下降。

观察到视觉后 6 层和语言前 8 层的校准输出存在大范围值：视觉 6–11 的各层 scale 范围为 **0.02597–4.88625**，语言 0–7 为 **0.18922–12.32941**；专家 0–7 仅为 **0.00628–0.01550**。这些是不同模块的 per-tensor min/max scale，提示离群值可能放大舍入步长，因此高动作偏移**不能直接归因于必须 FP16**。需要先试校准集 MSE 截断、适合后端的粒度，并在完整动作和实际 RKNN 模型上复验。

服务器复现命令：

```bash
cd /root/qvla
.venv/bin/python scripts/probe_action_sensitivity.py --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets --dataset-root data/libero --splits data/libero_splits.json --partition config/evaluation_partition_v2.json --output-dir runs/action_sensitivity_v1 --frames-per-task 2 --max-tasks 40 --groups vision_0_5 vision_6_11 language_0_7 language_8_15 expert_0_7 expert_8_15 expert_0_only
```

## 目前可用于 HAQ 的判断

这是“动作质量代价”一轴的早期排序。视觉 6–11、语言 0–7 值得优先做校准优化与层内定位；专家 MLP 输出舍入对动作影响较小，值得优先检验真正可编译的 INT8 候选。**不能**由此认定整个动作专家可以无损 INT8，也不能由某组动作 MAE 直接推断闭环成功率、板端节省或执行速度。后续要把实际量化格式、转换边界、权重量化误差和板端资源加入同一候选表，质量未达标的方案即使更快也不采纳。
