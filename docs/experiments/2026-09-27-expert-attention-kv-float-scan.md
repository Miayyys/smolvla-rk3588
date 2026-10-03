# 四个高误差专家 K/V 层：W8A8、BF16、FP16 对照

> **2026-09-27 复核更正：本页作为失败实验保留，所有 K/V 数值与 FP16 选择结论无效。** 旧 `KV` 子图把 K 的输入复用为 V 的输入，而原加载模型分别给 K/V 不同输入。第 1 层单条样本的 K 半幅与原加载输出几乎一致、V 半幅 MAE 约 `0.9108`；因此表中所谓 W8A8/BF16/FP16 误差均针对错误子图，不能进入 HAQ。正在重建独立 K/V 输入图；下文是原错误实验的原始记录，不作选型依据。

## 问题和配置

[全 16 层专家注意力 W8A8 数值筛查](2026-09-27-all-expert-attention-rknn.md)提示奇数层 1、11、13、15 的 K/V 输出误差较高。本轮仅在这四个已选开发候选上比较 FP16/BF16，不用冻结测试选层或选格式。两条权重路径仍分别为原始 FP 独立 PTQ 和 QAT 第 50 步浮点主权重；ONNX、模型、processor、划分及输入哈希沿用前述实验记录。每个子图在 **240 条隔离开发输入**上用 RKNN 主机模拟器运行，奇数层可变 KV 长度补零到 256 token，并仅统计真实有效 token。

[`compile_rknn_projection_with_eval.py`](../../scripts/compile_rknn_projection_with_eval.py)配置 `target_platform=rk3588`、`float_dtype=float16|bfloat16`、`do_quantization=False`；W8A8 对照使用原实验的 `quantized_algorithm=mmse, quantized_method=channel, quantized_dtype=w8a8` 和同一批 240 条校准输入。浮点子图无 PTQ 校准步骤，PTQ/QAT 标签只区分其**输入权重来源**。16 份浮点子图由[`scan_expert_attention_kv_float.py`](../../scripts/scan_expert_attention_kv_float.py)以 2 worker 编译，全部通过并在主机模拟器得到输出。指标仍是逐样本有效位置 MAE 对原始权重 FP32 ONNX 的均值，不是原加载 BF16 模型的动作误差。候选[原始数据](../../runs/expert_attention_kv_float_v1/report.json)、[W8A8 数据](../../runs/expert_attention_all_rknn_v3/summary.json)与[对照图](../../figures/expert_attention_kv_float_v1.png)均已保存；绘图脚本是[`plot_expert_attention_kv_float.py`](../../scripts/plot_expert_attention_kv_float.py)。

| 层 | 原 FP→W8A8 MAE | 原 FP→BF16 MAE | 原 FP→FP16 MAE |
| ---: | ---: | ---: | ---: |
| 1 | 0.032911 | 0.002030 | 0.000252 |
| 11 | 0.038496 | 0.002206 | 0.000276 |
| 13 | 0.044771 | 0.002498 | 0.000309 |
| 15 | 0.035320 | 0.002578 | 0.000318 |

QAT 权重路径四层的 W8A8/BF16/FP16 结果分别为 1 层 `0.032906/0.002031/0.000261`、11 层 `0.038498/0.002205/0.000281`、13 层 `0.044773/0.002495/0.000313`、15 层 `0.035318/0.002570/0.000323`。完整 16 行见原始数据。本次 FP16 在这四层的局部数值显著好于 W8A8 和 BF16，但该结果**没有证明闭环质量或 RK3588 执行速度会改善**。FP16 对 FP32 ONNX 更接近，也不等同于对原加载 BF16 模型更接近。

每层每条路径的错误单输入 `.rknn` 字节为 W8A8 `399,775`、BF16 `457,093`、FP16 `493,573`；这些文件大小也不能代替修正后的独立 K/V 子图大小。FP16 候选**撤销，待正确图结构重测**。每图算子精度表、板端时延/RAM、完整模型包和收益函数 $G$ 均未测量。

服务器复现命令：

```bash
.rknn-probe/bin/python scripts/scan_expert_attention_kv_float.py --export-dir runs/expert_attention_all_export_v3 --calibration-dir runs/expert_attention_all_calibration_v3 --development-dir runs/expert_attention_all_development_v3 --output-dir runs/expert_attention_kv_float_v1 --workers 2
```
