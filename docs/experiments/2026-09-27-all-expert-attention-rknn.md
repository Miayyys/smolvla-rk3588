# 16 层专家注意力投影的 RKNN QAT/PTQ 对照

> **2026-09-27 复核更正：奇数层 K/V 的 8 组数值实验无效。** 原始采集只取 `k_proj` 输入，旧导出却将同一输入同时送入 `k_proj` 和 `v_proj`；实际加载模型里 V 输入不同。第 1 层首条开发样本的 K 半幅 ONNX 与原加载输出 MAE 约 `2.2e-7`，V 半幅约 `0.9108`，足以证实旧 KV 子图图结构错误。下文 80/80 仅说明这些构造出的图能编译运行；40 组误差中 8 个 KV 组及其排序、位宽建议**撤销**，不得用于 HAQ。原始数据保留为失败记录；正确的 K/V 独立输入修复和新结果见[更正后的实验](2026-09-27-expert-attention-kv-correction.md)。

**问题**：先检查专家注意力的各投影能否按 RK3588 W8A8 配方真实编译，并用隔离开发激活定位数值风险。这里的子图输出误差只用于筛选，不能代替完整动作、闭环成功率或板端收益。

## 原理、输入与配置

专家 16 层交替使用不同注意力输入。偶数层把 Q/K/V 合为一个投影子图，奇数层分别导出 Q 和 K/V；每层另有输出投影，因此共 40 组、每组 QAT 和独立 PTQ 各一份，共 80 份 `.rknn`。奇数层 K/V 的原始 token 长度可变，采集时补零到 `1×256×320`，开发数值只统计原始有效 token；Q/QKV 输入为 `1×50×720`，输出投影为 `1×50×960`。该 padding 仅验证了本次采集长度均不超过 256；后续完整动作接口仍须处理动态长度。

使用锁定的 [`config/evaluation_partition_v2.json`](../../config/evaluation_partition_v2.json)，模型权重 SHA-256 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`，划分文件 SHA-256 `f755546a6b074d2fe248333fc42c3dbf30f9b54b9f0fa0b58a16506a54d942c2`，QAT 第 50 步浮点主权重 SHA-256 `521e5f1968c1e74c48ad8fe5ceedeb59085baf92e441e1a7e61cfaf9fad88680`。校准用 40 个 `ptq_calibration` episode，开发用不重叠的 40 个 `qat_train` episode；各任务采 2 帧、每帧 3 个去噪步骤，每组分别有 240 条校准输入和 240 条开发输入。逐样本路径、哈希、原长度见[校准原始记录](../../runs/expert_attention_all_calibration_v3/report.json)和[开发原始记录](../../runs/expert_attention_all_development_v3/report.json)。冻结测试未参与。

[`capture_all_expert_attention_inputs.py`](../../scripts/capture_all_expert_attention_inputs.py)采集激活；[`export_all_expert_attention_qat_ptq.py`](../../scripts/export_all_expert_attention_qat_ptq.py)从原始 FP 权重及 QAT 浮点主权重各导出一次 FP32 ONNX，均经 ONNX checker 和参考前向检查，[逐图哈希和输入输出形状](../../runs/expert_attention_all_export_v3/report.json)可复核。随后以 Toolkit2 `target_platform=rk3588, quantized_dtype=w8a8, quantized_method=channel, quantized_algorithm=mmse, optimization_level=3` 编译。QAT 是训练后的浮点主权重再校准、转换；PTQ 从原始 FP 独立校准、转换，没有对低比特权重二次 PTQ。[`compile_all_expert_attention_qat_ptq.py`](../../scripts/compile_all_expert_attention_qat_ptq.py)以 6 个 CPU worker 并行，按输入和 ONNX 哈希断点恢复；单图编译、模拟器和逐样本统计见[`compile_rknn_projection_with_eval.py`](../../scripts/compile_rknn_projection_with_eval.py)。

每条输入分别用原始 FP32 ONNX、当前候选 FP32 ONNX 和 RKNN 主机模拟器计算。报告指标为

$$E_{g,m}=\frac1{240}\sum_{i=1}^{240}\operatorname{mean}_{t<c_i,f}|Y^{\mathrm{RKNN}}_{g,m,i,t,f}-Y^{\mathrm{原始FP32}}_{g,i,t,f}|,$$

其中 $c_i$ 为有效 token 长度；这和原模型加载 BF16 的全链输出不完全等价。逐输入另存 `mae_vs_own_fp32` 与 `master_mae_vs_original_fp32`，便于区分 QAT 权重变化与量化误差。原始统计在服务器各子图 `report.json`，本地有[80 图汇总](../../runs/expert_attention_all_rknn_v3/report_layers_00_16.json)、[40 组配对数据](../../runs/expert_attention_all_rknn_v3/summary.json)和[绘图](../../figures/expert_attention_all_rknn_v3.png)；图由[`plot_all_expert_attention_rknn.py`](../../scripts/plot_all_expert_attention_rknn.py)生成。

## 实测结果与选择

80/80 个子图编译及主机模拟器评测成功。单条路径 40 个分离 `.rknn` 合计 **30,580,128 B**，QAT/PTQ 相同；这是子图文件之和，含重复封装，**不是完整模型包**。在 40 组中，QAT 的 $E$ 低于 PTQ 有 21 组，高于 PTQ 有 19 组，故不能认为 QAT 在注意力整体必然更好。奇数层 K/V 是本轮数值风险最高的组：第 13 层 PTQ/QAT 为 `0.044771/0.044773`，第 11 层 `0.038496/0.038498`，第 15 层 `0.035320/0.035318`，第 1 层 `0.032911/0.032906`。该排序只供浮点候选优先级，不直接确定最终位宽，因为各组输出量纲及全链误差传播不同。

上述 K/V 排序和高精度候选因图结构错误已撤销，不能依其确定格式。其余投影的 W8A8 仍需用原加载输出核对输入对应关系，并做闭环验证。虽然用 W8A8 配方导出真实 `.rknn`，本轮没有逐图解析算子表，不能声称 80 图内所有计算都在 NPU INT8 执行。板端运行、RAM、端到端 p50/p95、功耗和完整收益函数 $G$ 均**未测量**。

复现命令（服务器 `/root/qvla`）：

```bash
.venv/bin/python scripts/capture_all_expert_attention_inputs.py --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets --dataset-root data/libero --splits data/libero_splits.json --partition config/evaluation_partition_v2.json --output-dir runs/expert_attention_all_calibration_v3 --split calibration --frames-per-task 2 --step-samples 3
.venv/bin/python scripts/capture_all_expert_attention_inputs.py --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets --dataset-root data/libero --splits data/libero_splits.json --partition config/evaluation_partition_v2.json --output-dir runs/expert_attention_all_development_v3 --split development --frames-per-task 2 --step-samples 3
.venv/bin/python scripts/export_all_expert_attention_qat_ptq.py --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets --qat-snapshot runs/qat_w8a8_stage1_lr1e7/expert_master_step_50.safetensors --partition config/evaluation_partition_v2.json --output-dir runs/expert_attention_all_export_v3
.rknn-probe/bin/python scripts/compile_all_expert_attention_qat_ptq.py --export-dir runs/expert_attention_all_export_v3 --calibration-dir runs/expert_attention_all_calibration_v3 --development-dir runs/expert_attention_all_development_v3 --output-dir runs/expert_attention_all_rknn_v3 --workers 6
```
