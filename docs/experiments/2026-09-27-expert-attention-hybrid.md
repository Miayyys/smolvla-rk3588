# 专家注意力 Q/K/V：QAT、PTQ 与 RKNN 混合精度扫描

**结论范围**：仅 SmolVLA 动作专家第 0 层 self-attention 的 Q/K/V 投影子图；40 个隔离开发任务的主机模拟器数值和真实 RKNN 文件字节。没有完整注意力计算、动作、闭环或 RK3588 板端执行。**未确定最终注意力位宽**。

## 问题、固定输入和量化规则

前一轮[专家 MLP](2026-09-27-stage1-rknn-qat-ptq.md)的 QAT W8A8 数值比独立 PTQ 略好。本轮检验该结论能否迁移到注意力 Q/K/V，并按[质量优先的混合精度路线](../quantization-technique-plan.md)测试敏感投影保留 FP16 的体积—误差代价。模型 `lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de`，原始权重 SHA-256 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`，QAT 低学习率第 50 步浮点主权重 SHA-256 `521e5f1968c1e74c48ad8fe5ceedeb59085baf92e441e1a7e61cfaf9fad88680`。校准和开发分区与训练实验一致，分区 SHA-256 `f755546a6b074d2fe248333fc42c3dbf30f9b54b9f0fa0b58a16506a54d942c2`；校准 40 个 `ptq_calibration` episode，开发 40 个隔离的 `qat_train` episode，各任务首帧、首个 flow 步骤，**没有使用冻结测试**。逐输入 episode、文件 SHA、shape 见[校准捕获](../../runs/expert_attn_q0_cal_v1/report.json)和[开发捕获](../../runs/expert_attn_q0_dev_v1/report.json)。固定原模型、processor、输入和噪声；输入来自真实动作推理而非随机导出数据。

本层为 `self_attn_every_n_layers=2` 中的自注意力层；模型实现对共享的 `1×50×720` 隐状态分别执行 Q、K、V 投影。先用[`capture_mlp_calibration.py`](../../scripts/capture_mlp_calibration.py)挂在 `q_proj` 输入捕获，再由[`export_expert_qkv_probe.py`](../../scripts/export_expert_qkv_probe.py)导出三个投影及特征轴 concat 为 `1×50×1600` 的**分析子图**。concat 是方便一次比较三个投影的导出边界，不代表完整 SmolVLA 中已经部署了这个新子图。原始 FP→PTQ 与 QAT 浮点主权重→训练后校准/转换分别导出 FP32 ONNX；两者使用相同的校准输入、RKNN-Toolkit2 `2.3.2`、`target_platform=rk3588`、`quantized_method=channel`、`quantized_dtype=w8a8`、`quantized_algorithm=mmse`、optimization level 3。`q_proj` 单投影也用同法单独验证；[`export_expert_mlp_probe.py`](../../scripts/export_expert_mlp_probe.py)支持该单输入 Linear。

原 checkpoint 中 Q/K/V 实际运行权重为 **BF16**。FP32 ONNX 仅为编译中间格式；按[`capture_loaded_expert_qkv_outputs.py`](../../scripts/capture_loaded_expert_qkv_outputs.py)对同一 40 个输入捕获的[原始 BF16 输出报告](../../runs/expert_qkv0_original_loaded_v1/report.json)，它与 FP32 权重计算的平均 MAE 为 **0.00091823**。因此下面的**主表对标原模型加载后的 BF16 输出**，并保留[FP32 ONNX 对照图和原始数据](../../figures/expert_qkv0_hybrid_scan.json)供排查转换误差。

## 单投影和 QKV QAT/PTQ 对照

[`compile_rknn_mlp_probe.py`](../../scripts/compile_rknn_mlp_probe.py)生成真实 `.rknn`；[`check_exported_rknn_parity.py`](../../scripts/check_exported_rknn_parity.py)在主机上从相同 ONNX/校准输入**重建**模拟器。Toolkit2 2.3.2 不允许主机模拟器直接 `load_rknn` 后运行导出文件，故此数值不是板端或导出文件重新加载的执行结果。下表 MAE 是逐输出元素绝对差的 40 输入平均，越低越接近所标参考。

| 子图 / 路径 | `.rknn` 字节 | 输出 MAE vs 原加载 BF16 | 输出 MAE vs 原 FP32 ONNX | 状态 |
| --- | ---: | ---: | ---: | --- |
| 单独 Q 投影：原 FP→W8A8 PTQ | 732,189 | 未测量 | 0.00853074 | NPU INT8 Conv 权重 |
| 单独 Q 投影：QAT→W8A8 | 732,253 | 未测量 | 0.00853011 | 与 PTQ 差异几乎为零 |
| 单独 Q 投影：原 FP→FP16 | 1,411,011 | 未测量 | 0.00012262 | 单投影高精度对照 |
| QKV：原 FP→W8A8 PTQ | **1,229,200** | **0.00900762** | 0.00894345 | 三个投影均 NPU INT8 |
| QKV：QAT→W8A8 | 1,229,200 | **0.00952447** | 0.00947276 | 三个投影均 NPU INT8，质量代理退化 |
| QKV：原 FP→FP16 | 2,361,014 | **0.00091819** | 0.00011479 | 三投影高精度对照 |

单 Q 投影的[QAT/PTQ 逐输入差及任务 bootstrap](../../runs/expert_attn_q0_paired_comparison.json)为 QAT−PTQ `−0.00000063`，描述性 95% 区间 `[−0.00000269,+0.00000143]`，21/40 个任务偏向 QAT；该变化不支持“QAT 在该投影有效”。QKV 组合的[逐输入对照](../../runs/expert_qkv0_paired_comparison.json)则显示，相对 FP32 ONNX，QAT 比 PTQ **高 0.00052930 MAE（约 5.9%）**，40/40 个任务均更差。对原 BF16 输出重算后，退化仍约 **5.7%**。这表明单 MLP 的改善不能泛化到本层注意力。QKV 的[PTQ 编译日志](../../runs/expert_qkv0_ptq_v1/compile.log)与[QAT 编译日志](../../runs/expert_qkv0_qat50_v1/compile.log)均显示 Q/K/V 三个 `Conv` 计算落在 NPU、权重 INT8；[PTQ `.rknn`](../../runs/expert_qkv0_ptq_v1/mlp_int8_mmse_rk3588.rknn)和[QAT `.rknn`](../../runs/expert_qkv0_qat50_v1/mlp_int8_mmse_rk3588.rknn)已保存。

## 官方混合精度扫描

从原始 FP ONNX 运行 RKNN 官方 `hybrid_quantization_step1`，得到[完整校准配置](../../runs/expert_qkv0_hybrid_step1_v1/expert_layer0_qkv_fp32.quantization.cfg)。[`generate_rknn_qkv_hybrid_configs.py`](../../scripts/generate_rknn_qkv_hybrid_configs.py)仅修改 `custom_quantize_layers`，生成 Q、K、V 单独 FP16 及双投影 FP16 的六份[配置与 SHA 清单](../../config/qkv_hybrid_scan_v1/manifest.json)，其余层保留 INT8；随后使用[`probe_rknn_hybrid_step2.py`](../../scripts/probe_rknn_hybrid_step2.py)导出并对同一 40 个开发输入执行主机模拟器。方法依据 Rockchip 的[官方混合量化示例](https://github.com/airockchip/rknn-toolkit2/blob/master/rknn-toolkit2/examples/functions/hybrid_quant/README.md)。例如[V 保留 FP16 的编译日志](../../runs/expert_qkv0_hybrid_v_v1/compile.log)明确显示 V 为 NPU FLOAT16 Conv，Q/K 为 NPU INT8 Conv，并有 INT8↔FP16 转换。

| 从原 FP 编译的候选 | `.rknn` 字节 | MAE vs 原加载 BF16 | 比全 INT8 的误差变化 | 子图字节/误差 Pareto |
| --- | ---: | ---: | ---: | --- |
| 全 INT8 | 1,229,200 | 0.00900762 | 基准 | 是 |
| Q FP16 | 1,916,816 | 0.00899374 | −0.00001388 | 否，被 V FP16 支配 |
| K FP16 | 1,461,136 | 0.00921872 | **+0.00021110** | 否 |
| V FP16 | 1,461,136 | **0.00891157** | −0.00009605 | 是 |
| Q+K FP16 | 2,144,848 | **0.00889284** | −0.00011478 | 是 |
| Q+V FP16 | 2,144,848 | 0.00889669 | −0.00011093 | 否，被 Q+K FP16 支配 |
| K+V FP16 | 1,689,168 | 0.00912167 | **+0.00011405** | 否 |
| 全 FP16 | 2,361,014 | **0.00091819** | −0.00808943 | 是 |

[完整逐任务统计及 10,000 次、seed 0 的任务 bootstrap](../../figures/expert_qkv0_hybrid_scan_loaded.json)、[体积—误差图](../../figures/expert_qkv0_hybrid_scan_loaded.png)由[`summarize_expert_qkv_hybrid_scan.py`](../../scripts/summarize_expert_qkv_hybrid_scan.py)生成。V FP16 在 40/40 个任务上相对全 INT8 输出误差更低，但平均只降 **0.00009605**，文件增加 **18.87%**。Q+K FP16 再降低很少的输出误差，文件比全 INT8 增加 **74.49%**。全 INT8 相对该 FP16 子图文件缩小 **47.94%**，但数值误差显著增加。图上的 Pareto 只使用**这个分析子图的文件字节和离线输出 MAE**，不包含任务成功率、真实内存、延迟或数据转换成本；不是项目最终收益函数 $G$。此处没有选定最终混合位宽。

## 复现与结论边界

服务器 `/root/qvla` 中，先用 `capture_mlp_calibration.py` 对 `--module model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj` 分别运行 `--split-name ptq_calibration` 和 `--split-name qat_train`，两边均传 `--partition config/evaluation_partition_v2.json --max-tasks 40 --frames-per-task 1 --step-samples 1`，得到 `runs/expert_attn_q0_cal_v1` 与 `runs/expert_attn_q0_dev_v1`。再用上述导出、编译、混合量化脚本和记录中的路径重建所有候选。所有实际配置、逐输入数值、图数据、编译日志与 `.rknn` 均已保存；模型和数据原件仍在 Git 忽略目录。

由于训练 fake quant 的 min/max 与 RKNN MMSE 阈值不完全一致，QAT 浮点主权重经 RKNN 后出现退化；仅凭这一层不能判定整个 QAT 配方无效，也不能据此直接把所有专家注意力改成 FP16。子图 concat 不等于完整含 RoPE、缓存、注意力乘法及输出投影的模块；主机模拟器也不代表板端时延或 RAM。下一步应检查注意力与 MLP 的**完整动作误差/闭环质量**及 RKNN 分区边界，再按质量硬约束选位宽；RK3588 板端 $G$、RAM、p50/p95 和能耗仍是**未测量**。
