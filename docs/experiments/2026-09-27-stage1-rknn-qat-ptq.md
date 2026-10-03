# Stage 1 专家 MLP：QAT 后 RKNN W8A8 与独立 PTQ

**范围**：真实 SmolVLA 专家第 0 层 MLP 的 RK3588 子图。QAT 与 PTQ 均已生成真实 INT8 `.rknn`，并在 Toolkit2 主机模拟器上按相同配方重建、比较独立开发激活。**这不是 112 个 Linear 的完整专家图，更不是可部署的完整 VLA**；板端与闭环未测量。

## 问题、来源与事先方案

检验[stage 1 QAT 训练](2026-09-27-stage1-w8a8-qat.md)产生的浮点主权重，经 RKNN 的 W8A8 校准/转换后，能否保持真实 INT8 计算，并在同一模块、格式、校准集下比**原始 FP checkpoint 独立 PTQ**更接近原始 FP 子图。训练后第 50 步候选来自学习率 $10^{-7}$ 的 40 任务动作开发扫描；该选择本身未证明 QAT 优于 PTQ。QAT 主权重快照 SHA-256 `521e5f1968c1e74c48ad8fe5ceedeb59085baf92e441e1a7e61cfaf9fad88680`，原始模型权重 SHA-256 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`。

两条路径都以同一个真实 MLP `model.vlm_with_expert.lm_expert.layers.0.mlp` 为目标，源模型原运行 dtype 为 BF16。使用[`export_expert_mlp_probe.py`](../../scripts/export_expert_mlp_probe.py)导出 FP32 ONNX opset 17；QAT 路径仅将该层 `gate_proj`、`up_proj`、`down_proj` 的权重替换为训练后的 FP32 主权重。QAT 路径随后用隔离 `ptq_calibration` 输入做**训练后范围校准与真实转换**；没有对已转换 INT8 权重再次 PTQ。PTQ 路径则直接从原始 FP checkpoint 开始。

转换两边统一为 RKNN-Toolkit2 `2.3.2`，`target_platform=rk3588`、`quantized_method=channel`、`quantized_dtype=w8a8`、`quantized_algorithm=mmse`、optimization level 3。240 条校准输入来自 `ptq_calibration` 40 个任务各 1 episode、2 帧×3 个 flow 步骤，由[原始捕获记录](../../runs/rknn_expert_mlp_calibration_v2/report.json)及 `rknn_dataset.txt` 固定；240 条开发输入来自另外 40 个 `qat_train` 开发 episode，同样 2 帧×3 步，见[开发捕获记录](../../runs/rknn_expert_mlp_development_v2/report.json)。QAT 训练排除了所有 40 个开发 episode，校准和开发 episode 互不重叠。输入 shape 均为 `1×50×720`，数值来自真实动作推理的中间激活，而非随机导出样本。完整命令参数由[`compile_rknn_mlp_probe.py`](../../scripts/compile_rknn_mlp_probe.py)和[`check_exported_rknn_parity.py`](../../scripts/check_exported_rknn_parity.py)给出，原始逐输入数值报告在下方。

服务器 `/root/qvla` 下的导出与编译命令；`.venv` 用于原模型/ONNX，`.rknn-probe` 用于 Toolkit2：

```bash
.venv/bin/python scripts/export_expert_mlp_probe.py --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets --output-dir runs/qat_ptq_expert0_original
.venv/bin/python scripts/export_expert_mlp_probe.py --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets --output-dir runs/qat_ptq_expert0_qat50 --master runs/qat_w8a8_stage1_lr1e7/expert_master_step_50.safetensors
.rknn-probe/bin/python scripts/compile_rknn_mlp_probe.py --onnx runs/qat_ptq_expert0_original/expert_layer0_mlp_fp32.onnx --output-dir runs/qat_ptq_expert0_original --dataset runs/rknn_expert_mlp_calibration_v2/rknn_dataset.txt --mode int8 --algorithm mmse
.rknn-probe/bin/python scripts/compile_rknn_mlp_probe.py --onnx runs/qat_ptq_expert0_qat50/expert_layer0_mlp_fp32.onnx --output-dir runs/qat_ptq_expert0_qat50 --dataset runs/rknn_expert_mlp_calibration_v2/rknn_dataset.txt --mode int8 --algorithm mmse
```

两条主机模拟器命令调用同一个[`check_exported_rknn_parity.py`](../../scripts/check_exported_rknn_parity.py)，分别传各自 `--rknn`、`--onnx`、`--output`，并共用 `--reference-onnx runs/qat_ptq_expert0_original/expert_layer0_mlp_fp32.onnx`、`--dataset runs/rknn_expert_mlp_calibration_v2/rknn_dataset.txt`、`--inputs runs/rknn_expert_mlp_development_v2/heldout_inputs.txt`；实际原始 JSON 保存了所有这些路径。

## 转换结果

| 路径 | ONNX SHA-256 | RKNN SHA-256 | RKNN 字节 | 主要计算权重 |
| --- | --- | --- | ---: | --- |
| 原始 FP → PTQ | `fb9e669995abff913834d0e0e061d787ed1442c4f640bccddf5d95d74057474b` | `9b8a2dd8aedd928512d525848415eee48d267b62551dde9ae25b765669c26839` | 4,524,893 | 3 个 Conv/融合计算权重 INT8 |
| QAT 第 50 步 FP 主权重 → RKNN | `443c526cd4d339c7c97c282a20e28312fd610850af5ac7b366852cc5704e3648` | `2cdfef74ce6d632af581106e14fcee9fa794e4acb17273543d47b216d74ba659` | 4,524,893 | 同样 3 个 INT8 |

两份[PTQ `.rknn`](../../runs/qat_ptq_expert0_original/expert_layer0_mlp_int8_mmse_rk3588.rknn)与[QAT `.rknn`](../../runs/qat_ptq_expert0_qat50/expert_layer0_mlp_int8_mmse_rk3588.rknn)均已备份；[PTQ 编译报告及日志](../../runs/qat_ptq_expert0_original/int8_mmse_compile_report.json)、[QAT 编译报告及日志](../../runs/qat_ptq_expert0_qat50/int8_mmse_compile_report.json)可复核 INT8 权重落点。既有同一 MLP 的 [FP16 RKNN 编译报告](../../runs/rknn_expert_mlp_probe/fp16_compile_report.json)为 8,910,851 B，因此这个**单层子图**的 W8A8 文件比 FP16 小 $1-4{,}524{,}893/8{,}910{,}851=49.22\%$。该比例不等于整模型压缩率，也不含板端峰值 RAM。

## 相同开发输入的数值比较

Toolkit2 2.3.2 明确拒绝在主机模拟器中 `load_rknn` 后运行；因此使用**相同 ONNX、校准文件和编译参数重新构建模拟器**，没有把下表伪称为“导出文件重新加载后的执行”。原始 FP32 ONNX 是共同参考；每条输入计算 $E_i=\operatorname{mean}|y_i^{\mathrm{RKNN}}-y_i^{\mathrm{FP32}}|$。QAT 路径另算与其训练后浮点 ONNX 的偏差。原始[PTQ 240 条](../../runs/qat_ptq_expert0_original/development_simulator_parity.json)、[QAT 240 条](../../runs/qat_ptq_expert0_qat50/development_simulator_parity.json)报告保留输入路径与逐条误差。

| 路径 | 240 条平均输出 MAE vs 原始 FP32 ONNX | vs 自身浮点 ONNX | 自身浮点 ONNX vs 原始 FP32 |
| --- | ---: | ---: | ---: |
| 原始 FP → PTQ W8A8 | 0.00941949 | 0.00941949 | 0 |
| QAT 第 50 步 → W8A8 | **0.00923784** | 0.00923601 | 0.00013177 |

由[`compare_qat_ptq_subgraph.py`](../../scripts/compare_qat_ptq_subgraph.py)按同名输入配对、先按 40 个任务求均值后得到 QAT−PTQ 平均差 **−0.00018165**，相对 PTQ 降低 **1.93%**；40/40 任务的均值差为负。[逐输入、逐任务与重采样原始结果](../../runs/qat_ptq_expert0_paired_comparison.json)给出固定 seed 0、10,000 次任务 bootstrap 的描述性 95% 区间 `[−0.00018598, −0.00017744]`。[任务误差图](../../figures/qat_ptq_expert0_rknn_paired.png)和[SVG](../../figures/qat_ptq_expert0_rknn_paired.svg)由[`plot_qat_ptq_subgraph.py`](../../scripts/plot_qat_ptq_subgraph.py)生成。**这不是独立确认置信区间**：QAT 第 50 步曾用同一批开发任务的动作代理指标选取，尽管此处是另外捕获的帧与步骤；冻结测试仍未使用。

## 结论与边界

本轮证实“QAT FP 主权重 → RKNN 校准/编译”在真实专家 MLP 上可得到计算权重 INT8 的文件，且与原始 FP 的独立 PTQ 有相同的位宽、模块、校准方法和子图体积。该层主机数值的 QAT 改善很小但方向一致；它**不能**替代完整动作质量、闭环成功率、全图可编译性或 RK3588 板端实测。两份 `.rknn` 尚未在板端重新加载执行；整模型字节、RAM、p50/p95 延迟、功耗与收益函数 $G$ 均**未测量**。下一步须把同一方法扩到所有 112 个专家 Linear 所在的可导出子图，处理 FP/NPU 边界，再做完整动作和闭环配对；若子图改善未转化为任务质量，则优先维持模型质量而不是追求此误差代理。
