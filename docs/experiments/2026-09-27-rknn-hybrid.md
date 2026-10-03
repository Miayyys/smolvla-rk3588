# RKNN 官方混合量化：动作专家 MLP 的两个 FP16 保留层

**状态**：`step1 → 改配置 → step2` 流程对真实 SmolVLA MLP 子图编译成功；两个候选在独立开发集的平均输出误差均未改善，因此**不采用**。这里只测了单个子图的主机模拟器；没有完整动作、RK3588 板端执行或延迟。

## 假设与原理

针对[扩展校准实验](2026-09-27-mlp-calibration.md)中 RKNN 原生 MMSE 的候选，从 RKNN `step1` 生成的量化参数查找截断明显的中间激活，让单个输出保持 FP16，其余部分继续 INT8。混合量化允许敏感层使用较高精度，但也可能增加转换、数据搬运和文件开销；本轮仅检查可编译性、字节数和固定开发集上的输出 MAE。根据 [RKNN 官方示例](https://github.com/airockchip/rknn-toolkit2/blob/master/rknn-toolkit2/examples/functions/hybrid_quant/README.md)，先运行 `hybrid_quantization_step1`，在生成的 `custom_quantize_layers` 中指定输出名及 `float16`，再运行 `hybrid_quantization_step2` 导出 `.rknn`。使用 [`probe_rknn_hybrid_mlp.py`](../../scripts/probe_rknn_hybrid_mlp.py)和[`probe_rknn_hybrid_step2.py`](../../scripts/probe_rknn_hybrid_step2.py)实现。

## 固定配置与实际量化参数

模型、processor、ONNX、episode 划分、采样均与[扩展校准实验](2026-09-27-mlp-calibration.md)一致。具体为 SmolVLA-LIBERO 权重 SHA-256 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`、FP32 ONNX SHA-256 `fb9e669995abff913834d0e0e061d787ed1442c4f640bccddf5d95d74057474b`；240 个 `ptq_calibration` 激活只用于校准，另 240 个 `qat_train` 开发激活只用于比较。这 40 个开发 episode 已排除后续 QAT 训练，清单见 [`evaluation_partition_v2.json`](../../config/evaluation_partition_v2.json)。工具为 RKNN-Toolkit2 2.3.2，目标 `rk3588`，`quantized_algorithm=mmse`，`quantized_method=channel`，`quantized_dtype=w8a8`，`float_dtype=float16`，`proposal=False`。

`step1` 成功耗时 6.749 秒，生成 `.model` 17,697,008 B、`.data` 236,560 B 和[原始量化配置](../../config/rknn_mlp_mmse_base.cfg)；[报告](../../runs/rknn_expert_mlp_hybrid_v2/hybrid_step1_report.json)。原始配置 SHA-256 `1800042f7e44ec9dcdbf6e25924d99441054e06df441f9edaf35a36c77e3fe73`。此配置显示了**RKNN 实际生成的部分激活量化参数**，与手写对称 INT8 直方图扫描分属不同量化器：

| 输出名 | 实际规则 | 原始 min/max | 配置 min/max | scale / zero point |
| --- | --- | --- | --- | --- |
| `hidden_states_rs` | `asym`, `layer`, INT8 | -5.78125 / 6.28125 | -5.78125 / 6.28125 | 0.04730392 / -6 |
| `/gate_proj/MatMul_output_0_mm_tp_sw` | `asym`, `layer`, INT8 | -0.27846 / 3.24110 | -0.27846 / 2.02727 | 0.00904211 / -97 |
| `/Mul_output_0-rs` | `asym`, `layer`, INT8 | -7.39039 / 5.24019 | -4.62260 / 4.62260 | 0.03625569 / 0 |

这里的 `layer` 是该**激活**配置显示的粒度；`quantized_method=channel` 是编译选项，不能由此推断每个激活都按通道量化。`step1` 文件并未给出一条可复核的逐候选 KL/MSE 曲线，因此不能把[独立扫描图](../../figures/rknn_mlp_clipping_v2.png)的阈值说成 RKNN 的阈值。上表第三、四列分别来自配置里的 `ori_min/ori_max` 和 `min/max`，并不说明每个元素的饱和比例。

两个试验仅修改 `custom_quantize_layers`：候选 A 保留 `/Mul_output_0-rs`，候选 B 保留 `/gate_proj/MatMul_output_0_mm_tp_sw`。完整文件分别为 [`mul_fp16.cfg`](../../config/rknn_mlp_mmse_mul_fp16.cfg)（SHA-256 `0e24f0d0e62262d1e3342eaf994cf456471e82a8e52ce9a2dc5548f27d56c14f`）和[`gate_fp16.cfg`](../../config/rknn_mlp_mmse_gate_fp16.cfg)（SHA-256 `1cfc64101375330f8d5c9e7817c42cae71b8db06aa15705b59535454c12e319e`）。选择这两个输出的依据是配置里原始范围和编译范围存在明显差别；这只是敏感度假设，须用开发集证伪。

编译日志确认了配置被识别：乘法候选出现 `Meet hybrid type, dtype: float16, tensor: /Mul_output_0-rs`，算子表将对应 `Mul` 标为 `FLOAT16 NPU`，并插入 INT8/FP16 格式转换；门控候选也出现对应 `Meet hybrid type` 记录。它们说明编译器确实应用了混合精度及转换，仍不能代替 RK3588 板端执行验证。日志位于服务器 `/root/qvla/runs/rknn_expert_mlp_hybrid_v2/hybrid_mul_fp16.log`、`hybrid_gate_fp16.log`。

## 开发集结果

在同一 240 个 `[1,50,720]` 开发输入上，以 FP32 ONNX Runtime 为参考。混合产物由 `step2` **真实编译并导出**，随后同一 RKNN 对象在主机模拟器执行；INT8 MMSE 对照使用相同 ONNX、算法和校准输入重新 build。逐任务均值比较按每任务的 6 个开发激活计算。

| 方案 | `.rknn` 字节 | 平均输出 MAE | 最大逐输入 MAE | 最低余弦 | 相对 INT8 MMSE 平均 MAE | 40 任务中更优数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 全 INT8 MMSE | 4,524,893 | **0.00941949** | 0.01254902 | 0.99882072 | 基准 | — |
| 乘法输出 FP16 | 4,533,917 | 0.00942201 | 0.01253900 | 0.99881762 | +0.027% | 11 |
| 门控输出 FP16 | 4,538,269 | 0.00952368 | 0.01378973 | 0.99884599 | +1.106% | 3 |

两个混合配置文件分别比全 INT8 多 9,024 B 和 13,376 B，平均 MAE 均略增。最低余弦与最大逐输入 MAE 的排序并不完全一致，因此选择依据明确限定为**开发集平均输出 MAE**。这两个单层 FP16 配置目前被否决，不代表其他层或完整模型的混合精度没有收益。`step2` 编译与 240 次模拟器推理总耗时为 170.248 秒、145.118 秒；这不是板端推理延迟。

原始逐输入报告：[`乘法输出 FP16`](../../runs/rknn_expert_mlp_hybrid_v2/hybrid_mul_fp16_development_parity.json)、[`门控输出 FP16`](../../runs/rknn_expert_mlp_hybrid_v2/hybrid_gate_fp16_development_parity.json)、[`全 INT8 MMSE`](../../runs/rknn_expert_mlp_development_probe_v2/int8_mmse_development_parity.json)。产物在服务器 `/root/qvla/runs/rknn_expert_mlp_hybrid_v2/`；SHA-256 分别为 `a61ed0a1c3b4bc533d683a96fb8839a1732165fada8b1bd9cc14d19bfe8fab70`、`78384d9261b8a2338452df10f2ad599dd9b07e9b1a3d8564dd597fb94faf1544`。完整日志也在服务器同目录。当前转换环境仍借用 PyTorch 2.7，超出 RKNN 2.3.2 官方依赖声明上限；正式复现须在声明支持的环境复核。

复现关键命令（服务器 `/root/qvla`；`.model` 和 `.data` 由第一条生成）：

```bash
.rknn-probe/bin/python scripts/probe_rknn_hybrid_mlp.py --onnx runs/rknn_expert_mlp_probe/expert_layer0_mlp_fp32.onnx --dataset runs/rknn_expert_mlp_calibration_v2/rknn_dataset.txt --output-dir runs/rknn_expert_mlp_hybrid_v2 --algorithm mmse
.rknn-probe/bin/python scripts/probe_rknn_hybrid_step2.py --model runs/rknn_expert_mlp_hybrid_v2/expert_layer0_mlp_fp32.model --data runs/rknn_expert_mlp_hybrid_v2/expert_layer0_mlp_fp32.data --config config/rknn_mlp_mmse_mul_fp16.cfg --onnx runs/rknn_expert_mlp_probe/expert_layer0_mlp_fp32.onnx --inputs runs/rknn_expert_mlp_development_v2/heldout_inputs.txt --output-dir runs/rknn_expert_mlp_hybrid_v2 --label mul_fp16
```

## 下一步

把此 MLP 的 MMSE INT8 作为待验证子图候选，继续测其他动作专家层、视觉/语言接口及完整动作 chunk 的误差。混合精度只给在开发动作质量或真实板端资源指标上有收益的模块；先确定后端可实现的共同方案，再做面向该方案的 QAT，训练后真实转换，再从原始 FP 模型独立做 PTQ 对照。完整模型和板端指标仍未测量。
