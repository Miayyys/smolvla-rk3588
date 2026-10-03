# BF16、FP16 与 MX 格式：RK3588 子图探针

**范围**：核对 checkpoint 文件与 LeRobot 实际运行 dtype；在四个真实 SmolVLA MLP 上比较 BF16/FP16 的直接计算和 RKNN-Toolkit2 2.3.2 主机模拟器输出；探测 Toolkit2 对 MX 名称的配置接受情况。尚无 RK3588 板端执行、完整动作质量或延迟数据。

## 为什么不能把 BF16 与 FP16 混同

两者均为 16 bit。BF16 为 8 位指数、7 位尾数，指数范围更宽；FP16 为 5 位指数、10 位尾数，在可表示范围内舍入更细。BF16 可能减少溢出/下溢，但**不是普遍更精确**。将 BF16 转成 FP16 不带来名义权重位宽节省；实际文件差异要看编译器布局。选择须按原始**运行时** dtype、相同输入的动作质量和目标后端决定。

[`model.safetensors`](../../runs/checkpoint_dtype_inspection/report.json) SHA-256 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`、文件 906,712,520 B，其中 **474 个 BF16 张量（446,772,624 元素）**、**26 个 FP32 张量（3,273,552 元素）**。这是**磁盘存储**。按当前 LeRobot 环境加载后，本次检查的视觉 10/11 MLP 参数为 **FP32**，语言 3/4 MLP 参数为 **BF16**；[`inspect_checkpoint_dtypes.py`](../../scripts/inspect_checkpoint_dtypes.py)和[运行时报告](../../runs/rknn_float_dtype_compare/report.json)可复核。故此前仅凭 checkpoint 头推断整个运行模型是 BF16 不准确。

## 固定输入、计算与结果

沿用[同一批子图输入和分区](2026-09-27-rknn-sensitive-mlp-probe.md)：`ptq_calibration` 任务 0–3 各首帧，共 4 个仅用于 INT8 校准；与之隔离的 `qat_train` 开发 episode 首帧共 4 个用于下表比较，不使用冻结测试集。模型、processor、输入 SHA、环境见原始[运行时报告](../../runs/rknn_float_dtype_compare/report.json)。直接计算取加载后的 MLP，分别复制为 FP32、BF16、FP16，输入对应转换，令 `E_d=mean(|Y_d-Y_FP32|)`；FP32 是**加载后同一权重的 FP32 计算参考**，不是原始完整模型的统一 dtype。RKNN 主机模拟器沿用相同 FP32 ONNX 与开发输入，分别编译 `float_dtype=bfloat16/float16`，以及已记录的 INT8 MMSE 对照。脚本为[`compare_mlp_float_dtypes.py`](../../scripts/compare_mlp_float_dtypes.py)。

| MLP | 原加载参数 dtype | 直接 BF16 MAE vs FP32 | 直接 FP16 MAE vs FP32 | FP16 MAE vs 原加载行为 |
| --- | --- | ---: | ---: | ---: |
| 视觉 10 | FP32 | 0.001071 | 0.000128 | 0.000128 |
| 视觉 11 | FP32 | 0.004115 | 0.000483 | 0.000483 |
| 语言 3 | BF16 | 0.003862 | 0.000479 | **0.003903** |
| 语言 4 | BF16 | 0.003605 | 0.000446 | **0.003636** |

FP16 对 FP32 参考更接近，但语言 3/4 的**原始运行行为是 BF16**；对它们来说，FP16 相对原行为的变化见最后一列，不应仅以 FP32 参考误差更小就认定 FP16 提高任务质量。四层的开发输入 FP16 转换均未出现无穷大或非零值变零；权重 FP16 转换没有溢出，极少数非零权重变零（各层 1–7 个），细节见报告。视觉输入转换为 BF16 的最大单元素差约 0.24，语言输入原本可无误差地重现为 BF16。以上仅 4 个开发输入/层，尚不能推出全数据的动态范围。

| MLP | RKNN BF16 MAE vs FP32 ONNX | RKNN FP16 MAE vs FP32 ONNX | BF16 `.rknn` 字节 | FP16 `.rknn` 字节 |
| --- | ---: | ---: | ---: | ---: |
| 视觉 10 | 0.001028 | 0.000151 | 9,890,823 | 9,938,567 |
| 视觉 11 | 0.003849 | 0.000501 | 9,890,823 | 9,938,567 |
| 语言 3 | 0.003862 | 0.000478 | 14,852,165 | 14,857,285 |
| 语言 4 | 0.003605 | 0.000445 | 14,852,165 | 14,857,285 |

BF16 子图均成功导出真实 `.rknn`，编译日志出现内部 `BFLOAT16` 张量；例如[视觉 11 编译日志](../../runs/rknn_vision11_bf16_probe_compile.log)及[编译报告](../../runs/rknn_vision11_bf16_probe/bf16_compile_report.json)。四层的[BF16/FP16/INT8 图](../../figures/rknn_float_dtypes_v1.png)、[SVG](../../figures/rknn_float_dtypes_v1.svg)、[绘图数据](../../figures/rknn_float_dtypes_v1.json)与[脚本](../../scripts/plot_rknn_float_dtypes.py)汇总直接计算及模拟器误差。完整逐输入/逐格式数值在 `runs/rknn_*_bf16_probe/parity.json`、`runs/rknn_*_fp16_probe/parity.json`。这批 BF16 文件与报告的[备份包](../../runs/qvla_rknn_dtype_mx_probe_20260927.tar.gz)本地/服务器 SHA-256 同为 `a13a1092ef54799984fd1b8224b2c65f2f7c23673cb4ca581871746dae07f072`。

**收益边界**：BF16→FP16 对这四层的名义权重位宽收益为 **0 bit/参数**。BF16 `.rknn` 反而比 FP16 略小：视觉每层少 47,744 B，语言每层少 5,120 B；这是编译产物差异，不是可推广的模型压缩率。板端吞吐、延迟、峰值内存、能耗及完整动作/闭环质量均**未测量**。RKNN 主机模拟器从 ONNX 重新构建，尚未在板上加载导出的文件。BF16 编译成功也不能替代板端兼容性验证。

### 追加：直接对标原始加载模型

为避免 FP32 ONNX 参考掩盖语言层原本的 BF16 行为，另将**原加载 MLP 在相同输入上的输出**保存为逐样本参考（[报告与输出 SHA](../../runs/rknn_float_dtype_compare/report_vs_loaded.json)，[`original_mlp_outputs/`](../../runs/original_mlp_outputs/)）。RKNN BF16/FP16 主机模拟器重建模型分别与此参考逐元素配对：

| MLP | 原加载 dtype | RKNN BF16 MAE vs 原加载输出 | RKNN FP16 MAE vs 原加载输出 | 本轮质量优先候选 |
| --- | --- | ---: | ---: | --- |
| 视觉 10 | FP32 | 0.001028 | **0.000151** | FP16 若需保留浮点 |
| 视觉 11 | FP32 | 0.003849 | **0.000501** | FP16 |
| 语言 3 | BF16 | **0.0000398** | 0.003903 | BF16 |
| 语言 4 | BF16 | **0.0000420** | 0.003639 | BF16 若需保留浮点 |

上述是**隔离 MLP 输出**的数值误差，仍不能代替完整动作/闭环质量。按原始运行精度比较，视觉的 FP16 更接近原 FP32，语言的 BF16 更接近原 BF16；这修正了仅看 FP32 ONNX 参考时“全部优先 FP16”的判断。见[对原模型图](../../figures/rknn_vs_original_v1.png)、[SVG](../../figures/rknn_vs_original_v1.svg)、[逐格式数据](../../figures/rknn_vs_original_v1.json)与[绘图脚本](../../scripts/plot_rknn_vs_original.py)。各项原始数值在 `runs/rknn_{vision10,vision11,language3,language4}_{bf16,fp16}_probe/parity_vs_loaded.json`；复核传输包 [`qvla_rknn_original_reference_20260927.tar.gz`](../../runs/qvla_rknn_original_reference_20260927.tar.gz)本地/服务器 SHA-256 同为 `5c702899e988807e49c920399c019942ce5ae044feba3354d989cddc2b870900`。

## MX4/MX8 与普通 INT4/INT8 的界限

[`info.md`](../../info.md) 中的“MX4”是每 2 个值共享 micro-exponent、每 16 个值共享大 exponent 的**两级共享指数示例**，平均 4 bit/值；它**不等于** OCP 标准 MXFP4。OCP [MX v1.0 规范](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)列出 MXFP4（FP4 E2M1）、MXFP8（FP8 E4M3/E5M2）、MXINT8：均为 32 元素共享一个 8-bit E8M0 scale，名义有效位宽分别是 `4+8/32=4.25`、`8+8/32=8.25`、`8.25` bit/值，尚未计填充和其他元数据。“MX8”名称本身不足以指定 MXFP8 还是 MXINT8。

共享 exponent 相对 **16-bit BF16/FP16** 确实可省空间：标准 MXFP8 的理论权重位宽少约 48.4%，MXFP4 少约 73.4%。但相对纯 8-bit 权重，标准 MXFP8 的 `8.25` bit/值**并不更小**；普通 W8 也需要自身 scale 元数据，实际文件须用后端编译产物比较。若采用笔记中的**自定义两级 MX4**，理论为 4 bit/值，仍须实现相应解码和算子，不能按 OCP MXFP4 的支持情况推断。

在当前 Toolkit2 2.3.2 上，对 `rk3588` 的[`配置探针原始报告`](../../runs/rknn_numeric_formats/report_extended.json)显示：

| 配置字段 | 候选 | 结果 |
| --- | --- | --- |
| `quantized_dtype` | `w8a8` | 接受，且项目已有实际子图编译 |
| `quantized_dtype` | `w8a16` | 名称合法，但对 `rk3588` 报不支持 |
| `quantized_dtype` | `w4a8` | 报 `Invalid quantized_dtype`，当前 Toolkit2 不接受此名称 |
| `quantized_dtype` | `w4a16` | 名称合法，但报 `not support in 'rk3588'` |
| `quantized_dtype` | `mx4`, `mx8`, `mxfp4`, `mxfp8`, `mxint8` | 均报 `Invalid quantized_dtype` |
| `float_dtype` | `float16`, `bfloat16` | 均接受；本项目四层 BF16/FP16 子图实际编译通过 |
| `float_dtype` | `mx4`, `mx8`, `mxfp4`, `mxfp8` | 均报 `Invalid float_dtype` |
| `quantized_method` | `group32`（默认 `w8a8`） | 报仅支持与 `w4a16` 配合；但当前 `rk3588` 对 `w4a16` 配置又报不支持 |

探针脚本为[`probe_rknn_numeric_formats.py`](../../scripts/probe_rknn_numeric_formats.py)。官方 [`rknn_api.h`](https://github.com/airockchip/rknn-toolkit2/blob/master/rknpu2/runtime/Android/librknn_api/include/rknn_api.h)列有 BF16 张量枚举；[`rknn_matmul_api.h`](https://github.com/airockchip/rknn-toolkit2/blob/master/rknpu2/runtime/Android/librknn_api/include/rknn_matmul_api.h)列有若干普通 INT4 矩阵乘法类型，但没有 MX 类型。因此**普通 INT4、RKLLM 的 W4A16、MXFP4、笔记中的两级 MX4 是不同执行格式**；一个接口支持 INT4 不代表 SmolVLA 子图能原生执行 MX。就当前 Toolkit2 主线路径，MX4/MX8 暂不列入 RK3588 NPU 混合精度候选；可以单独研究软件仿真/打包的数值质量，但不得将其文件变小当作 NPU 加速。未来版本或自定义内核的支持情况不由本探针判定。

`W4A8`/`W8A16` 的逐项新探针记录在[`report_w4a8_w8a16.json`](../../runs/rknn_numeric_formats/report_w4a8_w8a16.json)。这些报错发生在 `rknn.config`，所以未进入 ONNX 编译或数值质量测试；不代表其它软件栈或底层定制 MatMul 的能力。

**能否“包装成 W8 配置”**：MXFP8 每个元素是 FP8 编码，另有每 32 个元素共享的 E8M0 exponent；RKNN W8A8 则按 INT8 值和编译器的 scale/zero point 解码。直接把 8-bit MX 元素码交给 W8A8 会改变数值含义。要保持 MX 的原始数值，必须在计算中逐组应用共享 exponent；[扩展配置探针](../../runs/rknn_numeric_formats/report_extended.json)显示 RK3588 上 `w8a16` 也被拒绝，`w8a8 + group32` 报“group32 只适用于 w4a16”，而该平台的 `w4a16` 又被拒绝。可把 MX 权重**先解码**为 FP16/BF16 再编译，或重新量化成 RKNN W8A8，但生成的计算格式就不再是 MX，不能宣称原生 MX 推理或保有 MX 的运行内存/吞吐收益。若只在磁盘存 MX、运行时解码，须单独计入解码开销和展开后内存。

## 复现与后续决策

```bash
cd /root/qvla
.rknn-probe/bin/python scripts/probe_rknn_numeric_formats.py --output runs/rknn_numeric_formats/report.json
.venv/bin/python scripts/compare_mlp_float_dtypes.py --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets --runs-dir runs --output runs/rknn_float_dtype_compare/report.json
.rknn-probe/bin/python scripts/compile_rknn_mlp_probe.py --onnx runs/rknn_vision11_export_probe/mlp_fp32.onnx --output-dir runs/rknn_vision11_bf16_probe --mode bf16
.rknn-probe/bin/python scripts/check_rknn_mlp_parity.py --mode bf16 --onnx runs/rknn_vision11_export_probe/mlp_fp32.onnx --inputs runs/rknn_vision11_dev_probe/heldout_inputs.txt --output runs/rknn_vision11_bf16_probe/parity.json
```

视觉 11/语言 3 属于高敏感层。下一个完整动作候选应按原运行 dtype 分别试视觉 FP16、语言 BF16，并以原加载模型动作作配对参考，再考虑硬件资源。只在板端确认 BF16 运行、算子落点和性能后将其作为最终硬件格式。
