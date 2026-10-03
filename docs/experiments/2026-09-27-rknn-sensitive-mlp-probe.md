# RK3588 视觉/语言 MLP 的真实子图编译与主机数值探针

**结论范围**：视觉第 10/11 层、语言第 3/4 层 MLP 的 FP16 与 INT8 W8A8 子图都可由 RKNN-Toolkit2 2.3.2 编译为真实 `.rknn` 文件。在各 4 个隔离开发输入的**主机模拟器重建模型**上，INT8 MMSE 输出 MAE 依次为 0.013936、0.482821、0.812640、0.066083；四者 FP16 都接近 FP32 ONNX。本轮暂把视觉 10、语言 4 列为 INT8 候选，高敏感的视觉 11、语言 3 列为保留浮点精度候选。**后续[BF16 对照](2026-09-27-float-mx-format-probe.md)已证明 BF16 也可编译，且语言层原运行 dtype 为 BF16，因此不能预先定成 FP16。**这仍不是完整 VLA 的混合量化决选，也未在 RK3588 板端执行。

## 问题、原理与判据

[逐层动作探针](2026-09-27-action-layer-selection.md)显示视觉 11、语言 3 的输出 INT8 舍入最敏感，视觉 10、语言 4 相对低敏感。这里分别从真实 checkpoint 导出四个 MLP，使用真实模型执行时的**输入激活**为 RKNN INT8 校准；FP16 无校准。检查 `rknn.build` 和 `export_rknn` 是否成功，并以独立开发输入计算 `MAE=mean(|Y_RKNN-Y_ONNX|)`、余弦相似度。MAE 仅是子图质量代理；若 INT8 偏差明显大于 FP16，先保留 FP16 候选。文件大小记录的是独立子图 `.rknn` 字节，不能相加推断完整模型大小。

## 固定配置与数据隔离

| 项目 | 记录 |
| --- | --- |
| 原始模型 | `lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de`；权重 SHA-256 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`；checkpoint 自带 processor |
| 数据分区 | [`evaluation_partition_v2.json`](../../config/evaluation_partition_v2.json)，SHA-256 `f755546a6b074d2fe248333fc42c3dbf30f9b54b9f0fa0b58a16506a54d942c2`；原 split SHA-256 `ca851a1bdc8fd60ad1e5b8d08dc7c405f971a8d4f999c4f0c2ecef154272d55f` |
| 校准输入 | 任务 0–3 各 1 个 `ptq_calibration` episode 的首帧，episode `85,108,2,40`；每层 4 个真实输入。文件 SHA-256 与形状见[视觉 10](../../runs/rknn_vision10_cal_probe/report.json)、[视觉 11](../../runs/rknn_vision11_cal_probe/report.json)、[语言 3](../../runs/rknn_language3_cal_probe/report.json)、[语言 4](../../runs/rknn_language4_cal_probe/report.json)报告 |
| 独立开发输入 | 任务 0–3 各 1 个 `qat_train` 开发 episode 的首帧，episode `18,1,3,6`；每层 4 个。见[视觉 10](../../runs/rknn_vision10_dev_probe/report.json)、[视觉 11](../../runs/rknn_vision11_dev_probe/report.json)、[语言 3](../../runs/rknn_language3_dev_probe/report.json)、[语言 4](../../runs/rknn_language4_dev_probe/report.json)报告。冻结测试 episode 未使用 |
| 配对执行 | `select_action`、checkpoint processor、各样本固定 PyTorch/CUDA 初始噪声；同一 `.npy` 输入用于 ONNX 与 RKNN 主机模拟器 |
| 导出 | BF16 源 MLP 转 FP32 ONNX，opset 17，静态实际输入形状；[视觉 10](../../runs/rknn_vision10_export_probe/report.json)、[视觉 11](../../runs/rknn_vision11_export_probe/report.json)、[语言 3](../../runs/rknn_language3_export_probe/report.json)、[语言 4](../../runs/rknn_language4_export_probe/report.json)导出报告 |
| RKNN | Toolkit2 `2.3.2`，`target_platform=rk3588`，`quantized_method=channel`，INT8 `quantized_dtype=w8a8`、`quantized_algorithm=mmse`；FP16 `float_dtype=float16`。主机环境为阿里 GPU 服务器，不是 RK3588 |

视觉 10/11 输入固定形状 `1×1024×768`，导出 ONNX 含 `Add, Constant, MatMul, Mul, Tanh`；语言 3/4 输入固定形状 `1×177×960`，含 `MatMul, Mul, Sigmoid`。ONNX ReferenceEvaluator 相对 PyTorch 的最大绝对差依次为 `0.000000954`、`0.000153`、`0.000244`、`0.00000668`。本轮只检查 4 个任务，未证明所有任务/输入长度都兼容这两个静态形状。

## 编译和独立开发输入结果

绘图：[PNG](../../figures/rknn_sensitive_probe_v1.png)、[SVG](../../figures/rknn_sensitive_probe_v1.svg)、[原始绘图数据](../../figures/rknn_sensitive_probe_v1.json)、[脚本](../../scripts/plot_rknn_sensitive_probe.py)。每项数值来自 4 个开发输入的逐元素输出误差，图的纵轴为对数尺度。

| MLP 子图 | 格式 | `.rknn` 字节 | 平均输出 MAE vs FP32 ONNX | 平均余弦 |
| --- | --- | ---: | ---: | ---: |
| 视觉 10 | FP16 | 9,938,567 | 0.000151 | 0.9999975 |
| 视觉 10 | INT8 MMSE | 5,293,473 | **0.013936** | 0.999268 |
| 视觉 11 | FP16 | 9,938,567 | 0.000501 | 0.9999996 |
| 视觉 11 | INT8 MMSE | 5,293,473 | **0.482821** | 0.923343 |
| 语言 3 | FP16 | 14,857,285 | 0.000478 | 0.9999999 |
| 语言 3 | INT8 MMSE | 7,522,271 | **0.812640** | 0.858829 |
| 语言 4 | FP16 | 14,857,285 | 0.000445 | 0.9999992 |
| 语言 4 | INT8 MMSE | 7,522,271 | **0.066083** | 0.997558 |

原始编译记录与逐样本输出误差：视觉 10 [FP16 编译](../../runs/rknn_vision10_fp16_probe/fp16_compile_report.json)、[INT8 编译](../../runs/rknn_vision10_int8_probe/int8_mmse_compile_report.json)、[FP16 parity](../../runs/rknn_vision10_fp16_probe/parity.json)、[INT8 parity](../../runs/rknn_vision10_int8_probe/parity.json)；视觉 11 [FP16 编译](../../runs/rknn_vision11_fp16_probe/fp16_compile_report.json)、[INT8 编译](../../runs/rknn_vision11_int8_probe/int8_mmse_compile_report.json)、[FP16 parity](../../runs/rknn_vision11_fp16_probe/parity.json)、[INT8 parity](../../runs/rknn_vision11_int8_probe/parity.json)；语言 3 [FP16 编译](../../runs/rknn_language3_fp16_probe/fp16_compile_report.json)、[INT8 编译](../../runs/rknn_language3_int8_probe/int8_mmse_compile_report.json)、[FP16 parity](../../runs/rknn_language3_fp16_probe/parity.json)、[INT8 parity](../../runs/rknn_language3_int8_probe/parity.json)；语言 4 [FP16 编译](../../runs/rknn_language4_fp16_probe/fp16_compile_report.json)、[INT8 编译](../../runs/rknn_language4_int8_probe/int8_mmse_compile_report.json)、[FP16 parity](../../runs/rknn_language4_fp16_probe/parity.json)、[INT8 parity](../../runs/rknn_language4_int8_probe/parity.json)。编译得到的 `.rknn` 文件及 ONNX、输入 `.npy` 已从服务器备份至 Git 忽略的 `runs/`。视觉 11/语言 3 [传输包](../../runs/qvla_rknn_sensitive_probe_20260927.tar.gz)本地与服务器 SHA-256 均为 `75dee9a1ff6696b38ec30e2a9a6d83463e8495439d32f41e7897abf3252a71c1`；视觉 10 [传输包](../../runs/qvla_rknn_vision10_probe_20260927.tar.gz)双方 SHA-256 均为 `94c332103b2e15d72eefe3ae17077d903c32c5ae2bcf578a5f988906c3b668ff`；语言 4 [传输包](../../runs/qvla_rknn_language4_probe_20260927.tar.gz)双方 SHA-256 均为 `af5c8d767846c11978435c0efd059514bd495b0abf0f426d3b15fcafa3626427`。

这里的 parity 工具从 ONNX 重新 `build` 后调用 RKNN **主机模拟器**，未对导出的 `.rknn` 文件在真实 RK3588 上执行；模拟器耗时不作为板端延迟。原始 parity JSON 的 `scope` 沿用了旧版工具的“action-expert MLP”字样，属于元数据标签错误；实际模型身份由各报告的 ONNX 路径和导出报告确认，现已修正脚本标签。INT8 校准只有 4 个任务且未扫描 `normal/KL`、更多代表性帧或混合层内格式，因此较大的误差不能证明 INT8 对这两层永远不可用。真实板端算子分配、边界转换、完整模型字节、峰值内存、功耗、动作质量及闭环成功率均**未测量**。

## 复现命令与下一步

以视觉 11 为例，语言 3 把 `vision_model.encoder.layers.11.mlp` 换成 `text_model.layers.3.mlp` 并更换输出目录：

```bash
cd /root/qvla
.venv/bin/python scripts/capture_mlp_calibration.py --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets --dataset-root data/libero --splits data/libero_splits.json --partition config/evaluation_partition_v2.json --output-dir runs/rknn_vision11_cal_probe --module model.vlm_with_expert.vlm.model.vision_model.encoder.layers.11.mlp --split-name ptq_calibration --max-tasks 4 --frames-per-task 1
.venv/bin/python scripts/capture_mlp_calibration.py --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets --dataset-root data/libero --splits data/libero_splits.json --partition config/evaluation_partition_v2.json --output-dir runs/rknn_vision11_dev_probe --module model.vlm_with_expert.vlm.model.vision_model.encoder.layers.11.mlp --split-name qat_train --max-tasks 4 --frames-per-task 1
.venv/bin/python scripts/export_expert_mlp_probe.py --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets --output-dir runs/rknn_vision11_export_probe --module model.vlm_with_expert.vlm.model.vision_model.encoder.layers.11.mlp --sample-input runs/rknn_vision11_cal_probe/task_00.npy
.rknn-probe/bin/python scripts/compile_rknn_mlp_probe.py --onnx runs/rknn_vision11_export_probe/mlp_fp32.onnx --output-dir runs/rknn_vision11_fp16_probe --mode fp16
.rknn-probe/bin/python scripts/compile_rknn_mlp_probe.py --onnx runs/rknn_vision11_export_probe/mlp_fp32.onnx --output-dir runs/rknn_vision11_int8_probe --mode int8 --algorithm mmse --dataset runs/rknn_vision11_cal_probe/rknn_dataset.txt
.rknn-probe/bin/python scripts/check_rknn_mlp_parity.py --mode fp16 --onnx runs/rknn_vision11_export_probe/mlp_fp32.onnx --inputs runs/rknn_vision11_dev_probe/heldout_inputs.txt --output runs/rknn_vision11_fp16_probe/parity.json
.rknn-probe/bin/python scripts/check_rknn_mlp_parity.py --mode int8 --algorithm mmse --dataset runs/rknn_vision11_cal_probe/rknn_dataset.txt --onnx runs/rknn_vision11_export_probe/mlp_fp32.onnx --inputs runs/rknn_vision11_dev_probe/heldout_inputs.txt --output runs/rknn_vision11_int8_probe/parity.json
```

下一步把视觉 11、语言 3 的 BF16/FP16 作为需配对比较的保护候选，视觉 10、语言 4 的 INT8 作为低精度候选；扩大隔离校准集并比较实际动作质量。只有累积的真实权重/激活量化模型与闭环评测通过后才确定 QAT 共用的位宽方案。
