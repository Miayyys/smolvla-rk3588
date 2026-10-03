# RK3588 动作专家 MLP：扩展校准与截断探针

**范围**：SmolVLA 第 0 层动作专家 MLP，一个 `[1,50,720]` 输入子图；RKNN-Toolkit2 2.3.2 主机编译和模拟器。这里没有完整 VLA 动作、闭环成功率或 RK3588 板端数据。

## 问题与采样

[首轮探针](2026-09-26-rknn-mlp.md)每任务只取首帧和首个 MLP 调用，覆盖的扩散过程和相机观测有限。本轮用同一个原始 checkpoint、processor、ONNX 和 episode 划分，每任务仍各取一个校准 episode 和一个测试 episode，但对每个 episode 取首帧、末帧，以及每次动作生成中 10 次 MLP 调用的第 0、4、9 次：40 任务 × 2 帧 × 3 次调用 = **240 个校准激活和 240 个测试激活**，每个 `[1,50,720]`。按 `0 + 1009×episode_index + 9176×frame_rank + 17×task_index` 固定采样噪声；调用 checkpoint 自带的前处理器。校准和测试 episode 不重叠，采集报告附逐文件 SHA-256。

原始权重 `lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de`，SHA-256 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`；划分 SHA-256 `ca851a1bdc8fd60ad1e5b8d08dc7c405f971a8d4f999c4f0c2ecef154272d55f`；FP32 ONNX SHA-256 `fb9e669995abff913834d0e0e061d787ed1442c4f640bccddf5d95d74057474b`。采集实现见 [`capture_mlp_calibration.py`](../../scripts/capture_mlp_calibration.py)，原始索引见 [`校准报告`](../../runs/rknn_expert_mlp_calibration_v2/report.json)和[`测试报告`](../../runs/rknn_expert_mlp_heldout_v2/report.json)。两个激活集合存于 Git 忽略的 `runs/`，可由报告与脚本重新生成。校准 `|x|` 最大值 6.28125，测试最大值 6.15625。

**评测边界**：本轮先在已指定为冻结测试的 40 个 episode 上查看了 `normal`、`kl_divergence` 和 `mmse` 的结果。这构成算法选择信息泄漏。因此下面第一组 RKNN 排名是**探索结果**，不能当作最终测试。随后从 `qat_train` 中单独取 40 个开发 episode 复核，后续 QAT 训练须排除这些开发 episode；本轮查看过的 40 个测试 episode 从最终冻结评价中排除。剩余 **213 个测试 episode**，40 个任务各仍有 3–7 个。固定 ID 在 [`evaluation_partition_v2.json`](../../config/evaluation_partition_v2.json)。独立的手写截断扫描也查看了同一批旧测试激活，只能作诊断。

## 截断技术与曲线

用全部校准 MLP 输入元素构造 `|x|` 的 2048-bin 直方图，范围 `[0,6.28125]`，每 16 bin 扫一个候选截断阈值 `α`，另含全范围端点。对称 INT8 诊断器使用 `s=α/127`、`z=0`、`q=clip(round(x/s),-127,127)`、`x̂=sq`。校准 MSE 使用直方图 bin 中心估计，选出 `α` 后再对全部校准和测试元素逐值计算 MAE、MSE 与饱和比例。这是**单一输入激活的独立诊断器**，不等于 RKNN 编译器的实际内部量化参数。

KL 代理：把候选阈值之外的直方图质量合并到最后一个保留 bin，将保留的 bin 合并为 127 个正量化 bin，再按原始非零 bin 均匀展开为重建分布 `Q`；对 `P` 和 `Q` 加 `10⁻¹²` 后归一化，计算 `D_KL(P‖Q)=ΣPᵢlog(Pᵢ/Qᵢ)`。这套基于 `|x|` 的近似及其候选空间**不是 RKNN `kl_divergence` 的复现**；RKNN 后续 `step1` 生成了另外一套实际激活参数，见[混合量化实验](2026-09-27-rknn-hybrid.md)。完整候选表、公式实现和阈值—KL/MSE/饱和曲线分别在 [`clipping_scan.json`](../../runs/rknn_expert_mlp_clipping_v2/clipping_scan.json)、[`analyze_mlp_clipping.py`](../../scripts/analyze_mlp_clipping.py)和[图：截断扫描](../../figures/rknn_mlp_clipping_v2.png)（[SVG](../../figures/rknn_mlp_clipping_v2.svg)）。

| 独立诊断规则 | 校准选出的 `α` | `s` | 校准饱和 | 测试饱和 | 测试激活 MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Min/max | 6.281250 | 0.04945866 | 0 | 0 | 0.0002036482 |
| 最小直方图 MSE | 5.397949 | 0.04250354 | 0.01375% | 0.01448% | 0.0001607928 |
| 最小 KL 代理 | 0.588867 | 0.00463675 | 50.40139% | 50.28183% | 0.3954042 |

MSE 截断在此**单一输入激活重建指标**上较 min/max 降低约 21.0%，尚未证明端到端子图或动作收益。该 KL 代理的最小值为 0.00672650，却截断约半数元素，导致测试激活 MSE 极高，因此此规则的最小 KL 阈值被判为无效，不送入编译器。该失败提示要检查直方图代理的目标与截断约束，不能推断 RKNN 原生 KL 校准失败。

## RKNN 原生校准方法对照

固定 ONNX、240 个校准输入、`target_platform=rk3588`、`quantized_method=channel`、`quantized_dtype=w8a8`、`float_dtype=float16`，分别设置 RKNN 的 `quantized_algorithm=normal/kl_divergence/mmse`；FP16 是同子图的浮点参照。RKNN 官方提供 [MMSE 算法示例](https://github.com/airockchip/rknn-toolkit2/blob/master/rknn-toolkit2/examples/functions/quantize_algorithm_mmse/README.md)。同一批 240 个测试激活在主机上以 FP32 ONNX Runtime 为参考，RKNN 模拟器从 ONNX **重新 build** 后逐输入计算输出 MAE 和余弦相似度。编译和检查脚本分别为 [`compile_rknn_mlp_probe.py`](../../scripts/compile_rknn_mlp_probe.py)、[`check_rknn_mlp_parity.py`](../../scripts/check_rknn_mlp_parity.py)。

| 方案 | 编译秒数 | `.rknn` 字节 | 240 输入平均输出 MAE | 最大逐输入 MAE | 最低余弦相似度 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP16 | 未重新计时 | 8,910,851 | 0.00008166 | 0.00010942 | 0.99999946 |
| INT8 normal | 5.160 | 4,524,893 | 0.01144322 | 0.01209607 | 0.99818516 |
| INT8 KL | 6.588 | 4,524,893 | 0.01093688 | 0.01769519 | 0.99816442 |
| INT8 MMSE | 172.636 | 4,524,893 | 0.00944018 | 0.01136893 | 0.99884349 |

此探索集上，MMSE 平均输出 MAE 比 normal 低 **17.50%**、比原生 KL 低 **13.68%**；逐任务平均输出 MAE 对 normal 与 KL 均为 40/40 更低。相同文件字节仅说明此子图的文件体积相同，不能推断量化参数或执行时间相同；上表“编译秒数”也不是推理延迟。逐帧/调用分组曲线见[图：RKNN 校准对照](../../figures/rknn_mlp_calibration_v2.png)（[SVG](../../figures/rknn_mlp_calibration_v2.svg)），数值汇总见 [`calibration_compare.json`](../../figures/calibration_compare.json)。

结构化原始结果：[`normal 编译`](../../runs/rknn_expert_mlp_probe_v2/int8_compile_report.json)、[`KL 编译`](../../runs/rknn_expert_mlp_probe_v2/int8_kl_divergence_compile_report.json)、[`MMSE 编译`](../../runs/rknn_expert_mlp_probe_v2/int8_mmse_compile_report.json)；[`FP16`](../../runs/rknn_expert_mlp_probe_v2/fp16_heldout_parity.json)、[`normal`](../../runs/rknn_expert_mlp_probe_v2/int8_normal_heldout_parity.json)、[`KL`](../../runs/rknn_expert_mlp_probe_v2/int8_kl_divergence_heldout_parity.json)、[`MMSE`](../../runs/rknn_expert_mlp_probe_v2/int8_mmse_heldout_parity.json)逐输入误差。INT8 RKNN SHA-256 分别为 normal `97e9edc2f0e9eaf5bfb7284c970e70ed8cc0d5de4ba6f2a02356a02713c410a8`、KL `a65de7e5209f20da6d1ceff427aa86e7ccad06cea055611fafe53cc407cf8c90`、MMSE `0575c8b69101cacb67e87aac8d2e6e9c05ebd1580c4b4f04c6394150d1e335d0`。产物在服务器 `/root/qvla/runs/rknn_expert_mlp_probe_v2/`。

复现关键命令（服务器 `/root/qvla`，先按已有环境放置模型、数据和脚本）：

```bash
.venv/bin/python scripts/capture_mlp_calibration.py --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets --dataset-root data/libero --splits data/libero_splits.json --output-dir runs/rknn_expert_mlp_calibration_v2 --split-name ptq_calibration --frames-per-task 2 --step-samples 3 --seed 0
.rknn-probe/bin/python scripts/compile_rknn_mlp_probe.py --onnx runs/rknn_expert_mlp_probe/expert_layer0_mlp_fp32.onnx --output-dir runs/rknn_expert_mlp_probe_v2 --dataset runs/rknn_expert_mlp_calibration_v2/rknn_dataset.txt --mode int8 --algorithm mmse
.rknn-probe/bin/python scripts/check_rknn_mlp_parity.py --onnx runs/rknn_expert_mlp_probe/expert_layer0_mlp_fp32.onnx --dataset runs/rknn_expert_mlp_calibration_v2/rknn_dataset.txt --inputs runs/rknn_expert_mlp_heldout_v2/heldout_inputs.txt --mode int8 --algorithm mmse --output runs/rknn_expert_mlp_probe_v2/int8_mmse_heldout_parity.json
```

## 独立开发集复核与当前候选

按相同采样规则，从 `qat_train` 每任务另取 1 个 episode，共 240 个开发激活；与上述 40 个校准 episode 和旧测试 episode 均不重叠。采样报告 [`report.json`](../../runs/rknn_expert_mlp_development_v2/report.json)记录每个 episode、帧、MLP 调用、噪声 seed 与激活 SHA-256；这 40 个 episode 已由固定分区清单排除出后续 QAT 训练。用**相同的 240 个 PTQ 校准激活**从同一 ONNX 分别构建三种 RKNN 模拟器，并在开发激活上比较 FP32 ONNX 输出。

| 方案 | 开发集平均输出 MAE | 最大逐输入 MAE | 最低余弦相似度 |
| --- | ---: | ---: | ---: |
| FP16 | 0.00008135 | 0.00011035 | 0.99999952 |
| INT8 normal | 0.01144585 | 0.01210967 | 0.99815911 |
| INT8 KL | 0.01084569 | 0.01823490 | 0.99815357 |
| INT8 MMSE | **0.00941949** | 0.01254902 | 0.99882072 |

开发集上 MMSE 的平均输出 MAE 比 normal 低 **17.70%**、比原生 KL 低 **13.15%**，40 个任务的任务均值分别对两者都是 40/40 更低。因此**仅针对这个 MLP 的主机数值代理指标**，暂定 RKNN 原生 MMSE 为后续 INT8 候选。注意 normal 的单输入最大 MAE 略小于 MMSE；候选依据是本轮比较的平均输出 MAE，尚未经过动作或板端验证。开发集[图](../../figures/rknn_mlp_calibration_development_v2.png)（[SVG](../../figures/rknn_mlp_calibration_development_v2.svg)）、[分组汇总](../../figures/calibration_development_compare.json)及逐输入原始报告：[`FP16`](../../runs/rknn_expert_mlp_development_probe_v2/fp16_development_parity.json)、[`normal`](../../runs/rknn_expert_mlp_development_probe_v2/int8_normal_development_parity.json)、[`KL`](../../runs/rknn_expert_mlp_development_probe_v2/int8_kl_divergence_development_parity.json)、[`MMSE`](../../runs/rknn_expert_mlp_development_probe_v2/int8_mmse_development_parity.json)。

```bash
.venv/bin/python scripts/capture_mlp_calibration.py --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets --dataset-root data/libero --splits data/libero_splits.json --output-dir runs/rknn_expert_mlp_development_v2 --split-name qat_train --frames-per-task 2 --step-samples 3 --seed 0
.rknn-probe/bin/python scripts/check_rknn_mlp_parity.py --onnx runs/rknn_expert_mlp_probe/expert_layer0_mlp_fp32.onnx --dataset runs/rknn_expert_mlp_calibration_v2/rknn_dataset.txt --inputs runs/rknn_expert_mlp_development_v2/heldout_inputs.txt --mode int8 --algorithm mmse --output runs/rknn_expert_mlp_development_probe_v2/int8_mmse_development_parity.json
```

## 判断与限制

此轮证实扩展采样能提供更广的真实 MLP 激活，且 RKNN 三种 INT8 算法均能编译；独立开发集支持 MMSE 作为该 MLP 的**候选校准算法**。当前不可把 MMSE 定为完整模型的混合精度/QAT 目标，也不可把这个 MLP 子图的 INT8 结果推广到视觉、语言、完整动作专家或 RK3588 板端。主机 RKNN 环境仍使用超出官方依赖声明上限的 PyTorch 2.7，正式复现需在受支持环境核对。下一步扩展子图层敏感度、官方 INT8/FP16 混合量化和 QAT 对齐；完整动作与板端测量仍是正式取舍依据。
