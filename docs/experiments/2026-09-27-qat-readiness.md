# 首轮 QAT 前的动作路径、联合精度与 RKNN 转换验证

**结论范围**：已锁定[首轮动作专家 W8A8 QAT 目标](../../config/quantization_map_qat_stage1.json)，可以开始此**实验性 QAT 训练**。它不是最终量化图、训练结果或 RK3588 板端部署结论。直接导入 PyTorch/ONNX 已量化 QAT 子图在本轮真实 Linear 上落为 FP16 权重；首轮真实转换须从 QAT 训练后的浮点主权重重建 RKNN W8A8，并与原始 FP 的独立 PTQ 保持同格式。

## 问题、固定输入和方法

检查[候选图 v0](../quantization-map-v0.md)中未测的 `lm_head`、词嵌入、connector、时间投影是否参与实际动作；比较大范围 W8A8 输出舍入的组合风险；检查 connector 的真实 RKNN 子图；验证 QAT 假量化或转后 TorchScript 能否保留为 RK3588 INT8 计算权重。验收首轮训练目标的原则是：选动作代理风险低、涉及一定权重规模且已有代表性 RKNN W8A8 子图可编译的范围。这里**不设动作 MAE 硬阈值**，也不根据源权重比例估算最终收益函数 $G$。

模型 `lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de`，原权重 SHA-256 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`；checkpoint 自带 processor 与本地固定的 SmolVLM2 tokenizer。原数据划分 SHA-256 `ca851a1bdc8fd60ad1e5b8d08dc7c405f971a8d4f999c4f0c2ecef154272d55f`，隔离分区 SHA-256 `f755546a6b074d2fe248333fc42c3dbf30f9b54b9f0fa0b58a16506a54d942c2`。A10 服务器 PyTorch `2.7.0a0+7c8ec84dab.nv25.03`、CUDA `12.8`；RKNN-Toolkit2 `2.3.2`，`target_platform=rk3588`。动作路径使用 `qat_train` 内固定的 40 个**开发** episode，每任务首帧；输出舍入的静态 min/max 则由与之隔离的 40 个 `ptq_calibration` episode 每任务首帧得到。固定每条动作的 seed、processor、10 个 flow 步骤，比较完整 50×7 动作 chunk。冻结测试未使用。

## 1. 动作路径与未测模块

[`probe_action_path.py`](../../scripts/probe_action_path.py)给 `lm_head`、token embedding、connector、时间投影及两层保护 MLP 安装 forward hook，在 40 个开发任务上运行真实动作 chunk，并检查 `lm_head` 与 embedding 的运行存储别名。[逐任务原始报告](../../runs/action_path_v1/report.json)记录每个调用次数和动作形状。

| 模块 | 40 个动作 chunk 的总调用数 | 说明 |
| --- | ---: | --- |
| `lm_head` | **0** | 当前固定动作推理路径未调用；与 embedding **不共享**运行存储 |
| token embedding | 40 | 每条动作一次 |
| connector | 80 | 每条动作两次，对应两路实际视觉输入 |
| 时间投影 in/out | 各 400 | 每条动作各 10 次 |
| 视觉 MLP 11 / 语言 MLP 3 | 80 / 40 | 保护层确实参与推理 |

`lm_head` 的 94,617,600 B 是原 checkpoint 载荷；它在本协议未调用，**仍保留在当前完整 checkpoint**。是否可从一个自定义部署包删除及删除后的加载、动作等价性尚未验证，因此不计入已取得的体积收益。上述结论只覆盖本次 SmolVLA 动作路径，不推断其它生成模式。

[`probe_action_sensitivity.py`](../../scripts/probe_action_sensitivity.py)新增 connector、embedding 与时间投影分组。输出张量使用静态逐张量仿射 INT8：$s=(x_{max}-x_{min})/255$，$z=\operatorname{clip}(\operatorname{round}(-128-x_{min}/s),-128,127)$，$\hat x=s[\operatorname{clip}(\operatorname{round}(x/s+z),-128,127)-z]$。这**只舍入模块输出**，权重保持 FP，因而不是实际 W8A8。40 个独立开发观测的[逐样本报告及量化参数](../../runs/action_missing_groups_v1/report.json)如下；FP 重复执行最大动作差为 0。

| 单组输出舍入 | 平均完整动作 MAE vs FP | 逐观测 p95 | 最大逐观测 MAE | 首动作对记录动作 MAE 增量 |
| --- | ---: | ---: | ---: | ---: |
| connector | 0.002662 | 0.003354 | 0.003574 | -0.000375 |
| token embedding | 0.001968 | 0.002768 | 0.003415 | +0.000019 |
| 时间投影 in/out | 0.001647 | 0.001811 | 0.001831 | +0.000104 |

embedding 的输出舍入**不能证明** RKNN 支持 INT8 embedding/Gather，也不等于量化了 94.6 MB 权重。connector 23.6 MB 源权重值得单独探查，但保留为后续候选。

## 2. 联合候选与首轮范围

用同一 40 个校准和 40 个开发 episode，对 v0 里的目标模块组合做输出舍入；每个候选均从同一原始 FP 模型重新执行。原始[stage 1/2 报告](../../runs/action_mixed_stages_v1/report.json)和[全 v0 报告](../../runs/action_mixed_v0_combo/report.json)含逐观测动作差、校准 min/max、scale/zero point、模块调用次数和初始噪声规则。[实测图](../../figures/qat_stage_selection_v1.png)、[SVG](../../figures/qat_stage_selection_v1.svg)、[绘图数据](../../figures/qat_stage_selection_v1.json)由[`plot_qat_stage_selection.py`](../../scripts/plot_qat_stage_selection.py)生成。

| 候选输出舍入范围 | 活跃模块数 | 涉及源权重载荷 | 平均完整动作 MAE vs FP | 逐观测 p95 | 最大逐观测 MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 仅专家 16 层注意力 + MLP | 80 | 199,720,960 B | **0.001097** | 0.001361 | 0.001693 |
| 再加视觉/语言后段 | 149 | 432,580,096 B | 0.004944 | 0.006916 | 0.007534 |
| v0 全部活跃候选 | 218 | 660,138,496 B | 0.018851 | 0.032271 | 0.042992 |

这三档的源权重载荷由[固定 checkpoint 覆盖清单](../../figures/quantization_map_v0_inventory.json)按分组求和，**不是压缩收益**。逐层响应不能相加；更大组合的动作代理误差明显增加。选“仅专家”为首轮 QAT 目标，是为了先验证质量与真实转换链路，并保留后段扩展候选；并非宣称该档已经达到任务质量门槛。对应 16 层专家的注意力四个投影和 MLP 三个投影，共 **112 个 Linear**；其它层按原始加载行为保留。原[旧 W8 weight-only QAT](2026-09-26-w8-pilot.md)出现质量退化，不能直接重用旧训练配置。

## 3. connector 的真实 RKNN 子图

用[`capture_mlp_calibration.py`](../../scripts/capture_mlp_calibration.py)捕获 40 个校准、40 个独立开发输入，shape 均 `1×64×12288`，逐输入 SHA 在[校准捕获](../../runs/connector_probe_v1/calibration_capture.json)与[开发捕获](../../runs/connector_probe_v1/development_capture.json)。从原加载 FP32 connector 导出 opset 17 ONNX，只有 `MatMul`；[导出报告](../../runs/connector_probe_v1/report.json)，[原 ONNX](../../runs/connector_probe_v1/connector_fp32.onnx)。`rknn.config` 使用 `target_platform=rk3588`、`quantized_method=channel`、W8A8 `quantized_algorithm=mmse`；FP16 不做激活校准。INT8 与 FP16 均导出真实 `.rknn`，并从同一 ONNX 重新构建**主机模拟器**与 40 个开发输入配对。

| 格式 | RKNN 文件 | 文件字节 | 开发输出 MAE vs FP32 ONNX | 开发输出平均余弦 |
| --- | --- | ---: | ---: | ---: |
| FP16 | [文件](../../runs/connector_probe_v1/connector_fp16.rknn) | 23,774,405 | 0.001245 | 0.9999997 |
| W8A8 MMSE | [文件](../../runs/connector_probe_v1/connector_int8_mmse.rknn) | 12,174,367 | **0.244674** | 0.9990477 |

[FP16 编译](../../runs/connector_probe_v1/fp16_compile_report.json)与[FP16 逐样本](../../runs/connector_probe_v1/fp16_parity.json)；[INT8 编译](../../runs/connector_probe_v1/int8_mmse_compile_report.json)与[INT8 逐样本](../../runs/connector_probe_v1/int8_parity.json)已独立备份。当前 INT8 仅在四十个校准输入、单个 FP32 ONNX 子图、MMSE 下比较，较大误差尚不能证明 QAT 后或其它校准方法永不可用；因此首轮图保留 connector 原精度。板端算子落点、完整动作与端到端成本未测。

## 4. QAT → RKNN 真正低比特转换的探针

用真实视觉 10 MLP 的 `fc1`（原加载 FP32，输入 `1×1024×768`）验证两条**不训练**的转换路径，4 个 `ptq_calibration` 输入仅用于这个格式探针，不能代表全数据质量：

1. [`probe_rknn_qat_qdq.py`](../../scripts/probe_rknn_qat_qdq.py)把输入、逐输出通道权重及输出静态 fake quant 导出 ONNX opset 19，得到各 3 个 `QuantizeLinear/DequantizeLinear`；Torch vs ONNX 输出平均绝对差 0.000652，详见[导出报告](../../runs/qat_qdq_vision10_fc1_v2/report.json)。RKNN `do_quantization=False` 在 optimization 3/2 都识别为 QAT 并编译，但[optimization 2 日志](../../runs/qat_qdq_vision10_fc1_v2/compile_opt2.log)明确显示计算 `Conv FLOAT16`、常量权重 `FLOAT16 (3072,768)`、权重内存约 4622 KB；**这不是 W8A8 计算**。
2. [`probe_rknn_qat_torchscript.py`](../../scripts/probe_rknn_qat_torchscript.py)按 PyTorch `fbgemm` eager QAT prepare/convert，把该真实 `Linear` 转为 `torch.qint8` 权重并保存 TorchScript；[报告](../../runs/qat_torchscript_vision10_fc1_v1/report.json)记录打包 dtype、文件 2,429,716 B 和 fake→converted 平均误差 0.03011。RKNN 2.3.2 `load_pytorch` 首先无法解析当前 PyTorch 开发版号 `2.7.0a0`；仅探针中临时把版本**报告字符串**改为 `2.7.0` 后才完成编译。即使如此，[optimization 2 日志](../../runs/qat_torchscript_vision10_fc1_v1/compile_opt2.log)仍显示计算权重为 `FLOAT16 (3072,768)`、约 4622 KB；TorchScript 自身的 QINT8 存储没有变成 RKNN W8A8。此临时版本字符串兼容仅用于查明编译行为，不用于正式训练与交付。

两条路径的 `.rknn` 文件均已备份，见[Q/DQ 产物](../../runs/qat_qdq_vision10_fc1_rknn_opt2/mlp_fp16_rk3588.rknn)和[TorchScript 产物](../../runs/qat_torchscript_vision10_fc1_rknn_opt2/qat_int8_rk3588.rknn)。这里只对一个 Linear 证明**当前两条直接导入路径会落成 FP16**，不泛化到所有 RKNN QAT 模型。Rockchip 的[官方 QAT 示例](https://github.com/airockchip/rknn-toolkit2/blob/master/rknn-toolkit2/examples/pytorch/resnet18_qat/README.md)要求已量化 QAT 模型 `do_quantization=False`；该示例是 ResNet18，不能推断本项目的 Linear/MatMul 同样保持 INT8。

**首轮可执行转换路线**：QAT 使用与 RKNN W8A8 尽量对齐的逐通道权重、静态激活 fake quant 与 STE 更新**浮点主权重**；训练完移除训练用 fake quant，导出该浮点主权重及其它保留层；用独立 `ptq_calibration` 重新估计激活范围，再由已探通的 RKNN W8A8 编译路径生成**真实**低比特子图。它属于 QAT 的训练后校准与真实转换，**不是对已转换低比特权重重复 PTQ**。从原始 FP checkpoint 独立沿同一模块范围、编译格式与校准 episode 做 PTQ 对照。由于 RKNN MMSE 内部阈值未导出，训练 fake quant 与最终 RKNN 的逐层量化参数未证实完全相同，转换后必须测真实动作质量与 fake→real 误差；若差异失控则调整 QAT 模拟或编译设置。

## 结论边界和下一步

首轮 QAT 可以按[锁定的 stage 1 图](../../config/quantization_map_qat_stage1.json)开始**训练实验**，后续再验证视觉/语言后段并决定是否扩大范围。当前既没有 stage 1 的真实 QAT 模型，也没有独立同格式 PTQ、闭环成功率、整模型真实文件大小、RK3588 全模型 RAM/延迟/功耗，故 $G=\mathrm{NA}$。RKNN 代表性 MLP/connector 编译并不保证专家注意力、完整动作链或板端能运行。训练实现须在运行前核对 112 个 Linear 的位置、激活校准与冻结分区排除规则；旧 [`qat_train.py`](../../scripts/qat_train.py) 仅有 weight-only QAT，不可直接当作此图的 W8A8 实现。
