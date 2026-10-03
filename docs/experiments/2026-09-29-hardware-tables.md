# 模块候选配置表与 RK3588 成本表

## 目的与固定输入

为完整 HAQ 精度动作空间提供可复现的硬件成本查表；同一算子、输入/权重 shape、格式和边界只测一个代表签名，不逐个复制权重实例。板测成本不用于直接判定任务质量。

- 固定 SmolVLA checkpoint SHA256：`9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`。
- RKNN Toolkit / Lite2 / runtime：2.3.2；RK3588 NPU `core0`；CPU embedding 环境 Python 3.10.12、NumPy 1.26.4。
- 板端系统 wall clock 比本地慢约两个月（解包时出现未来时间戳告警）；所有延迟用单调时钟计量，不依赖板端日历时间。
- 量化输入/校准集：线性/卷积成本子图用固定 seed `20260929` 合成输入和两条合成校准数组，只测硬件成本；真实 MLP 融合子图用保存的 held-out 模块激活，未把其用于编译校准；注意力和 QKV 精度边界用同一个固定 seed 的合成输入。
- shape 补全依据及逐模块来源：[硬件 shape manifest](../../config/hardware_shape_completion.json)。原始权重覆盖397个参数路径模块、500个张量、906,639,456 B。302个活动 Linear 中291个有输入调用采样，另11个根据匹配的同类捕获签名或模型调用图补齐；`lm_head` 未进入当前动作调用路径。
- 缓存读写按用户要求排除。

## 两张表如何阅读

[模块配置表](../hardware/module_options.csv)逐参数路径列出权重 shape、候选格式、代表 shape、成本测量 ID 与每种格式的测量状态。线性层和 patch Conv2D 的候选集合为 `w8a8`、`float16`、`bfloat16`、`w16a16i`、`w16a16i_dfp`；词 embedding 单列 `native_bf16_row_lookup` 与 `cpu_int8_row_lookup`。Norm/位置参数当前没有独立精度动作，标为 native-only。`current_precision` 只复述已有 PTQ artifact，不是 HAQ 的预选精度。

[RK3588 成本表](../hardware/measured_costs.md)包含100个基础签名及18个融合、转换和 CPU embedding 补测。每项成本记录 shape、格式、模型/权重大小、p50/p95、三轮稳定性和状态。CSV/JSON保存逐项 hash；[补充明细](../hardware/supplemental_costs.md)保留层输出误差的用途边界。

## 基础算子成本矩阵

302个活动 Linear 根据调用采样及 shape manifest 形成19个 Linear shape 签名；每签名5种格式共95项。patch embedding Conv2D 的 `[1,3,512,512]` 输入和 `[768,3,16,16]` 权重另测5种格式。合计100项，每项20次预热、100次计时、3轮；共300份成功板端报告、30,000次有效计时和6,000次预热。每项使用实际 checkpoint 代表权重构建独立子图，RKNN 输入/输出为FP32；调用计时包含Lite2/NPU执行，不含模型加载及上游预处理。

95个 Linear、5个 Conv2D 均完成 Toolkit 编译及三轮板端运行。每项p50/p95是300次有效延迟的第50/95百分位；稳定标记定义为 `max(三轮 p50) / min(三轮 p50) <= 1.2`。9项超过1.2：`3c0d39c688834aa7`、`a275a4c27d22d47e`、`22be0a07b3245214`、`21354414d63b06d5`、`88edc22597b0e53d`、`4948a0119faed73b`、`6800fd88001c4243`、`a0ff7a9a0eef18ed`、`3b413b25aeeaeedc`。板子未锁频，超阈值项保留原始结果，HAQ 最终候选需复测。

新增的35项包含30个补齐 shape 的 Linear 格式配置和5个 patch Conv2D 格式配置。原有65项加上新增35项，得到上述完整100项矩阵。完整报告和编译产物位于 `runs/hardware_cost_v1/<case_id>/` 与 `runs/hardware_cost_v2/<case_id>/`；每个 case 保存 shape、模型/输入 hash、三轮逐次延迟和环境报告，编译格式及配置见 `compile.json` 与编译日志。

## 融合 MLP 与注意力核心

使用真实专家、语言、视觉 MLP 图各自测 FP16 与 W8A8：6种配置、每项20次预热及100次计时×3轮，共18份板端报告、1,800次计时。输入是各子图独立的 held-out 激活；MAE 是板端子图输出对原始 FP32 ONNX 输出的差，不是LIBERO任务通过率。

MLP 的 W8A8/PTQ 与 FP16 编译参数、校准集和模型路径分别记录在专家 `runs/qat_ptq_expert0_original/int8_mmse_compile_report.json`、语言 layer 3 `runs/rknn_language3_int8_probe/int8_mmse_compile_report.json` / `runs/rknn_language3_fp16_probe/fp16_compile_report.json`、视觉 layer 11 `runs/rknn_vision11_int8_probe/int8_mmse_compile_report.json` / `runs/rknn_vision11_fp16_probe/fp16_compile_report.json`。专家 FP16 由 `scripts/prepare_supplemental_hardware_tests.py` 以 `target_platform=rk3588,float_dtype=float16,do_quantization=false` 从对应 FP32 ONNX 构建。输入、参考输出和实际RKNN SHA见 `runs/hardware_supplemental_v1/cases.json` 及板端报告。

| 子图 | 格式 | p50 ms | 模型 B | held-out 输出 MAE |
| --- | --- | ---: | ---: | ---: |
| 专家 MLP | W8A8 | 2.8946 | 4,524,893 | 0.008809 |
| 专家 MLP | FP16 | 6.1600 | 8,910,851 | 0.000111 |
| 语言 MLP layer 3 | W8A8 | 11.8468 | 7,522,271 | 0.824544 |
| 语言 MLP layer 3 | FP16 | 26.9201 | 14,857,285 | 0.001435 |
| 视觉 MLP layer 11 | W8A8 | 32.0934 | 5,293,473 | 0.512492 |
| 视觉 MLP layer 11 | FP16 | 101.6799 | 9,938,567 | 0.002066 |

实测说明融合图延迟必须作为独立配置测量，不能把逐 Linear 的 Lite2 调用延迟简单相加。三个 W8A8 图的成本均低于 FP16，但语言/视觉 MLP 的单次 held-out 输出误差明显较大；这是精度诊断，不据此单独决定任务质量或 HAQ 位宽。专家 W8A8 三轮延迟波动也超过20%，其值需要复测。

另外编译并板测参数化注意力核心代理：`QKᵀ → scale → Softmax → PV`，Q/K/V 输入各为 `[1,15,50,48]`（板端打包输入 shape `[3,1,15,50,48]`），FP16/W8A8各三轮。该图不含真实注意力权重，用合成Q/K/V只检查算子组合的编译与耗时：FP16 p50 4.4983 ms，W8A8 p50 3.7020 ms。FP16配置为 `target_platform=rk3588,float_dtype=float16`；W8A8配置为 `target_platform=rk3588,quantized_dtype=w8a8`，其他参数使用Toolkit默认值，两条合成校准输入。详细编译状态见 `runs/hardware_supplemental_v1/attention_core_source/attention_*_compile.json`。合成输入的 MAE 只用于数值通路检查，不是质量结果；该 proxy 不是完整 Transformer attention block。

## 混合精度转换边界

对真实专家 layer 0 QKV 投影图复用已编译的8种RKNN模型：全W8A8、全FP16，以及仅Q、仅K、仅V、Q+K、Q+V、K+V为FP16的6种混合方式。共同输入 shape 为 `[1,50,720]`，固定 seed 合成，仅比较成本；每项20次预热、100次计时×3轮，24份成功报告、2,400次有效计时。

| 图内配置 | 模型 B | p50 ms |
| --- | ---: | ---: |
| 全 W8A8 | 1,229,200 | 2.1505 |
| 全 FP16 | 2,361,014 | 3.0580 |
| Q FP16，其余 W8A8 | 1,916,816 | 2.4939 |
| K FP16，其余 W8A8 | 1,461,136 | 2.3542 |
| V FP16，其余 W8A8 | 1,461,136 | 2.2449 |
| Q/K FP16，V W8A8 | 2,144,848 | 2.6597 |
| Q/V FP16，K W8A8 | 2,144,848 | 2.6094 |
| K/V FP16，Q W8A8 | 1,689,168 | 2.4156 |

编译器日志可看到混合图中的 `exDataConvert`；量化主体采用Toolkit 2.3.2、RK3588、MMSE、per-channel W8A8，FP16覆盖配置由各自混合配置文件指定，config SHA保存在 `runs/expert_qkv0_hybrid_*_v1/hybrid_*_development_parity.json`。表中是整张QKV投影图的端到端子图成本，不能把两种图的差值解释成单独转换算子的耗时。此处没有用合成输入评估量化质量。逐项模型和计时报告在 `runs/hardware_supplemental_v1/qkv_boundary_*/`；全精度对照配置见 `runs/expert_qkv0_ptq_v1/int8_mmse_compile_report.json` 与 `runs/expert_qkv0_fp16_v1/fp16_compile_report.json`。

## 词 embedding CPU 行量化

使用真实 checkpoint 的 BF16 token embedding `[49280,960]`，对每行采用 `scale=max(abs(row))/127`、round-to-nearest、截断到 `[-127,127]`，scale 存FP32。BF16表为94,617,600 B，INT8权重加行scale为47,505,920 B，表存储减少49.79%。在板上比较177个固定seed token ID的 gather、解码/反量化及 BF16 输出舍入；177取自下游语言 MLP输入长度，ID本身是合成的。每种路径20次预热、100次计时×3轮。

- Native BF16：p50 0.2208 ms，p95 0.2476 ms，输出对自身参考 MAE 0；轮间p50比超过1.2，延迟不稳定。
- CPU INT8 row lookup：p50 0.6390 ms，p95 0.7471 ms，输出对原生BF16 MAE 0.002159；延迟稳定。

查表微基准不含tokenizer、完整文本处理与CPU到NPU传输。INT8节省约一半embedding表空间，但本测试中的CPU查表/反量化慢于直接BF16查表，不能据此声称端到端更快。原始权重、ID及逐次计时hash记录在 `runs/hardware_embedding_v1/`。

## 计算口径、复现和边界

- 原始单次延迟及输入/模型由板端runner记录；每轮p50用于波动判断，汇总p50/p95直接从三轮合并的300个有效样本计算。模型大小取实际 `.rknn` 文件字节数；词 embedding 取权重加量化scale字节数。
- 基础测试脚本：`scripts/build_hardware_tables.py`、`scripts/benchmark_cost_compile.py`、`scripts/benchmark_cost_board.py`、`scripts/summarize_hardware_cost.py`。补充子图脚本：`scripts/prepare_supplemental_hardware_tests.py`、`scripts/summarize_supplemental_hardware.py`；embedding脚本：`scripts/prepare_embedding_cost_test.py`、`scripts/benchmark_embedding_lookup_board.py`。
- 汇总表：[硬件表入口](../hardware/README.md)、[成本实测表](../hardware/measured_costs.md)、[补充表](../hardware/supplemental_costs.md)、[机器可读数据](../hardware/tables.json)。补充原始逐次计时、板端温度、输入和模型hash保留在 `runs/hardware_supplemental_v1/`；embedding逐次延迟在 `runs/hardware_embedding_v1/board_embedding_lookup.json`。
- 100项基础表覆盖了当前可调 Linear/Conv 精度候选；18项补测覆盖代表融合图、注意力组合、QKV混合边界和语言embedding CPU路径。缓存不在测试范围。Norm/残差/位置参数没有独立精度动作；其融合方式最终由完整导出图确认。
- 尚未完成完整 SmolVLA 的 RKNN 全图转换、端到端任务评测、整策略峰值RAM与总延迟。不能把这些子图的结果写成全模型部署结论，也不能用p95或RSS逐层求和代替整图板测。
