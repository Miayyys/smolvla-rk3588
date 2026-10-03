# V1 QAT RKNN后端数值修复

## 起点

[三个短任务](2026-10-02-v1-qat-board-short-tasks.md)所选RKNN适配版本0/3，原FP及历史FP16板端2/3。固定输入回放动作MAE0.192；同master FP16视觉替换降低为0.125，但仍7/50夹爪符号不同。目的为定位和修复后端数值差异，质量优先；任何精度/校准修改均作为独立适配候选，检查体积缩小40%的约束，不声称保持原HAQ图的数值等价。

## 固定边界隔离检查（已完成）

从同一QAT FP32 master完整回放捕获的边界输入，独立运行RKNN语言和专家。所有输入/output SHA、精确名字和shape保存在 `runs/qat_v1_rknn_deploy_v1/{prefix,expert}_isolated_report.json`，引用源master/样本/40校准划分及hash均与[部署记录](2026-10-02-v1-no-teacher-rknn-deployment.md)一致。边界文件不是本次校准数据，噪声保持不变。

语言输出hidden MAE1.024164、RMSE1.417766、相对RMSE1.250418；key0 MAE0.285021、相对RMSE0.532909，所有33个输出最大相对RMSE1.326173。语言的独立误差明显，不能把完整回放失败只归因视觉。动作专家单步velocity MAE0.051452、RMSE0.072255、相对RMSE0.069224；单步偏差较小但不能推断10步或闭环没有问题。relative_RMSE = RMSE(actual,reference) / max(RMS(reference),1e-12)，仅用于定位，不能代替任务成功率。

### 首轮对照（均已完成）

1. 视觉改用KL散度校准，仍W8A8/channel，相同40个独立校准episode/80路图像及同master ONNX；实际结果见后续对照。normal与KL分别保留目录/产物，脚本通过显式algorithm选择，不覆盖原候选。KL旨在改变激活截断阈值，实际是否改善以板测为准，不预设结论。
2. 语言同master FP16转换作为诊断，固定master边界输入和其他子图。首轮磁盘满失败，不代表后端不支持。只清理已结束/失败编译的check-number ONNX调试快照20个，共9,225,874,834 B；源ONNX、权重、校准、最终图与日志均保留，服务器compiler_debug_cleanup.json记录逐文件路径和大小。重启编译成功，固定输入结果见后续对照。

脚本 `scripts/rknn_board_boundary_audit.py` 保存固定输入独立子图实测，不把执行成功标为精度通过。FP16诊断不自动成为部署方案，须先检查质量和总实际文件大小；不存在闭环成功率改善的已测结论。

### 非参数计算保留FP16的语言修复候选（已完成板测，整体无改善）

复用同source ONNX SHA与normal算法的语言hybrid profile，保留原权重位宽例外。将中间图中非带权Conv/Gemm/MatMul的可量化输出设置FP16，同时将带权投影的激活输入设INT8，避免意外把整个权重投影变FP16；特殊语言down保持INT16。681个FP16 tensor、64个显式INT8投影输入、112个带权计算节点。实际编译日志包含INT8 Conv权重，最终文件172,673,382 B，比原语言图仅增加884,870 B；编译31.09秒，SHA `b868df198dd74f836459feb40e263c754745e64d52d4968508a277f2e9db3728`。不能由配置数量直接推断每个融合算子执行精度，实际仍需输出核对。

该实验检验是否RKNN将归一化、残差、注意力等额外量化造成误差（本地QAT主要量化权重投影）；它是后端修复候选，非重新RL搜索结果。先比较相同master边界，再完整动作与闭环任务。

`master_action_audit.json` 显示同一回放GPU pack相对FP32 master动作MAE0.007438/夹爪符号0差异，原板端INT8相对master MAE0.191587/夹爪7差异。说明该输入的巨大差异不单是pack参考与FP32 master本身不一致。

## 后续对照与无效结果

- 原语言FP16同master固定输入：hidden MAE0.002388、相对RMSE0.004774；key0 MAE0.002046。这排除了该输入下导出/FP16后端本身导致1.0级误差的解释，仍不能单独区分所有INT8量化位置。
- 视觉KL同master固定输入：features MAE2.667286/RMSE3.643797，比normal MAE6.270622降低57.46%，仍明显偏大。编译529.59秒、111,952,847 B，SHA `f6c014d86fb6315dfcc4fc03968260523b6cfb90c50c22d5f0222692b2ede347`；不据此宣称闭环已改善。
- 语言非参数FP16修复：hidden MAE1.023400/相对RMSE1.246655，基本未改善；key0 MAE0.225020。保留为无效/有限改善结果，不替换原部署模型。
- 尝试将graph input `prefix`直接写入custom_quantize_layers失败：Invalid operands name；custom API用于算子输出，不能按入口名字直接强制FP16。失败日志 `prefix_float_inputs_compile_console.log`，已移除无效脚本选项。
- 完整回放使用FP16视觉+FP16语言+原INT8混合专家：动作相对GPU pack MAE0.029730、夹爪1/50符号不同（此前0.192/7个），仅单回放无warmup，诊断候选，不是满足40%约束的最终策略。

## 入口动态范围假设与可逆缩放（已测，无效）

固定master输入图像128 token RMS241.654/absmax7130.523，文字48 token（含padding）RMS3.185/absmax74.071。原hybrid profile的prefix量化区间[-8130.87890625,3119.26806640625]，按非对称INT8重建公式步长44.118223、zero56，该固定输入文字元素约99.92%落在表示零的编码。是依据profile范围的解析重建，不是内核trace；不能单凭这一项宣布完整失败唯一原因。原始计算/输入hash保存在prefix_input_quantization_audit.json。

可逆入口缩放：只从40个独立校准prefix收集每个位置/通道最大绝对值，S=max(max_abs_calibration,1)，CPU输入X/S，图内Mul恢复(X/S)*S，然后仍使用带权投影W8A8、语言原INT16例外与FP16非参数计算。新的图、校准输入、固定回放与scale单独保存 `runs/qat_v1_rknn_prefix_balanced_v1`。不使用本次回放拟合scale；手工固定入口范围[-2,2]为校准normalized范围的2倍预留，不从任务成功率选择。该入口平衡是硬件接口适配，不等同于已实施全网络SmoothQuant，也不是重新RL结果。

浮点ONNX同master边界33输出parity全部通过rtol/atol2e-4，精确输入hash、scale SHA、公式和输出误差见balance_export_report.json；实际RKNN量化结果如下。部署必须将CPU缩放与图内恢复一起启用，单独换图会产生错误。


### 可逆入口缩放实测

`runs/qat_v1_rknn_prefix_balanced_v1/prefix_balanced_isolated_report.json`：hidden MAE1.023386、key0 MAE0.225029，未恢复语言质量。FP16视觉＋平衡语言＋INT8专家完整回放动作MAE0.121488、夹爪仍7/50符号变化，见 `prefix_balanced_full_report.json`。平衡图173,028,655 B、SHA `a9da8d346c40d5771978e6bc3ee75f17eea6de40ccc69d4d7c613f22a8b3866e`。这些结果不支持把入口动态范围认定为唯一根因；该版本不替换部署版本。

## 归一化输出保留FP16（已完成）

首个非参数修复把投影输入设INT8时也覆盖了共享Norm输出。新 `--float-nonparam --preserve-norm` 保留Norm与非参数输出FP16，明确请求带权节点输出INT8；实际编译日志检查Norm FLOAT16及独立转换节点，不能仅依据custom配置判断融合算子精度。语言745个FP16 tensor、112个INT8加权输出请求。脚本记录改用 `int8_tensors` 与 `int8_request_location`，避免把输出请求误写成投影输入数量；旧实验JSON保留原始字段，实际含义见本段。

语言图172,635,622 B，SHA `7c567a6d1ddc016787be3030e64ac8272343f39594b3af2d4c9db0f3ab232883`。固定输入key0 MAE从0.285021降到0.033432，relative_RMSE从0.532909降到0.039623，但最终hidden MAE仍1.022915、后续KV明显偏离。例如key4 MAE1.1453、value4 MAE0.5759，value15 MAE0.8958。**早期输出改善不等于完整语言恢复**。原始33输出报告 `prefix_preserve_norm_isolated_report.json`。实际逐层key/value相对RMSE对照图保存为 `runs/qat_v1_rknn_deploy_v1/prefix_layer_error.png`（三版使用相同固定master输入，非任务成功率图）。

视觉同样保护非参数计算未改善：

| 同master视觉版本 | feature MAE | 相对RMSE | 结论 |
| --- | ---: | ---: | --- |
| 原normal INT8 | 6.270622 | 见原报告 | 有明显偏差 |
| KL INT8 | 2.667286 | 0.555066 | 偏差降低，仍不合格 |
| FP16 | 0.156662 | 见原报告 | 大幅接近master |
| 非参数FP16、投影输入INT8 | 7.733449 | 见原报告 | 无效 |
| Norm保留FP16、投影输出INT8 | 7.860743 | 1.550450 | 无效 |
| 主干FP16、仅两块connector INT8 | 8.657827 | 1.691925 | 无效 |

报告分别为 `vision_{kl,float_nonparam,preserve_norm,connector_int8}_isolated_report.json`。最后候选200,844,853 B，SHA `7ed49871b7109c21e45e686aa962c0608c6072049e05b3e34a0b5b6edeb54168`，文件相对视觉全FP16减少约11.8MB，但质量明显退化，不按小文件选择它。不得把这些修复候选称为重新完成HAQ搜索。

## CPU权重无损打包与FP16前端诊断候选

同一CPU INT8数组从未压缩 `np.savez` 改为 `np.savez_compressed`：47,546,790 B →36,628,908 B；逐数组名字、dtype与数值完全一致。SHA `0fa0a04beec0a9c48ae5948e0dcfbaabd013194b63c99a9949480d07df1d5daa`。仅无损文件存储优化，不是新增量化或执行内存收益。

板端候选 `/root/qvla_board_test/qat_v1_repair_fp16_frontend_v1/` 使用同QAT master FP16视觉、FP16语言、原corrected量化专家和该CPU文件。三图＋CPU共692,615,798 B，相对906,712,520 B原checkpoint减少23.61%，不满足40%条件；完整运行资产还要加前处理文件。清单、hash与回放保存 `runs/qat_v1_repair_fp16_frontend_v1/`。

单次无warmup完整原始观测回放MAE0.029730、RMSE0.107372、最大误差1.89506、夹爪1/50符号变化；峰值RSS1,511,148 KiB。这证明该固定输入误差明显下降，不能证明任务恢复，也不能把单次时间当p50/p95。

2026-10-03已尝试重测Spatial/Object task0（与之前失败筛查同init0/seed0），但SSH在握手阶段超时，常驻runtime未初始化、任务没有开始。日志 `runs/qat_v1_repair_fp16_frontend_short_tasks_v1/{rollout,board_runtime}.log`，不能将本次连接失败计为任务失败。

## 语言KL＋Norm保护（编译完成，板测待测）

相同独立40校准episode，`--algorithm kl_divergence --float-nonparam --preserve-norm --bf16-as-fp16`，保留原语言INT16例外；编译165.069秒，172,635,622 B，SHA `6f06bca86059cac785818438f833e202c361f5e38517103d59dc833211433179`。文件 `prefix_selected_kl_divergence_float_nonparam_preserve_norm.rknn`，报告 `prefix_kl_preserve_norm_compile_report.json`。本地下载完成且文件大小/SHA与编译报告逐项核对一致；实际板测待板子恢复后进行，不将编译成功写成质量改善。

下一步先验证FP16前端短任务与语言KL独立输出；若语言仍失真，使用逐算子量化误差分析定位首次显著偏离的计算，再决定原生量化器对齐的训练或局部精度保护。仍按闭环质量优先且最终体积≥40%条件选择，不能拿离线MAE代替最终任务评价。


### 2026-10-03恢复连接后的实测（进行中）

板子恢复连接，KL图板端SHA与编译报告一致。严格所需资产清点（3个图＋CPU参数＋图像/文本/状态前处理配置；不含公共runtime、代码和回放/诊断文件）共542,679,241 B，相对原checkpoint减少40.1487%，达到40%文件约束。逐文件实际大小保存 `runs/qat_v1_repair_kl_prefix_v1/package_inventory.json`；CPU无损压缩贡献单独记录，不将其称为量化收益。总文件大小不代表峰值执行内存。

FP16前端诊断的两个短任务复测已启动，路径 `runs/qat_v1_repair_fp16_frontend_short_tasks_v2`；此前v1目录保留SSH超时原始记录。本轮真实任务结果待完成后填写，不与KL版本混淆。


### FP16前端两个短任务完成

`runs/qat_v1_repair_fp16_frontend_short_tasks_v2/summary.json`：原FP2/2、板端FP16前端＋量化专家2/2，初态与共同query噪声hash配对核对通过。Spatial task0原FP78步/板79步，Object task0原FP125步/板129步，共4回合54.28秒（不含GPU模型载入）。与此前原native量化部署这两个任务0/2相比，诊断版本恢复2个成功；这只覆盖2个任务，不推导全面成功率，且该版本文件压缩不足40%。

### KL语言独立板测完成

相同固定master边界：hidden MAE0.653138、RMSE1.061453、相对RMSE0.936163；key0 MAE0.266773、value0 MAE0.006780。相对原normal hidden MAE1.024164有所下降，但仍远大于FP16的0.002388，Norm保护的早期key优势也未保留；不能因此单独否定或认定最终任务质量。原始33输出、输入/model/reference SHA保存在 `runs/qat_v1_rknn_deploy_v1/prefix_kl_preserve_norm_isolated_report.json`。正在执行完整回放与相同短任务对照。


### KL版本完整回放完成

`runs/qat_v1_repair_kl_prefix_v1/board_raw_report.json/.npz`：原始RGB、文本与状态在板端前处理，2次FP16视觉、1次混合KL语言和10次量化专家，NPU_CORE_0。相对所选GPU pack动作MAE0.087892、RMSE0.270519、最大1.920386，夹爪仍7/50符号变化；相对最初native MAE0.192有所降低，但不及FP16前端0.029730。1次warmup＋3次重复输出完全一致，p50 7961.52ms、p95 7972.01ms、峰值RSS1,261,348 KiB。重复次数较少且启动环境/日期与历史不同，不直接由此宣称板端加速。

这些数值只用于定位；仍按两个配对真实闭环任务判断质量，不因MAE单独淘汰。KL版本短任务路径 `runs/qat_v1_repair_kl_prefix_short_tasks_v1`，固定init0/env_seed0/noise seed100000与200000，GPU原FP作为同时重测对照；任务运行中。


### KL版本两个短任务完成：仍退化

`runs/qat_v1_repair_kl_prefix_short_tasks_v1/summary.json`：原GPU FP2/2、KL板端0/2，两个任务均运行到280步上限；4回合105.66秒，常驻runtime与已测完整CLI回放完全一致，初态和共同query噪声hash均通过配对。Spatial原FP78步成功、KL失败；Object原FP125步成功、KL失败。该版本即使文件减小40.15%、固定动作MAE降低，仍未通过这两个任务，不能作为质量合格的部署版本。

| 同两个task0配对协议 | 原FP | 板端成功 | 核心或运行资产缩小 | 结论 |
| --- | ---: | ---: | --- | --- |
| 初始native量化适配 | 2/2 | 0/2 | 运行资产50.13% | 质量退化 |
| FP16视觉＋FP16语言＋量化专家 | 2/2 | 2/2 | 核心文件23.61%（前处理另计） | 诊断恢复，未达40% |
| FP16视觉＋KL混合语言＋量化专家 | 2/2 | 0/2 | 所需运行资产40.15% | 体积达标，质量未通过 |

以上不能推导唯一错误层或40任务全面成功率；原GPU pack同协议对照仍未重测。下一步优先定位语言子图中首次明显偏离的中间算子，核对本地QAT与RKNN原生激活/权重量化规则，再决定局部保护或重新校准/原生规则对齐训练。保留FP16前端诊断作为板端有效对照，不把它替换成满足体积目标的最终HAQ产物。
