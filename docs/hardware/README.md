# SmolVLA 模块候选配置与 RK3588 成本表

由 `scripts/build_hardware_tables.py` 根据固定 checkpoint、40 条校准调用记录及板端 JSON 自动生成。
**候选格式不等于该模块已验证支持，也不是最终HAQ动作空间。未测量值为空，不能视为零。** `current_precision` 描述既有PTQ候选，不代表本轮FP基线或HAQ已决定的精度。

`deployment_stage`/`deployment_backend`/`baseline_deployment_format`描述当前已运行的RKNN＋CPU基线；`stage_calls_per_action_chunk`是阶段调用次数。`full_graph_format_status`与`format_cost_status`分别表示整图执行证据与独立签名测量，不能互换。独立精度控制边界仍待导出图核查；没有据人工敏感度固定搜索位点。

原始checkpoint覆盖：397 个按参数路径分组的模块、500 个张量、906,639,456 B。既有PTQ候选曾量化291个Linear并使用1个CPU行量化embedding；它不是本轮HAQ固定精度图。
输入调用记录覆盖 291 个Linear；302 个活动Linear中 302 个已有实测或来源可追溯的shape，0 个仍无shape；另有1个不活跃lm_head。
按算子、shape、格式和边界去重得到 100 个基础成本配置，其中 100 项完成三轮板测；另有 18 项融合、注意力、精度边界和CPU embedding补充测量。
模块数仅指有持久化参数的模块。缓存按项目范围不测；当前清单没有为Norm/位置参数登记多精度候选，这不证明它们只能原精度；需由完整执行图和后端探针核实是否可独立配置。

[完整模块配置表](module_options.csv) · [去重成本配置/状态](cost_cases.csv) · [基础及补充实测表](measured_costs.md) · [实测CSV](measured_costs.csv) · [融合/转换明细](supplemental_costs.md) · [机器可读数据](tables.json) · [Linear延迟图](linear_costs.png)

| 分组 | 参数模块数 | 原权重 MB | 有输入记录 |
| --- | ---: | ---: | ---: |
| action_interface | 3 | 0.314 | 3 |
| action_time_mlp | 2 | 6.227 | 2 |
| expert_norm | 33 | 0.048 | 0 |
| expert_mlp_0_7 | 24 | 70.779 | 24 |
| expert_attention_0_7 | 32 | 29.082 | 32 |
| expert_mlp_8_15 | 24 | 70.779 | 24 |
| expert_attention_8_15 | 32 | 29.082 | 32 |
| lm_head | 1 | 94.618 | 0 |
| connector | 1 | 23.593 | 1 |
| language_embedding | 1 | 94.618 | 0 |
| language_norm | 33 | 0.063 | 0 |
| language_mlp_0_2 | 9 | 44.237 | 9 |
| language_attention_0_7 | 32 | 39.322 | 32 |
| language_mlp_8_15 | 24 | 117.965 | 24 |
| language_attention_8_15 | 32 | 39.322 | 32 |
| language_mlp_3 | 3 | 14.746 | 3 |
| language_mlp_4_7 | 12 | 58.982 | 12 |
| vision_patch | 1 | 1.181 | 1 |
| vision_position | 1 | 1.573 | 0 |
| vision_norm | 25 | 0.077 | 0 |
| vision_mlp_0_5 | 12 | 56.669 | 12 |
| vision_attention_0_5 | 24 | 28.348 | 24 |
| vision_mlp_6_10 | 10 | 47.224 | 10 |
| vision_attention_6_11 | 24 | 28.348 | 24 |
| vision_mlp_11 | 2 | 9.445 | 2 |

## 基础可调算子成本

95个Linear配置覆盖19个shape签名×5种格式；5个patch Conv2D配置覆盖1个shape×5种格式。每项20次预热、100次计时、3轮；使用真实checkpoint代表权重和固定种子合成输入/校准数据。独立FP32 I/O子图包含Lite2调用开销，不代表完整模型延迟或量化质量。

表中的格式按每个Linear/Conv shape的可编译候选列出；稳定性阈值为三轮p50最大/最小≤1.2。板子未锁频，超过阈值项需在HAQ最终候选阶段复测。

## 其他历史板端实测

| 单元 | 格式 | p50 ms | p95 ms | 文件 B |
| --- | --- | ---: | ---: | ---: |
| synthetic_matmul | float16 | 0.4063 | 0.5256 | 77103 |
| synthetic_matmul | bfloat16 | 1.4581 | 2.1557 | 39727 |
| synthetic_matmul | w8a8 | 0.2094 | 0.2362 | 293897 |
| synthetic_matmul | w16a16i | 0.2450 | 0.2632 | 41997 |
| synthetic_matmul | w16a16i_dfp | 0.2446 | 0.2638 | 41997 |
| expert_mlp_layer0 | ptq_w8a8 | 2.1176 | 2.4769 | 4524893 |
| expert_mlp_layer0 | qat_w8a8 | 2.3674 | 2.7667 | 4524893 |

以上历史测试未统一锁频/核心配置，不作直接性能排名；合成小图不代替真实模块测量。RSS是含Python/runtime的进程峰值。

## 补充子图与边界实测

18项额外测试见[融合/转换明细](supplemental_costs.md)：真实专家/语言/视觉MLP各测FP16与W8A8；QKᵀ→Softmax→PV形状代理测FP16/W8A8；真实专家QKV图测全INT8、全FP16及6种混合精度；真实BF16 token embedding测CPU行INT8查表。
融合子图数据用来校正“逐层相加”的估算；QKV记录的是含内部格式转换的整图延迟，不能解释为纯转换算子单独耗时。embedding微基准只含CPU查表/解量化，未包含tokenizer和CPU到NPU传输。

## 覆盖边界

- 11个原先缺失输入shape的活动Linear已由捕获shape或可追溯的同结构输入签名补齐；不活跃的lm_head不进入当前动作路径。
- 缓存独立微基准按用户要求排除；实际完整流程仍包含必要的张量传递。当前清单未为Norm、残差和位置参数建立独立精度候选，暂列待执行图核实；完整FP16策略已在RKNN＋CPU执行，任意混合精度整图尚未验证。
- W4A16不属于目前确认可执行的RK3588候选；W4A4未通过，未放入格式候选。质量选择仍需回到固定LIBERO任务评估，层输出误差只是诊断项。
- 延迟估计须结合真实调用次数；p95与峰值RSS不能逐层直接求和。

## 当前完整部署：RKNN＋CPU

以下从既有真实板端记录生成：单条开发观测完整回放预热1次、计时3次；闭环行来自4任务22次请求。三张RKNN图均为FP16构建，CPU保留浮点处理。不是最终HAQ/QAT/PTQ结果。

| 单元 | 后端 | 每动作块调用次数 | p50 ms/调用 | p95 ms/调用 | 测量次数 | 文件 B |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| vision_connector | rknn_2.3.2 | 2 | 1905.2817 | 2188.4714 | 6 | 212621173 |
| prefix_with_kv | rknn_2.3.2 | 1 | 613.8538 | 616.6909 | 3 | 326154086 |
| expert_step_v2 | rknn_2.3.2 | 10 | 323.8783 | 377.2159 | 30 | 217425832 |
| raw_preprocess | cpu_numpy_tokenizers | 1 | 144.0578 | 144.1368 | 3 | — |
| prefix_glue | cpu_numpy | 1 | 2.8798 | 3.0035 | 3 | — |
| time_embedding | cpu_numpy | 10 | 0.6944 | 0.8323 | 30 | — |
| euler_integration | cpu_numpy | 10 | 0.0799 | 0.1086 | 30 | — |
| action_postprocess | cpu_numpy | 1 | 0.1062 | 0.1101 | 3 | — |
| full_raw_policy | rknn_2.3.2+cpu | 1 | 8028.9742 | 8162.1893 | 3 | 945564365 |
| full_policy_closed_loop | rknn_2.3.2+cpu | 1 | 7444.5892 | 7630.8421 | 22 | 945564365 |

视觉共享同一模型文件、每动作块运行两次；前缀一次；专家十次。表内完整流程行已经包含子阶段，不能再次相加；单阶段p50之和不等于完整流程p50，p95/RSS也不能求和。
三张图共756,201,091 B，CPU参数文件189,363,274 B，合计945,564,365 B，未达到原checkpoint至少40%压缩要求；大小未包含tokenizer/配置等资产。CPU参数不按重复调用复制计费。
阶段调用次数是执行图次数，不保证各参数算子都逐次执行；独立精度控制边界仍待导出图核查。Norm/融合节点不凭缺少独立测量固定精度。
Lite2图调用时间含输入准备、运行及输出交接；纯CPU↔NPU拷贝/格式转换没有独立计时，不从总耗时相减推算。CPU分项为实测合并阶段，未把分词/lookup/状态投影拆出。
混合精度整图配置成本仍未测量；现有100个基础签名和18项补充可作代理查表，不能把FP16流程参考行作为任意配置的已测收益。
[完整部署证据](../experiments/2026-10-01-board-libero-closed-loop.md) · [表格更新记录](../experiments/2026-10-01-deployment-hardware-tables.md)

