# RKNN＋CPU 部署后的模块与成本表更新

## 已完成与结论边界

根据已完成的真实板端 FP16 部署及四任务闭环，重新生成两个主表：

- [模块可选配置表](../hardware/module_options.csv)：397 个参数模块、500 个原始张量；增加实际阶段、后端、基线格式、阶段调用次数、整图执行与独立精度控制状态。
- [硬件成本表](../hardware/measured_costs.csv)：原有100项独立算子＋18项补充均保留，新增10项部署阶段/完整流程参考，共128项；[可读表](../hardware/measured_costs.md)。

这次复用已有原始报告，没有启动新的板测或RL搜索。新增成本均为现有FP16基线，**不能用作任意混合精度整图的已测成本**。独立位宽控制边界仍待映射；没有删减或按人工敏感度固定候选。

## 原理与配置

当前执行路径为 RKNN Toolkit/Lite/runtime 2.3.2、NPU core0＋CPU NumPy/tokenizers，RKLLM未参与。按当前 `scripts/smolvla_board_runtime.py` 映射参数路径：

| 阶段 | 包含参数 | 阶段调用次数/50动作块 |
| --- | --- | ---: |
| vision_connector | vision 参数及 connector；共享视觉图供两路相机使用 | 2 |
| prefix_with_kv | 语言16层 attention/MLP/norm；显式输出逐层K/V | 1 |
| expert_step_v2 | 专家16层、动作输入/输出及时间MLP | 10 |
| prefix_glue / CPU | token embedding lookup、state_proj | 1 |
| inactive | lm_head | 0 |

调用次数是阶段级次数，不冒充每个参数算子的独立trace。原表 `current_precision` 仍指历史PTQ候选，新增 `baseline_deployment_format` 才描述当前部署，避免把历史候选误读为最终HAQ图。INT8/BF16/INT16等独立签名支持不自动等同于整图可部署；Norm/位置参数候选未完整探查的状态仍保留。

## 数据来源与计算

- 固定checkpoint SHA256：`9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`。
- `runs/smolvla_raw_board_v1/raw_full_board_report.json`：episode18/task0/frame0，seed `2416662958`，预热1次、完整流程计时3次。CPU预处理/组装/时间编码/积分/后处理及三类NPU调用均已分项记录。
- 四个 `runs/smolvla_board_libero_v1/board_{suite}_0/result.json`：四suite各task0、初始状态0/envseed0，共22次请求。仿真暂停等待板端；不是实时控制验证。
- 每行CSV附 `board_reports` 与 SHA256；三图SHA及CPU权重hash见[原始输入板端实验](2026-10-01-board-raw-preprocessing.md)。

将报告中的各阶段 `*_ms` 按真实调用展开，用 `numpy.percentile(values,50/95)` 计算p50/p95。视觉6次、前缀3次、专家30次；CPU每次完整动作块调用一次的阶段取3次，时间编码/积分各30次。这里是同一回放的相关测量，不当作30个独立观测。完整流程取3个 `total_ms`，闭环取22个 `inference_ms`。协议与原来的三轮×100次独立签名微基准分别标注。

| 新增测量 | p50 ms/调用 | 每动作块调用次数 |
| --- | ---: | ---: |
| 两路视觉共享图 | 1905.2817 | 2 |
| 语言前缀 | 613.8538 | 1 |
| 专家单步 | 323.8783 | 10 |
| CPU原始输入预处理 | 144.0578 | 1 |
| 完整原始输入回放 | 8028.9742 | 1 |
| 四任务闭环中的完整模型请求 | 7444.5892 | 1 |

完整回放峰值RSS为1,892,432KiB；闭环为1,887,076KiB，均为进程峰值，不能分配/求和成单模块成本。完整回放动作MAE 0.0020128；闭环FP/板端均2/4，仅开发筛查。

文件范围：三张RKNN图756,201,091B＋CPU参数189,363,274B＝945,564,365B；tokenizer/config等资产未计入。共享图和共享参数只计一份。没有达到原checkpoint≥40%压缩目标。纯CPU↔NPU传递/格式转换未独立计时；不得从不同实验总耗时相减得到转换成本。

## 使用及复现

```bash
.venv-haq-local/bin/python scripts/build_hardware_tables.py
MPLCONFIGDIR=/tmp/qvla-mpl .venv-haq-local/bin/python scripts/summarize_hardware_cost.py
.venv-haq-local/bin/python scripts/build_hardware_tables.py
.venv-haq-local/bin/python scripts/build_haq_search_space.py
```

生成逻辑在 `scripts/hardware_deployment_tables.py`，两个既有生成器均接入，重新生成不会丢失此次新增行。`tables.json` 同步保存 `deployment_contract`/`deployment_costs`；`formal_search_ready=false`。部署行标记 `deployment_stage` 和 `deployment_reference_only; overlapping_rows_do_not_sum`，它们不进入独立签名候选索引，避免完整模型与子阶段重复计费。旧基础/补充case_id保持不变。重建的HAQ配置清单仍为临时清单，更新来源hash不意味着正式精度搜索已准备好。

核对：397模块均有阶段归属；500张量/906,639,456B原始tensor数据全覆盖；基础100项与补充18项不变；新增10项分位数逐项可由报告复算；完整回放与闭环调用计数一致。后续需要验证任意混合精度图的可独立控制范围、转换数值和真实整图成本。
