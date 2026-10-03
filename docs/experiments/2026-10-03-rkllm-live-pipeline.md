# RKLLM 接入完整板端流程：连续调用一致性未通过

> 后续更新：[工作池缓存映射修复](2026-10-03-rkllm-working-pool-coherency.md)已消除本配置的重复输出漂移。本页保留修复前的实验结果。

## 范围与结果

在上一轮固定 K/V＋专家桥接后，实现了实际图像输入→RKNN FP16视觉→CPU token/state接口→原生RKLLM语言→RKNN INT8专家10步→CPU动作后处理。语言worker常驻，每个观测先调用官方 `rkllm_clear_kv_cache(handle,0,nullptr,nullptr)` 并检查缓存为0；禁止跨帧复用。系统runtime和原RKNN部署目录没有替换。

**整条链路可以输出有限动作，但连续调用尚未通过验证，未开始新的闭环任务。** 上一轮的单帧K/V和动作数值是已测结果，不能据此认为多帧语言模块已解决。

## 代码组织与配置

- `scripts/smolvla_rkllm_worker.cpp`：独立常驻C++进程，只负责原生语言执行、缓存清空与保存。原生计算不通过CPU重做attention或K/V投影。
- `scripts/smolvla_rkllm_backend.py`：校验模型/库/worker SHA，压实有效token，解析固定SDK1.3.1缓存，恢复177位置供原RKNN专家使用。暂支持验证输入的151有效token。
- `scripts/smolvla_board_runtime.py`：根据manifest选择语言后端；RKLLM路径只加载视觉、专家两个RKNN图。
- `scripts/verify_smolvla_rkllm_persistent.py`：真实A→B→A输入对照；B将第一摄像头置零、state加0.1，噪声相同。保存三个输入边界、32个K/V和动作，以及失败报告。
- 后端日志中任何 `E RKNN`/`E rkllm` 都判失败，不能只检查 `rkllm_run` 返回0。

板根目录 `rkllm_native_patch_v1`；视觉/CPU资产来自 `qat_v1_repair_fp16_frontend_v1` 的链接；专家为 `expert_selected_corrected.rknn`。语言来自选定V1无教师QAT master；本轮FP16语言不是最终40%压缩候选。文件与报告在Git忽略的runs中。

## 诊断计算

同输入重复误差分别为 `max(abs(prefix_A1-prefix_A2))`、`max(abs(KV_A1-KV_A2))`、`max(abs(action_A1-action_A2))`。这用于检查连续调用，与任务成功率不同。脚本沿用现有回放流程的一致性假设，诊断容差1e-5；这是严格的重复检查，**不是经任务验证的FP16质量容差**。未修改runtime的对照也出现小漂移，因此不能仅以越过该门限宣称补丁引入所有误差。

以下是各配置真实日志中的A→B→A结果；数值为最大绝对差，未做重复统计或置信区间。

| 配置 | 前缀重复差 | K/V重复差 | 动作重复差 | 日志 |
| --- | ---: | ---: | ---: | --- |
| 151整段，3 NPU／4 CPU | 0 | 1.516602 | 0.0611862 | persistent_debug.log |
| 对齐160，3 NPU／4 CPU | 0 | 0.5703125 | 0.0611862 | aligned_live.log |
| 对齐160，3 NPU／3在线CPU | 0 | 1.711182 | 0.0611861 | valid_cpu_live.log |
| 对齐160，1 NPU／1 CPU | 0 | 1.269531 | 0.0611861 | single_npu_live_v2.log |
| 因果控制＋整段160，1 NPU | 0 | 1.2265625 | 1.733609 | causal_control_live.log |
| 最终整理后对齐160，1 NPU／1 CPU | 0 | 0.2236328 | 0.0611862 | final_live_gate.log |
| 未修改runtime，原分段，1 NPU | 0 | 0.02734375 | 0.0917793 | stock_runtime_live.log |

当前板在线CPU为0–6，原mask0xf0包含不在线的7；改为在线核没有解决漂移。SDK要求CPU线程数至少等于NPU核数，3NPU／1CPU配置初始化明确失败，不列入有效对照。

不同配置下视觉输出及语言第一层K/V重复一致，后续层存在差异。独立重建进程后固定输入的K/V也有差异，因此不能只归因于旧帧KV缓存没清掉。因果控制也有漂移，说明分块mask并非唯一待查因素。**目前没有确定底层根因**；整段prefill的形状、缓冲区及同步是后续检查方向。

## 对齐和dump检查

为排查padding，将有效token放在前面，尾部补零到160；真实state保留其原位置，其后所有dummy key对全部query屏蔽。分块state规则保持不变。独立补丁 `--aligned` 在可执行空隙增加真实token计数0x7137f0；worker在两次执行之间短暂调整该页权限写计数，再恢复RX。必须校验固定库SHA，非公共SDK接口。汇编集中在 `rkllm_native_aligned_mask.s/.ld`。它没有解决重复误差，不能作为已选修复。

对应Toolkit诊断转换补齐M160和attention所需shape；保留原151版生成方式。1核编译另含QK/QKV的160/320/480候选，实际缺失shape须依错误日志处理，不声称所有长度已覆盖。模型名分开保存：原151版、aligned_3cores、aligned_1core，原有原始数据不当作新的结果。

两次同输入dump中，第一层Q文件第一次全零、第二次非零；K/V文件一致。存在dump快照与执行时序不一致的证据，**不能把Q文件差异当作首个计算错误定位**。检查日志 `dump_repeat.log`、原Q文件 `repeat_dump/`。GET_LAST_HIDDEN_LAYER模式不生成prompt cache；独立三次同输入hidden对照最大差0.850702、MAE0.0135835（全160×960张量，含dummy行，未按有效token另报）。这说明漂移不限于保存cache接口，但尚未定位具体算子。

## 收尾与下一步

完整语言接口和连续回放代码已经集中整理，未创建或提交PR。最后保留独立诊断目录和失败manifest，不替换正常RKNN部署，不更新正式HAQ成本表。最终日志/报告和SHA索引见 `live_experiment_manifest.json`。整理后的最终三次完整推理为7185.72、6718.25、6627.00ms（含预处理和缓存交接，不含初始化/SSH；仅三次诊断，不是性能统计）。旧151版模型重新生成后SHA仍为e849550d5a4e79c7146b50f5971d176b117d348ed4bb0538a996e37ad5a8e161，历史产物路径已恢复。

复现板端诊断：`cd /root/qvla_board_test/rkllm_native_patch_v1 && OPENBLAS_NUM_THREADS=1 PYTHONPATH=/root/qvla_board_test/python_site python3 verify_smolvla_rkllm_persistent.py --root .`。当前配置应返回failed并保存负结果，不能当作部署成功。

下一步应围绕原分段与整段prefill的原生执行/缓冲区同步做最小对照；修复后先比较有效token的K/V、完整动作和重复误差，再跑原先几个短任务。当前不报告新RKLLM成功率、端到端加速或达成40%压缩。

后续定位见[长度／布局隔离记录](2026-10-03-rkllm-repeat-length-isolation.md)：已排除视觉和专家为必要触发条件，独立语言长输入仍漂移，根因尚未解决。
