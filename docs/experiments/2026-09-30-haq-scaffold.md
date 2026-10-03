# 全模型 HAQ 搜索代码骨架与就绪性检查

**后续更新**：本页的严格质量门槛与G奖励属于当时接口验证。用户随后将目标改为至少40%文件压缩、效果与查表速度联合优化，活动路线见[技术路线](../quantization-technique-plan.md)。固定开发观测与FP动作缓存已进一步实现并完成[本机完整模型测试](2026-09-30-offline-action-cache.md)；本页的“全配置evaluator未完成”仍指任意全模型精度图尚未接通，不代表没有可用的离线评分函数。

## 目的与结论

把 RK3588 硬件测试表接入 HAQ 搜索代码，确认模型参数清单能否形成覆盖全部可配置权重算子的候选空间，并验证候选生成、策略更新、检查点恢复和质量优先奖励接口。本记录是代码与接口验证，不是量化搜索实验。

结果：从当前硬件清单构造出 **304 个多候选参数化位点、1517 个候选选择**，所有这些选择都有对应的签名级板测记录。策略与奖励合约烟测通过。动作空间仍标记为 `search_ready=false`，没有启动模型精度搜索。

## 输入与配置

- 模型：`lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de`。
- checkpoint SHA-256：`9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`。
- 成本与模块来源：[`docs/hardware/tables.json`](../hardware/tables.json)，SHA-256 `ececfdd22f65bcc4258ecbfd9bbf6b63fae461189599fb42f89c74ef48136655`。
- 清单生成：`python3 scripts/build_haq_search_space.py`，输出 [`config/haq_action_space_v1.json`](../../config/haq_action_space_v1.json)。
- 搜索候选没有读取 v0 手工图中的 `v0_format`、既有 `current_precision` 或敏感度结论来固定、屏蔽或优先某个模块。

当前候选包含302个 Linear 和1个 patch Conv，各有 `w8a8`、`float16`、`bfloat16`、`w16a16i`、`w16a16i_dfp` 五种候选；token embedding 有 `native_bf16_row_lookup` 与 `cpu_int8_row_lookup` 两种候选。成本证据覆盖100项基础签名和18项补充配置。基础表9项、补充表2项三轮延迟波动超过20%，最终候选仍要复测。

清单中的397个参数模块里，另有93行没有两个以上已登记的配置选项：91个 Norm/参数行、1个未活动 `lm_head`、1个仅登记 native 格式的位置 embedding。它们暂列为未决清单，不据此宣布搜索图完整或永久锁定原精度。所有板测都只是独立签名/子图成本证据，没有提供这1517个候选的整模型任务质量。

策略的8维输入特征来自权重字节数、权重元素量、输入/输出维数、校准调用数及算子类型；不包含人工敏感度。当前策略按模块路径自然序产生动作，这个顺序还没有和完整导出图及运行调用顺序核对。

## 已实现代码

- [`qvla_haq/search_space.py`](../../qvla_haq/search_space.py)：读取硬件表，生成多候选位点、格式编号、形状特征和成本证据引用，并校验完整配置向量。
- [`qvla_haq/policy.py`](../../qvla_haq/policy.py)：NumPy 自回归掩码分类策略，使用简单循环网络和整配置回报的 REINFORCE 更新；支持策略检查点保存与恢复。它是用于打通控制器接口的基线，不等同于 HAQ 论文实现的完整复现。
- [`qvla_haq/reward.py`](../../qvla_haq/reward.py)：严格读取成对成功率差的逐 suite 95% 置信下界、动作异常率、完整板端运行、RAM、p95 和部署包字节。任何质量或资源硬门槛失败时返回契约中显式设置的惩罚；只有全部通过才按固定公式计算 `log(G)`。
- [`scripts/sample_haq_configs.py`](../../scripts/sample_haq_configs.py)：生成带 `provisional_unscored` 状态的候选配置，不会运行模型或伪造评测分数。
- [`scripts/smoke_haq_scaffold.py`](../../scripts/smoke_haq_scaffold.py)：验证候选合法性、策略更新和存取、奖励公式与不完整评价拒绝。

奖励合约要求显式配置每 suite 的质量容忍界、动作异常率上限、RAM/p95 预算、板端参考配置的 `(R0,T0,B0)` 及不可行惩罚。项目目前没有完成 FP 多种子质量波动估计，也没有锁定这些门槛与完整高精度板端参考数据，因此没有生成实际奖励配置。

## 验证

运行 `python3 scripts/build_haq_search_space.py`，报告为397个参数模块、304个多候选位点、1517个候选选择、0个缺少签名级成本记录；输出显式标注 `search_ready=false`。

运行 `python3 scripts/smoke_haq_scaffold.py`：策略对两个完整候选配置完成一次合成回报更新，检查点恢复后下一次采样一致；合成的完整评价记录通过公式校验，不完整的子图代理因范围错误不进入收益排名。该烟测不调用 SmolVLA、LIBERO、RKNN 或 RK3588，也不提供任何量化质量或硬件收益结论。

运行 `python3 -m py_compile` 覆盖新增模块和脚本，通过。另用 `scripts/sample_haq_configs.py` 生成两个临时提案，均标记 `provisional_unscored`。

## 启动真实 HAQ 搜索前仍需完成

1. 捕获/导出完整策略执行图，核对全部动作位点、精度可行性、动态 shape、融合边界和 NPU/CPU 转换；用真实图调用顺序替换当前模块路径顺序。
2. 实现对任意完整精度图生效的量化转换和开发质量 evaluator，能在隔离任务、初始状态和噪声上返回逐 suite 配对闭环结果。
3. 完成同环境 FP 多种子基线，在查看冻结测试之前锁定质量容忍界与动作异常门槛。
4. 在 RK3588 上测完整高精度参考配置和候选全图，建立可用 RAM、动作 p95 预算及部署包参考值；校准成本预测器。当前不能把独立节点/子图的延迟简单相加，也不能计算正式 `G`。
5. 确定上面各质量/资源失败项的惩罚值，并在搜索配置里固定版本，之后再开始策略训练。

在这些条件完成前，当前代码只负责构造与检查候选和策略接口；不运行正式 RL 搜索，生成的候选不参与最终位宽决策。
