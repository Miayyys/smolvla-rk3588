# 扩展蒸馏后台运行协议

用户授权后台启动完整蒸馏并提供tail命令；本轮未授权直接自动进入QAT。使用已准备v3配置：10个长任务各500独立观察，均衡轨迹及时间三段；真实OFT教师标签→契约过滤及逐帧审计→原始FP学生2000次AdamW更新，累积2、教师任务比例0.5、λ0.2、lr1e-7热身到1e-6再余弦降至1e-7，每250更新40开发观察完整动作评价/保存。完整训练不代表任务质量必然改善。

入口 `scripts/run_expanded_distillation_pipeline.py`，flock防重复。单阶段失败停止，状态/日志记录失败，不自动跳过审核。教师标签质量过滤仅有限值、动作范围、符号及明确排除，**不保证自动识别全部语义错误**，不按教师失败任务整体删除数据。

FP训练后自动运行10长任务×1初始状态，和已保存原始FP长任务面板逐项核对状态/双相机/噪声/环境seed，不再次跑40任务。其他suite只做周期离线开发检查，本轮未自动做其闭环回归。目标≥8/10仍需真实结果；完成蒸馏和long10后停止，不启动QAT。

服务器启动检查：A10空闲，磁盘可用21GiB。训练入口按FP master/AdamW动量/临时文件/快照计算磁盘需求，不足时明确停止；不删旧模型补空间。

日志 `/root/qvla/runs/distillation_expanded_v3.log`，状态JSON同目录 `distillation_expanded_v3_status.json`，PID在 `distillation_expanded_v3.pid`。阶段prepare/label/review/audit/fp/long10；原始输入/教师标签/训练输出由v3配置指定。启动状态已记录，结果待实际完成，不写成已训练或已提高成功率。

## 实际完成结果

2026-10-02 流水线完成，总耗时 4445.67 秒（约 74 分钟），其中 FP 训练 2233.75 秒（约 37 分钟）、长任务评测 574.92 秒。完成 2000 次优化更新、4000 个微批次，其中 2028 个使用真实教师损失。严格重载 FP master 后动作 MAE/max_abs 均为 0；未开启 fake quant，未启动 QAT。

同初始状态、双相机图像、噪声核验通过的 LIBERO-Long 单初始状态面板：原始 FP **7/10**，本次蒸馏 FP **5/10**。任务 ID 3、6 从成功变失败；原本失败的 0、7、8 仍失败，没有新增成功。未达到 ≥8/10 目标，本次最终 checkpoint 不作为已改善的 QAT 起点。其他 30 任务闭环回归未测量。这不是多初始状态的正式成功率，也尚未确定退步由教师标签、训练设置或其他因素中的哪项导致。

原始证据：服务器 `runs/distill_fp_expanded_v3/report.json`、`long10/paired_summary.json`、流水线状态及日志。配对结果和状态已同步至本地 `runs/expanded_distillation_code_evidence_v1/completed_v3/`。master SHA256 `f40e3aeadfbe566f8bd2d357954611fc7bc8fc40193e74138be8cbbb60624f80`；教师 manifest SHA256 `0d29b473b2c93e4d5a4b816ec82c7cbb3f3234738426bc07703d70cf3b3efd59`。
