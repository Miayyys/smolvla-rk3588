# 选择性蒸馏微调方案（尚未训练）

## 依据与限制

现有 89.29% 参数更新的三组对照均未超过原 FP 的 Long10 7/10，且仅 GT 也退步。建议冻结视觉和语言主干，缩小动作侧更新范围；该建议尚未验证，不能称为已找到需要微调的层。

LeRobot 官方 SmolVLA 当前配置源码包含 freeze_vision_encoder=True、train_expert_only=True、train_state_proj=True：
https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/configuration_smolvla.py
这支持专家侧微调作为合理起点，不证明当前 LIBERO checkpoint 的最优训练范围。我们的 prepare_mixed_qat 会重新设置 requires_grad，实际实现必须在替换算子后显式冻结并检查优化器清单；仅修改 policy 配置不足以保证生效。

## 第一候选范围

冻结 vision_model、text_model（含词嵌入）、视觉 connector、专家前12层、state_proj 和未指定参数。只训练专家 layers.12–15 的线性权重，以及 action_in_proj、action_out_proj、action_time_mlp_in/out。节点权重形状合计约 26157440 个元素（约 5.81% 原 checkpoint），偏置另计。若无效再比较完整动作专家+动作投影，约 22.2%；不直接放开视觉/语言。

## 如何决定哪些模块确实需要更新

在隔离训练观察上按任务和时间段均衡抽取小批次，统计每个模块教师梯度与GT梯度：cos(g_teacher,g_GT) 为冲突诊断，RMS梯度及短步对完整动作的影响辅助筛选。单步梯度、梯度大小或过去的量化敏感性都不能认定模块必须更新。最终通过相同范围的 GT-only/KD 短对照和闭环开发任务决定保留哪些层；保留新的冻结评测面板用于最终比较。该梯度探针尚未实现/运行。

先用 200–250 更新的短诊断而非直接 2000 步长训；LR、教师权重与可训练范围分开对照，不同时改变全部设置。教师损失有冲突时不能简单判定教师错误；应结合动作语义与轨迹审核。质量未改善前不进入 QAT。冻结主干属于训练配置，不修改 HAQ 位宽搜索自由度。
