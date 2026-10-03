# 初始量化实验设计（2026-09-26，已归档）

本页保留项目启动时的实验思路，不再作为当前执行计划。早期的手工敏感度选层、先固定量化范围和 W4A16 候选等提议，已由后续 RK3588 实测和全模型 HAQ-RL 路线更新。

当前方法、HAQ 搜索范围、质量优先收益规则及剩余前置条件见[量化技术路线](quantization-technique-plan.md)；正式质量比较口径见[固定评测协议](evaluation-protocol.md)。模型候选讨论见[历史选型记录](model-candidates-rk3588.md)。

早期方案固定了 `lerobot/smolvla_libero` 与 `lerobot/libero`，计划建立 FP 基线、进行 QAT 和独立 PTQ，并用 RKNN 探查 RK3588 子图。实际完成或失败的步骤均保留在[实验索引](experiments/README.md)，不因本页归档而删除。
