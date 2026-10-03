# 简历项目表述：QVLA 硬件感知混合精度量化

## 推荐项目名称

**SmolVLA 面向 RK3588 的硬件感知混合精度量化与部署探索（QAT/PTQ）**

## 中文简历版

- 基于 LeRobot SmolVLA 与 LIBERO 构建 VLA 量化评估链路，完成动作专家 112 个 Linear 的 W8A8 QAT 微调、训练后真实 INT8 转换，以及从原始 FP checkpoint 独立进行的 PTQ 对照；视觉语言主干等其余模块保留 BF16，形成模型级选择性混合精度方案。
- 实现静态仿射激活量化、逐输出通道对称权重量化与 CUDA `torch._int_mm` 整数 GEMM（INT32 累加），量化 checkpoint 经严格重载并完成动作推理。专家量化整模型文件由 906.7 MB 降至 806.6 MB（减少 11.04%）。
- 完成真实 W8A8 FP/PTQ/QAT 的 LIBERO 配对闭环开发筛查，覆盖四套件 40 个任务、每任务 1 回合；本轮 FP 为 32/40，PTQ/QAT 均为 31/40，各有一个相对 FP 的不同任务回归。该样本不足以证明成功率保持或非劣，因此只作为开发诊断，不声称量化保持了任务质量。
- 另实现覆盖 291 个 Linear 和 token embedding 的独立 INT8 PTQ，checkpoint 文件缩小 41.45%；但四个 LIBERO suite 的单回合开发筛查为 24/40，对照 FP 为 32/40，质量退化。该结果达到体积目标但没有通过质量筛查，不作为最终配置。
- 面向 RK3588/RKNN-Toolkit2 建立子图校准、ONNX 导出、W8A8 编译及留出激活评估流程；按 240 条校准与 240 条开发激活比较子图精度和 `.rknn` 文件体积，并检查 BF16/FP16 候选与模型计算图输入结构。在 R1 板上用 runtime 2.3.2 实测第 0 层 PTQ/QAT 两个 W8A8 子图，单图 p50 分别为 2.118/2.367 ms。
- 建立全模型 HAQ-RL 候选空间与循环策略梯度/质量优先奖励代码骨架，覆盖304个暂定可调权重算子位点、1517个候选选择；代码烟测通过，真实搜索尚未启动。已取得一个 MLP 子图级的真实 NPU 延迟、误差和进程 RSS；完整策略的端到端延迟、整体内存与任务质量仍未测，尚未得到最终 Pareto 配置。

如果简历空间有限，可压缩成前三条。项目状态建议标注“进行中”。

## English version

**Hardware-Aware Mixed-Precision Quantization and Deployment Exploration for SmolVLA (QAT/PTQ)**

- Built a quantization workflow for LeRobot SmolVLA on LIBERO, including W8A8 QAT fine-tuning of 112 action-expert Linear layers, post-training conversion to real INT8 weights, and an independent PTQ baseline from the original FP checkpoint. Kept the vision-language backbone and remaining modules in BF16 as a model-level selective mixed-precision design.
- Implemented static affine activation quantization, per-output-channel symmetric weight quantization, and CUDA integer GEMM with `torch._int_mm` and INT32 accumulation. Strictly reloaded the quantized policy and validated action inference; checkpoint size decreased from 906.7 MB to 806.6 MB (11.04%).
- Also built an independent PTQ candidate covering 291 Linear layers and the token embedding. Its checkpoint shrank by 41.45%, while the single-round LIBERO development screen reached 24/40 versus 32/40 for FP, so it did not pass the quality screen.
- Built an RKNN-Toolkit2 workflow for RK3588 calibration, ONNX export, W8A8 compilation, and held-out activation evaluation. Also ran the layer-0 PTQ/QAT MLP subgraphs on the RK3588 NPU with runtime 2.3.2; measured p50 call latency of 2.118/2.367 ms.
- Prepared RK3588 cost measurements for 100 primitive signatures and 18 supplemental cases. Full-model HAQ-RL is the selected search route, but its complete evaluator and search have not run; full-policy execution, end-to-end latency, peak memory, and final task quality remain unmeasured.

## 面试时的边界说明

这里列出的专家 QAT/PTQ 结果与 291 Linear PTQ 是两种独立候选。它们证明真实量化链路和质量退化边界已经测过，不表示最终混合精度图已经确定。

项目已选定全模型 HAQ 强化学习搜索作为精度分配方法；当前完成的是动作空间、控制器和奖励接口骨架，尚未训练/运行 RL agent。完整 evaluator、质量容忍界和整策略板端资源基线仍需补齐，不能称为已经得到 HAQ 最优位宽分配。

当前 A10 的整数推理原型比 FP 慢，量化目标是降低存储并验证硬件适配，还没有证明推理加速。简历不应写“端侧已部署”“RK3588 加速 X 倍”或“成功率保持不变”，直到板端和配对闭环实验给出对应证据。

## 证据入口

- [整专家真实 W8A8 QAT/PTQ](experiments/2026-09-27-full-expert-real-w8a8.md)
- [真实 W8A8 LIBERO 配对闭环筛查](experiments/2026-09-27-paired-w8a8-rollouts.md)
- [Stage 1 QAT 训练](experiments/2026-09-27-stage1-w8a8-qat.md)
- [RK3588 16 层专家 MLP 子图](experiments/2026-09-27-all-expert-mlp-rknn.md)
- [RK3588 板端 MLP QAT/PTQ 实测](experiments/2026-09-27-rknn-board-subgraph.md)
- [RK3588 专家注意力错误图及更正](experiments/2026-09-27-all-expert-attention-rknn.md)
- [当前实验索引](experiments/README.md)
