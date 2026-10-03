# QVLA：SmolVLA 硬件感知混合精度量化与部署探索

一个面向边缘机器人部署的 VLA 混合精度量化项目。用真实 LIBERO 操作任务对 SmolVLA 做 QAT 与独立 PTQ，在 A10 GPU 验证真实整数推理，再用 RKNN-Toolkit2 按 RK3588 的后端能力探索子图精度、体积与算子支持。简历表述见[项目摘要](docs/project-portfolio.md)。

## 当前交付

- 锁定 `lerobot/smolvla_libero`、LeRobot 0.6.1、`lerobot/libero` 和版本化 episode 分区；校准、开发评测与冻结测试分开管理。
- 已完成两种不同覆盖范围的真实 GPU 量化试验。112 个动作专家 Linear 的 W8A8 QAT/PTQ 文件各为 806,645,032 B，比原始文件缩小 11.04%；配对闭环筛查为 FP 32/40、PTQ 31/40、QAT 31/40。该结果只代表早期专家方案，不能说明全模型 HAQ 已完成。
- 扩展 PTQ 将 291 个 Linear 和 token embedding 量化为 INT8，文件为 530,903,920 B，比原始缩小 41.45%；同一轮开发闭环筛查为 FP 32/40、PTQ 24/40。压缩达到40%目标，但质量明显退化，因此不作为最终方案。详情见[扩展 PTQ 实验](docs/experiments/2026-09-28-mixed-int8-implementation.md)。
- 已实现并严格重载 GPU INT8 推理路径：INT8 激活 × INT8 权重、INT32 累加、逐输出通道权重 scale、仿射激活 scale/zero point。A10 上当前 `torch._int_mm` 实现比 FP 慢，不能宣称有推理加速。
- 16 个专家 MLP 的 QAT/PTQ W8A8 子图均已编译，并在每层 240 条隔离开发激活上完成主机模拟器评测；QAT 的输出 MAE 在 10/16 层较低、PTQ 在 6/16 层较低，局部误差没有给出单一胜者。
- 旧奇数层注意力 K/V 图已标为无效；修正版把 K/V 输入拆开，32 个 W8A8 子图均在每图 240 条留出激活上完成主机模拟器比较。第 0 层 PTQ/QAT W8A8 MLP 子图也已在真实 RK3588 上运行成功，使用 Toolkit2/runtime 2.3.2、driver 0.9.8；测试通过私有 mount namespace 使用 runtime，没有替换系统 1.4.0 库。单图结果和限制见[板端实验记录](docs/experiments/2026-09-27-rknn-board-subgraph.md)。
- 当前路线是**全模型 HAQ 强化学习搜索**：临时动作空间包含304个多候选位点、1517个选择；过去的人工敏感度和位宽选择没有用来固定搜索结果。本机已完成RL与随机各30轮×4候选的真实量化对照，240份文件均压缩超过40%，两组奖励与更新均可复核。RL末段代理奖励较高，但最佳代理奖励只高0.000208；12任务配对闭环RL/随机各6/12、FP7/12，尚未证明任务质量收益。当前成本只是RK3588签名查表，INT16/DFP/Conv 本地数值需对齐后端，完整SmolVLA尚未在板上运行；因此尚无正式HAQ最优位宽分配。见[等预算对照实验](docs/experiments/2026-09-30-haq-rl-vs-random-pilot.md)。

终版与原模型的速度、占用和40任务成功率见[全面对比](docs/final-model-comparison.md)，结果按后台测试进度更新。

跨实验质量、体积和推理开销见[模型测试结果总表](docs/model-test-results.md)，包含所选V1的全RKNN与RKNN＋RKLLM板端对照。完整数据、SHA、参数、原始报告和结论边界见[实验索引](docs/experiments/README.md)；量化和部署目标见[技术路线](docs/quantization-technique-plan.md)。

## 实验流程

```text
原始 FP + 固定数据 ─→ FP 质量基线 / 执行图 ─→ 全模型 HAQ-RL 搜索
                                                 │
                               全配置动作质量与硬件资源反馈
                                                 │
                     选定精度图 ─→ QAT ─→ 真实量化转换/重载
                          │                      │
                          └── 原始 FP 独立 PTQ ─┘
                                      │
                       同协议动作及 LIBERO 闭环对照
                                      │
                    RKNN 完整图验证 → RK3588 资源与延迟实测
```

当前要求是整个可行精度空间可搜索、实际模型文件至少缩小40%，效果与查表速度联合优化。每次候选用固定开发观测和缓存FP动作快速评分，定期用闭环任务校验代理，最终统一复核成功率；查表成本保留估算标记，完整板端执行仍须实测。[离线评价器](docs/haq-offline-evaluation.md)已在本机4060验证完整FP及已有真实专家PTQ/QAT：40条候选前向约8.34秒，报告总耗时约22.92秒；旧专家候选仅压缩11.04%，只作接口验证。

本机完整配置反馈试跑可用 `scripts/run_haq_local_loop.py`；其文件压缩率是真实保存字节数，约42.40%～44.21%。保存的动作评分与板测签名成本驱动控制器更新，但查表速度仍是代理，不能称为RK3588整策略加速。[曲线图](figures/haq-real-feedback-local-v3.png)和[实验原始记录](docs/experiments/2026-09-30-haq-real-feedback-loop.md)给出每份候选结果。

## 复现入口

以下命令在服务器 `/root/qvla` 执行。模型、数据、QAT 主权重等大文件放在 Git 忽略的目录。

### 已有专家 W8A8 QAT/PTQ 试验复现（不是最终 HAQ 图）

```bash
.venv/bin/python scripts/qat_train_w8a8_stage1.py \
  --model-dir artifacts/model \
  --vlm-assets-dir artifacts/smolvlm2_assets \
  --dataset-root data/libero \
  --splits data/libero_splits.json \
  --partition config/evaluation_partition_v2.json \
  --map config/quantization_map_qat_stage1.json \
  --output-dir runs/qat_w8a8_stage1_lr1e7 \
  --steps 200 --learning-rate 1e-7 \
  --calibration-tasks 40 --development-tasks 40 --eval-every 50

.venv/bin/python scripts/pack_real_w8a8_expert.py \
  --model-dir artifacts/model \
  --qat-snapshot runs/qat_w8a8_stage1_lr1e7/expert_master_step_50.safetensors \
  --calibration-ranges runs/qat_w8a8_stage1_lr1e7/activation_ranges.json \
  --partition config/evaluation_partition_v2.json \
  --map config/quantization_map_qat_stage1.json \
  --output-dir runs/expert_real_w8a8_v1
```

`pack_real_w8a8_expert.py` 同时从原始 FP 和 QAT FP 主权重独立生成 PTQ/QAT 两份真整数模型。动作诊断与 LIBERO rollout 命令见[整专家 W8A8 实验记录](docs/experiments/2026-09-27-full-expert-real-w8a8.md)。

### RK3588 专家 MLP 子图

```bash
.rknn-probe/bin/python scripts/capture_all_expert_mlp_inputs.py \
  --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets \
  --dataset-root data/libero --splits data/libero_splits.json \
  --partition config/evaluation_partition_v2.json \
  --output-dir runs/expert_mlp_all_calibration_v1 --split calibration \
  --frames-per-task 2 --step-samples 3

.rknn-probe/bin/python scripts/export_all_expert_mlp_qat_ptq.py \
  --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets \
  --qat-snapshot runs/qat_w8a8_stage1_lr1e7/expert_master_step_50.safetensors \
  --partition config/evaluation_partition_v2.json \
  --output-dir runs/expert_mlp_all_export_v1

.rknn-probe/bin/python scripts/compile_all_expert_mlp_qat_ptq.py \
  --export-dir runs/expert_mlp_all_export_v1 \
  --calibration-dir runs/expert_mlp_all_calibration_v1 \
  --output-dir runs/expert_mlp_all_rknn_v1 --workers 4
```

新完成的逐层 MLP、修正 K/V 和板端部署结果按证据写入[实验索引](docs/experiments/README.md)。服务器环境安装与数据准备见[下载说明](docs/download-transfer.md)。

## 目录

- `scripts/`：校准采集、QAT、真实 INT8 打包/推理、ONNX/RKNN 子图编译和评估。
- `config/`：候选精度图与冻结 episode 划分协议。
- `docs/experiments/`：每个成功、退化或失败实验的配置和原始证据索引。
- `docs/quantization-map-v0.md`：早期人工候选图和历史覆盖审计，不是 HAQ 动作空间。
- `qvla_haq/`：候选动作空间、循环策略梯度控制器及质量优先奖励接口。
- `info.md`：量化学习笔记。

## 目前不能声称的结果

当前单回合配对闭环筛查没有证明 QAT 优于 PTQ，也没有证明任一量化模型达到质量门槛；足量多初始状态/多种子结果仍待测。RK3588 上目前只执行了第 0 层一个 MLP 的 PTQ/QAT 子图，没有整模型 RKNN 导出、完整策略执行、峰值 RAM、端到端延迟/功耗测量或完整部署收益结果。全模型 HAQ 搜索代码也尚未形成可运行的完整评价闭环；无这些证据时，不把单子图结果称为完整端侧部署或 RK3588 加速。
