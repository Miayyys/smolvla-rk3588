# SO101仿真抓放：文档与产物索引（归档，2026-10-19）

> 迁移（2026-10-19）：下文 `runs/so101_*`、`runs/so100_*`、`artifacts/smolvla_base`、`artifacts/transfer/smolvlm2_assets`、`artifacts/datasets/*` 与 `.venv-haq-local` 已移到 `/home/loser/Study/so101-vla-edge`（`runs/`、`artifacts/`、`.venv`），代码在其 `so101vla/` 中，见 [cleanup-2026-10-19.md](cleanup-2026-10-19.md)。

分支`experiment/so101-goal-conditioned`（远端`smolvla-rk3588`）。本地RTX 4060 Laptop；GPUServer已停机。产物在主仓库`runs/`（Git忽略；worktree中的`runs/`、`artifacts/`为指向主仓库的软链接，已加入本地exclude）。环境：`.venv-haq-local`，Nexus源码`/home/loser/Study/so101-nexus`（`7bdb671`）。

## 1. 文档

| 文档 | 内容 |
|---|---|
| [so101-scripted-data.md](so101-scripted-data.md) | 脚本专家数据v1/v2、基线A（GPUServer）、第5000/10000步闭环、欠拟合诊断、服务器停机与本地重建v2 |
| [so101-goal-conditioned-design.md](so101-goal-conditioned-design.md) | 目标条件化分层策略设计；实验B（真值目标，第5000步止损）；评测渲染非确定性 |
| [so101-dagger.md](so101-dagger.md) | DAgger第一轮（专家接管标签）、目标结构在同一DAgger数据上（真值/规划头/空目标）、任务范围、链式长任务（部分） |
| [project-route.md](project-route.md) | 路线摘要 |

## 2. 结论速览（仿真，100个验证种子×2次，每块执行10步）

| 模型 | 结构 | 推理输入 | 完成 |
|---|---|---|---|
| A 第10000步 | 原SmolVLA | 图像+关节 | 13/200 |
| **DAgger第一轮** | 原SmolVLA | 图像+关节 | **135/200**（当前主线） |
| 目标结构+DAgger，规划头 | +目标嵌入/规划头 | 图像+关节 | 127/200（与主线无差别，p=0.89） |
| 目标结构+DAgger，真值目标 | 同上 | +仿真真值目标 | 149/200（上限；优势不显著，p≈0.2） |
| 目标结构+DAgger，空目标 | 同上 | 图像+关节，空目标 | 45/100 |

测试集（种子200000–200049）从未使用。

## 3. 保留的产物

| 路径（`runs/`下） | 大小 | 内容 | 校验（SHA256前16位） |
|---|---|---|---|
| `so101_scripted_data_v2/` | 16G | v2数据（本地重建，标签与A所用逐位一致） | manifest `a6896c8d72852b78`，train.npz `5f25470493b0a46a`，stats `cc2a6e1afb978a62` |
| `so101_backup/full_v2_step10000/` | 2.5G | 基线A第10000步（GPUServer）+续训状态，`SHA256SUMS`逐文件 | model `a3bc9ece544b612c` |
| `so101_goal_B/` | 4.2G（btrfs共享数据块） | 实验B第5000步：`last/`、`snapshot_step5000/best/`（同一模型）、续训状态 | model `ea475a537ac75d1f`，state `7d835cad261731c2` |
| `so101_dagger_r1_data/shard{0,1}/` | 1.2G | DAgger第一轮样本（rollout、跟踪专家快照、接管动作块、JPEG） | train.npz `8f5d9795cad2b023` / `23aeb71a0a85fa87` |
| `so101_dagger_r1_data_goal/shard{0,1}/` | 95M | 同一样本加目标标注（JPEG为软链接） | `7ce008b61e0eee96` / `0f9313d6925f4114` |
| `so101_dagger_r1/` | 2.5G | **主线模型**：DAgger第一轮第5000步 `last/`+续训状态 | model `4c10aa0555da9f6b`，state `7a2cd3782c95c5bf` |
| `so101_dagger_r1_goal/` | 2.5G | 目标结构+DAgger第5000步 `last/`+续训状态 | model `33826682fabb044b`，state `c93fcdf8461b16d0` |
| `so101_A_step10000_local_eval_100*/`、`so101_dagger_r1_eval_100_rep{1,2}/`、`so101_dagger_r1_goal_eval_*/`、`so101_goal_B_eval_*/` | 各≤116M | 闭环评测报告（部分含逐步轨迹`trace_seed*.npz`与GIF） | 见各文档 |
| `so101_chain_dagger_r1_rep1/`、`so101_chain_goal_planner_rep1/` | 109M/74M | 链式长任务评测（中途停止：只有GIF与`.log`逐回合结果，无`report.json`） | — |
| 各`*.log`、`*_pipeline.log`、`so101_dagger_takeover_check.json` | 小 | 流水线与检查记录 | — |

整理（2026-10-19）：`so101_dagger_r1/best`、`so101_dagger_r1_goal/best`、`so101_goal_B/best`与同目录`last`（及B的快照）逐文件相同（`diff -r`无差异、模型SHA256相同），已删除，保留`last`（续训状态校验的就是`last`）。文件系统为btrfs，复制产生的是共享数据块，删除后可用空间未变化。中止的对照训练`so101_dagger_r1_control/`只有prepare与27步记录、无checkpoint；测速/冒烟目录（`so101_goal_B_probe*`、`*_smoke*`、`so101_goal_B_check`）只含日志与事件，均保留。

## 4. 复现入口

- 数据生成：`config/so101_scripted_data_v2.json` → `qvla/evaluation/generate_so101_scripted_data.py`
- 训练：`qvla/evaluation/finetune_so100_real.py --config <config> --stage prepare|smoke|train [--resume] [--stop-step N]`；配置`config/so101_scripted_full_v2.json`（A）、`so101_goal_B.json`、`so101_dagger_r1.json`、`so101_dagger_r1_goal.json`、`so101_dagger_r1_control.json`（未完成）
- DAgger数据：`qvla/evaluation/collect_so101_dagger.py rollout|label|goal-labels`；流水线`scripts/so101_dagger_r1_pipeline.sh`、`so101_dagger_r1_goal_pipeline.sh`
- 评测：原结构`qvla/evaluation/finetune_so101_scripted.py`；目标结构`qvla/evaluation/so101_goal_eval.py --goal-source oracle|takeover|planner|null`；链式`qvla/evaluation/so101_chain_eval.py`
- 检查与分析：`check_so101_goal.py`、`scripts/so101_dagger_takeover_check.py`、`scripts/so101_grasp_drop_analysis.py`、`scripts/so101_eval_determinism_probe.py`
- 无人值守：`systemd-run --user ... systemd-inhibit --what=idle:sleep:handle-lid-switch ... bash <pipeline>`（未开启linger，运行期间不要注销）

## 5. 空间不够时可删除的候选（未删除，仅建议）

主仓库`runs/`里本实验线之前阶段的产物：`so101_scripted_data_v1/`（1.8G，v1数据，v2已覆盖其种子）、`so101_scripted_full_v1/`（1.8G）、`so101_scripted_overfit5_v1/`（1.7G）、`so101_sim_finetune_matched_v1/`（1.8G）、`so101_sim_overfit_source3_v1/`（1.7G）。它们是旧文档结论的依据，删除前应确认不再需要复查。本线保留的`so101_goal_B/`（停止的方向）若不再续训，可删`training_state.pt`。
