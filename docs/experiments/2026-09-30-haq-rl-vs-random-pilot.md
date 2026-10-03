# 真实模型混合精度 RL 与随机搜索等预算对照

## 问题与判据

问题：增加到30轮后，RL 是否能比**同样评价120份真实量化模型**的随机搜索更好地选择 SmolVLA 精度？本轮只检验当前本地评价代理的搜索价值，不以它认定最终 RK3588 部署位宽。主要比较同预算最高可行奖励、搜索末段奖励均值；再用未参与搜索的动作和同种子 LIBERO 配对任务检查代理外推。若只改善代理而不改善任务成功数，只能称为“学会优化代理”。

## 输入、方法与原始证据

- FP checkpoint `lerobot/smolvla_libero`：906,712,520 B，SHA256 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`。模型、processor 和资源逐文件哈希见 `runs/haq_rl_trial_30x4_rl/summary.json` 的 `identity`。
- FP 动作缓存40条，manifest SHA256 `058ba94ee81c19785edffb0bc3724d8916f3e8867e8abfa846b84c8a2ae69c87`。搜索只用固定任务索引 `[0,5,10,15,20,25,30,35]` 的8条动作；其他32条只用于最终最佳候选留出检查。40个独立校准 episode 用于激活范围；冻结测试 episode 未用。划分 SHA256 `f755546a6b074d2fe248333fc42c3dbf30f9b54b9f0fa0b58a16506a54d942c2`。
- RK3588 硬件签名成本表 SHA256 `ececfdd22f65bcc4258ecbfd9bbf6b63fae461189599fb42f89c74ef48136655`。304个可调位点、1517个列出的格式选择；INT8、FP16、BF16、两种INT16数值配方以及 CPU INT8 embedding 查表进入当前动作空间。W4A16/W8A16 不在当前 RK3588 可用候选内。没有按历史人工敏感度固定任何可调位点。
- RTX 4060 Laptop 8 GB，本地 `.venv-haq-local`；控制器 seed 29、hidden size16、REINFORCE、Adam学习率 $10^{-3}$、梯度裁剪范数5、熵权重0。两组初始参数和有限 INT8 logit `+5` 相同；前4个候选及反馈逐项相同。该先验只为更容易满足文件压缩约束，其他选项概率仍大于0。RL 每4份候选更新一次，随机组保持初始分布且不更新。一个种子、各30轮×4候选；**这不是收敛性或多种子显著性实验**。
- 每个候选从 FP 原始模块重建，压缩权重写入 `.safetensors` 并严格重载，计算完整 SmolVLA 动作块。仅保留各组当前最优模型文件；所有120份候选的配置、动作、评分、字节数、哈希及查表明细均保存。脚本 `scripts/run_haq_local_loop.py` SHA256 `b1bcfcc8ce4010b95947229a31caf9141588048ffcc5279afb7b14ddb9cf06bc`；审计脚本 SHA256 `95860f1ee7ca62de89720288463f15052e38a86d4f2ed2436a90d89410d8d989`。

沿用[真实反馈闭环实验](2026-09-30-haq-real-feedback-loop.md)的量化数值配方和奖励。令 $E$ 为8条搜索动作对 FP 的按任务宏平均动作块 MAE，$s=\operatorname{mean}|a_{FP}|=0.2544052330$，$A=\exp(-E/s)$。RK3588 签名成本求和为 $\widehat T$，参考求和 $\widehat T_{ref}=15757.5526$ ms；$G=\sqrt{A\widehat T_{ref}/\widehat T}$。若真实文件字节数 $B_q\le0.6B_{FP}$，奖励 $r=G/(1+G)$；否则给负的压缩缺口奖励。$\widehat T$ 是独立算子成本求和，**不是整模型板端延迟**。搜索评价动作误差也不是任务成功率。

复现命令（输出目录须不存在）：

```bash
.venv-haq-local/bin/python scripts/run_haq_local_loop.py --output runs/haq_rl_trial_30x4_rl --rounds 30 --batch-size 4 --seed 29 --int8-logit-prior 5 --score-task-indices 0 5 10 15 20 25 30 35 --checkpoint-retention best
.venv-haq-local/bin/python scripts/run_haq_local_loop.py --output runs/haq_rl_trial_30x4_random --rounds 30 --batch-size 4 --seed 29 --int8-logit-prior 5 --score-task-indices 0 5 10 15 20 25 30 35 --checkpoint-retention best --random-control
.venv-haq-local/bin/python scripts/audit_haq_local_loop.py runs/haq_rl_trial_30x4_rl
.venv-haq-local/bin/python scripts/audit_haq_local_loop.py runs/haq_rl_trial_30x4_random
.venv-haq-local/bin/python scripts/compare_haq_rl_random.py --rl runs/haq_rl_trial_30x4_rl --random runs/haq_rl_trial_30x4_random --output runs/haq_rl_trial_30x4_comparison
.venv-haq-local/bin/python scripts/run_haq_local_paired_panel.py --run runs/haq_rl_trial_30x4_rl --output runs/haq_rl_trial_30x4_rl_panel --fp-reference-output runs/haq_libero_panel_v1
.venv-haq-local/bin/python scripts/run_haq_local_paired_panel.py --run runs/haq_rl_trial_30x4_random --output runs/haq_rl_trial_30x4_random_panel --fp-reference-output runs/haq_libero_panel_v1
```

对照脚本 SHA256 `ab416b3aab61b8546c97d1f2017426bb381db942b7f0a1f5823913f2fbb06ff0`。逐候选原始数据在两组 `round*_candidate*/report.json`、`assignment.json`、`actions.npz`；更新在 `updates.json`，留出动作在 `heldout_best.json`，独立复算在 `audit.json`。对照摘要、CSV和图在 `runs/haq_rl_trial_30x4_comparison/`；可查看[奖励曲线](../../figures/haq-rl-vs-random-30x4.png)。

## 实际结果

| 指标 | RL 30轮 | 不更新的随机30轮 |
| --- | ---: | ---: |
| 真实量化候选数 / 达到40%文件压缩 | 120 / 120 | 120 / 120 |
| 最优代理奖励 | 0.590129（第95份） | 0.589921（第7份） |
| 前20份平均奖励 | 0.586395 | 0.586239 |
| 后20份平均奖励 | 0.588854 | 0.585631 |
| 最优模型文件 | 502,316,720 B | 507,198,944 B |
| 相对原始文件压缩 | 44.60% | 44.06% |
| 最优候选8条搜索动作 MAE vs FP | 0.016100 | 0.014968 |
| 最优候选32条留出动作 MAE vs FP | 0.027217 | 0.027665 |
| 最优候选签名成本求和代理 | 7135.20 ms | 7179.36 ms |
| 控制器参数相对初始 $L_2$ | 0.27647 | 0（无更新） |
| 完整运行时间 | 422.95 s | 420.64 s |

审计各自复算120份候选的动作评分、真实文件大小、成本和奖励，并重放采样与控制器状态：RL 30次更新、随机0次更新，均通过。RL 最优配置为299个 W8A8、2个 W16A16I、1个 FP16、1个 BF16及1个 CPU INT8 embedding；随机最优配置为298个 W8A8、2个 W16A16I、1个 W16A16I DFP、1个 FP16、1个 BF16及1个 CPU INT8 embedding。其余选项可采样但本轮在 `+5` INT8 先验下探索很少。

LIBERO 配对闭环小样本：四个 suite 各取任务ID 0、3、6；每任务同一初始 seed 0、每模型1回合，复用同环境的 FP 结果。RL 最优 **6/12**，随机最优 **6/12**，FP **7/12**。RL 对 FP 新增1个成功、丢失2个；随机新增3个、丢失4个。RL 与随机在4个任务上结果相反，合计通过数相同。原始任务记录在 `runs/haq_rl_trial_30x4_{rl,random}_panel/summary.json`。这个12任务面板单种子且是开发筛查，不能据它估计最终通过率。

另一个重要反例：更早 `v2` 离线代理选出的模型，在本地同环境的40任务配对测试中 FP 为26/40、候选18/40（`runs/haq_libero_full_v2/summary.json`）；不能拿这组本地FP与旧A10环境的32/40直接比较。这说明当前离线代理可能挑出任务质量退化的模型。

## 结论与边界

**30轮足以显示控制器确实学会提高当前代理的平均奖励；不足以证明它比随机搜索找到任务质量更好的混合精度。** 最高代理奖励差仅0.000208；留出动作差很小；12任务通过数持平且都低于FP。扩大到更多轮可能改善这个代理，当前证据却不支持直接把更多轮的最高奖励当成最终配置。下一次要先改进与任务成功率相关的快速质量反馈，并多种子重复同预算对照，再考虑扩大轮数。

当前 RK3588 数据来自签名级子图和 CPU 查表；SmolVLA 整策略未在板端运行。后续可能需要视觉 RKNN、语言 RKLLM、动作头 RKNN/CPU 的拆分，RKLLM 的精度控制粒度可能与本轮304个位点动作空间不同。因此这些数字不代表整策略 RK3588 速度、内存或可部署性；保留 `search_ready=false`，不把本次最优候选称作最终 HAQ 位宽图。
