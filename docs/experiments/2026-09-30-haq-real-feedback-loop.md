# 本地全策略真实反馈 RL 诊断闭环

## 问题、判据与范围

这次验证控制器能否把**同一份完整配置**用于模型打包、严格重载、真实 SmolVLA 完整动作计算、RK3588 签名成本查表、模型文件体积检查及下一轮 RL 更新。以前的多轮测试只用了人工指定的合成奖励，不能回答此问题。

预设成功判据：每份配置覆盖当前304个可调位点；每份量化文件实际生成并严格重载；40个开发观测的动作与原始 FP 缓存比较；奖励同时使用动作、体积、硬件表；连续至少3轮；关闭熵奖励后梯度非零；独立读取落盘证据能逐项重算奖励并精确复现控制器更新。

这是**本地数值参考实现的诊断搜索**。尚无完整 RKNN 执行图、整策略板端成功率/延迟/内存；本轮不会确定最终位宽。INT16、DFP、Conv INT8 的本地计算使用压缩权重及显式量化数值参考，未验证与 RKNN 后端逐算子一致。Linear W8A8 在 RTX 4060 通过 CUDA `torch._int_mm` 做 INT8×INT8→INT32 后缩放。CPU embedding 候选的本地执行在 GPU 进行行查表，速度只取另测的板端 CPU 行查表成本。

## 固定输入与复现配置

- 原始 checkpoint：`lerobot/smolvla_libero`，文件 906,712,520 B，SHA256 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`；processor、VLM 资源、数据和版本随本轮 `summary.json` 的 `identity` 固定。
- 分区：`config/evaluation_partition_v2.json`，SHA256 `f755546a6b074d2fe248333fc42c3dbf30f9b54b9f0fa0b58a16506a54d942c2`。40 个校准 episode 与 40 个开发 episode 不重叠；冻结测试 episode 没有参与。校准从每个 episode 的首帧收集全部被调用模块的输入范围。开发按 `fp_cache40` 固定每任务一条开始/中间/90%观测及其噪声。
- FP 缓存 manifest SHA256 `058ba94ee81c19785edffb0bc3724d8916f3e8867e8abfa846b84c8a2ae69c87`；重载 FP 后对40个动作块做逐元素相等校验。校准原始范围 SHA256 `d5d9c026bd0bf1b742877f523acb859a9d6c99f7adcf2a0e8940c5c65ab1fed3`；执行次数/形状 SHA256 `09b9b46ca753950384e31b863b3d4339b693d3cdadd9098c13f231876dc5a5b2`。
- 硬件表 `docs/hardware/tables.json` SHA256 `ececfdd22f65bcc4258ecbfd9bbf6b63fae461189599fb42f89c74ef48136655`；目前397个参数模块清单、304个多候选位点、1517个选择。没有根据旧人工灵敏度固定模块。初始化给 INT8 候选有限的 logit +5，利用体积硬条件提高抽到可行文件的概率；其他选项概率仍大于零。此先验**不来自人工位宽图或任务质量结论**。
- 本机 RTX 4060、隔离 `.venv-haq-local`、seed 29、3轮×每轮2份配置、hidden size 16、学习率 $10^{-3}$、梯度裁剪范数5、熵权重0。FP 模型每个候选重新从原始模块构建，不能把候选反复量化；safetensors 文件严格重载后才评分。

量化数值配方按输出通道（Conv 也按输出通道）保存权重：对 $b\in\{8,16\}$，$Q=2^{b-1}-1$，$s_{w,c}=\max_i|w_{c,i}|/Q$，$q_{w,c,i}=\operatorname{clip}(\operatorname{round}(w_{c,i}/s_{w,c}),-Q,Q)$。Embedding 的 INT8 也按行保存 scale。普通 INT8/INT16 激活使用校准输入范围 $[l,h]$：$s_a=(\max(h,0)-\min(l,0))/(2^b-1)$，$z_a=\operatorname{clip}(\operatorname{round}(-2^{b-1}-\min(l,0)/s_a),-2^{b-1},Q)$，$q_a=\operatorname{clip}(\operatorname{round}(x/s_a)+z_a,-2^{b-1},Q)$。零范围以最小正 scale 保护。W8A8 Linear 的整数累加再减 $z_a\sum q_w$ 并乘 $s_as_w$；INT16/Conv 在本地解码后用浮点核计算。`w16a16i_dfp` 在本地把最大权重/激活 scale 向上取2的整数幂，用对称零点；这是参考近似，须与RKNN实现比对。FP16/BF16 候选按对应格式保存权重，本地运算前转换输入并将输出还原调用者 dtype。

复现命令，输出目录须不存在：

```bash
.venv-haq-local/bin/python scripts/run_haq_local_loop.py \
  --output runs/haq_real_feedback_local_v3 \
  --rounds 3 --batch-size 2 --seed 29 \
  --int8-logit-prior 5 --entropy-weight 0
.venv-haq-local/bin/python scripts/audit_haq_local_loop.py runs/haq_real_feedback_local_v3
```

运行脚本 SHA256 `ebd7d17aeeb4bf6f65343dda0a30345d52860ba9eeb476288a2d155e0c4ddffa`，量化执行模块 SHA256 `d9832627af3075f264f4b4aa62d1dad7a395ad10aa8c5bbd9e7df0bd69249cd5`，控制器 SHA256 `810e4d5c1e8987029067cef80ca210ca48c835e2885634632964da91c07e451e`。原始每候选配置、模型、动作、质量分项和304行查表结果均保存在 `runs/haq_real_feedback_local_v3/round*_candidate*/`；每轮控制器参数在同目录 `controller_round*.npz`。

## 奖励计算

本轮用以下**待闭环验证的动作代理**；不能把它当作成功率：

\[
E=\operatorname{task\;macro\;mean}|a_{q,50\times7}-a_{FP,50\times7}|,\quad
s=\operatorname{mean}|a_{FP}|=0.2664106727,\quad A=e^{-E/s}.
\]

\[
\widehat T(\boldsymbol b)=\sum_l c_l\, t_{l,b_l}^{\rm signature,p50},\quad
\widehat T_{\rm ref}=15757.5526\;\mathrm{ms},\quad
G=\sqrt{A\,\widehat T_{\rm ref}/\widehat T(\boldsymbol b)}.
\]

参考成本为所有可调算子选择 BF16、embedding 保持原生 BF16 查表时的**同一套签名求和**。它不是原始 FP 的板端整策略延迟。候选成本用开发动作中实际观测的平均调用次数乘对应板测签名 p50，再求和；没有融合调整，不含 norm、softmax、注意力核心、Flow 调度和真实跨设备边界成本。每份候选有1个位点的输入长度与代表签名不一致：文本 embedding 实际 `[1,48]`，成本签名 `[1,177]`。因此 $\widehat T$ 仅是统一规则下的粗略成本排序。

压缩率 $C=1-B_q/B_{FP}$ 使用**严格重载过的真实 safetensors 文件字节数**，并用整数条件 $5B_q\le 3B_{FP}$ 判定至少40%。奖励为：

\[
r=\begin{cases}G/(1+G),&5B_q\le 3B_{FP},\\
-(0.4-C)/0.4-10^{-6},&\text{否则。}\end{cases}
\]

未达标候选的负奖励只提供距离体积线还有多远的信息；不能在最终排序中击败任何达标候选。动作代理与速度使用同样 $1/2$ 的指数，是本轮机制验证的临时设定；它没有经过闭环结果校准。以前3轮均给未达标候选相同 $-1$，6份随机配置只压缩5.1%～11.4%，梯度主要由熵项产生；此失败试验保留在 `runs/haq_real_feedback_local_v1/`。第二次体积初始化试验12份候选在 `v2/`，熵权重仍为0.002；最终以熵权重为零的 `v3/` 判断真实反馈更新。

## 实际结果

| 轮/候选 | 文件 B | 压缩率 | 完整动作块 MAE vs FP | 查表 $\widehat T$ (ms，代理) | 奖励 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1/1 | 508,149,600 | 43.96% | 0.025477 | 7276.59 | 0.583829 |
| 1/2 | 511,139,168 | 43.63% | 0.024572 | 7427.09 | 0.581754 |
| 2/1 | 512,231,688 | 43.51% | 0.025515 | 7403.57 | 0.581709 |
| 2/2 | 512,283,664 | 43.50% | 0.025554 | 7236.49 | 0.584466 |
| 3/1 | 505,834,256 | 44.21% | 0.025774 | 7250.76 | 0.584126 |
| 3/2 | 522,305,120 | 42.40% | 0.024728 | 7447.98 | 0.581341 |

6/6 都达到40%实际文件压缩。每份候选40条动作的本地主机总前向时间约8.77～8.90秒，绝非 RK3588 整策略时延。整次校准、FP校验、候选评估与更新耗时90.60秒。每轮奖励标准差分别为0.001038、0.001378、0.001393；关闭熵奖励后的梯度范数分别0.01113、0.01620、0.04284，三次参数更新后与初始参数 $L_2$ 差0.05254。轮均奖励分别0.582792、0.583087、0.582733，**目前没有稳定提升趋势**。

按当前临时代理奖励，`v3` 中最高的第2轮候选2有292个 W8A8、7个 W16A16I、1个 W16A16I DFP、2个 FP16、1个 BF16 和1个 INT8 embedding；文件512,283,664 B，SHA256 `9872b1b79b976b2525e8aed8be9ab8a10b8b3800910e0a4816df2300182f3dbe`。这是供下一轮验证的**诊断候选**，还不能当作最终 HAQ 配置。

审计脚本重新读取6份模型文件、动作数组、FP缓存和板测表，重算质量与奖励；从初始控制器重放全部6次采样和3次更新，逐元素比对每轮参数及Adam动量。`runs/haq_real_feedback_local_v3/audit.json` 为 `passed`。图与原始报告分别在 [PNG](../../figures/haq-real-feedback-local-v3.png)、[SVG](../../figures/haq-real-feedback-local-v3.svg) 和对应 `runs/` 目录；图中速度轴已标成独立成本求和代理，未画任务成功率。

## 结论边界与后续验证

本地**真实动作反馈的控制器循环已接通并复核**。这证明从完整配置到奖励再到下一轮更新能运行，也证明保存的混合精度文件满足体积条件。它尚不能说明搜索出的策略在 LIBERO 更好：历史专家 PTQ/QAT 更接近数据集首动作但闭环各31/40、低于 FP32/40，已构成离线代理失效的反例。当前约0.025的动作块偏差比历史仅专家 INT8 的约0.002大，必须做小规模配对闭环筛查。

完整执行图仍需检查非参数算子、融合、转换和93条非活动/待核实参数行。INT16/DFP/Conv 候选需与真实 RKNN 数值对齐；签名速度需经过整策略板端校准。最终质量需要独立闭环任务成功率及多种子检验，`search_ready=false` 保持不变。按用户顺序，HAQ 确认的配置之后再做 QAT 与从原始 FP 独立做同格式 PTQ，且都要严谨重载真实转换产物。
