# QVLA 固定评测协议（候选模型决策前）

本协议适用于同一任务 checkpoint 的 FP、RK3588 硬件感知 QAT 和独立 PTQ。模型更换时重新锁定数据、代码、processor、动作空间和协议版本；不同模型不能直接比较原始动作 MAE。

## 依据与主指标

- VLA 任务质量以 **LIBERO 完整闭环 rollout 成功率**为主指标，按 Spatial、Object、Goal、Long 四个 suite 分别报告，再报告四个 suite 的等权平均。OpenVLA 的公开评测脚本默认每个 suite 10 个任务、每任务 50 回合，并对论文结果取 3 个随机种子；LeRobot 的 LIBERO 文档给出每任务 10 回合的复现实例。正式结果采用 **40 任务 × 50 回合 × 3 种子/模型**，资源不足时先执行 40 × 10 × 1 的开发评测并明确标成开发结果。[OpenVLA 评测说明](https://github.com/openvla/openvla#launching-libero-evaluations) · [LeRobot LIBERO 文档](https://github.com/huggingface/lerobot/blob/main/docs/source/libero.mdx)
- 每个模型使用相同任务、初始状态索引、环境配置、回合上限、动作 chunk/重规划间隔、相机映射、processor 和推理随机种子。逐任务保存成功数/回合数及失败原因，报告跨种子均值、标准差和 95% 置信区间；比较 QAT/PTQ 与 FP 时同时报告配对成功率差（百分点）。如果量化模型的动作策略需要改变以上任一配置，单列为新实验。
- 离线训练、校准、开发选择和最终测试按 episode 隔离；模拟评测固定 benchmark 初始状态与种子，并登记与训练数据的关系。正式测试回合不得用于选 bit、阈值、QAT step 或子图切分。每个 checkpoint 与配置保留 hash。当前 40 条首帧对记录动作 MAE 仅是诊断，不是策略成功率。

2026-09-27 的真实 GPU W8A8 配对闭环筛查只运行 **40 个任务 × 每任务 1 回合 × 1 seed**，低于计划中的 40×10×1 开发评测和 40×50×3 正式协议；它用于捕捉明显任务回归。该批次重跑 FP 为 32/40，而此前独立单回合基线为 31/40，说明该小样本受策略采样影响。只用同批次 FP 与量化结果作配对描述，不合并两批，也不作统计等效结论；完整数值见[配对实验记录](experiments/2026-09-27-paired-w8a8-rollouts.md)。

## 离线诊断与系统指标

| 维度 | 固定测法 | 输出 |
| --- | --- | --- |
| 动作保真 | 冻结测试 episode 多时间点，同输入和采样噪声，对 postprocessor 后的完整动作 chunk 比较 FP；另与示范动作比较 | 全 chunk MAE/RMSE、首步与末步误差、夹爪误判率、越界率；每任务分布 |
| 数值定位 | 同一校准数据，按视觉编码、语言主干、动作专家和导出子图记录量化前后激活/输出 | 重建 MSE、余弦相似度、异常层及其位宽；只用于选型，不替代 rollout |
| 模型存储 | 统计实际可加载模型、scale/zero-point、processor、子图与运行依赖 | 各文件字节数、总安装占用、实际权重位宽及覆盖率 |
| 推理 | 板端 batch=1，固定图像数量/分辨率和动作生成步数，预热后至少 100 次；CPU、NPU 计时同步 | 纯模型与含前后处理/数据传输的端到端 p50/p95/p99、冷启动、动作频率、CPU/NPU 占比 |
| 内存与能耗 | RK3588 的 Linux+Zephyr 目标镜像；记录 Linux 可用内存、RSS/PSS 峰值、NPU/CMA 使用和 OOM 日志；有可靠电源计才测功耗 | 峰值与稳态 RAM、是否完整运行、可测时的每动作能量；不可测填 NA |

主结论按 **成功率—板端峰值内存—端到端延迟** 给出，不以模型文件大小或 GPU latency 代替板端结果。若 FP16 无法运行而量化模型可以，必须在相同输入与功能范围下记录 FP16 失败日志、可用内存和量化版完整 rollout；明确是内存不足还是算子不兼容。不能用“FP 无法导出、量化版只运行子图”证明量化使完整 VLA 在 4 GB 板上可用。

### 候选方案的决策优先级

2026-09-30用户更新要求：RL可探索全部硬件可行组合，完整推理模型实际文件至少减少40%，在这个条件下联合优化效果与速度。速度来自已有RK3588实测表的查表估算，明确标记为估算。旧的逐suite非劣门槛及RAM/延迟/体积G不再作为当前搜索排序规则；内存和完整板端执行仍是部署验证项目。

每次候选快速反馈采用固定开发观测、固定噪声与缓存FP完整动作进行比较，记录任务等权的动作偏差、连续控制与夹爪差异。离线代理不能称为闭环成功率：定期对有希望的候选做小规模闭环，最终候选采用本页既定独立测试协议统一确认，报告效果—查表速度Pareto及[当前排序设置](quantization-technique-plan.md#四模块精度的选择方法)。动作代理与真实成功率的对应关系尚未建立，不能据更低MAE宣布实际效果更好。实现和本地40观测验证见[FP动作缓存记录](experiments/2026-09-30-offline-action-cache.md)。

## RK3588 硬件感知 QAT 与 PTQ

先固定 **板端可实现配置**：导出并验证视觉、语言和动作子图；记录 RKNN/RKLLM 版本、驱动、支持的算子、shape、INT8/FP16/W4 路径、CPU fallback 及数据传输；用实际板端延迟和内存而不是 FLOPs 作选择。HAQ 论文的核心就是把目标硬件的延迟与能耗反馈放进混合精度搜索。[HAQ, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_HAQ_Hardware-Aware_Automated_Quantization_With_Mixed_Precision_CVPR_2019_paper.html) · [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2)

RKNN-Toolkit2 2.3.2 的 [Python 3.12 官方依赖文件](https://github.com/airockchip/rknn-toolkit2/blob/master/rknn-toolkit2/packages/x86_64/requirements_cp312-2.3.2.txt)写明 `torch>=1.10.1,<=2.4.0`、`numpy<=1.26.4`；仓库首页只列 Python 版本，没有列 PyTorch 上限。当前独立探针环境借用系统 PyTorch 2.7，实际已完成 MLP 子图 FP16/INT8 编译与主机模拟器推理，但这不属于依赖文件声明的版本组合，正式复现仍需用支持的版本核对。其[更新记录](https://github.com/airockchip/rknn-toolkit2/blob/master/CHANGELOG.md)明确把 W4A16 支持标在 RK3576；2.3.2 虽增加 W4A16 分组功能，却未据此证明 RK3588 支持该格式。因此 RK3588 的当前 NPU 目标先以 FP16/INT8 子图和混合量化为主，W4 路径须有具体版本的转换与板端运行实证才加入。

1. **FP 基线和候选子图**：FP16 为主要板端浮点对照；FP32 可作数值参考。对每个子图先测 FP16 可导出性和板端成本，列出不能导出的算子。沿同一切分做 INT8 与 FP16 混合配置；W4 仅在 RK3588 的相应后端、模型架构和转换版本实际通过时纳入候选。当前 torchao W8 checkpoint 只是 A10 研究产物。
2. **共享量化目标**：为 QAT 和 PTQ 固定同一子图、可量化层、粒度、权重/激活格式、浮点保留层、预处理与预算。用校准集比较 clipping/scale 及层敏感度，在开发集和板端测成功率代理、延迟、内存，形成 Pareto 候选；不得先看最终测试结果再改配置。
3. **QAT**：从原始 FP checkpoint 开始，按目标后端的 scale、rounding、clamp、激活量化边界做 fake quant 与 STE 微调；训练后导出所需 Q/DQ 或其他被工具链接受的格式，编译真实 RKNN/RKLLM 子图并在板端验证。Rockchip 仓库提供 QAT 示例，不能假定任意 torchao 格式可直接导入。[RKNN QAT 示例](https://github.com/airockchip/rknn-toolkit2/blob/master/rknn-toolkit2/examples/pytorch/resnet18_qat/README.md)
4. **PTQ**：从同一个原始 FP checkpoint 独立出发，对相同硬件目标用隔离校准集做权重与激活统计、混合量化和编译。与 QAT 比较同一批完整回合；不对已经转换的 QAT 低比特 checkpoint 再做 PTQ。
5. **硬件验收**：两个产物都要有实际可加载低比特数据、编译/运行日志、覆盖率和端侧测量。RKNN-Toolkit2 的官方混合量化示例采用 step1 分析和 step2 生成；版本 2.3.2 又加入自动混合精度，但每个子图仍需验证。[混合量化示例](https://github.com/airockchip/rknn-toolkit2/blob/master/rknn-toolkit2/examples/functions/hybrid_quant/README.md) · [版本说明](https://github.com/airockchip/rknn-toolkit2)

## 当前状态

现有 FP/QAT/PTQ 的 40 条单帧离线比较、实际文件大小和 A10 延迟属于**首轮诊断**；新服务器又完成 FP 闭环 40 任务各 1 回合的开发基线。RK3588 的一个真实动作专家 MLP 子图已完成 FP16/INT8 和两个单层 FP16 混合配置的编译及主机模拟器激活对照；混合配置未改善开发集平均输出误差。完整 VLA 的 RKNN 转换、正式多种子 rollout、板端内存/延迟/能耗仍未测量。扩展 MLP 校准探针查看了原测试划分中每任务 1 个 episode，属于探索分析；之后的冻结测试须使用[`evaluation_partition_v2.json`](../config/evaluation_partition_v2.json)指定的剩余 213 个 episode，选型须从训练划分内独立开发 episode 完成，且这些开发 episode 不参加后续 QAT 训练。QAT 400 步配方在旧的 40 条上退化，不能直接作为硬件感知 QAT 成品。已完成步骤的原理、配置、原始报告和结论边界见[实验记录索引](experiments/README.md)。
