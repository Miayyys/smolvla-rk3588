# QVLA 项目路线与交付记录

更新：2026-10-03。本文件是项目完整路线的主阅读入口，按实际决策和结果叙述；精确配置、公式、扫描原始数据与失败细节仍由链接的技术记录承载，尚未逐项迁移，因此暂不删除原记录。

## 1. 目标与评价原则

以SmolVLA量化为主目标，结合量化学习笔记、HAQ思想和RK3588真实后端。经历FP基线、硬件探测、混合精度搜索、教师蒸馏、QAT/PTQ对照、后端转换与完整板端评价。

质量优先；搜索要求候选文件相对原checkpoint压缩至少40%，结合动作质量和板测成本评价。MSE/MAE用于快速筛选，不能代替LIBERO闭环成功率。最终部署经过用户同意改回原始FP16视觉，实际压缩35.15%，未达到早期40%目标。未测指标不以估算冒充。

校准/开发/测试按episode分离；动作比较固定噪声、种子与处理器。早期不同协议成绩不混用。量化训练的fake quant与真正整数产物/板端执行分别验证。

## 2. 基线与数据准备

使用`lerobot/smolvla_libero`及`lerobot/libero`；模型、processor和修订号锁定。准备数据划分、校准输入、离线动作缓存和LIBERO仿真环境。原checkpoint文件906,712,520字节。

入口：`download_and_upload.py`、`split_libero.py`、`fp_baseline.py`、`haq_offline_eval.py`。学习基础见[info.md](../info.md)，后续技术路线见[量化技术计划](quantization-technique-plan.md)。

## 3. 硬件格式探测与成本表

对RK3588进行编译、板端加载与数值检查，按后端/shape/接口判断候选可行性；不只依据SDK声明。建立100个基础签名的三轮板测及18项补充测试，包含相关边界与子图开销。

这些是成本查表数据，不是每种完整策略的实测延迟。语言切换RKLLM后，原RKNN查表成本不能直接当作新的RKLLM开销。

详细格式、签名和数据见[硬件表](hardware/README.md)。

## 4. HAQ风格强化学习搜索

枚举304个可调算子位点；按后端可行性掩码选择精度，不用此前人工敏感度结论锁定节点。实现自回归REINFORCE控制器、熵探索、平台早停及等预算随机对照；奖励来自真实离线动作评价、候选体积和板测查表。保存候选反馈并审计更新。

这不是原HAQ论文DDPG的逐项复现，亦未完成每候选整策略板测反馈。200轮×4候选等实验用于实际搜索与对照，不保证遍历组合空间或全局最优。

- V1：504,117,472字节，相对原checkpoint缩小44.40%。
- V2：513,906,600字节，相对原checkpoint缩小43.32%。

记录：[200轮](experiments/2026-10-01-haq-200-round-trial.md)、[探索调整](experiments/2026-10-01-haq-adjusted-exploration.md)、[至少200轮实验](experiments/2026-10-01-haq-exploration-min200.md)。

## 5. 教师蒸馏与可训练范围

使用OpenVLA-OFT真实教师推理，生成离线动作标签并审核有效时间步；训练使用GT及教师flow监督。同一噪声/时间用于相应损失比较。通过失败任务模块恢复与冻结范围实验排查回归。

最终学生选择只更新前8层专家线性模块的蒸馏结果，视觉、语言、后8层专家及动作接口冻结。GT/KD学生长任务均7/10，与原模型该协议下7/10持平；不能据此宣称蒸馏总体提高成功率。

记录：[回归诊断](experiments/2026-10-02-distillation-regression-diagnosis.md)、[前8层训练](experiments/2026-10-02-expert-first8-distillation.md)、[专家及接口对照](experiments/2026-10-02-expert-interface-distillation.md)。

## 6. QAT与独立PTQ

在蒸馏FP学生上按V1/V2配置做QAT，比较有/无教师损失。量化图覆盖all_sites，并非仅前8层量化；蒸馏可训练范围与QAT范围分开。QAT使用浮点master、STE与代理梯度，训练后校准、打包并严格重载整数产物。

所选V1无教师损失QAT：1000步，学习率1e-7衰减至1e-8，warmup/cosine，梯度累积2。无教师是QAT阶段教师损失关闭，学生本身仍经过蒸馏。另从独立FP checkpoint执行PTQ对照，不对已转换低比特权重重复PTQ。

四候选19任务筛选：V1有教师12、V1无教师14、V2有教师14、V2无教师10；按该筛选选择V1无教师，并非此后所有板端转换都保持相同质量。

记录：[四路QAT比较](experiments/2026-10-02-qat-four-way.md)、[持续评测总表](model-test-results.md)。

## 7. 真实RK3588部署与问题修复

先完成全RKNN分图与CPU胶水，再将语言适配RKLLM。包括两路图像、token embedding、state归一化、多模态prefix、语言K/V、10步Euler专家计算及动作后处理。

关键排查包括BOOL子图边界、BF16 native I/O、prefix注意力mask、KV缓存读取、长度对齐和工作池缓存一致性。当前RKLLM补丁绑定1.3.1及已核验二进制hash/偏移，不能宣称通用SDK支持或上游已接受。工作池映射修复通过A→B→A重复输入验证。

终版：原始checkpoint FP16视觉/connector（RKNN）＋V1 QAT W8A8语言（RKLLM）＋混合精度专家（RKNN）＋CPU INT8 embedding与胶水。视觉回退及语言统一W8A8意味着它不是原V1 HAQ图的严格等价转换。

记录：[缓存一致性](experiments/2026-10-03-rkllm-working-pool-coherency.md)、[后端适配](experiments/2026-10-03-v1-rknn-rkllm-adaptation.md)、[原始FP16视觉](experiments/2026-10-03-original-fp16-vision-rkllm-v1.md)。

## 8. 终版全面评价及审计

质量：40个任务，每任务固定初始状态的一次配对闭环；原模型在GPU，终版在真实板子。速度/内存：原FP16部署基线及终版在同一板子，预热3次、重复20次，延迟为生成50步动作块的完整推理耗时。

| 指标 | 原模型/基线 | 终版 |
| --- | --- | --- |
| 核心文件 | 原checkpoint906.71MB；板端基线945.56MB | 588.04MB |
| 板端P50 | 7.5199s | 7.2626s |
| 板端进程树峰值PSS | 1824.4MiB | 1129.9MiB |
| Spatial成功 | 8/10 | 5/10 |
| Object成功 | 8/10 | 8/10 |
| Goal成功 | 7/10 | 7/10 |
| Long成功 | 7/10 | 5/10 |
| 总成功 | 30/40 | 25/40 |

相对原checkpoint文件缩小35.15%；PSS降低38.07%；本次P50约快3.42%。Object/Goal类别合计持平不意味着逐任务相同或总体无损。40任务单回合不是多种子统计。

资源审计修正了遗漏RKLLM worker的旧采样；旧861MiB结果撤回。质量审计核对输入/噪声/hash/动作/任务计数，预处理误差接近浮点舍入；三个回归任务冷启动重跑轨迹逐位一致，未发现统计配置错误。仍未通过全中间张量对齐排除确定性的部署语义/数值误差，不将全部失败简单归因于量化位宽。

完整数据口径见[终版对比](final-model-comparison.md)，审计见[测试审计](experiments/2026-10-03-final-evaluation-audit.md)。

## 9. 模型文件在哪里

以下本地文件在整理时实际检查存在；远端路径来自实验记录，本次未重新连接核验。模型与数据被Git忽略。

| 类型 | 路径 | 说明 |
| --- | --- | --- |
| 原始SmolVLA | `artifacts/transfer/model/model.safetensors` | 同目录有配置与processor |
| HAQ V1 | `artifacts/candidates/v1/model.safetensors` | 原FP搜索候选，不是终版部署模型 |
| HAQ V2 | `artifacts/candidates/v2/model.safetensors` | 原FP搜索候选 |
| 早期V1/V2 QAT备份 | `artifacts/qat_backup/qat_distilled_full_v{1,2}_v1/` | 含master及packed；不是所选无教师损失QAT备份 |
| 终版RKLLM语言文件 | `runs/v1_rkllm_w8a8_g128_v1/language_w8a8.rkllm` | 本地存在，210.70MB；不包含视觉、专家及CPU参数 |
| 所选QAT master（服务器） | `GPUServer:/root/qvla/runs/qat_distilled_v1_no_teacher_v1/qat_float_master.safetensors` | SHA256 `4aeb92854d2bb89bac84a2d791d2acb4389b934178948c05533d6a52fe0b9f81`；本次本地未找到同名完整备份 |
| 终版部署（板子） | `root@10.42.0.252:/dev/shm/qvla_v1_rkllm_original_fp16_vision_v1/` | 记录中的整套入口，包含链接；内存盘重启会丢失，需按manifest收集真实目标文件 |
| 终版部署清单 | `runs/final_model_comparison_v1/final_deployment_manifest.json` | hash、实际文件及图身份；不是权重本体 |

终版是多个文件：`vision_connector_fp16.rknn`、`language_w8a8.rkllm`、三个专家RKNN子图和CPU参数，不是一个safetensors文件。尚需将终版所有真实文件整理为持久部署包；不能把本地单个语言文件称为完整备份。

## 10. 代码整理方向

不设archive。将重复实验编排合并为可配置通用入口，诊断工具独立，已替代无依赖代码删除。实验以本文为主入口，原始记录在细节迁移前保留。

可以取消顶层config目录，但不能丢弃精度图、位点候选掩码、数据划分、训练参数、版本和hash；这些随对应功能模块放置或由命令行传入。不要全部写死进源码。

测试不一定单独设tests目录，可作为tools中的自检入口；冻结范围、教师标签mask、checkpoint严格加载等重要验证应保留。当前尚未迁移config/tests，避免破坏路径依赖。

源码盘点见[功能清单](code-inventory.md)和[逐文件初筛](code-role-index.md)。
