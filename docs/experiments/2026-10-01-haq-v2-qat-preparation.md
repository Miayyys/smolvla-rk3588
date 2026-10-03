# v1/v2 固定与 v2 全配置 QAT 准备

## 用户选择与当前边界

用户将旧200轮候选命名v1，将加强探索200轮候选命名v2；先对v2做QAT，v1保留对照。精度图不在训练中重新搜索。由于任意整策略RKNN转换反馈未接入，本轮为显式允许的训练链路诊断，不称正式硬件感知QAT完成。

`scripts/freeze_haq_candidates.py`复制权重、assignment、候选报告、space、calibration、表快照到Git忽略的`artifacts/candidates/v1`及`v2`，逐文件核对SHA，不允许不同内容覆盖已固定候选；可追踪配置在`config/haq_candidate_v1.json`、`haq_candidate_v2.json`。

| 名称 | 权重B | SHA256 |
| --- | ---: | --- |
| v1 | 504117472 | 9bb703ce1aef964c5c334bbd9b4d754a0e5744137f7d7817b8da3263e123a310 |
| v2 | 513906600 | a033f8f22535c896d14ae2facc3d5fc3dff565a762570196233b577c2bdea3aa |

v2包含292个W8A8、3个FP16、3个BF16、5个INT16非对称及1个CPU INT8 embedding。各模块完整配置/来源/校准hash见公开JSON和原始manifest。

## 技术与实现

`qvla_haq/qat.py`将全部304个可选模块替换成可微QAT参考算子，保留FP32 master；仅配置模块的权重/偏置训练，其余参数冻结。视觉、connector、prefix、expert、embedding均检查实际梯度。整数权重沿输出通道对称量化，qmax=2^(b-1)-1，scale=max(abs(W_channel))/qmax；INT16动态定点如存在候选使用tensor级2的幂scale。激活沿用隔离40校准episode的静态仿射min/max；使用与打包相同的显式舍入、截断及STE，保留范围内梯度。FP16/BF16候选在算子内部cast后计算并恢复外部dtype；embedding为INT8逐行fake quant。配置与本地RL数值参考对齐，不保证RKNN内部INT16/Conv量化器等价。

`scripts/qat_train_haq.py`从原始FP checkpoint开始，不训练已经打包的候选。保留v2精度图和校准范围。数据按40任务轮换、任务内随机episode/帧，50步action chunk，SmolVLA原生flow-matching监督损失，初始lr1e-5、batch1、seed29、AdamW、梯度裁剪1。本轮优先做2步全链探针，不把2步当成训练收敛。

训练排除40个开发episode和校准/测试，核对split/partition/source SHA，保存实际取样episode/frame。产物包括可重载FP master和本地真实整数打包；保存后严格重载并运行完整训练forward核对有限输出。冻结未配置tensor保留原BF16存储以免无意放大文件。两个产物均非RKNN模型，真实部署转换/压缩与闭环质量待测。训练中的W8A8 Linear使用真实CUDA整数GEMM前向，浮点fake matmul提供STE反向；CPU权重舍入按master版本缓存，每次optimizer更新后重新生成，以避免CPU/CUDA舍入边界差异。FP32 master与外部原始BF16/F32 dispatch dtype分开存储。

## 已完成验证

本地11种算子/格式组合通过有限且非零权重梯度检查：Linear五格式、Conv五格式、embedding INT8。除需要CUDA整数GEMM的Linear W8A8外，其他10种fake forward与既有本地打包参考在同一权重/输入上通过atol2e-6、rtol0对照。未用这些单算子检查代替整模型训练/部署结果。

GPUServer连接恢复，A10 23028MiB，PyTorch `2.7.0a0+7c8ec84dab.nv25.03`、Transformers5.5.4、CUDA可用；学生权重/processor与数据已存在，数据位于`/root/qvla/data/libero`。教师OpenVLA-OFT LIBERO-10文件已见远端目录，其真实文件完整性检查另保存原始报告；没有声称教师推理已跑通。

## 蒸馏准备边界

本轮训练代码没有教师损失，不能称蒸馏QAT。教师README标明此checkpoint针对LIBERO-Long/10，不能直接当作四suite通用强教师；其OFT action head/proprio projector需配套官方推理接口，并核对7D动作、归一化、夹爪符号、双相机及action chunk时间对齐。先验证教师在对应任务上的动作/成功率，才生成隔离训练标签并加入教师损失。用户要求先用v2 QAT，故此处仅完成链路准备和诊断，待教师接口验证后做蒸馏与教师损失QAT。

## 服务器两步全链探针及修复记录

初版v1完成两步与严格重载，但未检查同噪声动作，不能作为数值通过证据。加入完整动作检查后v2失败：MAE0.05552869、最大0.2666735。v4保留原dtype并调整整数前向后仍失败：MAE0.03193065、最大0.1371336。诊断发现原始prefix部分模块为BF16：FP32 master重载进原BF16参数会截断更新，外部权重dtype也影响SmolVLA dispatch。逐算子对照还发现CPU打包和GPU舍入在边界附近不同。联合修复master存储/外部dtype、精确整数前向、CPU舍入缓存后，v5全链通过；不能把一次联合修复的收益单独归因给其中某项。

最终服务器`runs/qat_haq_v2_smoke_v5/`：304/304逐算子同输入前向MAE与最大误差均0；相同训练观测及固定噪声的完整50步动作，训练态与真实整数打包严格重载产物MAE/最大误差均0（预设门槛分别0.0001/0.002）。这属于训练观测上的产物一致性检查，不是留出质量验证。

| 指标 | 实际值 |
| --- | ---: |
| 训练步数 | 2 |
| 配置模块 | 304 |
| 可训练参数 | 401857104 |
| 训练episode池（排除开发） | 1147 |
| CUDA峰值分配B | 9936812544 |
| 本地整数打包B | 514073656 |
| 原文件体积减少 | 43.3036% |
| 总耗时s | 37.0411 |
| 教师损失 | 未使用 |
| RKNN完整配置转换 | 未验证 |
| 本轮闭环质量 | 未测量 |

训练loss两步分别0.02772476、0.03894386，来自不同帧与随机flow time/noise，不能据此判断效果改善或退化。重载后的随机训练forward loss0.11981525只作有限输出检查，同样不能与训练loss直接比较。vision/connector/prefix/expert/embedding的首步梯度L1分别4079.19/382.46/700.25/1966.30/6.19。

master SHA256 `26887381e32e3acd05c8d43579ebdcbfa5e257f384323457b15aedc7b60926dc`；整数打包SHA256 `2bfbe9227cbb8fb3588b28f87b9667d77186369e61613b40d11c43b49af592de`。产物在服务器运行目录；小型原始报告、动作NPZ及失败记录同步到`runs/qat_haq_v2_server_evidence/`。

修复后本地22种格式/外部dtype组合梯度通过，20种CPU可执行组合与原数值参考对照通过；缓存随master版本更新的检查通过。服务器完整304算子/动作检查覆盖CUDA Linear W8A8。

## 教师文件与环境确认

远端25个文件、15939159245B，全部SHA256与本地一致，原报告`teacher_upload_check_sha256.json`。模型git revision `95220f9a3421a7ff12d4218e73d09ade830fa9a3`。文件上传完整不等于教师推理通过。

服务器缺少prismatic/timm/peft/sentencepiece；现有Transformers5.5.4不能直接作为OFT已验证环境。[官方依赖配置](https://github.com/moojink/openvla-oft/blob/main/pyproject.toml)指定timm0.9.10、peft0.11.1、sentencepiece0.1.99、tokenizers0.19.1及用于双向并行解码的专用Transformers fork，并将官方torch参考环境固定为2.2.0。后续应独立准备教师环境，避免覆盖学生环境；是否复用服务器现有torch需单独验证，不将安装假设当成已通过结果。checkpoint README旧代码地址返回404，当前官方源码入口为moojink/openvla-oft。

当前结论：v1/v2已保留，v2全图训练/打包/重载诊断可用；尚未启动正式蒸馏或教师损失QAT。下一步完成教师推理接口、动作格式与时间对齐，在其LIBERO-10适用范围验证后生成隔离标签并训练FP学生，再按v2图QAT。原始FP独立PTQ仍保留为必做对照。

`scripts/qat_train_haq.py` SHA256 `0e74df3a45bd4867e52c794942e5b3574a126cb836a8481a725a631c622e8bc4`。

`qvla_haq/qat.py` SHA256 `ba624d434113ab8886c2230b609e0a5ca011ebd31038d3e88738362d1799dc46`。

`scripts/freeze_haq_candidates.py` SHA256 `d0e082bb3e06f60e8fb33825eca8e9d74bf0b9df860e79944fc86cc390c0861f`。

`config/haq_candidate_v2.json` SHA256 `5991c7b17ac5ab1235318890a124f7127eade5850322e84c9451384d08a72ac4`。

`config/distill_qat_v2_plan.json` SHA256 `c66ca557094448fcdfc37b71a251f29527f0435bca01cc12a54d0ba389d332ce`。
