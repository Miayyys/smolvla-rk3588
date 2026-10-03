# 扩展蒸馏训练控制代码

## 本次范围

用户要求先改代码，启动完整QAT前汇报。只开发和短诊断，不执行5000观察教师标注、2000更新蒸馏或完整QAT。

## 技术与配置

- 教师观察按轨迹交错、任务时间三等分采样，唯一 `(episode,frame)`，至少有8个有效动作。扩展计划每长任务500观察，共5000；训练/开发/校准/冻结测试按现有hash隔离。
- 教师NPZ图像/状态只解压一次，不能在每个样本重复解压全部5000张图。标注保存partial动作和进度，`--resume`核对输入manifest、checkpoint root SHA、worker/source diff及fork，跳过已完成样本。
- `review_teacher_labels.py`按有限值、模拟器连续动作[-1,1]和夹爪符号筛选，另接受具体row/动作步/理由的人工排除。不依据教师任务失败把该任务全部删掉，不以GT MAE阈值认定教师语义错误。该过滤仅检查契约，不能证明任务动作正确。
- 学生教师监督使用明确的accepted timestep mask：被排除位置保留GT动作作为flow上下文，关闭对应教师loss；整条教师观察被拒绝时回退GT。保留其余suite监督。
- AdamW/clip1，支持梯度累积；配置样例FP2000更新、累积2、长任务教师池50%/其余GT池50%，λ0.2。QAT样例仅提出1000更新、lr1e-7，未运行/未优化。
- 线性warmup前5%更新，从最低lr1e-7升到峰值1e-6；剩余更新余弦衰减回1e-7：`lr=min+0.5*(peak-min)*(1+cos(pi*progress))`。`steps`是optimizer更新次数，gradient_accumulation是每次更新的微批次数。默认constant/accum1保持历史诊断入口。
- 每250更新在40隔离开发观察上固定噪声，评价完整50步有效动作块（padding剔除）、连续6维、夹爪、长10/其余30；保存预测/GT/mask。评价恢复随机状态和train/eval模式，不改变下一次训练采样。离线代理不是最终选模标准；最终用完整长任务和其他suite回归。
- 每250更新保存模型、AdamW动量、采样/全局Python/NumPy/CPU/CUDA随机状态、lr进度与历史。临时文件写完后替换，恢复前核对SHA和训练身份；不同源模型/图/标签/学习率/预算拒绝恢复。保留最近两个FP master快照供闭环选模，不按首步误差自动选最佳。
- 扩展入口要求review文件且十长任务各至少64个接受观察，防止把旧10帧短标签当完整蒸馏。计划500观察并不代表全部已审核或必须全用。

配置 `config/distillation_expanded_v3.json`，入口 `scripts/run_expanded_distillation.py --stage prepare|label|review|audit|fp|qat`，每次只执行一个阶段。`--dry-run`展示准确命令。FP启动前需完整契约audit；QAT需明确选FP run，不自动连跑；RKNN全混合图尚未对齐时仅允许显式本地诊断。

## 测试证据与边界

本地7项实质测试通过：教师泄漏/合成标签拒绝、归一化及padding、同flow噪声/时间反向传播、lr端点单调性、AdamW及随机恢复后下一次更新逐元素一致、取帧阶段/轨迹/唯一性、逐步review及GT回退。命令：`.venv-haq-local/bin/python -m unittest discover -s tests -p test_distillation_contract.py -v` 与 `test_training_control.py`。

服务器第一次两更新FP测试在初始开发评价发现 `target/targets` 变量名错误，发生在任何更新前；已修复并保留失败日志 `runs/distill_controls_fp2_v1.log`。重跑 `runs/distill_controls_fp2_v2` 验证真实模型累积2、每更新评价/保存、最终严格FP重载；真实FP测试已完成：累积2、2次更新/4次教师监督微批，40开发观察初始及每更新评价通过，lr为[1e-6,1e-7]，最终严格master重载动作MAE/max=0，总耗时127.342s。完整训练状态及SHA已保存。不能把CPU恢复测试说成完整GPU训练恢复已经验证。

**当前不具备完成正式硬件感知QAT的全部证据**：扩展真实标签/质量筛选和FP长任务提升尚未完成，任意v2完整RKNN混合策略数值转换仍未通过。代码可以准备扩展FP训练，但正式QAT与部署完成不能由代码检查代替。

### 真实QAT新控制验证

随后以该两步FP master初始化冻结v2，运行累积2×2更新及每更新40开发观察，lr[1e-7,1e-8]，4个微批使用真实教师；实际本地pack为514,073,656 B（缩小43.30%），fake quant/严格真实pack重载完整动作MAE/max=0，总耗时175.192s，CUDA峰值15,842,855,424 B。没有做这两个短诊断的闭环成功率，不称其新模型质量提升。两次预算只有2更新，warmup取整为0；warmup端点与单调性由100更新CPU配置测试验证。

原始报告本地 `runs/expanded_distillation_code_evidence_v1/{report.json,training_state.json,qat2/report.json}`，服务器 `runs/distill_controls_fp2_v2`、`runs/distill_controls_qat2_v1`。语法检查及7项测试通过。教师标注resume尚未进行真实中断续跑测试；实现了身份检查和partial保存，不能把代码存在说成已测效果。正式训练还需要扩展标签、审核和FP质量验证。

分阶段使用说明见[使用说明](../expanded-distillation-usage.md)。新增磁盘预检根据模型/AdamW动量、临时替换及快照计算所需空间，不足时提前停止，避免训练到保存时才失败；这项守卫在GPU小测试后加入，不改变已测训练数学路径。
