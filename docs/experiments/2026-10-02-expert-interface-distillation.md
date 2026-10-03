# 完整动作专家与动作接口短对照

用户授权只训练动作专家及接口。从原始FP开始，只开放专家16层112个线性节点及5个接口（state_proj、action_in_proj、action_out_proj、action_time_mlp_in/out）master_weight/bias。专家LayerNorm及其他未替换参数保持冻结。视觉、语言及词嵌入、连接器均冻结。预计可训练99857232参数元素（约22.19%），精确清单以training_scope.json为准。

两组GT-only λ0和GT+教师 λ0.2，各250次优化更新、累积2、恒定lr1e-6、seed29、教师任务采样0.5。均复用隔离训练的5000观察/真实OFT标签和审核mask。与前8层短对照保持其余设置一致，比较范围变化的效果。不是LoRA，不开启fake quant；训练后严格重载完整FP master，再各测10长任务初始状态0，核对状态、双相机、策略噪声和环境seed。

成功指标：原FP成功的1/2/3/4/5/6/9是否保住，失败0/7/8是否改善。离线40开发观察完整chunk MAE仅辅助。该Long10已用于多次诊断，不当新独立测试；其他suite闭环尚未测。不会自动启动长训或QAT。

实现新增expert_and_interface范围，在算子替换后显式requires_grad，优化器仅接收该清单；冻结参数原始字节SHA256训练前后相同才通过。测试覆盖全部专家层与接口开放、语言与norm冻结，AdamW更新后冻结hash不变。另新增FP训练权重overlay重建接口及完整性测试作为磁盘不足的备选，但清理后本轮使用完整FP master保存，未启用overlay。

入口 `.venv/bin/python -u scripts/diagnose_expanded_distillation.py --output runs/distillation_expert_interface_v1 --train-scope expert_and_interface --two-arm-controls`。后台顺序训练/Long10，互斥锁，失败停止。结果待测。产物为同名runs目录pipeline.log、status.json、results.json及各组training_scope.json、完整master、训练和评测日志。

## 完成结果

两组250更新及20回合Long10完成，耗时1500.14秒。GT 6/10，丢失原本成功任务3；KD 5/10，丢失3、6。两组均未新增成功，0、7、8仍失败。真实可训练99857232参数元素，冻结350188944；冻结前后hash完全一致，配对状态/图像/噪声检查通过，采样记录与前8层/广范围对照完全相同。GT master SHA256 `849fbd9e77bad64c4a2aba9c3023acd576e92b9119c19acea997eda202280631`，KD master SHA256 `303e8c3ae6c4dae418d27d77f166e9a13d56e851508ee9e1f2ff835d7379f1eb`。

在当前单初始状态Long10诊断面板下，前8层GT/KD均7/10，完整专家+接口为6/5，更广范围没有改善并造成退步；不能分离后8层与接口各自的训练因果，也不能从这一suite推断其他suite质量。按用户要求下一轮以三个其他suite各3任务补充对照，未新增训练，QAT未启动。

## 用户授权的旧权重清理

为释放空间删除旧合成/两步/早期v2验证产物，以及v4三组退步模型权重；v3滚动AdamW训练状态与两份周期重复快照也删除，v3最终master仍保留用于恢复实验。共释放37.704GiB。只删除.safetensors/.pt/.pth权重文件，全部报告/日志/动作证据保留；原始FP、v1/v2候选、前8层对照及v3最终master未删除。删除精确路径与字节清单在服务器 `runs/obsolete_weights_cleanup_20261002.json`，这些已删除模型后续不能原地重载复测，也不能从已删训练状态续训，历史结果仍可审核。
