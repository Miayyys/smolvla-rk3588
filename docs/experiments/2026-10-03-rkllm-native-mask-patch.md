# 自行修改RKLLM原生注意力：微型与真实语言验证

## 实测结论

自行修改独立runtime副本后，**RK3588实际NPU执行已支持本次SmolVLA的分块注意力**：前缀全双向，前缀不能关注最后state token，state能关注全部输入。微型2层和真实16层语言模块均已板测，原生缓存取K/V，无CPU attention或K/V投影重算。真实模型完整151-token prefill需同时补齐转换端NPU shape配置。

此前“公开API没有可用入口”仍成立，但不能据此判断只能等待官方。本次通过版本绑定的内部实现修改验证了自行解决的可行性。这是实验性runtime补丁，尚非通用可维护的正式部署后端。

## 原理与精确修改

官方ARM64 runtime1.3.1 SHA256 `f25e9b099db08aaacd0a3ac62b4697d3951d6ae61ae41ea09f6702cfa89eb32c`；RK3588 driver0.9.8。以context和mask代码中`causal_attn`断言交叉定位对应字段，未猜测公开参数reserved字节。

1. 地址`0x1e36c8`原指令`strb w3,[x24,#57]`写入内部context注意力类型。单指令改为写零，微型模型从causal实际变为noncausal；对照证明这次runtime修改有效，和此前只修改模型metadata不同。
2. 完整分块版本在同位置跳到隔离code cave，禁用causal，并将ubatch设为内部batch值，允许一次处理整段前缀。补丁不是公共ABI接口。
3. 原生mask生成位置`0x1cdbd8`的标量存储跳到cave：读取query/key索引，当key是最后token、query不是最后token时写负无穷；其他位置保留原值。softmax、Q/K/V和MLP仍沿用原native实现，不执行第二次语言计算。
4. code cave使用原ELF第一RX段之后的已验证零填充区`0x7136c0`/`0x713740`，仅扩展该PT_LOAD的file/memory size至`0x713800`，早于下一文件段`0x714138`。不改系统runtime；通过`LD_LIBRARY_PATH`加载独立副本并检查loader解析结果。

脚本严格限定原文件hash、指令值、ELF布局与空白区，版本变化时直接拒绝。生成补丁的机器码可由`scripts/patch_rkllm_block_mask_probe.py`和配套`.s`/`.ld`复现。原生分块库SHA `cd525c66534987956cf63793a99d574a72051208b524161a805646b0cc96f29b`。

## 微型模型验证

沿用固定2层FP16、8token、hidden256、2KV头、D64探针和原embedding输入。token输入＋embedding回调，GENERATE/1，dump0，保存原生cache。首次修改只禁用causal：第二层K/V对causal参考MAE约0.18684/0.18618，对全双向参考约0.000175/0.000182；最大误差<0.001，证明计算规则实际改变。

随后加入7token前缀＋最后state的分块规则。参考使用相同PyTorch权重、显式二维mask，K0/V0/K1/V1 MAE分别约0.0000914/0.0000920/0.0001702/0.0001764，最大误差0.0009326。INIT/RUN/CACHE均为0。原始记录`block_report.json`、`block_board.log`，不以文件生成代替数值验证。

## 真实16层模型与NPU shape修复

选定V1无教师QAT master SHA `4aeb92854d2bb89bac84a2d791d2acb4389b934178948c05533d6a52fe0b9f81`。全部16层language权重、FP16、3NPU核、context256，真实episode18/task0/frame0；去padding后151有效token，位置0..150。原mask压实的等价性见上一轮记录。

最初沿用普通RKLLM产物时，151token整段执行出现未知shape，**虽然RUN返回0，结果仍判无效**。原产物支持部分prefill大小，但不包含此处weight matmul的M152及attention feature matmul的新形状。两次中间转换分别遗漏了两种feature形状，均有错误且弃用。

最终在Toolkit `get_op_cmd.add_matmul_info`中补充由实际错误日志枚举出的339个op/shape组合；weight matmul新增M152，QK新增[320,64,160]和[160,64,160]，QKV新增[320,160,64]和[160,160,64]，每NPU split各自加入。原权重和原有shape保留。这些是本输入的额外转换配置，不是所有长度的完整覆盖。脚本`scripts/compile_rkllm_full_prefill_probe.py`。

最终实际板运行日志无`E RKNN`/`E rkllm`计算错误，INIT/RUN/CACHE_RC均为0，callback一次返回151token。CPU affinity err22警告仍存在。原生K/V由SDK固定格式解析，32个张量对**原分块规则**FP32参考的MAE均值0.00417753，最大绝对误差0.900731；对同151token causal参考的均值0.0310294。误差均值和最大值都保留，不能称为严格浮点对齐。

单次run含保存缓存1113.438ms，未计初始化/主机读取解析；未做预热重复统计、完整模型峰值RAM或与旧版同协议速度对照，因此不宣布端到端加速。

## 接到真实板端动作专家

将151有效token原生K/V插回177固定长度cache位置，padding位置置零，保留原prefix_pad_masks。padding位置不允许被expert关注；这里仅重排数据，不重算投影。使用同初始噪声和已选`expert_selected_corrected.rknn`，板端连续10步Euler denoising，CPU正常time embedding与动作后处理沿用原部署。

| 比较 | 最终动作MAE | RMSE | 最大绝对误差 | 夹爪符号差异/50 |
| --- | ---: | ---: | ---: | ---: |
| 原生RKLLM cache＋同RKNN专家 vs QAT FP master | 0.0264345 | 0.104645 | 1.864545 | 1 |
| FP master cache＋同RKNN专家 vs QAT FP master | 0.0260707 | 0.106506 | 1.905336 | 1 |
| 上述两个板端结果直接比较，仅cache来源不同 | 0.00732198 | 0.0113715 | 0.0509885 | 0 |

这表明本输入没有因更换语言后端新增夹爪符号变化；仍有其他部署数值误差，不能把这一帧推断为闭环成功率恢复。两次专家执行约1.77s/1.76s，包含10次独立inference，没有完整图像前端与真实逐帧RKLLM调用的联动计时。

GPU服务器此时SSH返回404，动作对照改用已有FP master参考与板端相同专家，两者原数据/初始噪声固定。未训练、未重新校准。检查脚本`scripts/verify_rkllm_cache_board_actions.py`；日志与报告在`runs/rkllm_native_patch_v1/real_compact/`。

## 记录与结论边界

全部原产物/修改库/模型/cache/report hash和大小见`runs/rkllm_native_patch_v1/experiment_manifest.json`。原系统runtime不变。板上旧的普通language FP16副本经SHA核对后被新实验模型替换以节省磁盘，本地旧产物保留。正式HAQ精度图和成本表尚不调整。

已完成：原生attention开关、分块mask、完整真实language一次执行、原生K/V数值和一次真实RKNN专家动作桥接。尚需：更多真实观测/长度覆盖、实时前端与语言/专家串联、完整动作与少量闭环任务质量、预热重复延迟/内存、最终量化和运行资产体积。当前mask要求一个compact前缀＋一个最后state，未支持任意二维mask、跨观测复用缓存或其他runtime版本。

## 后续连续调用复核

[完整流程记录](2026-10-03-rkllm-live-pipeline.md)已接入真实视觉前端与专家，但A→B→A出现后续层K/V和动作漂移；对齐和单核未解决。本文保留单帧已测值，不能据此宣称完整语言替换通过。
