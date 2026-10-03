# V1 原精度图的 RK3588 完整执行与接口修复

## 范围与版本

沿用[转换记录](2026-10-03-v1-native-precision-conversion.md)的六个新图、所选 V1 无教师损失 QAT master、W8A8 视觉和 CPU 参数。六图 SHA 均与 manifest 相符，未重编译、替换权重或改变 V1 模块精度。语言用 RKNN 分图，未用 RKLLM 替代 INT16 DFP 例外。Runtime 2.3.2，驱动 0.9.8，单 NPU core0。

## 实际遇到的接口问题

1. 首次完整回放出现 `input dtype is undefine!`，随后进程 SIGSEGV。逐分区打印张量名字、shape、NumPy dtype，错误来自独立 BF16 gate 的普通输入接口。C 查询逻辑输入得到 type=12（未定义）、size=4294931296；同模型 native input 明确为 BF16，shape=(1,50,720)，size=72000。输出为 BF16，shape=(1,50,2048)。旧微型 BF16 图在当前 runtime 也暴露相同逻辑输入异常，因此旧成功报告不保证当前普通接口适用。
2. 使用 native BF16 输入后，Python faulthandler 进一步将 SIGSEGV 定位到专家后段 Lite inference。前段的 BOOL frontier 被 Lite 返回为 FP32，但原 ONNX 和后段逻辑输入明确要求 BOOL：`/Concat_17_output_0` 与 `/Unsqueeze_16_output_0`。语言后段 `/Unsqueeze_11_output_0` 同样需恢复 BOOL。

原始失败证据：`runs/v1_exact_precision_v1/board_boundary_diagnostic.log`、`projection_interface_query.log`、`expert_after_interface_query.log`、`bf16_native_without_bool_fault.log`。失败不算板端质量评测完成。

## 修复方式与原理

- `rknn_bf16_projection.c` / `smolvla_bf16_projection.py`：只服务单输入单输出 BF16 投影。查询并验证 native dtype、元素数、无 padding 的输入字节数；将 FP32 输入按 BF16 round-to-nearest-even 编码为16位，绑定 native input memory、同步到设备、执行原 BF16 NPU 图，再通过 SDK 输出接口取 FP32 表示。无 FP16 回退，无 CPU 矩阵乘法；输出缓冲区逻辑大小必须匹配。底层输出原生 buffer 有 padding，由 SDK 输出接口处理。
- `split_v1_projection.py` / `build_v1_partition_manifest.py`：保存原始 ONNX input/output dtype，并显式标识独立 gate 的 native BF16 执行后端。
- `smolvla_rknn_partitions.py`：按原始 dtype 恢复 BOOL；转换前要求值严格属于0/1，避免把任意数值静默当作掩码。INT64 索引要求类型保持不变。浮点 frontier 不做额外降精度。
- `stage_v1_partitioned_board.py`：传输适配源代码和已有 RKNN 头文件，在板端用 gcc 编译小共享库；没有新增下载依赖。

这是接口表示修复，不证明 RKNN 量化 scale/rounding/fusion 与 GPU QAT 打包器完全相同。BF16 与 INT16 DFP 的原配置及六图 SHA 保持不变。

## 完整原始观测回放

`verify_v1_partitioned_replay.py --root . --warmup 1 --repeats 3`，含图像处理、视觉编码、词查表、语言全部16层/KV、专家10步去噪及动作后处理。实际35次图执行（视觉2、语言3、专家30）。修复后退出码0，日志无 `E RKNN`；三次输出均有限且完全一致。

| 指标 | 实测 |
| --- | ---: |
| 单块完整 inference p50 | 6066.006 ms |
| 三次计时 | 6585.354 / 6066.006 / 6061.617 ms |
| 与所选 V1 GPU pack 的 action MAE | 0.141102664 |
| 最大动作绝对差 | 1.953972161 |
| 夹爪符号不同 | 7 / 50 |
| 三次重复最大绝对差 | 0 |
| 进程 ru_maxrss | 1,024,508 KiB |
| 七图＋CPU参数 | 448,622,897 B，较原 checkpoint 减少50.5220% |

MAE = mean(abs(board_actions - selected_GPU_pack_actions))，max 同差值取最大；夹爪比较第7维是否大于0。参考是所选 GPU 量化 pack，**不是原 FP**；此处 MAE 不能标成对原 FP 的量化误差。相同 raw input/noise 重复，非校准输入；未把回放观测用于重校准。

输入 SHA `cca351635bc682a7915c708c1f10dcb42f4e9195f882f5002085ba649801a25c`；参考 SHA `946e12c7562a17fc717170ee23ea27061d84fdd75965d9ddf6d3df765d5015ac`；manifest SHA `98bd9b9502a1e863c31319d6bdece75cb82b0236e9301235d8dc5b8395523359`。原始结果 `partitioned_replay.json/.npz`、`typed_native_replay.log` 均位于 `runs/v1_exact_precision_v1`。

maxRSS仅为进程统计，不能称板端总内存峰值；文件字节不含处理器、分词器和共享 runtime。单输入三次回放不代表 p95 或全面质量。完整运行稳定，但动作差仍明显，尚不能认定质量合格。

## 配对短任务

Spatial/Object task0、各初态0/seed0，与原 FP 配对；一致初始化与逐块噪声校验，原始日志、视频、逐块输入/动作在 `runs/v1_exact_precision_v1/short_tasks`。持久服务首先精确复现已保存原始观测回放，handshake通过；运行退出码0，SDK无错误。

| 任务 | 原 FP | 板端 V1 原精度 | 原 FP 步数 | 板端步数 |
| --- | --- | --- | ---: | ---: |
| Spatial task0：黑碗放到盘子 | 成功 | 失败 | 78 | 280 |
| Object task0：alphabet soup 放入篮子 | 成功 | 失败 | 125 | 280 |
| 合计 | **2/2** | **0/2** | | |

配对测试98.77s，开发筛查仅两个episode，不是40任务全面成绩。模拟器在等待板端推理时暂停，因此不代表真实机器人实时控制。完整原精度图已能运行，质量尚未合格，不能将 GPU 14/19 成绩移植到板端。

## 单观测前端误差归因

`diagnose_v1_partition_frontend.py` 做三次同输入完整计算，只在诊断时替换边界输入，所有板端模型仍实际执行；不替换最终部署文件。参考 feature/prefix 均来自同一个所选 GPU pack 的回放 NPZ，未使用教师或测试任务重校准。

| 诊断输入 | 动作MAE | 夹爪符号不同 | 组装prefix MAE |
| --- | ---: | ---: | ---: |
| 当前 native V1 | 0.141102664 | 7/50 | 143.612381 |
| 用 GPU 视觉 features 喂给原板端后续图 | 0.081311951 | 7/50 | 0 |
| 用 GPU 完整 prefix 喂给原板端语言/专家 | 0.081311951 | 7/50 | 143.612381（替换前） |

native视觉 features MAE=6.409431，feature 会按 sqrt(960) 放大后进入prefix。换为 GPU features 后组装 prefix 与 GPU 完全相同，且动作与直接替换完整 prefix 的情况一致。这支持：当前观测的 prefix 错误主要由视觉编码产生，CPU查表/状态投影/组装接口在该观测上没有额外差异。动作MAE减少约42.37%，但夹爪错误不减少，说明视觉之外的语言/专家链路还有误差，不能只修视觉就宣布完成。两阶段误差有非线性交互，不能把MAE差当成严格可加贡献。

原始结果 `frontend_attribution.json/.log`。只是一帧归因，没有测替换输入后的闭环成绩；三个诊断值不能写成最终模型任务提升。下一步优先检查视觉 W8A8 的原生校准、量化边界及融合差异，再对语言/专家做独立边界比较；保持V1位宽，不把全前端升FP16当作本次修复。

## 与历史全 RKNN 的同任务对比

重新读取五次历史/当前 `summary.json`，按 Spatial/Object task0 筛选，共同初始观测 hash 与共同 query 的 noise hash 均一致。完整审计及逐任务步数保存 `historical_same_task_comparison.json`。本次也是全 RKNN，BF16 投影使用原生 RKNN C 接口，没有使用 RKLLM。

| 版本 | Spatial0 | Object0 | 两任务成功数 |
| --- | --- | --- | --- |
| 最初原模型 FP16 全 RKNN | 成功79步 | 成功125步 | 2/2 |
| 旧 V1 native 量化适配 | 失败280步 | 失败280步 | 0/2 |
| V1 FP16视觉＋FP16语言＋量化专家 | 成功79步 | 成功129步 | 2/2 |
| V1 FP16视觉＋KL混合语言＋量化专家 | 失败280步 | 失败280步 | 0/2 |
| 本次 V1 原精度分图 | 失败280步 | 失败280步 | 0/2 |

同一原始观测和所选 GPU pack 参考：旧native动作MAE约0.192，本次0.141；FP16前端诊断0.02973，KL诊断0.08789。动作MAE下降尚未转换成当前两个任务的成功率提升。最初原模型FP16并非所选QAT master；FP16前端诊断才是同QAT master的高精度前端对照。历史与当前编译、精度配置和日期不同，不把耗时差解读为RKNN/RKLLM后端性能因果比较。仍未测本次Goal/Long，不能补写为失败或成功。
