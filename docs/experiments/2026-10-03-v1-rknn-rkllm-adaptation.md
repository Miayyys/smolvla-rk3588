# V1 部署适配：RKNN视觉/专家＋RKLLM语言

## 用户选择与边界

用户同意替换 V1 语言 INT16 DFP 例外，要求视觉等使用 RKNN、语言使用 RKLLM。所选训练权重仍是 V1 无教师损失 QAT master，SHA `4aeb92854d2bb89bac84a2d791d2acb4389b934178948c05533d6a52fe0b9f81`。本次不是重新训练，也不是保持原 V1 位宽图：RKLLM 整段语言采用 W8A8；RKNN 视觉和专家沿用[原精度板测](2026-10-03-v1-native-precision-board.md)产物，专家 FP16/BF16 例外不变，词嵌入仍 CPU 逐行 INT8 查表。

从真实 FP QAT master 转换，不对已打包低比特权重再次 PTQ。原始 FP 的独立 PTQ 对照仍以此前实验为准，本次适配不能代替重新比较同格式的完整 QAT/PTQ。

## 分组 W8A8 不可用的实际证据

先尝试 `w8a8_g128`，本地 RKLLM-Toolkit 1.3.1 明确拒绝：`The tensor shape of this model can not be evenly divided into group_size[128]`。SmolVLA 宽度960无法整除128。没有为绕过限制改网络维度或使用假分组量化，改用 SDK 可接受的普通 W8A8。失败日志与报告保存在 `runs/v1_rkllm_w8a8_g128_v1/compile.log`、`language_w8a8_g128.compile.json`；该目录名保留最初尝试，最终成功产物文件名明确为 `language_w8a8.rkllm`。

## 真实多模态校准

`prepare_rkllm_v1_calibration.py` 复用原40个独立校准 episode 的 prefix、attention mask、position IDs，未使用 Spatial/Object task0 的开发回放。按 mask 取有效 token，最后状态 token 保留，position必须严格等于0..n-1。

采用[官方 embedding 校准示例](https://github.com/airockchip/rknn-llm/blob/main/examples/multimodal_model_demo/data/make_input_embeds_for_quantize.py)的 `sample` / `token_nums` JSON 与 pickle kwargs 格式。样本保存 FP32 `inputs_embeds`、4D additive原分块attention mask（可见为0，不可见为float32最小值）、INT64 `position_ids`。不是用普通文字聊天数据代替 VLA 前缀。SDK日志确认加载40个样本并逐层优化。

源40行dataset SHA `ac5097abb584b6b754d4bf8ac0720405ea7d5c94b75b65c9cee07605f59c77d8`；新JSON SHA `ec2a3140c7294076a1399eb71b5d1f1537f696fee31209394f443fc0bea491e3`。每个episode、有效token数量、pickle SHA保留在 `calibration/calibration_report.json`，没有把本轮观测加入校准。

## 实际转换配置与产物

RKLLM-Toolkit1.3.1，load_huggingface(device=cpu,dtype=float32)，target=rk3588，1 NPU core，max_context256，`do_quantization=True`、`quantized_dtype=w8a8`、`quantized_algorithm=normal`、optimization_level1、默认hybrid_rate0；`export_embedding=False`。借用已验证的 full-prefill command形状扩展，新增160-token投影与attention形状，保留原始最终tensor布局；不是任意长度支持声明。

产物 `language_w8a8.rkllm`：**210,701,524 B**，SHA `9e5d80fff09a7be0fcd7c3adbdccd325f95c5a8758366ddee33ce807f54c9ff7`。实际 build/export 成功，原始日志、源语言权重 hash、源 master 身份与编译参数保留在 `w8a8_compile.log`、`language_w8a8.compile.json`。只是编译成功，语言连续调用、prompt-cache格式及整策略质量尚未测量。

| 参数文件 | 实际字节 |
| --- | ---: |
| RKNN W8A8视觉＋连接器 | 111,973,263 |
| RKLLM W8A8语言 | 210,701,524 |
| RKNN专家三分图（保留FP16/BF16例外） | 117,175,409 |
| 所选CPU参数 | 47,546,790 |
| 合计 | **487,396,986** |

对原checkpoint906,712,520 B减少 **46.2457%**。包括RKLLM容器实际保存的参数，不按理想位宽估算；不含分词器/配置/运行库，不代表峰值内存。

## 板端准备与待验证项目

`stage_v1_rkllm_board.py --prepare-only` 已实际完成小代码/manifest准备，板端目录 `/dev/shm/qvla_v1_rkllm_language_v1`。引用已有RKNN图、CPU资产、处理器和native BF16适配库；RKLLM继承版本固定的分块attention patch、worker和工作池WC控制。新W8A8模型的工作池大小与缓存格式尚未验证，不能自动套用FP16的修复成功结论。

最初按既有约定仅准备小文件；随后用户明确“板子的你来就行，服务器和下载大文件我来”，本轮开始代传210.70MB语言模型。`python3 scripts/stage_v1_rkllm_board.py` 支持续传并完整准备。板端root满，使用内存盘，重启需重建。

传输已完成，hash核验通过。量化语言A→B→A与缓存解析已完成，具体修复如下；完整warmup1/repeats3回放和两个同任务筛查正在进行。若缓存格式或执行接口报错先修接口；不能退回全RKNN后称已实现用户要求。若仍0/2，分别诊断W8A8视觉与语言/专家，不能直接引用FP16 RKLLM诊断版2/2。

## 工作池配置与连续调用

首次新W8A8模型运行成功，缓存header/records按原ABI读取成功，但A→B→A重复K/V最大差1.7041016，动作最大差0.0815815。日志中的工作池实际为32,022,528 B，原FP16过滤条件32,432,128 B未匹配，因此cacheable映射未改变。这是新模型首次真实反馈，不将运行成功视为数值成功。

更新filter时另发现传输问题：`rsync --append-verify`会跳过同长度的已有文件，manifest里的两个8位数字替换未被真正传到板端，第二次仍未命中WC。上传脚本对代码/配置/模型改用 `--checksum --partial`，保留rsync块级续传而移除append模式；实际读取板端manifest确认32,022,528后再执行。第二次未生效的运行不能称作WC修复无效。

真正生效日志确认工作池 original=403→actual=405，其他分配未变。A→B→A的prefix、32个K/V及动作最大重复差均**0**；B相对A动作MAE0.2761007（验证不是缓存原结果）。三次完整耗时5965.59 / 5971.08 / 5762.33ms，仅三次混合输入诊断值，不当作p50/p95。退出码0，原始失败报告 `w8a8_before_pool_fix.json`、成功 `native_live_replay.json` 及 `native_live_wc_verified.log` 保留在本轮runs目录。

该修复是RKLLM worker内特定工作池的缓存映射配置，非修改模型精度、非CPU代算；注意力patch/worker/分配库SHA仍与已固定版本一致。语言token条件和完整回放的质量需继续测量。

## 完整回放与两个短任务结果

warmup1/repeats3，完整图像/语言/专家/动作处理，语言确为RKLLM。三次动作完全一致，退出码0、SDK无错误；p50 **5942.153ms**，三次6463.042 / 5716.216 / 5942.153ms。相对同一个所选GPU pack，动作MAE **0.317656893**、最大差2.044516921、夹爪38/50符号不同，明显差于全RKNN原精度版MAE0.141/7个。前处理输入与参考SHA同此前记录，manifest SHA `dbbd47a9cf3d4a36b367366b4693f214cb4f46aa3b49f31d5d5edc2c805d7877`。

原回放工具的scope文字沿用全RKNN旧描述，但计时backend明确为rkllm、实际hash核验和worker日志也证明RKLLM执行；现已修正工具文字。保留原始 `partitioned_replay.json/.npz` 和日志；`full_replay_verified.json`仅补充后端/语言hash元数据，测量数字未改动。父进程ru_maxrss639384KiB不包含独立语言worker，不能和全RKNN单进程RSS直接比较或宣布内存减少。

| 任务 | 原FP | 全RKNN原精度版 | 本次RKNN＋RKLLM W8A8 |
| --- | --- | --- | --- |
| Spatial task0，初态0/seed0 | 成功78步 | 失败280步 | 失败280步 |
| Object task0，初态0/seed0 | 成功125步 | 失败280步 | 失败280步 |
| 成功数 | 2/2 | 0/2 | **0/2** |

本次97.51s，逐任务初始观测和共同query噪声严格配对。完整回放握手精确复现；W8A8语言格式在实际149/141有效token的任务中无执行错误。视频、动作、输入和反馈保存 `short_tasks/`。这版实现了要求的后端拆分、参数压缩超过40%，但质量尚未合格；没有证明比全RKNN质量更好，也不据历史不同日期的少量耗时宣称后端加速。

## 边界替换：RKLLM语言与视觉误差的区分

扩展 `diagnose_v1_partition_frontend.py` 支持RKLLM接口。同一原始观测/噪声执行三次完整计算；只有诊断输入替换，所有板端图实际执行，未改最终模型或用GPU计算替代正式部署。

| 输入条件 | action MAE 对所选GPU pack | 夹爪符号不同 | 组装prefix MAE |
| --- | ---: | ---: | ---: |
| 本次完整native RKNN＋RKLLM | 0.317656893 | 38/50 | 143.612381 |
| GPU视觉特征输入原板端后续流程 | 0.030223209 | 1/50 | 0 |
| GPU完整prefix输入原RKLLM/专家 | 0.030223209 | 1/50 | 143.612381（替换前） |

视觉实际feature MAE6.409431。替换features与替换完整prefix产生相同动作，支持这一个观测的主要错误来自视觉编码，CPU查表/组装没有额外差异。RKLLM W8A8在正确prefix下的动作误差比同条件原精度全RKNN语言/专家链的0.081312更低，但不是任务成功率比较；仍有最大差1.818579与一个夹爪符号差，不能称语言/专家完全无误差。

原始 `frontend_attribution.json/.log`保留。**GPU特征替换不是最终板端方案**，不能把0.0302或其潜在效果写成当前部署成绩。下一步保持用户要求的RKNN视觉＋RKLLM语言拆分，修复视觉原生量化数值，兼顾至少40%实际文件压缩；没有为了这次诊断改成全视觉FP16或重新启动HAQ/QAT。

进一步复核两个短任务首块（50个动作），确认两模型原始输入/噪声完全相同后与原GPU FP直接比较：Spatial位置MAE0.350899、旋转MAE0.031218、夹爪12/50不同；Object分别0.292562、0.019342、13/50不同。旋转相对RMSE1.230/1.482，不能因绝对差小宣布旋转保留良好。详见[持续测试总表第13节](../model-test-results.md#13-当前板端对原fp两个短任务的动作分项)，原始 `short_tasks/original_fp_first_chunk_comparison.json`。当前完整Long闭环未测，不能判断长任务是否更差。
