# 原始 FP16 视觉＋V1 RKLLM/专家验证

## 配置与来源

用户接受体积缩小约35%，要求先试原始视觉权重。视觉与连接器采用原始 `lerobot/smolvla_libero` checkpoint，RKNN FP16（不做整数校准）；语言仍使用所选V1无教师损失QAT master转换的RKLLM W8A8；专家保持V1三分图及FP16/BF16例外，CPU参数不变。未重新训练。

原始checkpoint SHA `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`；V1 master SHA `4aeb92854d2bb89bac84a2d791d2acb4389b934178948c05533d6a52fe0b9f81`。原始/V1视觉＋连接器198个tensor、98,229,504个参数，转FP32后逐tensor严格比较147个不同；因此本实验既改变权重来源也改变执行精度，不能单独归因于位宽。

原始FP视觉ONNX已验证与GPU FP边界一致，SHA `99feded2e64d31791c7469ce2d915bb4e7ee445d974d49f8d1a726bea4337875`；RKNN SHA `f486c5de7f0bc085e9bb1157db761003b9d81d8c2e1e83cd11fb53942b209434`，212,621,173 B。复用板子已有文件，无大文件下载/转换。其余参数与[当前混合后端](2026-10-03-v1-rknn-rkllm-adaptation.md)相同。

核心参数文件588,044,896 B：视觉212,621,173＋语言210,701,524＋专家117,175,409＋CPU47,546,790。对原始906,712,520 B缩小35.1454%，不包含runtime/代码/分词器，不代表RAM占用。

manifest中的requested_assignment是历史V1搜索图，实际视觉/语言由deployment_override覆盖；历史prefix_down3 INT16审核项不适用于RKLLM。此次不是严格原V1精度图。

## 固定输入完整回放

板端 `/dev/shm/qvla_v1_rkllm_original_fp16_vision_v1`，warmup1、测量3次；原始输入、噪声、参考均与前次一致，参考为所选GPU QAT pack。真实两次视觉RKNN、一次语言RKLLM、30次专家RKNN调用。

| 指标 | V1 W8A8视觉 | 原始FP16视觉 |
| --- | ---: | ---: |
| 动作MAE | 0.317656893 | 0.032225054 |
| 最大动作差 | 2.044516921 | 1.798183382 |
| 夹爪符号差异/50 | 38 | 1 |
| 重复最大差 | 0 | 0 |
| 完整推理p50 ms | 5942.153 | 7259.192 |

三次计时7259.192/7658.643/7175.537ms。动作误差明显减小，仍有最大差；不能替代闭环成功率。父进程maxrss847948KiB未包括RKLLM子进程，不能作为整策略峰值内存。

原始数据：`runs/v1_rkllm_original_fp16_vision_v1/` 内 manifest、vision_source_comparison.json、full_replay.log、partitioned_replay.json/.npz。来源导出和编译配置：`runs/smolvla_vision_split_v1/vision_connector.json` 与 `vision_connector_fp16.build.json`。

## 配对短任务

Spatial0/Object0，init_state0、env_seed0、noise_seed100000/200000，每块50动作，与原始GPU FP配对。运行前握手要求完整回放严格复现。闭环完成：原始GPU FP 2/2，板端2/2。Spatial：FP78步、板端78步；Object：FP125步、板端141步。耗时57.625s（含GPU对照/仿真/板端），初始观测与噪声配对核验通过，握手严格一致、SDK未报执行错误。原始结果和视频在 `short_tasks/summary.json`、`progress.json`、各任务目录。仿真等待板端推理，不是实时控制验证。未测试Long或40任务，2/2不足以证明全面非劣。

闭环调用：

```bash
OPENBLAS_NUM_THREADS=1 .venv-haq-local/bin/python -u scripts/run_smolvla_board_libero.py \
  --board-root /dev/shm/qvla_v1_rkllm_original_fp16_vision_v1 \
  --suites libero_spatial libero_object --task-id 0 --seed 0 \
  --replay-inputs runs/smolvla_raw_board_v1/raw_inputs.npz \
  --replay-reference runs/v1_rkllm_original_fp16_vision_v1/raw_replay_reference.npz \
  --output runs/v1_rkllm_original_fp16_vision_v1/short_tasks
```

首次握手误传包含3次重复的 `partitioned_replay.npz`，因参考shape不符在任务启动前退出；改为提取第一个重复 `[1,50,7]` 到 `raw_replay_reference.npz`，没有修改动作内容或放宽一致性要求。

## 扩展筛查（同日，预先固定任务）

为覆盖四种任务类型，追加各suite的task_id1：Spatial1、Object1、Goal1、Long1。统一init_state0/env_seed0，noise_seed100001/200001/300001/400001，模型与前次不变；不按结果筛选任务。与已完成Spatial0/Object0合计6个不同任务，每任务一个回合。命令沿用上面的参数，改为 `--suites libero_spatial libero_object libero_goal libero_10 --task-id 1 --output runs/v1_rkllm_original_fp16_vision_v1/expanded_tasks`。日志 `expanded_tasks.log`，逐任务视频、动作、原始输入和配对报告在 `expanded_tasks/`。扩展结果见下方；仍不覆盖40任务或多seed统计。


扩展首轮Spatial1的GPU对照成功109步，但板端在首个query前因Python manifest仅白名单141/149/151有效token而拒绝147，未执行该任务，不记为失败。底层worker固定160-token编译/执行，实际token数动态设置mask并恢复177位置K/V；worker允许2..160。先用预定四条实际指令验证新增长度和跨指令A→B→…→A重复性，再将验证通过的长度加入manifest，未重新编译/修改量化参数，不裁短指令。验证脚本 `scripts/verify_smolvla_rkllm_instructions.py`，报告 `instruction_length_validation.json`。这是执行接口验证，不证明新长度的语言数值与GPU一致，后续配对闭环仍必做。


### 扩展结果与边界

四条指令（147/141/136/142有效token）完整执行与跨指令重复性验证通过，前后参考prefix/KV/动作最大差均0；manifest仅添加136/142/147，仍保留未验证长度的限制。保存更改前manifest为 `fixed_replay_manifest.json`，原固定回放报告SHA对应它，新增闭环采用更新后的manifest。

| 任务 | 原GPU FP | 当前板端 |
| --- | --- | --- |
| spatial0 | 成功，78步 | 成功，78步 |
| object0 | 成功，125步 | 成功，141步 |
| spatial1 | 成功，109步 | 成功，117步 |
| object1 | 失败，280步 | 成功，126步 |
| goal1 | 成功，91步 | 成功，92步 |
| Long1 | 成功，227步 | 成功，238步 |

合计原GPU FP **5/6**、板端 **6/6**：这6个单回合无成功任务回归，Object1出现一个改善案例。扩展4对回合耗时134.642s。Long1为把cream cheese和butter放入basket的组合任务：227→238步，证明这一个长任务可以完成，不代表完整Long10通过率。

所有6个首动作块的原始图像/state/指令/noise严格一致，原FP为参考，连续6维动作MAE范围0.025008–0.039629；夹爪每50步符号差异分别0/0/0/1/0/0。与固定回放比较GPU pack的0.0322参考对象不同，不混算。后续轨迹不同，不把后续动作差异当同观测量化误差。

18次板端闭环query推理p50 **7349.461ms**、p95 **7786.377ms**，覆盖真实图像预处理/视觉/语言/专家/动作处理，不含RPC和仿真，含每任务首query；不替代固定输入warmup1/repeats3的7.259s。

汇总 `combined_screening.json` 保留源summary SHA及图文件hash；`scripts/summarize_smolvla_board_screening.py` 检查前后模型身份、无任务重复、配对首输入再统计。各任务有视频/全部原始query输入/动作。扩展首轮接口拒绝单独保留 `expanded_tasks_token_guard_attempt.log`，未计入任务成功率。

这是一组预固定任务、一个seed/initial state的工程筛查。没有多seed、完整40任务、整策略峰值内存、功耗或实时控制验证，不能据6/6宣称普遍优于FP。
