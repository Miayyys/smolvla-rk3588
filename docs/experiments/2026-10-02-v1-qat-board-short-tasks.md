# V1无额外教师损失QAT：RK3588三个短任务闭环筛查

## 目的与固定协议

用户要求检验实际任务效果，不由夹爪符号差异推导失败率。所选模型是蒸馏学生→HAQ v1→无额外教师损失QAT的RKNN适配版本。使用部署manifest中的原INT8视觉、INT8语言（含INT16例外）、专家混合图；不使用FP16视觉诊断候选。详细图/来源hash见[部署记录](2026-10-02-v1-no-teacher-rknn-deployment.md)。

Spatial/Object/Goal的task0各一次，init_state_index0，env_seed0，noise_seed分别100000/200000/300000，chunk50动作、固定高斯噪声生成器。先原GPU FP、后板端所选量化模型，核对原始初始观测及每个共同query的noise SHA。图像/文本/状态前处理与完整网络在板端执行，LIBERO仿真在本地；仿真等待板端推理，不是机器人实时控制或全面成功率测量。

启动前常驻接口必须精确复现已保存的选中模型原始观测回放（动作逐元素相同），否则禁止rollout。复用此前原FP16板端同任务结果，只有初始观测与共同噪声hash再次核对一致时才进行历史配对比较。

| 任务 | 描述 | 历史FP16板端成功 | 步数 |
|---|---|---:|---:|
| libero_spatial/0 | pick up the black bowl between the plate and the ramekin and place it on the plate | True | 79 |
| libero_object/0 | pick up the alphabet soup and place it in the basket | True | 125 |
| libero_goal/0 | open the middle drawer of the cabinet | False | 300 |

## 运行与证据

运行目录 `runs/qat_v1_board_short_tasks_v1`，脚本 `scripts/run_smolvla_board_libero.py` 新增显式board-root/replay-inputs/replay-reference，旧默认路径保留。板端 `serve_smolvla_board_stdio.py` 通过SSH常驻模型，每个新观测返回整chunk动作。保存每次原始输入、动作、噪声hash、时间、视频和结果JSON。三任务共六回合完成，总耗时135.19秒；常驻接口精确回放通过，配对初始观测和噪声验证通过。不能把历史FP16成绩当作本次量化模型成绩。

查看进度：

```fish
tail -f /home/loser/Study/QVLA/runs/qat_v1_board_short_tasks_v1/rollout.log
```

## 实际结果

| 任务 | 原GPU FP | 历史FP16板端 | 本次QAT适配RKNN | 本次板端步数/上限 |
|---|---:|---:|---:|---:|
| libero_spatial/0 | 成功 | 成功 | 失败 | 280/280 |
| libero_object/0 | 成功 | 成功 | 失败 | 280/280 |
| libero_goal/0 | 失败 | 失败 | 失败 | 300/300 |

三个短任务原GPU FP与历史FP16板端均2/3，本次所选RKNN适配模型0/3。三个任务与历史FP16板端的初始观测hash、共同query初始噪声hash全部一致，见 `historical_fp16_pair_audit.json`。说明当前部署候选在这两个原来成功的任务上确实退化，夹爪符号差异不能解释为14%失败率。小样本不代表40任务总成功率；本次未重跑GPU所选QAT pack的同协议闭环，因此不能将原FP→板端的全部退化唯一归因RKNN转换，也不能单独归因视觉/语言/专家或夹爪。需结合已测转换边界误差进一步定位。

完整报告 `runs/qat_v1_board_short_tasks_v1/summary.json`，各任务 `board_<suite>_0/rollout.mp4` 与 `fp_<suite>_0/rollout.mp4`；权重和执行图SHA在握手报告，当前脚本/summary SHA在implementation_hashes.json。仿真等待NPU，不作为实时机器人控制性能声明。
