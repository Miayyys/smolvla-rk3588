<a id="final-comparison"></a>

# 终版模型与原始模型全面对比

运行状态：**completed**；当前阶段：`pipeline`。

## 1. 模型与比较口径

| 项目 | 原始模型 | 终版模型 |
| --- | --- | --- |
| 成功率执行设备 | 本地GPU，原始checkpoint | RK3588，真实整策略 |
| 速度/占用执行设备 | RK3588，原始checkpoint FP16适配 | 同一RK3588，终版真实整策略 |
| 视觉＋连接器 | 原始权重；板端RKNN FP16 | 原始权重，RKNN FP16 |
| 语言 | GPU原始浮点；板端RKNN FP16 | V1无教师损失QAT权重，RKLLM W8A8 |
| 动作专家 | 原始权重；板端RKNN FP16 | V1无教师损失QAT权重，RKNN混合INT8/FP16/BF16 |
| CPU参数 | 板端浮点参数 | 当前INT8嵌入及既有接口参数 |
| 板端核心参数文件 | 945,564,365 B | 588,044,896 B |
| 原始checkpoint文件 | 906,712,520 B | — |

终版对原始checkpoint缩小35.15%；对原始板端部署文件缩小37.81%。两者存储口径不同。文件体积不等于运行内存，不包含runtime/代码/分词器。终版是完整部署组合，不是严格保留原V1 HAQ精度图：视觉回到原checkpoint FP16、语言原INT16例外改为RKLLM W8A8。

原始checkpoint SHA `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`；V1 master SHA `4aeb92854d2bb89bac84a2d791d2acb4389b934178948c05533d6a52fe0b9f81`。图SHA固定于本轮 `final_deployment_manifest.json`、两份资源报告及质量握手。板端终版 `/dev/shm/qvla_v1_rkllm_original_fp16_vision_v1`，基线基于 `/root/qvla_board_test/smolvla_vision_v1` 的隔离代码目录。内存盘重启会丢失模型链接/新产物。

## 2. 固定测试协议

- 成功率：Spatial/Object/Goal/Long各10任务，共40任务；每任务initial state0、env_seed0，一个回合。原始GPU FP重新按同一脚本执行，与终版配对，避免套用历史31/40或32/40的不同运行结果。
- 噪声seed：100000 × (suite序号+1) + task_id；每块50动作，10步Euler；初始图像/state和共同query噪声严格核验；每次运行保留输入、动作、视频与结果。
- 速度：同一原始图像/state/指令/noise，完整预处理＋2次视觉＋语言KV＋10步专家＋后处理；warmup3、测量20次；同板顺序执行，不与任务测试并行。P50/P95不含SSH/RPC、仿真、模型初始化。
- 内存/利用率：0.5秒采样主进程及所有子进程，包括RKLLM worker；PSS作为主要进程树内存值，RSS同时记录（共享页会重复）；统计初始化及推理峰值。CPU百分比100%表示一个核，NPU为全局各核占用采样。
- 记录温度、NPU/CPU频率、MemAvailable、运行身份与原始样本。未强制固定频率，频率/温度差异从原始采样审查。未映射到进程的驱动DMA内存不包含在PSS中，0.5秒采样可能漏掉瞬间峰值。
- 40条指令涉及15种有效长度134..151，均在固定160-token RKLLM图完成执行与跨指令重复验证；只扩展验证过的白名单，无截短指令/更改权重。

## 3. 速度与资源（同板）

| 指标 | 原始板端FP16 | 终版板端 |
| --- | ---: | ---: |
| 推理P50 ms | 7519.918 | 7262.644 |
| 推理P95 ms | 7634.695 | 7579.030 |
| 推理均值 ms | 7522.336 | 7302.086 |
| 预处理P50 ms | 134.422 | 126.749 |
| 模型初始化 s | 12.462 | 2.406 |
| 初始化＋推理采样峰值PSS MiB | 1824.404 | 1129.911 |
| 推理阶段采样峰值PSS MiB | 1824.404 | 1129.899 |
| 进程树采样峰值RSS MiB | 1826.551 | 1135.996 |
| 最大进程数（含worker） | 1.000 | 2.000 |
| CPU均值%，100%=1核 | 23.780 | 33.103 |
| 相同输入动作重复最大差 | 0.000 | 0.000 |

同输入P50速度比：原始/终版 **1.035×**；进程树采样峰值PSS相对变化：终版降低 **38.07%**。这是完整后端组合比较，同时包含RKNN/RKLLM、分图和精度差异，不单独归因于量化。


原始模型NPU各核采样平均占用：[81.62589928057554, 0.0, 0.0]%；PSS采样完整：True。

终版模型NPU各核采样平均占用：[72.92424242424242, 0.0, 0.0]%；PSS采样完整：True。

## 4. 任务成功率（GPU原模型 vs 板端终版）

| 类型 | 已完成配对任务 | 原GPU FP成功 | 终版成功 |
| --- | ---: | ---: | ---: |
| libero_spatial | 10/10 | 8/10 | 5/10 |
| libero_object | 10/10 | 8/10 | 8/10 |
| libero_goal | 10/10 | 7/10 | 7/10 |
| libero_10 | 10/10 | 7/10 | 5/10 |
| 总计 | 40/40 | 30/40 | 25/40 |

**成功数持平的两类：Object 8/10→8/10（80%）；Goal 7/10→7/10（70%）。** Object改善task1/5、回归task2/3；Goal改善task0/3、回归task4/5。类别总数持平，不是所有具体任务相同，也不是多seed统计非劣。

已完成配对中的改善：['libero_object:1', 'libero_object:5', 'libero_goal:0', 'libero_goal:3', 'libero_10:6', 'libero_10:7']；回归：['libero_spatial:2', 'libero_spatial:7', 'libero_spatial:9', 'libero_object:2', 'libero_object:3', 'libero_goal:4', 'libero_goal:5', 'libero_10:2', 'libero_10:4', 'libero_10:8', 'libero_10:9']。

| 任务 | 原GPU FP | 终版板端 |
| --- | --- | --- |
| libero_spatial:0 | 成功，78步 | 成功，78步 |
| libero_spatial:1 | 成功，109步 | 成功，117步 |
| libero_spatial:2 | 成功，110步 | 失败，280步 |
| libero_spatial:3 | 成功，87步 | 成功，86步 |
| libero_spatial:4 | 失败，280步 | 失败，280步 |
| libero_spatial:5 | 失败，280步 | 失败，280步 |
| libero_spatial:6 | 成功，105步 | 成功，111步 |
| libero_spatial:7 | 成功，125步 | 失败，280步 |
| libero_spatial:8 | 成功，95步 | 成功，97步 |
| libero_spatial:9 | 成功，119步 | 失败，280步 |
| libero_object:0 | 成功，125步 | 成功，141步 |
| libero_object:1 | 失败，280步 | 成功，126步 |
| libero_object:2 | 成功，114步 | 失败，280步 |
| libero_object:3 | 成功，131步 | 失败，280步 |
| libero_object:4 | 成功，138步 | 成功，145步 |
| libero_object:5 | 失败，280步 | 成功，127步 |
| libero_object:6 | 成功，150步 | 成功，171步 |
| libero_object:7 | 成功，127步 | 成功，134步 |
| libero_object:8 | 成功，140步 | 成功，151步 |
| libero_object:9 | 成功，187步 | 成功，144步 |
| libero_goal:0 | 失败，300步 | 成功，257步 |
| libero_goal:1 | 成功，91步 | 成功，92步 |
| libero_goal:2 | 成功，88步 | 成功，93步 |
| libero_goal:3 | 失败，300步 | 成功，265步 |
| libero_goal:4 | 成功，91步 | 失败，300步 |
| libero_goal:5 | 成功，146步 | 失败，300步 |
| libero_goal:6 | 失败，300步 | 失败，300步 |
| libero_goal:7 | 成功，77步 | 成功，84步 |
| libero_goal:8 | 成功，79步 | 成功，79步 |
| libero_goal:9 | 成功，145步 | 成功，127步 |
| libero_10:0 | 失败，520步 | 失败，520步 |
| libero_10:1 | 成功，227步 | 成功，238步 |
| libero_10:2 | 成功，230步 | 失败，520步 |
| libero_10:3 | 成功，240步 | 成功，277步 |
| libero_10:4 | 成功，231步 | 失败，520步 |
| libero_10:5 | 成功，171步 | 成功，171步 |
| libero_10:6 | 失败，520步 | 成功，322步 |
| libero_10:7 | 失败，520步 | 成功，258步 |
| libero_10:8 | 成功，383步 | 失败，520步 |
| libero_10:9 | 成功，259步 | 失败，520步 |

## 5. 原始数据与复现

`runs/final_model_comparison_v1/`：`pipeline.log`、`progress.json`、`instructions40.json`、`length_inventory40.json`、`instruction_length_validation.json`、`final_deployment_manifest.json`；两版 `benchmark.json`/`resource_samples.jsonl`/固定输入动作及 `quality40/` 下每任务输入、动作、视频、结果。

入口：`qvla/evaluation/run_final_model_comparison.py`；资源测量：`qvla/evaluation/benchmark_smolvla_board_resources.py`；质量评测：`qvla/evaluation/run_smolvla_board_libero.py`；本文由 `qvla/evaluation/report_final_model_comparison.py` 根据已测产物更新，未完成项不得填作结果。

## 6. 资源采样修正

首次采样通过 `/proc/PID/task/TID/children` 枚举子进程；板子内核未提供此接口，导致终版只记录主进程。原报告861MiB/降低52.8%已撤销，保存在 `resource_sampling_v1_missing_worker/` 作为无效记录。修正为扫描 `/proc/*/status` 的PPid递归枚举，加入RKLLM worker必须被采到的检查，再对终版warmup3/repeats20补测。原始全RKNN没有子进程，原始单进程PSS保留有效。最终有效终版报告标记 `process_tree_method=proc_status_PPid_scan`，修正日志 `final_board_resource_corrected.log`。

## 7. 结论边界

这是固定40任务各1回合的完整任务覆盖，不是多seed非劣统计。任务属于已有项目开发评测集，部分任务参与过模型选择；不能写成独立、从未使用的测试集成绩。仿真暂停等待板端推理，不证明实时机器人控制频率。功耗、所有驱动DMA、长期稳定性未测。全面测试完成后据实总结；速度改善不能抵消质量回归。


## 8. 成功率测试复核

模型/worker/runtime/CPU参数hash匹配；固定测评脚本和checkpoint未变；40任务/80回合无重复，seed/初始状态/噪声及动作实际执行检查通过，未发现SDK/RPC报错。此前6任务的两模型完整动作轨迹与本次40任务逐项完全相同。

40任务真实初始观测的checkpoint处理器与板端NumPy处理复核通过：token IDs/masks逐元素一致，state误差0，图像最大差2.384e-7，动作后处理最大差2.980e-8（浮点舍入级）。

| 冷启动复测 | 原GPU FP | 终版板端 | 与40任务原轨迹一致 |
| --- | --- | --- | --- |
| libero_spatial:2 | 成功，110步 | 失败，280步 | 逐元素完全一致 |
| libero_goal:4 | 成功，91步 | 失败，300步 | 逐元素完全一致 |
| libero_10:2 | 成功，230步 | 失败，520步 | 逐元素完全一致 |

这3个回归冷启动后稳定复现，没有发现把30/40误统计成25/40的测评错误，也不支持用长时间运行状态漂移解释这3个失败。尚未逐层验证各任务的RKLLM K/V和RKNN输出与GPU参考，不能排除确定性的部署数值/缓存语义问题，亦不能直接把降分只归因于位宽。

完整复核见[实验记录](project-route.md#record-experiments-2026-10-03-final-evaluation-audit)，原始数据 `evaluation_audit/`。

<!-- HISTORICAL_RESULTS -->

<a id="historical-results"></a>

<details>
<summary>历史模型对照与任务明细（不同协议分开记录）</summary>

# 模型测试结果总表

更新：2026-10-03。本页作为持续维护的跨实验汇总，涵盖原FP、四个蒸馏QAT候选及所选V1的RK3588部署版本。每次新增已完成测试在此补充；技术原理、配置和失败排查仍保留在 `docs/experiments/`。

**最新板端结果：RKNN视觉/专家＋RKLLM W8A8语言，参数487.40MB（缩小46.25%），动作块p50 5.942s，两个短任务0/2。** 上一版保留V1原精度的全RKNN版本p50 6.066s、同任务0/2。详见第11、12节。

第1—10节为GPU原FP与四QAT的既有记录；第11节起为真实RK3588部署。两种平台、不同任务面板和误差参考分别统计，不合并成一个成功率，不将GPU成绩移植到板端。

## 1. 比较的模型

| 简称 | 模型起点 | 量化配置 | QAT额外教师损失 | QAT更新范围 |
|---|---|---|---|---|
| FP | 原始 `lerobot/smolvla_libero` | 无新增量化 | 不做QAT | 无训练 |
| V1＋教师 | 前8层KD蒸馏FP学生 | HAQ v1 | λ=0.2 | 全图304个配置节点的master/bias |
| V1无教师 | 同一蒸馏FP学生 | HAQ v1 | λ=0，仅GT示范损失 | 同上 |
| V2＋教师 | 同一蒸馏FP学生 | HAQ v2 | λ=0.2 | 同上 |
| V2无教师 | 同一蒸馏FP学生 | HAQ v2 | λ=0，仅GT示范损失 | 同上 |

四个候选都经过蒸馏。“无教师”仅指QAT阶段没有额外教师损失。蒸馏时只训练前8层，不代表QAT只量化或只训练前8层。原checkpoint为BF16/F32混合存储，不称全FP32模型。

## 2. 已完成测试列表

- [x] **第一轮闭环任务测试：每模型19回合。** LIBERO-Long全部10个任务；Spatial/Object/Goal各任务0、4、8，共9个。初始状态index0、env seed0、每任务1回合。策略噪声固定为 `100000×(suite编号+1)+任务ID`，suite编号按Spatial/Object/Goal/Long为0/1/2/3。
- [x] **第二轮新状态闭环测试：每模型12回合。** 四suite各任务1、2、5，初始状态index1、seed2、每任务1回合。策略噪声由实际audit核对；各模型初始状态、双相机画面及噪声逐任务配对。
- [x] **共同成功任务的完成步数：第二轮。** 每个实际仿真动作保存到trace；仅FP与候选均成功时比较完成步数，避免把提前失败当成更高效率。
- [x] **完整动作块推理耗时：第二轮。** A10上CUDA同步墙钟计时，排除各任务前2块预热；不把缓存动作取出或仿真时间当模型推理时间。
- [x] **GPU峰值显存：第二轮。** policy加载后重置峰值计数，记录CUDA allocated/reserved；不是RK3588系统RAM。
- [x] **真实量化产物体积：四QAT。** 实际pack文件字节与原FP文件906,712,520 B比较。
- [x] **量化产物严格重载及动作一致性：四QAT。** 检查pack/hash/源量化图，严格load，再比较同观察和噪声的完整动作块；与训练量化前向MAE/max均0，不表示与原FP误差为0。
- [x] **离线动作开发评价：四QAT。** 250/500/750/1000更新时各评价40条开发观察，比较完整有效动作块与GT示范的MAE、连续6维MAE、夹爪符号差异，保留逐任务结果。
- [x] **训练范围与来源审计：四QAT。** 同一FP学生hash、同一帧采样hash；1000次QAT更新、累积2；冻结参数hash不变，量化图配置节点各阶段存在梯度。
- [ ] 当前四个QAT候选没有完整40任务、多初始状态正式测试。
- [x] 所选V1无教师QAT权重的全RKNN原精度版、RKNN＋RKLLM语言适配版已完成完整板端执行、重复回放和两个短任务，见第11节；不代表另外三个QAT模型也完成了板测。
- [ ] 四个QAT候选尚未完成统一板端全面质量和系统RAM对照；RKLLM适配版修改了语言量化配置，不是原GPU精度图原样部署。

每模型共31个任务-状态回合（19+12），不是31种不同任务；共有28种任务，Long任务1/2/5在两个初始状态重复。五模型合计155个闭环回合；原FP的第一轮结果复用了已核对的对应基线。

## 3. 成绩总表

| 模型 | 第一轮Long /10 | 第一轮其他 /9 | 第一轮 /19 | 第二轮 /12 | 合计 /31 |
|---|---:|---:|---:|---:|---:|
| FP | 7 | 9 | 16 | 7 | 23 |
| V1＋教师 | 4 | 8 | 12 | 6 | 18 |
| V1无教师 | 6 | 8 | 14 | 8 | 22 |
| V2＋教师 | 5 | 9 | 14 | 7 | 21 |
| V2无教师 | 4 | 6 | 10 | 7 | 17 |

## 4. 第一轮：19任务，初始状态0

任务ID从0开始。成功/失败均来自LIBERO实际回合结果。

| 任务组 | ID | 任务内容 | FP | V1＋教师 | V1无教师 | V2＋教师 | V2无教师 |
|---|---:|---|---|---|---|---|---|
| libero_spatial | 0 | 把盘子与小烤碗之间的黑碗放到盘子上 | 成功 | 成功 | 成功 | 成功 | 失败 |
| libero_spatial | 4 | 把木柜上层抽屉里的黑碗放到盘子上 | 成功 | 成功 | 成功 | 成功 | 失败 |
| libero_spatial | 8 | 把盘子旁边的黑碗放到盘子上 | 成功 | 失败 | 成功 | 成功 | 成功 |
| libero_object | 0 | 把字母汤放进篮子 | 成功 | 成功 | 成功 | 成功 | 成功 |
| libero_object | 4 | 把番茄调味酱（ketchup）放进篮子 | 成功 | 成功 | 成功 | 成功 | 成功 |
| libero_object | 8 | 把巧克力布丁放进篮子 | 成功 | 成功 | 成功 | 成功 | 成功 |
| libero_goal | 0 | 打开柜子的中间抽屉 | 成功 | 成功 | 成功 | 成功 | 成功 |
| libero_goal | 4 | 把碗放到柜顶 | 成功 | 成功 | 失败 | 成功 | 成功 |
| libero_goal | 8 | 把碗放到盘子上 | 成功 | 成功 | 成功 | 成功 | 失败 |
| libero_10 | 0 | 把字母汤和番茄酱都放进篮子 | 失败 | 失败 | 失败 | 失败 | 失败 |
| libero_10 | 1 | 把奶油奶酪盒和黄油都放进篮子 | 成功 | 成功 | 成功 | 成功 | 成功 |
| libero_10 | 2 | 打开炉灶并把摩卡壶放上去 | 成功 | 失败 | 成功 | 成功 | 成功 |
| libero_10 | 3 | 把黑碗放进柜子下层抽屉并关上 | 成功 | 失败 | 成功 | 失败 | 成功 |
| libero_10 | 4 | 白杯放到左盘，黄白杯放到右盘 | 成功 | 失败 | 失败 | 失败 | 失败 |
| libero_10 | 5 | 把书放进收纳架后部隔间 | 成功 | 成功 | 成功 | 成功 | 成功 |
| libero_10 | 6 | 白杯放到盘子上，巧克力布丁放到盘子右侧 | 成功 | 成功 | 失败 | 成功 | 失败 |
| libero_10 | 7 | 把字母汤和奶油奶酪盒都放进篮子 | 失败 | 失败 | 失败 | 失败 | 失败 |
| libero_10 | 8 | 把两个摩卡壶都放到炉灶上 | 失败 | 失败 | 成功 | 失败 | 失败 |
| libero_10 | 9 | 把黄白杯放进微波炉并关上 | 成功 | 成功 | 成功 | 成功 | 失败 |

## 5. 第二轮：12任务，初始状态1

| 任务组 | ID | 任务内容 | FP | V1＋教师 | V1无教师 | V2＋教师 | V2无教师 |
|---|---:|---|---|---|---|---|---|
| libero_spatial | 1 | 把小烤碗旁边的黑碗放到盘子上 | 失败 | 失败 | 失败 | 失败 | 失败 |
| libero_spatial | 2 | 把桌子中央的黑碗放到盘子上 | 成功 | 成功 | 成功 | 成功 | 成功 |
| libero_spatial | 5 | 把小烤碗上的黑碗放到盘子上 | 失败 | 失败 | 失败 | 失败 | 失败 |
| libero_object | 1 | 把奶油奶酪放进篮子 | 失败 | 失败 | 成功 | 失败 | 失败 |
| libero_object | 2 | 把沙拉酱放进篮子 | 失败 | 失败 | 失败 | 失败 | 失败 |
| libero_object | 5 | 把番茄酱（tomato sauce）放进篮子 | 成功 | 成功 | 成功 | 成功 | 成功 |
| libero_goal | 1 | 把碗放到炉灶上 | 成功 | 成功 | 成功 | 成功 | 成功 |
| libero_goal | 2 | 把葡萄酒瓶放到柜顶 | 成功 | 成功 | 成功 | 成功 | 成功 |
| libero_goal | 5 | 把盘子推到炉灶前方 | 成功 | 成功 | 成功 | 成功 | 成功 |
| libero_10 | 1 | 把奶油奶酪盒和黄油都放进篮子 | 失败 | 失败 | 成功 | 成功 | 成功 |
| libero_10 | 2 | 打开炉灶并把摩卡壶放上去 | 成功 | 成功 | 成功 | 成功 | 成功 |
| libero_10 | 5 | 把书放进收纳架后部隔间 | 成功 | 失败 | 失败 | 失败 | 失败 |

## 6. 第二轮：成功任务完成步数

“—”表示该模型失败，失败回合长度不能称为完成步数。同一行中双方成功时才作配对比较。

| 任务组 | ID | FP | V1＋教师 | V1无教师 | V2＋教师 | V2无教师 |
|---|---:|---:|---:|---:|---:|---:|
| libero_spatial | 1 | — | — | — | — | — |
| libero_spatial | 2 | 94 | 96 | 96 | 96 | 95 |
| libero_spatial | 5 | — | — | — | — | — |
| libero_object | 1 | — | — | 123 | — | — |
| libero_object | 2 | — | — | — | — | — |
| libero_object | 5 | 123 | 121 | 124 | 121 | 122 |
| libero_goal | 1 | 88 | 92 | 85 | 87 | 86 |
| libero_goal | 2 | 100 | 97 | 99 | 96 | 107 |
| libero_goal | 5 | 114 | 115 | 118 | 118 | 113 |
| libero_10 | 1 | — | — | 239 | 282 | 279 |
| libero_10 | 2 | 242 | 241 | 240 | 238 | 243 |
| libero_10 | 5 | 340 | — | — | — | — |

## 7. 文件体积与第二轮GPU开销

MB使用十进制10⁶ B。时延列先对每任务的完整动作块求p50，再对12个任务p50取中位数。实际轨迹输入不同，非固定输入微基准；全部是A10本地参考实现，不能写成RK3588加速。

| 模型 | 权重文件 MB | 相对原FP减少 | GPU峰值allocated MB | GPU峰值reserved MB | 动作块时延 ms |
|---|---:|---:|---:|---:|---:|
| FP | 906.71 | — | 1264.30 | 1314.91 | 208.63 |
| V1＋教师 | 504.28 | 44.38% | 1029.95 | 1551.89 | 408.48 |
| V1无教师 | 504.28 | 44.38% | 1029.95 | 1551.89 | 410.75 |
| V2＋教师 | 514.07 | 43.30% | 1028.83 | 1543.50 | 407.73 |
| V2无教师 | 514.07 | 43.30% | 1028.83 | 1543.50 | 412.63 |

## 8. 四QAT离线开发与重载验证

开发MAE是相对GT完整有效动作块的误差，不是任务成功率。四组同为40条开发观察，记录有效时间步1559。当前汇总未纳入同指标原FP对照，不能从本表宣称超过FP。

| 模型 | 最终有效动作块MAE | 连续6维MAE | 夹爪符号差异比例 | strict pack重载 | 重载动作MAE / max |
|---|---:|---:|---:|---|---|
| V1＋教师 | 0.022880 | 0.021348 | 0.009589 | 通过 | 0.0 / 0.0 |
| V1无教师 | 0.023252 | 0.021698 | 0.009712 | 通过 | 0.0 / 0.0 |
| V2＋教师 | 0.023557 | 0.021292 | 0.011756 | 通过 | 0.0 / 0.0 |
| V2无教师 | 0.023616 | 0.022031 | 0.010212 | 通过 | 0.0 / 0.0 |

## 9. 官方任务名称（由实际reset audit读取）

中文任务内容是简述；以下官方名称保留场景信息，可用于回查BDDL。

| 任务组 | ID | 官方名称 |
|---|---:|---|
| libero_spatial | 0 | `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate` |
| libero_spatial | 1 | `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate` |
| libero_spatial | 2 | `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate` |
| libero_spatial | 4 | `pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate` |
| libero_spatial | 5 | `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` |
| libero_spatial | 8 | `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate` |
| libero_object | 0 | `pick_up_the_alphabet_soup_and_place_it_in_the_basket` |
| libero_object | 1 | `pick_up_the_cream_cheese_and_place_it_in_the_basket` |
| libero_object | 2 | `pick_up_the_salad_dressing_and_place_it_in_the_basket` |
| libero_object | 4 | `pick_up_the_ketchup_and_place_it_in_the_basket` |
| libero_object | 5 | `pick_up_the_tomato_sauce_and_place_it_in_the_basket` |
| libero_object | 8 | `pick_up_the_chocolate_pudding_and_place_it_in_the_basket` |
| libero_goal | 0 | `open_the_middle_drawer_of_the_cabinet` |
| libero_goal | 1 | `put_the_bowl_on_the_stove` |
| libero_goal | 2 | `put_the_wine_bottle_on_top_of_the_cabinet` |
| libero_goal | 4 | `put_the_bowl_on_top_of_the_cabinet` |
| libero_goal | 5 | `push_the_plate_to_the_front_of_the_stove` |
| libero_goal | 8 | `put_the_bowl_on_the_plate` |
| libero_10 | 0 | `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` |
| libero_10 | 1 | `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket` |
| libero_10 | 2 | `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` |
| libero_10 | 3 | `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` |
| libero_10 | 4 | `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate` |
| libero_10 | 5 | `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy` |
| libero_10 | 6 | `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate` |
| libero_10 | 7 | `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket` |
| libero_10 | 8 | `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` |
| libero_10 | 9 | `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` |

## 10. 原始记录与结论边界

- 第一轮逐任务汇总：`runs/qat_four_way_comparison.json`；服务器原文件 `runs/qat_four_way_v1/comparison.json`。
- 第二轮逐任务汇总：`runs/six_candidate_screen_summary.json`；服务器原文件 `runs/six_candidate_screen_init1_seed2_v1/summary.json`。
- 第二轮任务名及初始状态：`runs/six_candidate_screen_reset_audit.json`。
- 第一轮任务名/状态审计：`artifacts/qat_backup/qat_distilled_full_v1_v1/{long10,other9}/reset_audit.json`；第一轮汇总已核对四模型与原FP的配对。
- 训练/开发/量化报告：两教师组 `artifacts/qat_backup/qat_distilled_full_v{1,2}_v1/report.json`；两无教师组 `runs/qat_distilled_v{1,2}_no_teacher_v1_evidence/report.json`。
- 第二轮各模型服务器目录均保存 `eval_info.json`、`reset_audit.json`、`action_trace.json`、`runtime_metrics.json`、`rollout.log` 与 `command.json`。
- 详细协议：[四组QAT比较](project-route.md#record-experiments-2026-10-02-qat-four-way)、[新状态配对筛查](project-route.md#record-experiments-2026-10-02-six-candidate-screen)。

当前四候选中优先V1无教师组：新状态8/12对FP7/12、显存减少约18.5%、文件减少44.38%；合计22/31仍低于FP23/31。本地完整动作块更慢，没有实测推理加速。这是小规模开发诊断，不证明总体质量改善，也不将成功变化单独归因蒸馏。

## 11. RK3588 完整部署：同任务质量对照

以下均为真实板端图像前处理、编码、语言/KV、动作专家去噪及动作后处理。LIBERO仿真器在本地等待板端响应，不能称机器人实时控制验证。Spatial/Object task0、初始状态index0、env seed0、noise seed100000/200000，每回合最多280步。已核对相关历史版本的初始观测及共同query噪声hash一致；最新两版本额外审计见 `runs/v1_rkllm_w8a8_g128_v1/same_task_comparison.json`。

| 部署版本 | 视觉／语言／专家精度及后端 | Spatial0 | Object0 | 成功数 |
| --- | --- | --- | --- | ---: |
| 原FP板端对照 | FP16，全RKNN | 成功，79步 | 成功，125步 | 2/2 |
| V1旧量化适配 | RKNN W8A8视觉/语言＋混合专家；部分例外曾适配 | 失败，280步 | 失败，280步 | 0/2 |
| V1 FP16前端诊断 | RKNN FP16视觉/语言＋混合量化专家 | 成功，79步 | 成功，129步 | 2/2 |
| V1 KL语言诊断 | RKNN FP16视觉＋KL混合语言＋量化专家 | 失败，280步 | 失败，280步 | 0/2 |
| V1 RKLLM FP16语言诊断 | RKNN FP16视觉＋RKLLM FP16语言＋RKNN量化专家 | 成功，80步 | 成功，130步 | 2/2 |
| V1全RKNN原精度分图 | RKNN W8A8视觉；语言保留INT16 DFP例外；专家保留FP16/BF16例外 | 失败，280步 | 失败，280步 | 0/2 |
| **V1 RKNN＋RKLLM W8A8（当前）** | **RKNN W8A8视觉＋RKLLM W8A8语言＋RKNN混合专家** | **失败，280步** | **失败，280步** | **0/2** |

每次最新配对测试同时重测的GPU原FP为Spatial成功78步、Object成功125步，即2/2。原FP板端对照未经过本次蒸馏/QAT，其余V1部署使用所选V1无教师QAT master。原精度分图保留原V1配置；当前RKLLM适配将语言INT16 DFP例外及其他语言Linear统一为RKLLM W8A8。FP16前端诊断成功不代表满足至少40%体积压缩。

本次只测两个短任务，Goal、Long及其他任务对当前RKNN＋RKLLM W8A8版本均**未测**。GPU的14/19、8/12不能填入此版本成绩；不同协议的旧Goal/Long结果也不合并。

## 12. RK3588 固定完整回放：体积、误差、延迟

两最新版本采用相同原始观测/初始噪声，warmup1＋计时3次，NPU core0；reference为同一个**所选V1 GPU pack**，不是原FP，也不是第8节的GT动作。计时含板端前处理、全部网络和后处理，不含模型初始化与网络RPC。数据源见 `runs/v1_exact_precision_v1/partitioned_replay.json`、`runs/v1_rkllm_w8a8_g128_v1/full_replay_verified.json`。

| 指标 | 上一版：全RKNN原精度分图 | 当前：RKNN＋RKLLM W8A8 |
| --- | ---: | ---: |
| 核心图文件＋CPU参数 MB | 448.62 | 487.40 |
| 相对原checkpoint906.71MB缩小 | 50.52% | 46.25% |
| 完整动作块p50 ms | 6066.006 | 5942.153 |
| 完整动作MAE | 0.141102664 | 0.317656893 |
| 最大动作绝对差 | 1.953972161 | 2.044516921 |
| 夹爪符号不同 /50 | 7 | 38 |
| 三次重复最大动作差 | 0 | 0 |
| Spatial/Object两个任务 | 0/2 | 0/2 |

当前核心参数为RKNN视觉111,973,263 B＋RKLLM语言210,701,524 B＋RKNN专家三分图117,175,409 B＋CPU参数47,546,790 B，共487,396,986 B。实际保存的文件包含量化元数据/容器，不按理想bit估算；不包含处理器、分词器、共享runtime和测试材料。不可与历史“完整运行资产”字节或第7节GPU单权重文件直接当作完全同口径。

两版本p50相差约124ms（2.04%），每版仅三次计时，未证明稳定加速。整体动作质量变差、任务仍失败，不能称综合收益改善。板端系统总RAM峰值未统一测量；RKLLM父进程RSS不含语言worker，不能直接和全RKNN单进程RSS比较。

### 视觉边界替换诊断（非最终部署成绩）

| 当前RKNN＋RKLLM的输入条件 | 动作MAE | 夹爪符号不同 /50 |
| --- | ---: | ---: |
| 实际板端W8A8视觉 | 0.317656893 | 38 |
| 仅诊断时改用GPU参考视觉特征 | 0.030223209 | 1 |
| 仅诊断时改用GPU完整prefix | 0.030223209 | 1 |

正确参考features经CPU组装后prefix差0，说明该观测主要误差来自RKNN视觉编码；后续仍有剩余误差，不称语言/专家完全无损。GPU特征替换不是可交付板端方案，未测这种替换的闭环成功率，不能把0.0302写成当前最终模型误差。

### 板端原始证据与技术记录

- 历史全RKNN质量：`runs/smolvla_board_libero_v1/summary.json`、`runs/qat_v1_board_short_tasks_v1/summary.json`。
- FP16/KL前端：`runs/qat_v1_repair_fp16_frontend_short_tasks_v2/summary.json`、`runs/qat_v1_repair_kl_prefix_short_tasks_v1/summary.json`；[数值诊断记录](project-route.md#record-experiments-2026-10-02-v1-rknn-numerical-repair)。
- RKLLM FP16语言：`runs/rkllm_native_patch_v1/wc_short_tasks_v2/summary.json`；[缓存修复记录](project-route.md#record-experiments-2026-10-03-rkllm-working-pool-coherency)。
- 全RKNN原精度：`runs/v1_exact_precision_v1/short_tasks/summary.json`；[板端记录](project-route.md#record-experiments-2026-10-03-v1-native-precision-board)。
- 当前RKNN＋RKLLM：`runs/v1_rkllm_w8a8_g128_v1/short_tasks/summary.json`、`full_replay_verified.json`、`native_live_replay.json`、`frontend_attribution.json`；[转换与板测记录](project-route.md#record-experiments-2026-10-03-v1-rknn-rkllm-adaptation)。目录名保留最初g128失败尝试，最终模型实际为普通W8A8。

**当前结论：后端拆分、完整稳定执行与至少40%参数压缩已验证，端侧任务质量尚未合格；接下来修RKNN视觉量化数值。**

## 13. 当前板端对原FP：两个短任务的动作分项

从已保存的两次配对rollout取首个50步动作块，逐数组确认输入RGB、状态、指令和初始噪声完全相同。参考是同时测试的**原始GPU FP模型**，不同于第12节固定回放的所选GPU QAT pack参考。后续两策略轨迹分叉，不直接逐块相减当作同输入量化误差。

| 任务 | 位置3维MAE | 旋转3维MAE | 位置相对RMSE | 旋转相对RMSE | 夹爪符号不同 /50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spatial task0 | 0.350899 | 0.031218 | 0.855150 | 1.230267 | 12 |
| Object task0 | 0.292562 | 0.019342 | 0.945315 | 1.482491 | 13 |

MAE=mean(abs(board-fp))，relative_RMSE=RMSE(board-fp)/RMS(fp)，各组分别按自身参考幅度归一化。位置和旋转是不同动作分量，不能将其单位/幅度不同的MAE直接排序成质量。旋转绝对误差较小，但原旋转动作RMS也小（0.033391/0.017795），相对偏差仍大；没有证据称旋转能力已保留。夹爪符号约74%—76%一致，仅代表首块预测开关符号，不证明实际抓取成功。

两个短抓放任务均失败且初始块已有明显位置/夹爪差异，因此当前不能将问题归为“只在长任务退化”。当前适配版没有完整Long闭环测评，长任务相对退化未知。已证实较接近的是相同输入重复稳定性，以及输入正确视觉特征时的后续语言/专家动作（第12节单观测诊断）；不推广为某类任务成绩恢复。

原始分项报告 `runs/v1_rkllm_w8a8_g128_v1/short_tasks/original_fp_first_chunk_comparison.json`，源动作 `fp_*/actions_000.npy`、`board_*/actions_000.npy`，对应原始输入 `input_000.npz`。


## 14. 原始FP16视觉＋V1 RKLLM/专家（2026-10-03）

用户接受约35%压缩后，复用原始checkpoint的FP16 RKNN视觉＋连接器，语言仍V1无教师QAT master的RKLLM W8A8，专家保留V1混合精度。实际核心参数588,044,896 B，缩小35.15%。**同时改变视觉权重来源与执行精度**，不是原V1完整精度图。

| 配对任务 | 原GPU FP | 板端 | 此前W8A8视觉板端 |
| --- | --- | --- | --- |
| Spatial0 | 成功，78步 | 成功，78步 | 失败，280步 |
| Object0 | 成功，125步 | 成功，141步 | 失败，280步 |
| 合计 | 2/2 | 2/2 | 0/2 |

与此前相同init_state0/env_seed0/noise_seed100000和200000，每块50动作。固定输入对所选GPU pack动作MAE0.032225（此前0.317657），夹爪1/50（此前38/50），重复最大差0，warmup1/repeats3 p50 7.259s（此前5.942s）。实际恢复两项短任务，Object完成步数增加；Long、完整40任务和整策略峰值内存未测，不能推定与原模型全面等效。

[技术记录与原始数据](project-route.md#record-experiments-2026-10-03-original-fp16-vision-rkllm-v1)。


### 14.1 四suite扩展筛查

预先固定追加Spatial1/Object1/Goal1/Long1，与上面0号任务合计6个不同任务，每任务init_state0/env_seed0，原FP/板端严格配对，不按成功结果筛选任务。

| 任务 | 原GPU FP | 当前板端 |
| --- | --- | --- |
| spatial0 | 成功，78步 | 成功，78步 |
| object0 | 成功，125步 | 成功，141步 |
| spatial1 | 成功，109步 | 成功，117步 |
| object1 | 失败，280步 | 成功，126步 |
| goal1 | 成功，91步 | 成功，92步 |
| Long1 | 成功，227步 | 成功，238步 |

**原GPU FP5/6、板端6/6**。这组原FP成功的5项板端均成功；Object1原FP失败、板端成功。Long1是把cream cheese和butter放入basket的组合任务，板端较FP多11步。每项仅1回合，不能推断完整Long10或40任务成功率。

六项首块连续动作MAE0.0250–0.0396，夹爪符号差异0/0/0/1/0/0（各50步，相对原GPU FP）；首输入图像/state/指令/noise均严格一致。18次真实闭环query p50 7.349s、p95 7.786s；包含预处理及整策略，不含RPC/仿真，含首query，口径与固定回放不同。体积仍588.04MB/缩小35.15%。

新增有效指令长度136/142/147经执行及跨指令参考重复验证后开放，160-token底层图与权重未改变。详细记录、视频路径和汇总见[实验记录](project-route.md#record-experiments-2026-10-03-original-fp16-vision-rkllm-v1)，原始 `runs/v1_rkllm_original_fp16_vision_v1/combined_screening.json`。


### 14.2 用户定版与全面测试准备

2026-10-03用户将此组合确定为终版模型：原始FP16 RKNN视觉＋V1无教师损失QAT权重的RKLLM W8A8语言＋V1混合RKNN专家，实际参数588.04MB，接受35.15%压缩。模型身份登记 `runs/v1_rkllm_original_fp16_vision_v1/final_selection.json`；定版不代表全面测试已通过。

板上原始checkpoint完整FP16部署仍在 `/root/qvla_board_test/smolvla_vision_v1`：视觉212,621,173 B、语言326,154,086 B、专家217,425,832 B、CPU189,363,274 B，共945,564,365 B。都是原checkpoint的板端适配，GPU原checkpoint906,712,520 B与部署容器/CPU参数存储口径不同。三个FP16图历史编译14.793/25.610/19.501秒（不含导出/传输/接口验证），现文件齐全无需重转。全面速度/资源/成功率结果待测，不能将历史父进程RSS当整模型峰值。


### 14.3 终版全面对比运行入口

用户指定成功率只比较原GPU FP与终版板端；速度和资源比较原板端FP16与终版板端。已启动独立后台串行pipeline（无板端并行干扰），原始数据 `runs/final_model_comparison_v1/`，新建[终版全面对比文档](results.md#final-comparison)自动更新各阶段已测结果。40任务指令15个有效长度134..151已验证全部执行及参考prefix/KV/action前后重复差0；未改变模型权重/精度。全面成功率重新按同一脚本运行原GPU FP作配对，而不是混用历史31/40、32/40。


### 14.4 终版40任务全面对比结果

完整40个不同任务各1回合配对完成：原GPU FP30/40（75%）、终版板端25/40（62.5%），下降12.5个百分点。Spatial8→5、Object8→8、Goal7→7、Long7→5；6项改善、11项回归。同一初始观测和共同query噪声核验通过，40个首块原始图像/state/task/noise额外严格核验通过。此前6任务筛查6/6不能代表总体。

同板固定输入warmup3/repeats20：原始P50 7.519918s、终版（修正资源采样补测）7.262644s，本次约快3.42%；P95 7.634695→7.579030s。核心参数945.56→588.04MB（对原始板端减少37.81%；对906.71MB checkpoint减少35.15%）。有效进程树峰值PSS1824.404→1129.911MiB，减少38.07%，终版已包含RKLLM worker。20次重复动作差0，补测动作与首次资源实验严格一致。

首次内存861MiB/降低52.8%无效：板子内核缺失 `/proc/PID/task/TID/children`，旧代码漏算worker。旧报告保留在 `resource_sampling_v1_missing_worker/`；改成PPid扫描递归后补测终版，最大进程数2；原全RKNN无子进程，原单进程PSS仍有效。PSS不覆盖未映射的驱动DMA、0.5秒采样可能漏瞬时峰值。完整资源/频率/温度/利用率、40任务逐项结果见[独立对比文档](results.md#final-comparison)。

当前定版组合有体积和运行内存收益，但总体质量回归；不能宣传为无损量化或总体任务优于FP。


### 14.5 成功率测评复核

用户质疑下降后，核验模型/库hash、80回合实际动作拼接、seed/噪声/后端调用、SDK/RPC日志、40首观测前后处理均通过；之前6任务两模型全部执行动作与本次40任务相同。冷启动复测Spatial2/Goal4/Long2，原FP仍成功110/91/230步、板端仍失败280/300/520步，全部动作轨迹与原40任务逐元素一致。30/40对25/40统计不变。见[复核记录](project-route.md#record-experiments-2026-10-03-final-evaluation-audit)。没有发现测评计数或配置错误，但尚未验证回归任务所有中间计算与GPU等价，不能据重复稳定排除确定性的缓存语义/转换数值问题。


### 14.6 成功数持平的任务类别

本次40任务单回合配对中，**Object（物体识别/抓放）8/10→8/10**、**Goal（目标条件操作）7/10→7/10**，两类总成功率分别持平于80%和70%。Object改善task1/5、回归task2/3；Goal改善task0/3、回归task4/5。因此是类别总成功数持平，不是每个具体任务都保持原结果，也不代表多seed统计非劣。Spatial和Long仍有回归，全模型总体30/40→25/40的结论保留。


</details>


## 本机原始模型与30Hz异步供给测试（2026-10-14）

测试设备RTX4060 Laptop约8GB，Torch2.7.1+cu118；原始SmolVLA checkpoint未改权重，保持加载后的FP32/BF16混合浮点参数（225/275个参数张量）、use_amp=false、10步flow推理。双相机RGB uint8 CHW 256×256及8维state来自既有40任务观测缓存，不接摄像头或机械臂；输出50×7动作。两CPU线程，未启用compile、额外autocast、量化或减少num_steps。

| 项目 | 实测 |
| --- | ---: |
| 加载模型（不含首次预热） | 15.104s |
| 预热/正式完整动作块次数 | 5 / 60 |
| 完整动作块P50 / P95 / 最大耗时 | 0.153062 / 0.158795 / 0.176355s |
| PyTorch进程内峰值CUDA allocated / reserved | 1.185 / 1.225GiB |
| 30Hz下50动作的执行时间预算 | 1.666667s |
| 按P95推算的计算提前量 | 至少5动作（约0.167s），未含设备I/O余量 |
| 定时异步影子循环 | 30s / 900个tick |
| 后台推理次数 / 断粮 / 请求丢弃 | 90 / 0 / 0 |
| 后台动作块P50 / P95 | 0.151512 / 0.153313s |
| 取动作tick间隔P50 / P95 | 33.334 / 33.450ms |
| tick唤醒延迟P95 / 最大值 | 0.211 / 0.306ms |
| 延后超过一个33.33ms周期的tick | 0 |

串行时间包含CPU图像转float、processor、GPU传输、完整前向、后处理及动作回传CPU，显式CUDA同步。CUDA allocated只是PyTorch分配量，不等于整机nvidia-smi占用，更不等于此前板端PSS。异步先准备首块，再启动30Hz消费者；每10tick提交一条缓存观测，后台仅一个推理线程，将推理耗时对应的过期前缀跳过，替换未来动作。不是LeRobot官方async/RTC栈，也未做prefix conditioning、动作连续性或闭环任务质量验证。

结论：本次工作负载可满足**30Hz动作供给**；不满足每33.33ms从新图像推理一整块的**30Hz策略前向**。1.67s是把50步按30Hz执行的算术预算，不证明改变训练数据执行频率后质量保持。SO101通常使用六电机关节目标，本checkpoint是LIBERO七维动作，不能删一维或直接发送；需SO101数据微调/对应checkpoint、校准、摄像头和串口周期实测后再评价实机控制。

复现：`.venv-haq-local/bin/python scripts/evaluate.py local-async --repeats 60 --warmup 5 --seconds 30 --hz 30 --request-every 10`。原始报告/逐次耗时在`runs/local_async30_v1/report.json`、`timings.npz`和`runs/local_async30_v1.log`。

- checkpoint SHA256：`9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`。
- 观测缓存SHA256：`e1861f28f234b533c33b97104f64ea51351e0789f35afc8df72cb5850e7b311c`。
- 实际运行源码快照SHA256：`de2c4a60b979162da7bd2a060040f8cd7dc13904c34695865ac61e32ca9161d6`，快照`runs/local_async30_v1/benchmark_executed.py`；随后只补参数校验/hash记录字段和统一CLI入口。
- 官方参考：[SmolVLA实机微调](https://huggingface.co/docs/lerobot/smolvla)、[LeRobot v0.6.1 SO follower六电机接口](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/robots/so_follower/so_follower.py)。


## 板端运行优化：速度与内存权衡（2026-10-14）

本节比较**同一终版模型的旧运行代码与实验分支代码**，不是原GPU浮点模型与量化模型，也不更新此前40任务模型成功率。模型、量化精度、图hash和10步Euler不变；双相机独立RKNN context在核1/2并行，动作专家及native BF16小投影保留核0，RKLLM保持原native组件与CPU1线程配置。

| 指标 | 旧版单核 | 优化版双相机并行 |
| --- | ---: | ---: |
| 完整50步动作块P50 | 7.255339s | 5.405506s |
| 完整动作块P95 | 7.438293s | 6.209123s |
| 初始化耗时 | 2.041s | 2.615s |
| 采样峰值进程树PSS | 1126.136MiB | 1579.669MiB |
| 正式重复次数 | 20 | 20 |
| 固定输入重复最大动作差 | 0 | 0 |
| 新旧20次完整动作比较 | — | 逐元素完全相同，最大差0 |

完整延迟降低**25.50%**，速度比**1.342×**。但峰值PSS增加**453.533MiB（+40.27%）**。缓冲复用减少重复分配不等于总峰值下降，第二视觉context的代价更大；保留默认单核模式，通过配置显式选择更快模式。未采用不稳定的专家三核配置，不把加载成功当作精度通过。

速度为相同图像/state/指令/噪声的完整predict，不含模型初始化、SSH、仿真和报告序列化；两个版本各预热3次、正式20次，顺序执行，没有与闭环测试并行。0.5s采样PPid进程树，均观察到2个进程（含RKLLM worker），PSS完整；不含未映射驱动DMA，可能遗漏瞬时峰值。采样NPU频率均为1GHz，未强制修改系统频率；温度及逐次数据保留在原始资源报告。

优化版阶段中位数：视觉2.104s、语言及K/V处理0.483s、10步专家2.710s、预处理0.125s。各阶段中位数不能直接相加视为整体P50。专家仍是主要瓶颈之一；不能把总耗时归因于Python调度。

### 两项短闭环回归

| 任务 | 旧版 | 优化版 | 执行步数 | 完整轨迹 |
| --- | --- | --- | --- | --- |
| libero_object:0 | 成功 | 成功 | 141→141 | 逐元素完全一致 |
| libero_goal:0 | 成功 | 成功 | 257→257 | 逐元素完全一致 |

固定init_state0/env_seed0，噪声沿用`100000*(suite_index+1)+task_id`及CUDA Generator；新旧所有query输入hash一致。只作两任务运行代码回归，不是新的40任务或多seed质量证明。原始图像、噪声、每块动作和实际执行轨迹在`runs/runtime_optimization_v1/quality2/`。其他新观测不保证必然逐位一致，仍受后端可复现性边界约束。

### 30Hz结论与复现

50动作按30Hz执行仅有1.6667s预算，目前P50仍为5.406s，按计算供给约9.25动作/秒，尚需额外约3.24倍加速才刚达标。异步不能弥补持续吞吐不足；未连接SO101，也未减少flow步数或重训模型。

- 分支：`experiment/rk3588-runtime-optimization`；基线：`30bb6b2`。
- 配置：`config/runtime_optimization_v1.json`，默认runtime仍保留单核，不修改`models/final`。
- 面板入口：`python -m qvla.evaluation.runtime_optimization_panel --root models/final --inputs inputs.npz --output <目录> --sequence 0 --warmup 3 --repeats 20 --resources --options '{"vision_cores":["1","2"],"expert_core":"0"}'`。基线在`30bb6b2`源码快照下运行同一面板，不传options。
- 完整小结、准确source/model hash：`runs/runtime_optimization_v1/summary.json`、`board_identity.json`；板端报告与原始采样：`board_results/final_benchmark/{baseline,optimized}/`。
- 输入panel SHA256：`86f2c986b14035ebe1adbaefa0cb9416bb9ee7f9c7f29dd16c59b8d23f863c71`；由既有观测缓存索引0/4/25与NumPy seed42构建，测量只使用样本0。运行代码验证序列A/B/C/A也已通过。

## 专家多核诊断结果（2026-10-14，尚未修复）

固定输入、同一专家RKNN文件，每种有效配置回放6次：

| 对象与配置 | 与单核基准逐位一致 | 重复最大差 |
| --- | --- | ---: |
| 专家前半子图，分别单用核0/1/2 | 是 | 0 |
| 专家前半子图，双核0+1或三核协同 | 否 | 3.7893732190 |
| 专家前半子图，标准C API三核 | 否 | 3.7893732190 |
| 专家后半子图，核0 | 是 | 0 |
| 专家后半子图，三核协同 | 否 | 1.6336455941 |
| 独立MLP、QKV图，单/双/三核 | 是 | 0 |
| 小QK/注意力：5头FP16/INT8，15头INT8，单/双/三核 | 是 | 0 |

完整链路3次捕获显示：语言K/V和专家全部输入相同，首次可观察偏差发生在第0轮专家前半子图两个计算输出，早于BF16投影与Euler更新。标准C API核0先通过逐位对齐，再在三核复现偏差，基本排除Python分图调度和缓存复用。小图对照正常，不能把问题泛化成所有RKNN多核/注意力都不支持。

具体内部根因仍未确定，已测试的额外等待、perf模式、局部WC及no_sram不是修复；全局WC触发超时，已排除。诊断中的早期布局快照错误也已剔除，详细无效结果与纠正依据见[路线诊断记录](project-route.md)。

安全配置最终回放2次仍与旧基准逐位相同（最大差0），没有改模型、系统驱动或正式runtime库，维持已有视觉并行＋专家核0方案。此节是正确性诊断，不是新的任务成功率或性能评测。

## 忽略质量的速度探索与单相机代理（2026-10-14）

固定50步动作；CPU performance＋大核3/4/5/6，NPU 1GHz，语言RKLLM 3线程。此表是新速度实验，**不是已验证终版质量的部署成绩**。

| 配置 | Euler次数 | P50秒 | 实测次数 |
| --- | ---: | ---: | ---: |
| 终版图、双相机分核，performance | 10 | 3.738 | 5 |
| INT8视觉＋旧单体专家、双相机，performance | 10 | 3.091 | 5 |
| 单相机代理、FP16视觉三核 | 10 | 3.066 | 3 |
| 单相机代理、INT8视觉三核 | 10 | 2.913 | 3 |
| 单相机代理、FP16视觉三核 | 1 | 1.716 | 3 |
| 单相机代理、INT8视觉三核 | 1 | **1.552** | 3 |

各组预热1次。单相机代理每块只执行一次视觉，特征复制至固定双视图语言图，保留原两图预处理；没有跨块缓存和质量验证。输出仍为LIBERO 7维，不能直接用于SO-101。1.552秒对应50/1.552≈32.21动作/秒，但仅余约114ms预算、未含采集通讯/真实控制，也未验证1步Euler质量，不能称已实现稳定30Hz。单相机最快一次1.545秒；本轮有限配置探索不能证明理论极限。CPU governor已恢复schedutil。

完整配置表、技术边界见[路线文档](project-route.md#速度上限探索及单相机代理测速2026-10-14)；原始记录`runs/speed_ceiling_v1/board/`。

## 官方 SmolVLA base 本机运行检查（2026-10-15）

RTX4060 Laptop GPU，原始base权重FP32/BF16混合，三路模拟图像＋6维状态＋指令；10次Euler，完整前后处理，预热1次、测3次。输出`[1,50,6]`，全部有限，同输入重复最大差0。P50 **0.187秒/50步动作**（实测0.184/0.189/0.187秒），PyTorch峰值allocated约1.181GiB。只证明可以执行，不代表真实SO-101任务效果或控制频率；与此前LIBERO协议不同。模型、输入hash及原始报告路径见路线文档末尾的同名记录。

## SmolVLA base 单视图与三视图GPU延迟（2026-10-15）

同一RTX4060 Laptop，官方base、10次Euler、50步×6维、原始精度与完整前后处理，每组预热5次、测60次；固定模拟输入，未测任务成功率。

| 实际视图数 | P50秒 | P95秒 |
| --- | ---: | ---: |
| 1 | 0.123482 | 0.127477 |
| 3 | 0.199992 | 0.202698 |

单视图P50降低38.26%，为官方动态输入路径，缺失视图直接跳过；已核对每次只运行1次视觉编码。输出正常且固定输入可重复；未涉及真实相机、电机、SO-101闭环或板端部署。原始记录`runs/smolvla_base_camera_benchmark_v1/report.json`。

## INT8优先专家速度探测（2026-10-15，质量不约束）

相同INT8视觉＋RKLLM W8A8语言、单相机特征复用代理、performance/大核、50步×7维输出；每组预热1次、测5次。新专家以原始FP专家导出图重新默认W8A8转换，无手工FP16/BF16例外；源权重与终版V1不同，不能视为严格仅精度消融。

| 专家配置 | 1次Euler P50 | 3次Euler P50 | 10次Euler P50 |
| --- | ---: | ---: | ---: |
| 原混合精度专家、核0，同轮对照 | 1.559秒 | 1.805秒 | 2.862秒 |
| INT8优先单体专家、核0 | **1.531秒** | 1.838秒 | **2.802秒** |
| INT8优先单体专家、三核 | 1.540秒 | 1.798秒 | 2.831秒 |

本轮最短一次1.527秒；同轮1次Euler的P50差异约1.79%，收益有限，未证明显著性。最快组视觉约1.078秒、语言0.248秒、专家0.129秒。新专家的SDK性能表中609个NPU节点均标INT8（带权重Conv/融合Conv共116个），CPU保留索引/位置与其他算子，不能称整策略纯INT8。单相机接口仍复用特征、语言图长度未缩短；没有真实SO-101闭环或质量验证。CPU governor已恢复。

详细配置、转换hash、校准及边界见路线文档末尾的同名记录；原始数据`runs/lowest_precision_speed_v1/summary.json`及`board/audit/operator_perf.txt`。

## 默认10步时的耗时分布复核（2026-10-15）

| 执行配置 | 整体P50 | 视觉P50 | 专家10步P50 | 语言P50 |
| --- | ---: | ---: | ---: | ---: |
| 原始LIBERO FP16板端、双相机、核0，历史3次回放 | 8.047秒 | 3.852秒 | 3.452秒 | 0.614秒 |
| 当前板端混合专家、单相机复用代理、performance，本轮5次 | 2.862秒 | 1.090秒 | 1.436秒 | 0.235秒 |
| 官方base原权重、本机GPU、单相机，新20次CUDA-event细分 | 116.318毫秒 | 33.162毫秒 | 72.464毫秒 | 7.769毫秒 |

板端原FP16是已有逐段记录分析，未在今天重跑；不套用到后续7.520秒资源基准。各行模型/相机/后端/协议不同，不能据此计算量化加速比；各阶段中位数也不要求相加等于整体中位数。默认10步下，当前板端专家约占50%、视觉38%；“视觉70%”仅适用于此前1步极速配置。本机GPU细分同样显示专家是主要耗时，但不能把GPU内部比例直接套到NPU。GPU打点数据包含嵌套，不能重复累加。细节及原始文件身份在路线文档末尾同名记录，分析产物`runs/smolvla_base_profile_v1/comparison.json`。

## 保留10步积分的30Hz速度探测（2026-10-15，忽略效果）

以下全部在RK3588真实执行视觉＋语言＋10次专家＋前后处理，每块50步×7维。INT8优先、performance/大核、视觉三核/专家核0。低分辨率特征扩展至旧64token接口并复制第二视图；不是SO-101模型，也没有质量结论。

| 视觉尺寸 | 专家层数 | FFN宽度 | 完整10步P50 |
| --- | ---: | ---: | ---: |
| 512 | 16 | 2048 | 2.783秒 |
| 256 | 16 | 2048 | 1.838秒 |
| 128 | 16 | 2048 | 1.734秒 |
| 128 | 16 | 512 | 1.678秒 |
| 128 | **8** | 2048 | **1.066秒** |
| 128 | **4** | 2048 | **0.705秒** |

前四组预热1次测3次、后两组预热1次测5次。专家“层数”与“积分步数”不同：8/4层候选也执行10轮，未复用一轮结果冒充10轮。结构剪枝未训练恢复，权重来自原始FP导出，不能转移V1 QAT质量成绩；512对照视觉来自旧V1图，也不是只改变分辨率的严格消融。

最快4层候选完成10秒30Hz模拟消费：**300/300个动作向量、0断供**；6次后台推理P50 0.696秒，最大0.955秒；tick迟到P95 0.075ms，最大0.552ms。初始动作块预先准备、固定观测重复推理、无真实相机/网络/电机，不做过期动作替换；仅证明短时间计算供给有余量，**不等于真实SO-101控制或任务质量已达标**。

原始`runs/speed30_tenstep_v1/summary.json`及`board/shadow_depth30.json`；技术原理/构建配置及全部9组结果在路线文档末尾同名记录，选定极速配置`config/runtime_speed_probe_tenstep.json`。正式终版模型未修改。
