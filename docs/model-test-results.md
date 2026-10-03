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
- 详细协议：[四组QAT比较](experiments/2026-10-02-qat-four-way.md)、[新状态配对筛查](experiments/2026-10-02-six-candidate-screen.md)。

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
- FP16/KL前端：`runs/qat_v1_repair_fp16_frontend_short_tasks_v2/summary.json`、`runs/qat_v1_repair_kl_prefix_short_tasks_v1/summary.json`；[数值诊断记录](experiments/2026-10-02-v1-rknn-numerical-repair.md)。
- RKLLM FP16语言：`runs/rkllm_native_patch_v1/wc_short_tasks_v2/summary.json`；[缓存修复记录](experiments/2026-10-03-rkllm-working-pool-coherency.md)。
- 全RKNN原精度：`runs/v1_exact_precision_v1/short_tasks/summary.json`；[板端记录](experiments/2026-10-03-v1-native-precision-board.md)。
- 当前RKNN＋RKLLM：`runs/v1_rkllm_w8a8_g128_v1/short_tasks/summary.json`、`full_replay_verified.json`、`native_live_replay.json`、`frontend_attribution.json`；[转换与板测记录](experiments/2026-10-03-v1-rknn-rkllm-adaptation.md)。目录名保留最初g128失败尝试，最终模型实际为普通W8A8。

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

[技术记录与原始数据](experiments/2026-10-03-original-fp16-vision-rkllm-v1.md)。


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

新增有效指令长度136/142/147经执行及跨指令参考重复验证后开放，160-token底层图与权重未改变。详细记录、视频路径和汇总见[实验记录](experiments/2026-10-03-original-fp16-vision-rkllm-v1.md#扩展结果与边界)，原始 `runs/v1_rkllm_original_fp16_vision_v1/combined_screening.json`。


### 14.2 用户定版与全面测试准备

2026-10-03用户将此组合确定为终版模型：原始FP16 RKNN视觉＋V1无教师损失QAT权重的RKLLM W8A8语言＋V1混合RKNN专家，实际参数588.04MB，接受35.15%压缩。模型身份登记 `runs/v1_rkllm_original_fp16_vision_v1/final_selection.json`；定版不代表全面测试已通过。

板上原始checkpoint完整FP16部署仍在 `/root/qvla_board_test/smolvla_vision_v1`：视觉212,621,173 B、语言326,154,086 B、专家217,425,832 B、CPU189,363,274 B，共945,564,365 B。都是原checkpoint的板端适配，GPU原checkpoint906,712,520 B与部署容器/CPU参数存储口径不同。三个FP16图历史编译14.793/25.610/19.501秒（不含导出/传输/接口验证），现文件齐全无需重转。全面速度/资源/成功率结果待测，不能将历史父进程RSS当整模型峰值。


### 14.3 终版全面对比运行入口

用户指定成功率只比较原GPU FP与终版板端；速度和资源比较原板端FP16与终版板端。已启动独立后台串行pipeline（无板端并行干扰），原始数据 `runs/final_model_comparison_v1/`，新建[终版全面对比文档](final-model-comparison.md)自动更新各阶段已测结果。40任务指令15个有效长度134..151已验证全部执行及参考prefix/KV/action前后重复差0；未改变模型权重/精度。全面成功率重新按同一脚本运行原GPU FP作配对，而不是混用历史31/40、32/40。


### 14.4 终版40任务全面对比结果

完整40个不同任务各1回合配对完成：原GPU FP30/40（75%）、终版板端25/40（62.5%），下降12.5个百分点。Spatial8→5、Object8→8、Goal7→7、Long7→5；6项改善、11项回归。同一初始观测和共同query噪声核验通过，40个首块原始图像/state/task/noise额外严格核验通过。此前6任务筛查6/6不能代表总体。

同板固定输入warmup3/repeats20：原始P50 7.519918s、终版（修正资源采样补测）7.262644s，本次约快3.42%；P95 7.634695→7.579030s。核心参数945.56→588.04MB（对原始板端减少37.81%；对906.71MB checkpoint减少35.15%）。有效进程树峰值PSS1824.404→1129.911MiB，减少38.07%，终版已包含RKLLM worker。20次重复动作差0，补测动作与首次资源实验严格一致。

首次内存861MiB/降低52.8%无效：板子内核缺失 `/proc/PID/task/TID/children`，旧代码漏算worker。旧报告保留在 `resource_sampling_v1_missing_worker/`；改成PPid扫描递归后补测终版，最大进程数2；原全RKNN无子进程，原单进程PSS仍有效。PSS不覆盖未映射的驱动DMA、0.5秒采样可能漏瞬时峰值。完整资源/频率/温度/利用率、40任务逐项结果见[独立对比文档](final-model-comparison.md)。

当前定版组合有体积和运行内存收益，但总体质量回归；不能宣传为无损量化或总体任务优于FP。


### 14.5 成功率测评复核

用户质疑下降后，核验模型/库hash、80回合实际动作拼接、seed/噪声/后端调用、SDK/RPC日志、40首观测前后处理均通过；之前6任务两模型全部执行动作与本次40任务相同。冷启动复测Spatial2/Goal4/Long2，原FP仍成功110/91/230步、板端仍失败280/300/520步，全部动作轨迹与原40任务逐元素一致。30/40对25/40统计不变。见[复核记录](experiments/2026-10-03-final-evaluation-audit.md)。没有发现测评计数或配置错误，但尚未验证回归任务所有中间计算与GPU等价，不能据重复稳定排除确定性的缓存语义/转换数值问题。


### 14.6 成功数持平的任务类别

本次40任务单回合配对中，**Object（物体识别/抓放）8/10→8/10**、**Goal（目标条件操作）7/10→7/10**，两类总成功率分别持平于80%和70%。Object改善task1/5、回归task2/3；Goal改善task0/3、回归task4/5。因此是类别总成功数持平，不是每个具体任务都保持原结果，也不代表多seed统计非劣。Spatial和Long仍有回归，全模型总体30/40→25/40的结论保留。
