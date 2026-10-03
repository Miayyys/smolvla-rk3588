# 终版40任务成功率测试复核

## 目的与模型身份

用户质疑原GPU FP30/40、终版板端25/40是否由测评错误造成。保持模型、量化精度、runtime和评测脚本不变，先核对既有80回合原始数据，再冷启动复测预先固定的三个回归任务，不重新量化或训练。

固定40任务的GPU为本地RTX4060 Laptop，torch2.7.1+cu118、LeRobot0.6.1、Transformers5.5.4、hf-libero0.1.4、MuJoCo3.8.1，非历史A10服务器结果。原始checkpoint SHA `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`，当前测评脚本 SHA `47241bf4832a39cf9fafddcb24dc8e5fe479e736ea74d83300dc66e5ed57ea9d`。冻结终版manifest SHA `5533924f3974599f92d555ce3b04a5698403da7ba9805cde6a551444e00bea0f`。板子实际逐文件SHA与该manifest核对：视觉、语言、专家三分图、CPU参数、worker和两份RKLLM/分配控制库均匹配。

## 已有40任务数据审计

- 40个唯一 `(suite,task_id)`、每项FP/板端两回合，共80条记录，无重复或遗漏。成功数来自各任务 `info['is_success']`，不是离线MSE。
- env_seed0、initial state0；noise_seed=`100000*(suite序号+1)+task_id`。保存的每个query噪声SHA与trace一致；共同query FP/板端噪声配对。40个首输入图像/state/task/noise的严格配对见 `quality40_action_audit.json`。
- 每块实际执行50动作；把所有保存的预测块拼接、按simulation_steps裁到实际终止位置，与 `executed_actions.npy` 逐元素完全一致。回合步数不超过280/300/520上限。没有以额外重复episode填补失败。
- 全部板端query语言后端为RKLLM，调用2个视觉/1个语言/30个专家子图，未发现SDK/RPC错误、traceback、返回错误。板端有限性和shape检查在原评测中严格执行。
- 冻结checkpoint、测评脚本、manifest文件hash一致。资源采样脚本后来修正PPid枚举，与成功率脚本无关，不混淆两种复核。
- 之前Spatial0/Object0及新增Spatial1/Object1/Goal1/Long1的两模型结果、完成步数、**全部执行动作**均与40任务对应回合严格一致。小规模6/6不是模型被换掉，而是选到的6个任务表现较好。

原始证据 `runs/final_model_comparison_v1/evaluation_audit/static_audit.json`、`board_identity_audit.json`；脚本 `scripts/audit_final_model_evaluation.py`。

## 前后处理复核

重新使用checkpoint自带processor，将40任务实际首观测送入CPU版原始处理器＋`SmolVLAPolicy.prepare_images/prepare_state`，对照板端同一NumPy公式，检查摄像头顺序、缩放、state标准化/填充、完整40条语言token与mask。动作反归一化另外用固定random seed42、[1,50,7]输入对照。

| 项目 | 结果 |
| --- | --- |
| 40任务token IDs与mask | 逐元素完全相同 |
| 图像最大绝对差 | 2.384185791015625e-7 |
| state最大绝对差 | 0 |
| 动作后处理最大绝对差 | 2.9802322387695312e-8 |

图像阈值5e-7、state/动作后处理1e-6，均通过；属于浮点舍入级，未发现相机交换、归一化或文本处理不一致。本次对照的是40个首观测、checkpoint CPU处理器和部署NumPy公式，不能替代所有网络中间tensor的板端/GPU验证。原始 `preprocessing_audit.json/.log`；脚本 `scripts/audit_final_model_preprocessing.py`。

## 三个回归冷启动复测

预先固定Spatial2、Goal4、Long2，各用原评测脚本在独立进程加载GPU原FP和完整板端模型，同seed0/initial state0/动作噪声，任务内完整闭环；没有修改精度或仅重放第一块。

| 任务 | 原GPU FP再次结果 | 终版再次结果 | 与原40任务的执行动作 |
| --- | --- | --- | --- |
| Spatial2 | 成功，110步 | 失败，280步 | 两模型均逐元素一致 |
| Goal4 | 成功，91步 | 失败，300步 | 两模型均逐元素一致 |
| Long2 | 成功，230步 | 失败，520步 | 两模型均逐元素一致 |

证明这3个回归在冷启动时可重复，不能用长期运行状态漂移解释它们；没发现改变成功统计的测评错误。原始 `repeat_results.json`、`summary.json`、三个任务子目录中视频/输入/预测块/执行动作和 `pipeline.log`。

## 结论及边界

本协议实测仍是30/40对25/40：6个改善、11个回归、净少5个（75%→62.5%，下降12.5个百分点）。这是一组单seed开发任务回合；历史31/40、32/40不同运行不能代入本次配对对照，也不能把此次40任务当作多seed总体统计。

本复核未找到成功率测评配置/计数错误，结果可复现。**没有证明所有部署计算都正确等价于GPU**：尚未对这些回归任务逐层对照RKLLM K/V、RKNN专家中间输出，也未分离蒸馏/QAT权重变化与原生量化/后端变化。确定性的attention/cache/转换数值问题仍可能导致真实任务失败；不能将降分简单解释为低bit必然损失。后续如继续定位，应针对这些固定失败观测比较中间输出，而非重新跑40任务或凭MSE调参数。
