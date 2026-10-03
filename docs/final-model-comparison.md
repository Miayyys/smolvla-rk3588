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

入口：`scripts/run_final_model_comparison.py`；资源测量：`scripts/benchmark_smolvla_board_resources.py`；质量评测：`scripts/run_smolvla_board_libero.py`；本文由 `scripts/report_final_model_comparison.py` 根据已测产物更新，未完成项不得填作结果。

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

完整复核见[实验记录](experiments/2026-10-03-final-evaluation-audit.md)，原始数据 `evaluation_audit/`。