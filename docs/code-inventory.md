# 代码盘点、上传范围与简历素材

更新：2026-10-03。此页按已有源码和实测记录盘点；已删除五个无引用的早期独立脚本；尚未移动其他源码或推送远程。

## 1. 当前规模与结论

- `scripts/`：229个文件，211 Python、5 C、4 C++、1头文件、3汇编、3链接脚本、2 Shell。
- `qvla_haq/`：15个核心Python模块；`tests/`：5个测试文件；`docs/experiments/`：90个记录/索引文件。
- 已完成数据与基线、硬件精度/成本探测、RL候选搜索、教师动作蒸馏、混合精度QAT及独立PTQ、真实低比特权重重载、ONNX/RKNN/RKLLM适配、板端完整推理和配对评测。
- 模型部署不是仅加载：输入原始两路图像/state/语言，板端执行视觉、语言K/V、10步动作专家和后处理，完成真实LIBERO闭环。
- 当前选定部署为原始FP16视觉＋V1无教师QAT权重的RKLLM W8A8语言＋混合RKNN专家；不是原V1 HAQ精度图的严格等价转换。训练/本地pack与后端原生量化语义有区别。

## 2. 按功能分类：主代码与实际完成内容

| 功能 | 重点实现/入口 | 已有结果与适用范围 |
| --- | --- | --- |
| 数据与原始基线 | `download_and_upload.py`、`split_libero.py`、`fp_baseline.py`、`prepare_libero_sim.py`；`qvla_haq/offline_actions.py` | 版本/模型/processor锁定，episode隔离，离线动作缓存，LIBERO仿真准备；不能把旧历史FP成绩代入新协议 |
| GPU真实整数推理/PTQ | `real_int8_linear.py`、`mixed_int8.py`、`ptq_mixed_int8.py`、`pack_mixed.py`、`pack_real_w8a8_expert.py`、`pack_v2_ptq_control.py` | INT8权重/激活、INT32累加、通道scale/仿射zero point；独立原FP PTQ与QAT pack严格重载；A10整数原型未取得速度收益 |
| 硬件精度与成本表 | `probe_precision_support.py`、`probe_rknn_numeric_formats.py`、`build_hardware_tables.py`、`benchmark_cost_compile.py`、`benchmark_cost_board.py`、`prepare_supplemental_hardware_tests.py`；`docs/hardware/` | 100个基础签名三轮板测＋18补充案例；精度支持依shape/后端/接口判断，查表成本不是完整策略延迟 |
| HAQ风格RL精度搜索 | `qvla_haq/{search_space,policy,reward,feasibility_reward,action_quality,runtime}.py`；`build_haq_search_space.py`、`run_haq_local_loop.py`、`run_haq_200_trial.py`、`run_haq_adjusted_trial.py`、`compare_haq_rl_random.py`、`freeze_haq_candidates.py` | 304可调位点，多候选掩码、自回归REINFORCE控制器、真实动作评价、体积硬约束、板测查表反馈、熵探索/平台早停、等预算随机对照、更新审计、V1/V2冻结；不是HAQ原论文DDPG逐项复现，也没有逐候选全策略板端反馈 |
| 教师蒸馏与冻结训练 | `qvla_haq/{distillation,training_scope,training_control,training_development,module_recovery}.py`；`prepare_openvla_teacher_inputs.py`、`cache_openvla_teacher.py`、`review_teacher_labels.py`、`audit_openvla_teacher_labels.py`、`diagnose_expanded_distillation.py`、`run_module_recovery_panel.py` | OpenVLA-OFT教师真实推理，离线动作缓存、有效horizon/时间步mask、标签审查；GT＋教师flow loss；冻结hash、模块恢复、部分专家训练与学习率对照。最终学生来自前8层蒸馏；没有证明蒸馏总体提高任务成功率 |
| 混合精度QAT | `qvla_haq/qat.py`、`fp_checkpoint.py`；`qat_train_haq.py`、`run_first8_v2_qat.py`、`run_qat_no_teacher_comparison.py`；`config/qat_distilled_*.json` | STE、浮点master、整数前向/代理梯度、冻结/解冻、warmup/cosine、梯度累积、训练快照/重载，有/无教师损失及V1/V2对照。选定QAT为V1无教师损失、1000步、all_sites；“蒸馏只训前8层”不等于“QAT只量化前8层” |
| 导出、校准与RKNN分图 | `export_selected_qat_rknn.py`、`verify_selected_qat_onnx.py`、`compile_selected_qat_rknn.py`、`split_v1_projection.py`、`prepare_v1_partition_calibration.py`、`compile_v1_isolated_projection.py`、`run_v1_partition_conversion.py`、`build_v1_partition_manifest.py` | 图像/语言/专家切分、ONNX边界验证、隔离校准、保留FP16/BF16/INT16例外的分图接口；BOOL边界恢复，BF16 native I/O适配；不把GPU pack直接当RKNN文件 |
| 板端完整推理 | `smolvla_board_runtime.py`、`smolvla_board_preprocess.py`、`smolvla_numpy_glue.py`、`smolvla_rknn_partitions.py`、`rknn_bf16_projection.c`/`smolvla_bf16_projection.py`、`serve_smolvla_board_stdio.py` | 两路RGB、tokenizer/state归一化、视觉/语言/专家/KV接口、10步Euler动作、后处理、真实NPU子图及CPU胶水；严格shape/hash/有限性检查 |
| RKLLM语言适配与一致性修复 | `export_smolvla_rkllm_language.py`、`prepare_rkllm_v1_calibration.py`、`compile_rkllm_full_prefill_probe.py`、`smolvla_rkllm_worker.cpp`、`smolvla_rkllm_backend.py`、`rkllm_native_aligned_mask.s/.ld`、`patch_rkllm_noncausal_probe.py`、`patch_rkllm_block_mask_probe.py` | 多模态prefix全量prefill、block mask/state token适配、160-token对齐、原生KV缓存回读解析；定位工作池cached/WC映射一致性问题；A→B→A重复漂移修复。改动绑定已核验RKLLM1.3.1二进制hash/偏移，不能称通用官方SDK支持或已合入上游 |
| 全链路评价与审计 | `run_smolvla_board_libero.py`、`benchmark_smolvla_board_resources.py`、`run_final_model_comparison.py`、`report_final_model_comparison.py`、`summarize_smolvla_board_screening.py`、`audit_final_model_evaluation.py`、`audit_final_model_preprocessing.py` | 固定初始状态/噪声的40任务配对闭环；两版同板P50/P95；PPid递归进程树PSS/RSS/CPU/NPU/温度/频率；视频和输入/动作原始证据；失败冷启动复现 |

这些文件都在当前 `scripts/` 平铺目录，包模块在 `qvla_haq/`。各模块是共享依赖，不能仅按脚本前缀判断可删除性。

## 3. 哪些适合上传Git

### 主交付：应上传

1. 自写源码：`qvla_haq/`、主链路及其依赖的Python/C/C++/汇编/链接脚本，先保留当前路径保证import有效。
2. 配置：候选V1/V2、搜索空间、数据分区/版本锁、蒸馏/QAT配置、依赖锁；终版部署小manifest另导出为可分发配置，避免强依赖被忽略的runs目录。
3. 文档：新主README、当前量化路线、硬件格式/成本表、[终版对比](final-model-comparison.md)、[持续测试总表](model-test-results.md)、关键实验与失败修复记录。
4. 实验复现证据：将必要的小JSON/CSV、模型hash、配置/样本身份和统计结果从 `runs/` 提取到 `docs/results/`。此目录目前尚未建立；不要上传整个runs来替代结果整理。
5. 自写测试与必要的结果图；保留原始数据来源说明，图须来自实测。
6. `.gitignore`；`AGENTS.md`可作为开发约定保留，`info.md`可作学习附录。

### 研究过程：适合归档后上传

- KL/MMSE/clip扫描、敏感度测试、局部QAT/PTQ、奇数层K/V导出修正等，保留作为方法与失败边界证据。
- RKLLM dump/MatMul trace/工作池定位探针和历史构建变体；主入口只展示当前有效路径，诊断单独导航。
- 无效或退步实验的配置、说明、短结果可以保留；巨大的失败权重和调试dump不进入Git。
- 不根据 `probe_` 名称直接归档：例如当前使用的 `compile_rkllm_full_prefill_probe.py` 是终版编译入口之一。

首版可以提交完整自写源码＋索引，让实验可追溯；无需为“代码整洁”丢弃有效复现依赖或失败证据。

### 不上传Git

- `artifacts/`、`data/`、`runs/` 内的模型/数据集/完整缓存、教师标注、NPZ大激活、视频、日志和训练断点。
- `.venv*`、wheelhouse、官方SDK/动态库/编译二进制、第三方模型repo和第三方源码副本；改为版本及安装/下载说明。
- Toolkit临时 `check*.onnx`、各ONNX/RKNN/RKLLM/Safetensors/PT文件、`__pycache__`。
- 模型单独作为外部下载产物管理，并带hash与安装说明；当前终版主要在板端 `/dev/shm`，尚不是可长期存放/一键分发的release包。

现有.gitignore已屏蔽上述主要大目录和根目录check临时图；正式提交前应确认新增环境/临时文件仍被排除。

## 4. 整理时优先解决的实际问题

1. **README与简历文档过期**：现在仍写“完整模型未板端执行、RL尚未运行”，必须替换成当前事实，并区分早期112专家方案、本地RL候选、GPU QAT/PTQ和终版原生后端适配。
2. **结果依赖被忽略目录**：目前Markdown大量引用 `runs/`，只上传代码读者看不到关键证据；需要小结果快照及终版manifest。
3. **可迁移配置不足**：若干入口写死板子IP、`/root/qvla`、具体run目录、runtime偏移。提取配置/参数，runtime patch继续保留版本/hash约束；把个人机器路径改成可填配置。
4. **不能直接批量搬文件**：源码AST盘点有94个脚本对其他平铺脚本的import语句，多处 `ROOT=Path(__file__).parents[1]`。移到二级目录会改变import与ROOT，须分批修正并验证主入口。
5. **环境与部署包**：已有teacher requirements/lock，但缺少统一主链路环境说明、训练/转换/板端环境的清晰分组和终版部署资产准备入口。当前运行脚本常依赖历史目录/符号链接，需要补齐安装与打包流程。
6. **代码状态与模型状态分开**：C/C++/汇编修复有效但版本绑定；不能写已获官方支持/已提交PR。正式搜索逐候选整策略硬件反馈未接通，搜索得到的配置不保证后端等价。

建议目标导航：`data → hardware → search → distill/train → convert → deploy → eval`；诊断与历史实验另列入口。先补文档/结果快照，再分批搬代码，不修改数值实现或更换所选模型。

## 5. 简历可写的四个核心点

项目名称：**面向RK3588的SmolVLA硬件感知混合精度量化与部署**。

可用表述：

- 实现基于HAQ思路的混合精度RL搜索，覆盖304个算子位点，结合真实动作反馈、模型体积约束与RK3588实测成本查表；实现REINFORCE、熵探索、平台早停和等预算随机对照，冻结V1/V2候选。
- 构建OpenVLA-OFT教师离线动作蒸馏、标签有效时间步过滤与部分参数冻结流程，完成蒸馏学生的混合精度QAT及独立PTQ对照，实现真实整数权重打包、严格重载及闭环质量评估。
- 完成RK3588整策略部署：RKNN视觉/动作专家＋RKLLM W8A8语言＋CPU处理，适配多模态prefix、KV缓存、BF16 native I/O及跨子图dtype，定位并修复NPU工作池缓存一致性造成的重复推理漂移。
- 建立40任务配对闭环和整策略资源评测；核心参数相对原checkpoint缩小35.15%至588MB，同板进程树峰值PSS降低38.07%至1130MiB（含RKLLM worker），完整动作块P50由7.52s降至7.26s。

若只写3条，可以将蒸馏/QAT/PTQ和RL搜索合并。速度提升只有本次约3.4%，优先突出真实部署、内存与可复现工程能力，不包装成大幅加速。

## 6. 简历数值及必须保持的口径

| 可用事实 | 数值/限制 |
| --- | --- |
| 终版核心参数文件对原checkpoint | 906.71→588.04MB，减少35.15% |
| 同板部署文件比较 | 945.56→588.04MB，减少37.81%（不同于checkpoint比较） |
| 同板采样峰值进程树PSS | 1824.4→1129.9MiB，减少38.07%；含RKLLM worker，不含未映射驱动DMA |
| 同输入完整动作块P50 | 7.5199→7.2626s，warmup3/repeats20；生成50步动作，不是单个机器人控制步 |
| Object类别成功数 | 原GPU/板端终版各8/10，单回合80% |
| Goal类别成功数 | 原GPU/板端终版各7/10，单回合70% |
| 全部40任务 | 原GPU30/40，终版25/40；6个改善、11个回归，空间关系/长任务主要下降 |
| 搜索成本 | 100基础配置三轮板测＋18补充；终版语言后端变化，旧RKNN签名不能等同其RKLLM真实开销 |

不能写“总体成功率不变”“无损量化”“蒸馏提高总体成功率”“RL找到全局最优”“逐候选板端HAQ闭环已经完成”“终版严格保留V1精度图”“推理加速数倍”。类别持平不能冒充整体持平；不把历史112层专家31/40与当前终版588MB拼成同一模型结果。

GPU原型使用真实整数GEMM，本地RL的多格式数值参考与真实RKNN/RKLLM产物要分开说明；QAT训练中fake quant只是训练机制，不是最终板端量化证明。

## 7. 下一步整理顺序

1. 更新README和 `project-portfolio.md`，用当前交付和证据替换旧状态。
2. 建立 `docs/results/` 保存公开可复核的小结果，整理有效主配置和终版资产manifest。
3. 增加主链路入口导航、环境/安装/运行说明，将地址与路径参数化。
4. 再按功能分批迁移源码、修正依赖；保留历史实验索引。其余源码暂不删除、不推送。

## 8. 删除与归档审查

检查范围为脚本间AST导入/文件名引用，以及 `qvla_haq/`、`tests/`、配置、README和文档引用。未被引用不等于无用，独立CLI入口仍需结合当前流程判断。用户确认后已删除下表五个脚本；其他项仍为审查建议。

### 可以清理的生成文件

- 根目录 `check0_base_optimize.onnx`、`check1_fold_constant.onnx`、`check2_correct_ops.onnx`、`check3_fuse_ops.onnx`：Toolkit中间图，合计1,611,731,082字节（约1.61GB），已被Git忽略；保留源模型和最终部署资产。
- `__pycache__/` 和 `.pyc`：可重新生成。

### 已删除的早期独立脚本

以下五个文件在上述范围内没有发现引用（本清单除外），合计约16.6KB。已按用户要求删除；不再保留这些早期独立探针入口，未发现当前主流程依赖。

| 文件 | 作用与删除理由 |
| --- | --- |
| `scripts/qat_probe.py` | 最早的单批TorchAO QAT兼容性探针，已由真实QAT训练入口替代 |
| `scripts/smoke_fp.py` | 最早单观测FP冒烟测试，使用旧单图像样本格式 |
| `scripts/prepare_sample.py` | 为旧冒烟测试提取首条样本；当前校准与动作缓存另有入口 |
| `scripts/stage_offline_eval_model.py` | 创建离线模型目录并重写路径；当前加载器直接覆盖资产路径 |
| `scripts/compile_rknn_qat_torchscript.py` | 早期TorchScript INT8兼容性探针，含torch版本绕过；当前转换走ONNX路径 |

### 先归档或保留

- `qat_train.py`、`pack_mixed.py`、`eval_quantized_libero.py` 等旧量化路径：有历史复现文档或代码依赖，先处理引用再退役。
- 历史敏感度、KL/MMSE扫描、RKLLM dump/trace探针：归入诊断/历史导航，保留方法与失败证据。
- `convert_smolvla_vision_rknn.py`：终版原始FP16视觉的复现转换入口，保留。
- `compile_rkllm_full_prefill_probe.py`：当前有效语言转换入口，保留。
- 当前RKLLM worker、mask汇编/链接脚本、BF16接口与板端runtime：有运行或构建依赖，保留。

仓库当前源码尚未形成可恢复的Git提交；真正删除源码前应先建立本地提交或备份。无需为清理而推送远端。

## 9. 为什么仍有大量文件

剩余229个脚本文件并非全部是终版运行必需。按第2节列出的61个入口做AST导入和显式文件名引用闭包，涉及77个文件；余下152个未纳入该主功能入口集合。其中25个是绘图入口、38个是按前缀初筛的探测/诊断入口，另外89个包含历史实验编排、独立验证、准备与汇总工具。分类是整理依据，不是删除判定；主清单也包含历史对照，板端运行实际所需文件更少。

完整逐文件初筛见 [源码用途索引](code-role-index.md)。后续应合并重复实验编排和绘图框架、保留当前训练/转换/运行与重要复现入口，再对历史独立CLI逐项退役。不能仅按没有被import或名字带probe来删。
