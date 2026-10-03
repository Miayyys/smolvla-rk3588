# V1无教师损失QAT：RK3588转换与部署验证

## 所选模型与目标

用户选用蒸馏学生→HAQ v1固定图→无额外教师损失QAT的最终模型，要求转换并尝试在RK3588运行。当前本地诊断19任务14/19、新状态12任务8/12；这些成绩不能直接归给RKNN转换后的模型。

- 原始checkpoint SHA256：`9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`。
- QAT FP master SHA256：`4aeb92854d2bb89bac84a2d791d2acb4389b934178948c05533d6a52fe0b9f81`。
- 实际本地pack SHA256：`0b1fd3ece496b777fb145e33ac03e51e14c7205a55fb246250707b6a77c29474`，504,284,536 B。
- 300个W8A8节点，专家第3层V投影FP16、专家第10层gate投影BF16、语言第3层down投影INT16 DFP、词嵌入CPU逐行INT8。具体assignment保留在 `config/haq_candidate_v1.json`。

## 实施方法与精度边界

采用此前已验证接口：两路视觉+connector RKNN、16层语言前缀输出32个KV RKNN、16层动作专家与动作/时间投影RKNN；CPU执行词嵌入、状态投影、前缀组装、时间正弦、10步Euler与反归一化。

从所选QAT **FP master**导出ONNX，再使用独立校准数据做RKNN训练后校准/转换；没有对已量化pack重复PTQ。CPU词嵌入及状态投影直接保存实际pack的INT8值和scale/zero/bias，避免CPU部分退回原始权重。RKNN融合/输出量化与本地逐节点quantizer不同，不能预设量化参数完全一致，转换后必须比较完整动作。

RKNN混合量化通过官方step1产生profile，再按精确权重值匹配的ONNX输出定位高精度例外，不按模糊层号匹配。实际混合custom配置拒绝BF16；专家gate明确适配为FP16。语言down投影编译为INT16，但与本地DFP quantizer的等价性未证明；单节点支持不能直接推断融合图支持。官方[混合量化示例](https://github.com/airockchip/rknn-toolkit2/blob/master/rknn-toolkit2/examples/functions/hybrid_quant/step1.py)说明custom配置按输出tensor指定dtype；最终以实际2.3.2编译报告为准。如采用任何精度映射替换，会独立记录为后端适配候选，不能称原V1数值等价。

## 校准与回放隔离

沿用所选QAT报告中的40个独立 `ptq_calibration` episode（每任务1个），每episode第0帧。捕获80路图像、40个前缀输入、120组专家输入（去噪步0/5/9）。精确episode/task/frame、噪声seed、输入文件及源hash保存在服务器 `runs/qat_v1_rknn_deploy_v1/export_report.json` 和三份dataset.txt。不是随机校准输入，不使用本次开发回放输入校准。

开发回放为episode18/task0/frame0，固定原始噪声seed2416662958、语言长度48、前缀177、chunk50、去噪10步。`replay_inputs.npz`沿用已核对的处理后输入（SHA256 `62ee6635d9cca02f74e6aff59b733d575299e5d62eccb2d5d01ff2c13c7fb427`）。重算所选实际GPU整数pack的完整动作作参考，不拿原FP冒充所选模型。

## 已实际验证

1. 三个ONNX图已导出，ONNX checker通过。ONNX Runtime相对对应Torch FP32 master子图共35个输出（视觉1、前缀33、专家1）全部通过rtol/atol1e-4；最大绝对差异9.1552734375e-5。不是RKNN或任务质量验证。
2. 专家图继续使用已验证的单调位置ReduceMin→Gather等价替换，避免旧板端INT64 ReduceMin执行失败。
3. CPU逐行INT8词嵌入和W8A8状态投影相对真实GPU pack前缀组装逐元素一致（max_abs0）；旧FP NumPy glue回归最大误差5.9604645e-8，保持原有行为。
4. 所选GPU pack回放动作参考已生成，CPU参数/输入/参考hash均锁定在 `replay.json`。完整RKNN转换与板端输出已测，结果见下表。
5. 板子10.42.0.252已确认在线，需 `ssh -o ProxyCommand=none` 绕开当前代理；4GB内存、driver0.9.8，原始三图runtime基础存在。主机和服务器Toolkit2 2.3.2。
6. 三图编译和板端完整回放完成。高精度输出配置曾只产生INT8计算后转换，已修正专家投影的输入/输出配置；编译日志确认两个专家例外Conv为FLOAT16、语言down Conv为INT16。最终专家为 `expert_selected_corrected.rknn`，不能使用早期仅输出转换的产物。
7. BF16 custom尝试失败已保留，不据此推断全图BF16不支持。修正后编译曾报告 `Unkown op target:0`，随后返回成功；最终板端日志没有 `E RKNN`。这些边界和实际精度映射保存在编译报告。

## 脚本与证据

- `scripts/export_selected_qat_rknn.py`：来源校验、三图导出、精度例外定位、40任务隔离校准输入、实际pack CPU参数。
- `scripts/verify_selected_qat_onnx.py`：ONNX与master子图35输出核对。
- `scripts/prepare_selected_qat_board_reference.py`：严格重载所选pack、真实GPU动作参考、CPU量化前缀验证。
- `scripts/compile_selected_qat_rknn.py`：原生W8A8与显式混合profile；BF16→FP16只能通过显式参数另作适配实验。
- `scripts/smolvla_numpy_glue.py`：新增INT8 embedding和state路径，保留旧FP路径。
- 原始产物/日志：服务器 `/root/qvla/runs/qat_v1_rknn_deploy_v1`；本地同名runs保存小报告。首个导出因服务器缺少旧helper而失败，保留 `export_missing_helper.log`，补传代码后通过。

## 板端完整回放实测

同一原始观测、固定初始噪声；NPU_CORE_0，warmup1/repeats3。包括板端图像/文本/状态前处理，两次视觉、一次语言前缀、十次专家去噪及CPU后处理。原FP16基线本次重新测量。三次重复输出完全一致，均为有限值，输出 `[1,50,7]`。

| 指标 | 原FP16部署基线 | 所选QAT的RKNN适配产物 |
|---|---:|---:|
| 完整回放p50 | 8047.19 ms | 6245.06 ms |
| p95（仅3次，描述性） | 8216.36 ms | 6314.11 ms |
| 进程maxRSS | 1,898,036 KiB | 1,043,496 KiB |
| 三图+CPU参数实际字节 | 945,564,365 | 448,520,196 |
| 动作MAE | 0.002013，相对原FP | 0.192045，相对所选GPU整数pack |
| 动作最大绝对差异 | 0.012950 | 1.818409 |
| 夹爪符号不一致 | 0/50 | 7/50 |

注意两列动作误差各自参考不同；量化列测后端转换误差，不能据此直接计算相对原FP的质量下降比例。三图+CPU参数相对原部署减少52.57%，延迟减少22.39%，RSS减少45.02%。包含tokenizer/processor/配置/脚本的自包含运行资产452,136,998 B（452.14 MB十进制），相对原checkpoint906,712,520 B减少50.13%；不含测试输入、报告、共享RKNN库。实际文件表保存在 `package_inventory.json`。

执行通过不代表动作质量通过。7/50夹爪符号发生改变且动作误差显著，**当前不能作为质量合格的端侧模型**，未测闭环任务成功率，不沿用本地14/19或8/12。

### 初步误差定位与常驻接口

`boundary_error_audit.json`：两路视觉特征相对GPU pack的MAE6.409431/RMSE8.576339；CPU组装前缀MAE143.612364，包含视觉误差经sqrt(width)缩放后的影响，并非语言前缀RKNN输出误差。不能用它单独归因语言层。

`vision_error_audit.json`：第一路视觉相对同一FP32 QAT master，板端MAE6.270622，而GPU pack MAE0.489853。误差在视觉阶段已明显出现，仍不能排除后续语言和专家的独立误差。下一步只转换同一master视觉为FP16对照，区分原生INT8量化与其他后端差异；这是诊断候选，不是改称原HAQ配置通过，结果尚未测量。

常驻 `BoardSmolVLA.predict(raw)` 已在板端运行一次，与CLI回放动作逐元素相同（max_abs0）。单次常驻调用5201.58ms仅用于接口一致性验证，不与三重复p50混用。

### 产物与运行

板端目录 `/root/qvla_board_test/qat_v1_no_teacher_v1/`；manifest固定三图SHA及CPU参数SHA。本地 `runs/qat_v1_rknn_deploy_v1/` 保留报告、输出NPZ、图及日志，均被Git忽略。

```bash
ssh -o ProxyCommand=none root@10.42.0.252
cd /root/qvla_board_test/qat_v1_no_teacher_v1
env OPENBLAS_NUM_THREADS=1 PYTHONPATH=/root/qvla_board_test/python_site python3 -u rknn_board_full_replay.py --vision vision_selected.rknn --prefix prefix_selected.rknn --expert expert_selected_corrected.rknn --inputs raw_inputs.npz --weights cpu_weights.npz --reference fp_reference.npz --config replay.json --preprocessor-assets . --preprocessor-manifest preprocess_export.json --output board_raw_report.json --warmup 1 --repeats 3
```

最终三图SHA：视觉 `bc042d14dced2c61176e9e020a250ccf469a58931c8ec2791d06c7560ae03f92`；前缀 `685a095999b45b88efd396f0a778151ca481882dd3ec515dcf68f2270089b400`；专家 `7617eb65f2f2e0af50924fbff2a6c318f522d06245f115a1a1b911bd839c2090`。动作参考SHA `946e12c7562a17fc717170ee23ea27061d84fdd75965d9ddf6d3df765d5015ac`。

## 同权重FP16视觉对照（已完成）

`compile_selected_qat_rknn.py --graph vision --mode fp16` 从同一SHA的ONNX生成不量化的视觉图，独立目录 `vision_rknn_fp16_diagnostic`，不会覆盖所选混合图。编译28.88秒，产物212,621,173 B，SHA `e1e14945aaa611aecce3d90bdb7c76ae9eeccdf5ea91a1b10bfa0c4a838f21ae`。已完成同输入板测，warmup1/repeats3。只替换视觉，CPU/语言/专家和噪声固定；它是误差定位实验，不代表保留原v1位宽图或质量通过。

| 同权重对照指标 | 原RKNN INT8视觉 | 改为FP16视觉 |
|---|---:|---:|
| 第一路视觉相对FP32 master MAE | 6.270622 | 0.156662 |
| 第一路视觉RMSE | 8.193296 | 0.233465 |
| 完整动作相对所选GPU pack MAE | 0.192045 | 0.124975 |
| 夹爪符号不一致 | 7/50 | 7/50 |
| 完整raw回放p50 | 6245.06ms | 6907.86ms |

视觉误差降低97.5%，动作MAE降低34.9%。这说明同一视觉权重的原生INT8转换明显贡献了误差；但不区分校准截断、融合量化边界、算子实现等更细根因，仍不能称特定quantizer已被唯一定位。仅修复视觉并未恢复最终动作，语言/专家的独立误差与后端精度规则差异待逐子图固定输入对照；不得把剩余误差全部归因某个尚未测量模块。FP16视觉也不等于GPU pack视觉，仅是一项可控替换实验。原候选图与manifest不变。证据 `vision_fp16_diagnostic_full_report.json/.npz`、`vision_fp16_diagnostic_summary.json`、完整console log。
