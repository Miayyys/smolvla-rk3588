# 更正奇数层 K/V 独立输入后的 RKNN W8A8 评测

**状态**：32/32 个子图已编译并完成 RKNN 主机模拟器留出数值评测；板端执行尚未测。此实验取代旧 K/V 图上的误差排序，但保留旧记录以说明错误与修复过程。

## 问题与更正

原[16 层注意力扫描](2026-09-27-all-expert-attention-rknn.md)将奇数层 K/V 组成一个双输出图，却误把 K 的输入同时送给 V。逐层 hook 已确认：8 个奇数层的 240/240 条样本中，实际 `k_proj` 与 `v_proj` 输入均不同。因此旧图中奇数层 V 数值、相关排序和位宽建议无效。修正版将 K、V 导出为两个独立模型，分别使用实际输入；每条 `.rknn` 的结果再按文件名与原始加载 BF16 策略的对应输出比较。

## 数据、配置与计算

- 原始 SmolVLA checkpoint SHA-256：`9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`。
- 固定 episode 分区 SHA-256：`f755546a6b074d2fe248333fc42c3dbf30f9b54b9f0fa0b58a16506a54d942c2`。校准来自 40 个 `ptq_calibration` episode，开发来自不重叠的 40 个 `qat_train` episode；每个奇数层 K/V 分别 240 条输入。冻结测试未参与。
- 已加载 BF16 模型上实际 K/V 输入和输出的 1,920 条记录，以及输入逐样本哈希见服务器 `runs/expert_odd_kv_inputs_v2/report.json`；该报告 SHA-256 为 `5395b667b85832f9ec468a7da7ce491cb11de520a73f1907624e7d91f2f39b0a`。8 个奇数层各有 240/240 条 `kv_inputs_equal=false`。
- K 使用 `expert_attention_all_calibration_v3` / `expert_attention_all_development_v3` 中采集的真实 K 输入；V 则由 `capture_all_expert_attention_inputs.py --odd-v-only` 单独采集，另与加载模型 hook 得到的 V 输入核对。K/V ONNX 也分开导出，避免图结构复用错误输入。
- 两条路径均由原始 FP checkpoint 或 QAT 第 50 步浮点主权重开始；量化均在 RKNN 编译时进行：`target_platform=rk3588`、W8A8、`quantized_method=channel`、`quantized_algorithm=mmse`、optimization level 3。QAT 经过训练后重新校准并转换，PTQ 从原始 FP 权重独立校准转换，没有对低比特模型二次 PTQ。
- 每个子图在 RKNN host simulator 上运行 240 条独立开发输入。主要指标为逐样本有效 token 上，模拟器输出与原始已加载 BF16 投影输出的平均绝对误差：

$$E_{l,k,m}=\frac{1}{240}\sum_{i=1}^{240}\operatorname{mean}_{t<c_i,f}|Y^{\mathrm{RKNN}}_{l,k,m,i,t,f}-Y^{\mathrm{loaded\ BF16}}_{l,k,i,t,f}|,$$

其中 $l$ 为奇数专家层、$k\in\{K,V\}$、$m\in\{PTQ,QAT\}$，$c_i$ 为该输入原始 token 长度。主机模拟器使用与 `.rknn` 编译相同的 Toolkit2 build 进程；这不是板端精度或完整动作指标。

复现的关键命令（服务器 `/root/qvla`）：

```bash
.venv/bin/python scripts/capture_all_expert_attention_inputs.py --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets --dataset-root data/libero --splits data/libero_splits.json --partition config/evaluation_partition_v2.json --output-dir runs/expert_attention_odd_v_calibration_v4 --split calibration --frames-per-task 2 --step-samples 3 --odd-v-only
.venv/bin/python scripts/capture_all_expert_attention_inputs.py --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets --dataset-root data/libero --splits data/libero_splits.json --partition config/evaluation_partition_v2.json --output-dir runs/expert_attention_odd_v_development_v4 --split development --frames-per-task 2 --step-samples 3 --odd-v-only
.venv/bin/python scripts/capture_expert_kv_loaded_outputs.py --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets --dataset-root data/libero --splits data/libero_splits.json --partition config/evaluation_partition_v2.json --development-dir runs/expert_attention_all_development_v3 --output-dir runs/expert_odd_kv_inputs_v2
.venv/bin/python scripts/export_odd_expert_kv_separate.py --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets --qat-snapshot runs/qat_w8a8_stage1_lr1e7/expert_master_step_50.safetensors --partition config/evaluation_partition_v2.json --output-dir runs/expert_odd_kv_export_v4
.rknn-probe/bin/python scripts/compile_correct_odd_expert_kv.py --export-dir runs/expert_odd_kv_export_v4 --k-calibration-dir runs/expert_attention_all_calibration_v3 --v-calibration-dir runs/expert_attention_odd_v_calibration_v4 --k-development-dir runs/expert_attention_all_development_v3 --v-development-dir runs/expert_attention_odd_v_development_v4 --loaded-dir runs/expert_odd_kv_inputs_v2 --output-dir runs/expert_odd_kv_correct_rknn_v4 --workers 2 --precision w8a8
```

## 结果、图和边界

32/32 K/V×QAT/PTQ 子图均成功。每个独立 `.rknn` 为 **418,591 B**；16 个图/格式各自合计 **6,697,456 B**，只是重复导出的独立子图，不是完整模型体积。

与加载 BF16 投影输出相比，K 子图平均 MAE 为 PTQ **0.0281652**、QAT **0.0280999**（QAT 在 8 层中 5 层较低）；V 子图分别为 PTQ **0.00756458**、QAT **0.00756461**（PTQ 在 8 层中 5 层较低）。QAT 与 PTQ 的优劣随层/投影变化，差值很小；不能仅凭这些子图 MAE 决定位宽或声称 QAT 总体更优。修正后第 1 层 V 的 MAE 约 `0.004159`，与原先错误图的约 `0.9108` 形成强烈差异，说明此次主要收益是恢复正确的模型语义。

[逐层 K/V 图](../../figures/expert_attention_kv_corrected_v4.png)、[聚合数值](../../figures/expert_attention_kv_corrected_v4.json)和[32 图构建/数值汇总及各模型 SHA](../../figures/expert_odd_kv_correct_w8a8_v4.json)保存所有开发集对照。逐样本原始报告、ONNX、激活和 `.rknn` 文件留在服务器 `runs/`。

结论范围仅是奇数层独立 K/V 投影子图的 W8A8 数值和图连接正确性。当前结论没有确认这些节点都由 NPU INT8 执行，也没有证明全模型动作质量、闭环成功率、板端延迟或内存改进。位宽选择仍须由完整动作/闭环质量约束及实际板端资源共同决定。
