# 16 个专家 MLP：QAT/PTQ 批量 RKNN 编译

**范围**：16 个专家层各自的 MLP，合计 48 个 Linear；原始 FP 独立 PTQ 与 QAT 第 50 步 FP 主权重两条路径各 16 个 RK3588 W8A8 `.rknn`。32 个子图均已编译，不等于一张完整专家图或板端可运行。

## 原理与固定配置

本实验延续[第 0 层真实 MLP QAT/PTQ 配方](2026-09-27-stage1-rknn-qat-ptq.md)：`target_platform=rk3588`、`quantized_dtype=w8a8`、`quantized_method=channel`、`quantized_algorithm=mmse`、optimization level 3。QAT 路径的 FP 主权重先由已完成的 STE QAT 微调产生，再用与 PTQ 相同的独立激活集校准后**转换**；PTQ 路径从锁定的原始 FP checkpoint 独立开始。两者没有低比特模型重复 PTQ。

[`capture_all_expert_mlp_inputs.py`](../../scripts/capture_all_expert_mlp_inputs.py)一次模型加载采集 16 层：40 个 `ptq_calibration` episode 每任务 2 帧、每帧取 3 个去噪步骤，即每层 240 条 `1×50×720` 校准激活；另在 40 个互不重叠的 `qat_train` 开发 episode 采同样 240 条作后续数值评测，冻结测试未使用。[`export_all_expert_mlp_qat_ptq.py`](../../scripts/export_all_expert_mlp_qat_ptq.py)一次模型加载导出 32 份 FP32 ONNX，逐图通过 ONNX checker 与参考前向。第 0 层的 PTQ/QAT ONNX 与旧实验各自哈希相同，240 条校准激活也逐条哈希相同，复用已验证的两份 `.rknn`；其余 30 份由[`compile_all_expert_mlp_qat_ptq.py`](../../scripts/compile_all_expert_mlp_qat_ptq.py)以 4 个 CPU 进程并行 MMSE 编译，可按哈希恢复中断任务。服务器为 8 核、29 GiB RAM、A10 23 GiB；编译主要占 CPU，不把 GPU 显存占用率当成功指标。

服务器 `/root/qvla` 下的三步命令（采集时两条 split 可并行）：

```bash
.venv/bin/python scripts/capture_all_expert_mlp_inputs.py --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets --dataset-root data/libero --splits data/libero_splits.json --partition config/evaluation_partition_v2.json --output-dir runs/expert_mlp_all_calibration_v1 --split calibration --frames-per-task 2 --step-samples 3
.venv/bin/python scripts/capture_all_expert_mlp_inputs.py --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets --dataset-root data/libero --splits data/libero_splits.json --partition config/evaluation_partition_v2.json --output-dir runs/expert_mlp_all_development_v1 --split development --frames-per-task 2 --step-samples 3
.venv/bin/python scripts/export_all_expert_mlp_qat_ptq.py --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets --qat-snapshot runs/qat_w8a8_stage1_lr1e7/expert_master_step_50.safetensors --partition config/evaluation_partition_v2.json --output-dir runs/expert_mlp_all_export_v1
.rknn-probe/bin/python scripts/compile_all_expert_mlp_qat_ptq.py --export-dir runs/expert_mlp_all_export_v1 --calibration-dir runs/expert_mlp_all_calibration_v1 --output-dir runs/expert_mlp_all_rknn_v1 --workers 4
```

## 结果与限制

[逐文件编译及 240 条留出输入数值报告](../../runs/expert_mlp_all_rknn_v1/report.json)有 32/32 `success`；每份 `.rknn` **4,524,893 B**，PTQ 与 QAT 各 16 份分别合计 **72,398,288 B**。该合计仅是分离子图文件大小相加，含重复图元数据，**不是整专家或整模型体积**。第 0 层两份实际导出的文件 SHA-256 分别是 PTQ `9b8a2dd8aedd928512d525848415eee48d267b62551dde9ae25b765669c26839`、QAT `2cdfef74ce6d632af581106e14fcee9fa794e4acb17273543d47b216d74ba659`。

逐层主机模拟器输出对原始 FP32 ONNX 的开发集平均 MAE 为 PTQ **0.006688**、QAT **0.006636**；QAT 在 16 层中的 10 层 MAE 较低，PTQ 在 6 层较低。层间范围分别为 PTQ `0.004643–0.010703`、QAT `0.004637–0.010705`。差异较小且随层反转，说明这批数据不能支持“QAT 全面优于 PTQ”或据此确定逐层最终位宽。[完整逐层图](../../figures/expert_mlp_all_rknn_v1.png)与[汇总数值](../../figures/expert_mlp_all_rknn_v1.json)保存了全部候选；本地原始报告为[`runs/expert_mlp_all_rknn_v1/report.json`](../../runs/expert_mlp_all_rknn_v1/report.json)。

编译成功和 W8A8 构建配方不替代逐图算子表核验，也不证明每个节点均由 NPU INT8 执行。主机模拟器的 MAE 不等于完整动作质量；GPU 上 112 Linear 的真实 INT8 动作对照另见[整专家真实量化实验](2026-09-27-full-expert-real-w8a8.md)。本批次未构成完整专家图，整图 RAM、闭环成功率和端到端收益函数 $G$ 均未由这些子图结果测量。

用[板端 smoke 脚本](../../scripts/rknn_board_subgraph_smoke.py)已将一条隔离开发输入和第 0 层 PTQ/QAT `.rknn` 搬到真实 RK3588 上做 Lite2 推理；两个子图均运行成功。因 Lite2 固定从 `/usr/lib/librknnrt.so` 加载，测试用私有 mount namespace 临时映射 2.3.2 runtime，没有覆盖板端 1.4.0 系统库。实测配置、延迟、误差与限制见[板端记录](2026-09-27-rknn-board-subgraph.md)。其余专家 MLP 和完整模型板端执行仍未测。
