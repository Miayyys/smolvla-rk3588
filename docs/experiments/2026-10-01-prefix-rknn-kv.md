# 前缀 prefill：RKNN 显式导出 16 层 K/V 并接回动作专家

## 问题与成功判据

SmolVLA 的动作专家读取全部 16 层前缀 K/V，不能只使用语言模型最后隐藏状态。本轮验证：前缀完整子图是否能在 RK3588 接收 embeddings、mask、positions 并返回 32 个 K/V 张量；能否在同一固定观测和噪声下将真实板端缓存接回原始动作专家。输出维度、有限值和精确输入身份是接口判据，任务质量没有在本轮设置通过阈值。

RKLLM 1.3.1 的[固定提交 C API](https://raw.githubusercontent.com/airockchip/rknn-llm/f7390530443bf84f0394255a449d7cbe81e69d1c/rkllm-runtime/Linux/librkllm_api/include/rkllm.h)有 `RKLLMCrossAttnParam`，它接收外部 encoder K/V；该接口没有给出语言前缀逐层 K/V 的输出方法。因此本轮用 RKNN 前缀显式输出，并不证明 RKLLM 在所有非公开或未来接口上无法适配。

## 配置与导出原理

模型/processor、固定开发观测和动作噪声与[拆分记录](2026-10-01-smolvla-deployment-split.md)一致：权重 SHA256 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`，episode 18、task 0、frame 0、种子 `2416662958`。前缀是从原始策略实际调用捕获的，不是合成输入。采集时完整动作与 FP 缓存逐元素相同。无训练、校准或位宽搜索；FP16 作为实际部署诊断起点，未锁定 HAQ 最终格式。

[`export_smolvla_prefix_onnx.py`](../../scripts/export_smolvla_prefix_onnx.py)捕获输入和原始 BF16 K/V，将同一加载权重转换为 FP32 计算后导出 ONNX。为了避免原始 RoPE 的切片赋值生成 ScatterND，用拼接表达相同运算：`cat(x1*cos-x2*sin, x2*cos+x1*sin)`；FP32 前向逐元素验证与原实现相同。首次导出曾因推理张量参与梯度保存失败，之后冻结权重、克隆输入并在推理上下文导出，日志保留在 `runs/prefix_export_console.log` 和 `runs/prefix_export_retry_console.log`。

| 接口 | 类型与形状 |
| --- | --- |
| 输入 prefix | FP32 `[1,177,960]` |
| 输入 attention_mask | BOOL `[1,177,177]` |
| 输入 position_ids | INT64 `[1,177]` |
| 输出 prefix_hidden | `[1,177,960]` |
| 输出 key_0/value_0 … key_15/value_15 | 各 `[1,5,177,64]`，共 32 个张量 |

输出的 last hidden 是诊断用参照，原始动作去噪主要读取逐层 K/V。ONNX 为 629,529,273 字节，SHA256 `2f6b94f47ec097ce69435402674352eb7829fcc966afc9bd70dbf8de70fb6a5d`，opset 17，无 ScatterND。ONNX Runtime 1.26.0 / CPU 对本轮 FP32 PyTorch 参考的 33 个输出全部 `allclose(rtol=1e-4, atol=1e-4)`，最大绝对误差 `2.52724e-5`。FP32 计算与原始 BF16 加载路径的差异另外记录，不将两种参考混为一谈；各输出 MAE 最大约 0.0111682。

RKNN Toolkit2 2.3.2 配置为 `target_platform=rk3588, float_dtype=float16, optimization_level=3, do_quantization=False`。编译 25.61 s；真实文件 326,154,086 字节，SHA256 `9812496c93b2125b7a3c72c5b69ec3c63ef6ccfeb9090f137e46c4004e80706c`。这里只比较前缀子图文件，不能当作全模型达到 40% 压缩的证据。

## 真实板端与动作接口结果

[`rknn_board_prefix_smoke.py`](../../scripts/rknn_board_prefix_smoke.py)在 runtime/Lite2 2.3.2、driver 0.9.8、NPU_CORE_0 上实测；BOOL 和 INT64 输入已成功接受，33 个输出 shape 全部匹配且有限。一次预热、两次计时为 606.350 / 593.546 ms，p50 599.948 ms；进程最大 RSS 757,856 KiB。这个计时只用于子图诊断，非整策略耗时。

| 输出（相对本轮 FP32 参考） | 板端 MAE | RMSE | 最大绝对误差 |
| --- | ---: | ---: | ---: |
| prefix_hidden | 0.001936734 | 0.003657032 | 0.125875473 |
| key_0 | 0.002036711 | 0.004865605 | 0.097284913 |
| key_15 | 0.004448579 | 0.008551648 | 0.123347163 |
| value_15 | 0.001753129 | 0.002732332 | 0.047972083 |

全部 32 个 K/V 的 MAE 范围约 `2.45954e-5–0.004450843`；公式为 `mean(abs(y_board-y_ref))`，覆盖完整张量。完整逐层数据在 `runs/smolvla_prefix_split_v1/prefix_board_report.json`；真实输出包 `.npz` SHA256 `ec4e4f9b7584ed5c57a88c0a56667285f15bc02200a3bc1b6bb4decbfbe85c8b`。

[`probe_smolvla_board_prefix_action.py`](../../scripts/probe_smolvla_board_prefix_action.py)核对输入、模型、输出 hash 后，将板端 K/V 转成原始 GPU 专家使用的 BF16，再构建相同 DynamicCache。前缀调用 1 次、专家去噪 10 次，固定样本最终动作对原始 FP 的 MAE **0.000882302**、RMSE 0.001368278、最大绝对差 0.007336199；该样本没有夹爪符号变化。视觉和专家仍在 GPU，不能称为完整板端策略或闭环质量通过。原始结果 `prefix_action_report.json`、`prefix_action.npz` 留在同一 `runs/` 目录。

## 结论和下一步

已实测确认 RKNN 是当前前缀显式 K/V 输出的一条可执行路径，并验证了缓存接回原始动作专家的接口。视觉误差及夹爪异常仍保留，见[40 条视觉动作诊断](2026-10-01-vision-layout-and-action-diagnostic.md)。下一步导出完整专家去噪子图，再连接实际视觉→CPU 前缀组装→RKNN 前缀→专家→CPU Euler 更新；完成整策略板端质量与资源测量后再确定 HAQ 搜索粒度和格式。
