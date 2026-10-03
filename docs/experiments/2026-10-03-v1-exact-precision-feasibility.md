# 保留V1逐模块精度：后端接口验证与拆分准备

## 要求与当前状态

用户明确要求部署 `V1无教师损失QAT`，包括训练后权重及 `config/haq_candidate_v1.json` 中全部304节点的精度配置。300个W8A8；专家 `layers.3.self_attn.v_proj` FP16、`layers.10.mlp.gate_proj` BF16；语言 `layers.3.mlp.down_proj` INT16 DFP；CPU词嵌入逐行INT8。层号为代码零起始索引。不得将 FP16 前端诊断版或 BF16→FP16 替代称为完整V1。

本轮完成后端接口小图验证与FP图拆分工具，**没有完成真实V1全图转换或板端效果评测**。GPUServer SSH返回WebSocket HTTP404；用户确认将开机并更新SSH。本地仅保留该候选的完整语言FP权重、旧部署图及小报告，缺少无教师QAT的完整master和原始专家ONNX。不能借用有教师版本master或原始FP权重冒充所选候选。

## RKLLM 1.3.1 的 INT16 验证

本地安装SDK公开 `build` dtype列表没有INT16；官方手册 `hybrid_rate` 实际用于同位宽的分组/非分组量化混合，不是任意逐节点精度图。

使用此前已编译W8A8/FP16的两层tiny fixture，CPU加载成功（return0），分别调用 `build(do_quantization=True,quantized_dtype='w16a16i_dfp'/'w16a16i',target_platform='rk3588',num_npu_core=1,max_context=128)`。两者均 **return -1**，SDK日志明确 `rk3588 not support quantized_dtype`。内部 `MatMulType` 枚举也没有INT16×INT16；枚举包含的INT4×INT4→INT16只是累加输出，不能解释成W16A16。

原始证据：`runs/v1_exact_precision_feasibility/rkllm_int16/{report.json}` 与 `rkllm_int16.log`。首次探针清理调用不存在的release方法而失败，日志保留 `rkllm_int16_cleanup_failure.log`；修正清理后完整重跑两种请求，不把首次清理异常当成后端拒绝证据。

结论范围：现有公开RKLLM编译入口不能原样实现V1语言图。没有证明所有私有实现均不可能；也没有实现INT16外部回调注入。按用户保留精度的要求，后续使用RKNN语言路径；不静默改成全语言W8A8或FP16。

## RKNN 2.3.2 单独BF16例外

用seed29固定两条MatMul分支，输入 `[1,128]`，权重 `[128,64]`；4组独立随机输入只用于格式探针校准，**不是实际V1校准数据**。对比实际计算节点/权重dtype，不能只看图输出dtype。

| 请求 | 实际结果 |
| --- | --- |
| global float16，指定gate输入/输出bfloat16 | hybrid_step2拒绝：bfloat16 not supported in custom_quantize_layers |
| global bfloat16，指定gate输入/输出float16 | 成功；gate Conv及权重为FLOAT16，另一投影INT8，不能算BF16例外 |
| global bfloat16，修改指定quantize_parameters.dtype为float32 | 拒绝：dtype is not allowed to be modified |
| global bfloat16，修改指定quantize_parameters.dtype为bfloat16 | 同样拒绝 |
| global bfloat16，不量化，两条投影均浮点 | 成功；两条Conv及权重均BFLOAT16 NPU |

原始日志 `bf16_kernel_hybrid.log`，逐请求报告/配置/中间模型/产物 `bf16_kernel_hybrid/`。最初 `bf16_hybrid/` 探针只定位输出转换，生成了INT8计算后FP16转换；该版本**不支持任何核心精度判断**，已修正为按 `weight_gate` 唯一定位计算节点后重新运行。

这里只证明BF16独立图可编译、原混合接口不能直接指定该例外。BF16独立图板端支持另见[此前精度实测](2026-09-29-rk3588-precision-support.md)。本轮新产物未板测，不能转移旧数值或延迟。

## 按原精度保留的拆分方案

准备 `scripts/split_v1_projection.py`：

1. 校验源ONNX SHA，唯一定位指定MatMul/Gemm输出。
2. 提取计算该投影输入所需的前段；计算前段到后段仍使用的全部活跃中间张量（包括残差），不只保存投影输入。
3. 单独提取带静态权重的投影，后段同时接收该投影输出与其他活跃张量。
4. ONNX checker验证三个图；传入同名输入NPZ时，ORT顺序执行三段与原整图比较，成功阈值rtol/atol1e-4。量化和板端运行是后续独立验证。

该工具可以准备专家BF16 gate，也可以准备语言INT16 DFP down。后者须核实RKNN混合INT16的量化规则：旧图仅确认INT16计算，尚未证明等于V1的DFP。必要时将该投影独立编译为 `w16a16i_dfp`，保留其他投影W8A8；不把普通INT16自动称为DFP。

两个小图FP分割对照完成：一个原始输入直接作为切点，另一个包含前段与跨切点残差分支。后者的frontier为 `[hidden,feature]`；三段与原图MAE/max均0。原始输入、图SHA和逐输出对照保存于 `split_tool_smoke/`、`split_residual_smoke/`；最终通用工具的同残差对照保存于 `projection_split_residual/`。**没有在真实V1专家图上验证，也没有测拆分后的硬件额外开销。** 增加图边界可能增加调用、转换与内存开销，不能预设“无性能损失”。

## 后续实际转换步骤

服务器恢复后：核对master SHA `4aeb92854d2bb89bac84a2d791d2acb4389b934178948c05533d6a52fe0b9f81` 与原始ONNX/校准清单；实际专家拆分并做FP等价性门槛；用独立校准划分编译BF16例外与其余原V1投影；核实语言INT16 DFP规则；视觉保持V1 W8A8；严格检查最终计算dtype、实际文件体积、相同原始观测/噪声动作误差和少量配对闭环。当前FP16 RKLLM诊断版本保持可复测，不冒充V1最终版本。
