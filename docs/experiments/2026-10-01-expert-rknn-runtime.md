# 完整动作专家 RKNN：导出与板端算子诊断

## 输入、版本与原理

使用 checkpoint SHA256 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`，开发观测 episode 18 / task 0 / frame 0，动作 seed 2416662958。捕获前先核对同处理器、同噪声的完整 FP 动作与缓存逐元素相同。Toolkit2 / Lite2 / runtime 均为 2.3.2，RK3588 driver 0.9.8，板测使用 NPU_CORE_0。

专家子图包含全部 16 层，以及 action_in_proj、时间 MLP 和 action_out_proj。35 个输入为动作 `[1,50,32]`、CPU 时间编码 `[1,720]`、前缀有效位 `[1,177]` 和 32 个 `[1,5,177,64]` 的逐层 K/V；输出是 `[1,50,32]` 的速度向量。时间正弦编码按原函数的 float64 计算后转换 FP32，留在 CPU；Euler 更新为 $x_{k+1}=x_k-0.1v_k$，共 10 步。每步仅输出速度，前缀 K/V 保持不变。

导出以原加载权重转换 FP32 为计算参考，不把 FP32 导出称为原始权重格式。源 checkpoint 实际有 474 个 BF16 张量（446,772,624 元素）和 26 个 FP32 张量（3,273,552 元素），文件 906,712,520 B。`runs/smolvla_source_dtype_inventory.json` 是实际统计；大部分权重原本为 BF16。

## 已验证的导出修正

脚本：[export_smolvla_expert_step_onnx.py](../../scripts/export_smolvla_expert_step_onnx.py)、[verify_smolvla_expert_onnx.py](../../scripts/verify_smolvla_expert_onnx.py)。

- 旋转位置编码用拼接替代原地赋值，避免 ScatterND；FP32 输出与原运算完全相同。
- BOOL 的 CumSum 在 ONNX Runtime 中无效；显式转整数后计算位置，FP32 专家输出保持相同。
- 原时间编码使用 FP64 Cos，当前 ONNX Runtime 无对应内核；改成 CPU 时间编码输入。时间 MLP 仍在专家子图内部。
- NumPy CPU 前缀拼接与原 GPU 最大差异 `5.9604645e-8`；动作反归一化最大差异 `1.4901161e-8`，不是逐元素完全相同。10 个时间编码与原 CPU Torch 转 FP32 后逐元素相同。原始 GPU FP 动作仍与缓存逐元素相同。

失败日志保留在 `runs/expert_step_export_*console.log` 和 `runs/prepare_board_replay_console.log`；这些属于真实兼容问题，不能仅凭 ONNX checker 通过认定图可执行。

## v1 实际板测失败

`runs/smolvla_expert_step_split_v1/` 保留初版产物与输出：

| 项目 | 实测 |
|---|---|
| ONNX SHA256 | `16263cf87f13d2916723ccc60036a9362ec4d569f7d45503c8db1d81e66e13b6` |
| RKNN SHA256 | `d3616a0fdcf52edb7aeef1263a2b3cc6ee4eb2cdff62a51ced7f6165049242d7` |
| RKNN 大小 | 217,425,256 B |
| 配置 | float16 / optimization_level=3 / do_quantization=False |
| ONNX 与 FP32 参考 | 两组不同时间/噪声均通过 1e-4；最大差异 2.3841858e-6 |
| 板端速度 MAE / RMSE / 最大差异 | 0.821724 / 1.043242 / 4.148295 |
| 2 次计时 | 138.299 / 125.715 ms |
| 进程 maxRSS | 523,132 KiB |

日志 `expert_board_console.log` 明确报告 `Meet unsupported input dtype for ReduceMax` 和 `ReduceMin:/ReduceMin, fallback cpu failed`。这是 INT64 位置序列最小值。接口仍返回形状正确、数值有限的数组，因此 `expert_board_report.json` 中 `status=success` 仅表示接口返回及有限性检查成功，**数值验证实际失败，该时延不能用作有效专家硬件成本**。这张图不能接入正式 HAQ 或称为部署成功。

## v2 修正依据

专家后缀有效位全为 1，位置由 `prefix_valid_count + arange(50)` 生成，严格递增。故沿序列求最小值恒等于取第一个元素。导出后将唯一、已核对 axes=1 / keepdims=1 的 ReduceMin 替换成 Gather(index=[0], axis=1)，保持整数结果和形状，无浮点近似。不修改第三方安装包。

v2 独立保留在 `runs/smolvla_expert_step_split_v2/`。两组时间/噪声的 ONNX 核对误差与 v1 相同，最大 `2.3841858e-6`，说明等价替换未改变数值。

| 项目 | v2 实测 |
|---|---|
| ONNX SHA256 | `73679efcc44854bb26698bd13dea94cb8eb8c27600bc51df51956c4804a29eca` |
| RKNN SHA256 | `c85c63773b2fea2b2fa17a82ecb1538c68f50a0e6f8fe032cccfb5ad074c31eb` |
| RKNN 大小 | 217,425,832 B |
| 配置 | 与 v1 相同，FP16 / opt3 / 非整数量化 |
| 板端速度相对 FP32 MAE / RMSE / 最大差异 | 0.00173138 / 0.00276039 / 0.0323333 |
| 板端速度相对原加载 BF16 专家 MAE / RMSE / 最大差异 | 0.00398056 / 0.00514810 / 0.0389740 |
| 2 次计时 | 345.290 / 347.908 ms |
| 进程 maxRSS | 523,220 KiB |
| 输出 NPY SHA256 | `0057a556c23daefc1c3c27ebb198f885e41cf656f9c6ff1df932139e33d130a6` |

`expert_v2_board_console.log` 未出现 `E RKNN`。相同输入、同配置、仅等价修正位置最小值后，单步 MAE 从 0.821724 降到 0.00173138，确认 INT64 ReduceMin 执行失败是初版专家大误差的主要原因。初版约132ms的计时属于错误执行，不能拿来宣传加速；正确执行约347ms。仍有浮点误差，单步接近不代表 10 步最终动作或任务成功率通过。

板测脚本：[rknn_board_expert_smoke.py](../../scripts/rknn_board_expert_smoke.py)。每个 4D KV 都显式使用 NCHW；每次推理新建 data_format 列表，避免 Lite2 原地修改格式枚举影响后续调用。后续[完整网络板端固定输入回放](2026-10-01-full-board-fp16-replay.md)已经串联，单条最终动作 MAE 0.001997；闭环质量尚未测量。
