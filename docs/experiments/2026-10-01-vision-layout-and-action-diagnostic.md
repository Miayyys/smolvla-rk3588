# 视觉 RKNN：输入布局纠正、误差定位与离线动作诊断

## 问题和验证边界

先前视觉 FP16 RKNN 的输出 MAE 为 5.945478。假设 NCHW 图像被当成 NHWC 读取；用同一模型、同一浮点输入明确设置布局，验证两种等价传入方式是否逐元素一致。没有基于本次误差值设最终质量阈值；最终量化配置仍须依据全策略任务质量和资源约束。

模型、固定输入、processor 及文件 hash 沿用[部署拆分记录](2026-10-01-smolvla-deployment-split.md)。RKNN Toolkit2/Lite2/runtime 均为 2.3.2，driver 0.9.8；NPU_CORE_0。真实 `.rknn` 文件 SHA256 `f486c5de7f0bc085e9bb1157db761003b9d81d8c2e1e83cd11fb53942b209434`。本次无训练或校准；浮点格式对照和布局诊断不属于 PTQ/QAT。

## 修正原理和脚本

原始图像为 `[1,3,512,512]` NCHW。板端实际输入要求 NHWC；因此必须传 `data_format=['nchw']` 让 Lite2 转置，或以 `ascontiguousarray(x.transpose(0,2,3,1))` 生成 `[1,512,512,3]` 后传 `data_format=['nhwc']`。原脚本直接传 NCHW 数组而未明确布局，造成严重误差。[板端 smoke 脚本](../../scripts/rknn_board_subgraph_smoke.py)新增 `--data-format`、`--save-output`，并记录布局。Lite2 会把传入的格式列表改成内部整数枚举，每次推理必须新建列表；复用列表的第一次诊断曾出现 `Unsupport data format: 1`，已修正。

FP32 ONNX 通过 ONNX Runtime 1.26.0 CPU 实测；[浮点格式脚本](../../scripts/verify_smolvla_vision_float.py)用 Torch 2.7.1+cu118 / 本机 CUDA，在每次转换前恢复原 FP32 权重，避免连续舍入。误差统一相对原始 FP 的固定 connector 输出计算：MAE=`mean(abs(y-ref))`，RMSE=`sqrt(mean((y-ref)^2))`；相对 RMSE=`RMSE/sqrt(mean(ref^2))`。数值统计覆盖整个 `[1,64,960]` 张量。

## 一个固定相机输入的实测

| 路径 | 输出 MAE 对原始 FP | RMSE | 最大绝对误差 |
| --- | ---: | ---: | ---: |
| ONNX CPU FP32 | 0.000008693 | 0.000012740 | 0.000209808 |
| PyTorch CUDA FP16 | 0.005155604 | 0.007293198 | 0.106483459 |
| PyTorch CUDA BF16 | 0.038552059 | 0.054267732 | 0.581016541 |
| RKNN 主机模拟器 FP16，优化等级 3 | 0.010908458 | 0.015859465 | 0.263366699 |
| RK3588 FP16，明确 NCHW | 0.156709969 | 0.235166401 | 3.072341919 |
| RK3588 FP16，转置后明确 NHWC | 0.156709969 | 0.235166401 | 3.072341919 |
| 历史板端运行，布局未明确 | 5.945478439 | 7.981630802 | 67.726448059 |

明确 NCHW 与转置后 NHWC 的板端输出**逐元素完全一致**。这证明原来大部分误差来自输入布局；剩余误差不能归因于模型导出，也不能简单视为普通 FP16 舍入。优化等级 0/3 的模拟器输出误差相同，尚未对等级 0 做板测，不能据此宣称板端优化等级无影响。正确 NHWC 的两次板端调用为 1903.831/1934.780 ms；两次诊断计时不用于正式延迟统计。

原始报告与数组均在 Git 忽略的 `runs/smolvla_vision_split_v1/`：`onnx_parity.json`、`onnx_fp32_output.npy`、`board_report_nchw.json`、`board_report_nhwc.json`、两个 `board_output_*.npy`、`layout_comparison.json`、两个 `vision_opt*_fp16.build.json` 及 `.simulator.npy`。CUDA 浮点结果在 `runs/smolvla_vision_float_v1/report.json` 和各格式 `.npy`。错误布局历史报告保留，不把它作为 FP16 精度退化的证据。`instrumentation_check.json` 确认优化等级 0/3 的模拟器输出逐元素一致，诊断图与原图的最终板端输出也逐元素一致。

## 板端中间输出定位

[`prepare_vision_stage_diagnostic.py`](../../scripts/prepare_vision_stage_diagnostic.py)给固定 ONNX 添加阶段输出并保存 ORT 参考；[`rknn_board_multioutput_diagnostic.py`](../../scripts/rknn_board_multioutput_diagnostic.py)比较真实板端输出。诊断 RKNN SHA256 `67b82e4a3e6240d32296e1f819a7fd45e95fe54ed2db371c3edcbd06754f28c9`，213,059,770 字节。原始数据为 `runs/smolvla_vision_split_v1/vision_stages.json`、`vision_stages_reference.npz`、`board_stages.json`、`board_stages.npz`；加输出可能改变融合图，不能直接用该图推断原模型每个算子的误差。最终 connector 输出与原 RKNN 的输出另做逐元素核对。

| 阶段 | MAE | 相对 RMSE |
| --- | ---: | ---: |
| Patch 投影＋位置编码 | 0.000044056 | 0.0392% |
| 第 0 层输出 | 0.002534554 | 1.0957% |
| 第 3 层输出 | 0.007358011 | 2.9100% |
| 第 7 层输出 | 0.011180213 | 3.0946% |
| 第 11 层输出 | 0.062505212 | 2.9562% |
| 最后 LayerNorm | 0.035043820 | 5.3894% |
| Connector | 0.156711912 | 3.5960% |

误差随 Transformer 计算累积；尚未找到单个故障算子，也未据此锁定任何 HAQ 位宽。

## 真实视觉输出接回原始策略

[`probe_smolvla_board_vision_action.py`](../../scripts/probe_smolvla_board_vision_action.py)将同一观测的两路真实板端 connector 输出注入原始 GPU 策略；在注入前核验图像逐元素相同、原始动作与 FP 缓存完全一致，并保持原动作种子 `2416662958`。两路视觉输出 MAE 分别为 0.156710/0.117673；最终 `[1,50,7]` 动作 MAE 为 **0.001752089**，RMSE 为 0.002672535，最大绝对误差为 0.012048244。报告和动作数组在 `runs/smolvla_board_vision_action_v1/`。

这个结果只证明固定观测的实际板端视觉误差可被完整动作路径消费，并给出最终动作变化；不证明闭环任务成功率，也不代表语言/专家已在板上运行。

## 40 条开发观测扩展

[`probe_smolvla_board_vision_panel.py`](../../scripts/probe_smolvla_board_vision_panel.py)使用既有隔离开发缓存，40 个任务各一条观测；80 个视觉输入来自真实 FP 调用，所有原始动作与缓存逐元素一致。压缩输入包为 95,131,905 字节，SHA256 `376efbb96384cb0273b58efe5d147c9be8413a3e4b4990ae6c7cf34e9697e0bb`，原始身份和种子在 `runs/smolvla_board_vision_panel_v1/capture.json`。板端 [`rknn_board_vision_panel.py`](../../scripts/rknn_board_vision_panel.py)只加载一次模型，前两次为预热，随后完成 80 次逐相机推理；没有运行 40 次闭环回合。输出包 SHA256 `8ee2c92b8b6a93dd311eff8ecd7264573e39cd827357b54a0584b45304b78a3c`。所有相机输入、模型和输出哈希在接回 GPU 策略前核验，图像调用顺序逐元素检查。

| 指标 | 实测结果 |
| --- | ---: |
| 80 路视觉特征 MAE / RMSE | 0.140785038 / 0.211469667 |
| 单相机 core0 `inference` p50 / p95 | 1913.061 / 1942.542 ms |
| 40 条最终动作 MAE / RMSE | 0.002072447 / 0.024076513 |
| 最终动作最大绝对差 | 2.007855356 |
| 夹爪符号不一致 | 2 / 2000 个动作位置，涉及 2 / 40 条观测 |
| 接回板端视觉后的 GPU 前缀/专家＋处理器总时长 | 3.665 s；不含板端视觉计算及传输 |
| 闭环任务通过率 / 整策略板端耗时 | 未测量 |

两次夹爪变化：task 1（episode 1、frame 141、动作块 offset 4）从 `+0.980526` 变为 `-1.027330`；task 16（episode 388、frame 44、offset 44）从 `+0.961099` 变为 `-1.021693`。其余动作分量最大绝对差不超过 0.062751。这些开合方向变化不能被较小的总体 MAE 掩盖；尚未测量它们是否改变任务结果，不能因此宣称精度通过或一定失败。40 条观测是分任务开发面板，不是最终冻结评测集。没有保存本批温度/频率日志，单相机延迟只是本次运行条件下的结果。

原始板端 `board_report.json`、`board_features.npz`，最终 `action_report.json` 和 `actions.npz` 均在 `runs/smolvla_board_vision_panel_v1/`。可核对的[实测图](../../figures/vision_deployment_diagnostic_v1.png)、[图数据及来源哈希](../../figures/vision_deployment_diagnostic_v1.json)由 [`plot_vision_deployment_diagnostic.py`](../../scripts/plot_vision_deployment_diagnostic.py)生成；第一幅只比较固定相机输入，第二幅展示阶段相对 RMSE，第三幅展示 40 条观测的最终动作 MAE，不混用为成功率。

## 当前结论

已定位并纠正视觉输入布局错误，ONNX FP32 一致性和 40 条真实板端视觉到动作的接口已验证。较小的平均动作 MAE 与两次夹爪方向变化同时存在，剩余板端浮点计算误差仍需闭环质量判断；优先打通前缀逐层 K/V 和专家实际部署，然后用统一质量协议决定是否保留该视觉 FP16 路径。不能把本次诊断误差当成正式位宽门槛。
