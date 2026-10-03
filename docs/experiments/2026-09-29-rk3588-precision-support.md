# RK3588 精度支持实测：RKNN 子图与底层 MatMul

## 范围、判据与配置

验证当前 Toolkit2/runtime 2.3.2、driver 0.9.8、R1 Linux 6.12.69-lzamp+。不将 API 枚举、可编译或运行返回 0 单独作为数值正确证据；不推广为完整 SmolVLA 支持。

复用 `runs/tiny_w4a16_probe/tiny.onnx`：随机 seed=20260928，X=[1,16,128]，W=[128,64]，Y=XW，opset13/IR8；8 条校准、1 条独立留出输入。FP32 NumPy X@W 作参考。模型/数据来源哈希见 2026-09-28 tiny-w4a16 记录。无训练或任务 episode，无 QAT；测试格式是候选扫描，不进行阈值扫描。FP16/BF16 do_quantization=False；整数格式 do_quantization=True，使用版本默认量化粒度/算法。本次未提取内部 scale，不声称复现了编译器量化公式。

脚本 `scripts/probe_precision_support.py`。本地 Python3.12.13、torch2.4.0、numpy1.26.4、onnx1.17.0。板端使用已有 `rknn_board_subgraph_smoke.py` 和 Lite2 2.3.2，默认 core 配置，private mount namespace 绑定 2.3.2 runtime；输入输出 API 为 FP32。每格式预热10次、测100次，仅一轮、未锁频；板子系统时间与主机不同步。计时使用单调时钟，不受日历时钟偏差影响。

原始数据 `runs/precision_support/report.json`、`build.log`、`board/precision_probe/*_board.json` 和运行日志。板端报告包含每个产物、输入与参考 SHA256、文件大小、RSS。编译报告中的核心 MatMul 被改写成 Conv，分别标注 FLOAT16/BFLOAT16/INT8/INT16、NPU；输入输出接口仍包含 CPU 节点。

## RKNN 模型结果

| 配置 | 转换 | 板端结果 | p50 ms | MAE vs FP32 |
| --- | --- | --- | ---: | ---: |
| float16 | 成功，核心 FLOAT16 NPU | 成功 | 0.406274 | 0.000279007 |
| bfloat16 | 成功，核心 BFLOAT16 NPU | 成功 | 1.458122 | 0.002127522 |
| w8a8 | 成功，核心 INT8 NPU | 成功 | 0.209407 | 0.011085147 |
| w16a16i | 成功，核心 INT16 NPU | 成功 | 0.244989 | 0.061888717 |
| w16a16i_dfp | 成功，核心 INT16 NPU | 成功 | 0.244551 | 0.000086714 |
| float32 | config 拒绝 | 无模型 | 未测 | 未测 |
| w8a16 / w4a16 | config 明确拒绝 rk3588 | 无模型 | 未测 | 未测 |
| w4a4 | config 不认识此名称 | 无模型 | 未测 | 未测 |

MAE=mean(abs(Y_board-Y_FP32))，1024 个输出元素。延迟为 Lite2 inference 调用，包括接口成本，排除模型加载；单模型单输入，不能作为 SmolVLA 最终精度或通用性能排名。INT16 两种配置误差差别很大，未分析内部量化参数前不解释原因。

`tfloat32` 是 float_dtype 错误信息列出的合法名称，但单进程扫描在该项 build 阶段提前结束，连 Python BaseException 捕获也未记录异常，未得到产物。日志 `tfloat32.log` 保留；随后移除该项重跑其余配置。它不是完整 IEEE FP32 支持证据，也不能标为可用。脚本当前默认扫描不含该异常项。

## 底层 MatMul 探针与失败记录

使用官方 v2.3.2 头文件：
- https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v2.3.2/rknpu2/runtime/Linux/librknn_api/include/rknn_matmul_api.h
- 同目录 rknn_api.h

` scripts/probe_matmul_precision.c`（路径无前导空格）在板端 gcc 编译并直接链接已有 runtime 2.3.2。M=16,K=N=128，全1输入，参考每项128，检查2048个输出；无浮点容差外的量化误差预期。每类型独立进程、15秒超时。

- W8A16 type5/6、W4A16 type7/8、INT8×INT4 type11/15、FP16×INT4→BF16 type12：当前 runtime 在 create 阶段报 unsupported dtype in this platform。
- INT4×INT4→INT16 type10：默认布局执行中 abort；native布局可创建、run返回0，但数值不匹配。
- FP16和INT8控制组同样数值不匹配。因此底层探针链路仍有未定位问题，不能归因为所有这些格式硬件不支持。
- 复查尝试了 native 布局、显式 mem_sync、virt_addr+offset、固定 core0，仍未通过。保留 type_*、native_type_*、offset_type_*、core0_type_* 日志。

上述底层失败不影响独立的 RKNN 模型路线已通过的测试，但 INT4 不能进入“已验证可用”的 HAQ 成本表。RKLLM 没有在本次加载模型实测，仍标未验证；不将其视为已经支持 W4A16。

## 对项目的结论

当前可加入后续真实模块探测的格式：INT8、FP16、BF16，以及探索性的 INT16（两种校准格式分别记录）。每个 SmolVLA 模块仍须确认转换和边界开销。FP32留给CPU参考路径；W4A16/W8A16在当前RKNN及底层runtime所测入口被拒绝。INT4底层路径数值未验证，不能计入可部署收益。动作质量、完整模型RAM、闭环与功耗本实验未测。

## 第二轮：完整接口清单复核

完整结论表见 `docs/rk3588-precision-support.md`。官方Toolkit2 tag v2.3.2解析为commit `42aa1d426c0a9e0869b6374edba009f7208a1926`，下载其MatMul demo、matmul_utils和Float16.h，以板端g++ -O2编译，链接现有2.3.2 runtime，未升级驱动或替换系统库。官方demo以及全部日志保存在 `runs/precision_support/final_board/precision_official/`。

命令模板：`./official_demo TYPE 16,128,128 1 1 3 1 0`。类型1/2/4官方随机数值校验通过，类型3/9/10不通过。对失败项追加4,64,64与32,256,256，B_layout=1、AC_layout=0/1；另16,128,128调用100次，仍不通过。随机种子沿用官方示例默认，未额外指定。返回0但数值校验不通过不能计为支持。

自写常量探针将调用数改为3后，类型1/2/4/10均得到2048个正确输出128；说明首轮自写探针结果不能简单作为不支持证据。常量可能掩盖布局和符号问题，追加 `scripts/probe_int4_signed.c`：M16 K128 N128，A[i]=(i*7+i/17)%15-7，B[i]=(i*3+i/13)%15-7，包含正负非均匀数据，按native布局写A、官方函数转换B、输出按native布局还原，调用10次；与CPU整数精确参考对比，2038/2048不一致，最大绝对误差404。与官方随机测试失败一致，当前INT4路径仍不可用，根因未定。

`tfloat32` 改用独立进程 `scripts/probe_rknn_tfloat32.py`，日志 `tfloat32_isolated.log` 明确为 `Can not support request type: tfloat32`，exit1，不再仅标异常未知。config与load通过，build终止。

RKLLM下载官方1.3.1英文手册（仓库commit `f7390530443bf84f0394255a449d7cbe81e69d1c`），quantized_dtype段明确RK3588只列w8a8/w8a8_g128/g256/g512四种量化格式，W4A16属于其他平台。文档原始PDF和pdftotext文本保存于 `runs/precision_support/rkllm_sdk.pdf`、`rkllm_sdk.txt`；没有把文档确认写成RKLLM模型实机测量。
