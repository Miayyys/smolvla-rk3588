# RKLLM原生MatMul边界：一致输入产生异常注意力列

> 后续更新：[工作池缓存映射修复](2026-10-03-rkllm-working-pool-coherency.md)已消除本配置的重复输出漂移。本页保留修复前的实验结果。

## 目标及范围

承接[长度隔离](2026-10-03-rkllm-repeat-length-isolation.md)。本轮定位原生语言连续调用漂移的首个可观察异常，未训练、未进行新任务评测。使用同一个16层真实语言FP16模型，96-token诊断输入、1 NPU／1 CPU、相同输入连续运行5次。短输入只用于定位，不是完整策略部署方案。

成功定位的判据：捕获同一原生MatMul的两份输入和输出，核对输入一致性，并由输入直接计算参考；发现异常还不等于已找到底层根因。诊断钩子会改变时序，不能用其耗时评价性能，不能把首个异常编号认为所有执行都固定。

## 原生采样实现

- `patch_rkllm_matmul_trace.py`与独立汇编／linker文件：输入同步调用点0x331ccc、输出同步调用点0x332124。原生同步函数完整执行，之后调用worker中的诊断回调；原计算、mask和NPU操作保持原样。
- 钩子保留原始返回值和ABI要求的寄存器。只修改已校验SHA的独立库，代码放在经审计的可执行空隙；不修改系统库。
- `rkllm_matmul_trace.h`：读取当前Tensor flatbuffer名称、shape及pool描述，保存前40个MatMul边界；每次run重新计数。名称从flatbuffer取得，未再依赖已复用的临时op字符串。
- `analyze_rkllm_matmul_trace.py`：比较各轮输入／输出，记录第一处不同及独立QK参考。真实推理没有用CPU重做attention；CPU乘法仅分析已保存输入。

启用环境：`QVLA_RKLLM_MATMUL_TRACE=1`、`QVLA_RKLLM_TRACE_DIR=<空诊断目录>`，其余沿用重复探针。模型SHA：`dcf230cfe4b79d085dff33595da3b3e807ca4ab51ca9bc77175bb5b40f92d65f`；输入／输出双钩子库SHA：`f8a783f41d10cbb09b23d29dd3ebc35a1a40e21f395f6464e0eaa78b362b8ba3`。每轮worker SHA来自该轮report，所有诊断及raw snapshot放在Git忽略的runs。

最初输出单钩子抓到第一层Q/K/V完全一致，第一个QK输出不同。增加输入采样后首次变化后移。第一次尝试把旧op指针当作仍有效std::string解码导致worker退出，属于诊断代码错误，已改为读取Tensor名称；该失败不是模型数值结果。

## 一致输入、异常输出的实证

主要证据：`runs/rkllm_native_patch_v1/qvla_io_named96/`，包括worker.log、五轮快照、重复K/V及`matmul_comparison.json`。每轮前40个输出覆盖前两层和第三层部分计算。第0…22号输出在五轮之间完全一致；第23号是第二层中的一个QK分组：

- A：288×64 FP16，288=3个query head×96 token；各轮逐元素一致。
- feature B：96×64 FP16，各轮逐元素一致。
- C：288×96 FP32；第1轮与后四轮有288个元素不同，即一整列。

正常输入边界的实际行优先解释经参考和公开MatMul核对：

\[
C_{\rm ref}=\operatorname{float32}(A)\operatorname{float32}(B)^T.
\]

第1轮C的第10列（从0编号）最大误差63.196114；其他列接近参考。后四轮该MatMul最大误差约3.43e−5。第10列前3项：

| | 输出 | 参考 |
| --- | ---: | ---: |
| row0 | 0.027127 | −8.634809 |
| row1 | −2.091191 | −7.382381 |
| row2 | 15.088089 | 12.235666 |

这不是“只看第一步动作符号”或正常低精度误差的推断：同一FP16输入本应得到同一乘积，这次边界确有异常。**可定位为原生QK执行／数据交接问题；尚不能仅凭CPU快照判断NPU运算、内部转换、分配或同步哪一项是根因。** 1e−3列筛查阈值仅用于标记异常，不是任务质量门槛。

同一批次96-token K/V重复最大差为0.675781、1.468750、1.079102、0.375000，语言仍未通过重复检查。

## 使用完全相同输入的公开MatMul对照

`probe_rknn_matmul_repeat.cpp A.bin featureB.bin badC.bin`重放上述第23号真实输入。普通布局与native布局各运行20次，M=288、K=64、N=96、FP16×FP16→FP32、core0；B显式转置并按官方接口打包，A/C按返回布局转换，输入输出显式同步。

两种布局均重复差0；对CPU FP32累加最大误差均3.43322754e−5；被保存的RKLLM异常C对同一参考为63.1961136。原始日志`public_exact_qk96.log`。这比上一轮较小M和人工输入的对照更直接；仅证明这份输入和形状在公开路径可正确执行，不能保证RKLLM内部路径等价。

## 同步对照与内部区域采样边界

1. **完整输入分配范围同步**：输入回调额外按Tensor PlanSize同步，而非只按当前有效元素数。`QVLA_RKLLM_FULL_INPUT_SYNC=1`，关闭快照以减少采样干扰。五轮K/V重复最大差1.382813、0.960938、1.082031、0.637695，未解决。日志`full_input_sync96.log`。
2. **同一输出再次同步读取**：原输出快照后再调用原生输出同步、读取第二份快照。前40个MatMul×5轮，共200组首次／二次读取，全部逐元素相同；K/V重复最大差仍为1.708008、2.390625、2.143066、1.625000。证据`qvla_double_sync96/`、`double_output_sync96.log`。这个对照没有支持“再刷新一次CPU输出就能修好”；它不排除其他同步或执行时序问题。
3. **候选内部B区域**：按编译模型PlanOffset=856064，从pool显式同步并采样，尝试按公开native B布局解码。得到的内容在正常QK输出时也不能一致解释为当前B，因此此地址的CPU视图／live分配／布局尚未验证。**不据该快照认定内部K打包已损坏。** 原文件名含nativeB只表示当时的假设，证据在`qvla_native_b96/`。代码已改为明确的`TRACE_INTERNAL_REGION`，要求显式`INTERNAL_OFFSET`，避免称为已确认B。

## 产物与复现边界

库生成：`python3 scripts/patch_rkllm_matmul_trace.py --source runs/rkllm_native_patch_v1/mask_trace_lib/librkllmrt.so --output-dir <独立目录>`。板端编译worker时同时提供`rkllm_matmul_trace.h`及固定SDK头文件。探针使用`--tokens 96 --runs 5`。普通推理不启用这些环境变量；此库只是诊断产物。

原始文件与SHA索引为`matmul_boundary_experiment_manifest.json`。历史中间worker有记录SHA但未逐版保留二进制，不能声称每个旧版本可按hash重载；最新源码、保留worker与库能重跑当前诊断。当前代码支持原采样／输入扩范围／双输出读取三个独立模式，不把过去有快照的运行当作无快照对照。

## 结论与下一步

已从“后续层K/V漂移”定位到“同一QK输入出现异常整列”，并证明该输入在公开MatMul路径正确。输入扩范围同步、输出二次同步没有解决。内部区域的live布局未确认，仍需检查RKLLM特有的feature→native转换、命令任务以及内部缓冲区绑定。

本轮未取得修复结果；不进入新闭环任务、不更新正式HAQ成本、不报告模型压缩、部署加速或新成功率。板端默认恢复为此前无MatMul诊断钩子的配置，仍标记repeatability_failed。

恢复后96-token三次检查的重复最大差0.906250、0.560547，manifest与实际文件hash匹配；日志`matmul_trace_restored96.log`。恢复worker SHA为`ac4de51f57881de854f4e71e43bbc41b80f4f8f422c45110be849d0c646742de`，保存在`matmul_io_trace_lib/restored_worker`；库恢复为0030eb…，本轮双MatMul钩子不再生效。
