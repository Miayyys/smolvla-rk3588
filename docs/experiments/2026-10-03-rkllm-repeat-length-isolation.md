# RKLLM连续误差：长度、布局与公开MatMul隔离

> 后续更新：[工作池缓存映射修复](2026-10-03-rkllm-working-pool-coherency.md)已消除本配置的重复输出漂移。本页保留修复前的实验结果。

## 问题、输入和判据

承接[完整链路重复检查](2026-10-03-rkllm-live-pipeline.md)。目的为定位相同输入的非确定性，不是新的任务质量测试。选定V1无教师QAT语言master重容器化为16层Llama，FP16 RKLLM；权重没有重新训练。单NPU核、单CPU线程、同步执行，每次清空官方KV cache并验证为0。原系统runtime不变。

`probe_rkllm_repeatability.py`只运行语言worker，排除视觉RKNN和动作专家。输入来自已记录开发观测的`native_live_debug.npz`，并非新增冻结测试数据。长度N≤151时保留前N−1个prefix token，最后加入原state；N=160时保留150个prefix＋state＋9个零padding。短长度改变了模型上下文，只用于诊断，不能作为部署降长方案。相同输入连续运行3次；32个原生K/V FP16张量解包为FP32计算：

\[
E_{\rm mean}=\operatorname{mean}|KV_i-KV_0|,\quad
E_{\rm max}=\max|KV_i-KV_0|.
\]

这里检查逐元素完全一致；不把它当作经任务验证的质量容差。报告中旧版`assets`来自manifest，部分诊断换库后未同步，**不能据该字段认证当时文件**。脚本已改为读取实际文件计算SHA，并报告manifest是否一致；旧报告保持原样，产物和日志另建SHA索引。

## 长度与转换布局对照

全部原始日志位于Git忽略的`runs/rkllm_native_patch_v1/`。表中两组数分别为第2、3次与第1次比较；层编号从0开始。

| 配置／日志 | MAE（两次） | 最大绝对差（两次） |
| --- | --- | --- |
| 8，isolated_repeat8.log | 0 / 0 | 0 / 0 |
| 64，isolated_repeat64.log | 0 / 0 | 0 / 0 |
| 160，isolated_repeat160.log | 0.004834 / 0.007622 | 0.801758 / 1.368164 |
| 160，保留原布局，layout_preserved_repeat160.log | 0.010907 / 0.007310 | 2.836426 / 1.178711 |
| 96，length_scan_repeat96.log | 0.000934 / 0.000504 | 0.479004 / 0.859619 |
| 128，length_scan_repeat128.log | 0.000879 / 0.001045 | 1.074707 / 0.671875 |
| 96，mask尺寸跟踪，mask_trace96.log | 0.001584 / 0.002022 | 0.968018 / 1.687653 |
| 96，OMP_NUM_THREADS=1、OMP_DYNAMIC=FALSE，omp1_repeat96.log | 0.002364 / 0.001266 | 2.484375 / 1.109375 |

第0层K/V重复差为0，后续层开始放大。8/64与96/128/160之间还存在编译候选区别，不能单凭此表断言64是精确故障阈值。

Toolkit追加形状会改变最终tensor Shape/OrigShape，因此尝试将候选插到原候选前面，保持原布局并增大PlanSize。`compile_rkllm_full_prefill_probe.py --prepend-shapes --output ...`生成独立控制产物。**保留布局没有解决漂移**；不能把追加形状造成布局变化写成已确认根因。第一次128探针缺少NN shape，日志`isolated_repeat128.log`是执行失败；补齐后结果见表，不纳入第一次的数值统计。

## 原生mask尺寸核对

独立汇编钩子记录mask tensor的ne0、ne1、nb1、batch token和填充循环步长，不用dump内容推断计算。96-token三次均为：`96 96 384 96 96`。nb1=96×4字节，循环步长也为96。该输入未发现mask矩阵行跨度不一致；这不证明所有内部张量的stride正确。

跟踪版库SHA256：`0030eb3ea743b48a56286bea77ca723cfe33311c0508c59325302fb0bd309072`；worker：`734799bd1223028f9c933930b9221c9b245335637510484807ad3d70e115effb`；长度扫描模型：`dcf230cfe4b79d085dff33595da3b3e807ca4ab51ca9bc77175bb5b40f92d65f`。代码仍限定原始库SHA及SDK1.3.1内部地址，不是公共SDK能力。

## 匹配FP32逐层参考

`compare_rkllm_length_reference.py`加载同一HF权重，FP32、eager attention、原block规则：prefix互相可见而不能看state，state可看全部有效token，padding key全部遮蔽；位置为0…N−1。它对照同长度输入的有效token，不拿短输入与原完整策略比较。参考权重、每组输入和缓存SHA见`length_comparison/fp32_comparison.json`，逐层MAE/max完整保留。

| token数 | 3次中各自最大的K误差 | 3次中各自最大的V误差 |
| --- | --- | --- |
| 64 | 0.026753 / 0.026753 / 0.026753 | 0.017089 / 0.017089 / 0.017089 |
| 96 | 0.839306 / 0.620585 / 0.063876 | 0.493299 / 0.461133 / 0.047349 |
| 128 | 0.270034 / 1.103073 / 0.519747 | 0.167453 / 0.785138 / 0.351197 |
| 160 | 0.905656 / 0.734793 / 0.971860 | 0.537783 / 0.212749 / 0.447034 |

96-token第1次layer0…13差异较小，layer14的K最大误差跃升至0.839；其他运行的突增位置不同。重复变化不是一个固定mask数学差异就能解释的现象，但具体首个错误算子仍未确认。

## 公开MatMul隔离

`probe_rknn_matmul_repeat.cpp`在同一板子调用公开`rknn_matmul_create/run`，FP16×FP16→FP32，core0；输入按固定整数公式生成，分母64，显式TO_DEVICE/FROM_DEVICE同步。比较普通布局与AC/B native布局，后者按返回subK/subN打包及还原，B使用官方布局转换函数。

形状为(T,64,T)和(T,T,64)，T∈{64,96,160}，每个配置重复20次。共12个配置（64的两类形状相同），240次执行；**全部重复差0，与CPU FP32累加参考最大差0**。原始日志`public_matmul_layout_repeat.log`；普通布局先行对照另存`public_matmul_repeat.log`。

结论仅覆盖这些固定输入、公开接口和布局。它不复现RKLLM内部的内存复用、特殊batched packing与调度，不能据此保证其MatMul输出正确，也不证明RKLLM的同步已正确。没有用CPU替换attention，也没有作为新部署耗时写进HAQ表。

## 当前结论与后续

语言独立执行即可复现，视觉／专家并发不是必要条件；保留布局和限制OpenMP线程没有解决；mask已测尺寸正常；公开独立MatMul稳定。**底层根因未解决，完整RKLLM替换尚未通过。** 后续需定位RKLLM内部首次变化的中间计算及其输入打包／缓冲区交接，而不是继续大范围位宽或任务扫描。

本轮不训练、不更新HAQ成本表、不报告新的任务成功率、内存收益或部署加速。各日志及产物hash见`repeat_length_experiment_manifest.json`。
