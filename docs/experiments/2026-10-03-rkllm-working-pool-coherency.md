# RKLLM 工作池映射修复与复核

## 问题与判据

前序[MatMul 边界定位](2026-10-03-rkllm-native-matmul-boundary.md)发现同一 QK 输入出现异常整列，公开 MatMul 相同输入稳定。输入扩范围同步、输出二次同步无效。本次只修改语言 worker 内的 NPU 分配方式；保持权重、FP16 格式、原生 NPU MatMul 和分块注意力规则。

成功判据：固定输入重复 K/V 完全一致；完整视觉→语言→专家 A→B→A 的 prefix/KV/动作最大差均 <1e-5，且 B 动作确实改变。任务质量另做配对短任务筛查，不能以重复一致性代替通过率。

## 原理与实施

使用 `scripts/probe_rknpu_allocation.c` 作为进程内 LD_PRELOAD 适配器，拦截已核实的 `rknpu_mem_create` ioctl ABI。将指定分配的 `RKNPU_MEM_CACHEABLE` (bit 1) 换为 `RKNPU_MEM_WRITE_COMBINE` (bit 2)，其他标志保持不变。该映射没有普通 CPU cache，因此对应对象的 MEM_SYNC 不再转发；其他对象的同步照常执行。

先把所有可缓存对象改为 WC 验证方向，再缩小到 **32,432,128 字节工作池**。409,190,400 字节权重区和命令区仍保持原方式。这里只在独立 RKLLM worker 设置环境；RKNN 视觉与专家父进程没有设置 LD_PRELOAD。所有观测分配 SRAM=0，submit flags=5（PC|PINGPONG，未置 NONBLOCK）。

公开 MatMul 默认分配也使用 CACHEABLE（flags=0x403）。所以不能简单归因于“公开接口与 RKLLM 使用了不同的默认内存类型”，也没有证明定制内核某一行是根因。目前证据支持：**改变 RKLLM 工作池的缓存映射消除了已复现的输出漂移**，精确驱动/runtime 责任边界仍未确认。

## 固定配置

- RK3588，4 GB，内核 `6.12.69-lzamp+`，NPU driver 0.9.8。
- RKNN Runtime 2.3.2，RKLLM 1.3.1；语言编译模型 FP16、单 NPU、worker CPU threads=1。
- 完整 prefill 160 token，真实样本 151 有效 token；padding 压实后，最后有效 token 为 state，保持 prefix 双向、prefix 不看 state、state 看全部有效 token。
- language SHA256：`dcf230cfe4b79d085dff33595da3b3e807ca4ab51ca9bc77175bb5b40f92d65f`。
- mask runtime SHA256：`0030eb3ea743b48a56286bea77ca723cfe33311c0508c59325302fb0bd309072`。
- worker SHA256：`ac4de51f57881de854f4e71e43bbc41b80f4f8f422c45110be849d0c646742de`。
- allocation adapter SHA256：`a6bed6c6b6c87baecd37dbefcd2f7958d08963756cf2a9bdd9ecaa5a37df49d3`。
- 固定输入 SHA256：`e64f5d8b2423dc28b4c51fcb67ce8f68900a15fc5284620119dcd70e2e96b681`。
- FP32 参考权重 SHA256：`f6e1654eabed4741dbcb0d20655a9c21fc25c184a77e7f6657a71a3f5ffb33c0`。
- 完整 graph、processor、CPU 权重 hash：`runs/rkllm_native_patch_v1/wc_pool_live/deployment_manifest.json`，沿用 V1 无教师 QAT 的 FP16 前端诊断版本；不是完整低比特最终部署。

环境由 `smolvla_rkllm_backend.py` 根据 manifest 设置：

```json
"allocation_control": {
  "library": "libqvla_rknpu_alloc.so",
  "mode": "wc",
  "pool_bytes": 32432128
}
```

## 已测结果

所有误差均以 `max(abs(x_repeat-x_first))` 计算；FP32 对照对有效 token 计算 MAE/max，各层元素数相同，汇总 MAE 为32个 K/V 张量 MAE 的平均。

| 测试 | 原可缓存映射 | WC 映射 |
| --- | --- | --- |
| 96-token 固定输入、5次 | 重复最大差0.8574～3.2969 | 全分配 WC 后每次差0 |
| 160-token 固定输入、5次 | 前序已确认漂移 | 全分配 WC 后每次差0 |
| 160-token 固定输入、10次 | 本次计时对照最大差3.52295 | 仅工作池 WC 每次差0 |
| 真实视觉→语言→专家 A→B→A | 前序重复未通过 | 仅工作池 WC，prefix/KV/动作重复差均0；B动作 MAE0.0641606 |

同一固定输入的原生 K/V 与匹配 FP32 语言代数对照：32个张量平均 MAE **0.0006798843**，最大绝对差 **0.4989529**；五次误差结果一致。少数单点误差仍不可忽略，不把均值小解释成任务必然正确。完整 A→B→A 推理含预处理耗时 **6814.28 / 7098.73 / 6983.30 ms**。

### 开销对照

相同160-token输入，依次运行原映射与工作池 WC，各10次，排除首轮后的9次计算百分位。计时覆盖 run＋保存 prompt cache＋worker stdio，不含初始化和 Python cache 解析。

| 映射 | p50 ms | p90 ms | p95 ms | 重复最大差 |
| --- | ---: | ---: | ---: | ---: |
| 原 CACHEABLE | 508.95 | 512.18 | 512.45 | 3.52295 |
| 工作池 WC | 675.25 | 747.88 | 750.82 | 0 |

WC 本次语言中位延迟增加166.30ms，约32.7%。这两批不是交错随机顺序测试，且未锁定温度/频率；只记录观测，不推导精确因果加速比。未测量整策略相对原映射的配对延迟变化、峰值内存或功耗。不得宣称没有性能代价。

## 原始证据与复现

目录：`runs/rkllm_native_patch_v1/`。

- `alloc_trace96.log`、`alloc_wc96.log`、`alloc_wc160.log`：分配日志与重复对照。
- `wc_pool160.log`：只改工作池的10次复核。
- `latency_cached160.log`、`latency_wc160.log`：原始逐次计时及误差。
- `public_allocation_trace.log`：公开MatMul默认flags核对。
- `wc_reference/{input160.bin,cache160.npz,fp32_comparison.json}`：输入、原生缓存、FP32逐层结果。
- `wc_pool_live/`、`wc_pool_persistent_live.log`：真实完整链路输入、输出、manifest与A→B→A证据。

板端复核命令（root=`/root/qvla_board_test/rkllm_native_patch_v1`）：

```bash
cd /root/qvla_board_test/rkllm_native_patch_v1
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/root/qvla_board_test/python_site \
  python3 verify_smolvla_rkllm_persistent.py --root .
```

## 边界

适配器目前是针对固定 SDK/内核 ABI 和已测工作池大小的原型；不是官方支持的公开配置。语言 attention 修改仍为固定 runtime hash 的二进制适配，不声称已获得官方 API 支持。后续整理 PR 应提供最小复现、缓存映射证据和适配代码，暂未提交。

真实压缩体积未因本次映射改变；尚未完成 >=40% 压缩的 RKLLM 全策略质量验证。短任务结果单独补录，不能转移 GPU QAT 成绩到板端。

## 两个配对短任务：已完成

运行 `scripts/run_smolvla_board_libero.py`，suite=libero_spatial/libero_object，task_id=0，init_state_index=0，seed=0，GPU原始FP与真实板端分别执行。LIBERO版本与本地环境沿用先前板测；精确环境资产参考项目锁定记录。每50步查询一个动作块，仿真器在板端推理时暂停，不证明实时控制可用。

| 任务 | 原始FP | RKNN视觉＋RKLLM语言＋RKNN专家 |
| --- | --- | --- |
| Spatial task0 | 成功，78步 | 成功，80步 |
| Object task0 | 成功，125步 | 成功，130步 |

原FP **2/2**，板端 **2/2**。初始观测hash与逐次查询噪声hash严格配对，见 `wc_short_tasks_v2/summary.json`，视频、实际输入、执行动作、查询时延均保留在任务子目录。这只是两个已用作部署诊断的任务，不是新冻结测试集或40任务成功率。相较历史纯RKNN FP16前端版本仍为2/2，没有证明质量提升。

Spatial/Object 有效token分别149/141，连同原151样本，三个长度各先完成真实A→B→A（重复差均0、B动作改变）后才加入manifest允许列表。模型统一按160token计算，其他指令长度尚未验证，仍拒绝运行。

首次闭环启动失败原因是入口脚本仍为指向旧纯RKNN部署目录的符号链接，Python导入了旧 `BoardSmolVLA`。将RKLLM目录的入口改为独立普通文件后重跑，未改动旧部署目录；失败日志保留于 `wc_short_tasks/`，有效完整结果在 `wc_short_tasks_v2/`。

现在已测配置可以稳定完成两项实际任务。后续重点是低比特语言转换的动作/任务质量与体积验证，然后才更新HAQ完整后端选择和成本表。本次无需再扩大漂移排查。

## 用户明确要求：保留V1逐模块精度

用户要求使用V1无教师损失QAT的权重，且量化精度也按 `config/haq_candidate_v1.json`，不能只借用权重或把大量模块恢复FP16后称为V1部署完成。原图共304节点：300个W8A8、专家 `layers.3.self_attn.v_proj` FP16、专家 `layers.10.mlp.gate_proj` BF16、语言 `layers.3.mlp.down_proj` INT16 DFP，以及CPU逐行INT8 embedding（层号均为代码零起始索引）。

当前诊断版不满足此要求。先前RKNN专家BF16配置被拒绝并显式改用FP16，因此即使专家部分也不是原V1精度的完全复制。RKLLM 1.3.1 本地 `api/rkllm.py` 的 build 公开 dtype 列表有W8A8/W4A16及其分组格式，`hybrid_rate` 表示block分组量化比例；不能由这个参数推断它可实现V1指定的单独INT16 DFP投影。精确逐节点INT16路由尚未验证，不能把全语言W8A8称为完整V1图。

后续转换以原精度图为目标；先核实指定INT16投影及BF16例外的实现路径。无法保持时明确报告后端限制，任何精度替代作为另一个候选，不静默改变V1图。当前未启动新量化编译或宣称已完成该目标。
