# SmolVLA 混合精度候选图：权重覆盖审计

**实验性质**：只对固定 checkpoint 的 safetensors 头部做静态分组与原字节统计，形成[候选量化图](../quantization-map-v0.md)。不加载模型、不执行动作，也不推断 RK3588 的真实量化节省或速度。

## 问题与判据

在执行新的 W8A8 QAT / 独立 PTQ 前，先核对候选图是否覆盖整个模型，避免只量化已有敏感度探针覆盖的 MLP/注意力而漏掉大型嵌入、连接投影、时间投影。事前判据是：每个源权重张量**恰好匹配一个**分组，配置中没有空组，权重 SHA-256 等于锁定值；失败即停止图的使用。

## 原理、配置与原始数据

固定 `lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de`，权重 SHA-256 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`，processor 为 checkpoint 随附文件。数据分区 SHA-256 `f755546a6b074d2fe248333fc42c3dbf30f9b54b9f0fa0b58a16506a54d942c2`；本静态审计**没有使用任何 episode**、随机数、GPU 或 RKNN，因而没有校准/测试数据泄漏。

输入为 `artifacts/transfer/model/model.safetensors`；分组规则为[机器可读配置](../../config/quantization_map_v0.json)，实现为[`audit_quantization_map.py`](../../scripts/audit_quantization_map.py)。只读取前 8 字节头长度及 JSON 头；每个张量的原始载荷字节按 `data_offsets[1]-data_offsets[0]` 计算，每组求和，不对名义 bit 比例作压缩估计。每个正则从张量名起始匹配，要求命中数为 1。原始逐组计数、载荷字节、dtype 见[`quantization_map_v0_inventory.json`](../../figures/quantization_map_v0_inventory.json)。图是[候选图中的 Mermaid](../quantization-map-v0.md#smolvla--rk3588-混合精度候选图-v0)，表达目标模块连接与位宽，不是测量曲线。

复现：

```bash
cd /home/loser/Study/QVLA
python3 scripts/audit_quantization_map.py
```

## 结果与选择依据

脚本通过哈希及一对一覆盖检查：**500 个张量、25 个分组、906,639,456 B 源载荷**。文件总大小 906,712,520 B。候选 W8A8 分组的**源载荷**为 660,138,496 B，占 72.8%；该比例只是初筛覆盖范围。受保护视觉 11 MLP 源载荷 9,444,864 B、语言 3 MLP 14,745,600 B；词嵌入与 `lm_head` 各 94,617,600 B、connector 23,592,960 B，过去的动作敏感度探针没有覆盖这三组。它们已在图中单列为待测，不凭源文件大小直接选择 INT8。

v0 的视觉 11→FP16 与语言 3→BF16 是由已有输出激活敏感度、RKNN 子图误差和原加载 dtype 支持的**保护候选**。其余 W8A8 标签是待检验的搜索起点；对未探针模块保持 `native`。这些判断不等于证明组合图的 QAT、PTQ 或 RK3588 可行。

## 尚未测量及下一步

本次 FP/QAT/PTQ 动作质量与闭环成功率、真实低比特部署包 $B$、板端峰值 RAM $R$、p95 端到端延迟 $T$、功耗均**未测量**；固定收益公式 $G$ 为 NA。`lm_head` 是否参与动作推理、是否共享运行内存未知。下一步补这些模型路径和硬件探针，再从图 v0 派生少量完整候选，按质量门槛与板端实测收益比较。真实 QAT 必须与 RKNN 目标激活/权重量化规则对齐；旧的专家 W8 weight-only 训练脚本不能直接执行本图。
