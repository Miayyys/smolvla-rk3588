# SmolVLA 语言前缀接入 RKLLM 的可行性验证

## 结论与验证范围

当前 RKLLM 1.3.1 **未找到保持现有 SmolVLA 计算语义、直接替换 RKNN 前缀的公开接口路径**。阻碍是逐层 K/V 输出，以及自定义前缀注意力掩码/位置输入。不能将此结论写成“RK3588 无法运行语言模型”或“RKLLM 永远不支持 SmolVLA”。后续 SDK 或定制接口解决这些条件后可以重新验证。

本次实际完成：固定官方源码/手册审查、连接真实板子核对 runtime hash 与导出符号、本地原始 checkpoint 的注意力掩码消融。**没有生成 `.rkllm`、没有运行 RKLLM 模型、没有测量 RKLLM 的 SmolVLA 质量/延迟/内存**。本次不安装数 GB 转换依赖；即使标准 backbone 转换成功，也不能解决尚缺失的边界接口。

## 固定版本与原始证据

- checkpoint：`artifacts/transfer/model/model.safetensors`，SHA256 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`；处理器/数据划分 hash 随 `mask_ablation.json` 保存。
- 官方提交：`f7390530443bf84f0394255a449d7cbe81e69d1c`，RKLLM 1.3.1。
- [C API](https://github.com/airockchip/rknn-llm/blob/f7390530443bf84f0394255a449d7cbe81e69d1c/rkllm-runtime/Linux/librkllm_api/include/rkllm.h)，SHA256 `4ad19442de58df1b7d1ed0884ed75b8d4b5107cc4e75b7e44900b1f95a32c57c`。
- [官方英文手册](https://github.com/airockchip/rknn-llm/blob/f7390530443bf84f0394255a449d7cbe81e69d1c/doc/Rockchip_RKLLM_SDK_EN_1.3.1.pdf)，SHA256 `61ed7267ae0048977618bc6ae1311e3d70a3334af5894aa434d14b8d25fe0da6`；审查 3.1.6 自定义转换、3.2.6 推理、3.2.9 prompt cache、3.2.10 KV 管理、3.2.13 cross attention。
- 真实板子 `root@10.42.0.252`：`/usr/local/lib/librkllmrt.so` SHA256 `f25e9b099db08aaacd0a3ac62b4697d3951d6ae61ae41ea09f6702cfa89eb32c`，与此前安装的官方产物一致。
- 原始材料在 `runs/rkllm_prefix_feasibility_v1/`（Git 忽略）：官方 tree/header/PDF/custom config、`board_symbols.txt`、`interface_audit.json`、`mask_ablation.json/.npz`。动态库符号由真实板端 `nm -D --defined-only` 获取。

## 输入/输出合同核对

| 必要能力 | SmolVLA 当前需要 | RKLLM 1.3.1 公开证据 | 结论 |
| --- | --- | --- | --- |
| 输入 embedding | `[1,177,960]`，两路图像、文字、状态组装 | `RKLLM_INPUT_EMBED` 接收 embedding 和 token 数 | 输入形式有基础支持；这不证明计算等价 |
| 指定前缀 attention mask | `[1,177,177]`，图像/文字块内双向，状态另一个块，排除 padding | 公共 `RKLLMInput`/`RKLLMInferParam` 未提供 self-attention 二维 mask；custom 配置主要描述模块名称/结构 | 未找到等价配置入口，不将普通 causal prefill 当作替代 |
| 指定 position IDs | 根据有效 token 累加，padding 不增加位置 | 公共 embedding 输入未提供 self-attention position IDs | 未找到对应输入入口；需另外验证 padding/位置语义 |
| 逐层 K/V 输出 | 16 层 × K/V，每个 `[1,5,177,64]`，供动作专家使用 | 结果结构公开文字/token、最后一层 hidden、logits、统计；动态库公开符号未发现逐层 K/V tensor getter | 未找到直接交接路径 |
| prompt cache 文件 | 若要替代 K/V 输出，必须解码每层数值与布局 | 提供保存/加载以供 RKLLM 自身复用；审查材料未给外部张量解码合同 | 不能视为可供 RKNN 专家使用的 K/V 数组；未实验解析缓存文件 |
| cross attention | 当前问题是从前缀导出 K/V | `rkllm_set_cross_attn_params` **接收** encoder K/V、mask、position | 方向相反，不能解决前缀导出；也不证明完整动作专家可转换 |

`get_kv_cache_size` 返回每个 batch 缓存的位置数量，未返回 K/V tensor。最后一层 hidden 不能代替各层进入注意力计算前生成的 K/V。现有官方 custom conversion 示例及本地 wheel 的 `load_huggingface(custom_config=...)` 支持结构映射，不等于任意 ONNX 输入/输出图转换。

## 注意力语义的真实模型消融

运行脚本：

```bash
.venv-haq-local/bin/python scripts/probe_smolvla_prefix_causal_mask.py
```

环境：PyTorch `2.7.1+cu118`，本地 CUDA；沿用原始 checkpoint 加载 dtype 和处理器。只用隔离开发缓存中的 episode 18、task 0、frame 0，动作噪声 seed `2416662958`。原始路径重新推理与缓存动作逐元素一致。

只在前缀 prefill 中修改 mask，位置、权重、输入、专家与 10 步去噪、随机种子及前后处理保持相同：

\[
M_{causal}(i,j)=M_{original}(i,j)\land(j\le i).
\]

差异度量为 `MAE=mean(abs(candidate-reference))`、`RMSE=sqrt(mean((candidate-reference)^2))`、最大绝对误差。不是量化截断阈值扫描，也没有改变任何位宽。

| 实测项 | 结果 |
| --- | ---: |
| 原 mask 允许的 token 对数 | 22,651 |
| 其中指向后续位置的允许对数，causal 修改后移除 | 11,175 |
| 前缀最后 hidden 的 MAE / 最大误差 | 0.0592251 / 8.546875 |
| 第 15 层 K 的 MAE / 最大误差 | 0.0648733 / 4.367188 |
| 第 15 层 V 的 MAE / 最大误差 | 0.0501386 / 2.351563 |
| 完整 50×7 动作的 MAE / 最大误差 | 0.00915816 / 0.0843685 |
| 50 个动作的夹爪符号变化 | 0 |

脚本 SHA256 `61e6429280ffb122da63cf673c5db20ab4ca81ff43576853b6b384db8bd37a23`；原始 NPZ SHA256 `4831046bd5699a630d801d8fdfd6ae53896ea9842d752a6ad5707049becda935`。各层 K/V 的完整差异见 JSON。

该消融只证明 causal mask 不能等价替换原前缀语义；**不是 RKLLM 模型数值测量，更不证明任务一定失败**。单观测推理时间含冷暖差异，不用于后端性能比较。没有做新的闭环回合或最终测试集评价。

## 路线决定

保留已实际串联的 RKNN 前缀与 RKNN 动作专家作为当前部署路径。RKLLM 仍是候选，重新接入的必要条件是：

1. 找到支持原始二维 mask 与位置语义的转换/运行方式。
2. 得到每层 K/V 的公开布局、dtype 与数值交接方式，或完整等价的前缀＋专家执行方案。
3. 用相同 checkpoint 和固定输入/噪声比较逐层 K/V、完整动作，随后测板端成本及闭环质量。

拆成 16 个独立 RKLLM 单层模型再自行重建 K/V，或解析私有 prompt cache 文件，只是未验证的定制研究方向；本次不把它们写成可运行方案。此结论限制当前后端接口选择，不按人工敏感度锁定 HAQ 的任何精度位点。

## 2026-10-03补充：内部dump与缓存文件的取数候选

本次重新检索官方接口/示例和官方仓库的问题讨论，并从真实板子读取 `/usr/local/lib/librkllmrt.so`，7,674,816 B、SHA仍为 `f25e9b099db08aaacd0a3ac62b4697d3951d6ae61ae41ea09f6702cfa89eb32c`。只读 `nm -D -C` 和 `strings` 实测保存 `runs/rkllm_kv_access_v2/{audit.json,runtime_symbols.txt,runtime_strings.txt}`。未修改或替换板端库；未运行RKLLM模型。

### 新发现的具体路径

1. **内部dump开关**：真实1.3.1二进制包含 `RKLLM_DUMP_LEVEL`、日志 `RKLLM_DUMP_LEVEL=%d` 和输出路径 `rkllm_dump`。官方仓库[Issue #501](https://github.com/airockchip/rknn-llm/issues/501)中，报告者在1.2.3/RK3576用 `RKLLM_DUMP_LEVEL=1` 得到逐token/逐层Q文件 `{pos}-attn_q-3`。这是报告者一手测量，不是维护者保证，也不能据此认定本板1.3.1会输出完整K/V。该开关值得做真实小模型探针：先核对文件清单、层数、token覆盖、dtype/shape与RoPE前后语义，再拼成专家需要的32个张量。完整K/V能否获取尚未测量；落盘调试方式的I/O开销尚未测量。
2. **保存prompt cache再解析**：[官方示例](https://github.com/airockchip/rknn-llm/blob/main/examples/rkllm_api_demo/deploy/src/llm_demo.cpp)提供 `save_prompt_cache` 与路径参数。真实库字符串包含保存/加载token数、embed slots/embed floats、损坏embed payload的日志；说明文件不能不经验证就当作连续FP16 K/V。未找到公开张量布局约定，也没有实际cache样本；解析路线仍未验证。若实验，需多组token长度的文件差分及与已知K/V数值对照，而非按文件大小猜偏移。
3. **隐藏导出函数**：本板实际动态符号中有 `rkllm_accuracy_analysis`，但当前公开header未提供其声明；未按猜测ABI调用。没有新增公开逐层K/V getter或可直接调的llama缓存函数。二进制内部存在llama/cache与non-causal字符串，仅作研究线索，不代表RKLLM入口允许直接设置这些能力。

### 条件与优先顺序

先用小模型验证dump机制，再考虑cache解析；仍不能据此直接把SmolVLA前缀切换到RKLLM。拿到普通causal推理K/V与拿到原SmolVLA K/V是两个验证条件：原二维块mask、padding与position规则没有解决时，即使取出32个张量，也不保证动作计算等价。全双向non-causal也不能自动代替原分块mask。

本次板端 `/root`/`/home` 和本地Study/Downloads中未找到现成 `.rkllm` 模型，因此没有动态dump实测结果；不把字符串发现写成已取得K/V。下一实验需要一个可运行的小 `.rkllm` 产物（优先自建微型模型以免下载大权重），核对输出后再判断是否值得正式接入。此前结论应读作“公开API未找到直接路径”，不应扩大为“任何调试/定制办法都不可能取出K/V”。

### 同日dump动态验证结果

后续已实际完成[两层微型模型板测](2026-10-03-rkllm-dump-probe.md)：内部dump输出每层K/V文件，先前“未动态测试”的边界已更新。第一层Norm对齐，但K/V数值/排列尚未确认，不能认为SmolVLA替换条件已满足。优先继续dump语义核对与原mask兼容验证，不重复将问题归结为完全无法取数。
