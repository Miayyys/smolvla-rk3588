# RKLLM dump逐层K/V取数探针

## 结论

真实RK3588、RKLLM Toolkit/runtime 1.3.1：`RKLLM_DUMP_LEVEL=1` **确实导出两层各自的K/V文件**。输入8个embedding，返回hidden模式和普通生成模式均RUN=0，缓存位置8。该机制解决了“完全取不到中间文件”的问题，但数值/布局与原浮点模型尚未对齐，不能写成SmolVLA语言子图已可替换。

## 技术与配置

内部dump把计算中间张量写入工作目录 `rkllm_dump/`；不依赖公开K/V getter。微型随机Llama：2层、hidden256、MLP512、4个Q头、2个KV头、head_dim64、context128，词表沿用本地SmolVLM2 tokenizer49280，不下载模型权重。模型seed20261003，输入seed41，8×256 float32 embedding，规模与语义都不是SmolVLA。

脚本：`scripts/probe_rkllm_dump_prepare.py`、`probe_rkllm_dump_compile.py`、`probe_rkllm_dump_run.cpp`。原始数据 `runs/rkllm_dump_probe_v1/`，含master/输入/浮点KV参考、模型hash、编译/板端日志和导出文件。`probe_report.json`固定所有SHA。运行前后都未替换板端runtime或主部署模型。

转换暂用现有torch2.7.1+cu118、transformers5.5.4、NumPy2.2.6并叠加已下载SDK1.3.1与小依赖；不是官方全部依赖版本的资格验证。FP16和W8A8均真实导出并在runtime标明相应dtype，W8A8使用normal算法、单NPU核、optimization_level0、一条固定合成文本校准；该随机探针校准不是机器人数据质量实验。后续在SDK独立目录补NumPy1.26.4/SciPy1.15.3，SDK重新导入通过，未修改原LeRobot环境；原实验编译环境仍按上面的实际版本记载。

## 实际运行

板端目录 `/root/qvla_board_test/rkllm_dump_probe_v1/`。

```bash
cd /root/qvla_board_test/rkllm_dump_probe_v1/hidden_v2
RKLLM_DUMP_LEVEL=1 ../dump_probe ../tiny.rkllm ../input_embeds.bin 1
```

`1`=GET_LAST_HIDDEN_LAYER；`0`=GENERATE。清空chat template，CPU count4/mask0xf0，max_new_tokens1，max_context128。返回hidden[8,256]及cache8，INIT/RUN/CACHE_RC均0。

最初本地交叉编译二进制依赖GLIBC2.38，板子缺失，改用板端g++后解决；首个native初始化CPU默认mask与数量不符，显式count/mask后解决。日志保留。运行存在CPU affinity warning err22，但推理成功；本实验不据此报告性能收益。

## 实际文件与数值验证边界

返回hidden模式的输出：

| 文件 | 字节数 | 对应预期元素数 |
| --- | ---: | ---: |
| `0-attn_k-0` | 4096 | 1024 |
| `0-attn_v-0` | 4096 | 1024 |
| `0-attn_k-1` | 4096 | 1024 |
| `0-attn_v-1` | 4096 | 1024 |

按little-endian float32读取均有限，长度与2KV头×8token×64维吻合。但**仅凭长度与名字不能确认证明dtype、布局和RoPE后语义**。普通生成模式也有这些文件，部分其他输出只留下最后位置，不能默认多次decode会保留全量历史。没有获得prompt_cache.bin，不能称缓存文件解析已验证。

`RKLLM_DUMP_LEVEL=2`额外导出Norm等数据。第一层 `attn_norm-0` 按[token,hidden]与相同输入的PyTorch归一化输出MAE约1.600182e-8，确认了输入确实进入计算而不是读入了错误embedding。第一层V的常见三维排列候选仍未对齐：INT8最低MAE0.347152，FP16最低MAE0.347093；若干通道分块排列也未改善至可接受数值。`layout_audit.json`保存原三维候选；初步FP16对照说明不能简单归因INT8截断。原因未定位，不能按最小MAE猜测布局后直接交接动作专家。

尝试使用SDK GGUFReader读取原生格式失败；改变副本magic后也因metadata解码失败，原始产物未改动。这不构成已解析RKLLM权重或缓存的证据，不调用未声明的accuracy_analysis ABI。

## 下一步

优先核对dump是否位于RoPE/重排/缓冲复用之前或之后及转换权重映射，使用第二组独立输入验证锁定布局。若逐层张量与计算规则确认后，仍需解决SmolVLA二维块mask、padding/position输入；即使全部K/V可读，也不能将causal结果冒充原模型结果。调试文件落盘耗时未测量，本次没有SmolVLA闭环或HAQ表更新。


## 后续语义复核

[语义与重建实测](2026-10-03-rkllm-dump-semantics.md)已确认直接投影文件名与内容不符，并在两个输入上验证Norm dump＋板端CPU投影重建路径。先前未对齐不等于已证明真实FP16语言计算退化；目前第二层重建MAE约0.00014。非因果metadata候选没有改变板端实际输出，SmolVLA mask问题仍待解决。
