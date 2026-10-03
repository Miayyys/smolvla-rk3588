# RKLLM dump语义复核、板端K/V重建与非因果设置探针

## 实际结论

在当前探针配置下，直接按 `attn_k/attn_v` 文件名读取未得到预期K/V；两组输入和已知投影权重确认了该观察，但尚未排除转换环境、融合设置或dump时机等影响，不能认定SDK普遍存在内容错位。取数绕行：用 `RKLLM_DUMP_LEVEL=2` 的逐层 `attn_norm`，再用明确的FP权重在板端CPU重算K/V和RoPE，2层FP16模型、两组独立输入与浮点参考接近。**这是FP16微型模型的取数验证，不是RKLLM原生量化缓存导出，也不是SmolVLA整语言替换**。

非因果设置候选编译成功，但板端Norm SHA与原因果版本完全相同，未改变注意力行为。原SmolVLA的块mask/位置规则仍未解决，不更换主部署模型、不更新正式HAQ成本表。

## 配置、输入和版本

沿用[首次dump探针](2026-10-03-rkllm-dump-probe.md)的随机Llama（2层、hidden256、MLP512、Q头4/KV头2、head_dim64）、8 embedding及实际FP16/W8A8模型。模型/输入/参考SHA见 `runs/rkllm_dump_probe_v1/probe_report.json`。SDK/runtime1.3.1、板端driver0.9.8不变。

本轮独立已知权重探针：从相同tiny模型另建 `runs/rkllm_dump_identity_v1`，两层Norm权重1、Q为矩形I、K为2I、V为3I、attention输出与MLP权重全0；词embedding保持原值。第一输入复用之前seed41的8×256数据，第二输入为NumPy default_rng(42)生成N(0,0.02) float32。模型、输入、已知结构记录与SHA见 `construction.json`、`audit.json`；独立目录保留，不覆盖原随机探针或项目真实权重。

已知投影与非因果模型本轮转换叠加NumPy1.26.4、SciPy1.15.3，原torch2.7.1+cu118、transformers5.5.4；与SDK严格要求torch2.6/transformers5.8仍有差异。因此范围为当前栈的实际机制验证，不推断所有正式SDK环境的结论。未修改原LeRobot环境或SDK安装文件；非因果探针的Python进程内方法恢复见下文。

## 投影dump名字与数据内容不符

原随机FP16模型，第一层 `attn_norm` 与浮点输入Norm MAE1.60e-8；第一层直接Q文件全0。穷举基本维度与head_dim分块排列，发现：

- `0-attn_v-0` 经固定重排后对应**原K线性投影、RoPE之前**，MAE7.7425e-5，而不是V。
- `0-attn_k-0` 对应**原Q投影前4个token**，MAE7.6842e-5，而不是K。

固定重排：K内容文件先reshape[token=8,KV_head=2,32,2]，transpose(0,1,3,2)再合并64维；Q内容文件reshape[4token,4Q_head,32,2]并同样重排。此处确认的是特定文件内容对应关系，不能将它升级为32份原生cache的正确导出规范。原始扫描保存 `name_semantics_audit.json`、`all_file_v_scan.json`；未发现可直接取正确V的简单候选。

已知权重独立第二输入（布局保持第一输入的固定结果，不重新挑选）：

| 层 | 名为V的文件 vs 已知K MAE | 名为V的文件 vs 已知V MAE | 名为K的文件 vs Q前4token MAE |
| --- | ---: | ---: | ---: |
| 0 | 0.0002684 | 0.769991 | 0.0001392 |
| 1 | 0.0002684 | 0.769991 | 0.0001392 |

原始数值 `runs/rkllm_dump_identity_v1/second_semantics_audit.json`。第一层Q文件仍全0，第二层Q文件最大绝对值0.072968。**推断**：dump时机、融合映射或复用缓冲区可能导致数据错位；尚未确认唯一内部原因。不能把前面约0.35的文件对照误差直接归因“FP16语言计算坏了”或“权重没正确加载”。

## 绕行原理：从Norm输出重建K/V

公开可复核公式（本微型Llama，未改变权重）：

\[
N_l=\operatorname{RMSNorm}(H_l),\quad
K_l=\operatorname{RoPE}(N_lW_{K,l}^{T}),\quad
V_l=N_lW_{V,l}^{T}.
\]

每层Norm文件按[token,hidden] float32解析；显式FP权重分别为[KV_head×head_dim,hidden]，投影后reshape[token,KV_head,head_dim]并transpose到[1,KV_head,token,head_dim]。RoPE采用原参考cos/sin，半维旋转 `concat(-K[...,D/2:],K[...,:D/2])`。这是原浮点模型的K/V重建，不是假定SDK native W8A8权重与之相同；实际量化缓存等价性未验证。

脚本 `scripts/reconstruct_rkllm_norm_kv.py` 校验大小/shape/有限值，记录Norm、权重与参考SHA，保存输出npz；CPU使用NumPy，板端OPENBLAS_NUM_THREADS=1。Norm参考与结果此前先在PC对照，再上传板端重建，未将PC数值直接写成板测。

| 输入 | K0 MAE | V0 MAE | K1 MAE | V1 MAE | 重建耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原固定输入 | 3.15e-8 | 3.27e-8 | 0.00013617 | 0.00013891 | 3.0353ms |
| 独立第二输入 | 1.90e-8 | 2.04e-8 | 0.00014233 | 0.00013274 | 1.3542ms |

4个输出均[1,2,8,64]；第二层relative_RMSE约0.00053。报告 `norm_cpu_kv_board.json`、`second_norm_cpu_kv_board.json` 含精确SHA与全部数值。重建耗时只含该微型案例的读取/运算/比较流程，**不含RKLLM推理、dump写盘、权重加载、通信等**；不可外推SmolVLA耗时、正式p50或加速收益。

```bash
cd /root/qvla_board_test/rkllm_dump_probe_v1
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/root/qvla_board_test/python_site python3 reconstruct_rkllm_norm_kv.py \
  --dump dump_level2/rkllm_dump --weights reconstruction_weights.npz \
  --reference fp_kv_reference.npz --output norm_cpu_kv_board.json
```

## 非因果元数据设置：未生效

SDK converter暴露 `LLMWriter.add_causal_attention(bool)`，尝试在写header之前显式加入False。`scripts/probe_rkllm_noncausal_compile.py`只在该进程内包装 `write_header_to_file`，finally恢复；编译日志确有hook调用，FP16候选另存 `tiny_noncausal_fp16.rkllm`。不改动态库，不依据猜测ABI写入私有runtime参数。

相同输入，PyTorch全双向参考显式使用[1,1,8,8]全0 additive mask；与原因果native的第二层Norm MAE0.550960，足以区分两种规则。

新候选在板端运行成功，但两层Norm dump SHA**与原因果版逐项完全相同**；重建后对全双向FP参考K1 MAE0.186865、V1 MAE0.186180。因此该metadata设置不能作为非因果注意力支持的证据。原始文件 `noncausal_compile.log`、`noncausal_board.log`、`noncausal_flag_audit.json` 和 `noncausal_norm_cpu_kv_board.json`。

该失败只针对本设置方法和当前栈，未证明任意定制RKLLM都不可能实现；原生二维mask入口仍没有验证有效路径。

## 下一步与边界

已形成可测的Norm dump＋CPU浮点投影路径，且可以用原权重保留RoPE语义；这是后端接入研究进展，不是HAQ最终模型。接下来需要验证实际SmolVLA的16层Norm可取，并解决其双向前缀/分块状态/有效token位置规则；普通causal结果不能直接替代。若找不到正确mask执行路径，则暂不扩大真实模型转换与闭环成本，继续保留RKNN为实际可运行主路径。


## 后续纠正与替代路线

[配置复查与原生缓存解析](2026-10-03-rkllm-cache-access.md)修正了过强的错位定性，并验证官方embedding回调＋token输入可以生成缓存文件，关闭dump后仍可解析两层原生FP16 K/V，独立输入最大误差<0.001。无需采用Norm＋CPU重算作为唯一方案；实际SmolVLA的mask/position仍待解决。
