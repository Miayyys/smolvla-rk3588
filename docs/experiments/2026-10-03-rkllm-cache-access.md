# RKLLM配置复查与官方缓存路线实测

## 结论与纠正

之前“内容错位”的表述过强。已测事实是：当前微型模型配置下，按dump文件名解释的张量不符合对应K/V，某些文件匹配其他投影。尚未排除非标准转换依赖、图优化/融合、调试时机等因素，不能定性SDK通用bug或权重计算错误。

本轮修正回调只在RKLLM_RUN_NORMAL读取输出，并显式设置input.role=user。重新运行后第一层attn_k/attn_v文件SHA与旧值完全相同，说明这两项修正没有解决所观察到的dump现象。头文件与runtime沿用官方1.3.1固定提交，原始输入Norm和重建第二层K/V接近参考；转换环境尚非完整官方pin，仍不能排除所有配置因素。

**替代路线实测成功：关闭dump，通过官方embedding回调＋token输入保存prompt cache，再从缓存解析两层原生FP16 K/V。** 原权重投影不在CPU重算。两组独立输入与浮点参考最大误差均<0.001。当前仅固定2层、8token探针，不是通用缓存解析器，也未解决SmolVLA原mask。

## 使用方式对照

脚本 `scripts/probe_rkllm_dump_run.cpp`；Toolkit/runtime1.3.1、driver0.9.8，context128、单NPU核、CPU count4/mask0xf0，keep_history=1，save_prompt_cache=1。输出全部保存 `runs/rkllm_config_recheck_v1/`。

| 输入方式 | 推理模式/生成数 | 是否保存缓存 |
| --- | --- | --- |
| 直接EMBED | GENERATE/4 | 未生成 |
| TOKEN IDs100..107 | GENERATE/4 | 成功，212262B |
| TOKEN IDs100..107＋官方embed_callback返回原8个embedding | GENERATE/1 | 成功，首输入212265B |

最后一行另导出 `export_embedding=False` 的微型FP16模型，输入词embedding由回调提供，输出投影仍保留。两次均有EMBED_CALLBACK count=8、RUN=0、CACHE_RC=0/TOKENS=8，RKLLM_DUMP_LEVEL=0。未启用dump，不用CPU重算K/V。特征回调只接受测试8个token，更多token场景/真实机器人输入尚未泛化。

注意：缓存匹配以token标识为条件；未来用于每帧图像embedding必须清空旧状态或使用正确缓存生命周期，不能以相同占位token盲目跨观测复用cache。

## 解析与独立验证

原生文件不是单一连续张量，含token IDs、模型名、RNG/其他状态与KV。第一输入探索全张量1024元素匹配确定候选：每层K采用FP16，token/head/channel存储，通道有RoPE交错；V为head/channel/token。当前固定fixture从文件尾定位K0/K1/V0/V1分别为8244/6184/4124/2064B，含层间元数据，不能外推不同层数或token长度。

`probe_rkllm_cache_parse.py`明确限于该fixture，检查开头8token标记/指定token IDs，再读取4个[1,2,8,64]数组。布局从第一输入锁定，第二输入不再搜索、不拟合数据。K直接与PyTorch完成RoPE后的cache比较。

| 输入 | K0 MAE | V0 MAE | K1 MAE | V1 MAE | 最大绝对误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原固定输入 | 9.1384e-5 | 9.1987e-5 | 1.6247e-4 | 1.6427e-4 | 0.0006908 |
| 独立输入 | 9.0012e-5 | 8.5935e-5 | 1.7169e-4 | 1.6202e-4 | 0.0007775 |

文件、hash、解析数值见 `cache_kv_independent_validation.json`、`second_parser_report.json`；首cache SHA `3f6a83aa4f80dec7feedb136006d7f6fc588b2795eb8231bc4ed100aabbaaa1c`，第二cache SHA `1b1190d4440a8aad602a140b5d4c58bcc44c92019f979df9cc24233c862c0278`。实际buffer来自板端，解析与数值对照在本地完成；未测缓存落盘/解析完整延迟。

## 可选取数方式

1. 官方prompt cache保存/加载＋解析：本次微型FP16模型取得原生K/V，优先扩展不同token长度/层数，建立结构解析，不继续固定偏移到真实模型。
2. 官方last_hidden回调：只提供最后层，无法一次拿全16层；拆成多子模型再计算各层K/V需要额外转换/正确mask验证，尚未实测。
3. 申请官方中间张量/KV getter或支持自定义mask的接口：现公开接口没有已验证路径；本次未发送issue或联系第三方。
4. Norm dump＋CPU投影：之前已有微型FP16板测，但会重复计算/携带权重，只作为诊断回退。

官方证据：[C API](https://github.com/airockchip/rknn-llm/blob/f7390530443bf84f0394255a449d7cbe81e69d1c/rkllm-runtime/Linux/librkllm_api/include/rkllm.h)、[官方调用示例](https://github.com/airockchip/rknn-llm/blob/main/examples/rkllm_api_demo/deploy/src/llm_demo.cpp)。API提供保存缓存和embedding回调，不提供本次私有缓存格式稳定性保证。

下一步扩大缓存结构验证并检查SmolVLA的mask/position实现途径。读取正确causal K/V不证明原SmolVLA可替换；不转移QAT分数，不修改最终HAQ图/成本表。
