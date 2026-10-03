# RKLLM注意力与位置适配核查

## 实测结果

使用已选定V1无教师QAT语言权重，沿用真实语言回放边界与原mask参考。脚本`scripts/verify_smolvla_compact_prefix.py`，原始逐层结果`runs/rkllm_mask_resolution_v1/report.json`。只做一次FP32真实权重语义对照，没有重新转换或重复板端模型运行。

按原mask的有效key列剔除26个padding，177token变成151token；有效token原位置恰好是0..150，连续位置问题因此在此观测上可消除。原mask缩小后仍保持前150token全双向、前缀不能看最后state token、state能看全部有效token。

| 规则 | 32个K/V的MAE均值 | 最大绝对误差 |
| --- | ---: | ---: |
| 去padding，保留原分块mask与位置 | 1.17295e-7 | 1.09673e-5 |
| 去padding，改成causal | 0.0308698 | 5.85751 |
| 去padding，改成全部双向 | 0.00743792 | 4.02066 |

这证明对本观测，padding压实基本保持有效token的K/V；还未验证完整动作、其他观测与可变输入长度，也未测板端延迟收益。**压实不能消除attention语义差异。** 这几种MAE不能直接推断任务成功率。

## 原生入口核查

2026-10-03重新核对[官方main头文件](https://github.com/airockchip/rknn-llm/blob/main/rkllm-runtime/Linux/librkllm_api/include/rkllm.h)、已固定1.3.1头文件及SDK手册Custom Model Conversion。公开接口未提供语言self-attention的二维mask或逐token position_ids。`encoder_mask`/`encoder_pos`属于cross-attention。custom_config公开示例是算子/权重名字映射，没有此mask规则的声明。

本地runtime动态导出符号也没有可调用的causal/mask setter。内部字符串含`--attention {causal,non-causal}`和`cparams.causal_attn`，说明库含相关内部实现，但不证明命令行解析被rkllm_init使用，也不证明对应NPU执行支持。SDK writer有`add_causal_attention(False)`，前轮实际板测没有改变结果；本轮核查确认它写入模型metadata，不能把它当成已验证的runtime context控制。

不能通过重排token精确实现原双向规则：因果顺序中任意两个不同位置总有一方不能看另一方，而原前缀要求双方互相看见。仅重复token也改变softmax分母及层间状态，未经重训练不等价。

## 当前处理与可行条件

已有真实RKLLM转换和原生K/V访问，剩余阻碍集中在context的attention构图。公开配置路径暂时没有找到保持原模型规则的解法；并非声称芯片不能计算该attention。没有猜测reserved字节、调用未知ABI或修改runtime二进制，也没有引入逐层CPU重算、反复prefill等未经证实的性能绕行。

继续采用RKLLM需要原生提供分块mask/position控制，或在其内部执行图实现这些规则。当前只有预编译runtime，项目无法用公开源码直接修改内部attention构图。提供一份可提交给官方的接口需求草稿`docs/rkllm-smolvla-support-request.md`；未发送。使用现有RKNN图仍能显式保留原mask/position，后续部署质量修复应在其可控执行图中进行。不能把换成causal后的模型当作原SmolVLA部署成功。


## 官方支持路线准备

核查官方递归树commit `f7390530443bf84f0394255a449d7cbe81e69d1c`，runtime目录只有头文件与Android/Linux预编译库，没有可修改的attention实现源码。GitHub API找到同类请求[Issue285](https://github.com/airockchip/rknn-llm/issues/285)，其部分序列关闭causal mask需求与本项目有关，当前comments为空；没有官方实现方案回复，不能据此断言永不支持。Issue300中关于reshape的第三方回复针对ONNX维度，不能解决此处mask语义。

官方请求稿已补充SDK/driver版本、依赖pin差异、实测误差与四项接口问题。证据包 `runs/rkllm_native_support_request_v1/rkllm_smolvla_attention_evidence.zip` 包含脚本、原始数值报告、板日志，无SSH配置/凭据/训练模型权重。它是证据包而非独立SDK复现环境。当前gh未登录，因此未向官方发送；建议在已有285补充实测证据，避免重复问题。核查API响应均保存同目录。


后续更新：[自行修改原生runtime](2026-10-03-rkllm-native-mask-patch.md)已在微型与真实151token语言模块验证分块mask，并完成一次真实RKNN专家桥接。因此本页的“公开接口未找到解法”是接口核查结论，不再代表只能等待官方；当前已有版本绑定的实验性原生补丁，通用联动部署仍待验证。
