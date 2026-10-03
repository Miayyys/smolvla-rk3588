# 融合、混合精度边界与 CPU embedding 补充实测

三轮均为20次预热、100次计时；RKNN项运行于RK3588 NPU core0，CPU embedding项在RK3588 CPU上运行。所有延迟以完整`RKNNLite.inference`或完整NumPy查表调用计，CPU到NPU搬运和完整策略运行未包含。

| 单元 | 格式 | 输入shape | p50 ms | p95 ms | 模型/权重 B | 轮间稳定 | 误差诊断 MAE | 误差范围 | 状态 |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- |
| expert_MLP_layer0 | w8a8 | [1, 50, 720] | 2.8946 | 3.6527 | 4524893 | 否 | 0.008809 | heldout_activation_vs_fp32_subgraph_only | measured |
| expert_MLP_layer0 | float16 | [1, 50, 720] | 6.1600 | 6.2493 | 8910851 | 是 | 0.000111 | heldout_activation_vs_fp32_subgraph_only | measured |
| language_MLP_layer3 | w8a8 | [1, 177, 960] | 11.8468 | 13.9990 | 7522271 | 是 | 0.824544 | heldout_activation_vs_fp32_subgraph_only | measured |
| language_MLP_layer3 | float16 | [1, 177, 960] | 26.9201 | 28.5550 | 14857285 | 是 | 0.001435 | heldout_activation_vs_fp32_subgraph_only | measured |
| vision_MLP_layer11 | w8a8 | [1, 1024, 768] | 32.0934 | 44.2609 | 5293473 | 是 | 0.512492 | heldout_activation_vs_fp32_subgraph_only | measured |
| vision_MLP_layer11 | float16 | [1, 1024, 768] | 101.6799 | 104.5194 | 9938567 | 是 | 0.002066 | heldout_activation_vs_fp32_subgraph_only | measured |
| expert_layer0_QKV_projection | w8a8 | [1, 50, 720] | 2.1505 | 3.0396 | 1229200 | 是 | — | not_measured_cost_only | measured |
| expert_layer0_QKV_projection | float16 | [1, 50, 720] | 3.0580 | 3.7291 | 2361014 | 是 | — | not_measured_cost_only | measured |
| expert_layer0_QKV_projection | q_fp16+other_w8a8 | [1, 50, 720] | 2.4939 | 3.2724 | 1916816 | 是 | — | not_measured_cost_only | measured |
| expert_layer0_QKV_projection | k_fp16+other_w8a8 | [1, 50, 720] | 2.3542 | 3.1550 | 1461136 | 是 | — | not_measured_cost_only | measured |
| expert_layer0_QKV_projection | v_fp16+other_w8a8 | [1, 50, 720] | 2.2449 | 3.1636 | 1461136 | 是 | — | not_measured_cost_only | measured |
| expert_layer0_QKV_projection | qk_fp16+v_w8a8 | [1, 50, 720] | 2.6597 | 3.3461 | 2144848 | 是 | — | not_measured_cost_only | measured |
| expert_layer0_QKV_projection | qv_fp16+k_w8a8 | [1, 50, 720] | 2.6094 | 3.3039 | 2144848 | 是 | — | not_measured_cost_only | measured |
| expert_layer0_QKV_projection | kv_fp16+q_w8a8 | [1, 50, 720] | 2.4156 | 3.2222 | 1689168 | 是 | — | not_measured_cost_only | measured |
| attention_QK_softmax_PV_proxy | float16 | [3, 1, 15, 50, 48] | 4.4983 | 5.3259 | 204085 | 是 | 0.000150 | synthetic_activation_numerical_sanity_only | measured |
| attention_QK_softmax_PV_proxy | w8a8 | [3, 1, 15, 50, 48] | 3.7020 | 4.8008 | 239695 | 是 | 0.009330 | synthetic_activation_numerical_sanity_only | measured |
| language_token_embedding_cpu_lookup | native_bf16_row_lookup | [1, 177] | 0.2208 | 0.2476 | 94617600 | 否 | 0.000000 | synthetic_token_ids_embedding_output_vs_native_bf16_only | measured |
| language_token_embedding_cpu_lookup | cpu_int8_row_lookup | [1, 177] | 0.6390 | 0.7471 | 47505920 | 是 | 0.002159 | synthetic_token_ids_embedding_output_vs_native_bf16_only | measured |

## 结论边界

MLP子图的INT8误差是单层输出对原始FP32 ONNX的差异，不是LIBERO任务通过率。注意力代理使用固定shape和合成Q/K/V，只用于确认QKᵀ、Softmax、PV组合在板上的成本，不是完整注意力层。QKV混合精度的p50包含Q/K/V计算与RKNN插入的格式转换；编译器日志中的`exDataConvert`周期是静态估计，不等于隔离出的实测转换时延。
词嵌入采用真实checkpoint BF16行权重；INT8用逐行max-abs/127对称量化，行scale为FP32。CPU查表输入token ID为固定种子生成，长度177取自下游文本MLP样本；不含tokenizer和CPU到NPU传输。INT8行权重约节省一半表存储，但查表反量化p50高于直接BF16读取，因此只证明存储收益，不代表端到端更快。
原始每次计时数据保留于`runs/hardware_supplemental_v1/<case>/board_<round>.json`和`runs/hardware_embedding_v1/board_embedding_lookup.json`；本表JSON带输入/模型/报告hash。
