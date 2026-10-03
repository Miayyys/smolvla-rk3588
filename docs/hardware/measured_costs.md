# RK3588 去重成本实测

表中每行是一个去重的算子/输入形状/权重形状/格式/边界签名；不代表每个真实模块实例都单独板测。
真实 SmolVLA checkpoint 权重；固定种子合成输入和两条合成校准输入只用于成本，不用于量化质量判断。Toolkit/Lite/runtime 2.3.2，core0；20 次预热、100 次计时、3 轮交错顺序。板上时延包含 Lite2 调用，不含模型加载和输入预处理。未锁频。

基础算子配置 100 项：100 项三轮成功；另附 18 项融合、转换和CPU embedding补充测量。基础项状态分布 `{'measured': 100}`；9 项基础配置三轮 p50 极差比超过 1.2，最终候选需复测。

| 类型 | 格式 | 输入 shape | 权重 shape | 边界 | p50 ms | p95 ms | 模型 B | 轮间稳定 | 状态 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| linear | w8a8 | [1, 50, 32] | [720, 32] | standalone_float32_io | 0.5462670005726977 | 0.6177657502121292 | 60681 | 是 | measured |
| linear | float16 | [1, 50, 32] | [720, 32] | standalone_float32_io | 0.6554920009875786 | 0.7494910007153525 | 78383 | 是 | measured |
| linear | bfloat16 | [1, 50, 32] | [720, 32] | standalone_float32_io | 2.1381125002335466 | 2.651728599539638 | 77551 | 是 | measured |
| linear | w16a16i | [1, 50, 32] | [720, 32] | standalone_float32_io | 0.5929320004725014 | 0.7967389997702413 | 80717 | 是 | measured |
| linear | w16a16i_dfp | [1, 50, 32] | [720, 32] | standalone_float32_io | 0.6257425002331729 | 0.733581450731436 | 80717 | 是 | measured |
| linear | w8a8 | [1, 50, 720] | [32, 720] | standalone_float32_io | 0.8529405004082946 | 0.9817788999498589 | 48329 | 是 | measured |
| linear | float16 | [1, 50, 720] | [32, 720] | standalone_float32_io | 0.9526865001134865 | 1.286425299485927 | 71343 | 是 | measured |
| linear | bfloat16 | [1, 50, 720] | [32, 720] | standalone_float32_io | 2.7285645010124426 | 3.4818626997548563 | 70511 | 是 | measured |
| linear | w16a16i | [1, 50, 720] | [32, 720] | standalone_float32_io | 0.9195835000355146 | 1.128290650467534 | 70925 | 是 | measured |
| linear | w16a16i_dfp | [1, 50, 720] | [32, 720] | standalone_float32_io | 0.9108340000238968 | 1.2326595506237938 | 70925 | 是 | measured |
| linear | w8a8 | [1, 50, 1440] | [720, 1440] | standalone_float32_io | 2.390245998867613 | 2.924615149231613 | 1075659 | 是 | measured |
| linear | float16 | [1, 50, 1440] | [720, 1440] | standalone_float32_io | 3.791350500250701 | 4.962849898674904 | 2112689 | 是 | measured |
| linear | bfloat16 | [1, 50, 1440] | [720, 1440] | standalone_float32_io | 6.205950499406754 | 6.831956450560028 | 2111921 | 是 | measured |
| linear | w16a16i | [1, 50, 1440] | [720, 1440] | standalone_float32_io | 3.49998850015254 | 4.912073699142638 | 2115087 | 否 | measured |
| linear | w16a16i_dfp | [1, 50, 1440] | [720, 1440] | standalone_float32_io | 4.710205000264978 | 4.905073750342126 | 2115087 | 是 | measured |
| linear | w8a8 | [1, 50, 720] | [720, 720] | standalone_float32_io | 1.2800674999198236 | 1.7372052995597187 | 557258 | 是 | measured |
| linear | float16 | [1, 50, 720] | [720, 720] | standalone_float32_io | 1.7092354992200853 | 2.2111124998900777 | 1069104 | 是 | measured |
| linear | bfloat16 | [1, 50, 720] | [720, 720] | standalone_float32_io | 3.732144999958109 | 4.942697650221817 | 1068272 | 是 | measured |
| linear | w16a16i | [1, 50, 720] | [720, 720] | standalone_float32_io | 1.58601200018893 | 2.043894050666495 | 1071502 | 是 | measured |
| linear | w16a16i_dfp | [1, 50, 720] | [720, 720] | standalone_float32_io | 1.6531970004507457 | 2.0081069500520243 | 1071502 | 是 | measured |
| linear | w8a8 | [1, 32] | [960, 32] | standalone_float32_io | 0.19788750068983063 | 0.21788054982607719 | 59395 | 否 | measured |
| linear | float16 | [1, 32] | [960, 32] | standalone_float32_io | 0.16303400025208248 | 0.21174004996282747 | 85929 | 是 | measured |
| linear | bfloat16 | [1, 32] | [960, 32] | standalone_float32_io | 1.3735425000049872 | 2.0822463498006996 | 85929 | 否 | measured |
| linear | w16a16i | [1, 32] | [960, 32] | standalone_float32_io | 0.1609929990991077 | 0.21385490108514207 | 90119 | 否 | measured |
| linear | w16a16i_dfp | [1, 32] | [960, 32] | standalone_float32_io | 0.20576150018314365 | 0.21729660038545262 | 90119 | 否 | measured |
| linear | w8a8 | [1, 50, 2048] | [720, 2048] | standalone_float32_io | 3.175523499976407 | 3.848777600114772 | 1513419 | 是 | measured |
| linear | float16 | [1, 50, 2048] | [720, 2048] | standalone_float32_io | 6.645618499987904 | 6.933491349764154 | 2983153 | 是 | measured |
| linear | bfloat16 | [1, 50, 2048] | [720, 2048] | standalone_float32_io | 8.277857999928528 | 9.094050499857076 | 2982385 | 是 | measured |
| linear | w16a16i | [1, 50, 2048] | [720, 2048] | standalone_float32_io | 6.672742000091603 | 6.976236000105018 | 2990671 | 是 | measured |
| linear | w16a16i_dfp | [1, 50, 2048] | [720, 2048] | standalone_float32_io | 6.614557500142837 | 6.976863750105622 | 2990671 | 是 | measured |
| linear | w8a8 | [1, 50, 720] | [2048, 720] | standalone_float32_io | 2.5981955000133894 | 3.391391400032262 | 1523915 | 是 | measured |
| linear | float16 | [1, 50, 720] | [2048, 720] | standalone_float32_io | 4.2164360002061585 | 4.717379599969718 | 2977329 | 是 | measured |
| linear | bfloat16 | [1, 50, 720] | [2048, 720] | standalone_float32_io | 6.096580499843185 | 7.420372550268439 | 2976497 | 是 | measured |
| linear | w16a16i | [1, 50, 720] | [2048, 720] | standalone_float32_io | 4.188874499959638 | 4.718094500071857 | 2994447 | 是 | measured |
| linear | w16a16i_dfp | [1, 50, 720] | [2048, 720] | standalone_float32_io | 4.112752999844815 | 4.520002549884339 | 2994447 | 是 | measured |
| linear | w8a8 | [1, 50, 720] | [320, 720] | standalone_float32_io | 1.0139334999621497 | 1.4799377996496337 | 265930 | 是 | measured |
| linear | float16 | [1, 50, 720] | [320, 720] | standalone_float32_io | 1.1986970000634756 | 1.6858454499015352 | 489008 | 是 | measured |
| linear | bfloat16 | [1, 50, 720] | [320, 720] | standalone_float32_io | 3.2716240002628183 | 3.918744800012064 | 488176 | 是 | measured |
| linear | w16a16i | [1, 50, 720] | [320, 720] | standalone_float32_io | 1.1692389998643193 | 1.6675731996656398 | 492302 | 是 | measured |
| linear | w16a16i_dfp | [1, 50, 720] | [320, 720] | standalone_float32_io | 1.1721559999386955 | 1.6698041498329985 | 492302 | 是 | measured |
| linear | w8a8 | [1, 50, 960] | [720, 960] | standalone_float32_io | 1.622177000172087 | 2.125144250089761 | 730058 | 是 | measured |
| linear | float16 | [1, 50, 960] | [720, 960] | standalone_float32_io | 2.1615905000089697 | 2.6767299500306763 | 1410608 | 是 | measured |
| linear | bfloat16 | [1, 50, 960] | [720, 960] | standalone_float32_io | 3.9243439998699614 | 5.346957950109754 | 1409776 | 是 | measured |
| linear | w16a16i | [1, 50, 960] | [720, 960] | standalone_float32_io | 2.0668034999289375 | 2.4683802999788895 | 1417102 | 是 | measured |
| linear | w16a16i_dfp | [1, 50, 960] | [720, 960] | standalone_float32_io | 2.0102219998534565 | 2.4575021500368166 | 1417102 | 是 | measured |
| linear | w8a8 | [1, 50, 720] | [960, 720] | standalone_float32_io | 1.5182029999323277 | 1.8374898996626146 | 731850 | 是 | measured |
| linear | float16 | [1, 50, 720] | [960, 720] | standalone_float32_io | 2.076427500014688 | 2.484421599751841 | 1410608 | 是 | measured |
| linear | bfloat16 | [1, 50, 720] | [960, 720] | standalone_float32_io | 4.170355000042036 | 5.00182970001788 | 1409776 | 是 | measured |
| linear | w16a16i | [1, 50, 720] | [960, 720] | standalone_float32_io | 1.8988110002737812 | 2.3232543499261737 | 1419022 | 是 | measured |
| linear | w16a16i_dfp | [1, 50, 720] | [960, 720] | standalone_float32_io | 1.9027480000204378 | 2.248780199738576 | 1419022 | 是 | measured |
| linear | w8a8 | [1, 177, 320] | [320, 320] | standalone_float32_io | 1.7184229998292722 | 2.327336949883829 | 149132 | 是 | measured |
| linear | float16 | [1, 177, 320] | [320, 320] | standalone_float32_io | 2.3416865001308906 | 3.0546917999799916 | 235250 | 是 | measured |
| linear | bfloat16 | [1, 177, 320] | [320, 320] | standalone_float32_io | 4.401489499969102 | 5.312207150154791 | 234418 | 是 | measured |
| linear | w16a16i | [1, 177, 320] | [320, 320] | standalone_float32_io | 2.1734019999257725 | 2.617328149676723 | 238608 | 是 | measured |
| linear | w16a16i_dfp | [1, 177, 320] | [320, 320] | standalone_float32_io | 2.113904999987426 | 2.619486799903825 | 238608 | 是 | measured |
| linear | w8a8 | [1, 64, 12288] | [960, 12288] | standalone_float32_io | 24.103771499540017 | 27.025842200509942 | 12173900 | 是 | measured |
| linear | float16 | [1, 64, 12288] | [960, 12288] | standalone_float32_io | 60.025970499737014 | 61.74411204883654 | 23773938 | 是 | measured |
| linear | bfloat16 | [1, 64, 12288] | [960, 12288] | standalone_float32_io | 19.921209999665734 | 24.522534000243468 | 23699954 | 是 | measured |
| linear | w16a16i | [1, 64, 12288] | [960, 12288] | standalone_float32_io | 17.170436999549565 | 22.034418799921696 | 23723088 | 是 | measured |
| linear | w16a16i_dfp | [1, 64, 12288] | [960, 12288] | standalone_float32_io | 17.27168250044997 | 22.26450845028012 | 23723088 | 是 | measured |
| linear | w8a8 | [1, 177, 2560] | [960, 2560] | standalone_float32_io | 13.857627500101444 | 16.283982599929917 | 2528269 | 是 | measured |
| linear | float16 | [1, 177, 2560] | [960, 2560] | standalone_float32_io | 31.111998500136906 | 32.05522070018105 | 4982835 | 是 | measured |
| linear | bfloat16 | [1, 177, 2560] | [960, 2560] | standalone_float32_io | 32.189012499884484 | 34.07762619983714 | 4982003 | 是 | measured |
| linear | w16a16i | [1, 177, 2560] | [960, 2560] | standalone_float32_io | 27.577886500012028 | 31.296818999703646 | 4997009 | 是 | measured |
| linear | w16a16i_dfp | [1, 177, 2560] | [960, 2560] | standalone_float32_io | 27.503368999987288 | 29.0028472998074 | 4997009 | 是 | measured |
| linear | w8a8 | [1, 177, 960] | [2560, 960] | standalone_float32_io | 9.971217999918736 | 12.985860600224441 | 2529101 | 是 | measured |
| linear | float16 | [1, 177, 960] | [2560, 960] | standalone_float32_io | 18.49885649994576 | 20.373357899688926 | 4957107 | 是 | measured |
| linear | bfloat16 | [1, 177, 960] | [2560, 960] | standalone_float32_io | 21.09211500010133 | 23.554598450141388 | 4956403 | 是 | measured |
| linear | w16a16i | [1, 177, 960] | [2560, 960] | standalone_float32_io | 14.942578500040327 | 16.484361250036272 | 4980241 | 是 | measured |
| linear | w16a16i_dfp | [1, 177, 960] | [2560, 960] | standalone_float32_io | 17.77788899994448 | 19.72216774977369 | 4980241 | 否 | measured |
| linear | w8a8 | [1, 177, 960] | [320, 960] | standalone_float32_io | 3.935719000082827 | 5.018264549971718 | 360780 | 是 | measured |
| linear | float16 | [1, 177, 960] | [320, 960] | standalone_float32_io | 5.694098500043765 | 6.830853699875661 | 655858 | 是 | measured |
| linear | bfloat16 | [1, 177, 960] | [320, 960] | standalone_float32_io | 7.527974499907941 | 10.280044750197703 | 655154 | 是 | measured |
| linear | w16a16i | [1, 177, 960] | [320, 960] | standalone_float32_io | 5.233577499893727 | 6.198017499878006 | 661072 | 是 | measured |
| linear | w16a16i_dfp | [1, 177, 960] | [320, 960] | standalone_float32_io | 5.2802415000314795 | 6.234516150334457 | 661072 | 是 | measured |
| linear | w8a8 | [1, 177, 960] | [960, 960] | standalone_float32_io | 5.634455500057811 | 6.565967749929769 | 980300 | 是 | measured |
| linear | float16 | [1, 177, 960] | [960, 960] | standalone_float32_io | 8.49295200009692 | 9.796819950065583 | 1884658 | 是 | measured |
| linear | bfloat16 | [1, 177, 960] | [960, 960] | standalone_float32_io | 11.82655149978018 | 13.063056749751922 | 1883954 | 是 | measured |
| linear | w16a16i | [1, 177, 960] | [960, 960] | standalone_float32_io | 6.172410000090167 | 11.02422754993313 | 1894992 | 否 | measured |
| linear | w16a16i_dfp | [1, 177, 960] | [960, 960] | standalone_float32_io | 6.107516999918516 | 11.027785899750597 | 1894992 | 否 | measured |
| conv2d | w8a8 | [1, 3, 512, 512] | [768, 3, 16, 16] | standalone_float32_io | 125.64785599988681 | 147.54621400006727 | 4188752 | 是 | measured |
| conv2d | float16 | [1, 3, 512, 512] | [768, 3, 16, 16] | standalone_float32_io | 242.4558180000531 | 243.9053228492412 | 4196918 | 是 | measured |
| conv2d | bfloat16 | [1, 3, 512, 512] | [768, 3, 16, 16] | standalone_float32_io | 249.4697745009944 | 250.50768850001077 | 4196918 | 是 | measured |
| conv2d | w16a16i | [1, 3, 512, 512] | [768, 3, 16, 16] | standalone_float32_io | 227.65462950019355 | 247.63725190032346 | 4219668 | 是 | measured |
| conv2d | w16a16i_dfp | [1, 3, 512, 512] | [768, 3, 16, 16] | standalone_float32_io | 226.13825999906112 | 247.89863609885288 | 4219668 | 是 | measured |
| linear | w8a8 | [1, 1024, 768] | [3072, 768] | standalone_float32_io | 31.99135449995083 | 39.516239300155576 | 2835023 | 是 | measured |
| linear | float16 | [1, 1024, 768] | [3072, 768] | standalone_float32_io | 58.40677199989841 | 71.05201569993369 | 5034805 | 是 | measured |
| linear | bfloat16 | [1, 1024, 768] | [3072, 768] | standalone_float32_io | 52.74598499977401 | 64.62093540019397 | 4847029 | 是 | measured |
| linear | w16a16i | [1, 1024, 768] | [3072, 768] | standalone_float32_io | 39.20276000008016 | 47.24449824968816 | 4859795 | 是 | measured |
| linear | w16a16i_dfp | [1, 1024, 768] | [3072, 768] | standalone_float32_io | 38.743551000152365 | 44.9510350995979 | 4859795 | 是 | measured |
| linear | w8a8 | [1, 1024, 3072] | [768, 3072] | standalone_float32_io | 53.58794749986373 | 70.47568679988672 | 2885263 | 是 | measured |
| linear | float16 | [1, 1024, 3072] | [768, 3072] | standalone_float32_io | 114.07508450020032 | 120.53950700019415 | 5174005 | 是 | measured |
| linear | bfloat16 | [1, 1024, 3072] | [768, 3072] | standalone_float32_io | 122.9608730000109 | 134.43579039985707 | 5074293 | 是 | measured |
| linear | w16a16i | [1, 1024, 3072] | [768, 3072] | standalone_float32_io | 106.71522799998456 | 112.47771465004917 | 5077779 | 是 | measured |
| linear | w16a16i_dfp | [1, 1024, 3072] | [768, 3072] | standalone_float32_io | 106.54315150009097 | 112.48986215007335 | 5077779 | 是 | measured |
| linear | w8a8 | [1, 1024, 768] | [768, 768] | standalone_float32_io | 16.608066500111818 | 20.25232184973902 | 1004494 | 是 | measured |
| linear | float16 | [1, 1024, 768] | [768, 768] | standalone_float32_io | 28.217961999644103 | 33.3955423000134 | 1400052 | 否 | measured |
| linear | bfloat16 | [1, 1024, 768] | [768, 768] | standalone_float32_io | 33.79827349999687 | 38.34326135015545 | 1298868 | 是 | measured |
| linear | w16a16i | [1, 1024, 768] | [768, 768] | standalone_float32_io | 28.035032499929002 | 31.00302190016464 | 1302418 | 是 | measured |
| linear | w16a16i_dfp | [1, 1024, 768] | [768, 768] | standalone_float32_io | 27.98195199966358 | 31.05423660008455 | 1302418 | 是 | measured |

## 融合、转换与 CPU embedding 补充测量

融合 MLP 使用固定 held-out 激活与原始 FP32 ONNX 输出作数值检查；误差仅说明子图输出差异，不代表LIBERO任务质量。注意力核心是固定shape的参数化外代理，输入为合成Q/K/V。QKV混合精度表测整张投影图，包含精度转换和计算成本，不能把差值归因于单一转换算子。CPU embedding行不经过NPU，未计分词或CPU到NPU传输。

| 单元 | 格式 | 输入 shape | 边界 | p50 ms | p95 ms | 文件/权重 B | 稳定 | 状态 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| expert_MLP_layer0 | w8a8 | [1, 50, 720] | rknn_lite2_core0_fused_graph_io | 2.8945784997631563 | 3.6527428010231233 | 4524893 | 否 | measured |
| expert_MLP_layer0 | float16 | [1, 50, 720] | rknn_lite2_core0_fused_graph_io | 6.160015000205021 | 6.249262749042828 | 8910851 | 是 | measured |
| language_MLP_layer3 | w8a8 | [1, 177, 960] | rknn_lite2_core0_fused_graph_io | 11.846821999824897 | 13.99899180050852 | 7522271 | 是 | measured |
| language_MLP_layer3 | float16 | [1, 177, 960] | rknn_lite2_core0_fused_graph_io | 26.92006150027737 | 28.554983950652968 | 14857285 | 是 | measured |
| vision_MLP_layer11 | w8a8 | [1, 1024, 768] | rknn_lite2_core0_fused_graph_io | 32.093413000438886 | 44.260949350427836 | 5293473 | 是 | measured |
| vision_MLP_layer11 | float16 | [1, 1024, 768] | rknn_lite2_core0_fused_graph_io | 101.6799125009129 | 104.51942209856497 | 9938567 | 是 | measured |
| expert_layer0_QKV_projection | w8a8 | [1, 50, 720] | rknn_lite2_core0_fused_graph_io | 2.150507501028187 | 3.039612699922145 | 1229200 | 是 | measured |
| expert_layer0_QKV_projection | float16 | [1, 50, 720] | rknn_lite2_core0_fused_graph_io | 3.057987999454781 | 3.729112148812419 | 2361014 | 是 | measured |
| expert_layer0_QKV_projection | q_fp16+other_w8a8 | [1, 50, 720] | rknn_lite2_core0_fused_graph_io | 2.4939300001278752 | 3.272382200248103 | 1916816 | 是 | measured |
| expert_layer0_QKV_projection | k_fp16+other_w8a8 | [1, 50, 720] | rknn_lite2_core0_fused_graph_io | 2.3542275002910173 | 3.1549919991448405 | 1461136 | 是 | measured |
| expert_layer0_QKV_projection | v_fp16+other_w8a8 | [1, 50, 720] | rknn_lite2_core0_fused_graph_io | 2.244857500954822 | 3.1636243497814576 | 1461136 | 是 | measured |
| expert_layer0_QKV_projection | qk_fp16+v_w8a8 | [1, 50, 720] | rknn_lite2_core0_fused_graph_io | 2.659651500835025 | 3.3461410503150546 | 2144848 | 是 | measured |
| expert_layer0_QKV_projection | qv_fp16+k_w8a8 | [1, 50, 720] | rknn_lite2_core0_fused_graph_io | 2.6093619990206207 | 3.3038510009646416 | 2144848 | 是 | measured |
| expert_layer0_QKV_projection | kv_fp16+q_w8a8 | [1, 50, 720] | rknn_lite2_core0_fused_graph_io | 2.415558000393503 | 3.2222036013990873 | 1689168 | 是 | measured |
| attention_QK_softmax_PV_proxy | float16 | [3, 1, 15, 50, 48] | rknn_lite2_core0_parameter_free_attention_proxy | 4.498318499827292 | 5.325915300090856 | 204085 | 是 | measured |
| attention_QK_softmax_PV_proxy | w8a8 | [3, 1, 15, 50, 48] | rknn_lite2_core0_parameter_free_attention_proxy | 3.7019584997324273 | 4.800763499861205 | 239695 | 是 | measured |
| language_token_embedding_cpu_lookup | native_bf16_row_lookup | [1, 177] | cpu_numpy_lookup_and_decode_excludes_cpu_to_npu_transfer | 0.22078150050219847 | 0.24761399981798604 | 94617600 | 否 | measured |
| language_token_embedding_cpu_lookup | cpu_int8_row_lookup | [1, 177] | cpu_numpy_lookup_and_decode_excludes_cpu_to_npu_transfer | 0.6390124999597901 | 0.7470844508134178 | 47505920 | 是 | measured |

## 当前完整部署：RKNN＋CPU

以下从既有真实板端记录生成：单条开发观测完整回放预热1次、计时3次；闭环行来自4任务22次请求。三张RKNN图均为FP16构建，CPU保留浮点处理。不是最终HAQ/QAT/PTQ结果。

| 单元 | 后端 | 每动作块调用次数 | p50 ms/调用 | p95 ms/调用 | 测量次数 | 文件 B |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| vision_connector | rknn_2.3.2 | 2 | 1905.2817 | 2188.4714 | 6 | 212621173 |
| prefix_with_kv | rknn_2.3.2 | 1 | 613.8538 | 616.6909 | 3 | 326154086 |
| expert_step_v2 | rknn_2.3.2 | 10 | 323.8783 | 377.2159 | 30 | 217425832 |
| raw_preprocess | cpu_numpy_tokenizers | 1 | 144.0578 | 144.1368 | 3 | — |
| prefix_glue | cpu_numpy | 1 | 2.8798 | 3.0035 | 3 | — |
| time_embedding | cpu_numpy | 10 | 0.6944 | 0.8323 | 30 | — |
| euler_integration | cpu_numpy | 10 | 0.0799 | 0.1086 | 30 | — |
| action_postprocess | cpu_numpy | 1 | 0.1062 | 0.1101 | 3 | — |
| full_raw_policy | rknn_2.3.2+cpu | 1 | 8028.9742 | 8162.1893 | 3 | 945564365 |
| full_policy_closed_loop | rknn_2.3.2+cpu | 1 | 7444.5892 | 7630.8421 | 22 | 945564365 |

视觉共享同一模型文件、每动作块运行两次；前缀一次；专家十次。表内完整流程行已经包含子阶段，不能再次相加；单阶段p50之和不等于完整流程p50，p95/RSS也不能求和。
三张图共756,201,091 B，CPU参数文件189,363,274 B，合计945,564,365 B，未达到原checkpoint至少40%压缩要求；大小未包含tokenizer/配置等资产。CPU参数不按重复调用复制计费。
阶段调用次数是执行图次数，不保证各参数算子都逐次执行；独立精度控制边界仍待导出图核查。Norm/融合节点不凭缺少独立测量固定精度。
Lite2图调用时间含输入准备、运行及输出交接；纯CPU↔NPU拷贝/格式转换没有独立计时，不从总耗时相减推算。CPU分项为实测合并阶段，未把分词/lookup/状态投影拆出。
混合精度整图配置成本仍未测量；现有100个基础签名和18项补充可作代理查表，不能把FP16流程参考行作为任意配置的已测收益。
[完整部署证据](../experiments/2026-10-01-board-libero-closed-loop.md) · [表格更新记录](../experiments/2026-10-01-deployment-hardware-tables.md)

