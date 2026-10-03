# V1无教师QAT原精度例外：真实RKNN转换

## 结果与范围

GPUServer恢复后核对所选master SHA256 `4aeb92854d2bb89bac84a2d791d2acb4389b934178948c05533d6a52fe0b9f81`，与V1无教师损失QAT一致。不是有教师版本或原始FP权重。

真实专家/语言ONNX各拆成前段、独立例外投影、后段。专家保留BF16 gate，语言独立编译INT16 DFP down；其他投影按V1 W8A8，专家V投影FP16，CPU逐行INT8 embedding与状态参数沿用所选pack。**全部六个新图已编译，但未板测。** 保留模块dtype不等于与本地QAT quantizer数值等价，RKNN校准、融合及图边界仍可能改变动作。

## 图与精度

原ONNX校验：专家SHA `6fe93aa6886d982c69ff64cffdc1494514376e565192ca18b0a024bcfed90878`，语言SHA `d07849b6ff23800479b659096e164ecbaf3b506d0246dc574b6648646b8ca613`。原图及其精度例外元数据保存在服务器 `runs/qat_v1_rknn_deploy_v1`。

`scripts/split_v1_projection.py` 验证真实专家velocity与语言全部33输出的浮点分割parity，rtol/atol1e-4；实际每项MAE/max均0。保留跨切点残差、mask和其他仍需使用的张量。语言早期K/V已由前段产生，运行时直接保留这些输出，不再作为后段的无计算输入/输出穿过另一次量化。对应 `forwarded_outputs` 和原始输出顺序明确记录在split_report。

编译日志实测：

- 专家V3：`Conv:/v_proj_3/MatMul#2`，FLOAT16 NPU。
- 专家gate10：`Conv:/mlp/gate_proj_10/MatMul#2`，BFLOAT16 NPU；输入720、输出2048，chunk50。
- 语言down3：`Conv:/mlp/down_proj_3/MatMul#2`，INT16 NPU；输入2560、输出960、prefix177。此独立图的SDK配置为 **quantized_dtype=w16a16i_dfp, quantized_method=layer**，不是只把原W8A8混合profile输出改成普通INT16。

未声称DFP的具体scale、饱和与舍入同GPU完全相等；后续比较真实GPU pack完整动作。

## 校准与环境

服务器A10 23GB，29GiB内存。此次图切分、ORT校准边界传播和RKNN编译主要使用CPU。磁盘只剩约3.2GB，因此中间文件位于 `/dev/shm/qvla_v1_exact_precision`，最终rknn、报告和日志持久化到 `/root/qvla/runs/v1_exact_precision_v1`。

重新确认环境：`.rknn-probe`具有RKNN/ONNX/ORT；`.rknn-venv`没有ONNX/ORT，不能依名字猜测环境。最初SFTP方式scp失败，改用scp -O；最初调用不具备ONNX的环境失败，均未进入模型计算。

沿用原40个隔离PTQ calibration episode，专家取去噪0/5/9共120组，语言40组。`prepare_v1_partition_calibration.py` 按原输入列顺序，运行前段和独立投影的FP ORT，生成后续分区的校准输入；原始K/V输入路径复用。源episode列表、dataset hash、逐行hash和分区dataset hash记录在两个calibration_report。未使用本次开发回放或闭环观测校准。

W8A8主体采用SDK channel/normal/optimization_level3；BF16投影不做整数校准；INT16 DFP投影单独校准。每个compile_report保留精确配置、源图/产物hash、实际字节和耗时。

## 实际文件大小

| 文件/部分 | 实际字节 | MB（十进制） |
| --- | ---: | ---: |
| 专家前段 | 75,667,745 | 75.67 |
| 专家BF16 gate | 2,977,778 | 2.98 |
| 专家后段 | 38,529,886 | 38.53 |
| 专家合计 | 117,175,409 | 117.18 |
| 语言前段 | 39,886,496 | 39.89 |
| 语言INT16 DFP down | 4,994,041 | 4.99 |
| 语言后段 | 127,046,898 | 127.05 |
| 语言合计 | 171,927,435 | 171.93 |
| 原V1 W8A8视觉（复用） | 111,973,263 | 111.97 |
| 所选GPU pack CPU参数（复用） | 47,546,790 | 47.55 |
| 七图＋CPU参数 | **448,622,897** | **448.62** |

六个新图合计289,102,844 B，需要传输约289.10MB；视觉/CPU参数在本地与板端旧部署目录已存在。七图＋CPU参数对原checkpoint906,712,520 B减少 **50.5220%**。这是参数资产大小；完整运行包还需配置、分词器及运行代码，尚未统计新自包含运行包的精确字节。模型文件大小不代表运行峰值内存。

专家三个转换阶段合计112.06s，语言107.37s；两条链的开始时刻不同，不能将两耗时之和解释为整体墙钟耗时。每条内部顺序编译；仅删除新内存盘工作目录中的重复check*.onnx快照，原图、profile、校准数据、最终图与日志保留。

## 板端接口与后续验证

`smolvla_rknn_partitions.py` 按hash验证六个子图，按显式输入/输出名字路由活跃张量。`smolvla_board_runtime.py` 增加可选partitioned_graphs；原三图/RKLLM入口仍可运行。新接口一次动作块实际包含2次视觉＋3次语言＋30次专家，共35次RKNN inference，**这只是调用次数，不是测出的延迟**。

`verify_v1_partitioned_replay.py` 准备warmup1/repeats3，比较所选GPU pack动作MAE/max、夹爪符号、重复误差、实际时延及进程maxRSS；随后做少量配对短任务，不直接套用GPU成绩。

转换完成时，用户此前指定大文件下载/上传自行操作，因此当时仅取回小报告，未代下载六个模型。随后用户完成传输，板端完整回放已开展，接口修复及实测结果见[板端原精度执行记录](2026-10-03-v1-native-precision-board.md)。`stage_v1_partitioned_board.py` 先验证下载文件SHA，再用可断点续传rsync上传；板端root盘满，使用 `/dev/shm/qvla_v1_exact_precision_v1`，原量化CPU/处理器/视觉资产只引用已存在文件。内存盘部署重启后消失；最终模型和记录仍在服务器持久磁盘。

完整部署manifest及参数清单在本地/服务器 `runs/v1_exact_precision_v1`。后续须检查图边界量化、完整动作误差和拆分成本，不能仅因位宽与文件约束满足就称质量合格。
