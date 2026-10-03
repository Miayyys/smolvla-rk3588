# 真实 V1 QAT 语言模块转 RKLLM

## 结论

选定的 V1 无教师损失 QAT master 的全部16层语言权重已提取、转换为FP16 RKLLM，并在RK3588成功执行。关闭dump，官方embedding回调输入真实177×960特征，保存并解析全部16层原生K/V；没有CPU重算投影。

**尚不能等价替换SmolVLA语言模块。** 当前RKLLM因果注意力和连续位置不等价于SmolVLA分块双向mask与跳过padding的位置。真实权重对照已证实这会引起明显输出变化。没有为绕开这个问题引入多次重跑、CPU attention或逐层主机搬运，也未把此结果用于机器人闭环或更新HAQ成本表。

## 输入、配置与产物

- QAT master SHA256：`4aeb92854d2bb89bac84a2d791d2acb4389b934178948c05533d6a52fe0b9f81`，服务器`runs/qat_distilled_v1_no_teacher_v1/qat_float_master.safetensors`。
- 固定输入沿用`runs/qat_v1_rknn_deploy_v1/prefix_boundary.npz`，episode18/task0/frame0，开发回放；参考`prefix_reference.npz`。本次没有使用任务测试数据校准或训练。
- 16层，hidden960，FFN2560，Q15头/KV5头，head64，RMSNorm eps1e-5。
- **RoPE theta必须是10000**：来自SmolVLA实际`apply_rope`，不能照搬VLM资产config的100000。
- 采用Llama容器严格核对146个语言参数键；未重训练、未删层。为满足CausalLM容器提供绑定embedding的输出头，VLA不消费生成的token/logits。
- Toolkit/runtime1.3.1，RK3588，FP16不量化，3 NPU核，max_context256，`export_embedding=False`。转换环境沿用已有overlay，依赖尚非全部官方pin，因此不作SDK通用正确性断言。
- TOKEN IDs100..276经embedding回调返回对应真实特征；GENERATE/1，keep_history1，save_prompt_cache1，dump0。本次新进程运行，不跨观测复用相同占位ID的cache。
- 编译load/build/export均返回0；产物419,212,100B，SHA256 `f628b85805bd130fb61eb29e7b30dc37af3e0605e369a4a070f7e9f3a16bfc03`。仅语言FP16实验产物，不是全模型压缩率。
- 原始数据、日志、各文件hash：`runs/selected_language_rkllm_v1/experiment_manifest.json`。板目录`/root/qvla_board_test/selected_language_rkllm_v1/`。

## 最小必要数值对照

提取后的FP32语言容器使用原始mask/position，与原始选定QAT master语言输出比较；再仅切换到原生连续位置/causal mask建立RKLLM参考。MAE为逐元素绝对误差均值。

| 对照 | 实测 |
| --- | --- |
| FP32提取容器＋原mask vs原语言模块 | hidden MAE 2.4024e-7；全部hidden/K/V中最大MAE 6.2081e-7、最大绝对误差4.3273e-5 |
| FP32提取容器＋causal/连续位置 vs原语言模块 | hidden MAE0.141877，最大绝对误差9.39646 |
| RKLLM原生K/V vs对应causal FP32参考 | 32个K/V张量MAE的均值0.00553478；最大绝对误差1.86270 |
| RKLLM原生K/V vs原mask FP32参考 | 32个K/V张量MAE的均值0.104401；最大绝对误差10.11941 |

RKLLM相对causal参考仍有累计数值误差，不能表述为严格对齐。第一层V MAE约1.2e-5，K约0.000182，后层增大；详细数据见`native_cache_report.json`。目前无需在mask尚不等价时继续位宽扫描或40任务测试。

## 板端执行与缓存

INIT/RUN/CACHE_RC均为0，177tokens；runtime分64/64/49调用embedding回调。单次`rkllm_run`（含保存缓存）852.727ms，**未计初始化、主机缓存读取与解析，也不是预热重复统计或整策略延迟**。CPU affinity err22警告仍存在，不据此声称正式速度提升。未测本次峰值RAM和完整动作质量。

缓存3,832,049B；解析header为(0,8,177)，第二字段不能误当token数。末尾固定标记和32条记录元数据逐条检查，K记录(1,640,0)，V记录(1,2,320)。K的RoPE交错布局转为[1,5,177,64]，V由head/channel/token转为同形状。这是经过实际数值对照的固定版本/形状解析，不是官方承诺的通用文件格式。解析在本地完成。

传输途中板磁盘满：删除本次不完整文件，并在逐一验证本地/板端SHA一致后，清除板上五个微型RKLLM探针模型副本；本地模型、日志/缓存与原部署模型保留。

## 脚本与后续边界

- `scripts/export_smolvla_rkllm_language.py`：限定选定master hash、提取严格键集合、修正实际RoPE配置。
- `scripts/compile_smolvla_rkllm_language.py`：FP16实际转换及逐阶段返回码。
- `scripts/prepare_rkllm_language_reference.py`：原mask与causal参考。
- `scripts/run_rkllm_selected_language.cpp`：固定真实输入、官方embedding/cache接口。
- `scripts/check_rkllm_selected_cache.py`：校验记录并比较原生K/V。

官方现公开C接口没有传入语言self-attention二维mask/position_ids的入口；`custom_config`文档用于架构节点映射，`encoder_mask`属于cross-attention，均不能直接当作此问题的开关。此前微型模型设置noncausal metadata未改变执行，本轮不重复该无效办法。

下一步若继续RKLLM路线，前提是找到并验证原生支持这些attention/position语义的接口或后端实现；仅“能转成rkllm”已验证，不再为这件事重复试验。现有RKNN语言路径保留。未经授权未向官方发送issue。
