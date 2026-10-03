# 教师动作蒸馏接口与 v2 QAT 训练链路

## 状态与边界

蒸馏代码已实现；训练数据导出及 FP→QAT→本地整数打包链路已验证。**真实 OpenVLA-OFT 教师推理尚未验证，正式蒸馏没有开始，效果提升未测量。** 两步服务器训练使用显式 `synthetic_test_only` 零动作标签，只用于检查训练接口，不能作为教师蒸馏效果证据。默认训练拒绝合成标签，显式测试开关最多允许两步。

## 实现与原理

- `scripts/prepare_openvla_teacher_inputs.py`：仅导出 qat_train 去除开发划分后的观察，按官方 LIBERO-10 任务描述匹配；不是按假定 suite 顺序切任务号。本次匹配到数据集 task_index 0–9，实际观察 state 为8维、两幅256×256图像，10 Hz。
- `scripts/cache_openvla_teacher.py`：在独立教师环境调用官方 `get_vla`、`get_processor`、动作头及 proprio projector；需要完整 `moojink/openvla-oft` 源码和 `moojink/transformers-openvla-oft` fork。当前学生环境缺少教师依赖，不能直接运行该步骤。模型/处理器的小配置放在工作副本，原始核验权重保持作为输入。
- `qvla_haq/distillation.py`：校验标签 hash、划分 hash、episode 范围、suite、动作维度及采样率；教师输出先反归一化为 simulator 7维动作，再使用学生 processor 归一化。
- `scripts/qat_train_haq.py --mode fp-distill`：关闭所有假量化，用浮点计算、FP32 master 更新学生；输出 `distilled_float_master.safetensors`。仅 v2 中可配置的304个执行模块及其bias参与训练，其他参数冻结。
- 同一脚本 `--mode qat --initial-master ... --teacher-cache ...`：严格加载蒸馏后的 master，启用冻结的 v2 全模块精度图，训练后保存浮点 master 与真实本地整数打包产物，严格重载及固定噪声动作一致性检查。

教师 checkpoint `openvla-7b-oft-finetuned-libero-10` 只覆盖 LIBERO-10，不能推断它在其他三组也更强。教师8步动作仅监督学生50步中的匹配前缀；后续步仍保留真实数据动作作为上下文，教师损失将其 mask 掉，不重复教师短块。剩余不足8步时同时应用 episode padding mask。

令共享噪声为 ε、流时间为 t，使用学生原生 flow matching：

\[
L=L_{GT}+\lambda L_T,\quad\lambda=0.2
\]

GT 与教师分支使用完全相同的 ε、t。原生损失对有效时间步和7个真实动作维度取均值。非教师 suite 只计算 GT 损失。λ=0.2 是初始训练配置，尚未通过质量试验选优。

图像契约已在[后续核对](2026-10-01-real-teacher-environment-preparation.md)中修正：数据集图像不再旋转，仅实时模拟器原图旋转180°；之后使用官方 resize/crop。夹爪按官方路径做 `-sign(2g-1)`。这些规则已按源码实现，**仍须真实教师首样本检查图像朝向、动作尺度/方向、输出和闭环质量**。10 Hz 是数据步长契约，不证明实际机器人物理控制频率已验证。

官方源码：[推理接口](https://github.com/moojink/openvla-oft/blob/main/experiments/robot/openvla_utils.py)、[LIBERO 评测](https://github.com/moojink/openvla-oft/blob/main/experiments/robot/libero/run_libero_eval.py)。实际教师运行时记录源码 Git revision、Transformers fork 来源和 checkpoint 文件 SHA。

## 输入与复现

学生、候选和划分 hash 沿用 [v2 QAT 准备记录](2026-10-01-haq-v2-qat-preparation.md)。教师 checkpoint 文件25个、15939159245 B，全部 SHA256 已核对；仅完整文件不等于教师推理可用。

本地数据导出验证：

```bash
.venv-haq-local/bin/python scripts/prepare_openvla_teacher_inputs.py \
  --dataset-root artifacts/transfer/libero --splits data/libero_splits.json \
  --partition config/evaluation_partition_v2.json \
  --output runs/distill_teacher_inputs_smoke_v1 --samples-per-task 1
.venv-haq-local/bin/python -m unittest discover -s tests -p test_distillation_contract.py -v
```

结果：10条观察、10个LIBERO-10任务；3项测试通过，覆盖排除评测 episode、拒绝合成/被改写标签、短时域padding/学生归一化、共享流噪声/时间及梯度传递。观察 manifest 和原始npz位于忽略目录 `runs/distill_teacher_inputs_smoke_v1`。

服务器教师环境就绪后，按顺序运行（`TEACHER_PYTHON` 替换为独立教师环境的 Python；以下正式命令尚未运行）：

```bash
cd /root/qvla
.venv/bin/python scripts/prepare_openvla_teacher_inputs.py \
  --dataset-root data/libero --splits data/libero_splits.json \
  --partition config/evaluation_partition_v2.json \
  --output runs/teacher_inputs_v1 --samples-per-task 32
TEACHER_PYTHON scripts/cache_openvla_teacher.py \
  --teacher-repo third_party/openvla-oft \
  --checkpoint artifacts/teacher/openvla-oft-libero-10 \
  --inputs runs/teacher_inputs_v1 --output runs/teacher_actions_v1
.venv/bin/python scripts/qat_train_haq.py --mode fp-distill \
  --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets \
  --dataset-root data/libero --splits data/libero_splits.json \
  --partition config/evaluation_partition_v2.json --candidate config/haq_candidate_v2.json \
  --teacher-cache runs/teacher_actions_v1 --steps 40 --output-dir runs/distill_fp_v1
.venv/bin/python scripts/qat_train_haq.py --mode qat \
  --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets \
  --dataset-root data/libero --splits data/libero_splits.json \
  --partition config/evaluation_partition_v2.json --candidate config/haq_candidate_v2.json \
  --teacher-cache runs/teacher_actions_v1 \
  --initial-master runs/distill_fp_v1/distilled_float_master.safetensors \
  --steps 40 --output-dir runs/distill_qat_v1 --allow-unverified-backend-diagnostic
```

40步只是首轮训练诊断预算，不能据训练 loss 称收敛。先完成教师接口检查，再比较独立开发集；完整 RKNN 混合图未验证，因此 QAT 保留诊断标记。独立 PTQ 仍从原始 FP checkpoint 开始，不能把合成测试产物当正式模型。

本地教师预检查实际失败：缺少 `timm、peft、sentencepiece、tensorflow、json_numpy、diffusers`，且参考目录不是完整教师源码。尚需独立教师环境和官方 Transformers fork；不得升级/替换学生环境来绕过检查。代码 hash 记录于 `runs/distill_chain_server_evidence_v1/code_sha256.json`。

## A10 两步链路实际结果

精确命令原件：服务器 `scripts/distill_smoke.sh`；本地证据 `runs/distill_chain_server_evidence_v1`（fp_report、qat_report、reload_parity、原始日志）。训练 seed29，lr1e-5，AdamW weight_decay0，梯度clip1，GT+0.2教师损失；标签为10×8×7全零且明确标记为合成。

| 检查 | FP 蒸馏接口测试 | v2 QAT 接口测试 |
|---|---:|---:|
| 训练步数 / 教师损失参与步数 | 2 / 2 | 2 / 2 |
| 五阶段均有有限正梯度 | 是 | 是 |
| 严格重载完整动作 MAE / max abs | 0 / 0 | 0 / 0 |
| 用时（包含加载/保存） | 28.50秒 | 38.25秒 |
| QAT CUDA峰值 | — | 14231008256 B |
| 本地打包大小 / 相对原始文件缩小 | 未打包浮点master | 514073656 B / 43.3036% |

FP master SHA：`e7723e6a411e4b388d9e139f97f33de58518a98548abe01f18e0226eecb70d87`。

QAT pack SHA：`4afde4549cda0641731035074690b774509a01a67d3159d17850228a2d1c8bcc`。

两个loss取自不同观察，不能把它们的下降当质量改善。文件缩小只属于本地打包，不代表 RKNN 体积或板端收益。教师真实标签、训练后任务成功率、真实 RKNN 转换、独立 PTQ 对照、端侧速度和资源收益均未测量。

## 后续真实教师验证

本页合成标签测试只证明接口。后续真实教师环境、10观察80动作契约、两组FP40步→QAT40步及严格打包重载已经通过，见[真实实验](2026-10-01-real-teacher-distill-qat.md)。质量评价改为[完整有效动作与实际闭环](2026-10-01-distill-qat-quality-protocol.md)，不凭首步指标选最终模型。
