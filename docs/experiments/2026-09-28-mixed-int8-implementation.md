# 扩展混合精度 PTQ：291 Linear + INT8 embedding

## 目标与边界

目标为保持 SmolVLA 原结构、原始 checkpoint 独立 PTQ，实际 checkpoint 相比 906,712,520 B 缩小至少 40%。这是 GPU 真 INT8 推理路径，尚非 RKNN 整模型、QAT 或质量达标结论。质量仍以配对 LIBERO 闭环为准。

## 配置与原理

`scripts/ptq_mixed_int8.py` 复用 `config/quantization_map_v0.json` 中 W8A8 分组，选择 291 个 Linear；额外将 token embedding 按行 INT8 存储。视觉 MLP11、语言 MLP3、connector、patch、位置编码、Norm、动作和时间接口、lm_head 均保留原 checkpoint 存储，运行时保留模型原加载精度。没有在此步骤将保护模块统一转换 FP16。

权重：每输出通道 scale=max(abs(W))/127，q=clip(round(W/scale),-127,127)。Embedding 使用相同规则，每词条一行，仅反量化被查询的行，不展开整个表。Bias 保留 FP32。

激活：40 个固定 calibration episode 首帧，逐 Linear 输入统计 min/max，范围先包含零；s=max((max-min)/255,1e-12)，z=clip(round(-128-min/s),-128,127)。前向 xq=clip(round(x/s)+z,-128,127)，CUDA `torch._int_mm` 实际 INT8 乘法、INT32 累加；减去 z*sum(Wq) 再按 scale 反量化，输出回到输入 dtype。token 行数补齐到至少 32 且为 8 的倍数后裁回；未对中间 Softmax、残差或 cache 新增 INT8 处理。

这是固定 min/max 候选，没有声称执行 KL/MMSE 截断扫描。量化后保存所有 INT8 权重、scale、zero point、INT32 权重和，严格重载，然后用隔离开发 episode 做同噪声动作冒烟测试。40% 文件大小是硬检查，动作 MAE 只记录，不设置通过阈值。

## 数据与复现

模型 SHA256 与分区沿用 v0；脚本在启动时验证模型、split、partition 哈希及 calibration/development episode 隔离。产物 `calibration.json` 保存校准 episode、逐层范围、输入 dtype、shape 和调用次数；`report.json` 保存 manifest、模型/分区/配置/产物哈希、真实字节、压缩率和重载状态；`smoke_actions.npz` 保存 FP/PTQ 动作。

服务器 `/root/qvla`：

```bash
.venv/bin/python scripts/ptq_mixed_int8.py --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets --dataset-root data/libero --splits data/libero_splits.json --partition config/evaluation_partition_v2.json --map config/quantization_map_v0.json --output-dir runs/mixed_int8_v1 --calibration-tasks 40 --smoke-tasks 2
```

复用 40 任务评测入口（每任务 1 回合；输出必须使用新的目录，避免旧结果被复用）：

```bash
.venv/bin/python scripts/run_real_w8a8_libero_suites.py --mode ptq --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets --pack-report runs/mixed_int8_v1/report.json --output-dir runs/mixed_int8_v1_libero_seed0 --seed 0
```

`eval_real_w8a8_libero.py` 根据 mixed-int8-v1 manifest 替换模块，旧专家量化报告仍走原路径。若环境变化，需同协议重跑 FP。

## 当前验证

本地语法检查完成。服务器 A10、torch 2.7.0a0+7c8ec84dab.nv25.03 已连通。完整校准、打包重载与数值测试结果待下文实测更新。尚未启动本配置的 40 任务闭环。正式使用前须确认 report 中 reload_smoke=passed 和 meets_40_percent=true。

## 实测结果

A10 上完成 40 条校准输入、原始 FP 两条开发动作参考、真实打包、严格重载及两条量化开发动作。总耗时 62.28 秒（不含代码适配、上传、闭环）。

- 原模型 906,712,520 B；PTQ 文件 **530,903,920 B**，减少 **41.4474%**，通过 40% 大小门槛。
- 产物 SHA256：`1747dd458e96c8f3a99a1f85ee0ab12eeaad716d4bf312dfd3554dae4b21b540`。
- 291 个真实 INT8 Linear 和 1 个 INT8 embedding；严格 state_dict 重载成功。
- 两条开发动作均有限，动作块对 FP 的 MAE=0.008173462；样本太少且为代理指标，不能据此称质量达标。
- 本地报告：`runs/mixed_int8_v1/report.json`、`calibration.json`、`smoke_actions.npz`；权重仍在服务器同名目录，未额外下载。

额外运行 `scripts/test_mixed_int8_runtime.py` 时，发现 torch._int_mm 不接受补齐到 8 的极短序列（报行数须大于 16）；已将最小补齐行数改为 32。实际动作冒烟中的序列未触发该错误。修复后 **15 组 CUDA 整数 GEMM 数值对照**（FP32/BF16/FP16 × 行数 1/7/8/17/50）和 **3 组 embedding dtype 对照**全部通过，逐元素与相同整数公式的参考相等。

尚未运行本配置的 40 任务闭环、延迟对照或 RK3588 全图测试。41.45% 是持久化 checkpoint 文件缩减，不代表显存或 RKNN 文件缩减；层边界仍返回浮点激活。

## 40 任务闭环评测完成

服务器后台 runner 已完成四组，每任务 1 回合、seed 0、256×256；沿用配对策略噪声规则，FP 为历史同协议结果，未在本轮同步重跑。前两组完成后 runner 停止；后台续跑复用两组完整结果，完成 Goal 和 Libero-10。

| suite | FP | 混合 PTQ |
| --- | ---: | ---: |
| Spatial | 9/10 | 6/10 |
| Object | 9/10 | 9/10 |
| Goal | 7/10 | 5/10 |
| Libero-10 | 7/10 | 4/10 |
| 总计 | 32/40 | 24/40 |

成功率 80%→60%，降低 20 个百分点。逐任务有 10 个新增失败、2 个新增成功；Object 总数相同但任务结果并不完全相同。具体任务 ID 与输入报告 SHA256 见 `runs/mixed_int8_v1_libero_seed0/comparison.json`；汇总脚本 `scripts/summarize_mixed_ptq.py`。报告 SHA256：`78e26227578f0ead58e362ad9a32e1993be7679dda8b1cd67c73df183524f8bb`。

结论：实际文件压缩 41.45% 达标，但本次开发闭环出现明显退化，不能接受为最终质量方案。每任务单回合不足以量化稳定成功率；也不能从任务失败直接归因某一层。后续先做分组高精度恢复定位，并保持文件预算，再考虑 QAT/蒸馏补偿。完整板端、当前配置 QAT 尚未测量。
