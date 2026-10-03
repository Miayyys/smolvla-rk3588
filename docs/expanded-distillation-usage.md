# 扩展蒸馏使用说明

入口读取 `config/distillation_expanded_v3.json`，一次只执行一个阶段；当前未开始完整训练。下列命令在服务器 `/root/qvla` 内运行，不受本地fish语法影响。

先检查命令，不执行：

```sh
.venv/bin/python scripts/run_expanded_distillation.py --stage fp --dry-run
```

准备→教师标注→契约过滤→逐帧审计（分别运行，不能跳过审计）：

```sh
.venv/bin/python scripts/run_expanded_distillation.py --stage prepare
.venv/bin/python scripts/run_expanded_distillation.py --stage label
.venv/bin/python scripts/run_expanded_distillation.py --stage review
.venv/bin/python scripts/run_expanded_distillation.py --stage audit
```

`review`只筛掉格式/控制范围错误，不能自动认定语义质量；需要排除具体动作时：

```sh
.venv/bin/python scripts/review_teacher_labels.py --cache runs/teacher_actions_expanded_v3 --exclusions runs/teacher_exclusions_v3.json --output runs/teacher_actions_expanded_v3/review.json
```

排除文件包含 `teacher_actions_sha256` 与 `exclusions`，每条为 `{"row": 12, "timesteps": [3,4], "reason": "具体核验依据"}`。row对应manifest数组位置，不是任务ID，不能按开发任务失败批量删除训练标签。没有具体依据时不能把契约筛选结果称为质量审核通过。

FP训练（待用户知悉代码验证结果，并完成真实扩展标签）：

```sh
.venv/bin/python scripts/run_expanded_distillation.py --stage fp
```

教师标注续跑：`--stage label --resume`。FP训练从最新保存点恢复：`--stage fp --resume`；具体检查点路径可放在 `--resume PATH`。训练恢复不能改总更新预算、lr图或标签，否则身份检查拒绝；重新配置实验应另建输出目录。

每250更新保存完整训练状态及最近两份FP master。训练状态含模型/AdamW动量，约数GB，临时保存同时需要额外空间；启动前确认磁盘够用。`development/step_XXXXXX`保存完整有效动作块评价，未做真实整数转换的QAT中间快照只是训练master。完整FP训练后，可对选定FP run用 `eval_distill_qat_libero.py --qvla-panel long10 --qvla-mode distilled_fp --qvla-run PATH`，沿用长任务配对协议，并检查其他suite回归。

仅当已选定有效FP蒸馏模型，才指定 `--stage qat --fp-run PATH`。当前入口会明确拒绝正式RKNN硬件感知QAT：任意v2混合整图等价性仍未通过。显式 `--allow-unverified-backend-diagnostic` 才能做本地数值QAT，不得将其称为板端完整量化。启动完整QAT前先向用户汇报，当前没有启动该阶段。

代码与实际小测试记录见[扩展训练控制](experiments/2026-10-02-expanded-distillation-code.md)。
