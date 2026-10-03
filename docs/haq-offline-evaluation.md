# HAQ 快速离线动作评价：本地与服务器

代码：`qvla_haq/offline_actions.py` 与 `scripts/haq_offline_eval.py`。先在固定开发观测上缓存FP动作，之后候选仅运行自身前向。结果包含整体动作偏差、连续控制偏差、夹爪方向差异及首步示范动作偏差。它们不是LIBERO任务成功率；当前速度查表聚合和任意全精度图应用仍未接通，RL正式搜索没有启动。

## 本地已测环境

独立环境 `.venv-haq-local`，PyTorch2.7.1+cu118、LeRobot0.6.1，RTX4060 8GB。依赖来自本地已有wheel，无需重新下载。`esp-ml`不受影响。数据和模型位于 `artifacts/transfer/`。

```bash
.venv-haq-local/bin/python scripts/test_haq_offline_actions.py

.venv-haq-local/bin/python scripts/haq_offline_eval.py cache \
  --model-dir artifacts/transfer/model \
  --vlm-assets-dir artifacts/transfer/smolvlm2_assets \
  --dataset-root artifacts/transfer/libero \
  --phases 1 --device cuda --output runs/haq_local_new/fp_cache40

.venv-haq-local/bin/python scripts/haq_offline_eval.py evaluate \
  --model-dir artifacts/transfer/model \
  --vlm-assets-dir artifacts/transfer/smolvlm2_assets \
  --cache runs/haq_local_new/fp_cache40 \
  --pack-report runs/expert_real_w8a8_v1/report.json --mode ptq \
  --device cuda --output runs/haq_local_new/expert_ptq40
```

以上也适用于fish。缓存和候选输出目录须是新的目录，防止旧结果覆盖。`--phases 1`是覆盖40任务的40条快速面板，任务间轮流选择开始/中间/90%帧；`--phases 3`是120条扩展面板。把 `--mode ptq` 改为 `qat` 可测已有QAT产物；不传 `--pack-report` 则进行FP重载对照。目前两个真实量化加载器用于接口验证，均不代表完整HAQ位宽图。

## 服务器开机后的上传

已备好 `artifacts/haq_offline_eval_upload.tar.gz`，仅包含代码、分区/split和文档；不包含环境、模型权重、数据视频、动作缓存。服务器仍使用已有 `.venv` 和模型数据，运行前核查版本与CUDA。

```bash
scp -O -F /home/loser/.ssh/config artifacts/haq_offline_eval_upload.tar.gz GPUServer:/root/qvla/
ssh -F /home/loser/.ssh/config GPUServer 'cd /root/qvla && tar -xzf haq_offline_eval_upload.tar.gz'
```

服务器 `/root/qvla` 下建立自身FP缓存：

```bash
.venv/bin/python scripts/haq_offline_eval.py cache \
  --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets \
  --dataset-root data/libero --device cuda --phases 1 \
  --output runs/haq_offline_server_v1/fp_cache40

.venv/bin/python scripts/haq_offline_eval.py evaluate \
  --model-dir artifacts/model --vlm-assets-dir artifacts/smolvlm2_assets \
  --cache runs/haq_offline_server_v1/fp_cache40 \
  --pack-report runs/expert_real_w8a8_v1/report.json --mode ptq \
  --device cuda --output runs/haq_offline_server_v1/expert_ptq40
```

模型、processor、tokenizer/config、分区、torch版本、device或flow步数改变时重建FP缓存。候选文件的大小和SHA必须匹配pack report；路径迁移时可用 `--checkpoint` 指定实际文件。脚本强制HF离线模式，不会重新下载权重。

## 仅用已有动作文件重算

本地系统Python无需torch或LeRobot也可执行：

```bash
python3 scripts/haq_offline_eval.py replay \
  --actions runs/expert_real_w8a8_action_v1/actions.npz --candidate-key ptq \
  --output runs/haq_offline_local_v1/archived_ptq_rescore.json
```

这是历史数据重分析，没有新的模型前向或任务评测。[实际配置、原始哈希及本机计时](experiments/2026-09-30-offline-action-cache.md)。
