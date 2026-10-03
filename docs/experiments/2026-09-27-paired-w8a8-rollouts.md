# 真实 W8A8 QAT/PTQ 的 LIBERO 配对闭环筛查

**状态：开发筛查已完成，质量门槛未判定。** 对 FP、真实 GPU INT8 PTQ、真实 GPU INT8 QAT 分别运行四个 LIBERO suite 的全部 40 个任务，每任务 1 回合。结果可以发现候选回归，但不足以证明非劣或正式成功率。

## 问题与方法

验证 112 个动作专家 Linear 的真 W8A8 产物是否在闭环任务中出现明显成功率回退。PTQ/QAT 两个 checkpoint 来自[整专家 W8A8 实验](2026-09-27-full-expert-real-w8a8.md)，分别独立从原始 FP 和 QAT 第 50 步浮点主权重生成；其余视觉语言模块保持 BF16。三种模式使用同一 LeRobot LIBERO evaluator、相同 suite/task、相同初始状态索引、256×256 双相机输入、relative control 和同一 `--seed 0`。

评测 wrapper 按任务重置 Python、NumPy、PyTorch 与 CUDA 随机数生成器。任务随机种子为

\[
seed_{task}=0+100000\times(suite\_index+1)+task\_id,
\]

其中 `suite_index` 按 `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` 取 0–3。这样同一任务 FP/PTQ/QAT 的策略采样噪声种子一致；逐任务成功标志由 `eval_info.json` 逐项对齐检查。它是配对单回合筛查，不是多初始状态统计。

## 固定来源与复现

- 模型：`lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de`，原始权重 SHA-256 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`；processor SHA-256 `122ec5106602b1bf129f49690d05ab2f49748a0ac6119de55ea0677d4e90d248`。
- VLM 资源：`HuggingFaceTB/SmolVLM2-500M-Video-Instruct@7b375e1b73b11138ff12fe22c8f2822d8fe03467`。LIBERO 数据与资产修订号、软件环境见[锁定环境](../../runs/fp_libero_40x1_new/environment.json)：LeRobot 0.6.1、PyTorch `2.7.0a0+7c8ec84dab.nv25.03`、CUDA 12.8、A10；`lerobot/libero` revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`，libero assets revision `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`。
- 评测脚本 SHA-256：`eval_real_w8a8_libero.py` `e4bcd3123e2fbbd8c2a566a32829520860d58dd6d4cd2a0b3c5b9c253b1e3a96`；`run_real_w8a8_libero_suites.py` `f4525a5dba3107c56294036c3ca28b84e413602b8d5939ba118ba0565c36d2c6`；`run_paired_real_w8a8_rollouts.py` `d102201b5e10f72391136c11db32608c889b3743f333a81c91d2364e85ac157e`；汇总器 `summarize_real_w8a8_rollouts.py` `34d15082b2bf9d4211cc6b1db79ea1c8bf6141457d4c2dee65e92b90d71c0671`。
- 输出根目录 `runs/paired_noise_seed0_v2/`。配对报告 SHA-256 `66633d4a93b2b5af339e9a0aa3490ab94b5dd1a1ee08653f59fc64233`；绘图 SHA-256 `d437e60efb95e412803ca459a741ae265a99ba1a7feaaf5d4fa99a53c2bc9330`。各模式日志为 `fp.log`, `ptq.log`, `qat.log`；汇总器确认 12 份 suite `eval_info.json` 均各含 10 个唯一任务并且任务 ID 对齐。总控 `progress.json` 未写入最后的 QAT 子进程完成记录，因此完成状态以三份成功日志、12 个结果文件和可复核汇总为准。

服务器 `/root/qvla` 复现命令：

```bash
.venv/bin/python scripts/run_paired_real_w8a8_rollouts.py \
  --model-dir artifacts/model \
  --vlm-assets-dir artifacts/smolvlm2_assets \
  --pack-report runs/expert_real_w8a8_v1/report.json \
  --output-root runs/paired_noise_seed0_v2 \
  --seed 0 --workers 3

.venv/bin/python scripts/summarize_real_w8a8_rollouts.py \
  --runs-root runs/paired_noise_seed0_v2 \
  --output runs/paired_noise_seed0_v2/paired_report.json \
  --plot figures/paired_noise_seed0_v2.png
```

## 觀測結果

| LIBERO suite | FP | PTQ W8A8 | QAT W8A8 |
| --- | ---: | ---: | ---: |
| Spatial | 9/10 | 9/10 | 8/10 |
| Object | 9/10 | 9/10 | 9/10 |
| Goal | 7/10 | 7/10 | 7/10 |
| LIBERO-10 | 7/10 | 6/10 | 7/10 |
| **合計** | **32/40** | **31/40** | **31/40** |

PTQ 相對這次 FP 重跑少 1 個成功任務（−2.5 個百分點）：39/40 任務相同，`libero_10/6` 從 FP 成功變成 PTQ 失敗。QAT 也少 1 個（−2.5 個百分點）：39/40 相同，`libero_spatial/6` 從 FP 成功變成 QAT 失敗。QAT 和 PTQ 總數相同，但失敗任務不同。按 suite 分層、每 suite 對 10 個任務有放回抽樣 20,000 次的配對差值 95% 描述性區間，兩者均為 **[−7.5, 0] 個百分點**；固定 bootstrap RNG seed 0。每任務只有一個初始狀態，區間不估計 episode/種子變異，不是非劣性檢驗或預先設定的質量門檻。

![配對 LIBERO 四套件成功任務數](../../figures/paired_noise_seed0_v2.png)

匯總報告保存所有 40 個逐任務 Boolean 結果、套件時間和 bootstrap 差值，見[`paired_report.json`](../../runs/paired_noise_seed0_v2/paired_report.json)；完整圖像為[`paired_noise_seed0_v2.png`](../../figures/paired_noise_seed0_v2.png)。套件執行時間是在三個模式併行時記錄，不作模型速度比較。這次 GPU 閉環結果也不代表 RK3588 NPU 執行。

## 解讀與邊界

早前獨立 FP 開發基線為 **31/40**，LIBERO-10 是 6/10；本次採逐任務策略採樣種子後，配對重跑 FP 為 **32/40**、LIBERO-10 為 7/10。這一回合差異說明一次回合對隨機策略噪聲敏感；不將兩次 FP 分數合併，也不以新 FP 重跑代替正式多種子基線。此次 PTQ/QAT 的差值僅相對同批次 FP 重跑計算。

結果沒有顯示總成功數改善，也沒有足夠樣本判定量化品質是否可接受。按照質量優先規則，當前統一專家 W8A8 是已完成真量化和配對閉環推理的**開發候選**，還不是通過質量門檻的最終配置。下一步需對同一套件和任務增加多個初始狀態/策略種子，與新的同環境 FP 基線配對；再依逐任務質量、資源及板端完整執行狀態決定保留 W8A8 或做更細的混合精度。完整模型 RKNN、端到端板端成功率、峰值 RAM、完整策略延遲與能耗仍未測量。
