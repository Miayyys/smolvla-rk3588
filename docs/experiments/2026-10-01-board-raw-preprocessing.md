# RK3588 原始图像、任务文字与状态预处理

## 完成范围和来源

已将图像缩放/归一化、任务分词、状态归一化搬到板端，并接入[三个 RKNN 子图＋CPU的完整网络](2026-10-01-full-board-fp16-replay.md)。板端从 RGB uint8 图像、原始任务字符串、未归一化状态开始，输出最终 `[1,50,7]` 动作，运行时没有主机/GPU推理或预处理参与。

本次输入为已有 LIBERO 开发缓存中的解码图像数组，不是现场相机采集。NPZ 保存两路 RGB CHW `[3,256,256]`、原始8维状态、任务字符串，以及固定初始噪声 `[1,50,32]`。初始噪声沿用原模型实际捕获值，以隔离随机差异；本次没有实现与 Torch RNG 位级相同的板端噪声生成器。NPZ文件读取、tokenizer/权重加载不计入稳态推理计时；每次正式计时重新执行图像、文字和状态预处理。

使用相同 checkpoint 和匹配的 processor：SHA256 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`。完整原始输入推理样本为 episode18 / task0 / frame0，seed2416662958；预处理核对覆盖独立于冻结测试的全部40条开发观测。没有训练、校准或位宽搜索。

## 技术原理与精确配置

实现：[smolvla_board_preprocess.py](../../scripts/smolvla_board_preprocess.py)。配置由锁定的 `policy_preprocessor.json`、模型 `config.json`、`tokenizer.json`、`tokenizer_config.json` 和原状态统计读取，不手填新词表或新的归一化参数。

### 图像

先转 FP32 并除255。目标512×512，保持比例：$r=\max(W/512,H/512)$，$H'=\lfloor H/r\rfloor$，$W'=\lfloor W/r\rfloor$。双线性插值使用 `align_corners=False` 的坐标 $s=(d+0.5)\times\text{old}/\text{new}-0.5$，负坐标限制为0，末端索引限制为最后像素。按原 SmolVLA 规则在**顶部和左侧**补零，不用居中补零；随后 $I'=2I-1$。模型视觉配置为 IDENTITY，不额外套 ImageNet mean/std。

NumPy实现不需要在板端安装Torch或OpenCV。摄像头次序固定为camera1、camera2，RGB uint8 CHW；当前 checkpoint 的 `empty_cameras=0`，第三路缺失不插入额外空相机。

### 文字

如任务末尾没有换行，补 `\n`；使用原完整 tokenizer 的 BPE、特殊 token、预分词规则。长度48，右侧 padding，pad=`<|im_end|>`（ID2），**左侧 truncation**，并使用原 tokenizer 的特殊 token处理。输出 token ID int64 和 attention mask bool，不用手工切字符串代替分词。

最初按默认右侧截断实现时，40条短任务均匹配，但长文本检查出现44个不同token位置。检查 checkpoint 的 `tokenizer_config.json` 后发现 `truncation_side=left`，已改为读取该配置。失败日志保留 `runs/prepare_raw_board_console.log`；后续核对为空串、已有换行、标点、中文和超过48token的长字符串均与原 tokenizer 完全相同。

### 状态

读取 checkpoint normalizer 的原始8维 mean/std，$s'=(s-\mu)/(\sigma+10^{-8})$，再尾部补零到32维。没有因为配置中 feature 标注6维而错误截断实际8维输入；原始 processor 的真实输出已逐元素核对。

### 板端依赖

板端Python3.10，NumPy1.26.4复用原 `/root/qvla_board_test/python_site`。额外使用PyPI官网的 tokenizers0.22.2 ARM64 wheel，3,290,736B，SHA256 `2249487018adec45d6e3554c71d46eb39fa8ea67156c640f7513eb26f318cec7`；通过 `python3 -m zipfile -e` 解包到同一隔离目录。实际导入与执行已验证。直接使用 `Tokenizer.from_file`，没有增加板端 Transformers/PyTorch 环境或在线访问模型仓库。

## 实际核对结果

本机脚本：[prepare_smolvla_raw_board_inputs.py](../../scripts/prepare_smolvla_raw_board_inputs.py)，对照原始GPU processor及 `prepare_images/prepare_state`。板端脚本：[verify_smolvla_board_preprocess.py](../../scripts/verify_smolvla_board_preprocess.py)，使用40条同一原始观测以及之前原模型实际捕获的80路图像输入。

| 项目 | 本机与原 processor | RK3588 与原 processor |
|---|---:|---:|
| token ID / mask | 40/40逐元素相同 | 40/40逐元素相同 |
| 状态最大差异 | 0 | 0 |
| 图像最大绝对差异 | 2.3841858e-7 | 2.3841858e-7 |
| 5条额外文本边界检查 | token/mask完全相同 | token/mask完全相同 |
| 40条板端预处理 p50 / p95 | — | 78.575 / 79.826 ms |

图像检查使用最大绝对差异5e-7容限，状态容限1e-6；token ID和mask必须完全相同。这些是兼容性核对，不是任务质量阈值。图像仍有浮点计算差异，不能声称移植后整个模型逐元素等价。

## 原始输入到最终动作实测

保持原来的三个FP16 RKNN、NPU_CORE_0，CPU BLAS线程1；三个模型共同驻留。一次预热，三次计时，每次都从原始图像、任务文字、未归一化状态重新处理。输出重复数组已全部归档。

| 指标 | 实测 |
|---|---:|
| 模型加载/初始化 | 1.37984s |
| 完整三次计时 | 8176.991 / 8028.974 / 7879.550ms |
| 完整 p50 / p95 | 8028.974 / 8162.189ms |
| 三次循环内预处理 | 142.432 / 144.058 / 144.146ms |
| 相对原FP动作 MAE / RMSE | 0.00201280 / 0.00313272 |
| 最终动作最大绝对差异 | 0.0129501 |
| 当前样本夹爪符号变化 | 0/50 |
| 三次最终动作 | 逐元素相同 |
| 进程 maxRSS | 1,892,432KiB，约1.805GiB |
| 相对旧“预处理输入回放”的最终动作 MAE / 最大差异 | 0.000339599 / 0.00187591 |
| Runtime `E RKNN` | 未出现 |
| LIBERO闭环成功率 | 未测量 |

循环内预处理时长与单独40条核对的运行上下文不同，不能直接相减判断新增开销。三次整体延迟比上一轮略低，不据此宣称性能优化，因为未控制或采集温度/频率、样本数仅3。微小图像差异经FP16模型后产生非零动作差异，实际结果已保留。单样本没有夹爪变化不推翻先前40条视觉诊断的两个夹爪变化。

## 原始数据、hash与复现

本机 `runs/smolvla_raw_board_v1/`（Git忽略）保存 `preprocess_export.json`、原始输入/面板、token/状态参考、`board_preprocess_report.json`、`raw_full_board_report.json/.npz`、原始日志和 `raw_integration_check.json`；`implementation_hashes.json` 固定本次实现脚本及两个板测报告的SHA256。

| 文件 | SHA256 |
|---|---|
| 原始40条面板 raw_panel.npz | `ab8216772c2525d7d24d8af7dc6a7e8e9845b298c8db45e17ee974afe4c82498` |
| token/状态参考 preprocess_reference.npz | `613be8a3288838ee131a974cecc27396e431741c13fb442655abc79e3e39318f` |
| 原始单样本 raw_inputs.npz | `cca351635bc682a7915c708c1f10dcb42f4e9195f882f5002085ba649801a25c` |
| 状态统计 state_stats.npz | `d00c91ca18a56fbb3ac590175a6c4d2a71756aca90a65e5ec44c560b330a2489` |
| 配置及核对 manifest | `c283cbd5274e5485d71860376ffd8f11f2496e688bdd1606999cbe48689b740a` |
| 最终动作及中间数组 raw_full_board_report.npz | `ce63211a025c1ad62ff411acebf23857dfe408c900d85ac6e9a6420023875de6` |

四个模型/processor/tokenizer配置的hash见 `preprocess_export.json`，运行前验证；神经子图与CPU权重hash见原完整回放记录和新报告。板端目录 `/root/qvla_board_test/smolvla_vision_v1/`，执行：

```bash
cd /root/qvla_board_test/smolvla_vision_v1
env OPENBLAS_NUM_THREADS=1 PYTHONPATH=/root/qvla_board_test/python_site \
  python3 rknn_board_full_replay.py \
  --vision vision_connector_fp16.rknn --prefix prefix_with_kv_fp16.rknn \
  --expert expert_step_v2_fp16.rknn \
  --inputs raw_inputs.npz --weights cpu_weights.npz \
  --reference fp_reference.npz --config replay.json \
  --preprocessor-assets . --preprocessor-manifest preprocess_export.json \
  --output raw_full_board_report.json --warmup 1 --repeats 3
```

板端推理API接受原始数据；NPZ和FP参考是这轮可复核回放输入/评测材料。本轮尚未接真实相机、机械臂或LIBERO闭环；后续[四任务真实板端闭环](2026-10-01-board-libero-closed-loop.md)已完成，FP与板端均2/4，仿真等待推理，不是实时控制验证。当前仍是FP16部署参照，未完成最终HAQ搜索和QAT/PTQ量化，不新增固定精度限制。
