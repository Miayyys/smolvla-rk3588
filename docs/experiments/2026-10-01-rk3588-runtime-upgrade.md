# RK3588 RKNN / RKLLM 版本更新与真实视觉子图复测

## 问题、范围和判据

将板端默认 RKNN runtime 与本机 Toolkit2 对齐，并安装官方最新 RKLLM runtime。判据是板端能通过默认动态库路径加载真实 SmolVLA 视觉子图并执行；数值质量另行与固定 FP 输出比较。该步骤不是完整 VLA 部署，也不是量化质量验证。

## 版本与来源

- 日期：2026-10-01；板子：RK3588，Linux `6.12.69-lzamp+`，RKNPU driver `0.9.8`，Python `3.10.12`。
- 本机 RKNN-Toolkit2、板端 RKNN-Toolkit-Lite2、替换后的 `/usr/lib/librknnrt.so` 均为 `2.3.2`。[官方 RKNN 发布页](https://github.com/airockchip/rknn-toolkit2/releases)。旧默认库实际报告 `1.4.0`；按用户要求直接替换，不保留旧文件。新库来自板端此前独立验证的 `/root/qvla_board_test/rknnrt-2.3.2/librknnrt.so`，SHA256 `d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8`。
- 板端新装 `/usr/local/lib/librkllmrt.so`，来自 [Rockchip 官方 v1.3.1 提交](https://github.com/airockchip/rknn-llm/commit/f739053)，SHA256 `f25e9b099db08aaacd0a3ac62b4697d3951d6ae61ae41ea09f6702cfa89eb32c`。`ldd` 无缺失依赖，`ctypes.CDLL("librkllmrt.so")` 成功；**没有 RKLLM 模型推理测试**。
- 同一官方提交的 Python 3.12 `rkllm_toolkit-1.3.1` wheel 已下载并装入 `artifacts/rkllm-1.3.1-venv/`（Git 忽略），wheel SHA256 `ba4191cd50ec6a555367df3d6a88f5827e918210d2f209a84774301427fa0add`。目前仅 `--no-deps` 安装，导入因缺少 `torch` 失败，**转换环境尚不可用**。官方 metadata 要求 `torch==2.6.0`、`transformers==5.8.0` 等依赖；不得将 wheel 安装等同于完成模型转换。

## 固定模型和操作

- 视觉子图来自 `lerobot/smolvla_libero` checkpoint SHA256 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`；固定输入为一个 `[1,3,512,512]` FP32 图像张量，输出 `[1,64,960]`。此处无 episode 抽样或校准集；只对一个固定输入做板端数值复测。
- RKNN FP16 子图文件 `runs/smolvla_vision_split_v1/vision_connector_fp16.rknn`，212,621,173 字节，SHA256 `f486c5de7f0bc085e9bb1157db761003b9d81d8c2e1e83cd11fb53942b209434`。原始输入和 FP 参考输出 SHA256 分别为 `f0ab4cdcde55d38946d9cc8654a05eecfaa1bade5b58212e91b43ed8312f7648`、`fc17c24d71bc5d0b2b7db2dba031a334f6cc8eecd0e28184fe2f20794827fb5e`。
- 板端运行：`PYTHONPATH=/root/qvla_board_test/python_site python3 /root/qvla_board_test/rknn_board_subgraph_smoke.py --model vision_connector_fp16.rknn --input vision_input_0.npy --reference connector_output_0.npy --output board_report_default_232.json --warmup 1 --repeats 2`。使用 NPU_CORE_0；延迟只包围 `RKNNLite.inference`，不含模型加载/预处理。原始报告在 `runs/smolvla_vision_split_v1/board_report_default_232.json`。

## 实测结果

| 项目 | 结果 |
| --- | ---: |
| RKNN runtime / driver | 2.3.2 / 0.9.8 |
| 默认路径加载和推理 | 成功；无私有 mount 映射 |
| 推理耗时（2 次） | 1922.639、1927.328 ms；p50 1924.983 ms |
| 进程最大 RSS | 512,900 KiB |
| `MemAvailable` 前 / 后 | 3,675,248 / 3,192,052 KiB |
| 与 FP 输出的 MAE / RMSE / 最大绝对误差 | 5.945478 / 7.981631 / 67.726448 |
| 完整 VLA 任务成功率、整策略延迟与峰值内存 | 未测量 |

误差定义为 `mean(abs(y_rknn-y_fp))`、`sqrt(mean((y_rknn-y_fp)^2))`、`max(abs(y_rknn-y_fp))`，在完整 `[1,64,960]` 输出上计算。这个误差远超可接受数值一致性，说明**仅升级 runtime 没有解决视觉子图的数值问题**；原因仍未定位，可能涉及输入布局、RKNN 变换或导出图。不得把该子图用于完整动作质量或 HAQ 硬件收益结论。

## 下一步

核对 RKNN 输入输出属性、布局和预处理，保存板端中间量并与 ONNX / FP 对照；数值通过后再扩展语言前缀和动作专家子图。RKLLM 公开接口能否暴露 SmolVLA 所需的逐层 K/V 尚须验证；当前仅证实 runtime 可加载。
