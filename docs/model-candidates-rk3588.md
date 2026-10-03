# 较大 VLA 的候选调查（2026-09-26，历史选型记录）

当前项目基线已经固定为 `lerobot/smolvla_libero`。本页记录早期讨论的候选和未验证估算，不表示目前待选择模型；更换模型需要另行建立 FP 基线并复核数据、processor 与 RK3588 后端。当前执行路线见[量化技术路线](quantization-technique-plan.md)。

目标是找可公开复现 QAT/PTQ、带 LIBERO checkpoint、量化前在 4 GB RK3588S 不可运行而量化后可完整运行的模型。**目前没有一个候选已满足最后一项**：权重大小只说明存储量，尚缺该板 FP16 与真实量化版的编译、内存和闭环运行实测。选择模型前不下载权重、不替换现有 SmolVLA 基线。

| 候选 | 已公开材料与 checkpoint | 对 4 GB 实验的判断 | A10 23 GB QAT 判断 |
| --- | --- | --- | --- |
| SmolVLA，约 0.45B | 当前已固定模型、processor、数据和初轮 QAT/PTQ | 体积较小，FP16 也可能可运行；适合打通 RKNN 流程，难形成明确的“量化前不能跑”案例 | 已实测短程 QAT 可训练，但当前 400 步质量退化 |
| [X-VLA，0.9B](https://github.com/huggingface/lerobot/blob/main/docs/source/xvla.mdx) | LeRobot 和原作者都有代码、LIBERO checkpoint、训练入口；[LeRobot checkpoint 文件约 3.52 GB](https://huggingface.co/lerobot/xvla-libero/blob/main/model.safetensors) | 大于 SmolVLA，结构相对便于 PyTorch QAT；若权重可统一存成 FP16，理论权重约 1.8 GB，故 FP16 未必超过 4 GB 总内存。需板端实测 | 值得先做导出和单步 QAT 探针；训练显存未测 |
| [InternVLA-A1.5，Qwen3.5-2B 主干](https://github.com/InternRobotics/InternVLA-A-series/blob/master/README.md) | [LIBERO checkpoint 文件约 5.39 GB](https://huggingface.co/InternRobotics/InternVLA-A1.5-Libero/blob/718527d4a148434f938f60a175fec04aba0ea9cb/model.safetensors)，公开训练/评测代码；代码许可 CC BY-NC-SA 4.0 | 最接近“FP16 内存吃紧、W8 可能进 4 GB”的目标，但 checkpoint 是否含全部推理权重、量化后峰值和 Qwen3.5 动态算子能否在 RK3588 完整运行均未验证；不是已证实可部署方案 | 作者评测脚本默认寻找至少 30 GiB 空闲 GPU；当前 A10 23 GB 不能直接照搬，需冻结大模块、梯度检查点或换更大 GPU，QAT 可行性未测 |
| [π₀.₅ / OpenPI](https://github.com/Physical-Intelligence/openpi/blob/main/README.md) | 主流、开放训练与 LIBERO 评测代码；[LeRobot LIBERO checkpoint 约 9.35 GB](https://huggingface.co/lerobot/pi05-libero/tree/main) | FP16 完整模型很可能不适合 4 GB；量化后也未必可在 4 GB 内完整运行，且多模块动态推理的 NPU 转换工作大 | 官方给出的单卡 LoRA 微调内存需求 **>22.5 GB**，满参数 **>70 GB**；现有 A10 23 GB 不适合直接承担完整 QAT |
| [OpenVLA，7B](https://github.com/openvla/openvla) | 经典开放模型，有 LIBERO 专项 checkpoint 和 LoRA/4-bit 训练入口 | 即使理想 W4 权重约 3.5 GB，也几乎没有 4 GB 板上系统、视觉编码、激活和运行时空间；当前 RK3588 对所需 W4 路径也未获证实 | 官方 LoRA 示例至少约 27 GB 显存配置，现有 A10 对 QAT 更吃紧 |

上述 FP16/W8/W4 大小只是按参数量乘位宽的**下界估计**，还未计 scale、未量化层、KV/中间激活、图编译缓冲、Linux/Zephyr 和 CMA。不能据此宣布可运行。现有 LZAMP 镜像还为 Zephyr 与共享内存预留了 [32 MiB](</home/loser/Study/rk3588/LZAMP/docs/architecture.md>)。

## 硬件筛选结论

Rockchip [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2) 支持 RK3588，提供 QAT 例子、混合量化、板端性能分析；[v2.3.2 说明](https://github.com/airockchip/rknn-toolkit2/blob/master/rknn-toolkit2/doc/changelog-2.3.2.txt) 新增自动混合精度和部分 W4A16 功能。但不能把该发布说明理解成任意 VLA 的全部 W4A16 子图都能在 RK3588 跑。Rockchip 的 [RKLLM 支持列表](https://github.com/airockchip/rknn-llm)含部分 Qwen/SmolVLM，却没有完整 π₀.₅、X-VLA 或 InternVLA 策略。社区提交的 [Qwen3.5/RK3588 问题](https://github.com/airockchip/rknn-llm/issues/535)还显示其 W4A16 转换报不支持；这是一份特定版本与模型的报告，需要本项目复核。

**建议顺序**：保留 SmolVLA 作固定低成本方法学基线；若以“更大、容易 QAT”为先，第二模型选 X-VLA；若以“争取 4 GB 内存跨越”为先，优先调查 InternVLA-A1.5，但先只读预审配置、代码与算子，再决定是否下载大权重，在现有 A10 上做导出/显存探针或升级 GPU。π₀.₅ 是更主流但资源与 RK3588 转换风险显著更高的研究扩展。最终模型由用户决定。
