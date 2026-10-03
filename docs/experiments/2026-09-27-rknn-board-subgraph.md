# RK3588 板端真实 NPU 子图验证

**状态**：首个专家 MLP 的 PTQ 与 QAT W8A8 `.rknn` 子图均已在真实 RK3588 NPU 上成功运行。只证明这两个单层子图能由该板端软件栈执行；不是完整 SmolVLA 部署，也不构成最终混合精度分配。

## 目的和执行路径

验证 Toolkit2 2.3.2 导出的真实 RKNN 模型能否在板上加载、执行，并检查数值误差与单图调用延迟。使用已有 [`rknn_board_subgraph_smoke.py`](../../scripts/rknn_board_subgraph_smoke.py)，通过 `rknn-toolkit-lite2` 调板端 RKNN runtime。Lite2 是 Python 调用层，计算由 `librknnrt.so` 和 RKNPU 驱动完成；本实验没有测自写 C/C++ 调用器。

板上系统 `/usr/lib/librknnrt.so` 为 1.4.0，无法加载模型 version 6（日志为 `Invalid RKNN model version 6`）。仅设置 `LD_LIBRARY_PATH` 仍被 Lite2 的运行时扩展绕过：扩展固定打开 `/usr/lib/librknnrt.so`。为使用已下载的 2.3.2 runtime 且不替换系统库，在私有 mount namespace 中把该库临时 bind mount 到固定路径。namespace 退出后映射消失；测试后确认系统库仍为 1.4.0，SHA-256 `0ebc1b408f897863a91a1b9ed60f3838a801386c7b1ef7c54d55ead624cd8347`。

测试时使用的命令形式：

```bash
unshare --mount --propagation private bash -c "set -e; \
  mount --bind /root/qvla_board_test/rknnrt-2.3.2/librknnrt.so /usr/lib/librknnrt.so; \
  cd /root/qvla_board_test; \
  PYTHONPATH=/root/qvla_board_test/python_site python3 -u rknn_board_subgraph_smoke.py \
    --model model.rknn --input input.npy --reference reference.npy \
    --output ptq_board.json --warmup 5 --repeats 50"
```

QAT 运行使用相同命令和输入，改为 `--model qat_model.rknn --reference qat_reference.npy --output qat_board.json`。输入的预处理不在计时区间内。

板上没有 `python3-venv`/`ensurepip`，因此没有安装系统包。Lite2、NumPy、psutil、ruamel.yaml 的已准备轮子解压到 `/root/qvla_board_test/python_site`，通过 `PYTHONPATH` 使用。`LD_LIBRARY_PATH` 单独指定 runtime 的失败结果也已留在 board 目录的失败 JSON 中；正确运行的两份结果在本地 [`ptq_board.json`](../../runs/board_validation_v1/ptq_board.json) 和 [`qat_board.json`](../../runs/board_validation_v1/qat_board.json)。

## 可复现环境与输入

- 板卡：R1，AArch64，Linux `6.12.69-lzamp+`，glibc 2.35；物理内存 3.8 GiB、无 swap。板端报告的 `MemAvailable` 约 3.5 GiB。
- RKNN 编译器：Toolkit2 2.3.2，目标 `rk3588`；报告模型版本 6、静态 shape。板端 RKNPU driver 0.9.8。
- 运行时：私有路径中的 `librknnrt.so` 2.3.2，SHA-256 `d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8`；Lite2 2.3.2；Python 3.10.12。
- 测试脚本 SHA-256：`1899359d98d56321bd24c9edb4476f44ef5ca0621bb63bcb2d7a6f29ff847dad`。
- PTQ 子图：4,524,893 B，SHA-256 `9b8a2dd8aedd928512d525848415eee48d267b62551dde9ae25b765669c26839`。
- QAT 子图：4,524,893 B，SHA-256 `2cdfef74ce6d632af581106e14fcee9fa794e4acb17273543d47b216d74ba659`。
- 同一隔离开发输入：`input.npy`，float32 `[1, 50, 720]`，SHA-256 `67b835b4982be3df2e0fbc137a97921b60a9c167b40751a9dabedf903482edcb`。两模型各进行 5 次 warmup、50 次计时，指定 `NPU_CORE_0`。
- PTQ 参考：`reference.npy`，SHA-256 `0b67cd801ddf1f1e7e35d5124bb82a15f9ff6e80b4a41e1ed8e03b81f3fd5da7`。QAT 参考：`qat_reference.npy`，SHA-256 `ee69d1d788d710c0fc3f2bcf50f351331542b62a2d3838cc7e808da5e4a75134`。分别对其匹配的主机参考输出计算误差；因参考文件不同，不能把两项误差直接用作 QAT/PTQ 胜负判据。

## 板端结果

| 子图 | 单文件大小 | 推理 p50 / p95 | 相对对应参考 MAE / RMSE | 最大绝对误差 | 进程最大 RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| PTQ W8A8 | 4,524,893 B | 2.118 / 2.477 ms | 0.008773 / 0.010992 | 0.048870 | 67,740 KiB |
| QAT W8A8 | 4,524,893 B | 2.367 / 2.767 ms | 0.008550 / 0.010743 | 0.045103 | 67,844 KiB |

两种子图都成功通过 `rknn_init` 并输出 `[1, 50, 720]` float32。以上延迟是同一个输入重复调用时 `RKNNLite.inference` 的单层调用延迟，不包含模型加载、VLA 图像处理、CPU/NPU 边界往返或策略其余层；单次批次的 p50 差异不能推断稳定的 QAT/PTQ 性能优劣。进程 RSS 是 Python 测试进程指标，不含 NPU/CMA 总占用；前后 `MemAvailable` 也不是峰值内存。

静态模型触发的 `RKNN_QUERY_INPUT_DYNAMIC_RANGE` 查询警告是 Lite2 的通用动态 shape 检查提示，日志明确允许静态模型忽略；实际图已成功运行。

## 结论边界与下一步

1. 已验证板上 driver 0.9.8 + runtime 2.3.2 能执行这两个由 Toolkit2 2.3.2 生成的 W8A8 专家 MLP 子图。模型版本不匹配是先前失败的直接原因；`LD_LIBRARY_PATH` 没有覆盖 Lite2 的绝对库路径。
2. 单图体积、MAE/RMSE 和延迟不能代替完整部署包大小、完整策略 RAM、端到端 chunk 时延、动作质量或 LIBERO 成功率；当前硬件感知收益函数仍为 NA。
3. 下一步将更多已编译图接入可执行策略，测跨图调度与整体资源；随后比较板端完整动作路径和闭环质量。若继续使用 Lite2 的绝对 `/usr/lib` 加载方式，应保留私有 mount namespace 做隔离测试。系统 runtime 和 LZAMP RKLLM 服务未被修改。
