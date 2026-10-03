# 真实教师环境准备：独立依赖、图像契约与网络阻塞

## 当前结论

服务器独立 `.venv-teacher` 已创建并安装官方 bidirectional Transformers fork；源码与 checkpoint 已在服务器。**真实教师加载/推理尚未执行：缺少32个依赖wheel，官网连接速度太慢，等待用户本地下载上传。** 不能将环境准备或此前合成标签两步训练当成真实蒸馏验证。

## 已完成及精确配置

服务器2026-10-01重新检查：Python3.12.3、NVIDIA A10；探测时空闲显存22717MiB，磁盘40GiB可用。教师venv使用 `--system-site-packages` 复用系统 CUDA torch2.7.0a0，不修改学生 `.venv`；教师包安装在自己的venv中。

- 官方教师源码 Git revision：`e4287e94541f459edc4feabc4e181f537cd569a8`。
- 官方 Transformers fork revision：`bc339d9ad707454c0c115970db43c260067c61ab`，editable版本4.40.1，安装成功；尚未完成真实模型兼容性验证。
- 教师 checkpoint 仍是已核验完整文件的 LIBERO-10 checkpoint。
- `config/teacher_inference_requirements.txt` 声明直接依赖；`teacher_inference_lock.txt` 固定完整推理依赖；`teacher_missing_wheels.txt` 仅列服务器缺少/需覆盖版本的包。
- `scripts/setup_teacher_server.sh` 提供离线安装、源码适配和真正import预检查。预检查包含TF、官方helper及动作常量的实际import，不仅检查package名字。成功仍不等于模型推理成功。

官方配置锁定Python3.10/TF2.15/较旧torch，而服务器只有Python3.12。当前准备复用torch2.7、TF CPU2.17.1、sentencepiece0.2.0，其余主要模型依赖按官方版本；这些版本偏差尚待真实推理验证。参考：[官方环境](https://github.com/moojink/openvla-oft/blob/main/SETUP.md)、[依赖声明](https://github.com/moojink/openvla-oft/blob/main/pyproject.toml)。

### 移除无关训练依赖的等价导入适配

官方 `openvla_utils.py` 从 RLDS `data_utils` 导入 `NormalizationType`，该模块本身又从 `prismatic.vla.constants` 导入同一个Enum。前者会顺带导入整个数据训练管线，拉入dlimp、TFDS、TFGraphics和没有cp312 wheel的TF Addons；依赖解析实测失败。

`patch_openvla_inference_import.py` 验证Enum原始来源后，仅将这一行改为直接从constants导入。没有改模型前向、图像处理、归一化算法或动作计算。补丁、原始revision和patch SHA全部记录，并随每次真实标签缓存保存。

实际补丁后的helper SHA256：`5cfbeec4a92ac0e6204793d8cf7b9398079ae27c2cd236e3e4272a5b29ab7ccf`。完整diff：服务器 `runs/teacher_inference_import_patch.json`，本地同名证据。dlimp在初次排查时曾安装，离线setup脚本会移除这个未用的教师venv包。

### 修复重复旋转图像

首次代码把数据集图像当实时模拟器原图，再旋转180°，该假设不正确。实际导出的图像肉眼对比，结合 [LeRobot官方图像契约](https://github.com/huggingface/lerobot/blob/main/docs/source/env_processor.mdx)，确认记录数据已经是RLDS训练朝向；官方OFT evaluator的180°旋转用于实时模拟器输入。

修复：数据集缓存不再额外旋转；仍使用官方JPEG/resize/crop流程。输入manifest明确 `LIBERO_RLDS_training_orientation; no_extra_rotation_then_official_resize_and_crop`，worker拒绝旧契约，避免旧输入误用。此前没有运行过真实教师，因此没有教师推理结果需撤回；旧输入v1仅作历史证据。新的服务器输入 `runs/teacher_inputs_real_smoke_v2`，10条观察覆盖LIBERO-10全部10任务，来自独立训练episode，仍不构成真实教师动作验证。

首次服务器导出还触发LIBERO首次配置交互提示并以EOF失败，已改为选择现有config.yaml；无配置则明确报错，避免后台任务卡在交互输入。

## 下载测量与下一步

官网TensorFlow wheel范围请求，20秒只收到537687 B，平均26883 B/s；随后pip包日志约10–23KB/s。已停止慢下载，没有把等待当安装完成。

通过官方PyPI元数据验证32个缺失依赖都有Linux CPython3.12可用二进制wheel，总大小 **281733292 B（281.73 MB）**；其中libclang使用manylinux2010标签，下载脚本已纳入。元数据与SHA：`runs/distill_chain_server_evidence_v1/teacher_download_metadata.json`。并未下载这些完整wheel。

用户在本地fish中运行：

```fish
cd /home/loser/Study/QVLA
python3 scripts/download_teacher_dependencies.py --skip-upload
python3 scripts/download_teacher_dependencies.py --upload-only
```

脚本仅从官方PyPI下载指定缺包，固定cp312/x86_64，并使用现有SHA上传函数；服务器无rsync时使用scp -O，不重传已经核验完整的文件。scp模式未完整的单个文件需要重传，不保证文件内部断点续传。

包齐后由项目继续：离线setup→真实10观察教师动作生成→首样本图像/状态/动作检查→真实标签少量FP蒸馏→v2 QAT接入并严格重载→独立开发集配对质量检查。**训练闭环接通与效果提升是两项不同结论，均须真实证据。** 当前速度、教师质量、蒸馏收益均未测量。

## 后续完成状态

32个依赖已上传并安装，补充延迟导入训练factory后，独立环境真实教师加载/推理及动作契约均通过。两组真实FP→v2 QAT短训练已完成。精确配置、两处导入补丁及实际结果见[真实教师训练记录](2026-10-01-real-teacher-distill-qat.md)；本页前述等待依赖为历史状态。
