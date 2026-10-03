# 真实 OpenVLA-OFT 教师、FP 蒸馏与 v2 QAT 诊断

## 完成范围

独立教师环境已经安装、真实教师推理与动作契约审计通过。两组学习率各完成 FP 蒸馏40步 → v2 QAT40步，并严格重载真实本地打包产物。**这是短训练诊断，不证明蒸馏提升、训练收敛或 RKNN 混合整图完成。** 效果评价与首步判断的修正见[完整动作与闭环协议](2026-10-01-distill-qat-quality-protocol.md)。此前[环境准备](2026-10-01-real-teacher-environment-preparation.md)的缺包状态是历史记录。

## 教师环境及版本

服务器 A10，学生 `/root/qvla/.venv`，教师 `/root/qvla/.venv-teacher`；教师使用 system-site-packages 复用 CUDA torch，不改变学生 Transformers5.5.4。教师安装官方 Transformers4.40.1 fork（`bc339d9ad707454c0c115970db43c260067c61ab`），OpenVLA-OFT 源码 `e4287e94541f459edc4feabc4e181f537cd569a8`；torch2.7.0a0 NVIDIA25.03、TF CPU2.17.1、timm0.9.10、peft0.11.1、sentencepiece0.2.0。32个缺包共281,733,292 B已离线安装。配置、锁文件和实际freeze分别是 `config/teacher_inference_requirements.txt`、`config/teacher_inference_lock.txt`、`runs/distill_real_teacher_evidence_v1/teacher_environment_freeze.txt`。

第一次加载因为推理入口间接导入训练用dlimp而失败。两处非数值补丁：NormalizationType改为直接导入定义它的constants（同一个Enum）；`prismatic/vla/__init__.py` 延迟导入训练dataset factory，调用训练API时仍执行原函数。没有修改模型计算。补丁及原始Git diff保存在实际教师manifest；工具 `scripts/patch_openvla_inference_import.py`。官方checkpoint根目录哈希未改变，推理需要同步的小文件在私有工作目录处理。

## 输入与动作契约

教师 checkpoint `moojink/openvla-7b-oft-finetuned-libero-10`，修订 `95220f9a3421a7ff12d4218e73d09ade830fa9a3`，仅声明LIBERO-10。调用官方 L1 action head/proprio projector150000、BF16、双图像、8维状态、8步×7维动作、统计key `libero_10_no_noops`。该环境版本组合已真实推理，未验证论文任务分数。

LeRobot/RLDS数据已采用训练图像方向，缓存不再额外旋转180度，仍使用官方resize/crop；模拟器实时图像按标准processor旋转一次。task描述匹配官方任务，不能将数据task_index直接当官方ID。动作先教师反归一化，再夹爪 `-sign(2g-1)` 转为模拟器坐标，最后进入学生归一化。数据10fps表示记录动作步序，不等同于仿真20Hz物理时间保证。

实际导出10任务各一个训练观察，共80个教师动作。逐帧图像、状态、任务、episode/frame与学生输入一致，训练episode不与校准/开发/冻结测试交叉。连续6维对记录动作MAE **0.0058702752**；夹爪符号一致 **79/80**。数值有限且非恒定。教师自身闭环成功率未测，不能据此称40/40强教师。

教师总耗时98.8314s（含模型加载/hash），首样本75.5849s，后续约0.22s/8步。原始数据：服务器 `runs/teacher_actions_real_smoke_v2/{manifest.json,actions.npy,contract_audit.json}`，输入 `runs/teacher_inputs_real_smoke_v2`；本地证据 `runs/distill_real_teacher_evidence_v1/`。实际manifest未测量教师CUDA峰值，不能补造。

## 损失和训练

冻结v2的304位点不再搜索：292 W8A8、3 FP16、3 BF16、5非对称INT16、1 CPU INT8 embedding。FP蒸馏关闭fake quant；QAT开启原v2精度图，使用FP32 master及STE。原始SmolVLA权重大多BF16，计算按原dtype执行；FP32训练master不作为部署体积。

对教师覆盖的有效前8步：`L = L_GT + 0.2 L_teacher`。两项flow-matching共享同一随机噪声ε和时间t；监督目标为 `v = ε - action`。教师suffix mask关闭，不能把8步重复到50步。GT监督完整有效动作块。每阶段40步遍历40任务，其中10步使用真实教师损失，其余30步仅GT。

AdamW，weight_decay0、grad clip1、seed29，两组lr分别1e-5和1e-7；同数据帧、同校准和v2图。训练池1147 episode，隔离40开发/40校准及冻结测试。全五阶段梯度被测到。每步的数据和噪声不同，loss曲线只能诊断，不是同一质量函数的收敛曲线。

| 学习率 | FP40步耗时 | QAT40步含打包耗时 | QAT CUDA峰值 | 严格重载完整动作MAE/max |
|---|---:|---:|---:|---:|
| 1e-5 |46.8153s|97.2169s|14,257,236,992 B|0 / 0|
| 1e-7 |43.5063s|97.1376s|14,250,590,720 B|0 / 0|

两组真实本地pack均 **514,073,656 B**，原始文件906,712,520 B，`1 - B_q/B_FP = 43.303567%`。不是部署RAM节省或RKNN体积。W8 Linear CUDA整数矩阵前向；Conv/INT16是本地解码数值参考，不能当作RKNN完全等价。

## 身份与复现

- 原模型SHA256：`9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`
- split：`ca851a1bdc8fd60ad1e5b8d08dc7c405f971a8d4f999c4f0c2ecef154272d55f`
- partition：`f755546a6b074d2fe248333fc42c3dbf30f9b54b9f0fa0b58a16506a54d942c2`
- v2配置：`5991c7b17ac5ab1235318890a124f7127eade5850322e84c9451384d08a72ac4`
- 教师manifest：`d48508a7a12f39280a06831ec1d211e3a5b62a92d0c82b8e20c101f8e17fbb34`
- 1e-5 FP master：`0d689ef9dfc9e898f9278760b7525cc70b8a6850c47cbbce6b5be1ccb1e8e826`；QATpack：`4dd3138a09f2dcbcdab562b8e832bb9c57af3a513503838732f0171127da7bdb`
- 1e-7 FP master：`0013cbbe1156d30f49766c9a95b23f24cd8e5df75c314033a703aa087fb6fec1`；QATpack：`7ad50159a08683aac3c4c26ca74716f181d237155e905f214ce814e828ac10ac`

运行入口：`scripts/setup_teacher_server.sh`、`prepare_openvla_teacher_inputs.py`、`cache_openvla_teacher.py`、`audit_openvla_teacher_labels.py`、`run_real_distill_qat_diagnostic.sh`。后者默认tag real40_v1/lr1e-5；设置 `QVLA_DIAGNOSTIC_TAG=real40_lr1e7_v1 QVLA_DIAGNOSTIC_LR=1e-7` 执行低学习率对照。保留原始FP、原始v2和两组训练权重，不凭首步代理晋升最终模型。
