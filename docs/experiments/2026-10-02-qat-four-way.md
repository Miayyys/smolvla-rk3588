# 蒸馏学生四组QAT比较：v1/v2 × QAT教师损失开关

用户授权仅新增v1/v2关闭QAT教师损失两组，与已完成两组共同四选一。模型起点仍是此前真实教师蒸馏、只更新前8层的FP学生；关闭的是QAT中的额外教师目标，而非删除原有蒸馏成果，更不是关闭量化前向或梯度。

## 两个新运行

配置 `config/qat_distilled_v1_no_teacher_v1.json` 和 `config/qat_distilled_v2_no_teacher_v1.json`。统一初始master SHA256 `4e93e82cfc662471810ce46e4ccef968c306c54ba1e759c6e50c4f4142cec88a`，1000次更新、累积2、seed29，50warmup步1e-8→1e-7再余弦回1e-8，AdamW及clip1，全部304节点权重/bias可训练，未配置参数冻结。v1/v2原有量化图与校准范围固定，不重新HAQ搜索。

唯一损失变化 λ0.2→0，L=L_GT。GT示范动作经过完整量化forward及STE反向更新，最终实际整数打包、严格重载、完整动作parity后运行Long10+other9共19任务。沿用同一教师缓存选训练帧与教师任务采样比例0.5，以免改动采样分布；不执行教师目标forward，不施加教师梯度，实际teacher_supervised_steps应为0。缓存选帧不等于使用教师损失。

每250步开发动作评价/断点保存，保留2快照。启动时按此前教师组约15GiB峰值预算，23GiB A10顺序运行，避免同时占用超显存。入口 `scripts/run_qat_no_teacher_comparison.py`，总目录 `runs/qat_four_way_v1`。逐模型训练日志 `runs/qat_distilled_v1_no_teacher_v1_pipeline/qat.log` 与v2对应目录；包含每组真实pack及固定19回合。不额外PTQ、蒸馏或其他变体。

## 四组选取协议

先满足本地文件相对原FP减少≥40%；在可行组中优先19任务成功数，其次Long10成功数，再比较文件大小；若这些指标仍一致则报告并列，不编造唯一胜者。训练耗时不当推理速度，板端收益尚未测。各组均核对相同学生hash、训练帧采样hash、源图hash、冻结完整性与严格pack重载，并逐项比较任务新增/丢失。既有v1教师组12/19，v2教师组14/19，学生FP参照16/19。

这是同训练seed、单初始状态及重复使用诊断面板的四组比较，不证明总体优越性或蒸馏改善。后续正式评估须另设冻结面板；RKNN混合整策略尚未验证。只有实际结果支持时才能描述质量收益。

## 验证和资源

本地三项蒸馏契约测试通过，λ0确保仅GT forward及梯度保留；新增流水线语法检查通过。依照用户此前清理无用旧产物授权，删除已完成v1/v2 QAT的滚动AdamW状态和周期重复快照，约15.91GiB；最终master、实际pack、完整报告/日志/动作均保留。具体文件清单 `runs/completed_qat_duplicates_cleanup_20261002.json`。旧完成运行无法从已删AdamW状态续训，最终权重仍能复测/另开实验。新两组实际结果见下节。

## 完成结果

原始数据：`runs/qat_four_way_comparison.json`（服务器原路径 `runs/qat_four_way_v1/comparison.json`）。两组新增流水线共6618.09秒（110.3分钟）；四组训练帧采样SHA均为 `d4038fa35a83c0821ad8a5283654fd51e3f28a9c878f53d13bc546c6fc59bc85`，同一学生起点。

| 组别 | Long10 | other9 | 合计 | 实际字节 | 相对原FP减少 | 训练及打包秒数 |
|---|---:|---:|---:|---:|---:|---:|
| v1_with_teacher | 4/10 | 8/9 | 12/19 | 504284536 | 44.38% | 2973.53 |
| v1_without_teacher | 6/10 | 8/9 | 14/19 | 504284536 | 44.38% | 2298.62 |
| v2_with_teacher | 5/10 | 9/9 | 14/19 | 514073656 | 43.30% | 2984.53 |
| v2_without_teacher | 4/10 | 6/9 | 10/19 | 514073656 | 43.30% | 2394.50 |

原FP与蒸馏FP学生参照均16/19（Long7、other9）。四组均满足至少40%文件压缩，严格重载相对对应训练量化前向的动作MAE/max均0；不表示相对原FP无量化误差。

按预定协议，暂选 `v1_without_teacher`：总成绩与v2教师组同为14/19，但Long6/10高于5/10，并且文件小约9.79MB。两组任务成功集合不同，不能声称逐任务全面更好。v1无教师组相对FP丢失Long4/6、Goal4，新增Long8成功；仍净少2个任务。QAT不加教师损失仍保留此前蒸馏学生起点，不能表述为蒸馏被取消，也没有证明蒸馏提高总体成绩。

教师损失效果依赖图：v1关闭后12→14，v2关闭后14→10；单seed对照不支持“教师损失总是有害/有益”。后续保留四组，优先候选供进一步独立评测及RKNN转换验证，本次不启动额外训练。推理速度、板端全策略峰值内存及混合图收益未测量。

### 两组新增权重校验

- `v1_without_teacher`：`/root/qvla/runs/qat_distilled_v1_no_teacher_v1/qat_local_packed.safetensors`，SHA256 `0b1fd3ece496b777fb145e33ac03e51e14c7205a55fb246250707b6a77c29474`。
- `v2_without_teacher`：`/root/qvla/runs/qat_distilled_v2_no_teacher_v1/qat_local_packed.safetensors`，SHA256 `f5177111ce19db27bf2fcd0e4cbcc18fe1d0abdd7ff499aac9bcdbfa6e161718`。
