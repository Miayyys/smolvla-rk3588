# 原FP与蒸馏FP的独立v2 PTQ对照

用户授权试PTQ。分别独立从原始FP和前8层KD蒸馏FP转换，不使用已经量化的QAT文件继续PTQ，不更新任何模型权重。不依赖旧HAQ候选文件的历史量化结果代替本轮转换。

位宽及激活校准参数固定为原v2图，与刚完成的全图QAT相同。校准仍来自隔离40校准episodes，未拿19任务测试输入重新校准。蒸馏学生采用全FP master精确参数，替换模块时避免将FP32更新提前截断成BF16；算子外部dispatch dtype保留原值，和QAT产物一致。未配置参数核对与原始权重相同。此轮固定范围便于比较，不声称它是重新校准学生后的最佳PTQ。

入口 `scripts/pack_v2_ptq_control.py`：冻结源/位宽/校准hash检查，直接整数打包，严格重载全部state，refresh运行缓存，在同40开发观察/同噪声下完整50动作chunk对照保存前后parity，要求max_abs=0；再执行19任务闭环。最终可见文件是实际整数权重safetensors，并非仅fake quant。INT16/Conv仍本地数值参考，非RKNN整图内核。

后台 `scripts/run_v2_ptq_controls.py`，顺序蒸馏学生PTQ→19测评→原FP PTQ→19测评，共38回合。状态/命令/日志 `runs/v2_ptq_controls_v1`，每组ptq_local_packed.safetensors、report.json、reload_actions.npz、逐任务动作和reset审计。比较原FP/蒸馏FP均16/19、QAT14/19与本轮两种PTQ，报告哪些任务新增/丢失，不只看总数。

全部固定Long10与其他suite9任务，核对初始state、双相机、env seed和策略noise seed。面板多次用于诊断，非新的独立冻结测试集；不能据此称为正式成功率、端侧速度/内存或硬件收益已验证。

语法检查通过，后台结果待实测。未新增训练，RKNN完整策略部署仍待验证。

## 完成的闭环结果

两组独立打包及38回合全部完成，用时1959.99秒（32.7分钟）；固定初始状态/双相机/环境和噪声seed配对通过。两产物在40开发观察保存前后完整动作parity均MAE/max_abs=0。

| 模型 | Long10 | other9 | 合计19 |
|---|---:|---:|---:|
| 原始FP | 7/10 | 9/9 | 16/19 |
| 蒸馏FP学生 | 7/10 | 9/9 | 16/19 |
| 原始FP+v2 PTQ | 7/10 | 7/9 | 14/19 |
| 蒸馏FP学生+v2 PTQ | 5/10 | 7/9 | 12/19 |
| 蒸馏FP学生+v2 QAT | 5/10 | 9/9 | 14/19 |

原FP PTQ：Long10新增成功7、丢失4；other9丢失Spatial4、Goal8。学生PTQ：Long10丢失1、6，无新增；other9丢失Spatial4、Goal4。学生QAT相对学生PTQ救回Long1/6，但又失去3/4，总Long仍5；other9全部救回至9。不能把同Long总分解释为同一失败机制。

结论：蒸馏FP与原FP在该面板同16/19，但量化后的学生PTQ更差12/19，表明当前学生参数与固定v2图/校准组合可能更脆弱；未证明是蒸馏本身的一般性问题。QAT将学生量化结果从12提高到14/19，显示当前对照中有任务质量恢复，仍低于FP16/19。原FP PTQ与学生QAT同14/19但任务不同，不能认定后者全面优于原FP PTQ。

## 存储类型核查

初次PTQ文件610344248B，比QAT514073656B大96270592B。检查确认未配置的lm_head、视觉位置嵌入及norm从源BF16被加载为FP32后按FP32存储；QAT保存路径则保留源BF16。这个差异不是位宽图或真正压缩效果不同。修复pack脚本，在每个恢复源BF16的tensor上验证FP32解码数值完全相同，拒绝有损恢复。

`scripts/normalize_ptq_fp_storage.py` 对已完成两产物作存储规范化，并分别严格重载、重跑同40开发观察完整动作，与修复前保存的真实输出要求max_abs=0；结果以storage_normalization.log及每组report/summary的storage_normalization字段为准。只有数值保持与真实动作检查通过后才替换文件并更新hash/体积；旧闭环轨迹保留，因文件改动只无损保存等值BF16，不重复38回合。未将初次32.7%缩小误报为满足40%门槛。

规范化两组均完成：每组最终514073656B，缩小43.3036%，与QAT存储协议一致；全部tensor值严格保持，修复后真实40观察动作对修复前MAE/max_abs均0。已更新报告hash/summary，并同步本地summary/status及规范化日志。任务质量结论不变。
