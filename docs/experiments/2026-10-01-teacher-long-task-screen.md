# OpenVLA-OFT 教师长任务筛查

## 依据与范围

当前checkpoint专门针对LIBERO-10（LIBERO-Long），不是四suite合并教师。[官方说明](https://github.com/moojink/openvla-oft/blob/main/LIBERO.md)确认任务范围；[论文Table I](https://arxiv.org/html/2502.19645v1#S5.T1)报告双相机/状态输入OpenVLA-OFT长任务94.5%，不是100%。这不能当作本服务器实测分数。

此前仅验证教师10个训练观察的推理和动作契约。六学生模型的小闭环只有长任务ID0/3两项，不足以评价蒸馏对完整长任务suite的增益。下一步先测教师本身全部10任务，以确认成功/失败和教师相对学生的可用提升空间。

## 预先固定的协议

- 10任务ID0..9，各1回合，官方initial state index0；环境seed0，策略seed400000+task_id。
- 复用官方GenerateConfig、initialize_model、run_episode、图像/状态与动作处理；BF16、2图、8维状态、每次执行8步再查询、center crop开启。
- LIBERO-Long最长520个控制步，等待稳定10步，256×256相机；不更改超时或失败任务初始状态以追求10/10。
- 教师隔离环境使用自己的Transformers fork。缺失模拟器模块从学生site-packages作为末位fallback导入，不以学生Transformers替代教师fork。
- 私有checkpoint工作副本保存helper同步配置；原始权重只读，记录root文件SHA、source diff、版本、真实CUDA峰值。
- 每任务记录描述、成功、控制步数、耗时、初始状态SHA及10次等待后原始双相机SHA。与已有学生长任务ID0/3对照前核验这些SHA。教师8步/学生50步查询频率是各模型原始策略设置，不宣称二者action chunk相同。
- 运行异常独立标记并使任务退出，不能计作模型质量失败。视频暂关闭，避免无关耗时。

入口 `scripts/eval_openvla_teacher_long.py`；服务器后台输出 `runs/teacher_long10_init0_v1`，主日志 `runs/teacher_long10_init0_v1.log`。状态：10任务全部完成，无运行异常；8/10成功。。

## 对蒸馏的约束

长任务是当前教师适用的蒸馏重点，但“长任务蒸馏提升更多”是待验证假设。此前每个任务只有一个训练观察、教师仅监督对应前8步，无法覆盖长任务的中后段和阶段切换。后续数据应从隔离训练episode覆盖全程和关键阶段，保留其余suite的GT监督控制退化。先比较教师/学生任务执行，再决定新增数据；不因离线单步误差小就断言长程能力已传递。

## 实测结果

总耗时547.123s（约9.1分钟，含加载，不含启动前checkpoint hash），8/10成功，所有任务无runtime error。当前root checkpoint SHA与此前真实教师manifest逐文件一致。每任务只一个初始状态，因此8/10不是论文多种子成功率复现，也不能说某任务普遍失败。

| 官方任务ID | 任务描述 | 成功 | 实际控制步数 |
|---|---|---|---:|
|0|put both the alphabet soup and the tomato sauce in the basket|是|255|
|1|put both the cream cheese box and the butter in the basket|是|239|
|2|turn on the stove and put the moka pot on it|是|238|
|3|put the black bowl in the bottom drawer of the cabinet and close it|是|211|
|4|put the white mug on the left plate and put the yellow and white mug on the right plate|是|219|
|5|pick up the book and place it in the back compartment of the caddy|是|172|
|6|put the white mug on the plate and put the chocolate pudding to the right of the plate|否|520|
|7|put both the alphabet soup and the cream cheese box in the basket|是|247|
|8|put both moka pots on the stove|否|520|
|9|put the yellow and white mug in the microwave and close it|是|248|

任务ID6（白杯放盘子，再把布丁放盘子右侧）与ID8（两个摩卡壶放炉子）均耗尽520步，未完成。不能把失败当作环境异常：没有官方episode error。

### 与已测学生的边界

- ID0的初始状态与等待10步后的双相机SHA均完全相同。原FP、原v2及低学习率FP/QAT学生失败，教师255步成功。这给出可用于长任务蒸馏的一个具体改善对象。教师原生每8步重新查询、学生原生每50步执行块，不能仅凭该比较把差异归因于模型容量。
- ID3双方成功，初始状态SHA相同但图像SHA不同。**不是严格配对画面**，不据此统计教师对学生总体配对优势。其余8任务本轮没有同条件学生数据。
- 当前证据支持优先补充长任务全程训练数据和成功教师轨迹，再验证学生任务执行；不支持“所有教师动作都更好”或“蒸馏一定提升更多”。原始GT保留，教师失败任务另行诊断。不能把开发rollout直接塞进训练集。

原始JSON/每任务错误日志：本地 `runs/teacher_long10_evidence_v1`，服务器 `runs/teacher_long10_init0_v1`。summary SHA256：`a03839676aa7eff6afa4889f5e4c30d30a693fd6108872731dcdc6edca903572`。配置、根权重hash、source diff、CUDA峰值与每任务原始SHA都在summary.json，最终任务结果不由离线MAE代替。
