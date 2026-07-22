# 思考点一阶段报告：Fast-WAM 的 LIBERO-Plus 环境 OOD 鲁棒性

报告日期：2026-07-22  
当前阶段：工程链路与三卡 pilot 已通过；正式 Clean/OOD rollout 尚未执行  
被评测策略：Fast-WAM `libero_uncond_2cam224`

## 摘要

思考点一要回答的可识别问题是：**冻结同一份 Fast-WAM checkpoint 后，从标准 LIBERO 切换到 LIBERO-Plus 官方环境扰动变体，任务成功率下降多少，且哪类扰动、哪个难度最敏感？**

截至本报告日期，环境、模型、checkpoint、LIBERO/LIBERO-Plus adapter、单卡 smoke、三卡分片、EGL、断点续跑、轨迹/视频记录和聚合链路均已通过真实运行验证。三卡 pilot 计划 9 条，其中 8 条真实执行、1 条因官方分层无候选而按协议跳过；8 条均正常结束，0 exception。pilot 的 2/8 成功只用于验证链路和估算成本，样本太小且没有对应的正式 Clean baseline，不能作为最终鲁棒性结论。

正式 manifest 已重新生成并审计：

- Clean：800 个 runnable rollout。
- OOD：6,839 条计划，其中 6,771 个 runnable rollout、68 条 skipped 审计记录。
- 正式剩余真实计算量：**7,571 个 rollout**，不是只有 6,771 个；若不执行 800 个 Clean baseline，就只能报告 OOD 绝对表现，不能回答 Clean→OOD 的下降。
- 另有 121 条没有官方 difficulty 的 `libero_goal / Light Conditions` 记录，按预注册协议排除在 easy/medium/hard 主分析之外，不擅自分级。

因此，当前最准确的阶段结论是：**评测系统已经具备正式运行条件，研究结果尚待 800 Clean + 6,771 OOD rollout 完成后才能形成。**

## 1. 研究问题与结论边界

### 1.1 当前可以回答的问题

在以下条件保持不变时，测量标准环境与官方扰动环境之间的差异：

- checkpoint 与 dataset stats；
- Fast-WAM 模型配置及动作预/后处理；
- 基础任务、成功判定和配对 seed；
- 推理精度、control horizon、相机输入尺寸和最大步数。

变化因素仅为 LIBERO-Plus 预生成的五类环境扰动：

1. `camera_viewpoints`
2. `light_conditions`
3. `background_textures`
4. `robot_initial_states`
5. `objects_layout`

主结果将报告 Clean/OOD 成功率、绝对与相对下降、95% bootstrap CI，以及按 suite、task、扰动类别和 difficulty 的分层统计。

### 1.2 当前不能回答的问题

- 不能把四个 LIBERO suite 称为 unseen-object 或 unseen-task：release 训练配置已经包含这些 suite。
- 不能声称 cross-platform transfer：LIBERO 与 RoboTwin 的接口和 release checkpoint 不同。
- 不能声称“未来想象改善泛化”：当前 release 是 uncond Fast-WAM，动作不读取预测未来；缺少训练配方匹配的 Joint WAM/IDM checkpoint。
- 不能把仿真 OOD 结果直接外推为真机鲁棒性。

详细可识别性说明见 [thought1_generalization.md](thought1_generalization.md)。

## 2. 可复现环境与工件

| 项目 | 固定值 |
| --- | --- |
| Python | 3.10.20 |
| PyTorch / CUDA | 2.7.1+cu128 |
| GPU | 3 张可见 `NVIDIA GeForce RTX 4090`，运行记录约 47.37 GiB/卡 |
| 精度 | bf16，TF32 enabled，每 GPU 1 worker |
| Fast-WAM commit | `45d8e1458921d83f8ad6cf9ce993d371208dabd0` |
| LIBERO commit | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| LIBERO-Plus commit | `4976dc30028e805ff8094b55501d532c48fec182` |
| pilot 项目 commit | `2e1736aab70edcab680fc7e0b13354c7afb2fcdf` |
| checkpoint SHA-256 | `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579` |
| dataset stats SHA-256 | `30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638` |
| classification SHA-256 | `faa87cce3e3ba434da01df7c77523a391b5f2912e4774330b0aa1be5f6a999e6` |

所有正式 Clean/OOD 结果必须继续使用上述 checkpoint/stats 组合。配置或 classification 变化后必须重新生成 manifest。

## 3. 实验协议

### 3.1 Clean

- 四个 suite，每个 suite 10 个标准任务。
- 每个任务 20 个初始化 index/seed。
- 共 `4 × 10 × 20 = 800` 个 runnable rollout。
- backend 使用原版 LIBERO。

### 3.2 OOD

- backend 使用 pinned LIBERO-Plus。
- 五类扰动，官方 difficulty 映射为 easy=1–2、medium=3、hard=4–5。
- `variant_selection=all_once`，每个官方预生成 variant 只执行一次。
- 每个 runnable job 的 `episode_index=0`、`initial_state_index=0`；不执行 `10,030 × 20` 的重复采样。
- 没有候选的 task/category/level 组合保留为 skipped 审计记录，不进入成功率分母。

### 3.3 配对与统计

- seed 由 `(base seed, suite, base task, episode index)` 确定，condition 不进入 seed 公式。
- 同一策略的 Clean/OOD checkpoint hash 必须一致，否则聚合拒绝比较。
- success 只使用官方环境成功判定；`max_steps` 计任务失败，exception 也计失败，skipped 排除。
- 成功率 CI 使用固定随机种子的 2,000 次 bootstrap。
- 优先报告配对四格：Clean 成功/OOD 失败、Clean 失败/OOD 成功、双成功、双失败。

完整统计口径见 [experiment_protocol.md](experiment_protocol.md)。

## 4. 已完成的工程与实验阶段

| 阶段 | 真实执行结果 | 能证明什么 |
| --- | --- | --- |
| 环境与 doctor | checkpoint、stats、assets、运行时模型和 3 GPU 均可用 | 依赖与路径完整 |
| Clean smoke | 2/2 completed，2 success，0 exception | 原版 LIBERO、checkpoint、动作、视频链路可用 |
| OOD smoke | 4/4 completed，4 success，0 exception | Plus camera/light reset、扰动和 init-state 路由可用 |
| 三卡 OOD pilot | 9 planned，8 completed，1 skipped，0 exception | 三卡分片、EGL、并发模型加载、结果落盘与聚合可用 |
| 正式 plan | 800 Clean；6,771 OOD runnable；68 OOD skipped | 正式任务单位与数量已锁定 |
| 正式 evaluation | 尚未执行 | 尚不能给出正式鲁棒性结论 |

已解决的关键阻碍包括：

- 原版 LIBERO 与 LIBERO-Plus 同名 Python 包的进程级隔离；
- PyTorch 2.6+ 默认 `weights_only=True` 与旧 NumPy init-state pickle 的兼容；
- LIBERO-Plus camera/light 等变体复用基础 init state 的路径解析；
- assets 压缩包嵌套路径与国内下载断流；
- ModelScope commit SHA 不能直接作为 `snapshot_download` revision；
- 三卡 torchrun 下 CUDA/EGL 设备映射与 episode-level 稳定分片。

工程复盘见 [engineering_highlights.md](engineering_highlights.md)。

## 5. 三卡 pilot 结果

### 5.1 完整性与运行健康度

| 指标 | 实测值 |
| --- | ---: |
| planned / runnable / skipped | 9 / 8 / 1 |
| completed / exception | 8 / 0 |
| success / max_steps | 2 / 6 |
| action finite 且非全零 | 8 / 8 |
| 可解码 MP4 | 8 / 8 |
| 末端执行器首末位移 | 0.316–0.407 m |
| checkpoint hash 种类 | 1 |
| 三个 rank 分配 | 3 / 4 / 2 jobs |
| job 重复或遗漏 | 0 / 0 |

唯一 skipped 是 `libero_spatial` task 4 的 `objects_layout/easy` 没有官方候选；它是预期协议结果，不是运行异常。

### 5.2 诊断性成功率

| 扰动 | Attempted | Success | Pilot success rate |
| --- | ---: | ---: | ---: |
| Camera viewpoints | 3 | 1 | 33.33% |
| Robot initial states | 3 | 1 | 33.33% |
| Objects layout | 2 | 0 | 0.00% |
| 合计 | 8 | 2 | 25.00% |

总体 95% bootstrap CI 为 `[0.00%, 62.50%]`。区间极宽，且 pilot 只覆盖 `libero_spatial` 的 3 个 task、3 类 easy 扰动，没有配对 Clean 结果。因此这些数字只能说明正式实验可能观察到明显失败，不能用于类别排序或泛化结论。

### 5.3 运行成本

| 指标 | 实测值 |
| --- | ---: |
| 三个模型并发加载 Wan/Fast-WAM | 约 369 s |
| pilot 总墙钟时间 | 约 11 min 35 s |
| 单个 completed episode 平均时长 | 71.53 s |
| 平均推理延迟 | 983.42 ms |
| P50 / P95 推理延迟 | 970.31 / 1038.35 ms |
| 每卡记录的峰值显存 | 23,814.42 MB |

按 pilot 平均 episode 时长理想线性外推，6,771 个 OOD rollout 约消耗 134.5 GPU-hours，即三卡理想墙钟约 44.8 小时；800 个 Clean 若按相同速度约需 5.3 小时。考虑静态分片尾部、suite 切换、`libero_10` 使用 700 policy steps、任务难度差异和失败视频 I/O，正式 Clean+OOD 应预留 **60–72 小时连续三卡窗口**。该区间是容量规划，不是完成时长承诺。

机器生成的 pilot 汇总见 [outputs/ood_pilot/summary/report.md](../outputs/ood_pilot/summary/report.md) 和 [metrics.json](../outputs/ood_pilot/summary/metrics.json)。

## 6. 正式 manifest 审计

### 6.1 按 suite

| Suite | Clean runnable | OOD planned | OOD runnable | OOD skipped |
| --- | ---: | ---: | ---: | ---: |
| libero_spatial | 200 | 1,685 | 1,661 | 24 |
| libero_object | 200 | 1,755 | 1,742 | 13 |
| libero_goal | 200 | 1,692 | 1,681 | 11 |
| libero_10 | 200 | 1,707 | 1,687 | 20 |
| 合计 | **800** | **6,839** | **6,771** | **68** |

### 6.2 按扰动类别

| OOD 类别 | Runnable variants |
| --- | ---: |
| Camera viewpoints | 1,599 |
| Light conditions | 1,021 |
| Background textures | 1,076 |
| Robot initial states | 1,550 |
| Objects layout | 1,525 |
| 合计 | **6,771** |

### 6.3 按 difficulty

| Difficulty | Runnable variants |
| --- | ---: |
| Easy | 2,561 |
| Medium | 1,535 |
| Hard | 2,675 |
| 合计 | **6,771** |

审计同时确认：所有 manifest 内 job ID 唯一；所有 runnable Plus row 使用 `all_once`；官方变体 `(suite, upstream_task_id)` 无重复；全部 10,030 个 classification row 均能解析到真实 init-state 文件。

## 7. 剩余工作与完成门槛

### 7.1 必须执行的正式计算

1. 四个 suite 的 800 个 Clean rollout。
2. 四个 suite 的 6,771 个 OOD runnable rollout。
3. 保留 68 条 skipped 审计记录，不为它们伪造 variant 或 init state。
4. 对八个实验目录聚合，并生成一个显式包含 Clean 与 OOD 输入的 combined report。

默认 resume 会跳过已经落盘的 job，适合中断恢复。正式运行中不要把 `--rerun failed` 当作常规 resume，因为它也可能重跑正常的 `max_steps` 策略失败；只有定位并修复系统性异常后才使用，并保留旧 JSONL 审计记录。

### 7.2 正式结果验收

- 7,571 个 runnable job 均有最终记录；68 个 skipped 均有明确原因。
- 无未解释的 exception、CUDA OOM、EGL 冲突、重复 job 或分片遗漏。
- Clean/OOD checkpoint hash 与 stats 来源一致。
- action 均为 finite，异常的全零轨迹单独审计。
- 按 suite、task、category、difficulty 输出成功率、CI 和失败数。
- combined aggregate 输出 Clean/OOD drop 与配对四格。
- 抽检各类别/难度视频，确认扰动真实生效；对主要失败模式完成人工分类。

## 8. 当前可写与不可写的结论

### 当前可以写

- 构建并真实验证了 Fast-WAM 在 LIBERO/LIBERO-Plus 上的三卡、可恢复、可审计 OOD 评测系统。
- Clean、OOD smoke 与三卡 pilot 均无运行 exception；正式 6,771 个 OOD variant 的 manifest 已按官方 task-instance 协议生成并审计。
- pilot 显示系统能够捕获真实策略失败，同时 action、机器人运动、视频和成功判定保持正常。

### 当前不能写

- “Fast-WAM 的正式 OOD 成功率是 25%”或“object layout 最差”。
- “Fast-WAM 从 Clean 到 OOD 下降了 X%”。
- “模型具有 unseen-object/task/platform 泛化能力”。
- “未来想象能提升或不能提升 OOD 泛化”。

这些表述分别需要正式全量结果、真正的 holdout split、跨平台 adapter/checkpoint 或配方匹配的 future/no-future 训练对照。

## 9. 证据索引

- 实施步骤与停止条件：[thought1_execution_guide.md](thought1_execution_guide.md)
- 完成度审计：[thought1_readiness.md](thought1_readiness.md)
- 研究结论边界：[thought1_generalization.md](thought1_generalization.md)
- 实验协议：[experiment_protocol.md](experiment_protocol.md)
- 工程难点与简历素材：[engineering_highlights.md](engineering_highlights.md)
- Clean smoke：`outputs/clean_smoke/`
- OOD smoke：`outputs/ood_smoke/`
- 三卡 pilot：`outputs/ood_pilot/`
- 正式 manifests：`outputs/thought1/fastwam/<suite>/{clean,ood}/job_manifest.jsonl`

## 10. 阶段结论

思考点一的**工程准备阶段已经完成**：单卡与三卡真实链路、两套 backend、官方扰动、init-state、安全加载、分片、resume、记录与聚合均有运行证据。思考点一的**科学实验阶段尚未完成**：正式结果还需要 800 个 Clean 和 6,771 个 OOD rollout。完成这 7,571 个 rollout 并生成 combined report 后，才能回答相同 Fast-WAM checkpoint 在标准 LIBERO 到 LIBERO-Plus 环境 shift 下的成功率下降及敏感扰动分布。
