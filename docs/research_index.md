# 研究总控：Fast-WAM 在 OOD 环境中真的不需要未来想象吗？

更新日期：2026-07-28

本文是项目的研究入口与证据总账。详细命令、协议和实现分别链接到各阶段手册；这里只回答四件事：当前做到哪里、哪些数字可以使用、阶段之间如何隔离、下一步是什么。

## 1. 论文主线

论文不是直接提出一个新模型，而是依次建立三层证据：

1. **环境鲁棒性**：冻结官方 Fast-WAM，测标准 LIBERO 到 LIBERO-Plus 的成功率下降。
2. **未来一致性**：不改变控制动作，离线观察同一 checkpoint 生成的未来是否与实际变化相符。
3. **未来因果增益**：加入轻量 Future-to-Action Adapter，通过 K=0/1/2/4 对照判断多少未来信息能改善 OOD，以及代价多大。

三层结论不能互相替代。阶段一的失败不能证明未来有用；阶段二的一致性不能证明动作依赖未来；只有阶段三的配方匹配对照才允许讨论显式未来的因果增益。

## 2. 当前真实状态

| 阶段 | 工程状态 | 真实运行证据 | 科学结论状态 |
| --- | --- | --- | --- |
| 阶段一：ID/OOD 评测 | Clean/OOD、3 GPU、resume、聚合和全量 trace/video 审计均完成 | `P1-FORMAL-v1`：800 Clean + 6,771 OOD，68 skipped，0 exception | **FORMAL 完成**。Clean 97.25%→OOD 47.70%，drop 49.55 pp |
| 阶段二 A：unconditional future consistency | 五类正式 runner、task-level 派生分析、trace/media 审计均完成 | 200 Clean + 532 OOD；1,010 probes / 2,020 aligned frames / 0 error | **正式数据收集完成；自动关联结果完成**。OOD consistency proxy 下降；统计计划未预冻结，human endpoint 待完成 |
| 阶段二校准：static/no-op null | 独立命令、输出、resume、聚合和 freeze gate 已实现 | 100 Clean + 100 OOD；200/200 eligible；阈值 `0.0167421166` | **正式校准完成并在 diagnostic 启动前接受**。0 predicted-static、6 actual-static probes |
| 阶段二盲审 | public packet/private key 分离与泄漏校验已实现 | 真实 pilot 7 cases / 28 media 全量解码；0 sensitive public key | **流程 PILOT 完成，人工标注未开始**。不得写成人工质量结果 |
| 阶段二正式抽样 | outcome-blind planner、anchor、exact-ratification 与 formal gate 已实现 | 200 Clean + 532 OOD 已 exact-ratify 并全部运行 | **完成**。只认证 Phase 2 future 指标前 job ID 不变；不冒充阶段一 outcome 前 preregistration |
| 阶段二统计协议 | episode→task 分层、task bootstrap、首 probe/outcome gate 已实现 | 10,000 次 suite-stratified task bootstrap；730/732 outcome match | **post-run analysis 完成，非 preregistered confirmatory**。DRAFT 未在正式指标前冻结；人工 endpoint 待完成 |
| 阶段二 B：action-conditioned future consistency | 严格门禁、schema、runner、测试已实现 | CPU/mock 与门禁测试通过 | **阻塞**。官方 release 为 `action_conditioned=false`，且没有可信匹配 checkpoint |
| 阶段三：Future-to-Action Adapter | Phase A/B/C/D 完成；Phase E 真实小训练已执行但总 Gate 未通过 | A0/A1 各完成 resumed/uninterrupted 100 step；第 2 step 非 gate 梯度、Adapter-only resume、确定性 SHA、单卡显存通过；固定 loss probe 未稳定下降 | **阻塞在 Gate E.1 优化诊断**。不得扩 A2/A4 或启动 ID/OOD；无 future 增益或 K 优劣结论 |

因此，“阶段一已经完成”的准确说法是：**阶段一工程、正式全量计算、聚合与
完整性审计均已完成；失败机制人工 taxonomy 尚未完成，但不阻塞主成功率结论。**

## 3. 证据等级

以后所有表格、简历和论文数字都必须带证据等级：

| 等级 | 含义 | 能否写成论文结果 |
| --- | --- | --- |
| `PLAN` | manifest、doctor、dry-run | 否 |
| `TEST` | 单元/集成测试或 mock | 否 |
| `SMOKE` | 少量真实模型和环境运行，验证链路 | 否 |
| `PILOT` | 小规模真实样本，估算失败模式和成本 | 只能作为预实验，必须显式标注 |
| `FORMAL-COLLECTION` | 数据收集配置/分母前锁定，provenance、聚合与审计通过 | raw 数据可进入论文；统计 claim 资格另审 |
| `FROZEN-CONFIRMATORY` | estimand、统计方法与停止规则在指标产生前冻结 | 可以作为确认性推断 |
| `POST-RUN-ANALYSIS` | 正式 raw 数据上的事后或未冻结分析 | 可以披露为探索/关联结果，不能冒充预注册确认性 |

任何 `SMOKE/PILOT` 成功率都不得自动抄入摘要、主表或简历效果数字。

## 4. 阶段隔离规则

### 4.1 代码与配置

- 阶段一冻结点：tag `thought1-baseline-v1`，commit `0df5fe2`。
- 当前 `main` 保留阶段一评测路径；阶段二只通过显式
  `diagnose-future` 或 `calibrate-static` 命令进入。
- 阶段一配置位于 `configs/eval_*.yaml` 和 `configs/studies/thought1.yaml`。
- 阶段二 A 使用 `configs/studies/thought2_unconditional_*.yaml`。
- 阶段二 static calibration 使用
  `configs/studies/thought2_static_calibration_*.yaml`。
- 阶段二 B 使用 `configs/studies/thought2_shadow_*.yaml`，当前应在能力门禁处失败。
- 阶段三使用 `configs/thought3/`、`src/fastwam_ood_eval/thought3/` 和独立
  `outputs/thought3/` namespace；根 CLI 只做惰性加法式分发。

### 4.2 输出

```text
outputs/thought1/...                 # 阶段一正式结果
outputs/thought2_unconditional_*     # 阶段二 A
outputs/thought2_static_calibration_* # 阶段二独立 null 校准
outputs/thought2_outcome_blind_*     # 阶段二只读抽样 manifest
outputs/thought2_future_blind_*       # 阶段二 public packet/private key
outputs/thought2_shadow_*            # 阶段二 B
outputs/thought2/five_category_formal_v1/ # 阶段二五类正式 raw + 独立派生分析
outputs/thought3/...                 # 阶段三训练、cache、评测
```

阶段二 future diagnostics 只读阶段一 `experiment_manifest.json` 和
`job_manifest.jsonl`，并验证 checkpoint hash、Fast-WAM commit、控制协议与
source manifest hash。Static calibration 另行规划独立 task/seed，只运行标准
no-op，不读取 pilot success/OOD 标签。这些输出目录必须互不包含；程序会在模型
加载前拒绝混写。

`require_frozen_cohort=true` 会阻止 draft 被误运行。新 holdout 仍应在 source
outcome JSONL 前使用 `frozen_before_source_outcomes`；现有 v2 则走更窄的
`ratified_before_diagnostic_outcomes`：保留全部原 job ID 和 draft hash，明确
记录阶段一 outcome 在 ratification 时已经存在。它防止根据阶段二 future metric
改样本，但不能声称对阶段一结果盲化。Blind packet 与 private unblinding key
仍须使用彼此分离、且不位于任一 diagnostic source 内的目录。

### 4.3 重新运行阶段一

有两种合法方式：

1. **严格复现冻结版本**：在独立 git worktree checkout `thought1-baseline-v1`，继续使用原阶段一配置与全新输出目录。
2. **使用当前 main 重跑**：阶段一 `evaluate` 路径保持独立，但必须记录新的项目 commit，并与旧结果分目录；只有协议、checkpoint、stats 与 manifest hash 一致时才能合并。

不要在已有正式 JSONL 上用 `--overwrite`。正常中断使用 resume；只有确认系统异常并保留旧记录后才选择性重跑。

所有 `FORMAL` run 启动前必须确认项目和三个上游 checkout 的
`*_dirty=false`。新 manifest 会同时记录 commit 与 dirty 状态；dirty run
最多降级为 PILOT，不得进入论文主表。LIBERO-Plus 的
`.downloads/` 仅保存下载缓存，不进入 Python/runtime source；该唯一例外由
`libero_plus_dirty_ignored_untracked` 显式写入 provenance，tracked 修改永不忽略。

## 5. 当前可记录的关键数字

| 来源 | 关键数据 | 允许的解释 |
| --- | --- | --- |
| 阶段一 Clean smoke | 2/2 success，0 exception | Clean 链路可用；不是成功率估计 |
| 阶段一 OOD smoke | 4/4 success，0 exception | camera/light 链路可用；不是 OOD 结论 |
| 阶段一 OOD pilot | 8 attempted，2 success，1 skipped，0 exception；平均动作推理 983.42 ms | 可估算成本和发现失败；不能写成正式 25% OOD 成功率 |
| 阶段一正式 plan | 800 Clean；6,771 OOD runnable；68 skipped | 正式计算分母已锁定 |
| 阶段一 FORMAL 总体 | Clean 778/800=97.25%；OOD 3,230/6,771=47.70%；drop 49.55 pp；0 exception | 当前五类官方环境 shift 的正式主结果 |
| 阶段一 FORMAL 类别 | camera 15.13%、robot-init 42.84%、background 51.49%、layout 61.25%、light 81.88% | Camera 是跨四 suite 稳定最低类别；中间类别精确排序需看 task-cluster |
| 阶段一 FORMAL 难度 | easy/medium/hard 59.82%/49.51%/35.07%；五类内部均粗粒度下降 | 支持 severity 分层，不把官方 1–5 当作严格等距连续量 |
| 阶段一 FORMAL 工程 | 7,571 traces、3,563 failure videos、2,399,314 action steps；0 NaN/空动作/静止 episode | 排除评测链路故障；不替代人工失败机制分类 |
| 阶段二 A real smoke | 1 episode / 1 probe；2 个 aligned future frames；0 probe error | 真实未来诊断链路可用 |
| 阶段二 A real smoke 资源 | 2-step future generation 1,223.53 ms；完整诊断 4,616.06 ms；峰值 24,841.09 MB | 仅作 smoke 容量证据；不能外推 20-step 延迟 |
| 阶段二 A 20-step pilot | Clean 2 episodes/2 probes；camera-easy OOD 3 episodes/5 probes；合计 14 aligned frames、0 error | 正式去噪步数的诊断链路可用；不是总体样本 |
| 阶段二 A 动作隔离 | 7/7 probe 的执行动作与阶段一 trace 逐元素一致，最大绝对差 0；5/5 episode outcome 一致 | shadow future 没有改变这批基线执行 |
| 阶段二 A pilot 资源 | Clean/OOD episode-weighted generation 4,108.12/4,563.88 ms；完整诊断 7,214.25/8,200.65 ms；峰值均 24,841.09 MB | 20-step 小样本容量与延迟证据 |
| 阶段二 static null pilot | 7/7 eligible、0 error；同帧编码噪声全为 0；8-step no-op energy 中位数/最大值 `0.006615/0.013223` | 证明旧阈值 1.0 数量级错误；不是正式阈值 |
| static 候选敏感性 | 候选 `0.013223` 下 predicted/actual static 均 0/7；旧阈值下均 7/7 | 只读派生重分类；原 diagnostics 未改写，candidate 未冻结 |
| 阶段二 blind packet pilot | packet `16a1dbc...665aef`；7 cases / 28 media；public/private hash 校验通过；human labels 0/7 | 盲审工具链可用；不能评价 future 质量 |
| 阶段二 outcome-blind draft v2 | 200 Clean + 532 OOD = 732；68 unsupported；0 supported shortfall；Clean 强制 episode-0 anchor | 覆盖设计草案；`frozen=false`，不是预注册正式样本 |
| 阶段二 static FORMAL | 200/200 eligible，Clean/OOD 各 100、五类 OOD 各 20；阈值 `0.0167421166` | 本次 formal diagnostic 的 run-level 阈值锁；对应 re-encoded frame embedding |
| 阶段二五类正式收集 | 732/732 episodes；1,010 probes；2,020 aligned future frames；0 error | 正式数据完整；4,040 个媒体全量解码通过 |
| 阶段二 ID→OOD consistency | cosine distance `0.1025→0.1341`，task-equal 差 `+0.0316`，95% CI `[0.0254,0.0381]` | OOD 下自动 future–realized consistency proxy 变差；非语义正确率 |
| 阶段二视觉方向 | motion-direction cosine `0.7416→0.5518`，差 `−0.1898`，95% CI `[−0.2134,−0.1664]` | OOD 下预测与受保护动作实际造成的视觉变化方向更不一致；不是直接 action cosine |
| 阶段二 outcome 关联 | OOD failure−success cosine `+0.0249`；仅首 probe `+0.0197`，两者 CI 均不跨 0 | future inconsistency 与失败相关；不能写成 future error 导致失败 |
| 阶段二正式资源 | 20-step generation mean/p50/p95 `3354.66/3316.96/3564.12 ms`；full diagnostic `5816.77/5762.95/6271.52 ms`；峰值 `24841.09 MB` | Shadow 诊断成本；不是动作延迟或阶段三 K-step 在线成本 |
| 阶段三 Phase B | 默认 Adapter `1,371,137` 参数；native future schema `[48,2,14,28]`；K1/K2/K4 paired cache、resume/checksum/mock training/checkpoint/online-no-cache tests 通过 | 只证明工程 contract；不是模型效果、真实延迟或显存结果 |
| 阶段三 Phase C | 单条真实样本的 K1/K2/K4 分别为 `120.34/165.62/325.30 ms`；video-only parity max diff 0；zero-gate action bitwise equal；backbone gradient 0；执行峰值 `12.964 GiB` | 单卡真实工程可行性；latency 是单样本 telemetry，不是正式 P50/P95；不支持 OOD 增益 |
| 阶段三 Phase D | 32 base samples、96 entries、12 shards；完整 split 37/5，cache 内 28/4；K1/K2/K4 mean `127.54/186.62/362.99 ms`；0.806 base sample/s；执行峰值 `12.677 GiB` | 真实离线 cache 工程门禁；sampling 不含 action denoising，不能作为在线总延迟或 OOD 效果 |

20-step pilot 的 episode-weighted Clean→OOD 描述值为：latent L1
`0.1512→0.2002`、cosine distance `0.1168→0.1942`、motion-direction cosine
`0.7697→0.5283`。它只覆盖 2 个 Clean 与 3 个 camera/easy OOD episode，
严格配对只有 1 对，且 static 只有 7 条 null candidate、尚未冻结；因此只能登记为“值得正式检验的
OOD 一致性下降假设”，不能进入论文结论表。

## 6. 分阶段文档

- 阶段一报告与完成度：[thought1_report.md](thought1_report.md)、[thought1_readiness.md](thought1_readiness.md)
- 阶段一执行手册：[thought1_execution_guide.md](thought1_execution_guide.md)
- 阶段二概念与上游审计：[thought2_concepts.md](thought2_concepts.md)、[thought2_upstream_audit.md](thought2_upstream_audit.md)
- 阶段二执行与标注手册：[thought2_execution_guide.md](thought2_execution_guide.md)
- 阶段二盲审与 outcome-blind 抽样：[thought2_blind_review_and_sampling.md](thought2_blind_review_and_sampling.md)
- 阶段二统计分析计划（当前 DRAFT）：[thought2_statistical_analysis_plan.md](thought2_statistical_analysis_plan.md)
- 阶段二五类正式结果：[thought2_formal_results.md](thought2_formal_results.md)
- 阶段二 static/no-op 校准手册：[thought2_static_calibration.md](thought2_static_calibration.md)
- 阶段三审计与设计：[thought3_upstream_audit.md](thought3_upstream_audit.md)、[thought3_design.md](thought3_design.md)
- 阶段三 Phase B 验收：[thought3_phase_b_report.md](thought3_phase_b_report.md)
- 阶段三 Phase C 验收：[thought3_phase_c_report.md](thought3_phase_c_report.md)
- 阶段三 Phase D 验收：[thought3_phase_d_report.md](thought3_phase_d_report.md)
- 阶段三 Phase E 验收/失败诊断：[thought3_phase_e_report.md](thought3_phase_e_report.md)
- 阶段三 Gate E.1 单样本诊断预注册：
  [thought3_phase_e1_protocol.md](thought3_phase_e1_protocol.md)
- 阶段三 Gate E.1 单样本诊断结果：
  [thought3_phase_e1_report.md](thought3_phase_e1_report.md)
- 阶段三数据/训练/评测：[thought3_data_protocol.md](thought3_data_protocol.md)、[thought3_training.md](thought3_training.md)、[thought3_evaluation.md](thought3_evaluation.md)
- 阶段三分析 DRAFT 与限制：[thought3_analysis_protocol_DRAFT.md](thought3_analysis_protocol_DRAFT.md)、[thought3_limitations.md](thought3_limitations.md)
- 阶段三旧版路线摘要：[thought3_adapter_plan.md](thought3_adapter_plan.md)
- 实验、失败尝试和结论台账：[experiment_ledger.md](experiment_ledger.md)
- 工程难点与简历素材：[engineering_highlights.md](engineering_highlights.md)

## 7. 当前优先级

1. Gate E.1 已通过：A0/A1 单样本固定 loss 分别下降 92.93%/99.58%，且
   frozen-before/after SHA 完全相同；这只证明注入图可以 overfit。
2. 阶段三下一步先冻结 Gate E.2：8-sample train-only 的少量 LR/尺度诊断，
   同时约束 loss 与 BF16 `delta/action-hidden`，不读取 development/OOD outcome。
3. Gate E 多样本 fixed probe 尚未下降，且 Gate E.1 出现 A0 1.91×、A1 0.70×
   hidden correction；在稳定配方和完整 28/4 Gate E 通过前，不扩 A2/A4。
4. 在不按自动 metric/outcome 挑案例的前提下，冻结正式 human-review budget、
   seed 和每 job 至多一个 probe；至少两名 reviewer 独立标注并保留 agreement/
   adjudication。
5. 人工 endpoint 只回答 goal progress、physical plausibility、局部 action
   execution 和 future–actual agreement；不把它升级成 future-to-action 因果。
6. 阶段三 OOD 结果不用于边训练边选择 K，
   Phase F 后先冻结分析协议再解锁正式 cohort。
