# 研究总控：Fast-WAM 在 OOD 环境中真的不需要未来想象吗？

更新日期：2026-08-03

本文是项目的研究入口与证据总账。论文正文与图表见
[完整论文](../paper/manuscript.md)，分层论证见
[论文证据链](../paper/evidence_chain.md)。详细命令、协议和实现分别链接到各
阶段手册；这里只回答四件事：当前做到哪里、哪些数字可以使用、阶段之间如何
隔离、下一步是什么。

## 1. 论文主线

论文不是直接提出一个新模型，而是依次建立五层证据：

1. **环境鲁棒性**：冻结官方 Fast-WAM，测标准 LIBERO 到 LIBERO-Plus 的成功率下降。
2. **未来一致性**：不改变控制动作，离线观察同一 checkpoint 生成的未来是否与实际变化相符。
3. **动作技术依赖**：用 K=1 correct/null/shuffle 输入干预，判断 future 的
   具体内容是否改变动作。
4. **未来任务效用**：用 matched K=0/K=1 训练判断动作敏感性是否形成
   held-out 收益；只有先通过该门禁，才允许扩展 K=2/K=4 和 OOD rollout。
5. **缺口定位**：冻结 Video/Action 主干，以 exact-state Camera/Lighting probes
   和 geometry-subspace intervention 区分 representation、interface 与
   camera-equivariance gap。

五层结论不能互相替代。阶段一的失败不能证明未来有用；阶段二的一致性不能
证明动作依赖未来；动作受 future 影响也不能证明任务收益。当前第四层在冻结
K=1 配方上得到离线负结果，因此 OOD success 因果问题保持未回答。
Thought4 进一步将下一方法假设定位为 `camera_equivariance_gap`，但仍没有产生
任何新方法 success 结果。

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
| 阶段三：Future-to-Action Adapter | Phase 0 审计完成；Phase 1 分支 A；Phase 2 完整双卡训练完成 | Phase 1 correct-null/shuffle/action-hash 均 `8/8`；Phase 2 A0 reduction `+1.845%`、A1 `−1.712%`，A1 比 A0 高 `3.624%` 且 4/4 dev sample 更差 | **有效离线负结果，路线停止**：future 内容会改变动作，但冻结 K=1 Adapter 未形成 held-out utility；Phase 3/OOD/A2/A4 锁定 |
| Thought4：Geometry–Action Gap | FP32 smoke v8 与 64-state formal v6 完成；1,586 工件只读审计 | 256 paired samples、12,544 features；Camera gap 0.020273 m > Lighting 0.011660 m；rank-3 shuffle 36/36 超过 replay floor | **FORMAL DIAGNOSTIC 完成**：`camera_equivariance_gap`；只解锁 Geo-REPA + relative pose / camera-ray equivariance，尚无方法/rollout 效果 |

因此，“阶段一已经完成”的准确说法是：**阶段一工程、正式全量计算、聚合与
完整性审计均已完成；失败机制人工 taxonomy 尚未完成，但不阻塞主成功率结论。**

阶段三不再由 surrogate Gate 阻塞。E5–E9 作为离线证据账本，Phase 1 证明
future-content action sensitivity；Phase 2 则得到有效离线负结果。按冻结停止
规则，不再挑 checkpoint、LR、sample weight、K 或 OOD outcome 重开该 Adapter
路线。

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
outputs/thought4/...                 # 冻结 geometry/action probes 与 formal diagnosis
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
| 阶段三 Gate E.2 | A0/A1 × 三 LR 共 1,200 step；六轨迹 execution/pairing/checkpoint/frozen SHA 全通过；峰值 `13,273.17 MiB`；Gate 总耗时 24.10 分钟 | A1 在 `1e-4/3e-4` 的 8-sample mean fixed loss 下降 24.19%/40.01%，但都只有 4/8 sample 不变差；无共同 eligible LR，属于 FAILED engineering gate，不是 future 效果 |
| 阶段三 Gate E.3 v2 | 320/320 held-out multi-flow forwards；0 optimizer/backward；执行、零权重、配对和 frozen SHA 全通过 | E.2 fixed-flow checkpoint 的 A1 `24.19%/40.01%` 降幅在 held-out flow 只剩 `0.025%/−1.31%`；有效负 Gate，不否定 future latent |
| 阶段三 Gate E.4 | 六轨迹共 1,200 optimizer steps、480 held-out objectives；峰值 `13,273.17 MiB`；总耗时 25.47 分钟；401,013,164 bytes | diversified train-flow 后六条 reduction 均为正但只有 `0.997%–1.948%`；A1@3e-4 为 7/8 但未达 10%，无共同 eligible LR |
| 阶段三 Gate E.5 | 六轨迹共 1,200 updates、9,600 train objectives、480 held-out objectives；120/120 execution checks；总耗时 114.65 分钟；413,198,197 bytes | A1@3e-4 held-out loss 下降 19.668%、8/8 不变差并单条过门；同 LR A0 仅 2.638%，故无共同 eligible LR。属于有效负总 Gate 与待独立复验的探索性 A1 信号 |
| 阶段三 Gate E.6 | 新 cohort 双轨共 400 updates、3,200 train objectives、160 held-out objectives；总耗时 43.89 分钟；全部 execution/paired/frozen checks 通过 | A1 下降 14.842%、7/8 且相对 A0 final mean 低 13.815%；A0 仅 4/8 不变差，故为有效负 Gate，但 A1 工程信号得到序贯复现 |
| 阶段三 Gate E.7 | 冻结 8 个既有 checkpoint、primary flow `6..10`、continuity flow `1..5`、800 个只读 forward objective；0 backward/optimizer/新训练；13.98 分钟 | Primary A0 50/100 通过、150/200 因 5/8 失败；step-200 mean 仍比 step 50 低 5.651%，故分类为 `not_supported_no_material_late_degradation`；A1 信号增强但无 joint candidate |
| 阶段三 Gate E.8 | A0 step 100/200；全新 flow `11..74`，双 32-flow block；1,536 forward；20k paired bootstrap + 20k five-flow sensitivity；0 backward/optimizer/训练；18.51 分钟 | 工程 Gate 通过；step 100 三 panel 全过，step 200 pooled reduction 为 3.472%/4.047%/3.728%，但稳定性为 4/8、5/8、4/8；仅 1/3 target 确认恶化，另有 1 条非 target 确认恶化，故为 `mixed_or_inconclusive` |
| 阶段三 Gate E.9a-v1/v2 | v1 为 0-objective invalid；v2 四轨各完成 200 updates/1,600 objectives，held-out `75..106`，88.60 分钟 | v2 raw A0/A1 reduction `4.175%/12.994%`，normalized `2.983%/11.010%`；tail harm `2/0→0/0`，但 normalized paired `8.274%<10%`。RNG identity 字段未落盘使 engineering Gate invalid；无 E.9b candidate |
| 阶段三 Phase 0 E9a-v2.1 audit | CPU-only、0 forward/backward/optimizer/checkpoint tensor load、0 CUDA、父目录 0 write；27/27 checks true | 恢复登记为 `audit_valid_scientific_failed`；normalization tail signal 合法，但 `sample_tail_mitigation_not_supported`、无独立复验 candidate、E9b locked |
| 阶段三 Phase 1 K=1 online CF | 单卡 8 sample；B0 replay/null L2/L∞ 均 0；correct-null L2 mean/p50/p95 `0.011052/0.011001/0.015738`；correct-shuffle `0.012092/0.011685/0.017690`；两者及 action hash 均 `8/8` 过冻结门槛；paired correct-null overhead `258.95 ms` mean | **有效 SMOKE、分支 A**：future 内容会改变该 checkpoint 的动作，但 action cosine 仍约 `0.9997`、单 task/无 rollout；不能写 success/OOD。模型加载峰值 `23,679.51 MiB`，policy 峰值 `13,009.92 MiB` |
| 阶段三 Phase 2 full 28/4 | A0/A1 各 200×28 training objectives；12/12 hard checks、32/32 manifest descriptors、8/8 checkpoint provenance；双卡 tracks 约 69 分钟 | **VALID OFFLINE NEGATIVE**：A0 `+1.845%`、A1 `−1.712%`；A1 final 比 A0 高 `3.624%`，4/4 dev sample 更差；`phase3_unlocked=false` |
| Thought4 formal v6 readability | Video geometry error `0.032814` vs mean/shuffle `0.061369/0.067243`；Action current geometry `0.021851` vs `0.061369/0.077118`；Action SE(3) `0.105583` vs `0.197027/0.225015` | 三组冻结 linear probe 均越过 5% controls；支持 Clean geometry/motion 可读，不等于策略正确使用 |
| Thought4 formal v6 Camera gap | Video Camera−Clean RMSE `+0.020273 m`、Lighting `+0.011660 m`；rank-3 coordinate Camera−Lighting `0.146284`，95% CI `[0.088519,0.200310]` | 冻结规则支持 Camera-specific equivariance gap；Robot-init 非 exact-state 且 distinct pattern=false |
| Thought4 formal v6 intervention | 12 test states×3 action seeds；36/36 correct bitwise、36/36 shuffle 超 replay floor；action L2 mean `0.000768`；backbone SHA 前后相同 | probe-defined geometry subspace 对动作有技术因果影响；未执行 action rollout，不能写 success/OOD improvement |

20-step pilot 的 episode-weighted Clean→OOD 描述值为：latent L1
`0.1512→0.2002`、cosine distance `0.1168→0.1942`、motion-direction cosine
`0.7697→0.5283`。它只覆盖 2 个 Clean 与 3 个 camera/easy OOD episode，
严格配对只有 1 对，且 static 只有 7 条 null candidate、尚未冻结；因此只能登记为“值得正式检验的
OOD 一致性下降假设”，不能进入论文结论表。

## 6. 分阶段文档

- 论文正文、证据链与复现：[manuscript.md](../paper/manuscript.md)、
  [evidence_chain.md](../paper/evidence_chain.md)、
  [reproducibility.md](../paper/reproducibility.md)
- 阶段一报告与完成度：[thought1_report.md](../thought1/report.md)、[thought1_readiness.md](../thought1/readiness.md)
- 阶段一执行手册：[thought1_execution_guide.md](../thought1/execution_guide.md)
- 阶段二概念与上游审计：[thought2_concepts.md](../thought2/concepts.md)、[thought2_upstream_audit.md](../thought2/upstream_audit.md)
- 阶段二执行与标注手册：[thought2_execution_guide.md](../thought2/execution_guide.md)
- 阶段二盲审与 outcome-blind 抽样：[thought2_blind_review_and_sampling.md](../thought2/blind_review_and_sampling.md)
- 阶段二统计分析计划（当前 DRAFT）：[thought2_statistical_analysis_plan.md](../thought2/statistical_analysis_plan.md)
- 阶段二五类正式结果：[thought2_formal_results.md](../thought2/formal_results.md)
- 阶段二 static/no-op 校准手册：[thought2_static_calibration.md](../thought2/static_calibration.md)
- 阶段三审计与设计：[thought3_upstream_audit.md](../thought3/foundations/upstream_audit.md)、[thought3_design.md](../thought3/foundations/design.md)
- 阶段三 Phase B 验收：[thought3_phase_b_report.md](../thought3/phase_b_d/phase_b_report.md)
- 阶段三 Phase C 验收：[thought3_phase_c_report.md](../thought3/phase_b_d/phase_c_report.md)
- 阶段三 Phase D 验收：[thought3_phase_d_report.md](../thought3/phase_b_d/phase_d_report.md)
- 阶段三 Phase E 验收/失败诊断：[thought3_phase_e_report.md](../thought3/gate_e/phase_e_report.md)
- 阶段三 Gate E.1 单样本诊断预注册：
  [thought3_phase_e1_protocol.md](../thought3/gate_e/phase_e1_protocol.md)
- 阶段三 Gate E.1 单样本诊断结果：
  [thought3_phase_e1_report.md](../thought3/gate_e/phase_e1_report.md)
- 阶段三 Gate E.2 八样本 LR/尺度诊断预注册：
  [thought3_phase_e2_protocol.md](../thought3/gate_e/phase_e2_protocol.md)
- 阶段三 Gate E.2 八样本诊断结果：
  [thought3_phase_e2_report.md](../thought3/gate_e/phase_e2_report.md)
- 阶段三 Gate E.3 held-out multi-flow 预注册：
  [thought3_phase_e3_protocol.md](../thought3/gate_e/phase_e3_protocol.md)
- 阶段三 Gate E.3 v1 无效运行报告：
  [thought3_phase_e3_v1_failure_report.md](../thought3/gate_e/phase_e3_v1_failure_report.md)
- 阶段三 Gate E.3 v2 修复版预注册：
  [thought3_phase_e3_v2_protocol.md](../thought3/gate_e/phase_e3_v2_protocol.md)
- 阶段三 Gate E.3 v2 有效负结果：
  [thought3_phase_e3_v2_report.md](../thought3/gate_e/phase_e3_v2_report.md)
- 阶段三 Gate E.4 paired diversified train-flow 预注册：
  [thought3_phase_e4_protocol.md](../thought3/gate_e/phase_e4_protocol.md)
- 阶段三 Gate E.4 有效负结果：
  [thought3_phase_e4_report.md](../thought3/gate_e/phase_e4_report.md)
- 阶段三 Gate E.5 full-cohort objective aggregation 预注册：
  [thought3_phase_e5_protocol.md](../thought3/gate_e/phase_e5_protocol.md)
- 阶段三 Gate E.5 有效负结果：
  [thought3_phase_e5_report.md](../thought3/gate_e/phase_e5_report.md)
- 阶段三 Gate E.6 fresh-cohort 序贯复验协议与有效负结果：
  [thought3_phase_e6_protocol.md](../thought3/gate_e/phase_e6_protocol.md)、
  [thought3_phase_e6_report.md](../thought3/gate_e/phase_e6_report.md)
- 阶段三 Gate E.7 只读 checkpoint-trajectory 协议与结果：
  [thought3_phase_e7_protocol.md](../thought3/gate_e/phase_e7_protocol.md)、
  [thought3_phase_e7_report.md](../thought3/gate_e/phase_e7_report.md)
- 阶段三 Gate E.8 A0 flow-variance replication 协议与结果：
  [thought3_phase_e8_protocol.md](../thought3/gate_e/phase_e8_protocol.md)、
  [thought3_phase_e8_report.md](../thought3/gate_e/phase_e8_report.md)
- 阶段三 Gate E.9a-v1 失败证据与 v2 matched sample-tail mitigation：
  [thought3_phase_e9_v1_failure_report.md](../thought3/gate_e/phase_e9_v1_failure_report.md)、
  [thought3_phase_e9_v2_protocol.md](../thought3/gate_e/phase_e9_v2_protocol.md)、
  [thought3_phase_e9_v2_report.md](../thought3/gate_e/phase_e9_v2_report.md)
- 阶段三 Phase 0 E9a-v2.1 只读审计协议与结果：
  [thought3_phase_e9_v2_1_readonly_audit_protocol.md](../thought3/gate_e/phase_e9_v2_1_readonly_audit_protocol.md)、
  [thought3_phase_e9_v2_1_readonly_audit_report.md](../thought3/gate_e/phase_e9_v2_1_readonly_audit_report.md)
- 阶段三 Phase 1 K=1 在线动作反事实协议与结果：
  [thought3_phase1_k1_online_counterfactual_protocol.md](../thought3/phase1_action/protocol.md)、
  [thought3_phase1_k1_online_counterfactual_report.md](../thought3/phase1_action/report.md)
- 阶段三 Phase 2 完整 28/4 A0/A1 单配方协议与结果：
  [thought3_phase2_full_28_4_protocol.md](../thought3/phase2_adapter/protocol.md)、
  [thought3_phase2_full_28_4_report.md](../thought3/phase2_adapter/report.md)
- Thought4 冻结协议、FP32 预注册与正式结果：
  [thought4_protocol.md](../thought4/protocol.md)、
  [thought4_fp32_preregistration.md](../thought4/fp32_subspace_v6_preregistration.md)、
  [thought4_formal_v6_results.md](../thought4/formal_v6_results.md)
- 阶段三加速路线、Phase 2/3 草案与硬停止规则：
  [thought3_accelerated_roadmap.md](../thought3/foundations/accelerated_roadmap.md)
- 阶段三数据/训练/评测：[thought3_data_protocol.md](../thought3/foundations/data_protocol.md)、[thought3_training.md](../thought3/foundations/training.md)、[thought3_evaluation.md](../thought3/foundations/evaluation.md)
- 阶段三分析 DRAFT 与限制：[thought3_analysis_protocol_DRAFT.md](../thought3/foundations/analysis_protocol_DRAFT.md)、[thought3_limitations.md](../thought3/foundations/limitations.md)
- 阶段三旧版路线摘要：[thought3_adapter_plan.md](../thought3/foundations/adapter_plan_legacy.md)
- 实验、失败尝试和结论台账：[experiment_ledger.md](experiment_ledger.md)
- 工程难点与简历素材：[engineering_highlights.md](engineering_highlights.md)

## 7. 当前优先级

1. Phase 0 已完成：E9a-v2 由 telemetry-invalid 恢复为
   `engineering valid + scientific failed`；原 77 文件未改，E9b 继续锁定。
2. Phase 1 已完成并按冻结规则进入 A：
   `future_content_sensitivity_observed`。B0 replay、formal-null parity、
   no-cache/no-future-RGB 和 frozen SHA 全部通过。
3. Phase 2 已有效完成，但 development direction 未观察到：A1 比 A0 高
   `3.624%` 且 4/4 sample 更差，分类为
   `training_valid_dev_direction_not_observed`。
4. 按冻结规则停止在 Phase 3 之前：不做完整-checkpoint sensitivity recheck、
   OOD pilot、A2/A4、checkpoint/LR/weight/K 调参，也不消费 E9b reserve。
5. 负结果、Phase 1→Phase 2 证据链、论文图表与完整论文草稿已整理；若未来
   提出新方法，必须作为新的、独立预注册路线，不能覆盖本结果。
6. Thought4 formal v6 已完成并冻结 `camera_equivariance_gap`；下一项新研究只允许
   `Geo-REPA + relative pose / camera-ray equivariance`，先 held-out representation/
   SE(3)，再预注册 Camera rollout。当前不得写成方法已有效。
7. 当前不得覆盖 E.5–E.9 Run ID、事后放宽门槛、根据动作差异挑 checkpoint，
   或把 engineering smoke 写成 success/OOD 证据。
8. 在不按自动 metric/outcome 挑案例的前提下，冻结正式 human-review budget、
   seed 和每 job 至多一个 probe；至少两名 reviewer 独立标注并保留 agreement/
   adjudication。
9. 人工 endpoint 只回答 goal progress、physical plausibility、局部 action
   execution 和 future–actual agreement；不把它升级成 future-to-action 因果。
10. 阶段三 Phase F/G OOD 草案未解锁并保持 archived；不得用阶段一既有 OOD
   outcome 事后选择 Adapter、K 或 checkpoint。
