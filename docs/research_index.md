# 研究总控：Fast-WAM 在 OOD 环境中真的不需要未来想象吗？

更新日期：2026-07-23

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
| 阶段一：ID/OOD 评测 | Clean/OOD、3 GPU、resume、聚合均已打通 | Clean 2 条、OOD smoke 4 条、OOD pilot 8 条真实执行；正式 manifest 已生成 | **尚未完成**。缺 800 Clean + 6,771 OOD 正式 rollout，不能报告正式 drop |
| 阶段二 A：unconditional future consistency | 已实现；读取阶段一 source manifest，只写独立输出 | 2-step smoke 1 episode；20-step Clean/OOD pilot 5 episodes / 7 probes / 14 aligned future frames / 0 error | **PILOT 完成，正式分析未完成**。当前趋势只用于形成假设 |
| 阶段二校准：static/no-op null | 独立命令、输出、resume、聚合和 freeze gate 已实现 | 2 Clean + 五类 OOD 共 7/7 eligible；候选阈值 `0.013223` | **PILOT 完成，阈值未冻结**。仅 7/200，且 v1 未预先固定 quantile 插值法 |
| 阶段二盲审 | public packet/private key 分离与泄漏校验已实现 | 真实 pilot 7 cases / 28 media 全量解码；0 sensitive public key | **流程 PILOT 完成，人工标注未开始**。不得写成人工质量结果 |
| 阶段二正式抽样 | outcome-blind planner、anchor 与 formal frozen gate 已实现 | v2 草案 200 Clean + 532 OOD；68 unsupported、0 supported shortfall | **尚未冻结**。当前 tree dirty；五类还是四类也待研究者决定 |
| 阶段二统计协议 | episode→task 分层 estimand、task-cluster bootstrap、missing/uncertain/outcome gate 已写成 DRAFT | 合成双 reviewer 标签验证 agreement 工具；无正式效应估计 | **尚未冻结**。primary metric、bootstrap、human budget 和 outcome gate 待确认 |
| 阶段二 B：action-conditioned future consistency | 严格门禁、schema、runner、测试已实现 | CPU/mock 与门禁测试通过 | **阻塞**。官方 release 为 `action_conditioned=false`，且没有可信匹配 checkpoint |
| 阶段三：Future-to-Action Adapter | 研究与隔离方案已设计 | 无训练或真实评测结果 | **未开始** |

因此，“阶段一已经完成”的准确说法是：**阶段一工程准备和 pilot 已完成，阶段一正式科学实验尚未完成。**

## 3. 证据等级

以后所有表格、简历和论文数字都必须带证据等级：

| 等级 | 含义 | 能否写成论文结果 |
| --- | --- | --- |
| `PLAN` | manifest、doctor、dry-run | 否 |
| `TEST` | 单元/集成测试或 mock | 否 |
| `SMOKE` | 少量真实模型和环境运行，验证链路 | 否 |
| `PILOT` | 小规模真实样本，估算失败模式和成本 | 只能作为预实验，必须显式标注 |
| `FORMAL` | 预注册配置、完整分母、聚合与审计通过 | 可以 |

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
- 阶段三将使用新的 `configs/studies/thought3_*`、训练配置和 checkpoint namespace。

### 4.2 输出

```text
outputs/thought1/...                 # 阶段一正式结果
outputs/thought2_unconditional_*     # 阶段二 A
outputs/thought2_static_calibration_* # 阶段二独立 null 校准
outputs/thought2_outcome_blind_*     # 阶段二只读抽样 manifest
outputs/thought2_future_blind_*       # 阶段二 public packet/private key
outputs/thought2_shadow_*            # 阶段二 B
outputs/thought3/...                 # 阶段三训练、cache、评测
```

阶段二 future diagnostics 只读阶段一 `experiment_manifest.json` 和
`job_manifest.jsonl`，并验证 checkpoint hash、Fast-WAM commit、控制协议与
source manifest hash。Static calibration 另行规划独立 task/seed，只运行标准
no-op，不读取 pilot success/OOD 标签。这些输出目录必须互不包含；程序会在模型
加载前拒绝混写。

正式 outcome-blind manifest 还要求 source outcome JSONL 尚未出现，并以
`require_frozen_cohort=true` 阻止 draft 被误运行。Blind packet 与 private
unblinding key 也必须使用彼此分离、且不位于任一 diagnostic source 内的目录。

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
| 阶段二 A real smoke | 1 episode / 1 probe；2 个 aligned future frames；0 probe error | 真实未来诊断链路可用 |
| 阶段二 A real smoke 资源 | 2-step future generation 1,223.53 ms；完整诊断 4,616.06 ms；峰值 24,841.09 MB | 仅作 smoke 容量证据；不能外推 20-step 延迟 |
| 阶段二 A 20-step pilot | Clean 2 episodes/2 probes；camera-easy OOD 3 episodes/5 probes；合计 14 aligned frames、0 error | 正式去噪步数的诊断链路可用；不是总体样本 |
| 阶段二 A 动作隔离 | 7/7 probe 的执行动作与阶段一 trace 逐元素一致，最大绝对差 0；5/5 episode outcome 一致 | shadow future 没有改变这批基线执行 |
| 阶段二 A pilot 资源 | Clean/OOD episode-weighted generation 4,108.12/4,563.88 ms；完整诊断 7,214.25/8,200.65 ms；峰值均 24,841.09 MB | 20-step 小样本容量与延迟证据 |
| 阶段二 static null pilot | 7/7 eligible、0 error；同帧编码噪声全为 0；8-step no-op energy 中位数/最大值 `0.006615/0.013223` | 证明旧阈值 1.0 数量级错误；不是正式阈值 |
| static 候选敏感性 | 候选 `0.013223` 下 predicted/actual static 均 0/7；旧阈值下均 7/7 | 只读派生重分类；原 diagnostics 未改写，candidate 未冻结 |
| 阶段二 blind packet pilot | packet `16a1dbc...665aef`；7 cases / 28 media；public/private hash 校验通过；human labels 0/7 | 盲审工具链可用；不能评价 future 质量 |
| 阶段二 outcome-blind draft v2 | 200 Clean + 532 OOD = 732；68 unsupported；0 supported shortfall；Clean 强制 episode-0 anchor | 覆盖设计草案；`frozen=false`，不是预注册正式样本 |

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
- 阶段二 static/no-op 校准手册：[thought2_static_calibration.md](thought2_static_calibration.md)
- 阶段三 Adapter 方案：[thought3_adapter_plan.md](thought3_adapter_plan.md)
- 实验、失败尝试和结论台账：[experiment_ledger.md](experiment_ledger.md)
- 工程难点与简历素材：[engineering_highlights.md](engineering_highlights.md)

## 7. 当前优先级

1. 在任何阶段一正式 outcome JSONL 出现前，决定阶段二主分析保留五类
   732 条，还是四类 612/622 条；同时确认 primary metric、bootstrap 与
   outcome/human gate。将当前实现提交并在 clean tree 上用 `--freeze`
   重新生成八份 outcome-blind manifest。
2. 冻结后再决定并启动阶段一正式 800 Clean + 6,771 OOD；阶段二代码不能替代
   这批结果。
3. 将已完成的 7 条 no-op calibration pilot 扩展到预注册的 200 条
   （Clean/OOD 各 100、五类 OOD 各 20）；两份 formal 配置已通过
   `assigned=100, pending=100, skipped=0` 的只读计划审计，待在 clean commit
   上真实运行并人工冻结阈值。
4. 由两名 reviewer 对已生成的 7-case pilot packet 做第一轮流程演练；它不替代
   正式 packet。
5. 只有阶段一主表和阶段二正式一致性结果稳定后，冻结 Adapter 输入和缓存
   schema，进入阶段三。
