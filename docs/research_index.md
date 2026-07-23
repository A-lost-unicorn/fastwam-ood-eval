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
| 阶段二 A：unconditional future consistency | 已实现；读取阶段一 source manifest，只写独立输出 | 2026-07-23 完成 1 episode / 1 probe / 2-step 真实 GPU smoke | **链路完成，正式分析未完成**。当前单样本数值不能作为论文结果 |
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
- 当前 `main` 保留阶段一评测路径；阶段二只通过显式 `diagnose-future` 命令进入。
- 阶段一配置位于 `configs/eval_*.yaml` 和 `configs/studies/thought1.yaml`。
- 阶段二 A 使用 `configs/studies/thought2_unconditional_*.yaml`。
- 阶段二 B 使用 `configs/studies/thought2_shadow_*.yaml`，当前应在能力门禁处失败。
- 阶段三将使用新的 `configs/studies/thought3_*`、训练配置和 checkpoint namespace。

### 4.2 输出

```text
outputs/thought1/...                 # 阶段一正式结果
outputs/thought2_unconditional_*     # 阶段二 A
outputs/thought2_shadow_*            # 阶段二 B
outputs/thought3/...                 # 阶段三训练、cache、评测
```

阶段二只读阶段一 `experiment_manifest.json` 和 `job_manifest.jsonl`，并验证 checkpoint hash、Fast-WAM commit、控制协议与 source manifest hash。输出目录与 source 目录必须互不包含；程序会拒绝把 diagnostics 写进阶段一目录。

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

阶段二 smoke 的 latent L1、cosine、motion direction 等数值保存在机器报告中，但由于 `n=1`、仅 2 个去噪步、10-step 人为截断且静止阈值未校准，当前不进入论文结论表。

## 6. 分阶段文档

- 阶段一报告与完成度：[thought1_report.md](thought1_report.md)、[thought1_readiness.md](thought1_readiness.md)
- 阶段一执行手册：[thought1_execution_guide.md](thought1_execution_guide.md)
- 阶段二概念与上游审计：[thought2_concepts.md](thought2_concepts.md)、[thought2_upstream_audit.md](thought2_upstream_audit.md)
- 阶段二执行与标注手册：[thought2_execution_guide.md](thought2_execution_guide.md)
- 阶段三 Adapter 方案：[thought3_adapter_plan.md](thought3_adapter_plan.md)
- 实验、失败尝试和结论台账：[experiment_ledger.md](experiment_ledger.md)
- 工程难点与简历素材：[engineering_highlights.md](engineering_highlights.md)

## 7. 当前优先级

1. 决定并启动阶段一正式 800 Clean + 6,771 OOD；阶段二可以并行写代码，但不能替代这批结果。
2. 阶段二先校准静止/no-op 阈值，再运行 20-step Clean/OOD 小 pilot。
3. 盲审阶段二视频，验证自动 latent 指标是否与人工“正确目标进展/方向一致”标签相关。
4. 只有阶段一主表和阶段二正式一致性结果稳定后，冻结 Adapter 输入和缓存 schema，进入阶段三。
