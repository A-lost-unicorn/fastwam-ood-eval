# 思考点二正式结果：未来—实际变化一致性

更新日期：2026-07-27

## 1. 一句话结论

在冻结官方 Fast-WAM checkpoint、保持控制动作不变的 732-episode
五类扰动 cohort 中，unconditional future 与实际局部视觉变化的自动一致性
在 OOD 下稳定下降，并与失败相关；但局部一致性既不是任务成功的充分条件，
也不是必要条件。由于 future 是 control-loop 外的 shadow observer，这些结果
是关联证据，不能解释为“future error 导致失败”或“动作读取了未来”。

## 2. 数据与证据资格

| 项目 | 结果 |
| --- | ---: |
| Run | `P2A-FIVE-CATEGORY-COLLECTION-v1` |
| 时间 | 2026-07-26 18:18:15—2026-07-27 00:41:46 |
| Cohort | 200 Clean + 532 OOD = 732 episodes |
| Probes / aligned future frames | 1,010 / 2,020 |
| Completed / error probes | 1,010 / 0 |
| Current / predicted / actual / comparison media | 各 1,010；全量解码 0 error |
| Checkpoint SHA-256 | `1000437c...9579` |
| Protocol fingerprint | `3be4c456...32b75` |
| Static threshold | `0.0167421166`，200/200 null jobs 通过 freeze gate |
| 自动统计 | 40 task 等权；suite-stratified task bootstrap 10,000 次 |

正式数据收集使用 clean commit `0fb8350`，项目与三个上游均为 clean。
八份 200/532 cohort 在生成正式 future 指标前 exact-ratify，保留原 job ID；
阶段一 outcome 在 ratification 时已经存在，因此不能写成“阶段一 outcome 前
预注册”。

统计分析实现遵循运行前 DRAFT 中已经写明的 episode→task 聚合、主要 endpoint
与 task bootstrap，但该 DRAFT 没有在正式指标生成前冻结。准确证据标签是：

```text
formal data collection
+ protocol-consistent post-run analysis
+ not preregistered confirmatory analysis
```

机器权威报告：
`outputs/thought2/five_category_formal_v1/formal_analysis_v1/report.md`。

## 3. 预测未来是否正确？

“正确”必须拆成两个问题：

1. **是否像随后实际发生的变化**：自动 VAE proxy 可以回答。
2. **是否朝任务目标正确进展、物理合理**：尚需标签盲化人工评审。

自动结果按 probe→episode→task 聚合，40 task 等权：

| Metric | Clean | OOD | OOD−Clean | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Future cosine distance ↓ | 0.1025 | 0.1341 | +0.0316 | [0.0254, 0.0381] |
| Future latent L1 ↓ | 0.1431 | 0.1708 | +0.0277 | [0.0238, 0.0317] |

因此可以写“**OOD 下 future–realized consistency proxy 变差**”。不能把
`1−cosine` 或 L1 改写成语义“预测正确率”；这些 latent 是预测/实际解码帧
分别经冻结 VAE 重编码的近似表示，不是原生 temporal future latent。

## 4. 动作与未来方向是否一致？

7-DoF action delta 与视觉表示不在同一向量空间，本阶段没有构造直接
action-future cosine。当前实际测量的是：

```text
预测视觉变化方向
    vs
受保护原动作执行后的实际视觉变化方向
```

| Metric | Clean | OOD | OOD−Clean | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Motion-direction cosine ↑ | 0.7416 | 0.5518 | −0.1898 | [−0.2134, −0.1664] |

200-job no-op 校准得到阈值 `0.0167421166`。1,010 个 probe 中没有
predicted-static；6 个 OOD failure 的第二 probe 为 actual-static，它们不进入
decisive direction 分母。所有 732 episode 仍至少有一个有效方向 probe。

## 5. 失败与 future inconsistency 的关系

成功 episode 只有一个 probe，失败 episode 有两个，因此必须同时报告全部
probe 和仅首 probe。另有两个 `libero_10/light_conditions` episode 的
Phase 1/2 outcome 不一致；outcome 分析排除它们，保留 255 success +
275 failure，40/40 task 同时包含两种 outcome。

| OOD task-equal contrast | 全部可用 probes | 仅首 probe |
| --- | ---: | ---: |
| Cosine distance：failure−success | +0.0249 [0.0166, 0.0328] | +0.0197 [0.0116, 0.0282] |
| Direction cosine：failure−success | −0.2127 [−0.2328, −0.1923] | −0.0784 [−0.1046, −0.0541] |

关联在只看首 probe 后仍存在，因此不完全是第二 probe/轨迹截断造成的。
但一致性不能单独解释成败：

- 首 probe cosine 最低误差四分位仍有 `55/132=41.67%` failure；
- 最高误差四分位为 `87/133=65.41%` failure；
- 存在高 future error 但成功的案例；
- 存在 direction cosine `0.8700` 但最终失败的案例。

所以当前最强结论是：

> OOD 是一个共同压力源：它同时降低控制成功率和 future–realized
> consistency；两者相关，但短时 future consistency 不是任务成功的充分或
> 必要条件。

在基础 Fast-WAM 中，shadow future 不反馈给动作分支，所以它不可能是这次
执行失败的直接原因。失败究竟是 wrong goal、wrong object、动作选择、碰撞、
停滞还是长时序恢复不足，仍需要 blind human labels；显式 future 是否能减少
这些失败，则由阶段三 Adapter 因果对照回答。

## 6. 五类扰动

| Category | OOD n | OOD cosine | OOD−Clean / 95% CI | OOD direction |
| --- | ---: | ---: | ---: | ---: |
| Camera viewpoints | 104 | 0.1561 | +0.0536 [0.0453, 0.0620] | 0.4391 |
| Robot initial states | 120 | 0.1451 | +0.0427 [0.0352, 0.0505] | 0.5463 |
| Background textures | 103 | 0.1263 | +0.0238 [0.0150, 0.0332] | 0.5033 |
| Object layout | 110 | 0.1216 | +0.0191 [0.0110, 0.0274] | 0.6287 |
| Light conditions | 95 | 0.1199 | +0.0175 [0.0082, 0.0270] | 0.6454 |

Camera 对 consistency 的破坏最大，light 最小，与阶段一 robustness 大方向一致。
但 severity 不对所有 future 指标单调：camera/background 的 cosine 没有严格
easy→medium→hard 恶化，而 robot-initial-state 呈单调上升。因此不能把
任务难度等级直接当作 future error 的连续强度。

## 7. 动作隔离、重跑稳定性与资源

- 同一次 Phase 2 rerun 内：`1,010/1,010` probe 的 action hash
  before/after 完全一致，证明 shadow observer 没有换掉即将执行的动作。
- 与 Phase 1 历史 trace 跨运行比较：`996/1,010` probe 的 10-step action
  逐元素一致；13 个 probe 有数值差异，1 个因 Phase 1 source 已提前结束而无
  完整对照，最大绝对差 2.0。
- Phase 1/2 episode outcome：`730/732` 一致。

跨运行差异说明仿真/GPU rerun 不是逐位完全确定，不否定同一次 Phase 2 内的
动作保护；这也是 outcome 分析使用 Phase 2 同轨迹结果的原因。

20-step shadow future generation 的 mean/p50/p95 是
`3,354.66/3,316.96/3,564.12 ms`；完整离线诊断为
`5,816.77/5,762.95/6,271.52 ms`；峰值 `24,841.09 MB`。
它们不是阶段一动作延迟，也不能替代阶段三 K=1/2/4 无 RGB 解码的在线延迟。

## 8. 论文与简历表述

论文可写：

> Across 40 task clusters and 732 episodes, OOD perturbations increased the
> decoded-frame VAE future consistency distance by 0.0316
> (95% task-bootstrap CI 0.0254–0.0381) and reduced predicted-versus-realized
> motion-direction cosine by 0.1898 (95% CI 0.1664–0.2134). The association
> with failure remained under a first-probe-only sensitivity analysis.

简历可写：

> 搭建 Fast-WAM/LIBERO-Plus 多 GPU future-consistency 评测管线，完成
> 732 episodes、1,010 probes、4,040 媒体全量审计与 task-cluster bootstrap；
> 发现 OOD 下 future consistency distance 增加 0.0316，方向一致性下降
> 0.1898，并通过首 probe 敏感性分析识别轨迹截断混杂。

两种表述都必须同时保留：`unconditional future`、自动 proxy、非因果、
统计计划未预冻结。不得写“future error 导致失败”或“Fast-WAM 动作使用了
future imagination”。

## 9. 可复现命令

```bash
source scripts/activate_env.sh
fastwam-ood analyze-thought2-formal \
  --experiment-dir outputs/thought2/five_category_formal_v1 \
  --thought1-summary outputs/thought1/fastwam/combined/summary/episode_results.csv \
  --source-trace-root outputs/thought1/fastwam \
  --output-dir outputs/thought2/five_category_formal_v1/formal_analysis_v1 \
  --bootstrap-replicates 10000 \
  --bootstrap-seed 20260725 \
  --verify-media
```

输出目录要求全新；命令不会覆盖或改写阶段一、阶段二 raw JSONL/CSV/video。
