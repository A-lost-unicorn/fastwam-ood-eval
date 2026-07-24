# 阶段二盲审与 outcome-blind 抽样手册

更新日期：2026-07-23

## 1. 本手册解决什么问题

阶段二有两个容易污染结论的入口：

1. 看过 `success/failure`、扰动类别或自动指标后再挑视频，容易只保留符合预期的案例。
2. 用含有 condition、outcome、job ID 的页面做“盲审”，实际上没有隐藏标签。

因此本项目把两个动作分开：

- **抽样盲化**：正式 cohort 在读取 episode outcome 前，仅用规划期 job
  metadata 冻结。
- **标注盲化**：把媒体复制到公开 packet，用不透明 `case_XXXX` 替换所有
  source identifier；condition、outcome、metric 和映射只写入另一个私有目录。

这里的“盲审”准确含义是**标签盲化**，不是视觉上不可识别。相机视角、光照或
明显的任务成败可能从视频本身被 reviewer 推断出来，论文中必须如实说明这个
限制。

## 2. 为什么不能复用阶段一 failure review

阶段一 `review-failures` 页面只面向已知失败案例，并会展示 perturbation、
seed 和 termination。它适合故障排查，不适合比较 Clean/OOD、success/failure
下的未来一致性。

阶段二第一轮盲审只保留：

- 任务文本；
- 当前双相机帧；
- 完整预测未来；
- 动作执行后的实际对齐画面；
- 左预测、右实际的并排视频。

公开 packet 不包含 condition、perturbation、outcome、自动指标、动作数值、
seed、source path、experiment/job/diagnostic ID。第一轮也不填写
`primary_failure_hypothesis`，避免 reviewer 被“这是失败案例”的问题设定暗示。

## 3. 生成和校验公开/私有双目录

以下命令会合并当前 20-step Clean/OOD pilot 的 7 个 probe：

```bash
fastwam-ood prepare-blind-review \
  --packet-dir outputs/thought2_future_blind_pilot_packet \
  --key-dir outputs/thought2_future_blind_pilot_key \
  --input-dir outputs/thought2_unconditional_clean \
  --input-dir outputs/thought2_unconditional_ood \
  --seed 20260723
```

两个目标目录和所有 source 目录必须互不包含，而且必须是全新路径。程序会复制
并逐文件校验 SHA-256；私有 key 文件权限设为 `0600`，key 目录设为 `0700`。

公开包可单独校验：

```bash
fastwam-ood validate-blind-review \
  --packet-dir outputs/thought2_future_blind_pilot_packet
```

由不参与第一轮标注的人同时校验映射：

```bash
fastwam-ood validate-blind-review \
  --packet-dir outputs/thought2_future_blind_pilot_packet \
  --key-dir outputs/thought2_future_blind_pilot_key
```

不要把 `thought2_future_blind_pilot_key/` 发给 reviewer。公开
`index.html` 可离线打开，草稿保存在浏览器 localStorage，并可导出 JSON/CSV。
版本化结果时应保存导出文件，不能只依赖浏览器缓存。

正式人工子集建议预先固定一个 probe/episode，例如：

```bash
fastwam-ood prepare-blind-review \
  --packet-dir <fresh-public-dir> \
  --key-dir <fresh-private-dir> \
  --input-dir <all-formal-diagnostic-dirs> \
  --seed <pre-registered-review-seed> \
  --max-cases-per-job 1 \
  --max-cases <pre-registered-human-review-budget>
```

启用 per-job cap 后，程序先用 source-job 级 hash 排序，再在每个 job 内选择
probe；全局 case 上限不会让拥有两个 probe 的长 episode 比一个 probe 的 episode
更容易入选。case 预算必须在看标签或自动指标前确定。

## 4. 第一轮与第二轮标注

第一轮使用
[future_blind_annotations.csv](templates/future_blind_annotations.csv)，允许值由
packet manifest 固定：

| 字段 | 允许值 |
| --- | --- |
| `video_validity` | `valid/corrupt/unaligned/uncertain` |
| `future_goal_progress` | `correct/partial/wrong_object/wrong_direction/static/uncertain` |
| `future_physical_plausibility` | `plausible/minor_artifact/unphysical/uncertain` |
| `future_actual_agreement` | `aligned/partial/conflict/static/uncertain` |
| `action_execution_quality` | `realized/stalled/collision/oscillation/uncertain` |
| `confidence` | `low/medium/high` |

`future_actual_agreement` 只能比较精确对齐的 offset `0/4/8`（0–0.4 s）；
predicted 视频余下帧没有实际对照。`action_execution_quality` 也只表示这段可见
窗口中的局部执行现象，不能据此区分 action selection 与低层 controller 故障。

正式主表子集至少由两名 reviewer 独立完成。原始导出建议命名为
`annotations_blind_<reviewer>.csv`；在解盲前计算逐字段 agreement 或 Cohen's
kappa，并保留缺失/`uncertain` 的分母。

第二轮才由保管 private key 的研究者解盲，并使用
[future_case_annotations.csv](templates/future_case_annotations.csv) 填写
`primary_failure_hypothesis`。讨论后的 adjudicated 文件必须另存；不得覆盖两份
原始盲审标签。

### 4.1 解盲前的导入校验与一致性统计

两名 reviewer 导出后，先在**不提供 private key**的进程中运行：

```bash
fastwam-ood analyze-blind-review \
  --packet-dir <public-packet-dir> \
  --annotation annotations_blind_reviewer_a.csv \
  --annotation annotations_blind_reviewer_b.csv \
  --output-dir <fresh-blind-agreement-dir>
```

CSV 和页面导出的 JSON 都受支持。程序会拒绝：

- annotation schema、packet ID 或 case set 不匹配；
- 同一 case 重复、缺失或出现 packet 外 case；
- reviewer 为空、一个文件混入多人、两个文件使用相同 reviewer ID；
- `review_round` 不是 `blind`；
- 任一字段出现未注册枚举值；
- 覆盖已有 output，或把分析写回 public packet。

输出为：

```text
analysis_manifest.json
normalized_annotations.csv
reviewer_completion.csv
pairwise_agreement.csv
agreement_summary.json
agreement_report.md
```

`analysis_manifest.json` 固定 public manifest、两份原始 annotation 和五个派生
文件的 SHA-256，并声明 `source_files_rewritten=false`、
`private_key_read=false`、condition/outcome/metric fields 均未读取。复核命令：

```bash
fastwam-ood validate-blind-review-analysis \
  --analysis-dir <blind-agreement-dir>
```

每个字段同时报告两套分母：

1. `nonmissing`：任一 reviewer 缺失则排除，但把 `uncertain` 保留为真实类别；
2. `decisive`：进一步排除任一 reviewer 为 `uncertain` 的 pair。

报告 exact agreement 与 nominal、unweighted pairwise Cohen's κ。若两人的边际
分布完全退化，例如所有 case 都选 `plausible`，即使 agreement=1，κ 也必须是
`undefined/degenerate_marginals`，不能手工写成 1。超过两名 reviewer 时给出所有
两两结果及 defined κ 的 macro mean；它不是 Fleiss' κ。

## 5. 2026-07-23 真实 packet 演练

已经生成并做完整解码审计的 packet：

| 项目 | 值 |
| --- | --- |
| packet ID | `16a1dbc38c93c5367e665aef` |
| public manifest SHA-256 | `273c4b67b8a642c4b724289c6c56854952322c0ebb90d57bc11b74f942587b7f` |
| cases / media | 7 / 28 |
| sensitive public keys | 0 |
| private key match | 通过 |

解码审计覆盖全部 28 个媒体：7 张 current PNG；每个 predicted MP4 为
9 帧 `224×448×3`，每个 actual MP4 为 3 帧 `224×448×3`，每个 comparison
MP4 为 3 帧 `224×896×3`。

这只是**工作流 PILOT**。当前 `annotations.csv` 仍是空模板，尚无人为盲审，
因此也没有真实 agreement report；不能把 7 条写成“人工未来质量结果”。

## 6. Outcome-blind 正式 cohort 的固定规则

planner 只读取阶段一 `job_manifest.jsonl`。它允许使用：

- suite、base task、condition、category、difficulty；
- episode index；
- 规划期 `skip_reason`，仅用于排除上游没有可运行 variant 的空分层；
- source job ID 及 source job-manifest SHA-256。

它明确不读取 `success`、`termination_reason`、latency、自动指标，也不打开
worker/summary 的 `episode_results.jsonl`。每个候选的排序键是
`SHA256(seed, source-job-manifest-hash, job-id)`，因此相同 source 和协议可精确
重放。

当前 v2 设计为：

- seed：`20260724`；
- Clean：每个 suite×task 抽 5 个 job，并强制包含
  `episode_index=0`，合计 `4×10×5=200`；
- OOD：每个可运行 suite×task×category×difficulty cell 抽 1 个官方 variant，
  合计 532；
- 所有 OOD job 都是 `episode_index=0`，Clean anchor 因而提供相同
  suite/task/index 的基准参考；
- 68 个 skipped-only cell 保持 unsupported 审计记录，不进入分母；
- supported cell 的 shortfall 为 0。

| Suite | Clean | OOD | Unsupported cell |
| --- | ---: | ---: | ---: |
| `libero_spatial` | 50 | 126 | 24 |
| `libero_object` | 50 | 137 | 13 |
| `libero_goal` | 50 | 139 | 11 |
| `libero_10` | 50 | 130 | 20 |
| **合计** | **200** | **532** | **68** |

OOD 532 条按类别为：background 103、camera 104、light 95、object-layout
110、robot-initial-state 120。

这份设计覆盖了当前阶段一计划中的五类扰动，尚未替用户把“布局或机器人初态”
二选一。若主文坚持四类：

- 保留 object-layout、排除 robot-init：总队列为 200 + 412 = 612；
- 保留 robot-init、排除 object-layout：总队列为 200 + 422 = 622。

这两种都必须用 category filter 重新生成全新的 manifest/ID，不能在 732 条
结果出来后再按效果选择类别。当前 732 也只是覆盖性设计，尚未做 statistical
power justification；冻结前仍可基于研究问题和算力缩小，但不能基于 outcome
方向缩小。

## 7. 当前 v2 草案及其资格

八份草案位于：

```text
outputs/thought2_outcome_blind_cohort_draft_v2/<suite>/clean.json
outputs/thought2_outcome_blind_cohort_draft_v2/<suite>/ood.json
```

它们全部可精确重放、`short_strata=0`，但状态均为：

```text
status = draft_not_frozen
frozen = false
```

原因是 planner 所在项目 tree 当前有尚未提交的实现和文档修改。旧目录
`thought2_outcome_blind_cohort_draft/` 是已废弃 v1：它没有强制 Clean
`episode_index=0` anchor，不能用于正式配对；保留该目录只为记录失败尝试。

v2 的 cohort ID：

| Suite | Clean | OOD |
| --- | --- | --- |
| spatial | `31fc87a1...c96727` | `61d05bc1...647047` |
| object | `8425b901...5c372` | `dbe87b36...848566` |
| goal | `7e1531fb...a2c7a` | `602cb486...6d65c4f` |
| libero_10 | `a79d716f...4a981` | `85d7e0a8...ec30b` |

缩写只用于阅读；机器分析必须使用 manifest 内的完整 ID 和 hash。

## 8. 如何真正冻结

冻结必须发生在**项目代码提交且 tree clean之后、任何阶段一正式 outcome JSONL
出现之前**。planner 会同时检查：

- 当前项目 tree 显式 `git_dirty=false`；
- source formal 目录不存在非空 episode-result JSONL；
- supported strata 没有 shortfall；
- 使用全新 output path，绝不覆盖 draft。

以 spatial Clean 为例：

```bash
fastwam-ood plan-diagnostic-cohort \
  --source-dir outputs/thought1/fastwam/libero_spatial/clean \
  --output outputs/thought2_outcome_blind_cohort_frozen/libero_spatial/clean.json \
  --seed 20260724 \
  --per-stratum 5 \
  --stratum-field suite \
  --stratum-field task_id \
  --anchor-episode-index 0 \
  --freeze
```

spatial OOD：

```bash
fastwam-ood plan-diagnostic-cohort \
  --source-dir outputs/thought1/fastwam/libero_spatial/ood \
  --output outputs/thought2_outcome_blind_cohort_frozen/libero_spatial/ood.json \
  --seed 20260724 \
  --per-stratum 1 \
  --stratum-field suite \
  --stratum-field task_id \
  --stratum-field condition \
  --stratum-field perturbation_category \
  --stratum-field perturbation_level \
  --category camera_viewpoints \
  --category light_conditions \
  --category background_textures \
  --category robot_initial_states \
  --category objects_layout \
  --level easy --level medium --level hard \
  --freeze
```

四个 suite 要用相同 selection rule 各自生成 Clean/OOD，并逐个运行：

```bash
fastwam-ood validate-diagnostic-cohort \
  --manifest <frozen-manifest.json> \
  --source-dir <matching-thought1-source-dir>
```

正式 diagnostic 配置必须同时写：

```yaml
diagnostics:
  cohort_manifest_path: outputs/thought2_outcome_blind_cohort_frozen/<suite>/<condition>.json
  require_frozen_cohort: true
```

安全门会在加载模型或 reset 环境前拒绝 `draft_not_frozen`。冻结后再改变 seed、
类别、分层、source job manifest 或 anchor 都会得到新的 cohort identity，必须
作为新协议登记。

## 9. 与阶段一、static calibration 和 matched case-control 的隔离

- outcome-blind manifest 只固定“阶段二对哪些阶段一 job 做 shadow rerun”，不会
  写入阶段一目录。
- static/no-op calibration 使用另一个 200-job null 计划，用于冻结 motion
  threshold；它不属于这 732 条。
- 阶段一完成后可以另建 success/failure matched case-control cohort，用于解释
  失败机制；它必须显式标为 outcome-aware，不能用于总体效应估计，也不能替换
  本 cohort。
- 正式 blind packet 应在 diagnostic rows 完成后一次生成，并将 packet manifest、
  private key、两份 reviewer 原始标签和 adjudication 分开存档。

## 10. 粗略算力预算

阶段二 runner 会重新执行环境，不是只读阶段一 trace。用现有 pilot 的 OOD
completed episode 平均 `71.53 s`，以及两次 20-step probe 约
`2×(7.21–8.20) s` 的完整 diagnostic latency 做线性外推：

```text
732 × (71.53 + 约 15.41) s ≈ 17.7 GPU-hours
```

这只是容量估算，不是 benchmark：正式 cohort 的 suite、成功/失败比例和
episode 长度都不同。理想均衡时约为 3 GPU 上 5.9 小时或 4 GPU 上 4.4 小时的
纯 workload；还需加八个 suite×condition 进程组的模型冷启动、尾部不均衡、I/O
和重试，实际应保留更宽的时间窗口。四类的 612/622 设计按同一粗略模型约为
14.8/15.0 GPU-hours。

当前 7-case public packet 为约 1.4 MiB，且 formal 默认
`save_latents=false`；相较 rollout 时间，当前媒体存储不是主要瓶颈。正式开始前
仍应以 20–50 条跨 suite pilot 重新测吞吐，而不是把上述外推当作完成时承诺。

## 11. 当前下一步

1. 决定正式主分析采用五类 732 条，还是四类 612/622 条。
2. 将当前实现提交，在 clean tree 上按最终类别方案重新生成 `--freeze` manifests。
3. 再启动阶段一正式 rollout；若 outcome JSONL 已经出现，planner 会拒绝补做
   “pre-outcome freeze”认证。
4. 完成并人工冻结 200 条 static/no-op threshold。
5. 使用 `require_frozen_cohort=true` 的 suite-specific 配置运行阶段二正式
   shadow diagnostics，然后生成正式盲审 packet。
