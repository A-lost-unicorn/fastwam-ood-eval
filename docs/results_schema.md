# Results schema

每个 worker 的 `episode_results.jsonl` 一行对应一个 job。核心字段包括实验/上游/checkpoint provenance，`policy_variant`、`test_time_future_imagination`、`comparison_group`、`training_recipe_id`，suite/base task、episode index/seed、condition、统一 perturbation/level、官方 variant parameters、success、steps、termination、latency、显存、视频和 error。

`termination_reason`：

- `success`：官方环境成功。
- `max_steps`：达到策略步数上限。
- `exception`：模型、环境、记录或运行错误；完整错误写入该行。
- `skipped`：官方分类没有兼容变体；不进入成功率分母。

聚合输出：

```text
summary/episode_results.jsonl
summary/episode_results.csv
summary/summary_by_task.csv
summary/summary_by_perturbation.csv
summary/summary_by_level.csv
summary/summary_by_policy.csv
summary/failures.csv
summary/metrics.json
summary/report.md
```

`experiment_manifest.json` 保存 resolved config、项目/上游 commit 与
`git_dirty/*_dirty` 状态、GPU/Python 环境和 job manifest 路径。checkpoint
SHA-256 在真实 episode result 中记录；plan 阶段不读取数 GB checkpoint，
因此 manifest 仅记录路径。正式结果要求项目和上游 dirty 状态均为 `false`。
唯一预声明例外是 LIBERO-Plus 的 `.downloads/` 下载缓存；排除前缀也会写入
manifest，任何 tracked diff 都不会被排除。

`metrics.json.future_imagination_comparisons` 保存 future/no-future 的 episode 配对数、discordant outcomes、配对成功率差及 CI、exact McNemar p-value 和因果解释资格。`causal_interpretation_allowed=false` 表示 checkpoint 训练配方未被证明匹配。

## 阶段二诊断结果

阶段二不追加或改写阶段一的 `episode_results.jsonl`。它只读取阶段一
`diagnostics.source_output_dir` 的 manifest/result，并在独立的
`experiment.output_dir` 下写入：

```text
diagnostic_manifest.json
source_manifest.json
checkpoint_hash.json
workers/rank_*/diagnostics.jsonl
workers/rank_*/completed_jobs.jsonl
workers/rank_*/current_frames/*.png
workers/rank_*/predicted_futures/*.mp4
workers/rank_*/actual_futures/*.mp4
workers/rank_*/side_by_side/*.mp4
summary/all_diagnostics.csv
summary/consistency_by_outcome.csv
summary/consistency_by_condition.csv
summary/consistency_by_perturbation.csv
summary/static_future_cases.csv
summary/diagnostic_metrics.json
summary/thought2_report.md
```

每个 probe 行显式记录：

- `mode`：`unconditional_future`（2A）或 `action_conditioned_future`（2B），二者不可混为同一 protocol。
- source experiment/job、checkpoint SHA-256 和 protocol fingerprint；source manifest
  SHA-256 固化在 `diagnostic_manifest.json`。
- current frame、predicted future、actual future、side-by-side 路径。
- 受保护动作的 shape、前后 SHA-256 和 `action_unchanged`；2A 还明确记录该动作未作为视频条件。
- 预测/实际时间戳、frame offset、alignment quality、可计入指标的 future frame 数。
- future 生成与完整诊断 latency、显存，以及 L1、cosine、motion-direction、motion-energy 等一致性指标。聚合器会分别输出
  `generation_*` 与 `diagnostic_*`，避免把 future-only 成本和媒体/编码开销混为一项。
- episode success/termination 和 error；诊断失败不能回写或重跑美化阶段一结果。

`diagnostic_manifest.json.status` 随诊断生命周期更新；成功聚合后为
`aggregated`，并记录聚合文件路径。2A 的
`causal_interpretation_allowed=false` 是 schema 约束，不因数值看起来较好而改变。

Clean/OOD 多输入聚合必须写到新的 comparison 目录。该目录也会生成
`diagnostic_manifest.json`，其中：

- `aggregation_kind=multi_input_comparison`；
- `planned_job_count=0`，真实分母来自各输入 manifest，避免重复计数；
- 保留唯一 mode、共同 checkpoint/Fast-WAM/项目 provenance；
- `comparison_inputs` 记录每个输入的 protocol fingerprint；
- `source_manifest_hashes` 记录所有阶段一 source hash。
- `provenance` 表示输入生成时的共同模型/代码身份，
  `aggregation_provenance` 单独记录生成 comparison 时的项目 commit/dirty 状态。

如果 mode 缺失，报告只能显示 provenance error，不能默认套用
action-conditioned 文案。

### Outcome-blind cohort manifest

正式阶段二不按 outcome 挑选 diagnostic job。独立
`thought2-outcome-blind-diagnostic-cohort-v1` manifest 保存：

```text
schema/kind/cohort_id/status/frozen
source:
  experiment_id
  experiment_manifest_sha256_at_selection
  job_manifest_sha256
  outcome_files_present_at_selection
selection:
  seed/per_stratum/stratum_fields/filters/anchor_episode_indices
  input_job_count/runnable_strata/selected_job_count
  short_strata/unsupported_skipped_only_strata
  outcome_fields_read=false
  episode_result_files_read=false
strata[]
unsupported_strata[]
selected_jobs[]:
  selection_order/job_id/selection_key/stratum
planner_provenance
```

validator 会重新读取 source job manifest，复算 source hash、每个 selection key、
精确顺序、stratum audit 和 cohort identity。`--freeze` 额外要求 source 中还没有
非空 outcome JSONL、项目 tree clean、supported stratum 没有 shortfall。正式
diagnostic config 必须设 `require_frozen_cohort=true`；否则
`draft_not_frozen` 只允许做流程测试。

### Label-blind review packet

公开 packet：

```text
blind_packet_manifest.json
annotations.csv
index.html
media/case_XXXX/current.png
media/case_XXXX/predicted.mp4
media/case_XXXX/actual.mp4
media/case_XXXX/comparison.mp4
```

公开 manifest 只包含 opaque case ID、任务文本、媒体相对路径/hash/bytes 和第一轮
annotation schema。condition、perturbation、outcome、metric、action、seed 与所有
source identifiers 都在独立 private
`unblinding_key.json`；key 同时固定 source manifest/diagnostic JSONL/media hash
以及 case mapping。可选 `max_cases_per_job` 先按 source-job hash 排序并对 probe
做 round-robin cap，避免双 probe episode 在有限人工预算中获得更高入选概率。
validator 会检查敏感 key/token 泄漏、路径逃逸、媒体大小与 SHA-256、case
order 和 public/private identity。

第一轮字段不包含 `primary_failure_hypothesis`；解盲后的失败归因必须保存在另一份
文件。完整流程见
[thought2_blind_review_and_sampling.md](thought2_blind_review_and_sampling.md)。

盲态 reviewer agreement 另写入全新目录，且不读取 private key：

```text
analysis_manifest.json
normalized_annotations.csv
reviewer_completion.csv
pairwise_agreement.csv
agreement_summary.json
agreement_report.md
```

manifest 固定 public packet、原始 CSV/JSON 和全部派生文件 hash，并声明 source
未改写、private key/condition/outcome/metric 均未读取。每个字段分别记录
nonmissing/uncertain/decisive 分母、exact agreement、Cohen's κ 及 κ 状态；
`degenerate_marginals` 必须保留为 undefined。多 reviewer 汇总是 pairwise
macro，不得标成 Fleiss' κ。

正式 episode/task-level estimand、cluster bootstrap、缺失和 multiplicity 规则见
[thought2_statistical_analysis_plan.md](thought2_statistical_analysis_plan.md)；
当前文档仍是未冻结 DRAFT，不能由现有 schema 自动升级为正式结论。

## 阶段二 static/no-op calibration

校准既不读取阶段一结果，也不读取 future diagnostic 的 success/OOD 标签。
`policy.act()` 不会被调用；环境只接收 manifest-pinned 标准 no-op。输出与前两类
namespace 互斥：

```text
calibration_manifest.json
checkpoint_hash.json
static_calibration_job_manifest.jsonl
workers/rank_*/static_calibration_samples.jsonl
workers/rank_*/completed_jobs.jsonl
workers/rank_*/artifacts/<job_id>/offset_000.png
workers/rank_*/artifacts/<job_id>/offset_004.png
workers/rank_*/artifacts/<job_id>/offset_008.png
```

每个 calibration sample 记录：

- job/task/seed/initial-state 与 Clean/OOD category/level；
- `policy_action_sampled=false`、标准 no-op 值和 SHA-256；
- settle/capture 实际步数、运行时 control frequency、model-frame shape；
- `frame_embedding_semantics` 与两组确定性 encoding seed；
- 同帧所有 pairwise energy 及每样本最大值；
- offset-4/8 no-op embedding energy 与像素 MAE；
- `eligible_for_threshold`、exclusion/error、artifact path 和 attempt 时间；
- checkpoint/upstream/project provenance 与 protocol/compatibility fingerprint。

Clean/OOD 聚合只接受相同 `compatibility_fingerprint`，并写到独立 comparison：

```text
summary/static_calibration_samples.csv
summary/static_calibration_summary.json
summary/static_calibration_aggregation_manifest.json
summary/static_calibration_report.md
summary/static_threshold_sensitivity.csv
summary/static_threshold_sensitivity.json
```

候选阈值为同帧噪声与 full-horizon no-op 两个分布的预注册 99% `higher`
分位数较大者。聚合 manifest 同时固定每个 raw job manifest、calibration
JSONL、diagnostic JSONL 和 source manifest 的 SHA-256。自动门禁还要求：
所有 source tree 显式 clean、运行时 control frequency 与 model-frame shape
跨样本一致、每条 job 完整执行预注册 settle/capture，且没有异常或排除样本。
`threshold_status` 只能是：

- `unavailable`：缺少有效 null 分布；
- `candidate_only`：存在数值，但样本/覆盖/异常/预注册门禁至少一项未通过；
- `eligible_for_manual_freeze`：自动门禁全部通过，仍需人工审核，绝不自动回写
  diagnostics。

敏感性文件只读取连续 motion energy，并固定源 manifest 与 JSONL hash；
`source_rows_rewritten=false` 是必需字段。完整口径见
[thought2_static_calibration.md](thought2_static_calibration.md)。
