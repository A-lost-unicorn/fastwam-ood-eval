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
- future 生成与完整诊断 latency、显存，以及 L1、cosine、motion-direction、motion-energy 等一致性指标。
- episode success/termination 和 error；诊断失败不能回写或重跑美化阶段一结果。

`diagnostic_manifest.json.status` 随诊断生命周期更新；成功聚合后为
`aggregated`，并记录聚合文件路径。2A 的
`causal_interpretation_allowed=false` 是 schema 约束，不因数值看起来较好而改变。
