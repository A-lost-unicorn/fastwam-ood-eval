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

`experiment_manifest.json` 保存 resolved config、upstream commits、GPU/Python 环境和 job manifest 路径。checkpoint SHA-256 在真实 episode result 中记录；plan 阶段不读取数 GB checkpoint，因此 manifest 仅记录路径。

`metrics.json.future_imagination_comparisons` 保存 future/no-future 的 episode 配对数、discordant outcomes、配对成功率差及 CI、exact McNemar p-value 和因果解释资格。`causal_interpretation_allowed=false` 表示 checkpoint 训练配方未被证明匹配。
