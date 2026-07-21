# Architecture

配置经过 dataclass 验证后，planner 把 `(suite, base task, condition, perturbation, level, episode)` 展开成确定性的 `job_manifest.jsonl`。Clean job 指向原版 task ID；OOD job 查询 LIBERO-Plus 官方 classification，保存变体 ID、名称和原始 difficulty。

`evaluate` 或每个 `torchrun` rank 按 job ID 稳定分片。worker 只加载一次 Fast-WAM policy，逐 job 重建环境并立即 append+fsync 一行结果。崩溃最多破坏最后一行；resume 会跳过所有已有完整 job ID。

真实调用链：

```text
CLI → config → job manifest → rank shard → backend package selection
    → FastWAMAdapter (official loader + official inference helper)
    → LiberoAdapter / LiberoPlusAdapter → episode result JSONL
    → aggregate → CSV + metrics.json → report/review HTML
```

原版与 Plus 不能在同一 Python 进程切换。一个实验配置只能选择一个 backend；`aggregate --input-dir CLEAN --input-dir OOD` 可把两个实验的 worker JSONL 合并后做配对分析。
