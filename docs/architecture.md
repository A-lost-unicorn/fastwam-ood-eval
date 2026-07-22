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

策略身份不是从视频保存选项推测，而是由配置显式声明并与上游 task/checkpoint 文件名交叉校验：

```text
fastwam  + *_uncond_* → 当前帧动作推理，无测试时未来想象
joint_wam + *_joint_* → 联合未来视频/action latent 去噪
idm      + *_idm_*    → 先预测未来视频，再反推动作
```

聚合器先在每个策略内部计算 Clean→OOD drop，再在同一 `comparison_group` 中对 future/no-future 结果做 episode 配对。多策略混合时不输出容易误读的全局 Clean→OOD drop。
