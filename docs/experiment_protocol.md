# Experiment protocol

1. 固定 Fast-WAM checkpoint、dataset stats、模型配置和推理超参数；聚合器发现 Clean/OOD checkpoint SHA-256 不同会拒绝比较。
2. 为 `(base seed, suite, base task, episode index)` 生成相同 seed；condition/category/level 不进入 seed 公式。
3. 先运行 Clean，再运行每个官方 Plus 类别和难度；Plus variant 的原始 classification metadata 保存在 job/result。
4. 每个 job 最多执行 suite 配置的 policy steps，之前执行与官方 evaluator 相同的 30 次 no-op。
5. success 只使用环境官方 done/check-success；异常不被悄悄忽略，skipped 也不进入成功率分母。
6. success rate 分母包含正常完成和 exception job（exception 计失败），但排除 skipped。这样环境崩溃不会让成功率虚高。
7. CI 是 episode Bernoulli 指标的 2,000 次确定性 bootstrap percentile interval。
8. 优先报告 paired seed 的 clean-success/OOD-failure、clean-failure/OOD-success、both-success、both-failure。

正式运行前确认：suite、任务集合、每条件 episode 数（至少 20）、checkpoint/hash、五类扰动、easy/medium/hard 映射、max steps、control horizon、4 个 GPU、视频策略和输出目录。
