# 实验、卡点与结论台账

更新日期：2026-07-23

本台账只记录可追溯事实。机器工件是权威来源，本文是便于论文、周报、简历和面试使用的索引。

## 1. 已运行实验

| Run ID | 日期 | 阶段/等级 | 配置或来源 | 分母与结果 | 结论资格 |
| --- | --- | --- | --- | --- | --- |
| `P1-CLEAN-SMOKE-v1` | 2026-07-22 | 1 / SMOKE | `configs/eval_clean_smoke.yaml` | 2 completed，2 success，0 exception | 只证明 Clean 链路 |
| `P1-OOD-SMOKE-v1` | 2026-07-22 | 1 / SMOKE | `configs/eval_ood_smoke.yaml` | 4 completed，4 success，0 exception | 只证明 camera/light 链路 |
| `P1-OOD-PILOT-v1` | 2026-07-22 | 1 / PILOT | `configs/eval_ood_pilot.yaml` | 9 planned，8 attempted，2 success，1 skipped，0 exception | 不得作为正式 OOD 成功率 |
| `P1-FORMAL-PLAN-v1` | 2026-07-22 | 1 / PLAN | `outputs/thought1/.../job_manifest.jsonl` | 800 Clean；6,771 OOD runnable；68 skipped | 分母已审计，rollout 未运行 |
| `P2A-CLEAN-SMOKE-v1` | 2026-07-23 | 2A / SMOKE | `configs/studies/thought2_unconditional_smoke.yaml`；只读 `outputs/clean_smoke` | 1 job，1 probe，2 aligned future frames，0 error | 只证明真实 future 工件与指标链路 |

### `P2A-CLEAN-SMOKE-v1` 机器证据

- checkpoint SHA-256：`1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579`
- Fast-WAM commit：`45d8e1458921d83f8ad6cf9ce993d371208dabd0`
- source manifest SHA-256：`8e9231615887e4a58053cec1ea7454247982b5fac1c73c4d30cd28c4429c4628`
- protocol fingerprint：`22f2ddbe80b18b07c6345e5dad4823a5e6e02842111ad8071c6c07a0714796da`
- action hash 前后完全一致：`42d23114...bbcad`
- 时间对齐：预测帧 0/1/2 对应环境 offset 0/4/8，即 0/0.2/0.4 s；运行时 control frequency 已验证为 20 Hz
- 生成：9 帧，2 个视频去噪步；generation latency `1,223.53 ms`
- 完整诊断：`4,616.06 ms`，不包含环境 step
- probe 峰值显存：`24,841.09 MB`；相对 probe 前增量约 `1,152.88 MB`
- 工件：当前帧 PNG、预测 9 帧 MP4、实际 3 帧 MP4、并排 3 帧 MP4，均已真实写入并抽检
- episode 在 `max_steps=10` 被 smoke 人为截断，`success=false` **不能解释为模型任务失败**
- 该 smoke 运行时阶段二实现尚未提交；旧 manifest 只记录 HEAD `9dfc254`，
  未记录 dirty 状态。因此它保持 SMOKE 资格，不作为可复现实验主结果。后续
  manifest 已增加 `git_dirty` 与三个上游 `*_dirty` 字段。

当前自动一致性数值仅留作管线检查：latent L1 `0.1437`、latent cosine distance `0.1025`、motion-direction cosine `0.7961`。不得将其写成模型能力结论。

## 2. 失败尝试与解决记录

| 日期/尝试 | 现象 | 根因 | 修复与证据 | 是否污染实验 |
| --- | --- | --- | --- | --- |
| 2026-07-23 / P2A smoke attempt 1 | policy 导入时报 Fast-WAM dependencies unavailable | 官方 Fast-WAM evaluator 在 policy 构造时 import `libero`，但 backend 路径此前只在环境构造时设置 | 抽出无仿真副作用的 `configure_libero_package()`，policy 和 environment 复用 | 否；未加载模型、未 reset |
| attempt 2 | checkpoint 加载后报不同 LIBERO package | 同一工作区有 `/home/...` 与 `/data/...` 路径别名，字符串比较误判 | 改为 `Path.resolve()` 后的父目录身份判断 | 否；未 reset、无 diagnostic row |
| attempt 3 | 仍报不同 LIBERO package | 顶层 `libero` 是 namespace package，`__file__=None`，真实来源在 `__path__` | 同时验证 `__file__` 与全部 `__path__`；新增 symlink/namespace 回归测试 | 否；未 reset、无 diagnostic row |
| attempt 4 | 完成 | 完整链路通过 | 1 job / 1 probe / 0 error；142 tests passed | 产生有效 SMOKE 工件 |

冷启动观测：失败尝试和成功尝试中 Wan 组件装载约 `336–433 s`。这不是单次 future latency；正式运行必须一 worker 多 episode 复用模型。

Provenance 补充核对：Fast-WAM 和 LIBERO checkout clean；LIBERO-Plus
只有非源码下载缓存 `.downloads/assets.zip`，无 tracked diff。后续 manifest
会显式记录 `.downloads/` 排除项，其他未跟踪文件和任何 tracked 修改仍会令
`*_dirty=true`。

## 3. 当前结论账

### 已支持

1. 阶段一评测工程链路和正式 manifest 已准备好。
2. 官方 `libero_uncond` release 在本机能够真实生成 unconditional future。
3. 阶段二 A 能在不改变动作哈希的前提下保存当前帧、预测未来、实际未来、动作、结果和资源指标。
4. 预测与实际帧可按官方 4 action/frame 比例精确对齐到控制步。

### 尚未支持

1. Fast-WAM 正式 Clean→OOD 成功率下降。
2. 哪类扰动最敏感。
3. unconditional future consistency 与成功/OOD 的稳定相关性。
4. “失败来自未来错误还是动作错误”的自动因果分类。
5. 显式未来能改善 OOD，或 K=1/2/4 中哪个最好。

## 4. 待填论文主表

以下表格是论文数字的唯一人工汇总入口。先从机器报告复制分母和估计值，再由
第二人核对 artifact；禁止手算后直接覆盖机器结果。

### 4.1 阶段一：ID→OOD 鲁棒性

| Suite | Perturbation | Level | ID n / SR / 95% CI | OOD n / SR / 95% CI | Absolute drop (pp) | Relative drop | Action latency p50 / p95 | Failure videos reviewed | FORMAL Run ID |
| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |
| 待正式运行 | — | — | — | — | — | — | — | — | — |

主文至少给总体和四类目标扰动；附录再展开 suite、difficulty 和变体。必须同时
报告 exception/skipped，并区分 episode-weighted 与 variant-weighted 口径。

### 4.2 阶段二：未来一致性

| Cohort | Mode | Condition/outcome | Episodes / probes / aligned frames | Video steps | L1 | Cosine distance | Motion-direction cosine | Human agreement | Generation / diagnostic latency | FORMAL Run ID |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 待阈值校准与正式抽样 | 2A | — | — | 20 | — | — | — | — | — | — |

自动指标与盲审标签分开保存。success/failure matched cohort 不能用于估计总体失败率；
2A 结果始终标记 `causal_interpretation_allowed=false`。

### 4.3 阶段三：部分未来 Adapter

| Variant | K | Train seed | Trainable params | ID SR | OOD SR | Absolute drop | Future latency | Action latency | Total latency | Peak memory | FORMAL Run ID |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `B0-base` | 0 | — | 0 | — | — | — | 0 | — | — | — | — |
| `A0-null` | 0 | — | — | — | — | — | 0 | — | — | — | — |
| `A1` | 1 | — | — | — | — | — | — | — | — | — | — |
| `A2` | 2 | — | — | — | — | — | — | — | — | — | — |
| `A4` | 4 | — | — | — | — | — | — | — | — | — | — |

所有行必须使用相同 episode manifest、Action DiT 去噪步数、训练预算和选模规则。
`B0-base` 对 `A0-null` 控制额外参数/训练效应；`A0-null` 对 A1/A2/A4 才隔离
future 信息效应。

### 4.4 结论—证据登记

| Claim ID | 拟写结论 | 所属阶段 | 必须满足的证据 | 当前状态 | Artifact/Run ID |
| --- | --- | --- | --- | --- | --- |
| `C1` | Fast-WAM 对特定环境 shift 敏感 | 1 | 完整 ID/OOD 分母、配对 drop、CI、0 未解释 exception | 未支持 | — |
| `C2` | future consistency 在 OOD 或失败样本下降 | 2A | 冻结 cohort、校准阈值、episode-level 统计与盲审 | 未支持 | — |
| `C3` | 显式部分未来提高 OOD 成功率 | 3 | B0/A0/A1/A2/A4 配方匹配、跨 seed 配对 CI | 未支持 | — |
| `C4` | 某个 K 位于效果—延迟 Pareto 前沿 | 3 | 在线 latency/memory 与同 manifest ID/OOD 结果 | 未支持 | — |

只有满足“必须证据”后才能把状态改为“支持/不支持”；负结果同样保留，不按预期
方向筛选。

## 5. 更新纪律

- 新实验先登记 Run ID、配置、输出目录、checkpoint/commit、证据等级和停止条件。
- 失败尝试也登记；不得只保留最终成功路径。
- 结果表必须写分母、exception/skipped、CI 和适用范围。
- 修改协议后使用新 output directory 和 protocol fingerprint。
- `FORMAL` run 必须在项目和上游 checkout 均为 clean tree 时启动；否则降级为 PILOT。
- 简历数字必须能回指 manifest、JSONL、聚合报告或测试输出。
