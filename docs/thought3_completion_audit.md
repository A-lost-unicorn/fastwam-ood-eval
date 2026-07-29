# Thought3 最终目标完成度审计

状态：`ACTIVE / INCOMPLETE`
审计日期：2026-07-29
目标文件 SHA-256：
`8459b07d6cf870ec6c1cf9f2d0254e61b543dd5c3532b116a0079a0c0c2ca972`

本文按最终 Goal 核对“已有证据能证明什么、还缺什么”。绿色单元测试、代码存在或
一次 smoke 只能证明其覆盖范围内的工程事实，不能代替六组真实 OOD 实验。

## 1. 总结

当前阶段三没有完成。最准确的状态是：

```text
Phase A/B/C/D        completed
Gate E/E.2           failed overall
Gate E.3 v1          invalid telemetry run; no gate conclusion
Gate E.3 v2          valid failed gate; 320/320 probes complete
Gate E.4             valid failed gate; 1,200 steps complete
Gate E.5             valid failed gate; 1,200 updates/9,600 objectives complete
full 28/4 Gate E     not passed
A2/A4 real training  not started
Phase F real pilot   not started
Phase G formal       not started
paper claim          unavailable
```

已经有充分证据证明：

- future latent 的真实 shape、K-step sampler、Adapter 注入和冻结边界可工作；
- 小型 K1/K2/K4 cache 可恢复、可校验、无真实未来输入；
- Adapter-only backward、checkpoint 和单卡显存可工作；
- A0/A1 单目标可拟合；
- E.2/E.4 各六条多样本训练轨迹工程上完整；
- diversified train-flow 缓解 fixed-flow 退化，但未达到冻结训练门槛；
- E.5 的 full-cohort mean-gradient、双层 telemetry 和原子恢复已完成真实
  六轨迹验证；A1@3e-4 出现 19.668%/8-of-8 候选信号，但没有共同 eligible LR。

尚无证据证明：

- 正确 future 相对 A0 提高 ID/OOD success；
- A1/A2/A4 任一 K 在真实在线控制中有效；
- shuffle future 会在真实 Fast-WAM policy 中产生超过 replay floor 的动作变化；
- OOD 收益覆盖 latency/memory；
- 三卡正式训练/评测无重复遗漏；
- 可以冻结论文主结论。

## 2. 核心科学问题

| 问题 | 当前证据 | 判定 |
| --- | --- | --- |
| 动作读取 future 后 OOD 是否提高 | 尚无六组真实 rollout | **Missing** |
| 收益是否来自 future 而非 Adapter/重训 | B0/A0 真实正式对照未运行 | **Missing** |
| K=1/2/4 的收益–成本曲线 | 只有 Phase C/D sampler latency，无在线 action/rollout | **Missing** |
| correct 与 shuffle 是否可预测地改变动作/success | 只有 CPU/mock counterfactual | **Missing** |
| 收益能否覆盖延迟/显存 | 无真实在线 P50/P95 + success 联合表 | **Missing** |

无提升、负提升或 Adapter 忽略 future 都仍是允许结果；当前不是因为结果为负而
未完成，而是关键对照尚未产生。

## 3. 架构与组别

| 要求 | 权威证据 | 状态 |
| --- | --- | --- |
| Video DiT/VAE/Action DiT 主体冻结 | Phase C/E frozen grad 与 SHA；`model_wrapper.py` | **Proved for smoke** |
| 单点 future→action 注入 | `injection.py`；Phase C parity/backward | **Proved for smoke** |
| zero-init、null、mask、统一 K schema | Adapter tests；native `[48,2,14,28]` | **Implemented/tested** |
| Adapter-only checkpoint | Phase E/E.2 round-trip | **Proved for A0/A1 smoke** |
| B0 | 配置/旧官方 policy 存在 | **Not run in Phase F/G** |
| A0 | 真实小训练已运行 | **No rollout evidence** |
| A1 | 真实小训练已运行 | **No rollout evidence** |
| A2 | 配置/mock/cache 存在 | **No real training checkpoint** |
| A4 | 配置/mock/cache 存在 | **No real training checkpoint** |
| A-shuffle | cross-task/episode derangement 和 mock action metric | **No real online policy/rollout** |

六组不得因 E.2 失败而减少。A-shuffle 应复用冻结 AK checkpoint，不另训练。

## 4. K-step future 与 cache

| 要求 | 权威证据 | 状态 |
| --- | --- | --- |
| K 是同一初始噪声的 1/2/4 sampler update | Phase C/D parity、schedule 与 paired-noise checks | **Proved** |
| native latent、不经 VAE decode | Phase C/D tensor telemetry | **Proved** |
| current+language+noise，无真实未来 | Phase C/D source audit、no-leakage tests | **Proved for pilot cache** |
| stable ID、shard、resume、checksum、corruption | Phase B/D tests 与真实 12 shards | **Proved for 32-sample pilot** |
| K 错配拒绝 | cache reader/training schema tests | **Tested** |
| 正式训练规模 cache 容量与三卡分片 | 只有估算和纯分片测试 | **Missing real-scale evidence** |
| online evaluator 不读取 cache | mock API 无 cache 参数 | **Tested only for mock** |

Phase D cache 只能用于训练；不得把它接入 Phase F/G 在线 policy。

## 5. 训练门禁

| Gate | 结果 | 能证明什么 |
| --- | --- | --- |
| C | passed | 单真实样本的 K1/2/4、forward/backward、冻结与显存 |
| D | passed | 32-sample cache build/resume/checksum/leakage |
| E v1–v3 | failed overall | 梯度/resume 子门禁通过，固定多样本 loss 未稳定下降 |
| E.1 | passed | A0/A1 单固定目标可 overfit；不是泛化 |
| E.2 | failed | 六轨迹完整；无共同 LR 达到 6/8 稳定门槛 |
| E.3 v1 | invalid run | 官方 `t=1000` 零权重 objective 触发非门控 ratio 实现错误；冻结 SHA 不变，未生成 gate result |
| E.3 v2 | valid failed gate | 320/320 probe 与全部执行检查通过；三个 LR 均未达到 A0/A1 共同 `10% + 6/8` |
| E.4 | valid failed gate | 1,200 optimizer steps、480 held-out objectives、108 execution checks 完整；六条 reduction 仅 0.997%–1.948%，无共同 LR |
| E.5 | valid failed gate | 1,200 updates、9,600 train objectives、480 held-out objectives、120 execution checks 完整；A1@3e-4 单条过门，A0@3e-4 仅 2.638%，无共同 LR |
| full E | not passed | 仍缺新的 28 train / 4 development 完整闭环 |

在 full E 通过前：

- 不训练 A2/A4；
- 不按 OOD outcome 选 LR/K/checkpoint；
- 不启动真实 Phase F rollout；
- 不把 E.2–E.5 的 A1/A0 mean-loss 差写成 future 效果。

## 6. 反事实与真实在线推理

当前 `thought3/evaluator.py` 和根 CLI 的 `thought3-evaluate` 明确是 Phase B
CPU/mock；`thought3-counterfactual` 在 Fast-WAM backend 上仍返回
`phase_c_fastwam_not_implemented`。

真实 Phase F 仍需：

1. 当前 RGB 只编码一次；
2. prompt/proprio context 只构造一次；
3. 在线运行 K-step Video DiT，不读取 cache；
4. 将 latent 通过 Adapter 注入 20 次 Action DiT denoising；
5. 固定 action seed 做 correct/null/shuffle/random/different-K；
6. 输出 action L1/L2、cosine、gripper、EEF trajectory、hash；
7. 记录 preprocessing/current/future/Adapter/action/total latency；
8. 保存 peak allocated/reserved memory；
9. 与官方 B0 `infer_action()` 做数值 parity/replay-floor gate；
10. 之后才接 LIBERO/LIBERO-Plus paired rollout。

这部分当前是最终 Goal 中最大的代码缺口之一。

## 7. 数据、评测与统计

| 要求 | 当前状态 |
| --- | --- |
| 标准 LIBERO demo only | Phase C–E 满足 |
| episode-level 90/10 split | Phase D 28/4 pilot 满足 |
| 不用 Thought1/2 rollout 训练 | 数据 API/审计满足 |
| 独立 Phase F 技术 cohort | 未冻结、未运行 |
| 新 Phase G seed/job manifest | 未生成 |
| task-equal OOD 主指标 | DRAFT 中定义，未产生 |
| 五类 OOD + Clean | Phase G 未运行 |
| paired episode seed | DRAFT 中定义，未冻结 |
| ≥10,000 task bootstrap | 实现思想存在，Phase G 未执行 |
| primary K 或 Holm | 尚未选择 |
| analysis protocol FROZEN | 文件不存在；DRAFT 不能冒充 |
| failure video/taxonomy | Phase F/G 尚未产生 |

## 8. 隔离与回归

| 要求 | 证据 | 状态 |
| --- | --- | --- |
| 独立分支/目录/CLI/schema | `feature/thought3-partial-future-adapter` 与 Thought3 namespace | **Satisfied** |
| 不修改 `third_party/FastWAM` | worktree/status checks | **Satisfied to current commit** |
| 不修改 Thought1/2 outputs | path guards + status/hash checks | **Satisfied to current commit** |
| 旧 CLI 行为不变 | old CLI regression；完整测试 314 passed | **Satisfied to current commit** |
| 正式运行前 dirty=false | Phase F/G 未到运行点 | **Pending** |
| 三 GPU shard union/intersection | 纯函数测试存在，真实 run 缺失 | **Partial** |

## 9. 完成最终 Goal 的依赖链

```text
run preregistered fresh-cohort sequential replication of A0/A1 @ 3e-4
  ↓
if replicated: freeze a new full 28/4 Gate E protocol and candidate recipe
  ↓
new full 28/4 Gate E training + dev selection
  ↓
train matched A0/A1/A2/A4 checkpoints
  ↓
real online parity + counterfactual + latency smoke
  ↓
Phase F: B0/A0/A1/A2/A4/A-shuffle small Clean/OOD pilot
  ↓
freeze model-selection rule + primary K/Holm + formal seeds/jobs/statistics
  ↓
Phase G multi-seed six-group Clean/five-category OOD evaluation
  ↓
aggregate, bootstrap, failure taxonomy, latency/memory Pareto
  ↓
paper/resume conclusions with evidence levels
```

若下一诊断或 full E 失败，应继续报告负结果并设计单变量工程诊断，不能跳过 A0、
shuffle、在线 no-cache 或统计冻结要求。

## 10. 当前最近一步

Gate E.5 已完整运行并有效失败。六条 full-cohort 轨迹完成 1,200 optimizer
updates、9,600 train objectives 和 480 held-out objectives；120/120
execution、21/21 paired、7/7 cross checks、zero-weight、checkpoint、
memory、provenance 和 frozen SHA 检查全部通过。

`A1@3e-4` held-out mean loss 下降 19.668%、8/8 sample 不变差，并在同 LR
下对 8/8 sample 都取得低于 A0 的 final loss；但 A0 只下降 2.638%，所以
预注册的 A0/A1 共同 `10% + 6/8` 不成立，`selected_lr=null`。该结果不能
事后改判为 future effect 或正式 LR。

E.6 已用未使用 train cohort、全新 flow slots 和新 Run ID 完成
A0/A1@3e-4 序贯复验的预注册与实现；协议明确披露 LR 来自 E.5
post-selection，development、OOD 和 success outcome 继续不可见。该复验尚未
真实运行，未产生任何 E.6 outcome，也未解锁完整 Gate E。
相关协议、结果与父结果见
[thought3_phase_e6_protocol.md](thought3_phase_e6_protocol.md)、
[thought3_phase_e5_protocol.md](thought3_phase_e5_protocol.md)、
[thought3_phase_e5_report.md](thought3_phase_e5_report.md)、
[thought3_phase_e4_report.md](thought3_phase_e4_report.md)。
