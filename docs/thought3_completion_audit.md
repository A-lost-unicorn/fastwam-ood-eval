# Thought3 最终目标完成度审计

状态：`ACTIVE / INCOMPLETE`
审计日期：2026-07-30
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
Gate E.6             valid failed gate; 400 updates/3,200 objectives complete
Gate E.7             valid read-only diagnosis; primary hypothesis not supported
Gate E.8             valid read-only diagnosis; mixed/inconclusive classification
Gate E.9a-v1         invalid engineering run; 0 objectives/updates
Gate E.9a-v2         four tracks complete; parent status remains invalid
Phase 0 audit        valid read-only recovery; scientific gate failed
E.9b replication     locked; no candidate
Phase 1 online CF    valid engineering smoke; branch A
full 28/4 A0/A1      preregistered/implemented/dry-run passed; GPU not started
A2/A4 real training  not started
directional OOD pilot not started
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
- E.6 在未使用 cohort 和全新 flow slots 上复现 A1 信号：14.842%/7-of-8，
  相对 A0 final mean 低 13.815%；但 A0 仅 4/8 不变差，故总 Gate 有效失败。
- E.7 以 800 个只读 objective 完成八个 checkpoint 的冻结诊断；primary 上
  A0 step 50/100 稳定、150/200 不稳定，但 endpoint mean 比 step 50 低
  5.651%，故不支持预注册的实质晚期退化模式；A1 随 step 增强但无 joint
  checkpoint candidate。
- E.8 以 1,536 个只读 objective、64 个全新 flow、双 32-flow block 和 FWER
  bootstrap 完成真实诊断；step-200 pooled mean 改善 3.728%，但只有 1/3
  预识别 target 被确认恶化，另有一条非 target 确认恶化，故分类为
  `mixed_or_inconclusive`。
- E.9a-v1 在 objective 1 前工程失效；v2 四轨完成 800 updates、6,400 train
  与 2,048 held-out objectives。normalization 将 A0 confirmed harm 从 2
  降到 0，但 paired gain 只有 8.274% < 10%。Phase 0 只读审计已证明缺失字段
  可由冻结代码路径恢复、父 77 文件未改；结果登记为
  `engineering valid + scientific failed`，无 E.9b candidate。
- K=1 在线 B0/correct/formal-null/shuffle 已完成真实单卡运行：B0 replay 与
  formal-null 逐位一致，correct-null、correct-shuffle 和 action-hash 均为
  `8/8`；62/62 工件 SHA、frozen/no-grad/no-cache/no-future-RGB checks 通过，
  冻结分类为分支 A。

尚无证据证明：

- 正确 future 相对 A0 提高 ID/OOD success；
- A1/A2/A4 任一 K 在真实在线控制中有效；
- Phase 1 的小幅动作变化会改善机器人轨迹或 success；
- OOD 收益覆盖 latency/memory；
- 三卡正式训练/评测无重复遗漏；
- 可以冻结论文主结论。

## 2. 核心科学问题

| 问题 | 当前证据 | 判定 |
| --- | --- | --- |
| 动作读取 future 后 OOD 是否提高 | 尚无六组真实 rollout | **Missing** |
| 收益是否来自 future 而非 Adapter/重训 | B0/A0 真实正式对照未运行 | **Missing** |
| K=1/2/4 的收益–成本曲线 | K1 已有 online action latency；K2/K4 与 success 均无 | **Missing** |
| correct 与 shuffle 是否改变动作 | Phase 1 同 checkpoint 技术反事实 `8/8` 改变 | **Proved for one-task smoke** |
| correct 与 shuffle 是否改变 success | 无 paired rollout | **Missing** |
| 收益能否覆盖延迟/显存 | Phase 1 已有在线 P50/P95/peak，但没有 success 联合表 | **Missing** |

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
| A-shuffle | 固定 other-episode derangement；真实 K=1 online action CF 已运行 | **Action smoke proved / no rollout** |

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
| online evaluator 不读取 cache | Phase 1 source telemetry 与 API hard guard | **Proved for K=1 action smoke** |

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
| E.6 | valid failed gate | 新 cohort 上 400 updates、3,200 train objectives、160 held-out objectives 完整；A1 absolute/paired superiority 通过，A0 4/8 stability 未过门 |
| E.7 | valid read-only diagnosis | 800 forward、0 backward/optimizer/write；primary 分类 `not_supported_no_material_late_degradation`，continuity 描述性分类相反；无 joint candidate |
| E.8 | valid read-only diagnosis | A0 step 100/200、flows 11–74、双 32-flow block、1,536 forward、20k bootstrap + 20k five-flow sensitivity；工程通过，分类 `mixed_or_inconclusive` |
| E.9a-v1 | invalid engineering run | raw/A0 initial probe 前被旧 `1..5` flow 硬编码拒绝；0 training objective、0 optimizer update、无 checkpoint/result；frozen SHA 未变 |
| E.9a-v2 | compute complete / engineering invalid | 四轨 800 updates、6,400 train + 2,048 held-out objectives；raw A0/A1 reduction 4.175%/12.994%，normalized 2.983%/11.010%；tail 2/0→0/0；paired 8.274% 未过 10%；RNG identity telemetry 缺失 |
| Phase 0 E9a-v2.1 | valid audit / scientific failed | 27/27 checks true；0 forward/backward/optimizer/GPU/parent write；恢复 RNG identity，确认 `sample_tail_mitigation_not_supported` 与 E9b locked |
| Phase 1 online CF | valid engineering smoke / branch A | 固定 E6 A1 step-200 与 E6 8-sample cohort；B0/null 精确 parity，correct-null、correct-shuffle、hash 均 `8/8` |
| full 28/4 A0/A1 | valid offline negative / complete | 两轨各 200×28；12/12 hard checks；A0 `+1.845%`、A1 `−1.712%`，A1 比 A0 高 `3.624%`；Phase 3 locked |

Phase 1 分支 A 和 Phase 2 有效负结果均已产生。当前停止条件为：

- 不新增 surrogate Gate；
- 不重跑 full 28/4，不启动 A2/A4 或 OOD rollout；
- 不按动作差异或 OOD outcome 选 LR/K/checkpoint；
- 不把 E.2–E.9 的 A1/A0 loss 差写成 future 动作效应。

## 6. 反事实与真实在线推理

旧 `thought3-evaluate`/`thought3-counterfactual` 仍是 Phase B CPU/mock，未被
改写。真实技术反事实通过独立命令
`thought3-k1-online-counterfactual` 实现，当前已覆盖：

1. strict current-only Dataset column selection 与一次 current encode；
2. prompt/proprio context；
3. online frozen Video DiT 严格 K=1，不读取 cache、不 decode future RGB；
4. Adapter 在 20-step Action DiT 中的精确 hook-call contract；
5. 原始 B0 `infer_action()`×2 replay、正式无 tensor null 与 other-episode
   shuffle；
6. action tensor/hash、四组 pair metric、逐 timestep/component difference；
7. synchronized stage latency 与 peak allocated/reserved；
8. replay floor、B0/null parity、A/B/C fail-closed 分类；
9. frozen checkpoint/cohort/config/SHA、atomic artifacts 与 checksum resume。

真实单卡 Phase 1 已得到分支 A，完整 28/4 A0/A1 也已完成。Phase 2 direction
为负，因此按预注册规则不复验完整 checkpoint、不启动 LIBERO/LIBERO-Plus
paired rollout。当前缺口是该单-task offline 负结果不能回答 OOD success 因果
问题，而不是等待继续运行本路线。

## 7. 数据、评测与统计

| 要求 | 当前状态 |
| --- | --- |
| 标准 LIBERO demo only | Phase C–E 满足 |
| episode-level 90/10 split | Phase D 28/4 pilot 满足 |
| 不用 Thought1/2 rollout 训练 | 数据 API/审计满足 |
| 独立 Phase 1 技术 cohort | E6 已消耗 8 条 train ID 已冻结并完成真实 GPU action CF |
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
| 旧 CLI 行为不变 | old CLI regression；完整测试 386 passed | **Satisfied to current commit** |
| 正式运行前 dirty=false | Phase F/G 未到运行点 | **Pending** |
| 三 GPU shard union/intersection | 纯函数测试存在，真实 run 缺失 | **Partial** |

## 9. 完成最终 Goal 的依赖链

```text
Phase 0 E9a-v2.1 audit completed:
engineering valid + scientific failed; E9b locked
  ↓
Phase 1 real K=1 B0/correct/null/shuffle action counterfactual
  ├─ A future-content sensitivity（已观测）
  │    ↓
  │  Phase 2: one matched 28/4 A0/A1 recipe
  │    ↓
  │  recheck K=1 action sensitivity on full checkpoint
  │    ↓
  │  Phase 3: 240-rollout B0/A0/A1/A-shuffle
  │           Clean/camera/robot-init directional pilot
  │    ↓ only if success signal is positive
  │  Phase 4: A2/A4 + formal multi-seed protocol
  ├─ B latent-presence only
  │    ↓
  │  one single-variable injection fix + one Phase 1 repeat
  └─ C no material action sensitivity
       ↓
     stop Adapter-only/full/A2/A4/OOD route
```

Phase 1 已结束 surrogate 循环并进入 A。后续不得用新的 surrogate Gate 延迟或
改写该结果，也不得跳过 A0、shuffle、online no-cache 或统计冻结要求。

## 10. 当前最近一步

Phase 0 已完成：E9a-v2.1 的 27/27 hard checks 全部为 true，0
forward/backward/optimizer/checkpoint tensor load、0 CUDA、父目录 0 write。
因此 v2 训练可登记为工程有效；但 normalized paired `8.274%<10%` 不变，
科学分类为 `sample_tail_mitigation_not_supported`，E9b locked。

Phase 1 已用固定 E6 A1@3e-4 step-200 和八条 E6 train demonstration 完成
真实单卡运行。正式 null 不创建零 tensor；B0 replay/null parity 精确为 0；
correct-null、correct-shuffle 与 action hash 均为 `8/8`，分类为
`future_content_sensitivity_observed`。paired correct-null overhead mean 为
`258.95 ms`，但 normalized action L2 mean 只有 `0.011052`，且没有 rollout。

Phase 2 已在修复 commit `8e41d0b` 上由原目录 resume 后完整完成。A0/A1
各完成 200×28 objectives；12/12 hard checks、32/32 manifest descriptors 和
8/8 checkpoint provenance 通过。固定 step-200 development 上，A0 reduction
为 `+1.845%`，A1 为 `−1.712%`；A1 final 比 A0 高 `3.624%`，4/4 sample
方向一致更差。冻结分类为
`training_valid_dev_direction_not_observed`，因此 Phase 3、完整-checkpoint
反事实、OOD pilot 与 A2/A4 均不解锁。

相关协议、结果与父结果见
[thought3_accelerated_roadmap.md](thought3_accelerated_roadmap.md)、
[thought3_phase1_k1_online_counterfactual_protocol.md](thought3_phase1_k1_online_counterfactual_protocol.md)、
[thought3_phase1_k1_online_counterfactual_report.md](thought3_phase1_k1_online_counterfactual_report.md)、
[thought3_phase2_full_28_4_protocol.md](thought3_phase2_full_28_4_protocol.md)、
[thought3_phase2_full_28_4_report.md](thought3_phase2_full_28_4_report.md)、
[thought3_phase_e9_v2_1_readonly_audit_report.md](thought3_phase_e9_v2_1_readonly_audit_report.md)、
[thought3_phase_e9_v2_1_readonly_audit_protocol.md](thought3_phase_e9_v2_1_readonly_audit_protocol.md)、
[thought3_phase_e8_protocol.md](thought3_phase_e8_protocol.md)、
[thought3_phase_e8_report.md](thought3_phase_e8_report.md)、
[thought3_phase_e9_v1_failure_report.md](thought3_phase_e9_v1_failure_report.md)、
[thought3_phase_e9_v2_protocol.md](thought3_phase_e9_v2_protocol.md)、
[thought3_phase_e9_v2_report.md](thought3_phase_e9_v2_report.md)、
[thought3_phase_e7_report.md](thought3_phase_e7_report.md)、
[thought3_phase_e7_protocol.md](thought3_phase_e7_protocol.md)、
[thought3_phase_e6_report.md](thought3_phase_e6_report.md)、
[thought3_phase_e6_protocol.md](thought3_phase_e6_protocol.md)、
[thought3_phase_e5_protocol.md](thought3_phase_e5_protocol.md)、
[thought3_phase_e5_report.md](thought3_phase_e5_report.md)、
[thought3_phase_e4_report.md](thought3_phase_e4_report.md)。
