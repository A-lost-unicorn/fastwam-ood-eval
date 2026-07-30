# Thought3 Gate E.6：未使用 Train Cohort 序贯复验报告

状态：**Gate E.6 有效失败；A0/A1 两条轨迹完整，A1 信号复现**

运行日期：2026-07-29

## 1. 结论摘要

E.6 按冻结协议完成了未使用 train cohort 上匹配的 `A0/A1@3e-4` 两条真实
Fast-WAM Adapter 训练轨迹。程序以非零状态退出不是工程崩溃，而是预注册硬门槛
忠实触发：

- A0 held-out mean loss 下降 `1.191%`，满足“mean 不变差”，但只有 `4/8`
  sample 不变差，未达到冻结的 `6/8`；
- A1 held-out mean loss 下降 `14.842%`，`7/8` sample 不变差，A1 绝对复现
  门槛全部通过；
- A1 final sample-equal mean loss 比 A0 低 `13.815%`，且 `6/8` sample 的 A1
  final loss 不高于 A0，配对优势门槛全部通过；
- 两条轨迹所有 execution、配对、冻结、调度、恢复、显存和无泄漏检查全部通过。

因此 E.6 必须保持 `failed`，不能事后把 A0 的 `6/8` 改为 `4/8`。同时可以把
更窄的结果登记为：

> 在第二组未使用的八条 train demonstrations 和全新 action-flow objectives
> 上，post-selected `A1@3e-4` 的 held-out action-loss 改善及其相对 A0 优势均
> 复现；但 null-future A0 的逐样本稳定性没有达到冻结要求，故尚未形成可进入
> 完整 28/4 Gate E 的稳定配方。

这仍是离线 action-loss 工程证据，不是 success、OOD 或 future 因果效果。

## 2. 预注册与执行身份

| 项目 | 冻结值/实测值 |
| --- | --- |
| 预注册代码 commit | `cb6f311fe1154722eaaeaf1f02f26cfde4922d56` |
| Config fingerprint | `8cb2ab718eed2cc226491038423c92f1c59128246d966a2a9c3700d505f292d9` |
| Learning rate | `3e-4`，明确为 E.5 后 post-selection |
| Cohort | Phase D train 排序位置 9–16，8 个不同 episode |
| Cohort SHA | `6a354151d6d3e93335b66743f16be1908abc8d0fe835ee3811562b2eeb63d7c3` |
| E.5 overlap / development overlap | `0 / 0` |
| Training flow slots | `31001..32600` |
| Identity schedule SHA | `419b09a2ec30ce7bffc99c95aff1a343f77d39e83e77a752fc67bc984508febc` |
| Held-out flow steps | `1..5` |
| Tracks | A0、A1，各 200 updates、1,600 objectives |
| 总预算 | 400 updates、3,200 train objectives、160 held-out objectives |
| Future RGB / dev / OOD / success / rollout | `0 / false / false / false / false` |

结果保留了以下披露字段：

- `learning_rate_chosen_after_e5=true`；
- `learning_rate_selected_by_e5_frozen_gate=false`；
- `independent_confirmatory_test=false`；
- `thresholds_chosen_after_e5=true`。

因此本实验是序贯复验，而非独立确认性检验。

## 3. 主要结果

| Variant | Initial mean loss | Final mean loss | Reduction | Non-worsened | Catastrophic | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A0 | 0.0034140585 | 0.0033733908 | 1.191% | 4/8 | 0 | Fail：逐样本稳定性 |
| A1 | 0.0034140585 | 0.0029073545 | 14.842% | 7/8 | 0 | Pass |

Hidden correction 尺度均远低于冻结上限：

| Variant | Median delta/hidden | Max sample delta/hidden | Max objective delta/hidden |
| --- | ---: | ---: | ---: |
| A0 | 0.06971 | 0.08535 | 0.10814 |
| A1 | 0.05783 | 0.09708 | 0.15197 |

配对对照：

| 指标 | 结果 | 门槛 | 状态 |
| --- | ---: | ---: | --- |
| A1 相对 A0 final mean 改善 | 13.815% | >=10% | Pass |
| A1 final loss 不高于 A0 | 6/8 | >=6/8 | Pass |
| 相同 sample/init/probe/schedule/budget | 全部相同 | 全部相同 | Pass |

## 4. 逐样本结果

`loss ratio = final / initial`，小于等于 1 表示不变差。

| Episode | A0 ratio | A1 ratio | A1 final / A0 final | A1 比 A0 更低 |
| --- | ---: | ---: | ---: | --- |
| 000014 | 1.0023 | 0.8747 | 0.8727 | 是 |
| 000010 | 1.1315 | 0.9142 | 0.8079 | 是 |
| 000011 | 0.9748 | 1.0226 | 1.0491 | 否 |
| 000030 | 0.8700 | 0.8729 | 1.0033 | 否 |
| 000019 | 0.9798 | 0.9015 | 0.9202 | 是 |
| 000038 | 0.8933 | 0.7394 | 0.8277 | 是 |
| 000000 | 1.0141 | 0.8462 | 0.8345 | 是 |
| 000012 | 1.1032 | 0.9020 | 0.8176 | 是 |

A0 的四个变差幅度分别约为 `0.23%`、`13.15%`、`1.41%`、`10.32%`；这不是
单纯由极小数值容差造成，因为其中两条超过 10%。A1 只有 episode 000011
变差，幅度约 `2.26%`。

## 5. 与 E.5 的关系

| Cohort | A0@3e-4 reduction | A0 non-worsened | A1@3e-4 reduction | A1 non-worsened |
| --- | ---: | ---: | ---: | ---: |
| E.5 train positions 1–8 | 2.638% | 7/8 | 19.668% | 8/8 |
| E.6 train positions 9–16 | 1.191% | 4/8 | 14.842% | 7/8 |

两个 cohort 都显示 A1 的 mean reduction 大于 A0，并且 E.6 的预注册相对优势
门槛通过。这提高了“A1 工程信号可重复”的可信度。但 E.6 同时证明 A0 的样本级
稳定性不能从 E.5 外推，因此不能只看 A1 或 pooled mean 就越过总门禁。

## 6. 工程与完整性审计

- A0/A1 均完成 200 updates、1,600 objectives、四个 checkpoint；
- 两条 objective/update/probe JSONL 行数分别为 `1600/200/2`；
- 所有 19 个预知 zero-weight objective 与冻结位置完全一致；
- update 1 仅 gate 获得梯度，update 2 projector/attention 获得非零梯度；
- Fast-WAM 参数 SHA 前后均为
  `ac0dd59de78495dc8ed5c5601e24951b78f3fd6ad40843d76b83106eccceb4f8`；
- train peak 为 `13277.440 MiB`，model-load peak 为 `23679.513 MiB`；
- A0/A1 mean update time 分别为 `4.962/4.972 s`；
- 总 wall time 为 `43.89 min`，模型加载 `7.19 min`；
- E.5 五个冻结父工件运行后复核不变；
- 原始 JSONL 独立重算的 execution、performance、paired 和 artifact SHA 与
  `gate_e6_result.json` 完全一致。

根工件 SHA：

| 工件 | SHA-256 |
| --- | --- |
| `gate_e6_result.json` | `464d9d3e02c52c2b1f2838ce59fe71a9b35716884d4d1da4b3d0e2ad78b42af6` |
| `run_status.json` | `b6dd1edf41375e4ecd5d6495976298b6246307eafe16946be6662a99cb3b9adc` |
| `pre_validation_result.json` | `3639032aa3d8faed5fd20d9f5da313ee51fb7605cf81aac95465a78390d83ec2` |
| `data_preparation.json` | `4f8c6d02c06a4f6a80bc01ec54e88c13a39d4bab4be9f04b7cb547347af552df` |
| `logs/phase_e6.log` | `b888d48f3b45dedc7577f616a6910400d950d38aa38be75ebbacf4f8d90eb81d` |

## 7. 失败含义与下一步

当前禁止：

- 对本 Run ID 再执行 `--resume`；
- 把 A0 的 `6/8` 门槛追溯降为 `4/8`；
- 把两 cohort 的 A1 强信号直接写成 future 因果增益；
- 进入完整 28/4 Gate E、A2/A4 或 ID/OOD rollout。

最省算力且不消耗剩余 train cohort 的下一步，是只读 checkpoint trajectory
诊断：在同一 E.6 cohort 上离线评估已保存的 step 50/100/150/200 A0/A1
checkpoint。为避免复用已知 step-200 outcome，主要判断使用新 flow `6..10`；
旧 flow `1..5` 只做 continuity reproduction。该 E.7 已按预注册协议完成：
primary 不支持实质晚期退化，且没有 joint candidate；详见
[协议](phase_e7_protocol.md)与
[结果报告](phase_e7_report.md)。

该分析只能是 post-run engineering diagnosis；若据此选择 checkpoint，仍须在
尚未使用的样本上重新预注册验证，不能在 E.6 上自证。
