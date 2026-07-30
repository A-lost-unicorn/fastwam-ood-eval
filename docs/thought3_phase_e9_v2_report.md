# Thought3 Gate E.9a-v2 结果与工程审计

状态：**PARENT STATUS PRESERVED INVALID / V2.1 AUDIT VALID / SCIENTIFIC FAILED**

运行日期：2026-07-29
预注册 commit：`694d1d0d98b11d24ca16a0c261dc2fa89e750d6a`
输出目录：`outputs/thought3/phase_e9_sample_tail_mitigation_v2/`

## 1. 结论

E.9a-v2 的四条 matched 轨迹全部完成 step 200，但根 Gate 不能登记为有效
科学结果。唯一失败的工程检查是四轨共同的
`heldout_rng_and_zero_weight_identity_exact=false`：probe row 没有保存 checker
要求的 `action_noise_seed`、`action_timestep_seed` 和
`flow_objective_sha256`。这是 telemetry/checker schema 缺陷，不是训练中断、
NaN、OOM、样本/flow 错配、参数泄漏或 checkpoint 失败。

冻结结果中的预注册分类为：

```text
classification=sample_tail_mitigation_not_supported
independent_replication_candidate=false
```

原因不是 normalized absolute Gate 或 tail Gate 失败，而是 normalized A1
相对 A0 的 final mean 优势只有 `8.274%`，低于冻结的 `10%` paired 门槛。
因此：

- 不运行 E.9b；
- 不降低 `10%` 门槛；
- 不选择 step 100；
- 不 resume、覆盖或重训 E.9a-v2；
- 下一步只允许预注册的只读 artifact audit。

Phase 0 只读审计已经通过，因此以下数值可写成经 provenance 恢复的
post-run engineering metrics；仍不能写成 confirmatory Gate 或 mitigation
成功。

## 2. 计算完整性

| 项目 | 结果 |
| --- | ---: |
| 轨迹 | 4：raw/normalized × A0/A1 |
| optimizer updates | 800 |
| training objectives / backward | 6,400 / 6,400 |
| held-out objectives | 2,048 |
| 每轨 checkpoint | step 50/100/150/200 |
| step-200 checkpoints | 4 |
| training samples | 8 |
| reserved 17–28 decoded/trained | 0 |
| development/OOD/success outcome | 0 |
| future RGB input | 0 frame |
| frozen Fast-WAM SHA before/after | `ac0dd59d...ceb4f8` / 相同 |

四个子轨 `run_status.json` 均为：

```text
status=complete
completed_steps=200
completed_objectives=1600
```

所有四轨共享：

- 相同 initial Adapter；
- 相同 initial probe；
- 相同八条 sample；
- 相同 `40001..41600` training schedule；
- 相同 `75..106` held-out schedule；
- 相同训练预算；
- raw/normalized 之间只改变固定 sample weighting。

## 3. 四轨主要结果

共同 initial held-out mean action loss 为
`0.004866404297096949`。

| Recipe / variant | Final mean loss | Loss reduction | Non-worsened | Catastrophic | Confirmed harm |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw/A0 | 0.0046632422 | 4.175% | 6/8 | 0 | 2 |
| raw/A1 | 0.0042340717 | 12.994% | 8/8 | 0 | 0 |
| normalized/A0 | 0.0047212370 | 2.983% | 7/8 | 0 | 0 |
| normalized/A1 | 0.0043306018 | 11.010% | 8/8 | 0 | 0 |

四条 absolute performance Gate 均通过：

- A0 mean 不变差；
- A1 mean reduction 至少 10%；
- 每条轨迹至少 6/8 sample 不变差；
- 无 sample 超过 initial loss 的 2 倍；
- Adapter correction 尺度均低于冻结上限。

### 3.1 Paired A1-vs-A0

| Recipe | A1 final 比 A0 低 | A1 不高于 A0 | 10% paired Gate |
| --- | ---: | ---: | --- |
| raw | 9.203% | 8/8 | **Fail** |
| normalized | 8.274% | 8/8 | **Fail** |

normalized A1 的 absolute improvement 为 `11.010%`，但候选判定同时要求
normalized A1 相对 normalized A0 至少低 `10%`；该条件差 `1.726` 个百分点。

### 3.2 Tail 结果

20,000 次 paired-flow bootstrap、32-comparison family-wise 校正得到：

| Track | Confirmed harmed samples |
| --- | ---: |
| raw/A0 | 2 |
| raw/A1 | 0 |
| normalized/A0 | 0 |
| normalized/A1 | 0 |

raw/A0 的两条 confirmed harm 为：

- `5f82a5db...def44`，对应 `episode_000011`，full relative change
  `+2.180%`；
- `81363fef...e3b9b`，对应 `episode_000012`，full relative change
  `+4.173%`。

sample normalization 将 A0 的 confirmed harm 从 `2` 降为 `0`，并把 A0
point stability 从 `6/8` 提到 `7/8`。但它同时把：

- A0 pooled reduction 从 `4.175%` 降到 `2.983%`；
- A1 pooled reduction 从 `12.994%` 降到 `11.010%`；
- A1-vs-A0 paired advantage 从 `9.203%` 降到 `8.274%`。

所以当前证据表现为“tail stability 与平均/paired gain 的权衡”，不是达到
预注册门槛的 mitigation candidate。

## 4. Adapter 尺度与资源

| Track | Median delta/hidden | Max sample delta/hidden | Max objective delta/hidden | Final gate |
| --- | ---: | ---: | ---: | ---: |
| raw/A0 | 0.06559 | 0.07085 | 0.09866 | -0.006669 |
| raw/A1 | 0.05786 | 0.09118 | 0.13551 | -0.011988 |
| normalized/A0 | 0.04571 | 0.04938 | 0.06882 | -0.005663 |
| normalized/A1 | 0.04870 | 0.05971 | 0.09488 | -0.010731 |

资源记录：

| 项目 | 结果 |
| --- | ---: |
| 总 wall time | 5,316.27 s / 88.60 min |
| 模型加载 | 387.34 s |
| 模型加载峰值 | 23,679.51 MiB |
| 训练执行峰值 | 13,277.44 MiB |
| mean optimizer update | 4.932–4.967 s |
| 输出大小 | 268 MiB |
| 输出文件 | 77 |

这些是单卡训练/held-out 诊断资源，不是 online rollout latency。

## 5. 工程 invalid 的精确原因

四轨唯一 false execution check 均为：

```text
heldout_rng_and_zero_weight_identity_exact=false
```

实际 probe 工件满足：

- initial/final 每轨各 256 rows；
- sample × flow grid 精确覆盖 `8 × 32`；
- flow 精确为 `75..106`；
- initial/final 的 `(sample_id, flow_step, timestep, action_weight)` 全量相同；
- 两个零权重位置精确为预注册的 0-based
  `(sample index 1, flow 80)` 和 `(sample index 7, flow 93)`；
- 两个零权重 objective 的 action loss 精确为 0；
- outcome 可从逐 objective row 精确重算。

但 `evaluate_multiflow_probe_grid()` 保存的 row 只有 loss、weight、timestep、
sample/flow 和 Adapter diagnostics，没有保存：

```text
action_noise_seed
action_timestep_seed
flow_objective_sha256
```

`_track_checks()` 又逐 row 要求这三个字段等于
`_flow_objective_identity()` 的派生值，因此该检查在当前 schema 下必然为
false。根 `engineering_passed=false`、`status=invalid` 和命令非零退出是
fail-closed 行为，不能事后直接改写。

其他 cross checks 全部通过：

- repository provenance 未变；
- E.8 冻结工件未变；
- Fast-WAM 无 requires-grad、无 grad、SHA 未变；
- 四轨 pairing contract 通过；
- reserved cohort 未解码/训练。

## 6. 冻结工件

### 6.1 根工件

| 工件 | SHA-256 |
| --- | --- |
| `gate_e9a_result.json` | `022e80868b56d7af7979e3c43a995061945ff7f615c6c21c5cf79256c8e25e24` |
| `run_status.json` | `c6e546c1fbeb64ac64462c2183dd8622a4e68c0b639070c415837c9f955a7fe9` |
| `pre_validation_result.json` | `bfd83683c9b50a52cd4d284c53917b110e1a5933edb76b265dacbd9bc1db4333` |
| `data_preparation.json` | `68ab4dbee025afaa1ceeaf2c10a5429f9da126f9d6115939e63a0476e206c5f4` |
| `logs/phase_e9a_v2.log` | `17cb5abe1982d075412b1e6555e94329d20265732ee4345f164cbf0e7fceb5d3` |

### 6.2 每轨核心工件

| Track | training manifest SHA | held-out metrics SHA | step-200 Adapter SHA |
| --- | --- | --- | --- |
| raw/A0 | `07a2cc94fd3885fa56a9cc4cc1bca1e29fbd3679751bb7042caccec22a0f4710` | `8a53c38c11d688fa246b2be2b8e163f283050dfb935a8c904c4160c1c3d15753` | `0d850504f8e5199a737c3b66daaa57b1075cfccfcb6918fd34c71ce356a58eef` |
| raw/A1 | `c3f3ae80a5c59f260a0a10dc5e9280386b30bfd43ab6348d8ce012b6339da39b` | `48149b03ee8228a322df9397680ea863a6feb11dc27e6abe9ffe74ebaf598046` | `237a92d5820ca3c657efd94abfea61b05f14874165669053e8700d2e3e4432e8` |
| normalized/A0 | `13c72f643a1ae7dc0f635659a6b3db5a745bc554a35b7e3e49c711556b87a4b5` | `29652779af2ac243bbe5413fbf6f45b70931c454eed888472e409427e05d6248` | `61965a4ffb474055b546a3d7bfe7520bbaf694546e63da86d5e8933659aa9bba` |
| normalized/A1 | `ef57b5e74eb2675853efbc3fe2e59334bc436d0eb1bbe81a59085f1daf98638c` | `c695df922799c9dfa703fe40b9318957a8a0bb3247910442b997015ceffcaa12` | `e63ccbf440a25154d810711abbc72c6f98edeb9b867dcb5d09fdcfa223c7d169` |

运行输出保持原样，不新增、不覆盖任何 result/status/metric/checkpoint 文件。

## 7. 可以写与不能写

可以写：

> 完成了 raw/normalized × A0/A1 的 matched Adapter 工程实验，共 6,400
> training objectives 和 2,048 held-out objectives；sample normalization
> 在冻结 panel 上消除了 A0 的两条 FWER-confirmed harm，但 paired
> A1-vs-A0 advantage 只有 8.274%，未达到预注册 10% 门槛。

必须同时披露：

> 运行因 held-out RNG identity telemetry 未落盘而被工程 Gate 判为 invalid；
> 上述数字是冻结工件上的 provisional post-run metrics，等待只读审计。

不能写：

- sample normalization 已通过 Gate；
- future Adapter 已改善 OOD；
- normalized 配方是 E.9b 或完整 Gate E 候选；
- 8.274% 接近 10%，因此可放宽门槛；
- E.9a 证明未来 latent 有或没有因果价值。

## 8. Phase 0 审计结果与下一步

`E.9a-v2.1 read-only artifact audit` 已执行：

1. 27/27 hard checks true；
2. 父目录 77 文件前后完全不变；
3. 0 CUDA/model/checkpoint tensor/forward/backward/optimizer；
4. 2,048 stored probe objectives 的 grid/timestep/weight/zero positions 精确；
5. 恢复 256 个唯一 RNG objective identity；
6. 输出 `audit_valid_scientific_failed`。

冻结性能门槛仍给出
`sample_tail_mitigation_not_supported`，所以 E.9b 继续锁定。
下一步不再增加 surrogate Gate，而是运行固定 E6 checkpoint 的 K=1 online
correct/null/shuffle action counterfactual。详见
[审计报告](thought3_phase_e9_v2_1_readonly_audit_report.md)与
[Phase 1 协议](thought3_phase1_k1_online_counterfactual_protocol.md)。
