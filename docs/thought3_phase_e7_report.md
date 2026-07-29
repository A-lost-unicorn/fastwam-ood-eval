# Thought3 Gate E.7：只读 Checkpoint-Trajectory 诊断报告

状态：**工程 Gate 通过；主要结论为 `not_supported_no_material_late_degradation`**

运行日期：2026-07-29

预注册 commit：
`703b57fd017d73beed16ab5c9f7484683b62772d`

## 1. 一句话结论

E.7 按预注册协议只读评估了 E.6 的 A0/A1 step
`50/100/150/200` checkpoint。新 primary flow `6..10` 上：

- A0 在 step 50/100 通过稳定性门槛，在 step 150/200 因只有 `5/8`
  sample 不变差而失败；
- 但 A0 step-200 final mean loss 比最早稳定的 step 50 **低 5.651%**，
  non-worsened 数只从 `6` 降到 `5`，没有达到预注册的“下降至少 2 条且 mean
  loss 增加”；
- 因此主要分类是不支持“实质性晚期过训练”：
  `not_supported_no_material_late_degradation`；
- A1 信号随 step 增强，step 150/200 通过绝对门槛；step 200 也通过相对 A0
  的 paired superiority，但 A0 同 step 不稳定，所以没有 joint diagnostic
  candidate。

最准确的机制描述是：

> A0 的逐样本稳定性在 100 update 后开始恶化，但 pooled mean 仍继续改善。
> 现有证据更像 flow-sensitive 的样本间 trade-off，而不是简单的“训练越久，
> 整体 loss 越差”。

这仍是 train-cohort action-loss 工程诊断，不是 future 因果效应或 OOD 成功率
结果。

## 2. 运行有效性

`run_status.json`：

- `status=complete`
- `gate_e7_passed=true`
- `diagnostic_classification=not_supported_no_material_late_degradation`

所有工程检查通过：

- project commit 前后均为
  `703b57fd017d73beed16ab5c9f7484683b62772d`，worktree clean；
- FastWAM commit 前后均为
  `45d8e1458921d83f8ad6cf9ce993d371208dabd0`，worktree clean；
- E.6 root 工件和八个 checkpoint 的 24 个文件前后未变；
- continuity step-200 的 A0/A1 outcome 与 E.6 完全一致；
- primary/continuity RNG identity、完整 grid、zero-weight 位置全部匹配；
- initial Adapter SHA 对 A0/A1 完全相同，zero gate 精确为零；
- Fast-WAM 参数 SHA 前后均为
  `ac0dd59de78495dc8ed5c5601e24951b78f3fd6ad40843d76b83106eccceb4f8`；
- Fast-WAM 无 grad、不可训练；Adapter probe 无 grad；
- 0 backward、0 optimizer、0 checkpoint write；
- 0 development/OOD/success/rollout/future-RGB 输入；
- 8 个 train sample 和 sample payload SHA 与 E.6 完全一致。

因此这是有效的只读诊断，不是工程崩溃或 partial run。

## 3. Primary panel：冻结主结果

Primary 使用结果前冻结的新 flow `6..10`，40 objectives/checkpoint。初始
zero-gate mean action loss 为 `0.007737525469747197`。

| Step | A0 reduction | A0 non-worsened | A0 gate | A1 reduction | A1 non-worsened | A1 gate | A1 mean 优于 A0 | A1≤A0 samples | Joint |
| ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| 50 | 0.874% | 6/8 | pass | 1.295% | 6/8 | fail：<10% | 0.425% | 6/8 | no |
| 100 | 3.628% | 6/8 | pass | 8.560% | 7/8 | fail：<10% | 5.117% | 8/8 | no |
| 150 | 3.880% | 5/8 | fail：A0 sample stability | 11.763% | 8/8 | pass | 8.201% | 8/8 | no |
| 200 | 6.476% | 5/8 | fail：A0 sample stability | 16.870% | 7/8 | pass | 11.114% | 8/8 | no |

四个 step 均满足：

- catastrophic sample `=0`；
- median delta/action-hidden `<0.5`；
- max objective delta/action-hidden `<1.0`。

因此 A0 step 150/200 的唯一失败项都是
`at_least_6_of_8_samples_non_worsened`，不是数值发散或注入尺度超限。

## 4. 预注册 A0 trajectory 判定

冻结规则要求 `late_overtraining_supported` 同时满足：

1. 至少一个 early checkpoint 稳定；
2. step 200 不稳定；
3. 从最早稳定点到 step 200，non-worsened 数下降至少 2；
4. step-200 final mean loss 高于最早稳定点。

观测为：

| 条件 | 结果 |
| --- | --- |
| Early stable | step 50、100 |
| 最早 stable | step 50 |
| Step 200 stable | 否，5/8 |
| Non-worsened drop | `6→5`，只下降 1 |
| Step-200 minus step-50 mean | `−0.0004333992` |
| Step-200 mean 相对 step 50 | 低 5.651% |

所以条件 3、4 均不成立，冻结分类必须是：

```text
not_supported_no_material_late_degradation
```

“不支持”不是证明不存在任何晚期变化。两个 early step 通过、两个 late step
失败，说明逐样本稳定性确实在 step 100 后发生变化；但它没有表现为预注册所要求
的实质 pooled-loss 退化。

## 5. 逐样本 A0 线索

Primary step 200 中三条变差样本为：

| Demonstration | Final / initial loss |
| --- | ---: |
| `episode_000010` | 1.1292× |
| `episode_000011` | 1.0194× |
| `episode_000012` | 1.1814× |

时间轨迹：

- `episode_000012` 在四个 checkpoint 均变差，且 ratio 逐步增大；
- `episode_000010` 从 step 100 起变差；
- `episode_000011` 从 step 150 起变差；
- 其余样本的改善足以让 pooled mean 从 step 50 到 200 持续下降。

这解释了为什么 mean reduction 与 6/8 stability 会给出不同判断：平均目标在
改善，但改善并非均匀分布到全部 demonstration。

A1 的样本稳定性明显更好：

- step 150 为 `8/8` 不变差；
- step 200 为 `7/8`，唯一变差的 `episode_000011` 只高 `0.302%`；
- 但 matched A0 同 step 未通过，所以不能单独选择 A1。

## 6. Continuity panel：描述性结果

Continuity 使用已知 E.6 endpoint 的旧 flow `1..5`，只允许做 exact reproduction
与描述，不参与主要分类。

| Step | A0 reduction | A0 non-worsened | A1 reduction | A1 non-worsened |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 1.343% | 7/8 | 1.176% | 8/8 |
| 100 | 2.238% | 6/8 | 7.825% | 7/8 |
| 150 | 1.583% | 4/8 | 14.966% | 8/8 |
| 200 | 1.191% | 4/8 | 14.842% | 7/8 |

该 panel 的描述性 A0 分类为 `late_overtraining_supported`：

- 最早 stable step 为 50；
- non-worsened 从 `7→4`；
- step-200 mean 比 step 50 高 `5.19e-06`，即约 `0.154%`。

它与 primary 的共同点是 A0 都在 step 50/100 通过、150/200 失败；差别在于旧
flow 上下降 3 条 sample 且 mean 极小幅变差，新 flow 上只下降 1 条且 mean
明显改善。

因为 continuity endpoint 在预注册前已知，且 primary 没有复现其 material
degradation，该描述性分类不能覆盖主要结论。两个 panel 的差异反而说明五个 flow
draw 对 8-sample stability 判定仍有明显影响。

## 7. 为什么没有 checkpoint candidate

冻结 joint candidate 要求同一 step 同时满足：

1. A0 stability；
2. A1 absolute；
3. A1-vs-A0 paired superiority。

实际形成错位：

- step 50/100：A0 stable，但 A1 reduction 和 paired mean 优势不足；
- step 150：A1 absolute 通过，但 A0 只有 5/8，paired mean 优势也只有 8.201%；
- step 200：A1 absolute 和 paired superiority 都通过，但 A0 仍只有 5/8。

因此：

```text
diagnostic_candidate_steps=[]
selection_status=no_joint_diagnostic_candidate
```

不能事后选择 step 100，也不能仅因 step 200 的 A1 强信号而跳过 A0。

## 8. 资源

| 指标 | 结果 |
| --- | ---: |
| 总 wall time | 838.765 s / 13.98 min |
| 模型加载 | 400.233 s / 6.67 min |
| 模型加载峰值 | 23,679.513 MiB |
| 数据准备峰值 | 13,002.816 MiB |
| Probe 最大峰值 | 12,945.219 MiB |
| Forward objectives | 800 |
| Backward / optimizer step | 0 / 0 |

实际耗时落在预注册的 12–18 分钟估计内。

## 9. 权威工件

输出根：
`outputs/thought3/phase_e7_checkpoint_trajectory_v1/`

| 工件 | SHA-256 |
| --- | --- |
| `gate_e7_result.json` | `9b242a3a38638cf2f67c31dd343af0e0d1ec39941d3e784dcd3e167bf14baa4b` |
| `run_status.json` | `207dc70a5a83bd67787f038559a4262708b9fb4e355f628cbc6cca90a162e125` |
| `pre_validation_result.json` | `cbe4bf697c07307bca3f9708fefd235160ccb6bcf355920c85913ac979616b5f` |
| `data_preparation.json` | `f6635c8d0e80d052ad06ce5848bbd2d2ee14635fd0594d44095ccc3461a57fc4` |
| `logs/phase_e7.log` | `e32a9bbbd74582f39d4593f851235e29c6145dd01b6c4cd3188f77ac8a78d899` |

Config fingerprint：
`3823a3403e2d94c4690cf210209e1b530388722446fe64220a79560c18209af2`

## 10. 结论边界与下一步

当前允许写：

- A0 primary stability 在 step 50/100 通过，在 150/200 失败；
- 预注册的 material late-overtraining pattern 未被 primary 支持；
- A1 improvement 随 step 增强，但没有任何 step 同时通过三组 joint 门槛；
- continuity 与 primary 的差异表明稳定性判断对 flow panel 敏感。

当前禁止写：

- A0 没有晚期变化；
- step 100 是最佳 checkpoint；
- step 200 A1 已可进入 OOD；
- future 已改善 action、ID/OOD success；
- continuity 的已知 endpoint 可以覆盖 primary 结论。

最省算力的后续不是立即改 optimizer，而是先预注册一个**全新、更大的 flow
replication panel**，检验 A0 5/8 与 panel 分歧是否来自五个 flow draw 的方差。
该实验必须：

- 使用尚未评估的 flow slots；
- 在结果前冻结 exact RNG identity 和 zero-weight 位置；
- 明确为 E.7 结果后的 sequential diagnosis；
- 不把 primary/continuity 合并统计事后升级为 E.7 主结论；
- 即使产生 candidate，也要在尚未使用的 train cohort 上独立复验。

在更大 flow panel 仍确认 A0 不稳定后，才进入 matched A0/A1 的单变量优化诊断，
例如对 per-sample tail risk/稳定性进行约束。完整 Gate E、A2/A4 和 OOD 继续
锁定。

2026-07-29 更新：上述后续现已冻结为 E.8，采用 A0 step 100/200、全新 flow
`11..74`、双 32-flow block、paired bootstrap 和三种互斥分类；尚未运行。详见
[thought3_phase_e8_protocol.md](thought3_phase_e8_protocol.md)。
