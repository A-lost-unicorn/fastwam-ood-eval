# Thought3 Phase 2：完整 28/4 A0/A1 训练结果与审计

状态：`COMPLETED / VALID OFFLINE NEGATIVE RESULT / PHASE 3 LOCKED`

运行日期：2026-07-30

证据等级：`VALID ENGINEERING + OFFLINE DEVELOPMENT`。本实验使用一个
`libero_goal` task 的 28 条 train、4 条 development 当前观测和动作监督，
不是机器人 rollout，也没有读取 success、Clean/OOD outcome 或真实 future RGB。

## 1. 结论先行

预注册 Phase 2 有效完成，冻结分类为：

```text
training_valid_dev_direction_not_observed
```

12/12 hard checks 全部通过，但固定 step-200 development 结果不支持 K=1
future 增益：

| 版本 | Initial mean loss | Final mean loss | Reduction |
| --- | ---: | ---: | ---: |
| A0，K=0 | 0.004234104 | 0.004155979 | +1.845% |
| A1，K=1 | 0.004234104 | 0.004306583 | −1.712% |

A1 final mean 比 A0 高 `0.000150604`，即相对 A0 **高 3.624%**；4/4
development sample 的 A1 loss 都高于配对 A0。因此：

```text
development_direction_observed = false
phase3_unlocked = false
next_required_stage =
  stop_before_phase3_and_register_negative_direction
```

结合 Phase 1，当前最重要的研究结论是：

> K=1 future 内容会改变固定 checkpoint 的动作输出，但这种技术敏感性没有在
> 本次冻结的轻量 Adapter 配方中转化为更低的 held-out action objective。

这支持“future 影响动作不等于 future 对控制有用”，但不支持“未来想象在所有
Fast-WAM、task 或 OOD 环境中都无用”。

## 2. Run identity

| 项目 | 冻结/实测值 |
| --- | --- |
| 输出 | `outputs/thought3/phase2_full_28_4_a0_a1_v1/` |
| 项目运行 commit | `8e41d0b701fe240b7fd0b0cf5a9c7cb20cbb08be` |
| Fast-WAM commit | `45d8e1458921d83f8ad6cf9ce993d371208dabd0` |
| Config fingerprint | `fabb96a97b7e137ca39a5477c2090deab1844909887b10cd22fa92ebbee66468` |
| Split fingerprint | `ea5402955023ccd48d790d821a73f98549b31d1ace8af035a90ceae2ad3951eb` |
| Cache fingerprint | `63a70e1af38f68bc894fc11d03c84f212e6c6328a5051256c9d045741156d9c5` |
| Task | `open the middle drawer of the cabinet` |
| 样本 | 28 train / 4 development，不同 demonstration episode 的 frame 0 |
| 版本 | A0 K=0 / A1 K=1 |
| 训练 | LR 3e-4，seed 3407，200 updates × 28 objectives |
| 主终点 | 固定 step 200，无 checkpoint selection/fallback |
| GPU | A0 物理卡 1；A1 物理卡 2；各进程只见 `cuda:0` |
| 完成时间 | `2026-07-30 07:12:15Z` |

首次 calibration 在 manifest bookkeeping 处中断；修复 commit `8e41d0b`
只统一 path representation。原 1,024 calibration rows 被原目录 `--resume`
复验和复用，数据、flow、weight、训练与终点均未改变。

## 3. 数据量与配对协议

| 阶段 | A0 | A1 | 合计 |
| --- | ---: | ---: | ---: |
| Train calibration | 896 shared | 896 shared | 896 |
| Development initial | 128 shared | 128 shared | 128 |
| Training objective | 5,600 | 5,600 | 11,200 |
| Development final | 128 | 128 | 256 |
| 总 objective |  |  | 12,480 |

关键配对身份：

| 项目 | SHA-256 |
| --- | --- |
| Calibration | `736de93f1957cf383c707293d37cda2813ba6beeada37800e83f0f6637b69fb4` |
| Sample weights | `4c36dece07b1fec2356bc7736b3d11baf2448e581d866c3f4a564fa9e6a1dc22` |
| Identity schedule | `75f58b8863afe52a6a09ec68441a77f710381d16714123c0592ae13e2ffdc048` |
| Train flow schedule | `c5ecd86f47ec5800b4cab9a55fa2b6cd0d68a599c203474f029cdf5a65c91c05` |

A0/A1 使用完全相同的 sample 顺序、normalized weight、action-flow identity
和 `50001..55600` training slots。Development 只在 step 0 和冻结 step 200
评测，没有用中间 checkpoint 选择结果。

## 4. Development 逐样本结果

正 reduction 表示相对 shared initial 改善；最后一列为 A1 相对 A0 的 loss
增幅。

| Sample ID 前 12 位 | Initial | A0 final | A0 reduction | A1 final | A1 reduction | A1 vs A0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `253c241e7c66` | 0.00595233 | 0.00600750 | −0.927% | 0.00604189 | −1.505% | +0.572% |
| `9c54127523fc` | 0.00268135 | 0.00253181 | +5.577% | 0.00281174 | −4.863% | +11.056% |
| `30d329649c06` | 0.00538263 | 0.00521795 | +3.059% | 0.00533706 | +0.847% | +2.283% |
| `08e50da358c` | 0.00292010 | 0.00286665 | +1.831% | 0.00303564 | −3.957% | +5.895% |

A0 相对 initial 为 3/4 sample 改善；A1 只有 1/4 改善。更关键的是 matched
A1-vs-A0 方向为 0/4 获益、4/4 更差。该一致方向是描述性结果；由于只有 4 条
development sample，本报告不额外生成事后显著性检验或放宽冻结判据。

## 5. 训练确实生效

负结果不是梯度断链：

- 两轨都完成 200 updates / 5,600 objectives；
- step 1 只有 zero-init gate 获得非零梯度，符合预期；
- step 2 起 attention、future projector 和其他 non-gate 参数均获得非零梯度；
- 两轨 `first_*_nonzero_gradient_update` 均为 2；
- step-200 raw gate：A0 `0.006218`，A1 `0.016266`；
- optimizer scope 为 `adapter_only`；
- final checkpoint round-trip 的 tensor state 均完全相同；
- frozen Fast-WAM SHA 前后均为 `ac0dd59d...ceb4f8`。

训练 step-200 当批 mean loss 为 A0 `0.00381483`、A1 `0.00347711`，但这是
training-flow 单批值，不能覆盖固定 development 负结果。这也说明不能根据
训练末批 loss 事后挑选 A1。

## 6. 完整性审计

| 检查 | 结果 |
| --- | --- |
| Root status | `complete` |
| A0/A1 track status | `complete / complete` |
| Hard checks | `12/12 true` |
| Manifest descriptor | root 5/5、calibration 5/5、A0 11/11、A1 11/11 |
| Checkpoints | A0/A1 × step 50/100/150/200，8/8 file SHA 与 provenance 通过 |
| Adapter-only round-trip | A0/A1 均 `state_equal=true` |
| Frozen Fast-WAM | 两轨前后 SHA 相同 |
| Matched weight/schedule | A0/A1 SHA 相同 |
| Future RGB | 未读取 |
| OOD/success/rollout | 未读取、未启动 |
| A2/A4 | 未训练 |
| Output tree | 60 files，141,360,940 bytes（134.81 MiB） |

权威工件 SHA：

| 工件 | SHA-256 |
| --- | --- |
| Root artifact manifest | `3f94c33fe14384acf1d7f11259964afe61419559bcdd4435e756038bf8ef6768` |
| Root result | `5ab57efa2747072a14170ef2ecdfc86cfb7bd36528d138cb14b27fdb17f53d93` |
| Root run status | `2584f8622abb5eff05db5d23443b6a019ae83d833e0a03a0541320ab412fac3d` |
| A0 result | `6695fbeea6fd52ac72a3bb5ef77fe151fb5118bc7f24081644841d5119a28e5f` |
| A1 result | `8b3bf8a25610dc7fc255da2b3d1407c3459a3318a99230c27972c081761fec11` |

## 7. 时间、显存与工程成本

| 项目 | A0 | A1 |
| --- | ---: | ---: |
| Model load | 453.60 s | 452.07 s |
| Track invocation | 58.75 min | 58.47 min |
| Mean update time | 17.339 s | 17.252 s |
| Peak allocated | 13,277.44 MiB | 13,277.44 MiB |
| Peak reserved | 13,286 MiB | 13,286 MiB |

双卡 matched tracks 从启动到两轨完成约 69 分钟；恢复 calibration、双轨和
CPU finalize 合计约 78 分钟。模型加载峰值为 `23,679.51 MiB allocated /
23,866 MiB reserved`，在本次设备上可运行。

这里没有重新测量在线 policy latency；不能把 update time、cache sampling time
或 Phase 1 的 `+258.95 ms` 互相替代。

## 8. 冻结决策

预注册 direction 要求同时满足：

1. A1 final development mean < A0；
2. A1 相对 shared initial reduction > 0；
3. 全部工程 hard checks 通过。

实测只有第 3 条成立，所以必须执行冻结停止规则：

- 不运行完整 checkpoint online correct/null/shuffle recheck；
- 不生成 Phase 3 Clean/OOD rollout pilot；
- 不训练 A2/A4；
- 不改 LR、weight、step、sample 或门槛后重跑同一 development；
- 不选择 step 50/100/150；
- 保留并登记负结果。

## 9. 可写与不可写的论文结论

### 可以写

> 在一个 LIBERO-Goal task 的预注册 28/4 matched offline ablation 中，轻量
> K=1 Future-to-Action Adapter 未改善 held-out action objective：K=0 改善
> 1.85%，K=1 恶化 1.71%，且 K=1 final loss 比 K=0 高 3.62%。

> 结合在线动作反事实，future 内容虽然会改变动作，但“被使用”不等于“有益”。

### 不可以写

- Fast-WAM 在 OOD 中不需要未来；
- K=1 会降低机器人成功率；
- A0 在 Clean/OOD rollout 中优于 A1；
- future 对其他 LIBERO task、其他 Adapter、其他 seed 或 K=2/K=4 都无用；
- 4/4 是统计显著或可推广的总体结论；
- 本实验回答了因果 OOD success 效果。

## 10. 对研究路线的意义

这不是工程失败，而是能收束路线的有效负结果：

1. Phase 1：确认 future 内容确实进入动作；
2. Phase 2：确认该影响没有在冻结的 K=1 轻量 Adapter 配方中形成 held-out
   offline objective 增益；
3. 按预注册规则停止本 Adapter 分支，避免继续用 LR、checkpoint、K 或 OOD
   outcome 寻找正结果；
4. 论文主线可以讨论“future sensitivity 与 future utility 的分离”，同时把
   OOD success 因果问题明确列为尚未回答。

简历可使用：

> 设计并执行双 GPU、预注册 matched K=0/K=1 Future-to-Action Adapter
> 消融，完成 11,200 个训练 objective、8 个可恢复 checkpoint 与多层 SHA
> 审计；发现 future-content action sensitivity 未转化为 held-out loss 增益，
> 并依据冻结停止规则保留负结果、阻止 outcome-driven 调参。
