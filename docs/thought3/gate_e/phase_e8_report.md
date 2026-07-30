# Thought3 Gate E.8：A0 大规模 Flow-Variance Replication 结果

状态：**COMPLETED / ENGINEERING PASS / MIXED OR INCONCLUSIVE**

运行日期：2026-07-29

预注册 commit：
`e6ebdf23650a9feb80461251926095621dff6d7d`

输出根：
`outputs/thought3/phase_e8_a0_flow_variance_replication_v1/`

## 1. 结论摘要

E.8 已完整跑完，`gate_e8_passed=true`，但这表示工程门禁通过，不表示某个科学
假设被支持。严格按预注册规则，主要分类为：

```text
diagnostic_classification = mixed_or_inconclusive
binary_answer              = inconclusive
onset_subclassification    = already_present_by_step100
```

两个简单解释都没有被支持：

1. **不能称为 persistent target tail risk supported**：E.7 预识别的三条 target
   中，step 200 只有 `episode_000012` 通过 full panel、两个 32-flow block 和
   Bonferroni bootstrap 的共同确认，低于冻结的 `2/3` 门槛。
2. **也不能称为 five-flow panel variance supported**：step 200 仍有两条
   sample 被确认恶化，且 full/Block A/Block B 的原 A0 sample-stability Gate
   分别只有 `4/8`、`4/8`、`5/8`，没有全部通过。

最准确的解释是：**E.7 的五-flow 失败有明显 panel 方差成分，但不是纯统计噪声。**
A0 step 200 的总体平均 action loss 在 64 个新 flow 上下降 `3.728%`，两个独立
block 也分别下降 `3.472%` 和 `4.047%`；与此同时，`episode_000012` 的
`+8.881%` 恶化稳定跨越两个 block，且校正后的单侧 bootstrap 下界仍为
`+4.679%`。这构成同一 demonstration 内的稳定 sample-level harm，但不能外推
到 LIBERO demonstration 总体，更不能证明 future/A1 或 OOD success 有改善。

## 2. 工程有效性

| 项目 | 结果 |
| --- | --- |
| Run status | `complete` |
| Engineering Gate | `true` |
| Forward objectives | `1,536 / 1,536` |
| Backward / optimizer / 新训练 | `0 / 0 / 0` |
| Checkpoints | 只读 A0 step 100/200 |
| Flows | 全新 `11..74`，64 flows/sample |
| Block A / B | `11..42` / `43..74` |
| Bootstrap / five-flow resampling | `20,000 / 20,000` |
| Frozen Fast-WAM SHA before/after | 均为 `ac0dd59d...ceb4f8` |
| Future RGB / actual future | `0 / 未读取` |
| Development / OOD / success / rollout | `0 / 0 / 0 / 0` |
| Model load peak | `23,679.513 MiB` |
| Probe peak | `12,945.219 MiB` |
| 总 wall time | `1,110.340 s / 18.51 min` |

以下 cross checks 全部为 `true`：

- initial/final probe grid、RNG identity、zero-weight 位置与 exact-zero loss；
- E.7 父工件和 E.6 checkpoint 文件运行前后不变；
- Fast-WAM 无 gradient、不可训练且参数 SHA 不变；
- 未创建 optimizer、未调用 backward、未写 checkpoint；
- project/FastWAM provenance 在运行中不变；
- 只解码 16 张当前观测相机帧，future RGB 为 0。

因此本次结果是**有效的只读诊断**，不是工程崩溃、partial run 或训练结果。

## 3. Full panel 与两个独立 block

正的 reduction 表示 final action loss 低于 zero-gate initial。

| Checkpoint | Panel | Initial mean | Final mean | Reduction | Non-worsened | 原 A0 Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| step 100 | Block A, 32 flows | 0.0050675 | 0.0049343 | 2.629% | 7/8 | PASS |
| step 100 | Block B, 32 flows | 0.0040646 | 0.0039386 | 3.099% | 7/8 | PASS |
| step 100 | Full, 64 flows | 0.0045660 | 0.0044365 | 2.838% | 7/8 | PASS |
| step 200 | Block A, 32 flows | 0.0050675 | 0.0048916 | 3.472% | 4/8 | FAIL |
| step 200 | Block B, 32 flows | 0.0040646 | 0.0039001 | 4.047% | 5/8 | FAIL |
| step 200 | Full, 64 flows | 0.0045660 | 0.0043958 | 3.728% | 4/8 | FAIL |

step 200 三个 panel 唯一失败的 performance check 都是
`at_least_6_of_8_samples_non_worsened`。其 pooled mean、catastrophic、
median/max hidden-correction 尺度门槛均通过：

- catastrophic sample：三个 panel 均为 `0`；
- full median delta/action-hidden：`0.0744`，门槛 `<=0.5`；
- full max objective delta/action-hidden：`0.1116`，门槛 `<=1.0`。

这说明当前矛盾不是总体 loss 发散或 correction 尺度爆炸，而是总体改善与
逐样本符号异质性同时存在。

## 4. Step-200 逐样本结果

`Relative change > 0` 表示比 zero-gate initial 更差。CI 为 20,000 次
paired-flow bootstrap、16 comparisons Bonferroni 校正后的单侧界。机器 ID
在表中缩写，完整 ID 保存在 `gate_e8_result.json`。

| Episode | ID 前 8 位 | E.7 target | Full change | Block A | Block B | 校正界 | Worsened flows | 判定 |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| `episode_000014` | `9610d2ae` | 否 | +2.050% | +3.115% | +0.878% | lower −0.507% | 92.19% | 未确认 |
| `episode_000010` | `75359438` | 是 | +0.261% | +0.604% | −0.201% | lower −4.991% | 85.94% | 未确认 |
| `episode_000011` | `5f82a5db` | 是 | −4.272% | −5.193% | −3.547% | upper −1.910% | 26.56% | 确认改善 |
| `episode_000030` | `8f34793b` | 否 | −10.282% | −8.978% | −11.626% | upper −6.066% | 1.56% | 确认改善 |
| `episode_000019` | `8c00174e` | 否 | −3.905% | −4.051% | −3.775% | upper −2.822% | 23.44% | 确认改善 |
| `episode_000038` | `461a673f` | 否 | −13.292% | −13.143% | −13.534% | upper −7.835% | 1.56% | 确认改善 |
| `episode_000000` | `739baab4` | 否 | +1.603% | +1.586% | +1.637% | lower +0.342% | 73.44% | **确认恶化** |
| `episode_000012` | `81363fef` | 是 | +8.881% | +8.517% | +9.320% | lower +4.679% | 98.44% | **确认恶化、material** |

两条 confirmed-worsened sample 中：

- `episode_000012` 是 E.7 预识别 target，且 `+8.881% >= 2%`；它是强且稳定的
  sample-level tail signal。
- `episode_000000` 不是 E.7 target，虽然统计确认，但幅度 `+1.603% < 2%`；
  它不能替换 target 来满足预注册的 `2/3` 主要规则。

另外两条 full point estimate 为正的 sample 没有通过校正确认。因此原
`non-worsened >=6/8` 点符号门槛会把“明确 harm”和“CI 跨 0 的小幅波动”都计为
同一种失败；E.8 不追溯修改该门槛，但后续新协议应明确区分二者。

## 5. 三条预识别 target 的复验

| Target | Step 100 full change | Step 100 判定 | Step 200 full change | Step 200 判定 |
| --- | ---: | --- | ---: | --- |
| `episode_000010` | −4.673% | 确认改善 | +0.261% | 未确认 |
| `episode_000011` | −2.227% | 确认改善 | −4.272% | 确认改善 |
| `episode_000012` | +4.034% | 确认恶化 | +8.881% | 确认恶化 |

step 200 只有 `1/3` target 被确认恶化，所以
`persistent_target_tail_risk_supported` 不成立。该唯一确认 target 在 step
100 已经恶化，因此 onset 子分类为 `already_present_by_step100`，不是
`late_emergent_after_step100`。这也进一步说明不能把 step 100 简单选成“安全的
最佳 checkpoint”。

## 6. Five-of-64 sensitivity

从同一 64-flow 工件中无放回抽 5 个 slots，重复 20,000 次：

| Checkpoint | 五-flow Gate pass | Gate fail | Stability fail | Pooled mean worsening | Reduction p05 / p50 / p95 |
| --- | ---: | ---: | ---: | ---: | --- |
| step 100 | 66.965% | 33.035% | 32.930% | 1.220% | 0.737% / 2.967% / 4.743% |
| step 200 | 13.630% | 86.370% | 86.360% | 3.555% | 0.310% / 3.800% / 7.319% |

step 200 的 20,000 个五-flow panel 中，`96.445%` 的 pooled mean 仍改善，但
`86.360%` 因逐样本 `6/8` 稳定性失败。这表明：

- E.7 的五-flow Gate failure 在当前 64-flow empirical distribution 下并不是
  罕见抽样事件；
- 失败几乎完全由 sample sign-count 驱动，而不是 pooled mean 变差；
- 但 full 64-flow 和两个 32-flow block 仍存在两条 confirmed harm，因此不能
  把 E.7 结果全部归因于小 panel 方差。

## 7. 允许与禁止的结论

当前允许写：

- E.8 工程 Gate 通过，完成 1,536 个只读 forward、0 训练；
- A0 step 200 的 pooled mean 在 full/两个 block 都改善约 3.5%–4.0%；
- 原 `6/8` sample-stability Gate 在三个 step-200 panel 均失败；
- 一条预识别 target 与一条非 target 在严格校正后确认恶化；
- `episode_000012` 的 harm 跨 64 flows 和两个 block 稳定存在；
- 主要分类为 `mixed_or_inconclusive`，五-flow 方差和真实 sample-level
  heterogeneity 同时存在。

当前禁止写：

- A0 instability 只是五-flow 噪声；
- E.7 三条 target 构成已确认的普遍 tail risk；
- step 100 是可选的最佳 checkpoint；
- A1/future 已修复上述 harm；
- 当前八条 sample 能代表全部 LIBERO demonstration；
- action-loss tail 等于 OOD failure tail；
- full E、A2/A4 或在线 OOD 已解锁。

## 8. 下一步决策

不建议继续在同八条 sample 上增加更多 flow，64-flow full panel和两个独立
32-flow block 已经把 flow-level 问题解析到足以看到“总体改善 + 个别稳定 harm”
的混合结构。继续堆叠同 cohort 阈值会增加结果后自由度，却不能增加
demonstration-level 证据。

下一步应先冻结 E.8 为 mixed 结果，再预注册一个**单变量、matched A0/A1 的
sample-tail mitigation**。建议方向是对每条 sample 相对 zero-gate baseline 的
恶化加入同构的 non-worsening/trust-region penalty，并保持：

- A0/A1 相同 Adapter、初始化、数据、flow schedule、更新数与 action denoise；
- 配方开发只使用已经用过的 cohort；
- Phase D train 排序 `17–28` 保留为一次性 demonstration-level 独立复验；
- 新 Gate 同时报告 pooled mean、点符号 `6/8` 和校正后的 confirmed harm，
  不追溯改判 E.6/E.7/E.8；
- 只有 matched A0/A1 在独立 cohort 形成稳定候选，才进入新的完整 28/4 Gate E。

该建议尚未预注册、没有脚本、也没有运行；在冻结新的 loss、系数、flow slots、
预算和判据前，不应启动 GPU 训练。

## 9. 权威工件

| 工件 | SHA-256 |
| --- | --- |
| `gate_e8_result.json` | `e3809eedaadc4eb7ce4c681151214f01304e08b0a45cd3bccf926ed003c989e1` |
| `run_status.json` | `03e9039b078ef5cd34c2a97d55b5d25fec29937959aff29c4dd322956ce8f53a` |
| `pre_validation_result.json` | `1a46e92af902e1613a87a4644912326f184b1517c289ea04d0d0becab8d6bc04` |
| `data_preparation.json` | `abdb800855e3bdedc5f8e9e267e5c7e1cef030050b88a32cd58ecdf81c983828` |
| `logs/phase_e8.log` | `68eda4a7b131a9cb82209df2c56ac67877ffc3dff564682877026d0abdc9743c` |

Config fingerprint：
`ed587c61cec3e386e5b44af11fca646dab527acbe46cce34d6badfd34ff09f7f`

Flow identity SHA：
`710b809614aeb502c944275c4c43759d2383b00e52fd9d5216898fb949b5772a`
