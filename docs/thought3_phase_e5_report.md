# Thought3 Gate E.5：Full-cohort Objective Aggregation 诊断报告

状态：**Gate E.5 有效失败；六条训练轨迹完整**
验收日期：2026-07-29
证据等级：`ENGINEERING DIAGNOSTIC / FAILED-GATE / NOT MODEL EFFECT`
Run ID：`P3-PHASE-E5-v1`

## 1. 结论

Gate E.5 已按冻结协议完整执行：

```text
3 learning rates × 2 variants × 200 optimizer updates
= 1,200 Adapter-only optimizer updates

6 tracks × 200 updates × 8 objectives/update
= 9,600 train action-flow objectives

6 tracks × step {0,200} × 8 samples × 5 held-out flows
= 480 held-out action-loss objectives
```

六条轨迹的训练、梯度聚合、checkpoint、恢复 provenance、held-out probe、
显存、zero-weight objective、配对 schedule 和冻结 Fast-WAM 检查全部通过。
命令最后的非零退出不是程序执行故障，而是完整结果落盘后按预注册性能门主动
fail-closed：

```text
lr_1e_04 = false
lr_3e_04 = false
lr_1e_03 = false
selected_lr_slug = null
gate_e5_passed = false
```

最强单条轨迹是 `A1@3e-4`：

```text
held-out mean loss reduction = 19.668%
non-worsened samples         = 8/8
catastrophic samples         = 0/8
```

它单独通过了五项 performance check；但同一 LR 的 A0 只下降 2.638%，未达到
冻结的 10% 门槛。E.5 的选择规则要求同一 LR 下 A0/A1 都通过，因此不能事后
选择 `3e-4`，也不能把总 Gate 改判为通过。

E.5 支持的最窄结论是：

> 在 matched optimizer-update、每次更新聚合完整八样本 cohort 的配方下，
> A1 的 held-out action-loss 信号明显强于 E.4，尤其是 `3e-4`；但 A0/A1
> 没有形成预注册的共同候选 LR，因此完整 Gate E、A2/A4 和在线 OOD 仍保持锁定。

这不是“future 已改善 OOD”的证据，也不是“future 无效”的证据。

## 2. 冻结运行身份

| 字段 | 值 |
| --- | --- |
| 预注册 commit | `a8245d11c145816e018dddb06a3693c70a2168fa` |
| config | `configs/thought3/phase_e5_objective_aggregation_diagnostic.yaml` |
| config fingerprint | `c4c681534cf4c143a1675c24ded719b7bf0a4c2964b2384704b4338e147122fc` |
| schema | `thought3.phase_e5.objective_aggregation.v1` |
| source Gate E.4 result SHA | `48314003c146327c93e3c5ecb173762cde09c27afb1b38124e741a222e974240` |
| samples / tracks | `8 / 6` |
| optimizer updates / objectives per track | `200 / 1,600` |
| train flow slots | `20001..21600` |
| held-out flow steps | `[1,2,3,4,5]` |
| Adapter parameters | `1,371,137` |
| Adapter fingerprint | `7c636482574a42165eb752b18a637b81668282d21c48f6533e47f0b1884ab2fd` |
| initial Adapter SHA | `77974a49c3d14fac142322244cc3613dccf0a329a25faa6e7053d99345ae627f` |
| frozen identity schedule SHA | `b6f9778d303a6ad2c4bef781f4a6027a800d013814110daa47eb7cb1d13af86d` |
| observed schedule SHA | `f84f664bce9c6f2e5d98e9b79b5e1761122514cf92e4db92bd9ed16400d2d682` |
| model dtype | `torch.bfloat16` |

冻结 Fast-WAM 参数 SHA：

```text
before = ac0dd59de78495dc8ed5c5601e24951b78f3fd6ad40843d76b83106eccceb4f8
after  = ac0dd59de78495dc8ed5c5601e24951b78f3fd6ad40843d76b83106eccceb4f8
```

## 3. 主结果

六条 initial held-out probe 的 sample-equal mean action loss 精确相同：

```text
0.005565503754223755
```

| LR | Variant | Final mean loss | Reduction | Non-worsened | Catastrophic | Median delta/hidden | Max delta/hidden | Track performance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `1e-4` | A0 | 0.00546036 | **1.889%** | 6/8 | 0/8 | 0.01370 | 0.01615 | fail：reduction |
| `1e-4` | A1 | 0.00520642 | **6.452%** | 7/8 | 0/8 | 0.01311 | 0.02058 | fail：reduction |
| `3e-4` | A0 | 0.00541867 | **2.638%** | 7/8 | 0/8 | 0.04069 | 0.04800 | fail：reduction |
| `3e-4` | A1 | 0.00447086 | **19.668%** | 8/8 | 0/8 | 0.06869 | 0.14167 | **individual pass** |
| `1e-3` | A0 | 0.00540463 | **2.890%** | 4/8 | 0/8 | 0.04308 | 0.05083 | fail：reduction、6/8 |
| `1e-3` | A1 | 0.00515836 | **7.315%** | 5/8 | 0/8 | 0.07064 | 0.10217 | fail：reduction、6/8 |

所有轨迹都满足：

- 0/8 catastrophic sample；
- median sample-mean `delta/action-hidden ≤ 0.5`；
- max sample-mean `delta/action-hidden ≤ 1.0`；
- 200/200 update、1,600/1,600 train objective 和 40+40 held-out objective
  完整；
- optimizer 只包含 Adapter；
- update 1 只有 gate gradient，update 2 的 projector、attention 和 non-gate
  gradient finite、nonzero；
- checkpoint round-trip、schedule、完整八样本 cohort、memory 和 frozen SHA
  全部通过。

真正的共同门禁失败点是：

- 三个 A0 都未达到 10% held-out mean-loss reduction；
- `A1@1e-4` 和 `A1@1e-3` 也未达到 10%；
- `1e-3` 的 A0/A1 还分别只有 4/8、5/8 sample 不变差。

## 4. E.4 与 E.5 对照

| LR | Variant | E.4 reduction | E.5 reduction | 变化 | Non-worsened |
| --- | --- | ---: | ---: | ---: | ---: |
| `1e-4` | A0 | 1.643% | 1.889% | +0.246 pp | 5/8 → 6/8 |
| `1e-4` | A1 | 1.751% | 6.452% | +4.701 pp | 3/8 → 7/8 |
| `3e-4` | A0 | 1.702% | 2.638% | +0.936 pp | 5/8 → 7/8 |
| `3e-4` | A1 | 1.787% | 19.668% | +17.881 pp | 7/8 → 8/8 |
| `1e-3` | A0 | 1.948% | 2.890% | +0.943 pp | 4/8 → 4/8 |
| `1e-3` | A1 | 0.997% | 7.315% | +6.319 pp | 4/8 → 5/8 |

六条 E.5 reduction 都高于对应 E.4，其中 A1 的增幅明显更大。这与
“完整 cohort aggregation 能保留更稳定的 A1 优化信号”一致，但不能把差异
唯一归因于梯度聚合：

- E.4 与 E.5 都是 200 次 AdamW update；
- E.5 每条轨迹看了 1,600 个 objective，是 E.4 的 8 倍；
- 因此两者不是 compute-matched 或 objective-count-matched。

“聚合方式”和“训练 objective 暴露量”在该跨 Gate 对照中不能完全分离。

## 5. A1 相对 A0：仅作探索性工程信号

同一 LR 的 A0/A1 使用完全相同的初始化、参数量、sample、slot、noise、
timestep、预算和 held-out probe。描述性配对结果为：

| LR | A1 相对 A0 的 reduction 差 | A1 final loss 更低的 sample | Mean(A0 final − A1 final) |
| --- | ---: | ---: | ---: |
| `1e-4` | +4.563 pp | 7/8 | 0.00025394 |
| `3e-4` | +17.030 pp | 8/8 | 0.00094782 |
| `1e-3` | +4.425 pp | 4/8 | 0.00024627 |

`A1@3e-4` 的 8/8 paired direction 是后续独立复验的最强候选信号，但它不是
本 Gate 预注册的模型效应 estimand，且 LR 是在看过三档结果后才显得突出。
当前只有一个 task 的八条 train sample，没有 development、rollout 或 OOD。
因此不得写成：

- “K=1 future 已优于 K=0”；
- “future latent 已提高策略成功率”；
- “3e-4 已是可用于 A2/A4 的正式 LR”。

## 6. 梯度与 scalar gate 遥测

以下是看到结果后的探索性诊断，不参与 Gate 判定。

每个 update 的 cancellation ratio 定义为：

```text
abs(sum of 8 mean-scaled gate-gradient contributions)
-------------------------------------------------------
sum of abs(8 mean-scaled gate-gradient contributions)
```

值越接近 0 表示 cohort 内抵消越强，越接近 1 表示方向越一致。

| LR | Variant | Mean within-update cancellation ratio | Global 200-update gate-gradient coherence | Final gate raw |
| --- | --- | ---: | ---: | ---: |
| `1e-4` | A0 | 0.618 | 0.176 | −0.004213 |
| `1e-4` | A1 | 0.619 | 0.422 | −0.007543 |
| `3e-4` | A0 | 0.603 | 0.069 | −0.005043 |
| `3e-4` | A1 | 0.576 | 0.344 | −0.013457 |
| `1e-3` | A0 | 0.637 | 0.027 | −0.003659 |
| `1e-3` | A1 | 0.676 | 0.052 | −0.006176 |

E.5 没有消除 micro-objective 抵消，但每次 update 都保留了 finite、nonzero
聚合梯度。`3e-4` 下 A1 的跨 200-update gate-gradient coherence 约为 A0 的
5 倍，final gate magnitude 约为 A0 的 2.7 倍；这与 A1 的较大 held-out
correction 和 loss reduction 同向。

这仍不能证明根因是 future 信息，因为 A0/A1 的 token 内容不同会直接改变
梯度场；它只说明当前日志中没有“梯度断链”或“scalar gate 完全不开”的工程
证据。此时立刻改变 gate parameterization 会同时丢失对这个信号的原配方复验。

## 7. 执行、配对与泄漏边界

```text
optimizer updates                       1,200
train objective rows                    9,600
held-out forward objectives               480
checkpoint directories                     24
per-track execution checks true          120/120
paired checks true                         21/21
cross checks true                           7/7
train zero-weight objectives              144
development outcomes read               false
OOD/success outcomes read                false
future RGB frames read                       0
uses ground-truth future                 false
rollout started                          false
```

每条轨迹的 24 个预知 `t=1000/weight=0` train objective 均被保留，并满足
official weight 0 → exact weighted loss 0。六条轨迹的 observed schedule SHA
完全相同。

`pre_validation_result.json` 的 `execution_error` 和 `execution_traceback`
均为 `null`。根 `run_status.json` 的 traceback 只来自结果完整写入后主动抛出的
failed-gate 状态。

六条轨迹记录的 manifest、objective metrics、update metrics、probe metrics
和 state 共 30 个 SHA 已逐文件复算，全部匹配 root result。

## 8. 资源

| 指标 | 值 |
| --- | ---: |
| Gate wall time | 6,878.881 s（1 小时 54 分 38.9 秒） |
| Model load time | 546.542 s（9 分 6.5 秒） |
| 六条 track wall sum | 6,080.086 s |
| 单 track mean wall | 1,013.348 s（16 分 53.3 秒） |
| Mean optimizer update | 4,961.190 ms / 8 objectives |
| Model-load peak | 23,679.513 MiB |
| Train/probe peak | 13,277.440 MiB |
| 输出大小 | 413,198,197 bytes（`du -sh` 为 395 MiB） |
| 输出文件数 | 113 |

这里的 held-out objective latency 和 optimizer-update latency 不是机器人在线
动作推理延迟，不能填入最终 K–latency 曲线。

## 9. 冻结工件

权威目录：

```text
outputs/thought3/phase_e5_objective_aggregation_v1/
```

| 工件 | SHA-256 |
| --- | --- |
| `gate_e5_result.json` | `c797a98f646855a9b37caa7e251c97e8001d2d4aecb7efbcb5a539f77911f7bd` |
| `run_status.json` | `cdc5944d35a03309230206ef817b75b17c1dbdea4b8f1706b98c1e7cec514f37` |
| `pre_validation_result.json` | `63061d304a4a3c77c4e95f782d061be478b2e03a7dd88a39de8861f8ccde63ae` |
| `data_preparation.json` | `ef95e5972ccabc455e7781afae19582f4f7880eb9e8800f0cd3e0a152f7261b6` |
| `logs/phase_e5.log` | `fc334690b893555c09d36a2eb288e562b6e8454531d601570a544b91911d8582` |

Gate E.4 的五个冻结父工件 SHA 在运行前后未改变。

## 10. 停止规则与下一步

E.5 没有共同 eligible LR，因此当前必须：

- 不放宽 10% 或 6/8；
- 不把 `A1@3e-4` 事后改成 E.5 的 selected LR；
- 不延长或覆盖同一 Run ID；
- 不直接进入完整 28/4 Gate E；
- 不实现/训练 A2/A4；
- 不启动 Phase F 或任何 ID/OOD rollout。

同时，`A1@3e-4` 已出现比“继续换 gate/optimizer”更值得先验证的、严格配对
信号。建议下一步先设计一个**新样本、序贯复验 Gate**，而不是立刻改架构：

```text
unused train cohort（development 保持不可见）
  × A0/A1
  × 仅复验 3e-4
  × 新 train-flow slots
  × 相同 200-update / 8-objective arithmetic-mean 配方
```

新协议必须在运行前明确披露 `3e-4` 是由 E.5 探索性选择，并冻结：

1. 新 cohort identity 与其不接触 E.5/开发集的证明；
2. A1 absolute reduction、A0 safety/stability 和 paired A1−A0 contrast 的
   三类独立门槛；
3. 新 flow-slot、noise/timestep schedule SHA；
4. 只有复验通过后，如何建立新的完整 28/4 Gate E；
5. 复验失败后才允许转向 gate parameterization、Adapter normalization 或
   optimizer 的单变量诊断。

该复验目前只是建议，**尚未预注册、实现或获 GPU 运行授权**。
