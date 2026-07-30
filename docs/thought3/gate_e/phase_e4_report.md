# Thought3 Gate E.4：Paired Diversified Train-flow 诊断报告

状态：**Gate E.4 有效失败；六条训练轨迹完整**
验收日期：2026-07-28
证据等级：`ENGINEERING DIAGNOSTIC / FAILED-GATE / NOT MODEL EFFECT`
Run ID：`P3-PHASE-E4-v1`

## 1. 结论

Gate E.4 已按冻结协议完成：

```text
3 learning rates × 2 variants × 200 optimizer steps
= 1,200 Adapter-only optimizer steps

6 tracks × step {0,200} × 8 samples × 5 held-out flows
= 480 held-out action-loss forward objectives
```

六条轨迹的训练、梯度、checkpoint、resume provenance、held-out probe、显存、
zero-weight objective、配对 schedule 和冻结 Fast-WAM 检查全部通过。本次非零
进程退出码来自预注册性能门槛失败，不是执行异常：

```text
lr_1e_04 = false
lr_3e_04 = false
lr_1e_03 = false
selected_lr_slug = null
gate_e4_passed = false
```

把训练 objective 从每条 sample 永久复用 `flow_step=0` 改为 200 个唯一 flow
slot 后，六条轨迹在 held-out flow 上都由 E.3 的部分负降幅变为约
`0.997%～1.948%` 的正降幅；`A1@3e-4` 也达到 7/8 sample 不变差。但所有轨迹
都远低于冻结的 10% mean-loss reduction，且没有一个 LR 让 A0/A1 同时满足
`10% + 6/8`。

因此，E.4 支持的最窄结论是：

> diversified train-flow 缓解了 fixed-flow checkpoint 的跨 flow 退化，但在
> 当前 scalar-gated Adapter、microbatch 1、200-step 配方下，改善幅度不足以
> 形成可进入完整 Gate E 的稳定训练信号。

它不能证明 future latent 无效，也不能回答 ID/OOD success、K 排序或
future-to-action 因果增益。

## 2. 冻结运行身份

| 字段 | 值 |
| --- | --- |
| 预注册 commit | `07d949d2fb2f9ae88b3b8ccd4ed4e656b8cea085` |
| config | `configs/thought3/phase_e4_diversified_flow_diagnostic.yaml` |
| config fingerprint | `e8c67a088c2c78e85e86c0cc0fac011e23303c59559d98c44dbc7051bdf578d1` |
| schema | `thought3.phase_e4.diversified_flow.v1` |
| source Gate E.3 result SHA | `517c1e0cfc198f0bc44ab03d0d59349f20131d5c00efd958dd10f67aee1defe3` |
| samples / tracks | `8 / 6` |
| train flow slots | `10001..10200` |
| held-out flow steps | `[1,2,3,4,5]` |
| Adapter parameters | `1,371,137` |
| initial Adapter SHA | `77974a49c3d14fac142322244cc3613dccf0a329a25faa6e7053d99345ae627f` |
| shared train schedule SHA | `1733c2298334f47a3d57689161e1c72aa0acd917099cb56895fa20498362c9d4` |
| model dtype | `torch.bfloat16` |

冻结 Fast-WAM 参数 SHA：

```text
before = ac0dd59de78495dc8ed5c5601e24951b78f3fd6ad40843d76b83106eccceb4f8
after  = ac0dd59de78495dc8ed5c5601e24951b78f3fd6ad40843d76b83106eccceb4f8
```

## 3. 主结果

所有 A0/A1 initial held-out probe 的 sample-equal mean loss 精确相同：

```text
0.005565503754223755
```

| LR | Variant | Final mean loss | Reduction | Non-worsened | Catastrophic | Median delta/hidden | Max delta/hidden | Eligible |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `1e-4` | A0 | 0.00547408 | **1.643%** | 5/8 | 0/8 | 0.00540 | 0.00632 | false |
| `1e-4` | A1 | 0.00546803 | **1.751%** | 3/8 | 0/8 | 0.00426 | 0.00663 | false |
| `3e-4` | A0 | 0.00547077 | **1.702%** | 5/8 | 0/8 | 0.00590 | 0.00691 | false |
| `3e-4` | A1 | 0.00546602 | **1.787%** | 7/8 | 0/8 | 0.00367 | 0.00533 | false |
| `1e-3` | A0 | 0.00545709 | **1.948%** | 4/8 | 0/8 | 0.00862 | 0.01014 | false |
| `1e-3` | A1 | 0.00551004 | **0.997%** | 4/8 | 0/8 | 0.00273 | 0.00347 | false |

六条轨迹都通过：

- 0/8 catastrophic sample；
- median `delta/action-hidden ≤ 0.5`；
- max sample-mean `delta/action-hidden ≤ 1.0`；
- 200/200 train metric 与 40/40 initial、40/40 final probe 完整；
- optimizer 只包含 Adapter；
- step 1 只有 gate gradient，step 2 projector/attention/non-gate gradient
  finite、nonzero；
- checkpoint round-trip、schedule、sample round-robin、memory 和 frozen SHA。

六条轨迹都未通过：

- held-out sample-equal mean loss 至少下降 10%。

另外，除 `A1@3e-4` 外，其余五条至少还未通过 6/8 sample 不变差。因为共同 LR
要求 A0/A1 同时通过，所以 `A1@3e-4` 的 7/8 不能单独成为候选。

## 4. E.3 与 E.4 对照

| LR | Variant | E.3 fixed-flow checkpoint held-out reduction | E.4 diversified-train-flow reduction | E.3 non-worsened | E.4 non-worsened |
| --- | --- | ---: | ---: | ---: | ---: |
| `1e-4` | A0 | 1.351% | 1.643% | 2/8 | 5/8 |
| `1e-4` | A1 | 0.025% | 1.751% | 2/8 | 3/8 |
| `3e-4` | A0 | −0.764% | 1.702% | 3/8 | 5/8 |
| `3e-4` | A1 | −1.308% | 1.787% | 2/8 | 7/8 |
| `1e-3` | A0 | 0.466% | 1.948% | 2/8 | 4/8 |
| `1e-3` | A1 | −8.254% | 0.997% | 1/8 | 4/8 |

diversified train-flow 的方向性改善在六条轨迹上都出现，因此 E.3 定位的
fixed-flow confound 不是纯粹的日志现象。但 E.4 最大 absolute reduction 仍只有
1.948%，不能把“比 E.3 好”替换成预注册的绝对通过条件。

## 5. A1 相对 A0：仅作探索性工程诊断

相同 LR 的 final held-out objective 严格配对。A1 相对 A0 的 final mean loss
差为：

| LR | A1 − A0 final mean loss | 相对 reduction 差 | A1 lower / equal / higher objectives |
| --- | ---: | ---: | ---: |
| `1e-4` | −0.00000605 | +0.109 pp | 26 / 1 / 13 |
| `3e-4` | −0.00000474 | +0.085 pp | 23 / 1 / 16 |
| `1e-3` | +0.00005295 | −0.951 pp | 23 / 1 / 16 |

这些差值没有在 E.4 前注册为模型效应 estimand，样本也只有一个 task 的八条训练
样本。因此只能用于判断下一工程诊断，不能写成“future 有正/负收益”。更重要的是，
A0 与 A1 的总体降幅都处于约 1%～2%，没有出现只有正确 future 才有的清晰增益。

## 6. Objective 与 scalar gate 遥测

以下分析是看到 E.4 结果后的探索性根因分析，不属于 Gate performance 判定：

- 六条训练轨迹的 positive per-step loss `max/min` 为
  `2268×～2342×`；
- per-step loss coefficient of variation 为 `1.919～1.931`；
- loss 最大的 10% step 占 positive loss 总和的 `58.7%～59.0%`；
- action scheduler weight 的 coefficient of variation 为 `0.753`；
- 198 个非零 gate-gradient step 中，连续 sign flip 为 `70～107` 次；
- `abs(mean gate gradient) / mean(abs(gate gradient))` 为
  `0.0014～0.108`；
- final scalar gate raw 只在 `−0.00370～0.00390`；
- final held-out median `delta/action-hidden` 只在
  `0.00273～0.00862`。

这组遥测与“microbatch 1 的单 objective 梯度方差很高、scalar gate 更新大量
抵消”一致，但不单独证明因果根因。它足以说明下一诊断不应继续增加 held-out
probe 或事后挑 LR，而应优先检验多 objective 梯度聚合/effective batch。

## 7. 执行、恢复与泄漏边界

```text
optimizer steps                    1,200
train metric rows                  1,200
held-out forward objectives          480
checkpoint directories                24
checkpoint files                       72
track execution checks true          108
development outcomes read          false
OOD/success outcomes read           false
future RGB frames read                  0
uses ground-truth future            false
rollout started                     false
```

每条轨迹的 train zero-weight steps 都精确为 `[49,142]`，共 12 个 training
zero-weight objective，全部满足 official weight 0 → exact weighted loss 0。
六条轨迹共享相同 sample/noise/timestep schedule SHA。

Phase D cache 仍为 96 entries / 12 shards，sample payload、cache fingerprint 和
split fingerprint 均通过冻结检查。Gate E.2、E.3 v2 工件在 E.4 前后 SHA 不变。

## 8. 资源

| 指标 | 值 |
| --- | ---: |
| Gate wall time | 1,528.287 s（25 分 28.3 秒） |
| Model load time | 383.845 s |
| 六条 track wall sum | 914.153 s |
| Mean optimizer step | 662.901 ms |
| Model-load peak | 23,679.513 MiB |
| Train/probe peak | 13,273.174 MiB |
| 输出大小 | 401,013,164 bytes（`du -sh` 为 383 MiB） |
| 输出文件数 | 107 |

## 9. 冻结工件

权威目录：

```text
outputs/thought3/phase_e4_diversified_flow_v1/
```

| 工件 | SHA-256 |
| --- | --- |
| `gate_e4_result.json` | `48314003c146327c93e3c5ecb173762cde09c27afb1b38124e741a222e974240` |
| `run_status.json` | `8c092f6aedbb67054e6853a49e35ec14f4cd3221b7867df6c72d6ff89a0acc43` |
| `pre_validation_result.json` | `4a74f33aa3af211854f86873c933530f904466c776c9ac97c969d7ef99cf8223` |
| `data_preparation.json` | `5cb61c57ab52feb93b395e3e3f379411e481f936839251b48048aa492c33a699` |
| `logs/phase_e4.log` | `6412697e39c55d5ba2c3232615d03007e69d517dff4b81701a12196814480886` |

`pre_validation_result.json` 的 `execution_error` 和 `execution_traceback` 均为
`null`。`run_status.json` 的 traceback 只来自完整结果写入后主动抛出的
failed-gate 状态，不是轨迹执行异常。

## 10. 停止规则与下一步

E.4 没有共同 eligible LR，因此：

- 不放宽 10% 或 6/8；
- 不从三个 LR 中事后选择 `3e-4`；
- 不增加更多 held-out flow；
- 不直接延长同一 Run ID；
- 不进入完整 28/4 Gate E；
- 不实现/训练 A2/A4；
- 不启动 Phase F 或任何 ID/OOD rollout。

下一项允许设计的单变量工程诊断应针对 optimizer 的 objective aggregation：

```text
当前：每次 optimizer update 只看到一个 sample × 一个 action-flow objective
候选：在同一次 update 内聚合多个预先冻结、严格配对的 objectives
```

在写代码和运行前，必须另行冻结：

1. matched-objective-budget 还是 matched-optimizer-update budget；
2. accumulation factor；
3. flow-slot 映射与是否复用 E.4 的 200 个 objective；
4. loss 是 mean 还是 sum；
5. checkpoint、zero-weight、梯度与显存门槛。

它仍须保持 A0/A1、八条 sample、LR grid、official loss、held-out probe 和
`10% + 6/8` 不变。该协议尚未预注册，更未获真实运行授权。
