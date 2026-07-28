# Thought3 Gate E.3 v2：Held-out Multi-flow 诊断报告

状态：**Gate E.3 有效失败；全部 probe 完成**
验收日期：2026-07-28
证据等级：`ENGINEERING DIAGNOSTIC / FAILED-GATE / NOT MODEL EFFECT`
Run ID：`P3-PHASE-E3-v2`

## 1. 结论

Gate E.3 v2 已完整读取 Gate E.2 的六个 step-200 Adapter checkpoint，并在每条
sample 的五个 held-out action-flow noise/timestep 上完成：

```text
2 initial variants × 8 samples × 5 flows
+ 6 final checkpoints × 8 samples × 5 flows
= 320/320 action-loss forward
```

所有数据访问、probe、pairing、finite、显存、冻结参数和零权重审计均通过；本次
非零进程退出码来自预注册性能门槛失败，而不是执行异常。三个 learning rate 均无
A0/A1 共同 eligible 配方：

```text
lr_1e_04 = false
lr_3e_04 = false
lr_1e_03 = false
selected_lr_slug = null
gate_e3_passed = false
```

E.2 中 `1e-4/3e-4` 的 A1 单固定-flow mean loss 曾下降
`24.19%/40.01%`；在 E.3 的五个 held-out flow 上只剩
`0.025%/−1.31%`。当前 checkpoint 的 fixed-objective 改善没有稳定迁移到新的
action noise/timestep。

该结果否定的是：

> 当前“每 sample 固定一个 action-flow objective、训练 200 step”的小训练配方
> 已具备跨 flow 稳定性。

它不否定 future latent 的潜在价值，也不能回答 ID/OOD success、K 的优劣或
future 的因果作用。

## 2. 冻结运行身份

| 字段 | 值 |
| --- | --- |
| v2 预注册 commit | `139742f861fd9d71f762fbf5c81a6ab4970035a0` |
| config | `configs/thought3/phase_e3_multiflow_diagnostic_v2.yaml` |
| config fingerprint | `eeab3e38c1fd7ce15afc0852c1cac1007455a5551758c37d068ad6ea470b392e` |
| schema | `thought3.phase_e3.multiflow.v2` |
| source Gate E.2 result SHA | `40f66bc50acd8e175ecb61ec150a04ef9ed5c55bf1fa9090802cc529104214bb` |
| flow steps | `[1,2,3,4,5]` |
| sample / checkpoint / objective | `8 / 6 / 320` |
| model dtype | `torch.bfloat16` |
| model load | `381.204 s` |
| model load peak | `23,679.513 MiB` |
| probe peak | `12,945.219 MiB` |
| gate wall time | `641.300 s`，即约 `10 分 41.3 秒` |

冻结 Fast-WAM 参数 SHA：

```text
before = ac0dd59de78495dc8ed5c5601e24951b78f3fd6ad40843d76b83106eccceb4f8
after  = ac0dd59de78495dc8ed5c5601e24951b78f3fd6ad40843d76b83106eccceb4f8
```

## 3. 主结果

所有 A0/A1 initial probe 的 sample-equal mean loss 精确相同：

```text
0.005565503754223755
```

| LR | Variant | Final mean loss | Reduction | Non-worsened | Catastrophic | Median delta/hidden | Max delta/hidden | Eligible |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `1e-4` | A0 | 0.00549030 | **1.351%** | 2/8 | 0/8 | 0.0181 | 0.0214 | false |
| `1e-4` | A1 | 0.00556413 | **0.025%** | 2/8 | 0/8 | 0.0187 | 0.0332 | false |
| `3e-4` | A0 | 0.00560802 | **−0.764%** | 3/8 | 0/8 | 0.3745 | 0.4420 | false |
| `3e-4` | A1 | 0.00563828 | **−1.308%** | 2/8 | 0/8 | 0.0519 | 0.0701 | false |
| `1e-3` | A0 | 0.00553956 | **0.466%** | 2/8 | 0/8 | 0.0766 | 0.0904 | false |
| `1e-3` | A1 | 0.00602489 | **−8.254%** | 1/8 | 0/8 | 0.0121 | 0.0149 | false |

六条轨迹均通过：

- 0/8 catastrophic sample；
- median `delta/hidden ≤ 0.5`；
- max sample-mean `delta/hidden ≤ 1.0`；
- 40/40 objective 完整、finite；
- probe memory `<43 GiB`；
- 无真实未来输入；
- A0/A1 sample、flow 与预算严格配对。

六条轨迹均未通过：

- mean loss 至少下降 10%；
- 至少 6/8 sample 不变差。

因此失败来源不是 hidden correction 过大，而是 action loss 在 held-out flow 上没有
形成预注册要求的稳定改善。

## 4. E.2 与 E.3 对照

| LR | Variant | E.2 fixed-flow reduction | E.3 held-out reduction | E.2 non-worsened | E.3 non-worsened |
| --- | --- | ---: | ---: | ---: | ---: |
| `1e-4` | A0 | 3.73% | 1.35% | 2/8 | 2/8 |
| `1e-4` | A1 | 24.19% | 0.025% | 4/8 | 2/8 |
| `3e-4` | A0 | 20.27% | −0.76% | 2/8 | 3/8 |
| `3e-4` | A1 | 40.01% | −1.31% | 4/8 | 2/8 |
| `1e-3` | A0 | 4.91% | 0.47% | 3/8 | 2/8 |
| `1e-3` | A1 | −13.97% | −8.25% | 0/8 | 1/8 |

E.2 optimizer 每次访问 sample 都使用同一 `flow_step=0`，即同一 deterministic
action noise/timestep。E.3 的 `flow_step=1..5` 从未进入该 optimizer。最强的
E.2 fixed-flow 改善在 held-out flows 上消失，符合“checkpoint 主要拟合固定
action-flow realization”的解释。

这是序贯工程诊断，不是预注册的模型效应统计。不能据此写成“A1 一定过拟合”或
“future 一定无效”；更准确的表述是：

> 现有证据没有证明当前固定-flow 配方能把 A0 或 A1 的 action-loss 改善泛化到
> 新 action noise/timestep。

## 5. 多 flow 是否关闭了 E.2 的混淆

五-flow 平均降低了单 draw 的极端差异，但没有完全消除 objective heterogeneity：

```text
E.2 单 flow initial max/min = 94.2842×
E.3 五 flow initial max/min = 33.1343×
E.3 五 flow sample-loss CV = 0.9920
```

因此，E.2 的单 draw 混淆确实很强；但即使每 sample 跨五个 draw 平均，八条 sample
的目标难度仍明显不同。后续训练与判定必须继续保持 sample-equal，而不能只依赖
被少数高-loss sample 主导的总体 token mean。

## 6. 零权重端点与 v2 修复验收

八个 probe 中每个都有且只有一个 `timestep=1000` objective：

```text
training_weight = 0
official weighted action loss = 0
```

全运行共八个 zero-weight row，全部满足 zero weight → exact zero loss。每个
initial/final outcome：

- 39 个 positive-denominator objective 进入非门控 loss-ratio 遥测；
- 1 个 zero-loss objective 仍保留在 sample mean 和全部正式门槛；
- 0 个 zero-initial objective 在 final 变为正 loss；
- 0 个 zero-weight/nonzero-loss 错误。

这证明 v2 修复按预注册边界工作，没有通过删除该 objective 来改善主结果。

## 7. 数据访问与执行边界

```text
optimizer steps                 0
backward calls                  0
checkpoints read                6
development outcomes read      false
OOD/success outcomes read       false
future RGB frames read          0
uses ground-truth future        false
rollout started                 false
```

`pre_validation_result.json` 的 `execution_error` 与 `execution_traceback` 均为
`null`。`run_status.json` 中的 error 只表示完整结果未通过 hard gate；
`gate_e3_result.json` 已存在且是本次有效负结果的权威判定。

## 8. 冻结工件

权威目录：

```text
outputs/thought3/phase_e3_multiflow_v2/
```

| 工件 | SHA-256 |
| --- | --- |
| `gate_e3_result.json` | `517c1e0cfc198f0bc44ab03d0d59349f20131d5c00efd958dd10f67aee1defe3` |
| `run_status.json` | `f1bfa70b18df2a9494a88dea52501659cfd10f7f368bf4531d7da12582dc70c3` |
| `pre_validation_result.json` | `68b7af97b5e17473ddb76472fe22c95abf5e1ec06e54ed7baeff324a2918ec14` |
| `data_preparation.json` | `0b505d9764cbf97e45fdebb9d95c68cbb4e3cd88bed2e0d73cebe95b1ce14ae6` |
| `logs/phase_e3.log` | `861c4bc58ac2bd3d3729d30e72aba3886908d996e01eb3e8f14858007191becc` |

Gate E.2 四个冻结 SHA 和 E.3 v1 四个失败证据 SHA 在 v2 后复核不变。

## 9. 停止规则与下一步

按 v2 预注册停止规则：

- 不放宽 `10%` 或 `6/8`；
- 不从三个 LR 中事后挑一个；
- 不扩 A2/A4；
- 不启动 Phase F rollout；
- 不继续增加 held-out probe 数量。

下一项允许的单变量工程诊断应直接针对 optimizer objective：

```text
旧：每 sample 的 25 次训练访问都复用 flow_step=0
新：每次访问使用不同、确定性、A0/A1 配对的 train flow slot
```

held-out probe、8 samples、LR grid、200-step budget、Adapter、cache、loss、阈值和
smallest-eligible 规则保持不变。训练 flow slots 必须与 probe `1..5` 完全不重叠，
并记录 timestep、official weight 和合法 zero-weight step。

该诊断必须使用新的 schema/config/output 和预注册 commit；在用户再次显式确认前
不得运行。冻结实现与停止规则见
[thought3_phase_e4_protocol.md](thought3_phase_e4_protocol.md)。
