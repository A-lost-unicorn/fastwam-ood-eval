# Thought3 Gate E.4：Paired Diversified Train-flow 诊断协议

状态：`EXECUTED / VALID FAILED GATE`
证据等级：`ENGINEERING DIAGNOSTIC / NOT MODEL EFFECT`

> 2026-07-28：六条 200-step 轨迹、1,200 optimizer steps 和 480 个 held-out
> objectives 已完整执行；全部 execution checks 通过，但没有 A0/A1 共同
> eligible LR，故 Gate E.4 有效失败。结果、边界与冻结工件见
> [thought3_phase_e4_report.md](thought3_phase_e4_report.md)。

## 1. 触发依据

Gate E.3 v2 已完整评估 Gate E.2 六个 checkpoint 的 held-out
`flow_step=1..5`：

- E.2 A1 的 fixed-flow reduction 在 `1e-4/3e-4` 为
  `24.19%/40.01%`；
- 相同 checkpoint 的 E.3 held-out reduction 变为
  `0.025%/−1.31%`；
- 六条轨迹的 hidden-scale、catastrophic、finite、pairing、memory 和 frozen
  checks 全部通过；
- 没有 A0/A1 共同 eligible LR。

E.2 optimizer 对每条 sample 的 25 次访问都复用 `flow_step=0`，即同一个
deterministic action noise/timestep。E.4 只检验：

> 把 optimizer objective 改为每次访问不同但严格配对的 action-flow slot 后，
> A0/A1 是否能在未用于训练的五个 flow 上形成稳定改善？

E.4 不是 future 效果实验，不读取 development/OOD/success，不启动 simulator。

## 2. 唯一训练变量

旧配方：

```text
all 200 optimizer steps:
flow_step = 0 for the visited sample
```

E.4：

```text
global_step = 1..200
training_flow_slot = 10_000 + global_step
```

因此训练 slots 精确为：

```text
10_001..10_200
```

性质：

- 200 个 slot 全部唯一；
- 与 E.2 fixed probe `0` 不重叠；
- 与 E.3/E.4 held-out probe `1..5` 不重叠；
- A0/A1、三个 LR 的同一 global step 使用相同 sample、noise seed、timestep
  seed、BF16 timestep 和 official loss weight；
- 每条 sample 仍按相同 round-robin 出现 25 次；
- RNG 仍使用既有 `thought3-real-action-{noise,time}-v1` namespace。

除 optimizer flow slot 外不得改变：

- 八条 train sample；
- Phase D cache；
- A0/A1 结构、初始化和参数量；
- LR grid；
- optimizer、weight decay；
- 200-step budget；
- checkpoint interval；
- official weighted velocity MSE；
- held-out probe；
- eligibility 阈值；
- smallest-eligible-LR 规则。

## 3. 冻结输入证据

运行前必须逐文件校验 Gate E.3 v2：

| 工件 | SHA-256 |
| --- | --- |
| `gate_e3_result.json` | `517c1e0cfc198f0bc44ab03d0d59349f20131d5c00efd958dd10f67aee1defe3` |
| `run_status.json` | `f1bfa70b18df2a9494a88dea52501659cfd10f7f368bf4531d7da12582dc70c3` |
| `pre_validation_result.json` | `68b7af97b5e17473ddb76472fe22c95abf5e1ec06e54ed7baeff324a2918ec14` |
| `data_preparation.json` | `0b505d9764cbf97e45fdebb9d95c68cbb4e3cd88bed2e0d73cebe95b1ce14ae6` |
| `logs/phase_e3.log` | `861c4bc58ac2bd3d3729d30e72aba3886908d996e01eb3e8f14858007191becc` |

同时校验：

- E.3 是完整的 valid failed gate，而不是 v1 工程异常；
- 320/320 objective 完整；
- E.3 initial/cross/paired/probe checks 全 true；
- 三个 LR eligibility 全 false；
- `selected_lr_slug=null`；
- Gate E.2 四个冻结 root SHA 不变；
- Phase D cache/inventory/split/checksum 不变。

任何一项不一致都在模型加载前 fail-closed。

## 4. 冻结组别、LR 与预算

```text
LR = 1e-4 / 3e-4 / 1e-3
variant = A0 / A1
steps per track = 200
total optimizer steps = 1,200
```

- A0：zero/null latent；
- A1：同 sample 的 K=1 model-generated cached latent；
- Adapter-only：1,371,137 参数；
- AdamW，weight decay `1e-2`；
- microbatch 1，gradient accumulation 1；
- deterministic CUDA math SDP；
- checkpoint `50/100/150/200`；
- 单物理 GPU、逻辑 `cuda:0`；
- 43 GiB hard memory limit。

不得根据 E.2/E.3 的“最好 A1 差值”缩减 LR grid。

## 5. Held-out probe 与门槛

每条轨迹只在 step `0` 和 `200` 运行同一冻结 probe：

```text
8 samples × flow_step 1..5 = 40 objectives
```

六条轨迹共：

```text
6 × 2 × 40 = 480 held-out forward objectives
```

不在 step 50/100/150 增加 probe，避免 E.3 失败后继续扩大 probe 预算。中间
checkpoint 只用于恢复和审计，不能按中间 loss 选择。

一个 LR 只有 A0 和 A1 分别都满足以下原门槛才 eligible：

1. held-out sample-equal mean loss 至少下降 10%；
2. 至少 6/8 sample 不变差；
3. 0/8 sample 超过 initial mean loss 的 2 倍；
4. median sample-mean `delta/action-hidden ≤ 0.50`；
5. max sample-mean `delta/action-hidden ≤ 1.00`。

选择 `1e-4 → 3e-4 → 1e-3` 中第一个共同 eligible LR。

## 6. 每 step 遥测

必须保存：

- global step、sample cursor、base sample ID；
- training flow slot；
- action noise seed、timestep seed；
- flow objective SHA-256；
- BF16 timestep、official training weight；
- weighted action loss；
- zero-weight flag；
- gate before/after、gradient/sign；
- projector/attention/non-gate gradient group；
- hidden、residual、future-token、gated-delta norm；
- `delta/action-hidden`；
- step time、peak memory、NaN/Inf。

按冻结 sample IDs 和 scheduler 预计算，slots `10_001..10_200` 中：

```text
global steps 49 and 142:
timestep = 1000
official training weight = 0
```

这两个 objective 合法地具有 exact-zero weighted loss。它们保留在 200-step
optimizer schedule 和 weight-decay 行为中，不能删除或重采样；运行时若零权重却
出现非零 action loss，立即失败。

六条轨迹的完整 objective schedule SHA 必须完全相同。

## 7. 恢复与执行硬门槛

每条轨迹必须满足：

- metrics 精确覆盖 1..200；
- 8-sample round-robin 精确；
- slots 精确为 10,001..10,200；
- seed/objective identity 可重算；
- step 1 只有 gate 非零 gradient；
- step 2 projector/attention/non-gate gradient finite/nonzero；
- 所有 loss/weight/timestep/gradient finite；
- optimizer 只含 Adapter；
- frozen Fast-WAM 无 gradient、SHA 前后相同；
- step-200 Adapter/optimizer checkpoint round-trip；
- A0/A1 参数量、初始化、sample、objective schedule 和预算相同；
- 无 development/OOD/success/ground-truth future；
- 显存小于 43 GiB。

真实中断时只允许从 checksum/provenance-valid 的 50-step checkpoint `--resume`。
若已生成有效的 pass/fail root result，则不得覆盖同一 Run ID。

## 8. 冻结运行身份与输出

| 字段 | 值 |
| --- | --- |
| config | `configs/thought3/phase_e4_diversified_flow_diagnostic.yaml` |
| config fingerprint | `e8c67a088c2c78e85e86c0cc0fac011e23303c59559d98c44dbc7051bdf578d1` |
| schema | `thought3.phase_e4.diversified_flow.v1` |
| output | `outputs/thought3/phase_e4_diversified_flow_v1/` |

输出：

```text
outputs/thought3/phase_e4_diversified_flow_v1/
├── data_preparation.json
├── pre_validation_result.json
├── gate_e4_result.json
├── run_status.json
├── logs/phase_e4.log
└── tracks/
    ├── lr_1e_04/{a0,a1}/
    ├── lr_3e_04/{a0,a1}/
    └── lr_1e_03/{a0,a1}/
```

按 E.2/E.3 实测预计：

- wall time 约 28–35 分钟；
- model load peak 约 23.7 GiB；
- optimizer/probe peak 约 13 GiB；
- 输出约 380–450 MiB。

## 9. 解释与停止规则

若存在共同 eligible LR：

1. 只解释为 diversified-flow 的工程候选 LR；
2. 使用新 Run ID 重新运行完整 28 train / 4 development Gate E；
3. 完整 Gate E 必须包含 resume、development-only selection、frozen SHA、loss
   和 hidden-scale 检查；
4. 完整 Gate E 通过后才允许训练 A2/A4 和实现真实 Phase F rollout。

若没有共同 eligible LR：

- 不放宽 10%/6-of-8；
- 不事后选择某个 LR；
- 不增加更多 held-out flow；
- 不扩 A2/A4 或 Phase F；
- 下一诊断必须针对 gate/Adapter parameterization、normalization 或 optimizer，
  且一次只改一个变量。

无论 E.4 pass/fail，都不能写成 future 改善或损害 OOD。

## 10. 显式授权

代码、测试和 dry-run 完成后不得自动运行。只有用户明确确认才可执行：

```bash
CONFIRM_THOUGHT3_PHASE_E4=YES \
THOUGHT3_GPU_ID=1 \
bash scripts/run_thought3_phase_e4_diversified_flow.sh
```

真实中断恢复：

```bash
CONFIRM_THOUGHT3_PHASE_E4=YES \
THOUGHT3_GPU_ID=1 \
bash scripts/run_thought3_phase_e4_diversified_flow.sh --resume
```
