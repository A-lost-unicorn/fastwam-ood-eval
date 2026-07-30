# Thought3 Gate E.3：Held-out Multi-flow 稳定性诊断协议

状态：`HISTORICAL V1 / INVALID ENGINEERING RUN`
证据等级：`ENGINEERING DIAGNOSTIC / NOT MODEL EFFECT`

> 2026-07-28：v1 已运行，但在官方 scheduler 的合法
> `timestep=1000, training_weight=0` objective 上触发非门控 loss-ratio
> 实现错误，未生成 `gate_e3_result.json`。本文件保留 v1 预注册内容；失败证据见
> [thought3_phase_e3_v1_failure_report.md](phase_e3_v1_failure_report.md)，
> 修复后的新 Run ID 协议见
> [thought3_phase_e3_v2_protocol.md](phase_e3_v2_protocol.md)。

## 1. 为什么需要 E.3

Gate E.2 的六条轨迹全部完成，但没有共同 eligible learning rate。只读根因分析
发现，E.2 把每条 sample 永久绑定到一个 `flow_step=0` 派生的 action
noise/timestep：

- 8 条 initial fixed loss 的 max/min 为 `94.2842×`；
- initial fixed loss 与该 BF16 timestep 的 Pearson 相关为 `−0.93466`；
- `1e-4/3e-4` 下 A1 的平均 loss 明显下降，但只有 4/8 sample 不变差；
- A0 在任一 learning rate 下最多只有 3/8 sample 不变差。

因此，E.2 的 per-sample stability 同时包含了 sample 内容和一次 action-flow
Monte Carlo draw。E.3 只诊断这一项，不改变模型、训练轨迹或 Gate E.2 结论。

## 2. 唯一协议变化

E.3 **不重新训练**。它只读取 E.2 已冻结的六个 step-200 Adapter-only
checkpoint，并将 fixed probe 从：

```text
每 sample：flow_step = 0 一个固定 draw
```

改为：

```text
每 sample：flow_step = 1, 2, 3, 4, 5 五个 held-out fixed draws
```

`flow_step=0` 是 E.2 训练和 probe 使用的目标；`1..5` 从未进入 E.2 optimizer，
因此是相对于该训练轨迹 held-out 的 action noise/timestep。

除 probe replicate count 外全部保持不变：

- 同一 8 条 train sample；
- 同一 A0/A1 checkpoint；
- 同一官方 action Flow Matching / velocity MSE；
- 同一 scheduler、shift、seed namespace 和 action normalization；
- 同一 LR grid；
- 同一 hidden-scale 和 loss 门槛；
- 同一 smallest-eligible-LR 选择规则；
- 不读取 development、OOD、success 或 rollout；
- 不使用真实未来；
- 不修改 E.2 输出或 checkpoint；
- 不训练，不创建 optimizer，不做 backward。

## 3. 冻结上游证据

只允许读取：

```text
outputs/thought3/phase_e2_eight_sample_v1/
```

必须在加载模型前校验：

| 工件 | SHA-256 |
| --- | --- |
| `gate_e2_result.json` | `40f66bc50acd8e175ecb61ec150a04ef9ed5c55bf1fa9090802cc529104214bb` |
| `run_status.json` | `570774031d338ee27754f460c46deaf2a12f77d39e1b68cd3b08cb6af1a91e58` |
| `pre_validation_result.json` | `7aa98cfb95fbc73ab409ef47545e8a912ae221586fe57f2afa841676c6a9a7bb` |
| `data_preparation.json` | `fb92b8c7f01129689c5a4ddd7ab96aaa184687dcec15b07b9f180d049dc01b4e` |

还必须确认：

- E.2 `gate_e2_passed=false`；
- 六条 track 均为 `status=complete`、200 step；
- 六条 execution checks 全部为 true；
- A0/A1 pairing 和 frozen Fast-WAM checks 全部为 true；
- 每个 final checkpoint 的 file checksum、semantic Adapter SHA、variant、K、
  config fingerprint、cache/split/frozen SHA 与 E.2 manifest 一致。

任何不一致都 fail-closed，禁止“就近找一个 checkpoint”。

## 4. Probe 定义

对每个 variant、sample 和 held-out flow step，固定：

- current observation；
- language/context；
- proprio；
- target action；
- Adapter checkpoint；
- action noise seed；
- action timestep seed。

只改变预先列出的 `flow_step=1..5`。每个 objective 保存：

- base sample ID；
- flow step；
- timestep；
- action loss；
- action hidden norm；
- attention residual norm；
- gated BF16 delta norm；
- `gated_delta/action_hidden`；
- finite 状态；
- latency 和 peak memory。

先在每个 sample 内对五个 held-out draw 做算术平均：

```text
sample action loss       = mean(loss_step1 ... loss_step5)
sample delta/hidden      = mean(ratio_step1 ... ratio_step5)
```

再对 8 条 sample 做 sample-equal 汇总：

- mean action loss；
- non-worsened sample count；
- catastrophic sample count；
- median sample mean delta/hidden；
- max sample mean delta/hidden。

同时记录 40 个 objective 中的最大 loss ratio 和最大 delta/hidden，作为诊断遥测，
但不在看到结果后增加新的 eligibility 条件。

## 5. 冻结门槛

一个 LR 只有在 A0 和 A1 分别都满足以下条件时 eligible：

1. held-out multi-flow sample-equal mean loss 至少下降 10%；
2. 至少 6/8 sample 的 final mean loss 不高于 initial mean loss；
3. 0/8 sample 的 final mean loss 超过 initial mean loss 的 2 倍；
4. median sample mean `delta/action-hidden ≤ 0.50`；
5. max sample mean `delta/action-hidden ≤ 1.00`。

初始 probe 使用相同 zero-init Adapter；A0/A1 的 initial action loss 必须逐
sample、逐 flow-step 精确相同。所有 LR 的同 variant initial probe 也必须相同。

选择规则仍为：

```text
1e-4 → 3e-4 → 1e-3
```

选择第一个 A0/A1 共同 eligible 的 learning rate。不得按 A1−A0 差值或最低 A1
loss 选择。

## 6. 解释与停止规则

如果 E.3 存在共同 eligible LR：

- 只将它解释为跨 held-out action-flow draws 的工程候选配方；
- 使用新的 Run ID 重新执行完整 28 train / 4 development Gate E；
- 完整 Gate E 仍须重新训练，不能直接拿 E.2 的 8-sample checkpoint 做 rollout。

如果 E.3 没有共同 eligible LR：

- Gate E.3 失败；
- 不放宽 6/8 或 10% 门槛；
- 不扩 A2/A4；
- 不启动 Phase F；
- 下一诊断必须针对 Adapter/optimizer 本身，而不是继续增加 probe 数量。

E.3 无论通过或失败，都不能证明 future 改善 success、ID 或 OOD。

## 7. 输出、成本和运行授权

独立输出：

```text
outputs/thought3/phase_e3_multiflow_v1/
├── data_preparation.json
├── pre_validation_result.json
├── gate_e3_result.json
├── run_status.json
└── logs/phase_e3.log
```

配置为 `configs/thought3/phase_e3_multiflow_diagnostic.yaml`，冻结 fingerprint：

```text
f2313eec175f26d7d0bc61a89c77127344f76012e234d9792fc3850131075652
```

E.3 不写 checkpoint，预计输出小于 5 MiB。需要加载一次 Fast-WAM、计算两次
frozen SHA，并完成：

```text
2 initial variants × 8 samples × 5 draws
+ 6 final checkpoints × 8 samples × 5 draws
= 320 action-loss forward probes
```

预计单卡 wall time 15–20 分钟，显存不应超过 E.2 的 13,273.17 MiB
probe/training 峰值；硬限制仍为 43 GiB。

真实运行前必须由用户显式确认：

```bash
CONFIRM_THOUGHT3_PHASE_E3=YES \
THOUGHT3_GPU_ID=1 \
bash scripts/run_thought3_phase_e3_multiflow.sh
```

代码生成、dry-run 或测试完成后不得自动执行该命令。
