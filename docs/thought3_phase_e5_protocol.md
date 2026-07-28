# Thought3 Gate E.5：Full-cohort Objective Aggregation 诊断协议

状态：`PREREGISTERED / IMPLEMENTED / NOT EXECUTED`
证据等级：`ENGINEERING DIAGNOSTIC / NOT MODEL EFFECT`

> 本协议在任何 E.5 loss/result 产生前冻结。代码完成后不得自动运行；真实执行
> 仍需用户显式确认。E.5 即使通过，也不能解释为 future 改善了 OOD。

## 1. 触发依据

Gate E.4 已完成六条 `A0/A1 × 3 LR` 的 200-update 轨迹：

- 1,200 次 optimizer update、480 个 held-out objective 完整；
- execution、pairing、checkpoint、frozen、memory、leakage checks 全部通过；
- 六条 held-out loss reduction 均为正，但只有 `0.997%–1.948%`；
- 没有一个 LR 同时让 A0/A1 达到冻结的 `10% + 6/8` 门槛。

E.4 的 post-run 工程分析还观察到：

- 单 objective weighted loss 跨 step 的动态范围约 `2,268×–2,342×`；
- 正权重 train loss 的 CV 约 `1.92`；
- 最高 10% objective 承担约 `58.7%–59.0%` 的正 loss；
- scalar gate gradient 在相邻 step 间频繁换符号；
- 部分轨迹的 `|mean gate gradient| / mean(|gate gradient|)` 很低。

这些是设计 E.5 的依据，不是已经证明的根因。E.5 只问：

> 在相同 optimizer-update 数下，每次更新先聚合同一完整 8-sample cohort 的
> 多个独立 action-flow objective，是否能形成通过原 held-out 门槛的稳定改善？

## 2. 唯一训练变量

E.4：

```text
one optimizer update
  = one sample
  = one unique action-flow objective
```

E.5：

```text
one optimizer update
  = all 8 frozen train samples, each exactly once
  = 8 unique action-flow objectives
  = arithmetic mean of 8 official weighted velocity losses
```

实现必须等价于：

```python
optimizer.zero_grad()
for objective in eight_sample_cohort:
    (objective_loss / 8).backward()
optimizer.step()
```

禁止：

- 对 8 个 loss 求和而不除以 8；
- 每个 micro-objective 调用一次 `optimizer.step()`；
- 在 update 内重复或遗漏 sample；
- 按 loss 大小选择、重采样或丢弃 objective；
- 用梯度裁剪、scheduler、warmup、gate regularization 等同时改变第二个变量。

## 3. 为什么使用 matched-update budget

本诊断预先选择 `matched optimizer-update`，不是 `matched objective-count`：

| 项目 | E.4 | E.5 |
| --- | ---: | ---: |
| optimizer updates / track | 200 | 200 |
| objectives / update | 1 | 8 |
| train objectives / track | 200 | 1,600 |
| AdamW weight-decay applications / track | 200 | 200 |
| checkpoints | 50/100/150/200 | 50/100/150/200 |

理由是保持 AdamW 更新次数、weight-decay 次数和候选 checkpoint 边界不变，隔离
“单 objective 更新”与“完整 cohort 均值更新”。代价是 E.5 看到了 8 倍 train
objectives，故它不能被描述为 compute-matched 或 sample-matched 对照，也不能把
E.5 与 E.4 的差异直接归因于“梯度方差”这一唯一机制。

若未来需要 matched-objective 对照，必须使用新协议和新 Run ID；不得在本次结果
后追溯改变预算定义。

## 4. 冻结输入与父 Gate

运行前逐文件校验 E.4 valid-negative 工件：

| 工件 | SHA-256 |
| --- | --- |
| `gate_e4_result.json` | `48314003c146327c93e3c5ecb173762cde09c27afb1b38124e741a222e974240` |
| `run_status.json` | `8c092f6aedbb67054e6853a49e35ec14f4cd3221b7867df6c72d6ff89a0acc43` |
| `pre_validation_result.json` | `4a74f33aa3af211854f86873c933530f904466c776c9ac97c969d7ef99cf8223` |
| `data_preparation.json` | `5cb61c57ab52feb93b395e3e3f379411e481f936839251b48048aa492c33a699` |
| `logs/phase_e4.log` | `6412697e39c55d5ba2c3232615d03007e69d517dff4b81701a12196814480886` |

同时要求：

- E.4 schema/status/failed gate/null selection 精确；
- 三个 eligibility 全 false；
- 108 个 per-track execution check、全部 paired/cross checks 为 true；
- E.4 scope 精确为 1,200 updates、480 held-out objectives；
- prevalidation 没有 execution error/traceback；
- 六条轨迹 sample IDs 完全相同；
- Phase D cache、数据 split 与 source audit 仍通过。

任何不一致都必须在模型加载前 fail-closed。

## 5. 冻结 sample 与 objective schedule

每次 optimizer update 的 sample 顺序固定为：

| micro index | base sample ID |
| ---: | --- |
| 1 | `4a0a595342e32200b9f7dc1266b0a110ef9c062370b524c6c5808102eade8bfb` |
| 2 | `68d4dae70bb0327cdc377526ade847937985e1ff41c75244fefbc051031f69c5` |
| 3 | `6e613283d05bfebe6060077a82ded94db241672411c900274d034f22ad765343` |
| 4 | `9075e1637a0fd4b70472e896c96b5aa21e766ee4d9d2bf398465e6e2065626d1` |
| 5 | `1b4b3db48f51a7590f163b75f2f9e4246bff8a563034e4c2b5401dd78627aa0d` |
| 6 | `83eaad65f24194306273012f47f20e043c1aa2c2e26e6dfb9140d181b096ec51` |
| 7 | `43b8eb04262bb651497f184ce1b5975cef7eda39c8670bd3904b26d0437c86b9` |
| 8 | `0643a55f92bbe4fd0da1abb9ab756422ca564f2f921b8f8b0b1f9d50514b5c69` |

slot 映射：

```text
optimizer_update = 1..200
micro_index      = 1..8
slot = 20_000 + (optimizer_update - 1) × 8 + micro_index
```

因此：

- 1,600 个 train slots 精确为 `20_001..21_600`；
- 与 fixed probe `0`、held-out probe `1..5` 不重叠；
- 与 E.4 slots `10_001..10_200` 不重叠；
- A0/A1 和三个 LR 共用完全相同的 sample/slot/noise/timestep schedule；
- RNG namespace 仍为 `thought3-real-action-{noise,time}-v1`。

运行前可知的完整 schedule identity SHA-256 冻结为：

```text
b6f9778d303a6ad2c4bef781f4a6027a800d013814110daa47eb7cb1d13af86d
```

该 hash 覆盖 objective/update/micro/sample index、base sample ID、slot、noise
seed、timestep seed 和 objective SHA；运行后另存含实际 BF16 timestep/official
weight 的 observed schedule SHA。

## 6. 合法零权重 objectives

按冻结 scheduler 和 sample/slot identity，预先计算出 24 个 `t=1000`、official
weight `0` 的 objective：

```text
(update,micro,slot)
(20,3,20155)   (22,5,20173)   (26,4,20204)   (41,5,20325)
(48,8,20384)   (51,1,20401)   (55,2,20434)   (61,6,20486)
(81,5,20645)   (84,3,20667)   (92,3,20731)   (94,1,20745)
(94,5,20749)   (108,5,20861)  (113,8,20904)  (118,3,20939)
(123,2,20978)  (130,3,21035)  (134,8,21072)  (151,2,21202)
(158,5,21261)  (173,6,21382)  (177,3,21411)  (190,5,21517)
```

这些 objective 必须保留；其 weighted action loss 必须 exact zero，但同一 update
通常仍由其余七个 objective 产生非零均值梯度。禁止删除或重采样。

## 7. 冻结组别、优化器与评估门槛

```text
variant = A0 / A1
LR = 1e-4 / 3e-4 / 1e-3
optimizer = AdamW
weight decay = 1e-2
optimizer updates / track = 200
gradient accumulation = 8
loss reduction = arithmetic mean
train objectives / track = 1,600
total optimizer updates = 1,200
total train objectives = 9,600
```

其余保持 E.4：

- A0 为同结构 zero/null latent，A1 为同 sample K=1 model-generated cache；
- Adapter-only，结构 fingerprint 与 1,371,137 参数不变；
- zero-init scalar gate；
- official Fast-WAM flow-matching weighted velocity MSE；
- 单物理 GPU、逻辑 `cuda:0`；
- deterministic CUDA math SDP、TF32 off；
- 43 GiB hard memory ceiling；
- 不读取 development/OOD/success/rollout/真实未来 RGB。

每条轨迹只在 update `0` 和 `200` 运行：

```text
8 samples × held-out flow_step 1..5 = 40 objectives
```

六条轨迹共 480 held-out objectives。一个 LR 只有 A0/A1 分别都满足以下条件才
eligible：

1. held-out sample-equal mean loss 至少下降 10%；
2. 至少 6/8 sample 不变差；
3. 0/8 sample 超过 initial mean loss 的 2 倍；
4. median sample-mean `delta/action-hidden ≤ 0.50`；
5. max sample-mean `delta/action-hidden ≤ 1.00`。

仍选择升序 LR 中第一个共同 eligible 值；不得事后选择“看起来最好”的单条轨迹。

## 8. 梯度与遥测硬门槛

每个 objective 保存：

- update、micro、objective index、sample cursor、base sample ID；
- slot、noise/timestep seeds、objective SHA；
- BF16 timestep、official weight、raw weighted action loss；
- `loss/8` 的 backward contribution；
- mean-scaled gate-gradient contribution、累计值和符号；
- gate before/after update；
- future token、attention residual、hidden、gated delta 与比例；
- zero-weight、NaN/Inf、update time/peak memory。

每个 optimizer update 保存：

- 8 个 loss/weight 的 sum 与 arithmetic mean；
- gate gradient、各 micro contribution 的绝对值和、cancellation ratio；
- gate before/after；
- projector/attention/non-gate gradient groups；
- objective index 范围、sample cursor；
- update time、peak memory、NaN/Inf。

必须满足：

- update 1：完整八样本都在 exact-zero gate 下反传，只有 gate gradient 非零；
- update 2：projector、attention、non-gate gradient finite/nonzero；
- 每个 update 的 8 个 mean-scaled contribution 精确重算 accumulated gradient；
- frozen Fast-WAM 无 gradient，参数 SHA 前后相同；
- optimizer 只包含 Adapter；
- 所有数值 finite，显存 `<43 GiB`；
- step-200 Adapter/optimizer checkpoint round-trip。

## 9. 原子恢复与输出

checkpoint 固定在 optimizer update `50/100/150/200`。每次先原子提交：

```text
train_objective_metrics.jsonl
train_update_metrics.jsonl
```

再保存 Adapter-only checkpoint。manifest 的 `sample_cursor` 必须等于
`optimizer_update × 8`。恢复时：

- 只接受 checksum/provenance-valid checkpoint；
- objective rows 截断到 `checkpoint_update × 8`；
- update rows 截断到 `checkpoint_update`；
- checkpoint manifest 同时绑定两份 metric prefix 的 canonical SHA-256 与
  observed schedule SHA，已提交前缀任一字段变化都拒绝恢复；
- 多出的原子 metrics 视为 checkpoint 前崩溃后的未提交计算并安全重算；
- 不允许缺行、重复、乱序或 partial cohort；
- 已存在完整 valid pass/fail root result 时不覆盖同一 Run ID。

冻结运行身份：

| 字段 | 值 |
| --- | --- |
| config | `configs/thought3/phase_e5_objective_aggregation_diagnostic.yaml` |
| config fingerprint | `c4c681534cf4c143a1675c24ded719b7bf0a4c2964b2384704b4338e147122fc` |
| schema | `thought3.phase_e5.objective_aggregation.v1` |
| output | `outputs/thought3/phase_e5_objective_aggregation_v1/` |

输出结构：

```text
outputs/thought3/phase_e5_objective_aggregation_v1/
├── data_preparation.json
├── pre_validation_result.json
├── gate_e5_result.json
├── run_status.json
├── logs/phase_e5.log
└── tracks/
    ├── lr_1e_04/{a0,a1}/
    ├── lr_3e_04/{a0,a1}/
    └── lr_1e_03/{a0,a1}/
```

基于 E.4 实测，预计单卡：

- model load peak 约 `23.7 GiB`；
- train peak约 `13–14 GiB`；
- wall time 约 `110–135 分钟`；
- 输出约 `400–470 MiB`。

这些只是执行规划，不是实测结果。运行后必须以 root manifest 为准。

## 10. 解释、通过路径与停止规则

若有共同 eligible LR：

1. 只解释为 full-cohort objective aggregation 给出了工程候选 LR；
2. 不能比较 A0/A1 差异并宣称 future effect；
3. 使用新 Run ID 运行完整 28-train/4-development Gate E；
4. 完整 Gate E 仍必须通过 resume、dev-only selection、hidden scale、frozen
   SHA 和 no-leakage；
5. 只有完整 Gate E 通过后才允许训练 A2/A4 和进入真实 Phase F。

若没有共同 eligible LR：

- 不放宽 `10% + 6/8`；
- 不事后挑一个 LR、sample 或 update；
- 不扩大 held-out flow 数；
- 不启动 A2/A4、OOD 或 rollout；
- 记录为有效负工程结果；
- 下一诊断只能一次改变一个已冻结变量，如 gate parameterization、Adapter
  normalization 或 optimizer，并使用新 Run ID。

无论 pass/fail，E.5 都不能回答显式 future 是否提升 OOD。

## 11. 显式授权

实现、测试和 dry-run 不会启动真实模型。无确认时，公共 CLI 在创建
`run_status.json` 或加载模型前即拒绝。只有用户明确确认后才运行：

```bash
CONFIRM_THOUGHT3_PHASE_E5=YES \
THOUGHT3_GPU_ID=1 \
bash scripts/run_thought3_phase_e5_objective_aggregation.sh
```

中断后只用：

```bash
CONFIRM_THOUGHT3_PHASE_E5=YES \
THOUGHT3_GPU_ID=1 \
bash scripts/run_thought3_phase_e5_objective_aggregation.sh --resume
```

不得同时启动其他占用该物理 GPU 的任务。
