# Thought3 Gate E.2：八样本 Train-only LR/尺度诊断报告

状态：**Gate E.2 未通过；六条工程轨迹全部完成**
验收日期：2026-07-28
证据等级：`ENGINEERING DIAGNOSTIC / FAILED-GATE / NOT MODEL EFFECT`
Run ID：`P3-PHASE-E2-v1`

## 1. 结论

Gate E.2 已在单张 RTX 4090 上完成 A0/A1 × 三个 learning rate 的六条真实
Fast-WAM 轨迹，共 1,200 optimizer step。运行没有崩溃、OOM、梯度断链、
checkpoint 损坏、冻结权重变化或数据泄漏。

Gate 最终仍按预注册规则失败，因为没有一个 learning rate 能让 A0 和 A1
**同时**满足：

- 8-sample sample-equal mean fixed loss 至少下降 10%；
- 至少 6/8 sample 的 final fixed loss 不高于自身 initial fixed loss。

三个 learning rate 的 hidden-correction 和 catastrophic-loss 门槛都通过，但
`eligibility` 全部为 false，因而没有选择 learning rate：

```text
lr_1e_04 = false
lr_3e_04 = false
lr_1e_03 = false
selected_lr_slug = null
gate_e2_passed = false
```

这不是“代码没跑好”，也不是“future 无效”。它说明当前八样本、每样本单个固定
action-flow noise/timestep 的诊断中，平均 loss 改善集中在少数初始 loss 较高的
固定目标上，尚未形成预注册要求的跨样本稳定改善。

## 2. 冻结身份

Gate E.2 的代码、LR 网格、门槛和选择规则在查看结果前冻结于：

```text
e10432868e1f7fa7116113a7c4eac8f1c64eb155
```

| 项目 | 值 |
| --- | --- |
| 分支 | `feature/thought3-partial-future-adapter` |
| 配置 | `configs/thought3/phase_e2_eight_sample_diagnostic.yaml` |
| config fingerprint | `f1a4cb39a2c6866331543a55c00f6d592be7a609c788802827f1848e244d6fd3` |
| cache fingerprint | `63a70e1af38f68bc894fc11d03c84f212e6c6328a5051256c9d045741156d9c5` |
| split fingerprint | `ea5402955023ccd48d790d821a73f98549b31d1ace8af035a90ceae2ad3951eb` |
| sample payload SHA-256 | `1bb4cfb6f4fc357f6227d7c369ad5fc00ed621b530270cd16cec9e1eba56973e` |
| seed | 3407 |
| trainable parameters | 1,371,137，Adapter-only |
| optimizer | AdamW，weight decay `1e-2` |
| LR grid | `1e-4 / 3e-4 / 1e-3` |
| budget | 每个 LR 的 A0/A1 各 200 step |
| checkpoint | 每 50 step |
| action-flow probe | 每 sample 固定 `flow_step=0` |

数据来自 Phase D 冻结的标准 `libero_goal` task 0 cache。完整 split 是
28 train / 4 development；source loader 只读取固定排序后的 8 条 train sample，
没有加载 development action target。

## 3. 主结果

所有轨迹的初始 sample-equal mean loss 都精确相同：

```text
0.010379154649854172
```

| LR | Variant | Final mean loss | Reduction | Non-worsened | Catastrophic | Median delta/hidden | Max delta/hidden | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1e-4` | A0 | 0.00999186 | 3.73% | 2/8 | 0/8 | 0.0172 | 0.0259 | 0.00430 |
| `1e-4` | A1 | 0.00786882 | 24.19% | 4/8 | 0/8 | 0.0174 | 0.0499 | −0.00736 |
| `3e-4` | A0 | 0.00827578 | 20.27% | 2/8 | 0/8 | 0.3563 | 0.5372 | 0.01189 |
| `3e-4` | A1 | 0.00622687 | 40.01% | 4/8 | 0/8 | 0.0436 | 0.1227 | −0.01136 |
| `1e-3` | A0 | 0.00986953 | 4.91% | 3/8 | 0/8 | 0.0729 | 0.1099 | −0.00247 |
| `1e-3` | A1 | 0.01182948 | −13.97% | 0/8 | 0/8 | 0.0126 | 0.0219 | 0.00410 |

逐轨迹判定：

- `1e-4/A0`：平均下降不足 10%，且仅 2/8 不变差；
- `1e-4/A1`：平均下降通过，但仅 4/8 不变差；
- `3e-4/A0`：平均下降通过，但仅 2/8 不变差；
- `3e-4/A1`：平均下降通过，但仅 4/8 不变差；
- `1e-3/A0`：平均下降不足 10%，且仅 3/8 不变差；
- `1e-3/A1`：平均 loss 反而增加 13.97%，0/8 不变差。

`1e-4` 和 `3e-4` 下 A1 的平均 train-only fixed loss 降幅大于 A0，是值得继续
诊断的信号；但这是同一小型训练子集上的结果，没有独立 development、rollout 或
OOD endpoint，不能写成 future 带来泛化或任务收益。

## 4. 为什么平均值和 6/8 判定分离

8 条 fixed objective 的初始 loss 范围为：

```text
min = 0.00038568
max = 0.03636319
max / min = 94.2842
```

两个高-loss objective 占 8 条初始 loss 总和约 87%。因此，只要这两个目标下降，
sample-equal mean 就能明显下降；多个低-loss objective 即使有小幅绝对上升，
仍可能被均值掩盖。

运行后的只读根因分析进一步发现：Gate E.2 把每条 sample 永久绑定到
`flow_step=0` 派生的一个 action noise/timestep。8 条样本的 timestep 为：

```text
152, 236, 472, 768, 820, 852, 896, 920
```

初始 fixed loss 与 timestep 的 Pearson 相关为：

```text
r = -0.93466
```

这说明当前“样本是否改善”同时反映 sample 内容和一次固定 flow
noise/timestep 抽样。该相关分析是看到结果后的工程根因分析，不是预注册统计
endpoint，也不能用于追溯放宽 Gate E.2。

## 5. 工程门禁全部通过

六条轨迹均满足：

- metrics 严格覆盖 step 1–200；
- sample order、预算和初始化配对一致；
- probe 严格覆盖 step `0/50/100/150/200`；
- step 1 只有 scalar gate 获得非零 gradient；
- step 2 起 projector、attention 和 non-gate gradient finite/nonzero；
- optimizer 只包含 Adapter；
- Fast-WAM 没有 gradient；
- Adapter/optimizer checkpoint round-trip 通过；
- 无 NaN/Inf；
- 峰值 optimizer-step 显存均为 `13,273.17 MiB`；
- 每条轨迹的 mean optimizer step 为 `659.65–673.28 ms`。

Fast-WAM 全参数 SHA：

```text
before = ac0dd59de78495dc8ed5c5601e24951b78f3fd6ad40843d76b83106eccceb4f8
after  = ac0dd59de78495dc8ed5c5601e24951b78f3fd6ad40843d76b83106eccceb4f8
```

其他成本：

- 模型加载：`387.77 s`，峰值 `23,679.51 MiB`；
- 8-sample 数据准备：`19.26 s`；
- 整个 Gate：`1,445.66 s`，即 24 分 5.66 秒；
- 输出目录约 381 MiB。

## 6. 数据访问与泄漏审计

```text
available train / development      28 / 4
selected train / development        8 / 0
current camera frames decoded          16
state rows read                          8
action chunks / rows                 8 / 256
future RGB frames decoded                 0
actual future read                    false
uses ground-truth future              false
development/OOD/success read          false
rollout started                       false
```

A0 使用同 shape zero/null latent；A1 只使用 Phase D 中由
`current observation + language + sampled noise` 生成的 K=1 latent。

## 7. 冻结工件

权威目录：

```text
outputs/thought3/phase_e2_eight_sample_v1/
```

根工件：

| 工件 | SHA-256 |
| --- | --- |
| `gate_e2_result.json` | `40f66bc50acd8e175ecb61ec150a04ef9ed5c55bf1fa9090802cc529104214bb` |
| `run_status.json` | `570774031d338ee27754f460c46deaf2a12f77d39e1b68cd3b08cb6af1a91e58` |
| `pre_validation_result.json` | `7aa98cfb95fbc73ab409ef47545e8a912ae221586fe57f2afa841676c6a9a7bb` |
| `data_preparation.json` | `fb92b8c7f01129689c5a4ddd7ab96aaa184687dcec15b07b9f180d049dc01b4e` |
| `logs/phase_e2.log` | `8bcb548821de88aab538322ace18183b526c346d14f7133cc5f43802a8cc4ef4` |

最终 Adapter semantic SHA：

| LR | A0 | A1 |
| --- | --- | --- |
| `1e-4` | `e1c7ef8f…2a4d45d` | `db76d9ac…574898` |
| `3e-4` | `36b82658…f4f12e` | `a40eaf6d…4bf1b` |
| `1e-3` | `700b6fa1…fcb50` | `f489dbde…887a00` |

六条 track 的 step 50/100/150/200 checkpoint、metrics、fixed probes 和 manifest
都必须保留。不得删除失败结果后复用本 Run ID。

## 8. 解释边界与下一步

Gate E.2 关闭了以下问题：

- A0/A1 多样本训练能完整运行；
- LR `1e-4/3e-4` 能在受控 hidden scale 下降低平均 fixed loss；
- A1 path 不是完全失活；
- checkpoint、resume 基础、冻结哈希和无泄漏边界成立。

它没有证明：

- A1 优于 A0；
- future 改善 development、ID 或 OOD；
- K=1 有用；
- 可以训练 A2/A4；
- 可以启动 Phase F rollout。

下一诊断不得事后把 6/8 门槛改成 4/8，也不得按本次最漂亮的 A1−A0 差值直接选择
`3e-4`。更可审计的单变量修改是：保持六条轨迹、LR 网格、200 step、官方 action
flow-matching loss 和所有门槛不变，只把每条 sample 从一个固定 action-flow
noise/timestep 扩为多个预冻结 flow slot。训练仍保持每 sample 相同出现次数，
fixed probe 先在 sample 内跨 flow slot 平均，再执行原来的 sample-equal 门槛。

该新协议必须使用新的配置、schema、输出目录和预注册 commit；真实运行前仍需用户
显式确认。
