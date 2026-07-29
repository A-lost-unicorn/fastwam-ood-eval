# Thought3 训练与恢复手册

状态：Phase D 小规模真实 cache 已通过；尚未启动真实 Adapter 训练
更新时间：2026-07-28

## 1. 当前能做什么

Phase B 已实现：

- 单次 action-encoder-output 注入；
- zero-gated Future-to-Action cross-attention；
- A0/A1/A2/A4 同构模型；
- action velocity MSE 的 CPU mock；
- Adapter-only safetensors checkpoint；
- optimizer、global step、sample cursor resume；
- train/development loss、grad norm、gate、attention norm、step time、NaN/Inf；
- 中断+恢复与不中断训练的最终 Adapter tensor hash 一致测试。

Phase C 已额外验证：

- 一条真实 LIBERO training sample 的 K=1/2/4 latent；
- BF16 Fast-WAM hidden 到 FP32 Adapter 的 zero-gate forward；
- 一次官方 action loss backward；
- backbone 0 gradient、MoT hash 不变；
- 单卡执行峰值 12.964 GiB、模型加载峰值 23.125 GiB；
- optimizer 0 step、真实 cache 0 条。

Phase D 已额外验证：

- 一个 `libero_goal` task、32 个不同 episode 当前观测的真实 cache；
- K=1/2/4 共 96 条 BF16 latent、12 个原子 safetensors shard；
- episode 级 37/5 完整 split，pilot cache 内为 28/4；
- paired seed/initial-state hash、shape、checksum、no-op resume 和主动损坏检测；
- 64 张当前相机帧、0 future RGB、0 action target 的 source access 审计；
- 0.806 base sample/s（不含模型加载），执行峰值 12.677 GiB；
- optimizer 0 step、Adapter training 0 step。

当前仍不能做：

- 启动正式或长时间 Adapter 训练；
- 把 32-sample cache 当作完整训练集或论文数据；
- 声称 mock loss 下降代表机器人控制有效；
- 自动启动 3 GPU 长训练。

## 2. Adapter 结构

```text
future [B,48,2,14,28]
  → Conv3d 48→256, kernel/stride [1,2,2]
  → [B,256,2,7,14]
  → LayerNorm + factorized T/H/W position
  → [B,196,256] K/V

action_encoder output [B,32,1024]
  → LayerNorm + Q projection
  → 8-head cross-attention, attention dim 512
  → output projection
  → x + tanh(gate) × residual
```

默认参数量精确为 `1,371,137`：

| 模块 | 参数 |
| --- | ---: |
| Conv3d projector | 49,408 |
| future LayerNorm | 512 |
| factorized position | 5,888 |
| action query LayerNorm | 2,048 |
| Q/K/V/out projections | 1,313,280 |
| scalar gate | 1 |
| 合计 | **1,371,137** |

gate 初始为 0，因此 A0 wrapper 与同一 B0 backbone 在固定输入上的 action 输出
逐元素完全一致。Adapter 不原地修改 action hidden。

## 3. 冻结与 hook

- backbone 全部 `requires_grad=false` 并保持 eval mode；
- wrapper 只允许 `adapter.*` trainable；
- optimizer 只接收 Adapter parameters；
- future context 用 context manager 激活；
- 每次 action prediction 检查 `action_encoder` hook 恰好调用一次；
- 异常退出必须清理 context，防止 future 泄漏到下一 batch。

真实 Phase C 已确认整个 backbone `requires_grad=false`、backbone gradient
count 为 0，并对 MoT 做 forward/backward 前后参数 hash。更细粒度的每个子模块
hash 必须在 Gate E 真实训练前扩充到训练前后可比的 frozen-backbone 锚点。

## 4. Phase B 可复现流程

先生成一次共享 mock cache：

```bash
source scripts/activate_env.sh

fastwam-ood thought3-plan-cache \
  --config configs/thought3/cache_smoke.yaml

fastwam-ood thought3-build-cache \
  --config configs/thought3/cache_smoke.yaml

fastwam-ood thought3-validate-cache \
  --config configs/thought3/cache_smoke.yaml
```

然后分别运行小型 mock：

```bash
fastwam-ood thought3-train \
  --config configs/thought3/train_a0_smoke.yaml

fastwam-ood thought3-train \
  --config configs/thought3/train_a1_smoke.yaml

fastwam-ood thought3-train \
  --config configs/thought3/train_a2_smoke.yaml

fastwam-ood thought3-train \
  --config configs/thought3/train_a4_smoke.yaml
```

中断后使用相同 config：

```bash
fastwam-ood thought3-train \
  --config configs/thought3/train_a1_smoke.yaml \
  --resume
```

identity 不一致时没有 `force` 或 `strict=false` 逃生口，必须新建 run。

## 5. 训练输入

Adapter 输入只有模型生成的 `future_latent` 与 mask。action flow batch 包含：

- current RGB/proprio/context；
- noisy action；
- action target 和 pad mask；
- model-generated future latent；
- identity/provenance。

action target 用于 loss，不能进入 future sampler。真实 future observation 不存在于
Trainer API。

## 6. 变体纪律

| 变体 | active K | Adapter 输入 | 是否训练 |
| --- | ---: | --- | --- |
| B0 | 0 | 无 | 不训练 |
| A0 | 0 | 全零 `[48,2,14,28]` | 独立训练 |
| A1 | 1 | K1 cache | 独立训练 |
| A2 | 2 | K2 cache | 独立训练 |
| A4 | 4 | K4 cache | 独立训练 |
| A-shuffle | K | 推理替换 | 不另训 |

所有 A0/A1/A2/A4 使用同一 Adapter class、参数量和 initialization seed。
动作噪声 seed 的推导不含 K/variant。

## 7. Checkpoint 内容

每个 checkpoint 目录只含：

```text
adapter.safetensors
optimizer.pt
manifest.json
```

不复制 6B backbone。manifest 绑定：

- official checkpoint/stats SHA-256；
- Fast-WAM commit；
- Adapter/config/split/cache fingerprint；
- variant/K/train seed；
- global step、sample cursor、world size；
- trainable name/count allowlist；
- frozen parameter hash；
- Adapter semantic-state SHA-256；
- 每个 checkpoint 文件 SHA-256；
- `contains_backbone=false`。

加载时逐项精确匹配，tensor key/shape 采用 strict load。

## 8. 记录指标

每 step 的 `train_metrics.jsonl`：

- action flow loss；
- gradient norm；
- raw gate 与 `tanh(gate)`；
- attention residual norm；
- trainable parameter count；
- step time；
- peak memory（Phase B CPU 为 0；真实运行记录 CUDA）；
- NaN/Inf；
- sample cursor。

每 checkpoint 原子刷新 metrics；因此 crash 后可从最近 commit 恢复。

## 9. Phase C 单卡结果

Phase C 已按以下顺序完成：

1. 获取一条标准 LIBERO train sample，确认没有 OOD/test 来源；
2. 只加载一个官方 Fast-WAM；
3. 生成 K=1/2/4 native future，核对 shape/dtype/device/schedule/noise；
4. 与 upstream joint/video path 做 same-seed parity；
5. gate=0 比较 B0/A0 action hash；
6. 执行一次 action-only loss backward；
7. 检查只有 Adapter 有 finite gradient；
8. 对 frozen 参数做 pre/post hash；
9. 记录 CUDA allocated/reserved peak，硬上限 43 GiB；
10. 完成真实 future mutation invariance。

全部硬门禁通过。精确 tensor、parity、gradient、memory、泄漏与工件 SHA 见
[thought3_phase_c_report.md](thought3_phase_c_report.md)。

## 10. Phase D 真实 cache smoke

Gate D 已完成：

- task：`open the middle drawer of the cabinet`；
- 完整 42 episodes 的 90/10 split 为 37 train / 5 development；
- pilot 选入 32 个不同 episode，cache 内为 28 train / 4 development；
- K1/K2/K4 共 96 条 BF16 `[48,2,14,28]` latent；
- 12 个 shard、96 个 metadata row、7,687,316 bytes；
- 32/32 base sample 的 paired seed 和 initial-noise hash 通过；
- 12/12 shard 文件/tensor/逐样本 checksum 通过；
- no-op resume 跳过 12/12 shard，且 `model_loaded=false`；
- 临时副本单字节损坏被拒绝，正式 cache 未改变；
- 只读取 64 张当前相机帧，future RGB/action target 均为 0；
- generation loop 39.70 s，0.806 base sample/s（不含模型加载）；
- 执行峰值 12.677 GiB，模型加载峰值 23.125 GiB。

权威机器结果、吞吐口径、冻结 SHA 和复核命令见
[thought3_phase_d_report.md](thought3_phase_d_report.md)。

## 11. Phase E 真实 smoke

Gate E 真实小训练先只做 A0/A1：

- 一个 suite、少量 task；
- 100–500 steps；
- microbatch 1；
- bf16；
- gradient accumulation；
- 一个 train seed；
- checkpoint selection 只看 development action loss；
- 不看正式 OOD success。

zero-init gate 的梯度有明确两步门禁：

1. 第 1 个 optimizer step 前，只有 gate 非零梯度是预期行为；
2. gate 被更新为非零后，从下一个有效 step 开始，projector、Q/K/V/out
   projection 等非 gate 参数必须出现 finite、nonzero gradient。

只看“Adapter 总 grad norm 非零”不足以通过 Gate E，因为它可能仍完全来自 gate。
训练日志必须按模块分别记录 gate 与非 gate grad norm，并至少保存首个
`non_gate_grad_nonzero=true` 的 step。

只有 loss/gate/分模块 gradient 可诊断、resume 一致、frozen hash 不变且单卡无
OOM，才扩到 A2/A4。三卡 DDP 和正式 ID/OOD rollout 仍需后续独立门禁。

### 11.1 2026-07-28 实际状态

Gate E 已真实执行，但总门禁未通过：

- A0/A1 各完成 resumed 50→100 和独立 uninterrupted 0→100；
- 第 1 step 只有 gate 非零 gradient，第 2 step 起
  projector/attention/non-gate gradient finite 且非零；
- 强制确定性 CUDA 后，两组 resumed/uninterrupted 最终 Adapter semantic
  SHA 分别完全一致；
- A0/A1 development-only checkpoint selection、Adapter-only checkpoint
  round-trip 和单卡显存通过；
- v2 A1 development 未低于初始化；v3 A0 fixed train probe 也未低于
  初始化，因此 `loss 有可诊断下降` 未通过；
- Gate 在训练后 frozen hash 之前 fail-closed，`frozen before==after` 尚未闭环。

该状态随后由 Gate E.1 单样本 fixed-noise overfit 继续诊断；Gate E v1–v3 的
权威数值、失败尝试 SHA 与边界见
[thought3_phase_e_report.md](thought3_phase_e_report.md)。

### 11.2 Gate E.1 实际状态

Gate E.1 已在预注册 commit `30ffc93` 上执行并通过：

- 同一条 train sample、同一 action noise/timestep，A0/A1 各 200 step；
- 固定 action loss 分别下降 92.93% 和 99.58%；
- 第 1 step gate-only，第 2 step non-gate gradient，最终 BF16 hidden delta
  非零；
- Fast-WAM frozen SHA before/after 完全相同；
- optimizer-step 峰值 13,273.17 MiB，无 future RGB/OOD/outcome。

但 A0/A1 step 200 的 `delta/action-hidden` 分别达到 1.91×/0.70×，所以
Gate E.1 只关闭“图是否能 overfit”问题，没有关闭多样本稳定训练问题。下一步是
先冻结 8-sample train-only LR/尺度诊断，再重跑完整 28/4 Gate E；A2/A4 仍锁定。
详见 [thought3_phase_e1_report.md](thought3_phase_e1_report.md)。

### 11.3 Gate E.2 实际结果

Gate E.2 已完成 A0/A1 的 8-sample train-only 工程诊断：

- LR 网格固定为 `1e-4 / 3e-4 / 1e-3`；
- 每条轨迹 200 step，共 1,200 optimizer step；
- 每 sample 固定 noise/timestep；
- 不读取 development/OOD/success outcome；
- 同时门控 fixed loss 与实际 BF16 `delta/action-hidden`；
- 多档通过时固定选择最小 LR；
- 六条轨迹的 execution、pairing、checkpoint、frozen SHA 和显存检查全部通过；
- 三个 LR 都没有达到 A0/A1 共同 `6/8 non-worsened`，总 Gate 失败；
- initial fixed loss max/min 为 94.28×，且与单 fixed flow timestep 的相关为
  `−0.93466`。

完整预注册和结果分别见
[thought3_phase_e2_protocol.md](thought3_phase_e2_protocol.md)、
[thought3_phase_e2_report.md](thought3_phase_e2_report.md)。

### 11.4 Gate E.3 held-out multi-flow 诊断

Gate E.3 不重新训练。它只读 E.2 六个 step-200 Adapter checkpoint，在未进入
E.2 optimizer 的 `flow_step=1..5` 上完成 320 个 action-loss forward：

- 每 sample 先跨五个 fixed flow draw 求均值；
- 再执行 E.2 原有 mean reduction、6/8、catastrophic 和 hidden-scale 门槛；
- LR 网格和 smallest-eligible 规则不变；
- 0 optimizer step、0 backward、0 development/OOD/success/rollout；
- v1 在官方 `timestep=1000, training_weight=0` 的合法零 loss 上触发了
  非门控 objective ratio 的实现错误，未产生 Gate 结论；
- v2 使用独立 schema/config/output，零权重 objective 仍进入 sample mean 和原
  门槛，只从未定义的非门控 `final/initial` ratio 遥测中排除；
- v2 已完成 320/320 forward，全部执行检查通过，但 A0/A1 在三个 LR 下均未达到
  `10% mean reduction + 6/8 non-worsened`，Gate 有效失败；
- E.2 的 fixed-flow 改善未迁移到 held-out flow，因此当前训练配方不能进入
  A2/A4 或 Phase F。

历史 v1、失败报告与 v2 预注册分别见
[thought3_phase_e3_protocol.md](thought3_phase_e3_protocol.md)、
[thought3_phase_e3_v1_failure_report.md](thought3_phase_e3_v1_failure_report.md)、
[thought3_phase_e3_v2_protocol.md](thought3_phase_e3_v2_protocol.md)、
[thought3_phase_e3_v2_report.md](thought3_phase_e3_v2_report.md)。

### 11.5 Gate E.4 paired diversified train-flow 诊断

E.3 之后唯一允许改变的训练变量是 action-flow objective：

```text
E.2：每条 sample 的所有 optimizer visit 都使用 flow_step=0
E.4：每次 visit 使用不同、确定性、A0/A1 配对的 train flow slot
```

E.4 必须保持：

- 同一 8 条 train sample 和 round-robin；
- A0/A1 × `1e-4/3e-4/1e-3`；
- 200 step、AdamW、weight decay、Adapter 和初始化；
- 官方 weighted velocity MSE；
- held-out probe `flow_step=1..5`；
- `10% + 6/8`、catastrophic 与 hidden-scale 门槛；
- 0 development/OOD/success/rollout/future RGB。

train flow slot 必须与 probe slot 不相交，并在每 step 保存 timestep、official
weight、noise/slot identity 和合法 zero-weight 计数。E.4 是序贯工程诊断，不是
future 模型效应实验。

E.4 已于 2026-07-28 完整执行：

- 六条轨迹共 1,200 optimizer steps、480 held-out objectives；
- 108 个 per-track execution checks、全部 cross/paired checks 通过；
- frozen Fast-WAM SHA 前后相同；
- 六条 held-out reduction 均为正，但只有 `0.997%–1.948%`；
- A1@3e-4 达 7/8 non-worsened，但 mean reduction 只有 1.787%；
- 三个 LR 均没有满足 A0/A1 共同 `10% + 6/8`，故有效失败。

因此不得进入完整 Gate E、A2/A4 或 Phase F。下一单变量诊断只能针对 optimizer
objective aggregation/effective batch；实现前必须冻结 matched-objective 或
matched-update budget、accumulation factor、flow slots 与 mean/sum loss。

冻结协议和结果分别见
[thought3_phase_e4_protocol.md](thought3_phase_e4_protocol.md)、
[thought3_phase_e4_report.md](thought3_phase_e4_report.md)。

### 11.6 Gate E.5 full-cohort objective aggregation

E.5 已按预先冻结的 `matched-optimizer-update` 预算完整执行：

```text
200 optimizer updates / track
8 frozen train samples / update
8 unique flow objectives / update
loss reduction = arithmetic mean
1,600 train objectives / track
```

唯一训练变化是把 E.4 的“单 sample objective 后立即更新”改成“同一次 update
内完整遍历 8 条 sample，对 `loss/8` 累积梯度后更新”。A0/A1、三档 LR、
Adapter/初始化、AdamW 更新次数、weight decay、checkpoint、held-out flow
`1..5` 和 `10% + 6/8` 门槛均保持不变。

训练 slots 冻结为 `20001..21600`，完整 pre-outcome identity schedule SHA 为
`b6f9778d303a6ad2c4bef781f4a6027a800d013814110daa47eb7cb1d13af86d`。
代码分别保存 1,600 条 objective 遥测和 200 条 update 遥测，可独立重算 mean
loss、每个 micro-objective 的 gate-gradient contribution 及 cancellation ratio。
24 个预知 `t=1000/weight=0` objective 原样保留。

该预算不是 sample/compute matched：E.5 的 train objective 数是 E.4 的 8 倍。
因此 E.5 与 E.4 的差异不能只归因于梯度聚合。

真实结果：

- 六条轨迹共 1,200 updates、9,600 train objectives 和 480 held-out
  objectives；
- 120/120 execution、21/21 paired、7/7 cross checks 全部通过；
- 六条 reduction 为 A0/A1：
  `1.889%/6.452%`、`2.638%/19.668%`、`2.890%/7.315%`；
- `A1@3e-4` 单条通过全部 performance checks，8/8 sample 不变差；
- 同 LR 的 A0 只下降 2.638%，所以三个 LR 仍全部不 eligible；
- frozen Fast-WAM SHA 前后相同，0 development/OOD/success/rollout/future
  RGB。

因此 E.5 是有效负总 Gate，不得进入完整 Gate E、A2/A4 或 Phase F。
`A1@3e-4` 的 19.668%/8-of-8 是需要在未使用 train cohort 上预注册复验的
探索性工程信号，不能事后充当 selected LR 或 future effect。

协议和结果见
[thought3_phase_e5_protocol.md](thought3_phase_e5_protocol.md)、
[thought3_phase_e5_report.md](thought3_phase_e5_report.md)。
