# Thought3 Gate E.9a-v1 失败报告

状态：**INVALID ENGINEERING RUN / FROZEN / DO NOT RESUME**

运行日期：2026-07-29
预注册 commit：`94255fa98499070cf89d20b48d101ba3696f3462`

## 1. 结论

E.9a-v1 没有跑完，也没有产生可分析的训练结果。它在第一条
`raw/A0` 轨迹的 initial probe 开始前，因为共用 multi-flow evaluator
只接受旧 Gate E.3 的 `1..5`，拒绝了 E.9 协议冻结的 `75..106`：

```text
RealTrainingError:
Gate E.3 multiflow probe requires flow steps 1..5
```

本次运行只完成了模型加载和八条真实 LIBERO train 样本准备。已确认：

- training objectives：`0`
- optimizer updates：`0`
- checkpoints：`0`
- `gate_e9a_result.json`：未生成
- 根状态：`failed`，`result=null`
- frozen Fast-WAM 参数 SHA：前后均为
  `ac0dd59de78495dc8ed5c5601e24951b78f3fd6ad40843d76b83106eccceb4f8`

因此它既不是负科学结果，也不能用于比较 raw/normalized 或 A0/A1。

## 2. 根因与影响边界

`ObjectiveAggregationProtocol` 已正确携带任意显式 held-out flow 集，但
`evaluate_multiflow_subset_probe()` 仍保留 Gate E.3 专用硬编码。这是 evaluator
复用错误，不是样本、cache、checkpoint、loss、梯度或显存失败。

异常发生在 Adapter objective、backward 和 optimizer 之前，所以没有训练状态
污染。子轨 `tracks/raw/a0/run_status.json` 仍显示 `running`，则是第二个独立的
状态机缺陷：原失败处理只覆盖 training loop 内异常，没有覆盖 initial probe。
根 runner 已正确以非零状态结束，当前没有残留进程。

## 3. 冻结证据

| v1 工件 | SHA-256 |
| --- | --- |
| `run_status.json` | `0542dc8b535e640cdde5be7ffa9021f312b78dbeb43e0e4f7da55ebef78b4658` |
| `pre_validation_result.json` | `e325211ac58f2021c6f5815325b1c205ad77145535147fa16c98f1d7e90dbe4f` |
| `data_preparation.json` | `f7c779eceaaff3eb0dfeab03757ab8c6aacf2004227396fd708f7aaf7f74695b` |
| `tracks/raw/a0/run_status.json` | `336c65b89f67c509d28038a8781ec96d2752966ce49ea3941308e8a20fb137f0` |
| `logs/phase_e9a.log` | `76db28f18b4f67dcd73263f02f32eabd809a2f015893cebd14d1b9bfb3485bfd` |

代码在 E.9a-v2 启动前逐项验证这些 SHA、`result=null`、零训练工件和 frozen
参数不变；任一项变化都会 fail closed。

## 4. 处置

- 永久保留
  `outputs/thought3/phase_e9_sample_tail_mitigation_v1/`，不修改、不覆盖、
  不 `--resume`。
- v1 runner 已改为只读归档拒绝器，固定返回 exit code 2。
- 修复只进入全新的 E.9a-v2 配置、runner、schema 和输出目录。
- E.9a-v2 保持 v1 的 cohort、权重、LR、200-step endpoint、train/probe flow
  身份、统计门槛和 E.9b reserve 不变。
- 完整 Gate E、A2/A4 和 ID/OOD rollout 继续锁定。

修复后的冻结边界与实际运行状态见
[E.9a-v2 协议](phase_e9_v2_protocol.md)和
[E.9a-v2 结果报告](phase_e9_v2_report.md)。
