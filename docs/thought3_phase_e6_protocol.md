# Thought3 Gate E.6：未使用 Train Cohort 的序贯复验协议

状态：**已预注册、已实现、尚未运行**

冻结日期：2026-07-29

> 本协议是在查看 Gate E.5 结果之后建立的。`3e-4` 来自 E.5 的探索性结果后
> 选择，因此 E.6 是 post-selection sequential replication，不是独立确认性
> 实验。代码完成后不会自动启动 GPU；真实运行仍需显式确认。

## 1. 问题与结论边界

E.5 的预注册共同门槛有效失败：`A1@3e-4` 的 held-out mean action loss
下降 `19.668%`、`8/8` 样本不变差，而匹配的 `A0@3e-4` 只下降
`2.638%`。E.5 因此没有 selected LR。

E.6 只问：

> 这个事后发现的 A1 工程信号，能否在未被 E.5 使用的 train demonstrations、
> 全新训练 flow objectives、匹配的 A0/A1 配方下复现？

E.6 不读取 development、OOD、success outcome，不做 rollout，也不使用真实未来
RGB。无论结果如何，都不能直接解释成 future 提升了 OOD，不能追溯把 E.5
改判为通过。

## 2. 冻结父证据与 LR 披露

运行前逐文件校验 E.5：

| 工件 | SHA-256 |
| --- | --- |
| `gate_e5_result.json` | `c797a98f646855a9b37caa7e251c97e8001d2d4aecb7efbcb5a539f77911f7bd` |
| `run_status.json` | `cdc5944d35a03309230206ef817b75b17c1dbdea4b8f1706b98c1e7cec514f37` |
| `pre_validation_result.json` | `63061d304a4a3c77c4e95f782d061be478b2e03a7dd88a39de8861f8ccde63ae` |
| `data_preparation.json` | `ef95e5972ccabc455e7781afae19582f4f7880eb9e8800f0cd3e0a152f7261b6` |
| `logs/phase_e5.log` | `fc334690b893555c09d36a2eb288e562b6e8454531d601570a544b91911d8582` |

结果元数据必须写明：

- `learning_rate_chosen_after_e5=true`；
- `learning_rate_selected_by_e5_frozen_gate=false`；
- `independent_confirmatory_test=false`；
- `thresholds_chosen_after_e5=true`。

## 3. 冻结样本

使用 Phase D K=1 cache 的 28 条 train 样本，按既有
`_training_order_key(base_sample_id, seed=3407)` 排序，取 1-based
位置 `9–16`。这八条均属于不同 demonstration，与 E.5 八条和四条
development 样本的交集均为零。

| 顺序 | episode | base sample ID |
| ---: | --- | --- |
| 1 | `episode_000014` | `9610d2aed3a6ddf382c514715ead977c9f9a25b56265b2705a9146ac28f6c0cc` |
| 2 | `episode_000010` | `75359438f810e6921754de327beda8bd974343f5e89fb54d7ac8852f79c89c9b` |
| 3 | `episode_000011` | `5f82a5db9be7a61f969fd32f5bca19dbb19a65106fb49d5357705be2d03def44` |
| 4 | `episode_000030` | `8f34793be5e051e0d62c0397b83cc341f17b626bd73660968f48ff1f6339d1b9` |
| 5 | `episode_000019` | `8c00174e915504c49a3c69057f9c199af1654a6ecef414070c1657316b1e4418` |
| 6 | `episode_000038` | `461a673f2745ab243d99d617f4514a737644d44ba2fc5fdece8b45f347e51564` |
| 7 | `episode_000000` | `739baab482230ba4ee1ae9c0cccf5886268db9ee37c895435af6c6891d22c3b0` |
| 8 | `episode_000012` | `81363feff988d3f3faaeeb66191e7ff9c4fd40c85d7b3b7cd0bda84cd41e3b9b` |

- cohort SHA：
  `6a354151d6d3e93335b66743f16be1908abc8d0fe835ee3811562b2eeb63d7c3`
- Phase D split SHA：
  `ea5402955023ccd48d790d821a73f98549b31d1ace8af035a90ceae2ad3951eb`

## 4. 冻结训练配方

只运行两条轨迹：

| Track | Future input | LR | Updates | Objectives/update |
| --- | --- | ---: | ---: | ---: |
| A0 | null future，K=0 | `3e-4` | 200 | 8 |
| A1 | cached K=1 future latent | `3e-4` | 200 | 8 |

两条轨迹保持同一 Adapter、zero-gate 初始化、AdamW、weight decay、20-step
action denoising 语义、官方 weighted velocity MSE、checkpoint 50/100/150/200
和 held-out flow steps `1..5`。每个 update 对八条样本各取一个 objective，
对 `loss/8` 做 arithmetic-mean gradient accumulation。

总预算固定为：

- 2 tracks；
- 400 optimizer updates；
- 3,200 train objectives；
- 160 held-out objectives；
- 单 GPU，预计约 45–55 分钟，其中模型加载约 9 分钟。

## 5. 全新 Flow Namespace

训练 flow slots 固定为 `31001..32600`，不与 held-out `1..5`、E.4
`10001..10200` 或 E.5 `20001..21600` 重叠。

- offset：`31000`
- pre-outcome identity schedule SHA：
  `419b09a2ec30ce7bffc99c95aff1a343f77d39e83e77a752fc67bc984508febc`
- 预知 `t=1000/weight=0` objective：19 个；
- 前两个 update 不含 zero-weight endpoint，必须重新观察到 update 1 只有
  scalar gate 获得梯度、update 2 projector/attention 开始获得非零梯度。

每个 objective 和 update 的 telemetry、prefix SHA、checkpoint marker、resume
身份和 observed schedule SHA 都绑定到 E.6 protocol；E.5 checkpoint 不允许
冒充 E.6 恢复点。

## 6. 冻结通过门槛

所有工程、配对、冻结、无泄漏和内存检查必须全部通过，此外：

### A1 绝对复现

- held-out mean loss reduction `>=10%`；
- `>=6/8` sample 不变差；
- catastrophic sample `=0`；
- median gated-delta/action-hidden `<=0.5`；
- 任一 objective 最大 ratio `<=1.0`。

### A0 稳定性负对照

- held-out mean loss 不变差，即 reduction `>=0%`；
- `>=6/8` sample 不变差；
- catastrophic sample `=0`；
- 同样满足两个 hidden-scale 上限。

A0 不要求达到 A1 的 `10%`：它是 null-future 稳定性负对照。这是 E.6 的新门槛，
不追溯修改 E.5 的共同 `10%` 门槛。

### A1 相对 A0 的配对优势

- A1 final sample-equal mean loss 至少比 A0 低 `10%`；
- 八条相同 sample 中至少 `6/8` 的 A1 final loss 不高于 A0。

该相对门槛是在看到 E.5 的 A1-vs-A0 差异后设定，也必须按 post-selection
披露。

## 7. 决策与停止规则

- 全部通过：只允许把 `A0/A1@3e-4 + full-cohort mean` 冻结为新的完整
  28-train/4-development Gate E 候选协议。
- 任一科学门槛失败：保留有效负结果，不调阈值、不换 cohort、不覆盖 Run ID。
- execution/provenance 失败：保留无效工件，修复后必须使用新 Run ID。
- E.6 不直接解锁 A2/A4、Phase F 或 OOD rollout；仍需新的完整 28/4 Gate E
  通过。

## 8. 运行与恢复

预注册本身不启动以下命令。获得运行授权并确认单卡空闲后：

```bash
CONFIRM_THOUGHT3_PHASE_E6=YES \
THOUGHT3_GPU_ID=1 \
bash scripts/run_thought3_phase_e6_fresh_cohort_replication.sh
```

中断后只可对同一合法 Run ID 使用：

```bash
CONFIRM_THOUGHT3_PHASE_E6=YES \
THOUGHT3_GPU_ID=1 \
bash scripts/run_thought3_phase_e6_fresh_cohort_replication.sh --resume
```

输出根为 `outputs/thought3/phase_e6_fresh_cohort_replication_v1/`。完整失败结果
不可用 `--resume` 覆盖。
