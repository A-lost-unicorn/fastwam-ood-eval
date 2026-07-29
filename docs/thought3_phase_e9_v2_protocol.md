# Thought3 Gate E.9a-v2：修复后预注册

状态：**PRE-REGISTERED / NOT RUN**

冻结日期：2026-07-29
证据等级：`POST-RUN SEQUENTIAL ENGINEERING`

## 1. 为什么需要 v2

E.9a-v1 在第一条 raw/A0 initial probe 前触发 evaluator 的旧 `1..5`
硬编码，完成 `0` 个 training objective、`0` 次 optimizer update，且没有产生
科学结果。v1 已作为 invalid engineering run 冻结，详见
[v1 失败报告](thought3_phase_e9_v1_failure_report.md)。

v2 只修复两项工程缺陷：

1. multi-flow evaluator 接受协议显式冻结的任意非空、无重复、正整数 flow 集；
2. setup 或 initial probe 失败时，子轨 `run_status.json` 必须从 `running`
   原子落盘为 `failed`，并记录 invocation ID、failure stage、error 和已完成
   objective 数。

不改变研究问题、训练配方、样本、RNG、预算、endpoint 或科学门槛。

## 2. 回归与兼容 contract

- E.9a-v2 的 held-out flow 必须精确等于 `75..106`，共 32 个 flow。
- 每次 initial/final probe 必须覆盖 `8 × 32 = 256` 个唯一 objective。
- aggregator 和 outcome checker 都从协议 flow 集推导 objective 数，不再假设
  flow 数量为 5。
- 空集、重复值、零、负数、布尔值和非整数必须 fail closed。
- 旧 Gate E.3 的专用 aggregation/outcome wrapper 仍严格冻结在 `1..5`；
  v2 不追溯修改 E.3 已执行结果。
- 自动回归测试必须构造完整 `75..106` grid，并验证 initial/final outcome。
- 故障注入测试必须证明 initial-probe 异常留下 `status=failed` 和
  `completed_objectives=0`，不能再留下假 `running`。

## 3. 不变的科学设计

四条 matched 轨迹仍为：

| Recipe | Variant | Future | 训练 objective |
| --- | --- | --- | --- |
| raw | A0 | K=0 | 八条 raw loss 的 arithmetic mean |
| raw | A1 | K=1 | 八条 raw loss 的 arithmetic mean |
| normalized | A0 | K=0 | 八条固定 sample-weighted loss 的 mean |
| normalized | A1 | K=1 | 八条固定 sample-weighted loss 的 mean |

冻结项如下：

| 项目 | v2 冻结值 |
| --- | --- |
| train cohort | E.6–E.8 train 排序 9–16，同八条样本 |
| sample payload SHA | `f5e61fd99d68244d7fa3cca6cc1ff59aabc12317840e4832ff2595f9ff78252f` |
| LR / endpoint | `3e-4` / step 200 |
| updates / objectives | 每轨 200 updates、每 update 8 objectives，共 1,600 |
| train flow slots | `40001..41600` |
| train identity SHA | `4c5c66f977e6f75dfaf3bb9db398a13c8a2807d6c065ae19307b19435440d64e` |
| held-out flows | `75..106` |
| held-out identity SHA | `76e96cb5be832908aff1510256bc058fa5023c8b71e51b57dfe6b3f277d899fb` |
| held-out zero-weight positions | `(sample 1, flow 80)`、`(sample 7, flow 93)`，均为 0-based sample index |
| weight calibration SHA | `edfb31e3fe1d6a8067a607ed20803ded33ba98f860c2a679067e70aa21105d70` |
| normalized weights SHA | `3e65b4f76f6cdee7176c49c9befd12bcd416fe9f60f2f719446a2896b05719f6` |
| action denoise steps | 20，四轨一致 |
| checkpoint endpoint | 只用 step 200 作科学判定 |

original absolute/paired Gate、32-comparison FWER paired bootstrap、
confirmed-harm 定义和互斥分类均完全沿用
[v1 归档协议](thought3_phase_e9_protocol.md)。不得选择 step 100，不得降低
E.6–E.8 门槛，也不得根据 v2 中间结果重设 sample weights。

## 4. 一次性独立复验继续锁定

train 排序 17–28 在 E.9a-v2 中仍为 identity-only reserve：

- cohort SHA：
  `0218d90eb6455d3297857423bfd34109469f308db9f69d5adeee02146ee42324`
- reserved flow：`107..138`
- identity SHA：
  `d5aeb3df50bbf11940ba545318327fd08df7f1e83dc27d7e3026ff6ed70b4f64`

v2 不解码、不训练、不 probe 这 12 条样本。只有 E.9a-v2 产生
`independent_replication_candidate=true` 时，才可另行执行一次性只读 E.9b。

## 5. 新身份与输出隔离

| 项目 | v2 值 |
| --- | --- |
| schema | `thought3.phase_e9.sample_tail_mitigation.v2` |
| config | `configs/thought3/phase_e9_sample_tail_mitigation_v2.yaml` |
| config fingerprint | `afcd6a7fdad6338a2a09a0d881494ea9e5e2a0c9fd9495d712fcafeb5507097a` |
| runner | `scripts/run_thought3_phase_e9_sample_tail_mitigation_v2.sh` |
| output | `outputs/thought3/phase_e9_sample_tail_mitigation_v2/` |
| confirmation | `CONFIRM_THOUGHT3_PHASE_E9A_V2=YES` |

v2 启动前会验证 v1 的五个冻结工件 SHA、`result=null`、零训练工件和 frozen
Fast-WAM SHA 不变。v1 目录绝不能作为 v2 resume source。

## 6. 正式运行门禁与命令

权威预注册版本是首次同时包含本协议、v1 失败报告、v2 配置、实现、runner 和
回归测试且 worktree clean 的 git commit。正式运行前必须记录：

```bash
git status --short
git rev-parse HEAD
```

`git status --short` 必须无输出。唯一正式入口为：

```bash
CONFIRM_THOUGHT3_PHASE_E9A_V2=YES \
THOUGHT3_GPU_ID=1 \
bash scripts/run_thought3_phase_e9_sample_tail_mitigation_v2.sh
```

runner 仍为单卡进程，只接受一个物理 GPU ID；显存已用超过 1 GiB 时拒绝启动。
正常中断只能在同一 v2 Run ID 上使用 `--resume`。不要同时运行 v1 runner，
不要把两个 GPU ID 写入 `THOUGHT3_GPU_ID`。

## 7. 结论边界

当前 v2 状态仍是 `NOT RUN`，所以没有 raw/normalized、A0/A1、tail mitigation
或 candidate 结果。代码和回归测试通过只证明修复后的工程 contract，不证明
sample normalization 有效，更不证明未来 latent 改善 OOD。

完整 Gate E、A2/A4、在线 latency 和 ID/OOD rollout 在 E.9a-v2 及必要的 E.9b
通过前继续锁定。
