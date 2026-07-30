# E.9a-v2.1 只读 Artifact Audit 预注册

状态：`FROZEN BEFORE AUDIT EXECUTION`

## 1. 问题与允许结论

本审计只回答一个工程证据问题：E.9a-v2 的 held-out probe 虽未把
`action_noise_seed`、`action_timestep_seed` 和 `flow_objective_sha256`
写入行内，这三项是否仍能由冻结运行代码和已保存的 objective grid 唯一重建。

只允许两个顶层输出：

1. `audit_valid_scientific_failed`
2. `audit_invalid_identity_unrecoverable`

若通过，含义是：

- 原始 v2 工件仍保持 `status=invalid`，不回写、不改名；
- v2.1 作为独立 post-run adjudication，将该次计算登记为
  `engineering valid + scientific failed`；
- 不改变冻结的科学结果，不解锁 E.9b。

## 2. 严格只读边界

审计禁止：

- model forward、backward、optimizer、checkpoint load；
- CUDA 或 GPU 模型；
- resume、重训、新 flow；
- 修改或补写 E.9a-v2 原目录；
- 读取 reserve 17–28、development、OOD、success、future RGB；
- 把派生 identity 冒充为原始 observed telemetry。

唯一新写入位置是：

`outputs/thought3/phase_e9_v2_1_readonly_audit_v1`

## 3. 冻结父工件与代码

父目录、17 个根/四轨核心 SHA、父配置 fingerprint、运行 commit、Fast-WAM
commit、四个运行代码 Git blob、运行首末时间均冻结于：

`configs/thought3/audits/phase_e9_v2_1_readonly_audit.yaml`

审计开始和结束各对父目录全部 77 个文件计算 SHA-256；两个快照必须完全相同。
已在 E.9a-v2 报告登记的核心 SHA 还必须逐项匹配配置。

## 4. Identity 重建规则

运行 commit `694d1d0...` 的冻结代码规定：

```text
action_noise_seed =
  stable_seed("thought3-real-action-noise-v1",
              train_seed, flow_step, base_sample_id)

action_timestep_seed =
  stable_seed("thought3-real-action-time-v1",
              train_seed, flow_step, base_sample_id)

flow_objective_sha256 =
  SHA256(base_sample_id, train_seed, flow_step,
         action_noise_seed, action_timestep_seed)
```

`stable_seed` 是 NUL 分隔字段的 SHA-256 前 8 bytes，并清除 signed-64 高位。
objective digest 沿用 v2 的 legacy unnamespaced 格式；本审计不会事后增加
namespace。

每条派生记录还保存：

- parent config fingerprint；
- `sample_index`；
- `flow_index`；
- `objective_position = sample_index * 32 + flow_index`；
- seed namespaces；
- initial/final 以及四轨中对应位置。

这些上下文字段证明保存 grid 到 deterministic call path 的一一映射；实际 seed
公式只依赖冻结 namespace、train seed、flow step 和 sample ID。报告必须明确
披露这一点，不能声称 config fingerprint/sample index 被编码进原 seed。

## 5. Hard checks

全部为真才允许 valid：

- 四轨各有 initial/final 两个 probe，每个精确 256 rows；
- flow 精确为 `75..106`，sample 精确为冻结的 8 条 E6 cohort；
- initial/final、raw/normalized、A0/A1 objective position 精确配对；
- 2,048 个保存行的 BF16 timestep 与 CPU 重建完全相等；
- 官方 weight 与 CPU 重建绝对误差不超过 `1e-6`。该容差在运行前冻结，
  只吸收 CPU/GPU `exp` 的最后一位差异；
- zero-weight 位置精确为 `(sample_index, flow_step)=(1,80),(7,93)`；
- raw/normalized 的 schedule 相同；
- A0/A1 除 future 条件和合法的 variant/output metadata 外匹配；
- 四条轨迹均完成 200 updates、1,600 train objectives；
- 原父 Gate 唯一失败的 execution check 是缺失的 held-out identity telemetry；
- frozen Fast-WAM SHA before/after 相同；
- scope 显示 reserve 解码 0、development/OOD/success false、future RGB 0；
- tail bootstrap 和最终性能分类可从父 JSONL 重算；
- 父目录审计前后 SHA 快照完全相同。

## 6. 冻结科学判定

即使工程审计通过，也必须登记：

- raw A0/A1 reduction：`4.175% / 12.994%`；
- normalized A0/A1 reduction：`2.983% / 11.010%`；
- raw confirmed harm：`2`；
- normalized confirmed harm：`0`；
- normalized A1-vs-A0 paired advantage：`8.274% < 10%`；
- `sample_tail_mitigation_not_supported`；
- `independent_replication_candidate=false`；
- E.9b locked。

E.9a-v2.1 是证据账本修复，不是新的 surrogate Gate，并且不阻塞 K=1 online
counterfactual。
