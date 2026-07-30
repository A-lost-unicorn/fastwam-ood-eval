# Thought3 Phase 0：E9a-v2.1 只读工件审计结果

状态：`COMPLETED / AUDIT VALID / SCIENTIFIC FAILED`

执行时间：2026-07-30

输出：`outputs/thought3/phase_e9_v2_1_readonly_audit_v1/`

## 1. 一句话结论

E9a-v2 的四条训练轨迹可从“engineering invalid”恢复登记为
**engineering valid + scientific failed**。训练本身完整、配对和 RNG
schedule 可恢复、原目录未被修改；但 normalized A1-vs-A0 的配对优势只有
`8.274%`，低于冻结的 `10%` 门槛，因此：

- 分类保持 `sample_tail_mitigation_not_supported`；
- `independent_replication_candidate=false`；
- E9b 保持锁定；
- 该结果不再阻塞 K=1 在线动作反事实。

## 2. 审计边界

本次审计只读既有 E9a-v2 工件，未执行：

- Fast-WAM 或 Adapter forward；
- backward 或 optimizer step；
- checkpoint tensor load；
- CUDA 初始化或 GPU 模型加载；
- reserve train 排序 17–28 的读取；
- development、OOD、success、rollout 或 future RGB 读取；
- 对父目录的补写、resume、覆盖或重训。

审计前后父目录 77 个文件完全一致，父 Gate SHA-256 为
`022e80868b56d7af7979e3c43a995061945ff7f615c6c21c5cf79256c8e25e24`。

## 3. 关键验证结果

`audit_result.json` 中 27 项 hard check 全部为 `true`。其中：

| 检查 | 结果 |
| --- | --- |
| 四轨训练完成度 | 每轨 200 updates / 1,600 train objectives |
| held-out probe grid | 每轨 initial/final 各 256 rows，严格 `8×32` |
| held-out flows | 精确为 `75..106` |
| timestep mismatch | 0 |
| weight mismatch | 0 |
| CPU/GPU weight 最大绝对差 | `4.76837158203125e-07`，低于 `1e-6` |
| zero-weight 位置 | 每个 initial/final 均精确为 `(1,80)`、`(7,93)` |
| 四轨 training schedule SHA | 同为 `d8a1dee3...52a0` |
| frozen Fast-WAM SHA | 审计前后不变 |
| CUDA | 前后均未初始化 |
| parent writes | 0 |

### RNG identity 的准确解释

父 probe row 原本没有保存：

- `action_noise_seed`；
- `action_timestep_seed`；
- `flow_objective_sha256`。

审计没有伪造“原始字段存在”，而是将其明确标记为
`not_observed_in_parent;deterministically_reconstructed`。冻结代码中的 seed
直接依赖：

- frozen namespace；
- train seed；
- flow step；
- sample ID。

`config fingerprint`、sample index 和 objective position 作为冻结 grid 的上下文
映射字段，没有直接编码进 seed。该映射在当前冻结 `8×32` grid 上是一一对应的，
共恢复 256 个唯一 identity。论文或简历不得写成“父 JSONL 原本完整保存了 RNG
identity”。

## 4. 科学结果

| 配方 | A0 reduction | A1 reduction | confirmed harm |
| --- | ---: | ---: | ---: |
| raw | 4.175% | 12.994% | A0=2，A1=0 |
| normalized | 2.983% | 11.010% | A0=0，A1=0 |

Normalization 呈现 tail-stabilization signal：A0 confirmed harm 从 2 降为 0。
但 normalized A1-vs-A0 paired advantage 为
`0.08273999465149648`，低于冻结阈值 `0.10`。因此不能登记为 sample-tail
mitigation 成功，也不能解锁 E9b。

## 5. 可证明与不可证明

本次结果可以证明：

- E9a-v2 的训练、配对 schedule 和 held-out grid 是可审计的完整工程运行；
- 原 invalid 原因是 writer/checker telemetry contract，而不是训练中断；
- normalization 有描述性的 tail-stabilization signal；
- 冻结的科学门槛仍然失败。

本次结果不能证明：

- future latent 真实改变了动作；
- future 的具体内容区别于任意 latent presence；
- future 提高了 Clean/OOD success；
- K=1 优于 K=2/K=4；
- E9 normalized checkpoint 应替换主在线 checkpoint。

Phase 1 继续固定使用 provenance 有效的 E6 A1@`3e-4` step-200 checkpoint，
不得根据在线动作差异回头挑选 E9 checkpoint。

## 6. 工件

| 工件 | SHA-256 |
| --- | --- |
| `audit_result.json` | `20f0662d6260563ddde7cc68614e7477b85d81fd7204f80ed98361a5f75a22b9` |
| `derived_identity_manifest.jsonl` | `331242c3ab858503746c01731a446b92157458973b196977fefed70c370f23e7` |
| `parent_artifact_manifest.json` | `d187f2aa84a3415ec0356e7a71ef76228a90eee43cdcab80cfb7d4c2a6cb73c8` |
| `run_status.json` | `ce3170427def457a0b91cc03d3dd45f295b593d1dfeba217615ba9d6a347955e` |

协议见
[thought3_phase_e9_v2_1_readonly_audit_protocol.md](phase_e9_v2_1_readonly_audit_protocol.md)。
