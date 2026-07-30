# Thought3 Phase B：CPU/mock 实现验收报告

状态：**完成（历史 Phase B 验收；后续 Gate C 已通过）**
验收日期：2026-07-27
证据等级：`TEST`，无真实模型、训练或 OOD 结论

## 1. 验收结论

用户已确认 Phase A 设计，Phase B 已在严格隔离条件下完成：

- 未加载 12 GB 官方 checkpoint；
- 未调用 GPU；
- 未运行真实训练；
- 未修改 `third_party/FastWAM`；
- 未修改 Thought1/Thought2 正式输出；
- 旧 CLI 与全量测试继续通过。

本文保留 2026-07-27 的 Phase B 验收快照。后续 Gate C 已于
2026-07-28 通过，真实结果见
[thought3_phase_c_report.md](phase_c_report.md)。Phase B 结束时的状态为：

```text
Phase A 设计审计       ✅
Phase B CPU/mock 系统  ✅
Phase C 单 GPU tensor  ⛔ 尚未开始
Phase D 真实 cache     ⛔
Phase E 真实小训练     ⛔
Phase F 技术 pilot     ⛔
```

## 2. 已实现组件

| 组件 | 实现 |
| --- | --- |
| Adapter | Conv3d projector、factorized position、8-head cross-attention、zero gate、mask |
| 注入 | `action_encoder` output scoped hook、exact call count、异常 cleanup |
| 冻结 | backbone eval/frozen、`adapter.*` trainable allowlist、frozen hash |
| K sampler | 完整 K=1/2/4 shifted schedule、CPU float32 paired noise、current slice 固定 |
| 数据 | suite×task episode split、stable base/cache ID、source allowlist |
| Cache | safetensors shard、atomic commit、resume、文件/tensor/sample checksum |
| Validation | K pairing、seed/noise hash、shape/dtype/source、corruption detection |
| Shuffle | 跨 task/episode 的确定性一一 derangement |
| Counterfactual | correct/null/shuffle/random/different-K action metrics |
| Trainer | CPU mock action velocity loss、grad/gate/step metrics、NaN guard |
| Checkpoint | Adapter-only state、optimizer/cursor/provenance、strict resume |
| Online boundary | API 不接受训练 cache；stage-separated latency schema |
| CLI | 9 个惰性 `thought3-*` 命令，全支持 help/dry-run/config/resume/device |
| Config | 12 份 Phase B/pilot/formal template，未知 key fail-closed |

## 3. 关键已测事实

- 默认 trainable params：`1,371,137`；
- A0/A1/A2/A4 参数量、state keys、structural fingerprint 相同；
- zero gate 下 A0 action 与 B0 action 逐元素完全相同；
- native cache schema：`[48,2,14,28]` / sample；
- 同一 base sample 的 K1/K2/K4 初始 seed 与 noise hash 相同；
- K1/K2/K4 全部走到 sigma=0；
- 人为翻转 cache 单字节能被 checksum 检测；
- resume 不重复已完成 shard；
- interrupted+resumed 与 uninterrupted mock 训练的最终 Adapter semantic hash 相同；
- checkpoint round-trip 后固定输入输出逐元素相同；
- mock development action loss 可下降；
- shuffle donor 不与 recipient 同 sample、episode 或 task；
- online evaluator 不存在 cache 参数，mock manifest 记录
  `online_cache_read=false`；
- dry-run 独立进程不 import `torch`/`safetensors`，不加载 checkpoint，不写输出。

## 4. 测试证据

Phase B 验收命令：

```bash
.conda/envs/fastwam-ood/bin/pytest -q
```

最终验收结果：

```text
235 passed, 5 warnings in 20.23s
60 Thought3 tests collected
```

5 条 warning 均为受限测试进程无法初始化 NVML，不是测试失败。`git diff --check`
与 Thought3 `compileall` 同时通过。

## 5. 冻结输出哨兵

验收使用八个只读 SHA-256；最终回归必须再次完全一致：

| 文件 | SHA-256 |
| --- | --- |
| Thought1 experiment manifest | `57dd93f51a2491423f1b14f0d90523f219218698e231a133dcef114caca132ee` |
| Thought1 metrics | `0aa1173038a1c37d37123570a83ff9f08667490e3f94276345c802151897dbb5` |
| Thought1 report | `889d567e4882b9982fb2121788dbbacdf983e1556faf8e4f9bb5a29768f8e137` |
| Thought2 diagnostic manifest | `ff4aef249d800dbcec44d8f319d89efbebd0c7b1be78a99213eb7e3b2f1d7e09` |
| Thought2 diagnostic metrics | `b47d29f5c176fea74797f32f872fc14d9c23f370bea8c667cfccf4ccdfc942c3` |
| Thought2 analysis manifest | `ad3793b3ef8a2042c6eff90c0f238ea56f1afdda62a3e79dbb6518dede1fe76f` |
| Thought2 formal analysis | `9d51e0f46c7af73340b390c3acdfd30fa05c8d1e2fa92794ebcae0f112c69f19` |
| Thought2 run status | `32128801f41bfad982645fb2a8358df40bee638206623805bc2dedf6d13be718` |

## 6. Phase B 本地演练

```bash
source scripts/activate_env.sh

fastwam-ood thought3-plan-cache \
  --config configs/thought3/cache_smoke.yaml
fastwam-ood thought3-build-cache \
  --config configs/thought3/cache_smoke.yaml
fastwam-ood thought3-validate-cache \
  --config configs/thought3/cache_smoke.yaml

fastwam-ood thought3-train \
  --config configs/thought3/train_a1_smoke.yaml
fastwam-ood thought3-counterfactual \
  --config configs/thought3/train_a1_smoke.yaml
fastwam-ood thought3-evaluate \
  --config configs/thought3/train_a1_smoke.yaml
fastwam-ood thought3-aggregate \
  --config configs/thought3/train_a1_smoke.yaml
fastwam-ood thought3-report \
  --config configs/thought3/train_a1_smoke.yaml
```

这些命令会写 `outputs/thought3/**`，不会触碰阶段一/二。若只检查计划，给任意
Thought3 命令加 `--dry-run`；不会创建目录。

## 7. Phase C 准入与阻塞

当前不应直接跑 A0/A1/A2/A4 长训练。Phase C 的两个外部前提：

1. 标准 LIBERO train demonstration 路径与不可变 revision；
2. 单卡可以安全加载官方 checkpoint 的窗口。

Phase C 只做一条真实样本：

- K1/K2/K4 latent；
- upstream parity；
- Adapter forward；
- 单次 action loss backward；
- only-Adapter gradient；
- frozen hash；
- zero-gate parity；
- future-mutation leakage test；
- CUDA peak <43 GiB。

通过后才能进入一个 suite/task 的真实 cache smoke。

## 8. 简历可用表述

当前只能写工程成果，不能写效果提升：

> 设计并实现面向 Fast-WAM 的 1.37M 参数 Future-to-Action Adapter 实验系统，
> 通过 zero-gated cross-attention 将 K=1/2/4 原生 future latent 接入冻结动作
> 分支；构建可分片/断点恢复/逐级 checksum 的 safetensors cache、Adapter-only
> checkpoint、跨 task/episode 反事实置换与信息泄漏门禁，并以全量回归验证旧
> ID/OOD 评测链路不受影响。

在真实 Phase F/G 前不得追加“提升 OOD 成功率”。
