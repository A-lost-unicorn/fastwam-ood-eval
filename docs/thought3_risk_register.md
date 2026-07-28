# Thought3 风险登记与停止条件

状态：Phase A–D 已通过；Gate E 未通过；Gate E.1 工程诊断已通过
范围：Partial-Future Adapter 的数据、cache、训练、推理、统计和阶段隔离

## 1. 等级

影响：

- `Critical`：会使科学结论无效、污染冻结结果或发生真实未来泄漏；
- `High`：会造成主要对照失效、训练不可恢复或正式资源浪费；
- `Medium`：会降低效率、稳定性或解释力；
- `Low`：可局部修复，不改变研究结论。

状态：

- `Open`：尚未通过对应 gate；
- `Controlled`：已有设计控制，但仍需测试；
- `Closed`：证据已保存且通过；
- `Accepted`：无法消除但会在论文限制中如实报告。

## 2. 风险总表

| ID | 风险 | 概率 | 影响 | 检测证据 | 控制与停止条件 | Gate | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R01 | Adapter 输入真实未来图像或其 VAE latent | 中 | Critical | schema audit、future-mutation invariance test、provenance | training API 禁止 future observation 字段；一旦输出随真实后续 RGB 改变立即停止 | B/C | Controlled |
| R02 | 在线正式评测读取训练 cache | 中 | Critical | file-open guard、online evaluator test、manifest | evaluator 不接收 cache path；发现任何 cache read 立即作废该 run | B/F/G | Controlled |
| R03 | 把 K 当重复编码、普通 forward 或 action steps | 中 | Critical | scheduler metadata、actual timestep/delta | K 只计 video scheduler update；不匹配即 fail-fast | B/C | Controlled |
| R04 | K=1/2/4 起始噪声不同 | 中 | High | initial noise hash/seed、paired sample audit | seed 从 base sample ID 推导且不含 K；任一 pair 不同即拒绝 cache | B/D | Controlled |
| R05 | K cache 与错误 sample 对齐 | 中 | Critical | base/cache ID、source hashes、random online recompute | fingerprint 全匹配；禁止仅按数组位置连接 | B/D | Controlled |
| R06 | cache 损坏、半写或 resume 重复 | 中 | High | file/per-sample checksum、sample cardinality | temp+fsync+atomic rename；只跳过完整且校验通过的 shard | B/D | Controlled |
| R07 | A-shuffle 意外取到同 episode/task | 中 | Critical | recipient→donor manifest、derangement validator | 排除 same sample/episode/task；无合法 donor 就失败 | B/F | Controlled |
| R08 | A-shuffle 被另训后学会忽略错误输入 | 中 | High | checkpoint identity | shuffle 是同一个 A-K checkpoint 的 inference intervention，不另训 | F | Controlled |
| R09 | A0 结构/参数/预算与 A-K 不匹配 | 中 | Critical | structural fingerprint、param count、step manifests | A0 零 latent 走同一 Adapter；除 future source 外配置一致 | B/E/F | Controlled |
| R10 | B0 与 zero-gate A0 初始动作不一致 | 中 | High | fixed-seed action allclose/hash | 初始差异超预注册容差即停止，先修注入路径 | B/C | Controlled |
| R11 | 注入 hook 未调用、重复调用或泄露上一 batch context | 中 | Critical | hook counter、context lifecycle test | context manager + exact call count + finally cleanup；异常即失败 | B/C | Controlled |
| R12 | 中层 hook 与 gradient checkpoint 重算冲突 | 中 | High | forward/backward call count、grad comparison | v1 改用 action encoder 输出；中层方案不进入第一版 | B | Controlled |
| R13 | zero gate 长时间不开，Adapter 实际忽略 future | 高 | High | gate、submodule grad norm、counterfactual | Gate E.1 已证实 step 2 non-gate gradient 和单目标可拟合；多样本与反事实使用仍需 E/F | E/F | Controlled |
| R14 | Adapter attention mask 全 false 导致 NaN | 低 | High | finite test、mask validator | 每 sample 至少 1 token；masked softmax 单测；NaN 立即停止 | B/C | Controlled |
| R15 | 上游 full `training_loss()` 把真实 future 带入图 | 高 | Critical | call audit、forbidden API test | 新 action-only loss；Thought3 trainer 禁止调用上游 full loss | B/C | Controlled |
| R16 | action loss 与官方 flow target/weight 不一致 | 中 | Critical | formula parity test、fixed tensor reference | 逐项复用 scheduler/target/mask/reduction；不一致即停止 | B/C | Controlled |
| R17 | action normalization/stats 漂移 | 中 | Critical | stats/config hash、round-trip test | 固定发布 stats，不重算；加载时强制 SHA-256 | B/C/F | Controlled |
| R18 | train/dev frame 泄漏自同一 episode | 中 | Critical | split intersection audit | suite×task 分层、episode 级 split；任一 episode 交集非空即失败 | B/D | Controlled |
| R19 | 使用 Thought1/2 或 OOD 正式轨迹训练/调参 | 中 | Critical | source-root allowlist、split manifest | 训练只允许标准 LIBERO demo roots；检测到 outputs/或 LIBERO-Plus 即失败 | B/D/E | Controlled |
| R20 | 用最终 OOD 反复选择 LR、层数或 checkpoint | 中 | Critical | experiment ledger、protocol timestamps | 只用 development/pilot；正式 manifest 与协议先冻结 | F/G | Controlled |
| R21 | backbone 参数被 optimizer 或 backward 更新 | 中 | Critical | requires-grad allowlist、grad audit、pre/post hash | optimizer 只含 Adapter；发现 frozen grad/hash 改变立即停止并作废 checkpoint | B/C/E | Controlled |
| R22 | upstream checkpoint `strict=False` 掩盖 identity 问题 | 中 | High | checkpoint/config/stats hash、live module shape | Thought3 checkpoint 绑定官方 SHA；禁止仅凭文件名恢复 | B/C | Controlled |
| R23 | Adapter-only resume 搭配了不同 backbone/cache | 中 | Critical | resume manifest verification | 所有 identity 精确匹配；不提供 `strict=False`/force override | B/E | Controlled |
| R24 | optimizer resume 恢复但 data cursor/RNG 未恢复 | 中 | High | uninterrupted vs resumed hash test | 保存 rank RNG、epoch、sample cursor；输出逐步一致性测试 | B/E | Controlled |
| R25 | 单卡训练 OOM | 中 | High | peak allocated/reserved、preflight | C/E/E.1 均低于 43 GiB；配方扩展后仍保留 hard abort | C/E | Controlled |
| R26 | 三卡 DDP shard 重复或遗漏 | 中 | High | union/intersection/cardinality test | hash/ordered deterministic shard；重复或缺失即不运行 | B/D/E | Controlled |
| R27 | FSDP/ZeRO 增加复杂度但无实际收益 | 中 | Medium | memory profile | v1 先普通 DDP；只有实测 OOM 才升级 | C | Controlled |
| R28 | cache 磁盘不足或产生过多 inode | 中 | High | plan-cache 容量预检、free-space check | Phase D 96-entry pilot 已通过；扩大正式 cache 前重新估算并保留 20% | D | Controlled |
| R29 | 把 Thought2 3.355 s 当作 K-step 在线延迟 | 高 | High | latency boundary manifest | 新 CUDA event 分段计时；旧数值只作上界背景 | C/F | Controlled |
| R30 | 离线 cache 读取时间替代部署 future latency | 中 | Critical | report schema、online_no_cache test | 正式 success/latency 只来自 online sampling | F/G | Controlled |
| R31 | correct 与 shuffle 的计算量不同 | 中 | High | per-stage latency trace | 两者均在线跑相同 K；donor loading 单独记录 | F/G | Controlled |
| R32 | Video-only sampler 与上游 joint path 数值不等价 | 中 | Critical | same-input/seed parity test | Gate C same-seed parity 已通过冻结容差，结果见 Phase C 报告 | C | Closed |
| R33 | current first-frame slice在 update 后漂移 | 低 | Critical | per-step equality assertion | 每 step 强制覆盖并断言；漂移立即停止 | B/C | Controlled |
| R34 | latent 被 VAE decode/re-encode，语义变成 Thought2 embedding | 中 | Critical | graph/provenance/schema | cache 只接收 native sampler state `z[:,:,1:]` | B/C | Controlled |
| R35 | 数据集版本/样本总量未知 | 高 | High | dataset revision inventory | LIBERO revision、archive SHA 和 42-episode task inventory 已冻结 | C/D | Closed |
| R36 | 上游 config 默认 `val_set_proportion=0` 导致无 dev | 高 | Critical | split manifest | Thought3 自建 90/10 episode split，禁止沿用默认 | B/D | Controlled |
| R37 | 数据 loader 虽只用 t0，却把后续 RGB 暴露给 model API | 中 | Critical | allowed-key schema、mutation test | 项目侧 current-observation dataset；模型调用对象不含未来帧 | B/C | Controlled |
| R38 | action target 被误传入 uncond future sampler | 中 | Critical | signature test、taint sentinel | sampler signature 没有 action；传入即 TypeError/config error | B/C | Controlled |
| R39 | future seed 与 action seed耦合，反事实不干净 | 中 | High | RNG stream manifest | 独立 namespace/Generator；counterfactual 固定 action seed | B/F | Controlled |
| R40 | 训练预算、初始化或 sample order 在 K 间不匹配 | 中 | Critical | cross-run manifest diff | 结构字段只允许 K/future fingerprint 不同；否则拒绝聚合 | E/F/G | Controlled |
| R41 | 单一 train seed 偶然增益被写成结论 | 中 | High | multi-seed aggregation | Phase F 仅技术 pilot；正式至少多 seed，并报告 seed 间分布 | G | Controlled |
| R42 | episode 被当完全独立样本，CI 过窄 | 中 | Critical | analysis protocol test | episode→task、task 等权、suite-stratified task bootstrap | G | Controlled |
| R43 | 参数/成功率提高但 latency 不可部署 | 高 | Medium | Pareto table | 同时报告 P50/P95、显存和成功率；允许结论为 trade-off 不划算 | F/G | Accepted |
| R44 | future 本身在 OOD 错误，Adapter 放大错误 | 高 | High | correct/shuffle/null、failure taxonomy | 作为核心科学结果报告；不隐藏负提升 | F/G | Accepted |
| R45 | Adapter 根本不使用 future | 高 | High | replay-calibrated counterfactual | 如实报告；不以 success noise 声称 future 有用 | E/F/G | Accepted |
| R46 | Phase A 文档变更意外修改 Thought1/2 输出 | 低 | Critical | 冻结文件 SHA-256 复核 | Phase A 前后八个冻结哈希一致；后续 gate 继续复核 | A/B | Closed for A |
| R47 | 旧 CLI 因 Thought3 import/参数变化而改变 | 中 | Critical | old CLI regression、dry-run | additive lazy import；旧命令输出/计划 fixture 不变 | B | Controlled |
| R48 | LIBERO-Plus 既有 `.downloads/` 被误归因或写入 | 低 | Medium | upstream status snapshot | 记录为审计前既有；Thought3 path guard 禁止写 third_party | A–G | Controlled |
| R49 | 单样本通过但 residual/hidden correction 尺度过大，导致多样本不稳定 | 高 | High | residual、gate、实际 BF16 delta/action-hidden | E.1 发现 A0 1.91×、A1 0.70×；E.2 六轨迹尺度门槛均通过，最大 sample ratio 0.537，但总 Gate 因跨样本 loss 门槛失败 | E | Controlled |
| R50 | 每 sample 只绑定一个 fixed action-flow draw，使 sample stability 被 timestep/noise 混淆 | 高 | High | initial loss/timestep、multi-flow held-out probe | E.2 initial loss max/min 94.28×，loss–BF16-timestep `r=-0.93466`；E.3 固定使用未训练的 flow step 1..5，先 sample 内平均再做 6/8 门槛 | E | Open |

## 3. 最高优先级风险详解

### 3.1 R01/R15/R37：真实未来泄漏

这是最高优先级风险。上游 RobotVideoDataset 和 `FastWAM.training_loss()` 原本会处理完整
demonstration video；即使 attention mask 理论上让 action 只访问 first frame，也不适合
用作阶段三的泄漏边界。

硬控制：

```text
Thought3TrainingBatch.allowed_keys =
  current_rgb
  current_proprio
  context
  context_mask
  target_action
  action_is_pad
  future_latent_from_model
  future_mask
  identity/provenance
```

任何 `next_*`、`future_frames`、`video` with T>1、`gt_future_latent`、
`success`、`termination` 字段默认拒绝。action target 是监督，不得被 sampler 或 Adapter
读取。

Gate C 的强制反事实：

1. 固定 current observation、language、proprio、noise；
2. 把同一 demo 的真实后续 RGB 替换成全零、全一和随机值；
3. 重建 cache；
4. 三次 native latent checksum 必须完全相同。

### 3.2 R03/R04/R05：K 与 cache identity

每条 cache 同时有：

- `base_sample_id`：用于跨 K 配对；
- `cache_sample_id`：包含 K、seed、schema；
- `initial_noise_sha256`；
- `sampler_schedule_sha256`；
- `latent_sha256`。

validator 必须检查：

```text
set(base_id in K1) == set(base_id in K2) == set(base_id in K4)
initial_seed(K1) == initial_seed(K2) == initial_seed(K4)
K1.schedule != K2.schedule != K4.schedule
checkpoint/config/input hashes all equal
```

错误不能用 warning 跳过。

### 3.3 R09/R10/R40：对照失配

Phase F/G 聚合前对 A0/A1/A2/A4 manifest 做 structural diff。允许不同：

- `variant_id`；
- `future_mode`；
- `K`；
- cache fingerprint；
- K 特有实测 latency。

除此之外的结构、训练预算、init hash、split、action seed schedule、评测 jobs 任何不同都
使该组不能进入主 K 曲线。

### 3.4 R21/R23：冻结与恢复

冻结证明不是只打印 `requires_grad`：

1. 训练前记录 frozen parameter name、shape、dtype、sample/full hash；
2. backward 后检查 `.grad is None`；
3. optimizer param ID 与 Adapter allowlist 完全一致；
4. checkpoint 后再次 hash；
5. resume 加载后再次 hash；
6. Adapter state round-trip 后 fixed input 输出一致。

任一失败，当前及此前未验证 checkpoint 全部标记 invalid。

### 3.5 R25/R27：48 GB 显存

现有 20-step shadow generation 峰值为 24,841 MiB，但训练多了 action backward。
第一次真实 backward：

- 单卡；
- microbatch=1；
- Adapter-only；
- bf16；
- 不加载多个 variant；
- 无 RGB decode/video writer；
- 43 GiB hard abort。

回退顺序固定为：

1. gradient accumulation；
2. action/MoT non-reentrant activation checkpoint；
3. 缓存 current-frame VAE latent；
4. CPU offload；
5. FSDP。

不能一开始同时改变多项，否则无法知道哪项解决问题或改变数值。

## 4. Gate 检查表

### Gate A：设计确认

- [x] 独立分支；
- [x] 主仓库/上游/资产 commit 与 hash；
- [x] Thought1/2 冻结文件 hash；
- [x] latent/scheduler/action call chain；
- [x] 参数、磁盘、显存估算；
- [x] 用户确认关键设计选择（2026-07-27）。

未确认前不得进入模型编码。

### Gate B：CPU/mock

- [x] old tests 全通过；
- [x] old CLI regression；
- [x] zero-gate parity；
- [x] A0/A1/A2/A4 param count 一致；
- [x] cache ID deterministic；
- [x] cache 与 training resume 不重复且 deterministic；
- [x] checksum 检出人为损坏；
- [x] shuffle derangement；
- [x] leakage allowlist、source guard 与 sampler signature；
- [x] mock training loss 下降；
- [x] Adapter-only checkpoint round-trip；
- [x] dry-run 独立进程不 import torch/safetensors、不 load checkpoint；
- [x] Thought1/2 hash 不变。

Phase B closure：全量回归、冻结哈希和最终测试分母见
`docs/thought3_phase_b_report.md`。真实 future mutation invariance 仍属于 Gate C，
不能由 mock schema 测试替代。

### Gate C：单 GPU

- [x] 真实 train sample；
- [x] `[B,48,2,14,28]` bf16 native latent；
- [x] K schedule/seed 配对；
- [x] video-only 与 upstream parity；
- [x] Adapter forward/backward finite；
- [x] only Adapter grad；
- [x] frozen pre/post hash；
- [x] zero-gate action parity；
- [x] peak <43 GiB；
- [x] current slice fixed；
- [x] no VAE decode/re-encode；
- [x] 真实 future mutation invariance。

### Gate D：cache

- [x] 一个 suite/task、32-base-sample pilot；
- [x] K=1/2/4 完整配对；
- [x] build/no-op resume/validate/checksum corruption；
- [ ] 独立 online recompute tolerance（正式扩 cache 前补）；
- [ ] 3-rank union/intersection（单卡 pilot 不执行）；
- [x] throughput 与 K 分段 latency telemetry；
- [x] 实际磁盘 bytes/inode 记录；
- [x] 20% free-space margin。

Phase D 已按用户批准的单卡小 cache smoke 范围通过。未勾选的两项不影响该
32-sample pilot，但必须在扩大为分布式正式 cache 前另设门禁。

### Gate E：训练

- [x] A0/A1 100–500 steps；
- [x] loss/gate/grad finite；
- [x] E.1 单样本固定 loss 有可诊断下降；
- [ ] 多样本 fixed train/development loss 有稳定下降；
- [x] uninterrupted/resumed 一致；
- [x] dev-only checkpoint selection；
- [x] E.1 frozen-before/after 完全相同；
- [x] 单卡无 OOM。

2026-07-28 状态：v2 已完成 A0/A1 各 100-step resumed/uninterrupted；
第 2 step 非 gate gradient、semantic SHA、checkpoint selection 和显存通过。
v3 fixed train probe 未低于初始化，且 fail-fast 发生在 frozen-after hash
之前，因此 Gate E 总体仍未通过。详见
[thought3_phase_e_report.md](thought3_phase_e_report.md)。

Gate E.1 随后证实 A0/A1 单样本固定 loss 分别下降 92.93%/99.58%，并关闭
frozen hash 缺口；但发现实际 hidden correction 达 A0 1.91×、A1 0.70×。
因此 Gate E 仍需 8-sample 尺度诊断和新的 28/4 完整运行，不能由 E.1
追溯改判。详见
[thought3_phase_e1_report.md](thought3_phase_e1_report.md)。

### Gate F：技术 pilot

- [ ] 六组都执行；
- [ ] same-job paired；
- [ ] replay numerical floor；
- [ ] correct/null/shuffle action sensitivity；
- [ ] online no-cache；
- [ ] P50/P95/peak memory；
- [ ] 无严重工程性 ID 崩溃；
- [ ] 未把 pilot 写成正式结论；
- [ ] Phase G 协议在看正式结果前冻结。

## 5. 全局立即停止条件

出现任意一项，当前 run 立即停止并标记 invalid：

1. `outputs/thought1/` 或 `outputs/thought2/` 任一冻结哈希改变；
2. `third_party/FastWAM` 出现 Thought3 修改；
3. Adapter/sampler 输入真实未来 observation/latent；
4. online evaluator 打开训练 cache；
5. train/dev/final episode 泄漏；
6. frozen parameter 有 grad 或 hash 改变；
7. K/seed/cache sample 错配；
8. A-shuffle donor 与 recipient 同 episode 或同 task；
9. A0/A-K 参数、init 或训练预算不一致；
10. action steps 不再固定为 20；
11. normalization/stats hash 改变；
12. NaN/Inf；
13. 单卡峰值超过 43 GiB；
14. rank shard 重复/遗漏；
15. resume identity 不匹配却继续；
16. 正式协议未冻结就查看/选择正式结果。

停止不代表研究失败。修复工程问题后要生成新 run ID 和 manifest，不能覆盖无效证据。

## 6. 允许接受并报告的科学风险

以下不是工程 bug，不应通过删结果或反复调正式集“修好”：

- future latent 对 OOD 无提升；
- K=4 比 K=1 更慢但不更好；
- A0 提升而 A-K 不再提升；
- shuffle 会改变动作但 correct 不提高成功率；
- gate 保持接近零；
- Camera/robot-init 之外的扰动没有收益；
- 某些 task 从 future 获益、另一些受损；
- seed 间方差较大；
- latency 增量覆盖成功率收益。

这些结果都直接回答“Fast-WAM 在 OOD 中是否真的需要未来想象”。

## 7. 当前未关闭项

| 风险 | 关闭它所需的下一证据 |
| --- | --- |
| R13 多样本是否实际使用 future | 通过 Gate E 后做 same-checkpoint correct/null/shuffle counterfactual |
| R28 正式 cache 容量 | 扩大样本前重新执行 inventory、容量估算和 free-space gate |
| R49 hidden correction 尺度 | E.2 六轨迹尺度门槛已通过；完整 28/4 Gate E 仍须复核 |
| R50 单 fixed flow draw 混淆 | 运行预注册 E.3 held-out flow step 1..5 probe |
| 3-rank cache 完整性 | 正式分布式 cache 的 union/intersection/cardinality 证明 |
| Gate E 多样本优化 | 稳定配方下重新跑 28/4 fixed train/development gate |

在这些证据出现前，文档必须使用“预计”“待验证”，不能写成已完成结果。
