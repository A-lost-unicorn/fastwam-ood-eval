# Thought3 风险登记与停止条件

状态：Phase 0/1 完成；Phase 2 有效离线负结果；Phase 3/A2/A4/OOD 路线停止
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
| R02 | 在线正式评测读取训练 cache | 中 | Critical | file-open guard、online evaluator test、manifest | Phase 1 真实 telemetry 为 0 cache read；后续 evaluator 继续不接收 cache path | B/1/F/G | Controlled / Phase 1 closed |
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
| R13 | zero gate 长时间不开，Adapter 实际忽略 future | 高 | High | gate、submodule grad norm、counterfactual | E.1 已证实 step 2 non-gate gradient；Phase 1 在 E6 checkpoint 上以 correct/null/shuffle `8/8` 证明内容使用；完整 Phase 2 checkpoint 仍须复验 | E/1/2 | Controlled / Phase 1 closed |
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
| R29 | 把 Thought2 3.355 s 当作 K-step 在线延迟 | 高 | High | latency boundary manifest | Phase 1 独立测得 K1 Video DiT `189.88 ms` mean、paired total overhead `258.95 ms`；Thought2 数值继续隔离 | C/1/F | Closed for K1 |
| R30 | 离线 cache 读取时间替代部署 future latency | 中 | Critical | report schema、online_no_cache test | Phase 1 全部 future 在线生成、0 cache read；正式 success/latency 继续只允许 online | 1/F/G | Controlled / Phase 1 closed |
| R31 | correct 与 shuffle 的计算量不同 | 中 | High | per-stage latency trace | Phase 1 两者均精确 1 个 K1 Video DiT 和 20 Adapter calls；donor loading 单独计入 shuffle policy total | 1/F/G | Closed for Phase 1 |
| R32 | Video-only sampler 与上游 joint path 数值不等价 | 中 | Critical | same-input/seed parity test | Gate C same-seed parity 已通过冻结容差，结果见 Phase C 报告 | C | Closed |
| R33 | current first-frame slice在 update 后漂移 | 低 | Critical | per-step equality assertion | 每 step 强制覆盖并断言；漂移立即停止 | B/C | Controlled |
| R34 | latent 被 VAE decode/re-encode，语义变成 Thought2 embedding | 中 | Critical | graph/provenance/schema | cache 只接收 native sampler state `z[:,:,1:]` | B/C | Controlled |
| R35 | 数据集版本/样本总量未知 | 高 | High | dataset revision inventory | LIBERO revision、archive SHA 和 42-episode task inventory 已冻结 | C/D | Closed |
| R36 | 上游 config 默认 `val_set_proportion=0` 导致无 dev | 高 | Critical | split manifest | Thought3 自建 90/10 episode split，禁止沿用默认 | B/D | Controlled |
| R37 | 数据 loader 虽只用 t0，却把后续 RGB 暴露给 model API | 中 | Critical | allowed-key schema、mutation test | 项目侧 current-observation dataset；模型调用对象不含未来帧 | B/C | Controlled |
| R38 | action target 被误传入 uncond future sampler | 中 | Critical | signature test、taint sentinel | sampler signature 没有 action；传入即 TypeError/config error | B/C | Controlled |
| R39 | future seed 与 action seed耦合，反事实不干净 | 中 | High | RNG stream manifest | Phase 1 使用独立 namespace，correct/shuffle 复用 recipient future seed并固定 target action seed | B/1/F | Closed for Phase 1 |
| R40 | 训练预算、初始化或 sample order 在 K 间不匹配 | 中 | Critical | cross-run manifest diff | 结构字段只允许 K/future fingerprint 不同；否则拒绝聚合 | E/F/G | Controlled |
| R41 | 单一 train seed 偶然增益被写成结论 | 中 | High | multi-seed aggregation | Phase F 仅技术 pilot；正式至少多 seed，并报告 seed 间分布 | G | Controlled |
| R42 | episode 被当完全独立样本，CI 过窄 | 中 | Critical | analysis protocol test | episode→task、task 等权、suite-stratified task bootstrap | G | Controlled |
| R43 | 参数/成功率提高但 latency 不可部署 | 高 | Medium | Pareto table | 同时报告 P50/P95、显存和成功率；允许结论为 trade-off 不划算 | F/G | Accepted |
| R44 | future 本身在 OOD 错误，Adapter 放大错误 | 高 | High | correct/shuffle/null、failure taxonomy | 作为核心科学结果报告；不隐藏负提升 | F/G | Accepted |
| R45 | Adapter 根本不使用 future | 高 | High | replay-calibrated counterfactual | E6 checkpoint 的 Phase 1 分支 A 排除“完全忽略”；完整 Phase 2 checkpoint 与 success relevance 仍须复验 | E/1/F/G | Accepted / Phase 1 negative ruled out |
| R46 | Phase A 文档变更意外修改 Thought1/2 输出 | 低 | Critical | 冻结文件 SHA-256 复核 | Phase A 前后八个冻结哈希一致；后续 gate 继续复核 | A/B | Closed for A |
| R47 | 旧 CLI 因 Thought3 import/参数变化而改变 | 中 | Critical | old CLI regression、dry-run | additive lazy import；旧命令输出/计划 fixture 不变 | B | Controlled |
| R48 | LIBERO-Plus 既有 `.downloads/` 被误归因或写入 | 低 | Medium | upstream status snapshot | 记录为审计前既有；Thought3 path guard 禁止写 third_party | A–G | Controlled |
| R49 | 单样本通过但 residual/hidden correction 尺度过大，导致多样本不稳定 | 高 | High | residual、gate、实际 BF16 delta/action-hidden | E.1 发现 A0 1.91×、A1 0.70×；E.2 六轨迹尺度门槛均通过，最大 sample ratio 0.537，但总 Gate 因跨样本 loss 门槛失败 | E | Controlled |
| R50 | 每 sample 只绑定一个 fixed action-flow draw，使 sample stability 被 timestep/noise 混淆 | 高 | High | initial loss/timestep、multi-flow held-out probe | E.3 v2 已确认风险；E.4 使用 200 个唯一 paired slots 后六条 held-out reduction 均转正，但绝对幅度仍不足 | E | Controlled |
| R51 | 官方 scheduler 零权重端点使逐 objective `final/initial` ratio 出现零分母 | 中 | Medium | action weight、timestep、zero-loss count、v1 traceback | v2 完成 320/320 probe；8 个零权重 row 全部 exact zero loss，保留于主统计且仅排除未定义 ratio；回归测试通过 | E | Closed |
| R52 | Adapter 在 200 step 内主要拟合固定 action noise/timestep，而非形成跨 flow 稳定 action objective | 高 | High | E.2 fixed-flow、E.3 held-out 与 E.4 diversified-train 对照 | E.4 将六条 held-out reduction 改善为 `0.997%–1.948%`，说明 fixed-flow 是混淆但不是全部原因；不得回用 E.2 checkpoint | E | Controlled |
| R53 | microbatch 1 的单 action-flow objective 方差过高，使 scalar gate/Adapter 更新抵消 | 高 | High | per-step loss CV/top-share、gate-gradient sign/cancellation、final gate 与 delta/hidden | E.5 已完成 full-cohort arithmetic mean；六条 reduction 均高于 E.4，A1@3e-4 达 19.668%，但 E.5 同时多看 8× objectives，不能把增幅唯一归因于聚合 | E | Controlled / mechanism unresolved |
| R54 | 看过三档 LR 后把 A1@3e-4 的强单条结果当作正式 LR 或 future 效果 | 高 | Critical | protocol timestamp、selected LR、fresh-cohort identity、paired A0/A1 outcome | E.5 保持 failed/null selection；E.6 显式 post-selection 并在新 cohort 复现 A1 信号，但总 Gate 因 A0 4/8 稳定性失败，full E/A2/A4/OOD 仍锁定 | E | Controlled / valid negative gate |
| R55 | A0 mean 改善掩盖逐样本不稳定，导致仅看 pooled loss 错误放行配方 | 高 | High | per-sample ratio、intermediate checkpoint trajectory、larger-flow replication | v2 provisional 结果显示 raw/A0 有 2 条 confirmed harm，normalized/A0 为 0；但 paired 8.274% 未过 10%，无 candidate | E | Controlled / invalid-run evidence |
| R56 | 把同一八条 sample 上增加 flow draw 错当成 demonstration-level 独立复现 | 高 | Critical | 明确 resampling unit、保留未使用 cohort、结果措辞门禁 | 排序 17–28 未解码/训练；v2 `independent_replication_candidate=false`，E.9b 保持锁定 | E | Controlled |
| R57 | `6/8` 点符号 Gate 把校正后未确认的小波动与稳定 harm 等价计数 | 高 | High | point sign、双 block、FWER paired bootstrap、material flag | v2 同时报告 6/8 point Gate 和 32-comparison FWER：raw/A0 2 harms，normalized 四轨 0；结果受 R60 invalid 边界约束 | E | Controlled / invalid-run evidence |
| R58 | 根据 E.8 的已知 tail outcome 调权或只跑 normalized 配方，造成 outcome-conditioned weight 与新 flow 混淆 | 高 | Critical | calibration field audit、weight SHA、raw same-flow controls、protocol timestamp | 四轨 pairing/weight/schedule checks 通过；normalization 降低 tail harm 但同时降低 pooled/paired gain，未被选择为候选 | E | Controlled |
| R59 | 复用旧 Gate evaluator 时隐含 flow 数/取值硬编码，导致新协议在 objective 前失败且子轨状态误留 `running` | 中 | High | 任意正整数 flow contract、75–106 完整 grid 回归、invocation-scoped failed-status test、v1 SHA 冻结 | v2 成功完成四轨 `75..106`，证明 generic evaluator 与 initial-probe 路径修复有效 | E | Closed |
| R60 | checker 强制验证 RNG identity 字段，但 probe writer 未持久化这些字段，导致完整运行被标 invalid | 高 | High | writer/checker schema contract、artifact SHA、完整 grid/timestep/weight/zero-position 只读审计 | Phase 0 27/27 checks 通过，父 77 文件未改；登记 engineering valid + scientific failed | E | Closed |
| R61 | surrogate Gate 无限增长，长期不触达动作或 success 变量 | 高 | Critical | 加速路线和硬停止规则；Phase 1 前禁止新 flow/checkpoint/LR/weight Gate | Phase 1 已真实触达 action 并得到 A；Phase 2 唯一配方已冻结，不再新增 surrogate Gate | 1–4 | Closed for action boundary |
| R62 | 用零 future tensor 冒充 formal null，使 correct-null 混入 projector 响应 | 高 | Critical | request-scoped parameter-free bypass；不构造 tensor、不跑 Video DiT、Adapter call=0；B0 hard parity | 真实 8/8 null 为 0 Video DiT/Adapter call，B0-null L∞ 精确 0 | 1 | Closed |
| R63 | shuffle 同时改变 target current/context/action noise，无法归因于 future 内容 | 高 | Critical | other-episode 一一 derangement；target RGB/language/proprio/cache/action seed 不变；复用 recipient future noise | 8/8 other-episode、only-future-replaced、same recipient seed/context/initial-state checks 通过 | 1 | Closed for Phase 1 |
| R64 | B0 非确定性被误报为 future sensitivity | 中 | Critical | 每样本 B0×2；预先定义 `max(1e-7,10×p95 replay L2)`；`L∞>1e-5` fail closed | 真实 B0 replay 的 L1/L2/L∞ 全 0；floor 冻结为 `1e-7` 后才运行 intervention | 1 | Closed |
| R65 | 尝试多个 checkpoint 后挑动作差异最大者 | 中 | Critical | 唯一固定 E6 A1@3e-4 step-200 path/file/state/config SHA；披露 post-training engineering | preflight 精确匹配唯一 checkpoint；项目 prereg commit `f516920` | 1 | Closed for Phase 1 |
| R66 | 把在线 action engineering smoke 写成 OOD/success 或 K 曲线 | 高 | Critical | schema/report claim boundary；runner 不读 OOD/success、不 rollout、不自动 Phase 2 | 结果报告保留 SMOKE、single-task/no-rollout 与 effect-size 边界 | 1 | Controlled |
| R67 | Phase 1 代码/dry-run 被误记成真实 action 结果 | 中 | High | 只有 output decision/run status 才产生 outcome | 真实 `run_status=completed`、decision、62-file manifest 和结果报告已审计 | 1 | Closed |
| R68 | replay 恰好为 0 导致 `1e-7` floor 极低，把确定但很小的动作差异误写成实用效应 | 高 | Critical | 同时报告 L1/L2/L∞、action cosine、相对 B0 RMS、逐样本方向与 latency | 分支 A 只表示 content sensitivity；轨迹/success relevance 必须由完整 checkpoint 和 paired rollout 证明 | 1–3 | Controlled / downstream open |
| R69 | 八条同 task train sample 的 `8/8` 被误推广到其他 task、dev 或 OOD | 高 | Critical | cohort fingerprint、sample-level table、证据等级 | 报告固定为 one-task SMOKE；Phase 2 full 28/4 复验与多 task directional pilot 才允许推广 | 1–3 | Controlled / downstream open |
| R70 | 两卡分别重算 normalized weights，微小差异破坏 A0/A1 单变量配对 | 中 | High | calibration artifact SHA、两轨 weight SHA | 两轨实测读取同一 `4c36dece...1dc22`；sample/identity/flow SHA 全部 matched | 2 | Closed |
| R71 | 查看 step 50/100/150 dev 后挑 checkpoint，重建 trajectory selection | 高 | Critical | fixed endpoint、checkpoint manifest、无 fallback | 主结果固定 step 200；development 只评 step 0/200，中间 checkpoint 未用于选择 | 2 | Closed |
| R72 | E9-derived normalized recipe 被误写成预先独立发现或 E9 科学通过 | 高 | High | recipe disclosure、E9 audit SHA、protocol timestamp | 明确登记 post-E8 engineering selection，paired 8.274%<10% 与 E9b locked 同时报告 | 2 | Controlled |
| R73 | Phase 2 dev 方向通过后直接进入 rollout，却未确认完整 A1 checkpoint 仍读取 future 内容 | 中 | Critical | finalize 强制 `phase3_unlocked=false`、下一 stage 字段 | 本次 direction 为负，完整-checkpoint recheck 与 rollout 均未触发；Phase 3 保持锁定 | 2–3 | Not triggered / route stopped |
| R74 | config-relative artifact path 与 resolved safety root 混用，导致计算完成后 manifest 写入失败 | 中 | High | mixed relative/absolute path regression、outside-root rejection、原 rows SHA、resume status | 原目录 resume 完成；calibration/track/checkpoint manifests 全部验证，原 1,024 rows 保留 | 2 | Closed |

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
`docs/thought3/phase_b_d/phase_b_report.md`。真实 future mutation invariance 仍属于 Gate C，
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
[thought3_phase_e_report.md](../gate_e/phase_e_report.md)。

Gate E.1 随后证实 A0/A1 单样本固定 loss 分别下降 92.93%/99.58%，并关闭
frozen hash 缺口；但发现实际 hidden correction 达 A0 1.91×、A1 0.70×。
因此 Gate E 仍需 8-sample 尺度诊断和新的 28/4 完整运行，不能由 E.1
追溯改判。详见
[thought3_phase_e1_report.md](../gate_e/phase_e1_report.md)。

E.2–E.4 的后续证据显示 diversified objective 能把六条 held-out direction
全部转正，但幅度仍只有约 1%–2%。E.5 随后完成 200 matched optimizer
updates/track、每 update 全 8-sample cohort、8 个独立 flow objective 的
arithmetic mean 和完整 contribution/cancellation telemetry：

- 六条 reduction 均高于对应 E.4；
- `A1@3e-4` 达 19.668% reduction、8/8 non-worsened；
- 同 LR 的 A0 仅 2.638%，所以预注册共同 Gate 有效失败；
- 每条轨迹 24 个 zero-weight objective、全部执行/配对/冻结检查通过。

R53 已从“mitigation 未测试”更新为“聚合配方真实可用，但机制仍未分离”：
E.5 在相同 update 数下使用 E.4 的 8 倍 objectives，无法区分 aggregation 与
exposure。R54 则阻止把看到结果后突出的 `3e-4` 直接升级为正式 LR。

E.6–E.8 随后证明 A1 信号可序贯复现，但 A0 pooled improvement 与逐样本
harm 并存。E.9a-v1 在 objective 前因 flow 硬编码工程失效；E.9a-v2 保持
科学内容不变，冻结 raw/normalized × A0/A1 四轨单变量协议：

- 不选 step 100，科学 endpoint 固定 step 200；
- 不降低 A0/A1/paired 的原门槛；
- normalized weights 只来自 E.8 zero-gate initial loss；
- raw/normalized 共用新训练与 held-out schedule；
- train 排序 17–28 identity-only 保留，E.9a-v2 不读取。

v1 不是负结果。v2 已完成四轨：normalization 将 raw/A0 的 confirmed harm
从 2 降到 0，但 paired gain 只有 8.274%，未达 10%。Phase 0 已关闭 R60：
冻结代码路径在父目录 0 write、0 GPU/forward/backward 下恢复 256 个唯一 RNG
identity，27/27 checks 通过。因此该权衡可登记为 engineering-valid
post-run evidence，但科学分类仍失败。下一步是 Phase 1，不运行 E.9b。

### Phase 1：K=1 技术反事实

- [x] B0/correct/formal-null/shuffle 四条件；
- [x] 同 target/action seed paired；
- [x] replay numerical floor；
- [x] correct/null/shuffle action sensitivity；
- [x] online no-cache；
- [x] P50/P95/peak memory；
- [x] A/B/C 分类与 frozen 后续动作；
- [x] 未把 action smoke 写成 success/OOD 结论。

Phase 1 真实结果为分支 A。correct-null 与 correct-shuffle 均 `8/8` 超过
`1e-7` replay floor；B0/null 精确相同。完整 effect size、latency 和工件审计见
[thought3_phase1_k1_online_counterfactual_report.md](../phase1_action/report.md)。

## 5. 全局立即停止条件

出现任意一项，当前 run 立即停止并标记 invalid：

1. `outputs/thought1/` 或 `outputs/thought2/` 任一冻结哈希改变；
2. `third_party/FastWAM` 出现 Thought3 修改；
3. Adapter/sampler 输入真实未来 observation/latent；
4. online evaluator 打开训练 cache；
5. train/dev/final episode 泄漏；
6. frozen parameter 有 grad 或 hash 改变；
7. K/seed/cache sample 错配；
8. A-shuffle donor 与 recipient 同 episode，或不符合本阶段冻结的 same/cross-task
   规则；
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
| R13 future 使用是否迁移到完整 checkpoint | Phase 1 已在 E6 8-sample checkpoint 上关闭技术使用问题；Phase 2 后按同协议复验 |
| R28 正式 cache 容量 | 扩大样本前重新执行 inventory、容量估算和 free-space gate |
| R49 hidden correction 尺度 | E.2 六轨迹尺度门槛已通过；完整 28/4 Gate E 仍须复核 |
| R50 单 fixed flow draw 混淆 | 已受控：E.4 使用唯一 paired slots，六条 held-out reduction 均转正但未过 Gate |
| R51 零权重 objective ratio | 已关闭：v2 单测与真实 320 probes 的 weight/count/exact-zero 检查均通过 |
| R52 fixed-flow objective 拟合 | 已受控：E.4 证明 diversification 有改善但不足以形成 eligible LR |
| R53 单 objective 梯度方差 | E.5 已完成真实 full-cohort mitigation；仍需 matched-objective 证据才能分离 aggregation 与 8× exposure，但该机制分离不应先于强 A1 信号复验 |
| R54 post-selection 候选污染 | E.6 已按披露协议运行；保持 failed/null formal selection，不把复现的 A1 信号写成 future/OOD 效果 |
| R55 A0 样本稳定性 | audit 后定级为 normalized tail-stabilization signal；Phase 2 matched A0 必须继续逐样本报告 |
| R56 flow pseudo-replication | 排序 17–28 未读取；v2 无 candidate，E.9b 不解锁 |
| R57 point-sign 与 confirmed harm 混淆 | v2 已同时报告 6/8 与 FWER；不得只挑 tail `2→0` 忽略 paired 失败 |
| R58 outcome-conditioned weighting | same-schedule、固定 weight SHA 和 calibration provenance 均通过；结果未达候选门槛 |
| R59 evaluator/status 复用缺陷 | 已关闭：v2 完成 `75..106` 四轨与 6,400 objectives |
| R60 probe identity telemetry | 已关闭：Phase 0 audit valid，父 77 文件未改；科学 Gate 仍 failed |
| R68 技术差异与实用效应混淆 | Phase 1 sensitivity 与 Phase 2 offline negative 已分开登记；本路线停止，因此 rollout utility 仍未回答 |
| R69 单 task/八样本外推 | Phase 2 扩到 28/4 仍只有同一个 task；不得把 4-sample negative 外推到其他 task/OOD |
| 3-rank cache 完整性 | 正式分布式 cache 的 union/intersection/cardinality 证明 |
| 28/4 结果的跨 task/seed 外推 | 当前只有一个 task、一个 seed、4 条 development sample；本路线已按负 direction 停止，不以事后扩样覆盖 |

已关闭项可以写成其精确覆盖范围内的完成事实；上述未关闭项仍必须使用“预计”
“待验证”，不能外推成 rollout、success 或 OOD 结果。
