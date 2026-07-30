# Thought3 当前限制与未决问题

状态：Phase 0/1 完成、Phase 2 有效离线负结果后的诚实边界
更新时间：2026-07-30

## 1. 已有工程证据与仍缺的下游结果

当前已有：

- 真实 Fast-WAM K=1/2/4 latent 与 32×3 cache；
- 真实 Adapter action-loss backward、Adapter-only checkpoint 和冻结 SHA；
- E5/E6 两个 cohort 的 A1 离线 action-loss 信号；
- E9a-v2.1 engineering-valid/scientific-failed 审计；
- K=1 online B0/correct/null/shuffle 真实分支 A：B0/null 精确 parity，
  correct-null、correct-shuffle 与 action hash 均 `8/8`；
- 完整 28/4 A0/A1 checkpoint 与固定 step-200 offline development 结果。

当前仍没有：

- ID/OOD success rate；
- A-shuffle 机器人 rollout；
- 完整 checkpoint 的机器人 rollout 或 OOD success 对照；
- K=2/K=4 的真实训练、在线动作反事实或收益—成本曲线。

Phase C–E 的真实离线工程结果不支持 future 改善 OOD。Phase 1 的真实
`decision.json` 只登记固定 checkpoint 的动作内容敏感性；它仍是 `SMOKE`，
不能升级成轨迹、success 或 OOD 结论。

## 2. Native latent contract 已在小规模真实运行确认

`[B,48,2,14,28]` bf16 已由 Phase C/D 在官方 checkpoint 与标准 LIBERO
training sample 上确认，包括：

- VAE/current shape；
- full diffusion state `[B,48,3,14,28]`；
- future tail；
- dtype/device；
- 无 VAE decode/re-encode；
- current slice 每 update 不漂移。

该证据仍只覆盖小规模 offline cache。Phase 1 已首次计量同一 native latent
进入 20-step Action DiT 后的真实动作响应，但仅覆盖一个 task 的八条 train
sample。

## 3. Video-only sampler parity 已在小规模关闭

Phase C 已把项目侧 K-step loop 绑定真实 Video DiT，并在相同 input/seed 下与
upstream joint path 做数值 parity；K=1/2/4 的最大差为 0。该结论覆盖当前
checkpoint/config 的小规模 smoke，不自动保证未来上游版本变更。

## 4. Action loss 已完成真实小规模语义核对

Phase C 与 E 系列已在真实 Fast-WAM 上逐项复用官方：

- timestep/sigma sampling；
- noisy action 公式；
- velocity target 符号；
- action pad mask；
- loss weighting/reduction；
- normalization/denormalization。

该实证只覆盖现有 action-only smoke/training path；完整 28/4 配方已绑定相同
scheduler/stats SHA，但真实训练尚未产生 checkpoint。

## 5. 训练数据已完成 task-level inventory，但正式规模仍未训练

标准 LIBERO LeRobot 数据已可读，`libero_goal/task_0` 的 42 episode inventory、
37/5 episode split 和 32-sample cache pilot 已审计。Phase 2 已冻结：

- 28/4 实际训练/development identity；
- 完整 action/frame selection 和 update schedule；
- 数据没有混入 LIBERO-Plus/test trajectory。

仍待真实 GPU run 验证完整 28/4 loss、checkpoint 与 frozen hash；license 与
公开分发边界仍是独立未决项。

## 6. Hook 依赖上游调用次数

第一版通过 `action_encoder` output hook 避免改 `third_party/FastWAM`，但真实模型可能在：

- gradient checkpoint recompute；
- torch compile；
- 多次 action sub-call；
- DDP wrapping

下改变调用次数。exact-call guard 会 fail-fast，而不是静默多注入。Phase C/E
已确认训练调用边界，Phase 1 已确认 online 20-step action path 每个 condition
精确调用；未来 DDP/compile 配方仍须重新验证。

## 7. Zero gate 的优化动力学

gate=0 保证初始 parity，但第一步主要只有 gate 获得梯度，projector/attention 信号会在
gate 打开后出现。E 系列真实训练已经观察：

- gate 离开零；
- 第 2 step 起非 gate 子模块出现 finite nonzero gradient；
- Phase 1 correct/null/shuffle 对 future 内容有动作响应。

仍未关闭的是完整 28/4 checkpoint 的泛化与 correction scale；不应在正式 OOD
上反复调到“有效”。

## 8. A-shuffle 的部署语义和成本

正式 shuffle 不能读取训练 cache。为了保持计算公平，donor current 也必须在线运行同 K。
这需要预注册 donor observation 获取方式，并区分：

- recipient policy latency；
- donor future generation latency；
-研究性反事实总成本。

A-shuffle 是因果干预诊断，不是可部署策略。

## 9. Cache 容量只完成 32-sample 真实 pilot

纯 bf16 latent 为约 220.5 KiB/sample（K1+K2+K4）。Phase D 已对 32 sample、
96 entries、12 shards 实测 `7,687,316 bytes`、throughput、inode、checksum 和
20% 空间余量。扩大到正式多 task cache 前仍须重新做容量与 rank
union/intersection gate。

## 10. 3×4090 仍需真实验证

单卡 Phase C/E/Phase 1 已实测可运行；Phase 1 模型加载峰值
`23,679.51 MiB`、policy 峰值 `13,009.92 MiB`。这仍不能证明三卡 cache/training
shard union/intersection 或 DDP 恢复正确。只有实测 OOM 才考虑 gradient
checkpoint、CPU offload、FSDP 或 ZeRO，第一版不同时启用多个回退手段。

## 11. Statistical protocol 仍是 DRAFT

primary K、train seed 数、smallest effect、multiplicity、formal job manifest 和
checkpoint selection 仍未冻结。Phase F 之前保持
`thought3_analysis_protocol_DRAFT.md`；看到 Phase G 结果后再改协议将使分析降级为
post-run exploratory。

## 12. 可能的负面科学结果

以下都不是工程失败：

- A0 与 B0 相近，A1/A2/A4 无提升；
- A0 提升但 future 没有增量；
- K4 比 K1 更慢且更差；
- correct/shuffle 都改变动作但 success 不变；
- future 在 camera/robot-init 下放大错误；
- 部分 task 获益、部分受损；
- OOD 增益不足以覆盖 latency。

论文必须保留这些可能性。

## 13. Phase 2 已得到单 task 离线负结果

完整 28/4 matched A0/A1 已完成：A0 development loss 改善 `1.845%`，A1
恶化 `1.712%`，A1 final 比 A0 高 `3.624%`，且 4/4 development sample 的
A1 loss 更高。这是工程有效、按冻结 endpoint 得到的负结果，不是梯度断链。

它仍有明确限制：

- 只有一个 `libero_goal` task 和 4 条 development sample；
- 只有一个 seed、一个 K=1 Adapter 配方和固定 step 200；
- objective 是 action flow-matching loss，不是 rollout success；
- 没有读取 Clean/OOD outcome，也没有执行 Phase 3；
- 不能外推到 K=2/K=4、其他融合结构、完整模型训练或其他 task。

因此可写“future-content sensitivity 未在该预注册配方中转化为 offline
utility”，不可写“Fast-WAM 在 OOD 中不需要未来”。
