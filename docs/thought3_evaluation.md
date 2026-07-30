# Thought3 在线评测手册

状态：Phase 1 真实 K=1 online counterfactual 已实现/测试/dry-run；GPU 待运行
更新时间：2026-07-30

## 1. 最重要的边界

正式评测必须在线生成 future：

```text
current observation
  → frozen current-state path
  → frozen K-step Video DiT
  → native future latent
  → trained Adapter
  → fixed 20-step Action DiT
  → action
```

离线 cache 只用于训练。online evaluator 的 API 不接收 cache path 或 reader；
manifest 必须记录 `online_cache_read=false`。

## 2. 六组运行语义

- B0：不运行 sampler、不构造 Adapter；
- A0：构造并训练同一 Adapter，在线输入全零 latent，不运行 sampler；
- A1/A2/A4：从本次 current observation 在线运行完整 K schedule；
- A-shuffle：同一个 A-K checkpoint，recipient 和预注册 donor 的当前观测分别
  在线生成相同 K；recipient 动作读取 donor future。

这里的 A0 是后续 matched training group；Phase 1 的 `null` 是 A1 checkpoint
上的 parameter-free bypass，用于 B0 parity，不等同于旧 A0 训练时的全零
future tensor。

Phase 1 技术 cohort 只有同一 task，因此 A-shuffle 固定为**同 task、跨
episode**的一一 derangement；这不是跨 task 语义检验。后续多 task pilot 必须在
manifest 中另行冻结 donor 规则，不能沿用或根据结果改变 Phase 1 mapping。

## 3. Action counterfactual

固定：

- current RGB、language、proprio；
- checkpoint；
- action diffusion seed；
- action steps=20。

首个真实 Phase 1 只替换：

- correct future；
- formal null（无 tensor、无 Video DiT）；
- other-episode shuffle。

保存：

- action hash；
- chunk L1/L2；
- action direction cosine；
- gripper change；
- cumulative EEF trajectory change；
- paired task success change。

Phase B `thought3-counterfactual` 仍保留为 mock。独立的
`thought3-k1-online-counterfactual` 已实现真实 Fast-WAM K=1 动作指标，但真实
GPU 尚未运行；它不读取或输出 success。正式 success change 必须来自后续
paired rollout。

## 4. Latency

每个 policy call 分段：

| 阶段 | 包含 | 不包含 |
| --- | --- | --- |
| preprocessing | 图像/proprio/text 输入准备 | 环境 step |
| current encoding | VAE current slice + current Video DiT path | future sampling |
| future sampling | K 个完整 update | RGB decode |
| Adapter | projector + cross-attention + gate | Action DiT |
| action denoising | 固定 20 steps | future sampling |
| total policy | 上述 policy wall time | simulator/render |

正式 CUDA 使用同步 CUDA events 并预热；报告 mean/P50/P95 与 peak
allocated/reserved memory。Thought2 的 20-step decoded shadow latency 不能替代这里。

## 5. Phase B mock 命令

训练完成后可验证工程闭环：

```bash
fastwam-ood thought3-counterfactual \
  --config configs/thought3/train_a1_smoke.yaml

fastwam-ood thought3-evaluate \
  --config configs/thought3/train_a1_smoke.yaml

fastwam-ood thought3-aggregate \
  --config configs/thought3/train_a1_smoke.yaml

fastwam-ood thought3-report \
  --config configs/thought3/train_a1_smoke.yaml
```

产物明确带 `mock_only_no_scientific_claim=true`。它们只证明：

- Adapter checkpoint 能进入在线动作路径；
- future 在线生成而非读取训练 cache；
- latency schema 可落盘；
- aggregate/report namespace 独立。

mock success、动作差异和 CPU latency 不得进入论文、简历效果数字或正式 K 曲线。

## 6. Phase 1：真实 K=1 技术反事实

固定 E6 A1@3e-4 step-200 checkpoint 和 E6 已消耗的 8 条 train sample。先运行
B0 `infer_action()` 两次定义 replay floor，再运行：

- correct：target current → online frozen Video DiT K=1 → Adapter；
- null：request-scoped identity bypass，0 tensor、0 Video DiT、0 Adapter call；
- shuffle：donor current 在线生成 K=1，但 action 端保持 target current/context/
  action noise。

null 必须与 B0 达到 `L∞<=1e-5` parity，否则 fail closed。A/B/C 分类只回答
future-content action sensitivity，不回答 success/OOD。配置、完整指标、恢复
和唯一运行命令见
[thought3_phase1_k1_online_counterfactual_protocol.md](thought3_phase1_k1_online_counterfactual_protocol.md)。

## 7. Directional OOD pilot 设计

仅当 Phase 1=A 且 Phase 2 完成后进入。当前草案为：

- 5 个预先冻结代表 task；
- Clean + camera + robot-init；
- 4 个独立但 paired episode seed；
- B0/A0/A1/A-shuffle；
- 相同 job/episode seed；
- failure-only video 加全量 action trace。

总预算 `4 groups × 3 environments × 5 tasks × 4 seeds = 240 rollouts`。
pilot 回答：

1. A1 success 是否高于 A0；
2. A1 是否高于 A-shuffle；
3. OOD gain 是否不弱于 Clean gain；
4. K=1 latency/显存是否可接受；
5. 是否值得扩展 A2/A4。

不得把 pilot success rate 写成正式模型提升。

## 8. Phase G 正式矩阵

冻结前创建独立 Thought3 job manifest。所有组保持：

- official checkpoint/stats；
- train recipe 与 checkpoint rule；
- Clean/OOD task、variant、episode seed；
- action denoise steps；
- max episode steps；
- success checker；
- recording；
- exclusion/missing 规则。

主表：

| Variant | K | Clean SR | OOD SR | Drop | AK−A0 | Correct−Shuffle | Future P50/P95 | Total P50/P95 | Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

分层至少报告 camera、robot-init、background、layout、lighting。不要只展示获益类别。

## 9. Failure artifacts

每个失败保存：

- 当前帧与必要观测；
- online future latent provenance，可选离线 decode 仅用于 review；
- action trace/hash；
- correct/null/shuffle counterfactual；
- termination/max-steps；
- policy latency breakdown；
- peak memory；
- 视频。

failure taxonomy 需盲化/分层抽样，不按结果挑好看的案例。

## 10. 正式运行停止条件

- online 发生任何 training-cache read；
- action step 数改变；
- checkpoint/split/job fingerprint 不匹配；
- A-shuffle donor 违规；
- frozen 参数变化；
- NaN/Inf 或 action shape 异常；
- peak allocated 超过该阶段冻结的物理卡 bound（Phase 1 为 23.8 GiB）；
- correct/shuffle 计算配方不等价；
- protocol 尚未冻结；
- 运行中用 OOD 结果改选 K/checkpoint。
