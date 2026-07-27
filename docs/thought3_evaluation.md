# Thought3 在线评测手册

状态：Phase B online boundary 与 CPU/mock 已实现；真实 Fast-WAM/LIBERO 接入待 Phase C/F
更新时间：2026-07-27

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

A-shuffle donor 必须跨 task、跨 episode、同 split/K/fingerprint。找不到 donor
时该 job 不得退化为 same-task donor。

## 3. Action counterfactual

固定：

- current RGB、language、proprio；
- checkpoint；
- action diffusion seed；
- action steps=20。

替换：

- correct future；
- null；
- shuffle；
- random；
- K1/K2/K4。

保存：

- action hash；
- chunk L1/L2；
- action direction cosine；
- gripper change；
- cumulative EEF trajectory change；
- paired task success change。

Phase B `thought3-counterfactual` 已能对 mock checkpoint 输出前五项；success change
为 `null`，不能写成机器人结果。正式 success change 必须来自 paired rollout。

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

## 6. Phase F pilot 设计

进入 pilot 前必须完成 C/D/E。建议范围：

- 一个 suite；
- 3–5 task；
- 一个 train seed；
- 少量 Clean + camera + robot-init OOD；
- B0/A0/A1/A2/A4/A4-shuffle；
- 相同 job/episode seed；
- failure-only video 加全量 action trace。

pilot 只回答技术可行性：

1. zero-gate/A0 没有工程性崩溃；
2. online sampler、Adapter、Action DiT 稳定；
3. shuffle/null 能否产生超过 replay floor 的动作变化；
4. K latency/显存是否可接受；
5. 是否值得冻结 Phase G 协议。

不得把 pilot success rate 写成正式模型提升。

## 7. Phase G 正式矩阵

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

## 8. Failure artifacts

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

## 9. 正式运行停止条件

- online 发生任何 training-cache read；
- action step 数改变；
- checkpoint/split/job fingerprint 不匹配；
- A-shuffle donor 违规；
- frozen 参数变化；
- NaN/Inf 或 action shape 异常；
- peak memory >43 GiB；
- correct/shuffle 计算配方不等价；
- protocol 尚未冻结；
- 运行中用 OOD 结果改选 K/checkpoint。
