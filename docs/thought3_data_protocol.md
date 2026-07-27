# Thought3 数据与 Future Cache 协议

状态：Phase B schema 已实现并通过 CPU/mock 测试；真实 LIBERO 数据 revision 尚未冻结
更新时间：2026-07-27
适用阶段：B–G

## 1. 目的与边界

该协议保证 Future-to-Action Adapter 只能读取在决策时可获得的信息：

```text
current observation + language + proprio + sampled noise
                              ↓
                    frozen Video DiT
                              ↓
                 model-generated future latent
```

真实后续 RGB、下一时刻 observation、执行结果和 success 不能成为 sampler 或
Adapter 输入。真实 action chunk 只作为 action flow-matching 的监督目标。

训练源只允许标准 LIBERO demonstration。以下来源一律拒绝：

- `outputs/thought1/**`；
- `outputs/thought2/**`；
- `outputs/thought3/**` 中的 rollout/evaluation；
- LIBERO-Plus / `libero_plus`；
- 正式 ID/OOD test episode；
- 环境执行后的 observation、success 和 termination。

## 2. 数据集冻结项

进入 Phase C/D 前必须生成只读 inventory，并冻结：

| 字段 | 要求 |
| --- | --- |
| dataset revision | commit、Hugging Face revision 或不可变 snapshot ID |
| suite/task | 标准 LIBERO suite 与 task ID/name |
| demonstration | 原始 demo ID，不用数组位置代替 |
| episode | 完整 episode identity |
| frame | action-rate frame index 与 timestamp |
| cameras | 当前协议固定两路，默认 `image`、`wrist_image` |
| language | 原始任务文本及其 SHA-256 |
| checkpoint/stats | 官方文件 SHA-256 |
| preprocessing | image range、拼接、尺寸和 proprio 配方 hash |

当前工作区仍缺正式的四个标准 LIBERO LeRobot 训练目录，因此 Phase B 只使用
`mock-phase-b-v1` inventory；它没有研究结论资格。

## 3. Train/development split

split 在 suite×task 内按完整 episode 做确定性 90/10 划分：

```text
selection_key =
SHA256("thought3-split-v1" || split_seed || episode_id)
```

规则：

1. 同一 episode 的任何 frame 不得跨 train/development。
2. 每个 suite×task stratum 至少两个 episode；否则 fail-fast。
3. 每个合法 stratum 至少保留一个 train 和一个 development episode。
4. 默认 `split_seed=3407`、`development_fraction=0.1`。
5. A0/A1/A2/A4 共用完全相同的 split manifest。
6. development 只用于 LR、gate、训练步数等工程选择；不能使用最终 OOD 调参。

split manifest schema 为 `thought3.episode_split.v1`，其 canonical JSON
SHA-256 进入每个 base sample identity、cache 和 checkpoint。

## 4. 两级 sample identity

### 4.1 `base_sample_id`

`base_sample_id` 不含 K，用于 K=1/2/4 配对。canonical identity 至少包含：

- dataset revision；
- suite、task ID/name；
- demonstration、episode、frame、timestamp；
- camera keys；
- 完整 language；
- checkpoint/stats SHA-256；
- sampler-config SHA-256；
- preprocessing SHA-256；
- split-manifest SHA-256。

### 4.2 初始噪声

同一 base sample 的 K=1/2/4 必须共享初始噪声：

```text
initial_seed =
int64_positive(
  SHA256("thought3-noise-v1" || global_cache_seed || base_sample_id)[0:8]
)
```

K 不进入 seed 推导。Phase B validator 同时比较 seed 与初始 noise tensor hash。

### 4.3 `cache_sample_id`

`cache_sample_id` 区分不同 K：

```text
SHA256({
  cache_schema,
  base_sample_id,
  K,
  initial_noise_seed
})
```

训练 join 必须按稳定 identity；禁止仅按数组位置连接。

## 5. Future latent contract

| 项目 | 冻结值 |
| --- | --- |
| 来源 | 当前观测经冻结 Video DiT 采样 |
| schema | `thought3.future_cache.v1` |
| 单样本 layout | `CTHW` |
| 单样本 shape | `[48,2,14,28]` |
| batch shape | `[B,48,2,14,28]` |
| 默认 dtype | bf16 |
| 是否 decode RGB | 否 |
| 是否真实未来 VAE latent | 否 |
| source kind | `model_sampled_from_current` |

每个 bf16 latent payload 为：

```text
48 × 2 × 14 × 28 × 2 bytes = 75,264 bytes = 73.5 KiB
```

K=1/2/4 合计为 225,792 bytes/sample（约 220.5 KiB），未计 mask、
metadata 和 safetensors header。512 样本三种 K 的纯 latent 约 110.25 MiB。
容量规划采用：

```text
latent_bytes = N × 3 × 75,264
```

并在真实数据上额外保留至少 20% 可用空间。粗略量级：10 万样本纯 latent
约 21.0 GiB，100 万样本约 210.3 GiB。

## 6. K-step schedule

K 是从同一 sigma=1 noisy state 到 sigma=0 的完整 shifted-flow update 数，
不是 20-step schedule 的前 K 步，也不是 Action DiT step。

冻结参数：

- K ∈ `{1,2,4}`；
- shift = 5；
- train timesteps = 1000；
- noise 在 CPU float32 独立生成，再转模型 dtype/device；
- action denoising 始终为 20。

Phase B 实现的 sigma nodes：

| K | sigma nodes |
| ---: | --- |
| 1 | `[1.0, 0.0]` |
| 2 | `[1.0, 0.8333333, 0.0]` |
| 4 | `[1.0, 0.9375, 0.8333333, 0.625, 0.0]` |

Phase C 必须与上游 `WanContinuousFlowMatchScheduler` 做同输入、同 seed 数值
parity；未通过时不能生成真实 cache。

## 7. Shard 与原子提交

目录：

```text
outputs/thought3/cache/<run>/
├── cache_plan_manifest.json
├── cache_plan.jsonl
├── split_manifest.json
├── k1/
│   ├── shard_000000.safetensors
│   ├── shard_000000.metadata.jsonl
│   └── shard_000000.manifest.json
├── k2/
└── k4/
```

默认每 shard 512 个样本。一个 shard 只能由一个 rank 写，rank 分配采用稳定
index modulo world size。

提交顺序：

1. tensor 写同目录临时文件、flush/fsync、atomic rename；
2. metadata 原子写入；
3. shard manifest 最后写入，作为唯一 commit marker；
4. 写后立即完整读取并校验。

resume 只跳过 commit manifest 存在且全部 checksum 通过的 shard。存在 manifest
但校验失败时拒绝继续，不能把损坏当 completed；未提交的残片需要人工审计后处理。

## 8. Checksum 与 validation

每个 shard 保存并验证：

- safetensors 文件 SHA-256；
- metadata JSONL SHA-256；
- `future_latents` 和 `future_masks` 语义 tensor SHA-256；
- 每样本 latent SHA-256；
- cache/base sample ID；
- K、schedule、seed、initial-state hash；
- checkpoint/stats/cache fingerprint；
- source kind 与 `uses_ground_truth_future=false`。

whole-cache validator 还检查：

```text
set(K1.base_id) == set(K2.base_id) == set(K4.base_id)
seed(K1) == seed(K2) == seed(K4)
initial_state_hash(K1) == ... == initial_state_hash(K4)
无 duplicate / missing / extra sample
```

## 9. 自动泄漏门禁

训练 batch 使用 allowlist；未知字段也拒绝。允许：

```text
sample_id, base_sample_id,
current_rgb, current_proprio,
context, context_mask,
target_action, action_is_pad,
future_latent, future_mask,
metadata
```

显式禁止包括：

```text
actual_future, future_frames, gt_future_latent,
next_image, next_observation,
success, termination_reason
```

video-only sampler API 不存在 `action`、`target_action`、future frame 或 success
参数。正式 online evaluator 的构造函数也不存在 cache path/reader。

Phase C 还必须增加真实 future mutation invariance：在不改当前帧的情况下替换
demo 后续 RGB，生成的 K latent checksum 必须完全不变。

## 10. Phase B 命令

```bash
source scripts/activate_env.sh

fastwam-ood thought3-plan-cache \
  --config configs/thought3/cache_smoke.yaml

fastwam-ood thought3-build-cache \
  --config configs/thought3/cache_smoke.yaml

fastwam-ood thought3-validate-cache \
  --config configs/thought3/cache_smoke.yaml
```

重复执行 plan/build 时必须显式加 `--resume`。这些命令当前 `backend=mock`，
不得把生成值当作真实 Video DiT latent。

## 11. 进入真实 cache 的门槛

- [ ] 标准 LIBERO dataset revision 与 inventory 冻结；
- [ ] 真实 `[B,48,2,14,28]` bf16 tensor smoke；
- [ ] K scheduler 与 upstream parity；
- [ ] current slice 每 step 完全固定；
- [ ] sampler 不读取 action 或真实 future；
- [ ] 单卡峰值小于 43 GiB；
- [ ] 一个 suite/task 的 online/cache tolerance 通过；
- [ ] 三 rank shard union 完整且 intersection 为空；
- [ ] 实际磁盘容量预检通过。
