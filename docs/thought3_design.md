# Thought3 设计：Partial-Future Adapter

状态：Phase 0 审计完成；Phase 1 K=1 在线动作反事实已实现、真实 GPU 待运行
科学问题：显式读取低成本 future latent，能否改善 Fast-WAM 的 OOD 控制，而收益不是由
额外参数、重新训练或错误对照造成？

当前执行顺序以
[thought3_accelerated_roadmap.md](thought3_accelerated_roadmap.md) 为准。
E5–E9 的 surrogate 结果不再阻塞 Phase 1；六组 K 曲线只有在 K=1 directional
OOD pilot 正向后才解锁。

## 1. 研究声明与非声明

阶段三要建立第一条真实的：

```text
model-generated future latent → action
```

路径，并通过 B0、A0、A1、A2、A4、A-shuffle 区分：

- 原模型能力；
- Adapter/重新训练本身的影响；
- 正确 future 的增量信息；
- future 计算步数；
- 错误 future 对动作和任务结果的干预效应。

阶段三不预设 future 有效，也不把 Thought2 的相关性证据改写成因果证据。允许的研究
结论包括：

- 正 future 提升 OOD；
- 只提升 ID 或部分扰动；
- 没有提升；
- 负提升；
- Adapter 学会忽略 future；
- Adapter 使用 future，但错误/低质量 future 使控制变差。

## 2. 隔离边界

### 2.1 只读

- `outputs/thought1/`
- `outputs/thought2/`
- `third_party/FastWAM/`
- 官方 checkpoint 和 dataset stats
- Thought1/Thought2 job ID、seed pairing、统计协议

### 2.2 新 namespace

```text
branch:  feature/thought3-partial-future-adapter
code:    src/fastwam_ood_eval/thought3/
config:  configs/thought3/
output:  outputs/thought3/
schema:  thought3.*
CLI:     fastwam-ood thought3-*
```

所有 Thought3 writer 在 resolve path 后必须拒绝：

- `outputs/thought1/**`
- `outputs/thought2/**`
- `third_party/**`
- cache 目录之外的任意训练 cache 写入

根 CLI 只做惰性注册/分发。旧命令不能 import torch/FastWAM/Thought3 heavy modules，
也不能改变默认值、job planning 或输出 schema。

## 3. 总体架构

```text
current observation ──────────────┬─────────────────────────────────────┐
                                  │                                     │
                                  ▼                                     ▼
                         frozen VAE encode                    frozen VAE encode
                                  │                                     │
                                  ▼                                     ▼
                    current latent [B,48,1,14,28]      initial video noise + current latent
                                  │                                     │
                                  ▼                                     ▼
                    frozen Video DiT current path        frozen Video DiT K updates
                                  │                                     │
                                  ▼                                     ▼
                     per-layer current K/V cache       native future tail [B,48,2,14,28]
                                  │                                     │
                                  │                                     ▼
                                  │                           small future projector
                                  │                                     │
                                  │                                     ▼
                                  │                             [B,196,256] tokens
                                  │                                     │
                                  └──────────────┬──────────────────────┘
                                                 ▼
                     noisy action → Action encoder [B,32,1024]
                                                 │
                                                 ▼
                                   gated future cross-attention
                                                 │
                                                 ▼
                                  frozen 30-layer Action DiT/MoT
                                                 │
                                                 ▼
                                      action velocity / action chunk
```

离线 cache 只替代训练时的右侧 K-step generation。在线评测必须重新从当前观测生成
future，不得读训练 cache。

## 4. 实验组的精确定义

| 组 | Backbone | Adapter | Adapter 训练 | future 输入 | 在线 video sampling |
| --- | --- | --- | --- | --- | --- |
| B0 | 官方 checkpoint | 无 | 无 | 无 | 无 |
| A0 | 同一官方 checkpoint | 有 | 有 | `[B,48,2,14,28]` 全零 tensor | 无 |
| A1 | 同一官方 checkpoint | 有 | 有 | 正确 K=1 cache/online latent | 1 update |
| A2 | 同一官方 checkpoint | 有 | 有 | 正确 K=2 cache/online latent | 2 updates |
| A4 | 同一官方 checkpoint | 有 | 有 | 正确 K=4 cache/online latent | 4 updates |
| A-shuffle | 与对应 A-K 完全相同 | 同一已训练权重 | 不另训 | 跨 task/episode 的错误 K latent | K updates |

### 4.1 配对原则

A0/A1/A2/A4 必须保持相同：

- Adapter 类、shape 和参数量；
- Adapter 初始化 state hash；
- backbone/checkpoint/stats；
- train/dev sample manifest；
- sample 顺序；
- action noise seed/timestep；
- optimizer、LR、weight decay；
- microbatch、gradient accumulation；
- update steps 和 checkpoint selection；
- action denoising 20 steps；
- 评测 jobs、环境 seed、variant；
- 除 K/future 输入外的配置。

四个模型独立训练，不共享一个 K-conditioned Adapter。这样 K 是清晰的模型处理变量。

### 4.2 A0

A0 使用与真实 latent 完全同 shape、dtype 和 mask 的零 tensor，并走同一个 projector、
cross-attention 和 gate。这样：

- 参数量相同；
- action 计算图相同；
- projector 仍在图中；
- 不运行 Video DiT K-step sampling；
- 能控制额外参数、训练和 Adapter 路径。

不在第一版使用 learned null token，避免额外未使用参数和 DDP unused-parameter 分支。

### 4.3 A-shuffle

A-shuffle 不是另训一个“适应错误 future”的模型，而是在同一个 A-K checkpoint 上做
推理时替换。否则模型可能学会忽略错误输入，不能直接证明正确训练后的 Adapter 是否使用
future。

主 pilot 先报告 `A4-correct` vs `A4-shuffle`；counterfactual bundle 同时覆盖 K=1/2/4。
如果进入正式实验，再预注册是否把 K=1/2/4 shuffle 全部纳入成功率矩阵。

合法 donor 必须同时满足：

```text
donor.base_sample_id != recipient.base_sample_id
donor.episode_id      != recipient.episode_id
donor.task_id         != recipient.task_id
donor.K               == recipient.K
donor.split           == recipient.split
donor.cache/model fingerprint == recipient fingerprint
```

使用固定 seed 构造确定性 derangement，并保存 recipient→donor manifest。找不到合法 donor
时必须失败，不能退化为同 task 或自己。

正式在线 A-shuffle 不能读取训练 latent cache。donor 只能来自预注册 donor 当前观测，
并由冻结 Video DiT 当场生成错误 latent；correct 与 shuffle 都各运行一次相同 K sampler，
分别报告计时。

## 5. K-step future sampler

### 5.1 输入

```text
current_rgb:      [B,3,224,448], range [-1,1]
current_proprio:  [B,8], official normalized
context:          [B,128,4096]
context_mask:     [B,128]
initial_seed:     uint64 per base sample
K:                one of {1,2,4}
checkpoint/config/scheduler fingerprint
```

发布模型的 video expert 是 `action_conditioned=false`。因此 sampler 不接收：

- action target；
- predicted action；
- future image；
- environment success；
- downstream action seed。

### 5.2 初始化

```text
z ~ N(0,I), generated as float32 on rand_device=cpu
z = z.to(model_device, bf16)
z shape = [B,48,3,14,28]
z[:,:,0:1] = VAE(current_rgb)
```

同一 `base_sample_id` 的 K=1/2/4 使用同一个 `initial_seed` 和初始 `z`。seed 推导不包含
K：

```text
initial_seed = uint64_be(
    SHA256("thought3-noise-v1" || global_cache_seed || base_sample_id)[0:8]
)
```

`cache_sample_id` 包含 K；`base_sample_id` 不包含 K。二者不能混用。

### 5.3 完整 K-step schedule

每个 K 调用同一个上游 `WanContinuousFlowMatchScheduler.build_inference_schedule(K)`，
shift=5、train timesteps=1000，并完整走到 sigma=0。

伪代码：

```python
with torch.inference_mode():
    first = frozen_vae.encode(current_rgb)
    z = fixed_initial_noise(base_sample_id)
    z[:, :, 0:1] = first
    timesteps, deltas = scheduler.build_inference_schedule(K)
    for timestep, delta in zip(timesteps, deltas):
        velocity = frozen_video_dit(
            z,
            timestep=timestep,
            context=context_with_current_proprio,
            action=None,
        )
        z = scheduler.step(velocity, delta, z)
        z[:, :, 0:1] = first
    future = z[:, :, 1:].contiguous()
```

不运行：

- Action DiT；
- VAE decode；
- RGB video writer；
- Thought2 frame re-encoding。

### 5.4 数值验证

Phase C 同一输入/seed 必须验证：

1. shape、dtype、device；
2. current temporal slice 每步都等于 `first`；
3. K schedule metadata 与上游 scheduler 返回完全一致；
4. video-only K output 与上游 uncond joint video path 在预注册容差内；
5. 同 seed 重跑 hash 相同，换 seed hash 改变；
6. 改写真实未来 observation 不改变 latent；
7. latent 全程未经过 VAE decode。

如果 video-only 与 joint path 不等价，停止并改为项目侧“video-only MoT runner”，不能
静默接受另一条模型路径。

## 6. Future-to-Action Adapter

### 6.1 默认结构

```text
future [B,48,T,H,W]
  → Conv3d(48,256,kernel=(1,2,2),stride=(1,2,2))
  → [B,256,T,H/2,W/2]
  → flatten + factorized learned (T,H,W) positions
  → LayerNorm
  → future tokens [B,N,256]

action hidden [B,32,1024]
  → LayerNorm
  → Q projection 1024→512

future tokens
  → K projection 256→512
  → V projection 256→512

scaled dot-product attention, 8 heads × 64
  → output projection 512→1024
  → x + tanh(zero_gate) * output
```

默认 future tail `T=2,H=14,W=28`，projected grid 为 `2×7×14`，N=196。

### 6.2 参数量

| 组件 | 参数 |
| --- | ---: |
| Conv3d projector | 49,408 |
| future LayerNorm | 512 |
| factorized T/H/W position embeddings | 5,888 |
| action query LayerNorm | 2,048 |
| Q/K/V/output projections | 1,313,280 |
| scalar zero gate | 1 |
| 合计 | **1,371,137** |

占官方 MoT 参数的 0.0228%，占 Action expert 的 0.1343%。

### 6.3 约束

- `gate` 初始化为精确 0；
- residual 必须是 out-of-place；
- future mask 为 `[B,N]` bool，至少一个 valid token；
- mask 后 attention 不能产生 NaN；
- 支持 T/H/W 在配置上缩小；position embedding 通过切片/确定性插值适配；
- A0/A1/A2/A4 structural fingerprint 必须完全相同；
- forward 返回诊断信息时只返回 detach 后的 attention/gate norm；
- 默认无 dropout，保证 counterfactual deterministic；
- LoRA 默认关闭。

zero gate 的第一步通常只有 gate 获得有效梯度，内层 attention/projector 要在 gate 离开
0 后才获得明显梯度。这是已知优化特性，训练日志必须分别记录 gate 和各子模块 grad norm。
如果 100–500 step smoke 中 gate 不打开，只能在 development split 上调整 LR 或 gate
参数化，不能看正式测试集后再改。

## 7. 注入与 model wrapper

### 7.1 推荐入口

注入到：

```text
model.action_expert.action_encoder output
```

即 noisy action 从 7 维投影到 1024 维之后、进入 block 0 之前。

项目侧 `PartialFutureFastWAM` wrapper：

1. 持有原 FastWAM，不改变其类或 state dict；
2. 持有 `FutureToActionAdapter`；
3. 给 `action_encoder` 注册一个 output hook；
4. 使用 context manager 设置当前 future/mask；
5. hook 执行 Adapter 并返回新 tensor；
6. context 退出时清空引用；
7. 每次 forward 断言 hook 恰好调用一次。

推理 action diffusion 有 20 次 action network forward，因此每个 policy call 的 hook
总调用次数应为 20；单次 training action loss 为 1。计数不符即失败。

### 7.2 为什么不默认注入中层 block

MoT 并不直接调用 `ActionDiTBlock.forward()`，而是拆开调用 self-attention、
cross-attention 和 FFN。中层注入需：

- hook `block.cross_attn`；
- 或复制 `forward_action_with_video_cache()`；
- 处理 non-reentrant activation checkpoint 的重算；
- 防止日志计数/状态在 backward 被执行两次。

这条方案可以作为后续开发集消融，但不是最低风险的第一版。

## 8. Action-only 训练目标

### 8.1 输入

```text
current RGB only
current proprio only
language/context
normalized action target [B,32,7]
action_is_pad [B,32]
cached model-generated future [B,48,2,14,28] or A0 zeros
identity/provenance
```

数据对象不允许包含真实 future image/latent。

### 8.2 官方 flow-matching 目标

完全复用上游 action scheduler：

```text
t_action ~ shifted flow training distribution
noise_action ~ N(0,I)
noisy_action = (1-sigma) * action + sigma * noise_action
target_action = noise_action - action
pred_action = frozen ActionDiT + trainable Adapter
loss_token = MSE(pred_action, target_action), mean over action dim
loss_sample = valid-action masked mean over time
loss = mean(training_weight(t_action) * loss_sample)
```

第一版 `total_loss == official_action_loss`。gate regularization 配置存在但默认 0；任何非零
regularization 都必须作为 development decision 写入冻结协议。

### 8.3 当前状态路径

当前帧 VAE/Video DiT 仍按官方 `infer_action()` 产生 30 层 K/V cache。它们在
`torch.no_grad()` 下运行并 detach，因为：

- backbone 冻结；
- current-state 路径不需要 Adapter 梯度；
- 可节省 activation memory；
- 不改变 Action DiT 对当前表征的读取。

Action DiT 本身参数冻结，但从 Adapter 注入点到 action head 的计算必须启用 autograd，
使 loss 能回传到 Adapter。

不得直接调用带 `@torch.no_grad()` 的上游 `_predict_action_noise_with_cache()`；项目侧只
组合公开子模块完成等价 action forward。

## 9. 数据协议候选

Phase B 已新增并实现 `docs/thought3_data_protocol.md`；本节保留设计摘要，机器
schema 以 `src/fastwam_ood_eval/thought3/` 为准。

### 9.1 来源

只使用官方标准 LIBERO-fastwam demonstrations：

- libero_spatial；
- libero_object；
- libero_goal；
- libero_10。

不使用：

- LIBERO-Plus；
- Thought1/Thought2 rollout；
- final evaluation observations；
- success/failure label。

### 9.2 episode 级 split

默认按 `suite × task` 分层，用 seed `3407` 对完整 episode ID 排序/打乱：

- train：90% episode；
- development：10% episode，且每个有足够 episode 的 task 至少 1 个；
- frame 永远继承 episode split；
- split manifest 生成后只读。

如果某 task episode 太少而无法同时分配，命令失败并要求显式规则，不能把相邻 frame
随机拆分。

最终 test 不是 demonstration dev split。它沿用 Clean/LIBERO-Plus simulator，但使用在
正式运行前冻结的新 evaluation seed/variant manifest。

### 9.3 ID 设计

先构造 canonical JSON：

```text
dataset_revision
suite
task_id + task_name_hash
dataset_repo
demonstration/episode_index
frame_index + timestamp
camera_keys/order + concat/preprocess hash
language_hash
checkpoint_hash
stats_hash
sampler_config_hash
split_manifest_hash
```

```text
base_sample_id  = sha256(canonical JSON excluding K and noise seed)
initial_seed    = deterministic(base_sample_id, global_cache_seed)
cache_sample_id = sha256(base_sample_id + K + initial_seed + cache schema)
```

这样既满足 cache identity 包含 K/seed，又能严格配对 K=1/2/4。

## 10. Future latent cache

### 10.1 目录

```text
outputs/thought3/cache/<cache_fingerprint>/
├── manifest.json
├── plan.jsonl
├── k_1/
│   ├── shard-00000.safetensors
│   ├── shard-00000.metadata.jsonl
│   └── shard-00000.manifest.json
├── k_2/
└── k_4/
```

默认每 shard 512 samples，约 36.75 MiB raw latent/K。写入流程：

1. 写同目录临时文件；
2. flush/fsync；
3. 计算 file SHA-256 和 per-sample tensor SHA-256；
4. 验证 tensor count、ID、shape、dtype；
5. atomic rename；
6. 最后写 shard manifest。

resume 只跳过 manifest 与 checksum 都通过的完整 shard。损坏或半成品不视为完成。

### 10.2 schema

全局：`thought3.future_cache.v1`。

每个样本至少记录：

- base/cache sample ID；
- suite/task/episode/frame/cameras/language；
- split；
- K；
- initial noise seed；
- actual timestep/sigma/delta arrays；
- latent shape/layout/dtype；
- checkpoint/stats/FastWAM/config hashes；
- sampler and preprocessing fingerprints；
- generation latency；
- peak GPU memory；
- latent checksum；
- source kind：`model_sampled_from_current`；
- `uses_ground_truth_future=false`。

训练 loader 必须精确匹配 K 和所有 fingerprint，禁止“找到相同文件名就读取”。

### 10.3 磁盘估算

future tail raw payload：

```text
75,264 bytes/sample/K
225,792 bytes/sample for K=1+2+4
```

| 样本数 | 单 K raw | 三个 K raw | 三个 K + 5% metadata/shard 规划 |
| ---: | ---: | ---: | ---: |
| 10,000 | 0.701 GiB | 2.103 GiB | 约 2.21 GiB |
| 100,000 | 7.010 GiB | 21.029 GiB | 约 22.08 GiB |
| 1,000,000 | 70.095 GiB | 210.285 GiB | 约 220.80 GiB |

真实样本数要在训练数据下载后由 `thought3-plan-cache` 读取 metadata 再给出。

可选共享 current-frame latent 为 36.75 KiB/sample，会额外增加：

- 10k：0.350 GiB；
- 100k：3.505 GiB；
- 1M：35.048 GiB。

第一版默认不缓存 current latent；只有 Gate C 显存/吞吐证明必要时才启用。

### 10.4 时间估算

Thought2 测得 20-step joint+decode 平均 3.355 s，不能直接视为 cache 速度。Phase D 先跑
100–500 sample benchmark，记录每 K 的 warm/cold P50/P95。

三卡理想墙钟公式：

```text
hours ≈ N × (t_K1 + t_K2 + t_K4) / (3 × 3600) × 1.10
```

其中 1.10 是写盘、校验和不均衡的规划余量。以 100k samples 为例的容量场景：

| 三个 K 合计耗时/样本 | 3 GPU 理想时间 | 加 10% 规划 |
| ---: | ---: | ---: |
| 1.0 s | 9.26 h | 10.19 h |
| 1.5 s | 13.89 h | 15.28 h |
| 2.0 s | 18.52 h | 20.37 h |

这些是容量表，不是尚未实测的性能承诺。

## 11. Adapter-only checkpoint

每个 checkpoint 是目录：

```text
checkpoints/step_000500/
├── adapter.safetensors
├── optimizer.pt
├── scheduler.pt
├── rng_rank_000.pt
├── trainer_state.json
├── manifest.json
└── checksums.json
```

`adapter.safetensors` 只含 Adapter/可选 LoRA。manifest 必须绑定：

- B0 checkpoint SHA-256；
- official stats SHA-256；
- FastWAM/main repo commit；
- Adapter structural fingerprint；
- variant/K；
- split/cache fingerprints；
- config hash；
- train seed；
- optimizer/scheduler；
- global step、epoch、sample cursor；
- trainable param names/count；
- frozen param hashes；
- precision、world size、GPU inventory。

resume 时先验证所有只读 identity，再恢复 optimizer/RNG/cursor。任何 backbone、cache 或
structural mismatch 都拒绝恢复，不能 `strict=False`。

## 12. 训练与并行

### 12.1 第一版默认

```text
precision: bf16
microbatch_per_gpu: 1
gradient_accumulation: 8（smoke 可更小）
optimizer: AdamW
trainable: Adapter only
LoRA: off
DDP: one process per GPU
gradient checkpoint: action path on if Gate C 需要
max memory abort: 43 GiB/card
```

LR、steps、warmup 和 weight decay 在 development split 上决定；Phase A 不伪造最优值。
第一轮建议从 Adapter LR `1e-4` 开始做 100–500 step smoke，并同时观察 gate 是否打开，
而不是直接启动正式训练。

### 12.2 必记训练指标

- action flow loss；
- development action loss；
- total/各子模块 grad norm；
- gate raw value 与 `tanh(gate)`；
- Adapter residual norm / action hidden norm；
- future attention output norm；
- finite/NaN/Inf count；
- LR；
- step time、samples/s；
- GPU peak allocated/reserved；
- cache read latency；
- trainable/frozen param count；
- frozen hash audit；
- checkpoint/resume identity。

## 13. Counterfactual：证明动作读取 future

固定：

- current RGB；
- language/context；
- proprio；
- action diffusion initial noise；
- action timesteps；
- checkpoint/Adapter；
- deterministic kernels；
- K（比较 same-K correct/null/shuffle 时）。

替换：

- correct future；
- null future；
- shuffled future；
- random future（补充）；
- K=1/2/4。

记录 normalized 与 simulator space：

- action chunk L1/L2；
- first executed action L1/L2；
- translation/rotation direction cosine；
- gripper sign/value change；
- open-loop end-effector integrated trajectory change；
- action SHA-256；
- gate/attention/residual norm；
- paired task success change。

先做“同一 future 重放两次”的数值噪声 calibration。只有干预差异显著超过 replay
噪声，才能称为可测量 action sensitivity。

解释规则：

| 观察 | 允许结论 |
| --- | --- |
| shuffle/null 后动作不变 | Adapter 没有实际使用 future，或 gate/路径失效 |
| 动作改变，但 correct 不优于 shuffle | 使用了 future，但信息质量/训练映射无控制价值 |
| correct 优于 null 和 shuffle | 支持 future 信息具有增量控制价值 |
| A0 优于 B0，A-K 不优于 A0 | 收益更可能来自 Adapter/再训练，而非 future |
| A-K OOD 增益伴随严重 ID 降低 | 不是无代价鲁棒性提升，需报告 trade-off |

## 14. 在线评测与 latency

正式 A1/A2/A4：

```text
current observation
  → online K-step video sampler
  → future Adapter
  → fixed 20-step action sampler
  → action
```

禁止打开 offline training cache。计时用 CUDA events + synchronize，并拆分：

- preprocessing；
- current-frame VAE encode；
- current Video DiT/KV prefill；
- K-step future sampling；
- Adapter 累计；
- 20-step Action DiT；
- official denormalization；
- total policy；
- peak allocated/reserved。

报告 warmup 后 P50/P95，外加 first-call warmup。明确：

- 是否 decode video：正式控制为 false；
- future 保持 latent：true；
- cache read latency：不属于部署延迟，也不得报告为它。

## 15. 统计与正式协议

在看正式结果前新增并冻结：

- `docs/thought3_data_protocol.md`
- `docs/thought3_analysis_protocol_DRAFT.md`
- 最终 `docs/thought3_analysis_protocol_FROZEN.md`

主指标：

```text
task-equal OOD success(A-K) - task-equal OOD success(A0)
```

重要对照：

- B0 vs A0；
- A0 vs A1/A2/A4；
- same checkpoint correct vs shuffle；
- Clean 与五类 OOD；
- latency/peak memory。

聚合顺序：

1. episode；
2. task 内平均；
3. task 等权；
4. suite-stratified task bootstrap，至少 10,000 replicates；
5. 多 train seed 时先 seed 内聚合，再报告 seed 间分布。

不能把大量 episode 当独立样本。正式排除规则、success checker、bootstrap seed、train
seeds、checkpoint selection 和最终 evaluation manifest 必须预先冻结。

## 16. CLI 设计

新增：

```text
fastwam-ood thought3-audit
fastwam-ood thought3-plan-cache
fastwam-ood thought3-build-cache
fastwam-ood thought3-validate-cache
fastwam-ood thought3-train
fastwam-ood thought3-counterfactual
fastwam-ood thought3-evaluate
fastwam-ood thought3-aggregate
fastwam-ood thought3-report
```

共同参数：

- `--config`
- `--device`
- `--dry-run`
- `--resume`

约束：

- `--dry-run` 不 import torch、Hydra 或加载 checkpoint；
- build/train/evaluate 需要显式确认环境变量，防止意外长跑；
- 正式长训练永不由代码生成过程自动启动；
- 每个 CLI 写独立 manifest 和 status；
- rank shard 必须集合完备、互斥。

## 17. 计划新增/修改文件

### 17.1 新增代码

```text
src/fastwam_ood_eval/thought3/
├── __init__.py
├── cli.py
├── config.py
├── schemas.py
├── future_sampler.py
├── future_cache.py
├── cache_builder.py
├── cache_validator.py
├── adapter.py
├── injection.py
├── model_wrapper.py
├── training_dataset.py
├── trainer.py
├── checkpointing.py
├── counterfactuals.py
├── latency.py
├── evaluator.py
├── aggregate.py
└── report.py
```

### 17.2 新增配置

```text
configs/thought3/
├── cache_smoke.yaml
├── train_a0_smoke.yaml
├── train_a1_smoke.yaml
├── train_a2_smoke.yaml
├── train_a4_smoke.yaml
├── pilot_b0.yaml
├── pilot_a0.yaml
├── pilot_a1.yaml
├── pilot_a2.yaml
├── pilot_a4.yaml
├── pilot_shuffle.yaml
└── formal_template.yaml
```

### 17.3 新增测试

```text
tests/test_thought3_adapter.py
tests/test_thought3_zero_init.py
tests/test_thought3_freezing.py
tests/test_thought3_cache_schema.py
tests/test_thought3_cache_resume.py
tests/test_thought3_cache_validation.py
tests/test_thought3_no_leakage.py
tests/test_thought3_counterfactual.py
tests/test_thought3_checkpoint.py
tests/test_thought3_old_cli_regression.py
tests/test_thought3_mock_training.py
tests/test_thought3_sharding.py
tests/test_thought3_online_no_cache.py
tests/test_thought3_provenance.py
```

### 17.4 只做加法式修改

- `src/fastwam_ood_eval/cli.py`：惰性注册 Thought3 子命令；
- `docs/research_index.md`：增加 Thought3 文档入口；
- 必要时 `README.md`：只增加独立入口，不改旧命令。

不修改现有 `fastwam_ood_eval.config.EvalConfig`；Thought3 使用自己的强类型 config，
避免改变旧 YAML 的解析。

## 18. Phase A–F 执行计划与门禁

### Phase A：审计与确认

交付：

- 本文；
- `thought3_upstream_audit.md`；
- `thought3_risk_register.md`。

门禁：用户确认第 19 节的关键选择。该门禁已于 2026-07-27 通过；没有因此启动训练。

### Phase B：CPU/mock

实现：

- schemas/config；
- Adapter；
- 注入 wrapper；
- cache plan/shard/resume/checksum；
- mock action loss/trainer/checkpoint；
- counterfactual；
- additive CLI；
- old CLI regression。

门禁：

- 全部旧测试通过；
- 新 CPU tests 通过；
- dry-run 不加载大模型；
- Thought1/2 文件哈希不变；
- 无 GPU/大 checkpoint load。

状态：**已通过**。验收见
`docs/thought3_phase_b_report.md`；Phase B 只产生临时测试工件，无真实研究结果。

### Phase C：单 GPU tensor smoke

只用一条真实标准 LIBERO train sample：

- 生成 K=1/2/4；
- 验证 native latent；
- Adapter forward；
- 一次 action loss backward；
- frozen grad/hash；
- zero gate B0/A0；
- checkpoint round-trip；
- peak memory。

门禁：

- 无泄漏；
- finite；
- only Adapter grad；
- peak <43 GiB；
- video-only 与上游路径在容差内。

### Phase D：小 cache smoke

范围：一个 suite、一个 task、少量 train/dev episode。

验证：

- plan/build/resume/validate；
- K 配对；
- checksum corruption detection；
- 三卡 shard 完备互斥；
- online vs cached latent；
- 实测磁盘/吞吐。

### Phase E：小训练 smoke

范围：

- A0/A1；
- 单卡；
- 100–500 steps；
- 不做成功率结论。

门禁：

- loss 可下降；
- gate/grad finite；
- resume 完全一致；
- frozen hash 不变；
- A0/A1 参数量与预算一致。

### Phase F：技术 pilot

范围：

- B0/A0/A1/A2/A4/A-shuffle；
- 一个 suite；
- 3–5 tasks；
- 一个 train seed；
- 独立 pilot Clean/OOD jobs。

目标：

- future 干预超过 replay numerical floor；
- correct/null/shuffle 动作可区分，或明确判定 Adapter 忽略 future；
- 在线 latency/显存可接受；
- 无明显工程性 ID 崩溃；
- 决定是否值得 Phase G。

pilot 只作 go/no-go 和协议固定，不写成正式效果结论。

## 19. 已确认的关键选择

用户于 2026-07-27 确认 Phase A，以下默认值已用于 Phase B：

1. **K 语义**：每个 K 使用同一连续 scheduler 的完整 K-step 离散化并到达
   sigma=0，不取 20-step prefix。
2. **注入位置**：`action_encoder` 后、block 0 前，仅一次。
3. **Adapter 尺寸**：future dim 256、attention dim 512、8 heads，1,371,137 参数。
4. **A0**：全零 native future，通过同一 projector/attention；无 learned null。
5. **A-shuffle**：不另训，作为同一 A-K checkpoint 的推理干预；pilot 主组用 K=4。
6. **split**：suite×task 分层、episode 级 90/10，seed 3407。
7. **LoRA**：第一版关闭，只有 Adapter-only 无信号后才建立独立实验族。
8. **current latent cache**：默认关闭，Gate C 证明有必要再启用。
9. **并行**：先 DDP，不启用 FSDP/ZeRO/CPU offload。
10. **训练 API**：项目侧 action-only flow loss，真实 future video 不进入对象。

Phase B 编码和 CPU/mock 测试已完成。该确认没有授权 GPU smoke、真实 cache 或训练；
这些动作仍受 Phase C/D/E 独立门禁约束。
