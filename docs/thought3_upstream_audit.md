# Thought3 上游与可行性审计

状态：Phase A 已完成静态审计，等待设计确认；尚未实现模型、生成 cache 或训练  
审计时间：2026-07-27（Asia/Shanghai）  
研究分支：`feature/thought3-partial-future-adapter`

## 1. 审计结论

阶段三可以实现，但不能直接复用上游 `Trainer.training_loss()` 或
`FastWAM.infer_joint()` 作为最终训练/缓存入口。

推荐的最小实现是：

1. 保持官方 Fast-WAM checkpoint、VAE、Video DiT、Action DiT 和 proprio
   encoder 冻结。
2. 新建项目侧 video-only K-step sampler，从当前观测、语言、proprio 和固定噪声
   生成原生 diffusion latent；不运行 Action DiT，不做 VAE decode。
3. 只缓存 latent 的 future tail，形状为 `[B, 48, 2, 14, 28]`。
4. 在 `ActionDiT.action_encoder` 输出后、block 0 前注入一次
   Future-to-Action gated cross-attention。
5. 新建项目侧 action-only flow-matching loss，严格复用官方 action scheduler、
   target、mask 和权重公式；训练输入 schema 不含真实未来图像。
6. 第一版只训练约 1.371M 参数的 Adapter，不启用 LoRA。

当前有两个开跑前阻塞项：

- 四个官方标准 LIBERO LeRobot 训练目录在本工作区不存在；
- `[B,48,2,14,28]` 是由真实上游代码与冻结配置推导出的运行时 contract，
  尚未通过 Phase C 单卡真实 tensor smoke 再实测一次。

这两个问题不妨碍完成 CPU/mock 实现，但在 Phase C/D 前必须解决。

## 2. Git、上游与冻结结果快照

### 2.1 主仓库

| 项目 | 审计值 |
| --- | --- |
| 当前分支 | `feature/thought3-partial-future-adapter` |
| 当前 HEAD | `37ef1dbce60eb79c36adb980b5637346c8671cdb` |
| `main` / `origin/main` | 均为 `37ef1dbce60eb79c36adb980b5637346c8671cdb` |
| 开始审计时工作树 | clean |
| Thought1 tag | `thought1-baseline-v1`，解引用到 `0df5fe224e5c5dd767ed105802821b69c141e041` |
| Thought2 runner commit | `0fb8350a4a6c3fe2976b04b6f1fbcdb0e6c2cc17` |
| Thought2 完整分析端点 | `37ef1dbce60eb79c36adb980b5637346c8671cdb` |
| Thought2 tag | 不存在；正式引用必须写 commit，不能虚构 tag |

Thought1 tag 的 tag object 是
`8822e75aa7ea8abd1f78f91dfdb1cef8813692b5`，上表记录的是解引用后的 commit。

### 2.2 固定上游

| 上游 | commit | 审计状态 |
| --- | --- | --- |
| FastWAM | `45d8e1458921d83f8ad6cf9ce993d371208dabd0` | clean |
| LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` | clean |
| LIBERO-Plus | `4976dc30028e805ff8094b55501d532c48fec182` | 存在既有未跟踪 `.downloads/`；阶段三不得读取或修改它 |

阶段三不修改 `third_party/FastWAM`。所有新逻辑放在
`src/fastwam_ood_eval/thought3/`。

### 2.3 官方资产

| 资产 | 值 |
| --- | --- |
| checkpoint | `checkpoints/fastwam_release/libero_uncond_2cam224.pt` |
| checkpoint size | 12,041,735,140 bytes |
| checkpoint SHA-256 | `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579` |
| checkpoint step | 21,700 |
| dataset stats SHA-256 | `30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638` |
| FastWAM root eval config SHA-256 | `c5b60228495987739fd222d620e9a643d9d527d44e2e63c0832c403041931df3` |
| task config SHA-256 | `27bdc30545aa06bca81137690a241d469d5a67a433881976bd79da3d4171d1a0` |
| model config SHA-256 | `ab3c2ffde9933e7576c747fecce82bd7d28c9c6478c1b53fcac02b3012be416c` |
| data config SHA-256 | `4d40ad0cd31337e0ccd04232c548a1a292bf3a0672644db781af4155bedb9b7d` |

使用 `torch.load(..., mmap=True, weights_only=True)` 只读统计得到：

| checkpoint 子模块 | 参数量 | bf16 payload |
| --- | ---: | ---: |
| Video expert | 4,999,787,712 | 9.313 GiB |
| Action expert | 1,020,886,023 | 1.902 GiB |
| MoT 合计 | 6,020,673,735 | 11.214 GiB |
| Proprio encoder | 36,864 | 72 KiB |

### 2.4 Thought1/Thought2 只读证据

审计前记录以下正式文件哈希，并在 Phase A 文档完成后再次核验；前后完全一致：

| 文件 | SHA-256 |
| --- | --- |
| `outputs/thought1/fastwam/combined/experiment_manifest.json` | `57dd93f51a2491423f1b14f0d90523f219218698e231a133dcef114caca132ee` |
| `outputs/thought1/fastwam/combined/summary/metrics.json` | `0aa1173038a1c37d37123570a83ff9f08667490e3f94276345c802151897dbb5` |
| `outputs/thought1/fastwam/combined/summary/report.md` | `889d567e4882b9982fb2121788dbbacdf983e1556faf8e4f9bb5a29768f8e137` |
| `outputs/thought2/five_category_formal_v1/combined/diagnostic_manifest.json` | `ff4aef249d800dbcec44d8f319d89efbebd0c7b1be78a99213eb7e3b2f1d7e09` |
| `outputs/thought2/five_category_formal_v1/combined/summary/diagnostic_metrics.json` | `b47d29f5c176fea74797f32f872fc14d9c23f370bea8c667cfccf4ccdfc942c3` |
| `outputs/thought2/five_category_formal_v1/formal_analysis_v1/analysis_manifest.json` | `ad3793b3ef8a2042c6eff90c0f238ea56f1afdda62a3e79dbb6518dede1fe76f` |
| `outputs/thought2/five_category_formal_v1/formal_analysis_v1/formal_analysis.json` | `9d51e0f46c7af73340b390c3acdfd30fa05c8d1e2fa92794ebcae0f112c69f19` |
| `outputs/thought2/five_category_formal_v1/run_status.txt` | `32128801f41bfad982645fb2a8358df40bee638206623805bc2dedf6d13be718` |

现有正式结果规模约为 Thought1 1.8 GiB、Thought2 267 MiB。阶段三输出只能写入
`outputs/thought3/`。

## 3. 十五项强制审计

| # | 审计项 | 证据与结论 | 状态 |
| ---: | --- | --- | --- |
| 1 | 当前 Git 状态与 commit | 独立分支，起点为 clean main `37ef1db` | 通过 |
| 2 | Thought1/2 tag/commit | Thought1 有 tag；Thought2 只有可追溯 commit，无 tag | 通过，需在 manifest 如实记录 |
| 3 | `FastWAMAdapter` 调用链 | Hydra compose → instantiate → official checkpoint load → official processor/stats → `_predict_action_chunk()` → `infer_action()` | 通过 |
| 4 | Action DiT 结构 | 30 blocks，hidden 1024，FFN 4096，24 heads × 128，action `[B,32,7]` | 通过 |
| 5 | future latent tensor | 原生 VAE/diffusion layout `[B,C,T,H,W]`；完整 `[B,48,3,14,28]`，future tail `[B,48,2,14,28]`，bf16 | 静态 contract 通过；Phase C 实测待办 |
| 6 | K-step sampling 入口 | 上游 scheduler 可直接构建 K 步完整 schedule；上游没有 video-only public sampler，需项目侧封装 `video_expert` | 可实现 |
| 7 | Adapter 注入位置 | `action_encoder` 后、block 0 前最少侵入；中层 `cross_attn` hook 受 MoT/checkpoint 耦合影响 | 推荐方案已定，待确认 |
| 8 | action normalization | 官方 stats + `FastWAMProcessor` min/max normalize；执行前官方 denormalize 与 gripper 变换 | 必须原样复用 |
| 9 | LIBERO 训练加载 | 上游 LeRobot 按 frame 取 33 observation、32 action；配置默认 `val_set_proportion=0.0` | 不能直接作为 Thought3 split |
| 10 | checkpoint save/resume | 上游支持 full MoT 与 Accelerate state，但会保存/训练过多参数 | 只借鉴机制，需 Adapter-only 实现 |
| 11 | 3 GPU 约束 | 3×RTX 4090，每卡 49,140 MiB；2026-07-27 11:17 快照可见三卡 | 通过 |
| 12 | 预计峰值显存 | 已测 shadow 20-step 峰值 24,841 MiB；Adapter train microbatch 1 暂估 28–36 GiB | 必须在 Gate C 实测 |
| 13 | FSDP/ZeRO/checkpoint/offload | 首选普通 DDP + bf16 + action activation checkpointing；暂不需要 FSDP/ZeRO/offload | 有回退方案 |
| 14 | 安全冻结模块 | VAE、Video DiT、Action DiT、MoT、proprio encoder 全冻结；只允许 Adapter trainable | 可自动验证 |
| 15 | 真实未来泄漏 | 上游训练 sample 含真实未来视频；Thought3 训练入口必须不接受该字段，Adapter 只读模型生成 cache | 高风险但可阻断 |

## 4. 当前动作调用链

项目侧入口见
`src/fastwam_ood_eval/policy/fastwam_adapter.py:31-108` 和
`src/fastwam_ood_eval/policy/fastwam_adapter.py:119-151`：

```text
FastWAMAdapter
  ├─ compose third_party/FastWAM/configs/sim_libero.yaml
  ├─ instantiate FastWAM
  ├─ official._load_model_checkpoint()
  ├─ load official dataset stats / FastWAMProcessor
  └─ official._predict_action_chunk()
       ├─ official image concat: two 224×224 cameras → [1,3,224,448]
       ├─ official proprio normalize
       ├─ FastWAM.infer_action(num_inference_steps=20)
       │    ├─ VAE encode current image → [1,48,1,14,28]
       │    ├─ Video DiT current-frame tokens → [1,196,3072]
       │    ├─ prefill 30-layer video K/V cache
       │    └─ 20 action flow updates
       │         └─ Action DiT: [1,32,7] → [1,32,1024] → [1,32,7]
       └─ official action denormalize / gripper conversion
```

关键事实：基础 Fast-WAM 的 Action DiT 已经读取“当前帧 Video DiT 表征”，但没有读取
生成的 future latent。上游 attention mask 明确限制 action query 只访问 first-frame
video tokens，见
`third_party/FastWAM/src/fastwam/models/wan22/fastwam.py:385-407`。

因此阶段三增加的是独立的 future→action 路径，不应替换或删除原 current-state 路径。

## 5. Action DiT 与安全注入位置

冻结配置：

- action horizon：32；
- action dimension：7；
- hidden dimension：1024；
- block 数：30；
- FFN dimension：4096；
- attention：24 heads、每头 128；
- text dimension：4096；
- action sampling steps：正式比较固定为 20。

`ActionDiT.pre_dit()` 先通过 `action_encoder: Linear(7,1024)` 得到
`[B,32,1024]`。之后 MoT 的 action query 在每层同时访问：

- 当前帧 Video DiT 的冻结 K/V；
- 当前 action token 的 K/V；
- text/proprio context。

审计过的候选位置：

| 位置 | 优点 | 风险 | 结论 |
| --- | --- | --- | --- |
| `action_encoder` 输出后 | 单一稳定入口；训练与推理都经过；避开 block checkpoint；gate=0 容易验证 | future 信息需要通过 30 层传播 | 第一版推荐 |
| 第 15 个 action block 的 text cross-attention 后 | 路径更短，可能信号更强 | MoT 内部调用、forward hook 与 gradient checkpointing 更耦合 | 仅作为开发集消融 |
| 每个 action block | 容量大 | 参数、显存、归因和侵入性都显著增加 | 第一版禁止 |

第一版通过项目侧 wrapper 给 `action_encoder` 注册受上下文管理的 output hook：

```text
action_encoder output x
  ├─ Query = LayerNorm(x)
  └─ K/V = projected future tokens
             ↓
       cross-attention
             ↓
x' = x + tanh(gate) * adapter_output
```

它不原地修改 `x`，不修改上游源文件，异常退出时必须清理 active future context，并检查
每次 action prediction 的 hook 调用次数。

## 6. 原生 future latent 的真实语义

官方 LIBERO 输入和模型配置：

- 两路相机各 `224×224`，横向拼接为 `224×448`；
- 33 个 action-rate observation、32 个 action；
- `action_video_freq_ratio=4`，因此 video horizon 为 9 RGB frames；
- Wan2.2 VAE38：channel 48、spatial factor 16、temporal factor 4；
- Video DiT patch size `[1,2,2]`。

由上游 `infer_joint()` 的实际分配公式：

```text
latent_T = (9 - 1) / 4 + 1 = 3
latent_H = 224 / 16 = 14
latent_W = 448 / 16 = 28
```

得到：

| 名称 | tensor |
| --- | --- |
| 完整视频 diffusion state | `[B,48,3,14,28]` |
| 固定当前帧 latent | `[:, :, 0:1]`，即 `[B,48,1,14,28]` |
| Adapter 的 future tail | `[:, :, 1:]`，即 `[B,48,2,14,28]` |
| layout | `[batch, channel, latent_time, latent_height, latent_width]` |
| 运行 dtype | `torch.bfloat16` |
| cache dtype | 默认 bf16，读取后按模型 dtype/device 转换 |
| 是否 native diffusion latent | 是 |
| 是否 VAE decoded | 否 |
| 是否来自真实未来视频编码 | 否 |

future tail 共 `48×2×14×28 = 37,632` 个元素；bf16 payload 为
75,264 bytes，即 73.5 KiB/样本/K。

Video DiT 的 `[1,2,2]` patch 后为：

```text
[B,48,2,14,28]
  → [B,256,2,7,14]     # Thought3 small projector
  → [B,196,256]        # 196 future tokens
```

这里的 K latent 是 K 次 Euler/flow sampling update 后的模型状态，不是 Thought2
“把预测/真实 RGB 帧重新编码”得到的诊断 embedding。两者不得混称。

## 7. K=1/2/4 的真实 scheduler 语义

上游 `WanContinuousFlowMatchScheduler` 使用：

```text
phi(u) = 5u / (1 + 4u)
u = linspace(1, 0, K+1)
t = 1000 * phi(u[:-1])
delta = phi(u[1:]) - phi(u[:-1])
z_next = z + velocity * delta
```

第一版定义每个 K 都走完从 `sigma=1` 到 `sigma=0` 的完整 K-step schedule：

| K | sigma nodes | model timesteps | deltas |
| ---: | --- | --- | --- |
| 1 | `[1, 0]` | `[1000]` | `[-1]` |
| 2 | `[1, 0.833333, 0]` | `[1000, 833.333]` | `[-0.166667, -0.833333]` |
| 4 | `[1, 0.9375, 0.833333, 0.625, 0]` | `[1000, 937.5, 833.333, 625]` | `[-0.0625, -0.104167, -0.208333, -0.625]` |

每次 update 后都把 temporal index 0 恢复为当前帧 VAE latent，和上游
`infer_joint()` 一致。K 间保持：

- 同一连续 scheduler 与 shift=5；
- 同一 checkpoint、context、latent shape 和 future horizon；
- 对同一 `base_sample_id` 使用完全相同的起始噪声；
- action 分支始终 20 步；
- 不解码 RGB。

不能把 K 定义为 20-step schedule 的前 K 个 prefix。那样 K=1/2/4 会停在不同的
非终止 sigma，混淆“求解精度”和“剩余噪声量”。

上游没有独立 video-only helper；项目侧 sampler 将调用冻结
`video_expert.forward()`。由于发布模型 `action_conditioned=false` 且 video query
不会访问 action keys，Phase C 必须用相同 seed 比较项目侧 video-only 输出和上游
joint video 输出，建立数值等价容差。

## 8. 官方 normalization/denormalization

训练和评测必须复用：

- `FastWAMProcessor.action_state_transform()`；
- 发布的 dataset stats；
- `norm_default_mode=min/max`；
- `use_stepwise_action_norm=false`；
- action 7 维、proprio 8 维；
- 官方 `_denormalize_action()`；
- 官方 gripper 映射、反转与可选 binarize。

不允许根据 Thought3 train/dev split 重算 stats，也不允许把 normalized action 与
simulator action 混合记录。训练 loss 在官方 normalized action space 中计算；在线执行前
仍走官方后处理。

## 9. 训练数据与 split 审计

发布配置要求四个标准 LIBERO LeRobot 目录：

```text
data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot
data/libero_mujoco3.3.2/libero_object_no_noops_lerobot
data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot
data/libero_mujoco3.3.2/libero_10_no_noops_lerobot
```

2026-07-27 审计时 `data/` 目录不存在。Phase C 前应按上游 README 下载
`yuanty/LIBERO-fastwam`，并把下载 revision、文件清单和数据根 hash 写入
Thought3 data manifest。

上游 `BaseLerobotDataset` 已按 episode 进行可选 split，但发布配置把
`val_set_proportion` 设为 0，因此所有 episode 都进入训练集合。阶段三必须建立独立、
显式、不可变的 split manifest，不能依赖这个默认值。

LeRobot 样本可提供：

- `dataset_index`；
- `episode_index`；
- `frame_index`；
- `task_index` / task string；
- timestamp；
- camera keys；
- action/proprio pad mask。

这些字段足以构造稳定 ID。为降低泄漏面，Thought3 dataset 输出只允许：

```text
current two-camera observation
current proprio
language/context
normalized target action chunk
pad masks
stable identity/provenance
```

真实后续 observation 不属于允许 schema。未来 action chunk 仅作为 action loss
supervision，不能输入 future sampler 或 Adapter。

## 10. 上游 training/checkpoint 机制是否可复用

上游已有：

- Accelerate/DeepSpeed 训练；
- optimizer/scheduler；
- step、epoch、dataloader cursor；
- full model weight checkpoint；
- training-state resume。

但上游 trainer 会把 `model.dit` 整体设为 trainable，且 `FastWAM.save_checkpoint()`
保存完整 MoT。这不符合 Adapter-only 要求。

此外，上游 `FastWAM.training_loss()`：

1. 编码完整真实 demonstration video；
2. 同时计算 video 和 action flow loss；
3. 虽然 action attention 只访问 first-frame video token，仍扩大了真实未来进入
   运行图的泄漏面与显存开销。

阶段三因此新建 action-only trainer：

- 复用 action scheduler 的 `sample_training_t()`、`add_noise()`、
  `training_target()`、`training_weight()`；
- 复用 action pad mask 与 MSE reduction；
- 当前帧 Video K/V 在 no-grad 下产生并 detach；
- Action DiT 保持 autograd 路径，只为 Adapter 求梯度；
- loss 输入结构中不存在真实 future image/latent。

Adapter checkpoint 只保存 Adapter/可选 LoRA、optimizer/scheduler/RNG/cursor 和
provenance，不复制 12 GB backbone；加载时强制核对 backbone SHA-256。

## 11. GPU、显存和并行策略

### 11.1 当前硬件

2026-07-27T11:17:39+08:00 的只读快照：

| GPU | 型号 | 总显存 MiB | 已用 MiB | 利用率 |
| ---: | --- | ---: | ---: | ---: |
| 0 | NVIDIA GeForce RTX 4090 | 49,140 | 1,412 | 50% |
| 1 | NVIDIA GeForce RTX 4090 | 49,140 | 15 | 0% |
| 2 | NVIDIA GeForce RTX 4090 | 49,140 | 202 | 40% |

GPU 状态是瞬时值，正式命令必须重新 preflight，不能依赖本表。

### 11.2 已测基线

Thought2 的 732 个 20-step shadow probes：

- future generation mean：3,354.66 ms；
- P50：3,316.96 ms；
- P95：3,564.12 ms；
- synchronized peak allocated memory：24,841.09 MiB。

该计时包含 joint video/action generation 和 VAE decode，不能当成 Thought3
video-only K=1/2/4 在线延迟，也不能按 K/20 直接线性外推。

### 11.3 Adapter 估算

默认 Adapter 为 1,371,137 参数：

- fp32 可训练 weights 约 5.23 MiB；
- bf16 inference weights 约 2.62 MiB；
- fp32 master/grad/Adam states 规划约 21–27 MiB；
- 相对 6.021B MoT 为 0.0228%；
- 相对 1.021B Action expert 为 0.1343%。

冻结 backbone 仍需驻留。基于 24.84 GiB 已测 inference 峰值和 action
backprop activation，microbatch=1、bf16 的保守规划区间是 28–36 GiB/卡。
这是预算，不是实测结果；Gate C 以 `torch.cuda.max_memory_allocated()` 为准。

安全阈值：

- 首轮只允许 microbatch 1；
- 峰值超过 43 GiB 立即停止；
- 先增加 gradient accumulation；
- 再启用 Action/MoT non-reentrant activation checkpointing；
- 仍超限才评估 CPU offload 或 FSDP。

### 11.4 是否需要 FSDP/ZeRO

第一版不需要。原因是：

- 每卡已能容纳完整 frozen inference model；
- optimizer 只覆盖约 1.37M 参数；
- ZeRO/FSDP 对 Adapter optimizer 的节省很小，却增加 checkpoint 与冻结验证复杂度。

三卡首选一进程一卡的 DDP，backbone 每卡复制、Adapter 梯度 all-reduce。
只有 Gate C 实测失败时才升级并行策略。

## 12. 可安全冻结的模块

第一版必须 `requires_grad=False`：

- `model.vae`；
- `model.video_expert`；
- `model.action_expert`；
- `model.mot`；
- `model.proprio_encoder`；
- text encoder/tokenizer（评测模型本来不加载 text encoder，使用缓存 context）。

必须保持 eval mode：

- VAE；
- Video DiT；
- Action DiT/MoT 中没有需要更新的 dropout/statistics，但仍统一保持 eval。

只允许 train mode/grad：

- future projector；
- future cross-attention；
- zero-init gate；
- 可选 LoRA（默认关闭，必须作为独立实验族）。

训练前后要对所有 frozen tensor 做分层 hash；每次 backward 检查：

- frozen param `requires_grad=false`；
- frozen param `.grad is None`；
- Adapter grad finite；
- optimizer param groups 只含 allowlist；
- checkpoint 前后 frozen hash 不变。

## 13. 信息泄漏判定

允许的 future 输入：

```text
current RGB observation
+ current proprio
+ language/context
+ deterministic sampled noise
+ frozen checkpoint/config/scheduler
```

禁止的 Adapter/sampler 输入：

- 后续 RGB observation；
- 真实 future video 的 VAE latent；
- 环境执行后的画面；
- Thought1/Thought2 正式测试 trajectory；
- LIBERO-Plus OOD 测试数据；
- success、termination 或 failure label；
- 与当前时刻之后状态有关的字段。

真实 future action 只可作为 action flow target；真实 future video 即使理论上只用于
另一个 loss，也不进入第一版 Thought3 training API。

自动泄漏测试至少包括：

1. schema 拒绝 `future_frames`、`next_observation`、`gt_future_latent` 等字段；
2. 对同一 current observation 固定 noise，任意改写数据集后续 RGB，cache hash 不变；
3. cache provenance 声明 `source_kind=model_sampled_from_current`；
4. online evaluator 禁止打开 `outputs/thought3/cache/`；
5. train/dev/final episode 集合交集必须为空；
6. cache sample 的 split、checkpoint、K、seed 与训练配置精确相符；
7. monkeypatch future observation 为随机值，Adapter action 输出保持不变。

## 14. 尚不能确定的问题

以下内容必须通过后续 gate 实测，Phase A 不伪造答案：

1. 真实 LIBERO-fastwam 数据版本、episode/frame 总量和磁盘规模；
2. video-only sampler 与上游 joint video path 的实际最大误差；
3. K=1/2/4 生成延迟和 cache 吞吐；
4. microbatch 1 的真实训练峰值显存；
5. zero gate 是否在 100–500 step smoke 内能稳定打开；
6. input-level injection 是否足够，还是需要开发集上的单个中层 block；
7. Adapter-only 是否足够，是否值得建立独立 LoRA 实验族；
8. future latent 是否提供任何 OOD 增益。

第 8 项允许答案为零或负；审计不预设 future 有效。

## 15. Phase A 判定

架构在工程上可行，且可以在不修改上游、Thought1/Thought2 命令和正式输出的前提下
隔离实现。进入 Phase B 前需要确认：

1. K 使用“每个 K 完整走到 sigma=0”的 schedule；
2. 第一版注入在 `action_encoder` 后；
3. Adapter 维度为 future 256、attention 512、8 heads；
4. A0 使用同形状全零 latent，通过完整 projector/attention；
5. A-shuffle 是同一已训练 A-K checkpoint 的推理时干预，主比较先以 K=4 为准；
6. train/dev 按 suite×task 分层、episode 级 90/10 划分；
7. LoRA 第一版关闭；
8. 当前帧 VAE latent 是否作为可选共享 cache（默认先不缓存）。

详细方案见 `docs/thought3_design.md`，风险与停止条件见
`docs/thought3_risk_register.md`。

## 16. Phase A 完成校验

2026-07-27 文档落盘后：

- `git diff --check`：通过；
- 完整旧测试：`175 passed`；
- 4 条 warning 均为受限测试进程不能初始化 NVML，不是测试失败；
- 第 2.4 节 Thought1/Thought2 八个冻结文件 SHA-256 全部与审计前一致；
- 未加载大模型、未生成 future cache、未训练、未写入 `outputs/thought1/`、
  `outputs/thought2/` 或 `third_party/`；
- 当前变更仅为三份新的 Phase A 文档。
