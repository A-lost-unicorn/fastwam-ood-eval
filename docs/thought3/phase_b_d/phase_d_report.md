# Thought3 Phase D：小规模真实 Future Cache 验收报告

状态：**Gate D 通过**
验收日期：2026-07-28
证据等级：`SMOKE`
Run ID：`P3-PHASE-D-v1`

## 1. 结论

Phase D 已在一张 NVIDIA GeForce RTX 4090 上完成一个标准 `libero_goal` task
的小规模真实 cache smoke。以下工程门禁全部通过：

- 42 个完整 demonstration episode 先按 episode identity 做确定性 90/10
  train/development split，再选择 32 个不同 episode 的当前观测；
- 冻结官方 Fast-WAM Video DiT，为同一批 32 个 base sample 生成 K=1/2/4，
  共 96 条真实 model-generated future latent；
- 12 个 safetensors shard 的 shape、dtype、文件/tensor/逐样本 checksum 均通过；
- 32/32 个 base sample 的 K=1/2/4 共享相同 seed 和 initial-state hash；
- 第二次 cache build 验证并跳过 12/12 已提交 shard，没有重新加载模型；
- 临时副本的单字节破坏被 checksum validator 拒绝，主 cache hash 保持不变；
- 数据源只解码 64 张当前相机帧，没有读取未来 RGB、action target 或 rollout
  outcome；
- 单卡模型加载与执行峰值显存均低于 43 GiB 硬上限；
- 没有创建 optimizer、没有 backward/optimizer step、没有启动 Adapter 训练。

Gate D 因此解锁的是 **100–500 step Adapter 工程 smoke**，不是论文效果结论。
本阶段不能证明 Adapter 能收敛、显式未来改善 OOD、K 越大越好，或离线 cache
latency 等于在线总推理 latency。

## 2. 冻结来源与运行身份

| 项目 | 冻结值 |
| --- | --- |
| 项目分支 | `feature/thought3-partial-future-adapter` |
| Phase C 收口 commit | `f37a66bd43399bff637e2d2ffb1b9fd4103bd942` |
| Phase D 实现/运行 commit | `02a010eb63897a97c911fb5f68e0bb209fe654ec` |
| Fast-WAM commit | `45d8e1458921d83f8ad6cf9ce993d371208dabd0` |
| checkpoint SHA-256 | `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579` |
| dataset stats SHA-256 | `30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638` |
| model config SHA-256 | `ab3c2ffde9933e7576c747fecce82bd7d28c9c6478c1b53fcac02b3012be416c` |
| LIBERO revision | `117413dc0ca99c7cd64036c4eaa4a316c537d692` |
| dataset archive SHA-256 | `a21ae10171535585fb43e6405d9efa09ff38ef34689e4176428ca005af3a39ea` |
| config fingerprint | `367296d012a9ba2f2579dbe6a6663a88e132a029b49146819e0e50be0bedb2fc` |
| split fingerprint | `ea5402955023ccd48d790d821a73f98549b31d1ace8af035a90ceae2ad3951eb` |
| cache fingerprint | `63a70e1af38f68bc894fc11d03c84f212e6c6328a5051256c9d045741156d9c5` |
| physical/logical GPU | GPU 1 / `cuda:0`，visible device count = 1 |

运行前重新校验了 Phase C 的 status/result/log 三个冻结 SHA。运行后重新校验
Thought1/Thought2 的八个只读 sentinel，均与 Phase B/C 前完全一致。

## 3. 数据、任务与 split

本次只使用标准 training demonstration：

| 项目 | 值 |
| --- | --- |
| suite | `libero_goal` |
| task index | `0` |
| task | `open the middle drawer of the cabinet` |
| 候选 episode | 42 |
| 每 episode 当前观测 | frame 0，一条 |
| split seed | 3407 |
| 完整 split | train 37 / development 5 |
| 选入 cache | 32 个不同 episode |
| cache 内 split | train 28 / development 4 |
| camera | `image`、`wrist_image` |

split 在 pilot 截断前作用于全部 42 个 episode。同一 episode 不会跨
train/development；K=1/2/4 也共享同一 base sample 与 split。该 32-sample cache
只用于 Phase D/E 工程 smoke，不能代表完整任务分布。

## 4. Cache tensor 与配对结果

32 个 base sample × 3 个 K = 96 条 cache entry。`shard_size=8`，所以每个 K
有 4 个 shard，共 12 个。

| 项目 | 实测 |
| --- | --- |
| 单 shard latent | `[8,48,2,14,28]`，BF16 |
| 单 shard mask | `[8,2,14,28]`，bool |
| 单样本 latent | `[48,2,14,28]` |
| safetensors shard | 12/12 |
| metadata rows | 96 |
| base samples | 32 |
| cache 总文件 | 41 |
| cache 总大小 | 7,687,316 bytes |
| whole-cache validation | `valid` |

独立逐 metadata/tensor 复核得到：

- 32/32 base sample 的三个 K 使用同一 `initial_noise_seed`；
- 32/32 base sample 的三个 K 使用同一 `initial_state_sha256`；
- 32/32 base sample 的 K1/K2/K4 `latent_sha256` 两两不同；
- 32/32 episode 只属于一个 split；
- 96/96 row 的 source access 都是当前两相机、0 future RGB、0 action target。

“输出 hash 不同”只说明不同 K 确实产生了不同 latent，不等价于 K 越大越准确。

## 5. Checksum、断点恢复与损坏检测

每个 shard 在写入后验证：

- safetensors 文件 SHA-256；
- metadata JSONL SHA-256；
- `future_latents` / `future_masks` tensor SHA-256；
- 每样本 latent SHA-256；
- sample/base identity、K、schedule、seed、initial-state hash；
- checkpoint/stats/cache fingerprint；
- `source_kind=model_sampled_from_current`；
- `uses_ground_truth_future=false`。

首次 build 提交 12/12 shard。紧接着的 no-op resume 结果为：

```text
built_shards=0
skipped_valid_shards=12
total_shards=12
model_loaded=false
resume_validation_only=true
```

主动损坏测试只复制 `k1/shard_000000.safetensors` 到 `/tmp/thought3/`，翻转最后
一个 byte。validator 以 `tensor file checksum mismatch` 拒绝该副本；正式 shard
操作前后 SHA-256 相同。没有删除或改写正式 cache。

## 6. 无真实未来泄漏

真实 source loader 每个 base sample 只向 PyAV 请求一个 timestamp 的两路相机帧，
再读取同一当前行的 proprio。构建器 API 不接收后续 RGB 或 action target。

| 审计项 | 实测 |
| --- | --- |
| current camera frames decoded | 64 |
| future RGB frames decoded | 0 |
| inventory requested future RGB | false |
| actual future read | false |
| action target read | false |
| cache uses ground-truth future | false |
| configured / actual decode backend | TorchCodec / PyAV |

当前环境中的 TorchCodec 无法链接 FFmpeg，因此本次明确固定 PyAV
`16.0.1`，torchvision 为 `0.22.1+cu128`。torchvision 的视频接口弃用 warning
不影响本次输出，但扩展正式 cache 前应继续冻结可用的解码后端与版本。

## 7. 吞吐、延迟与显存

| 指标 | 实测 |
| --- | ---: |
| Gate D wall time | 1,150.50 s |
| cache builder wall time | 1,137.42 s |
| 模型加载 | 888.44 s |
| cache generation loop | 39.70 s |
| 不含模型加载吞吐 | 0.806 base sample/s |
| 当前双相机 decode mean / min / max | 330.74 / 222.48 / 507.44 ms |
| 当前帧 VAE encode mean / min / max | 111.56 / 21.04 / 2,446.25 ms |
| 执行阶段峰值显存 | 12.677 GiB |
| 模型加载峰值显存 | 23.125 GiB |

| K | n | sampling mean | min | max |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 127.54 ms | 90.77 ms | 779.51 ms |
| 2 | 32 | 186.62 ms | 162.75 ms | 213.31 ms |
| 4 | 32 | 362.99 ms | 319.00 ms | 378.06 ms |

K=1 的 max 和 current encode 的 max 含首轮 warm-up。上表是离线、单卡、
video-only future sampling 遥测；不含在线 action 20-step denoising，也不是
ID/OOD rollout 的端到端 latency。论文中的效果—延迟曲线必须在 Phase G 使用同一
在线协议重新计时。

## 8. 冻结机器工件

| 工件 | SHA-256 |
| --- | --- |
| `phase_d_cache_smoke_v1/run_status.json` | `d302cd63d3fd18161775f92ac3aa9d18e84842ee97b3316fe0f427df2e819baa` |
| `phase_d_cache_smoke_v1/gate_d_result.json` | `a636d649491ad9df67a1ea2cb91d8e9bf708784a410ba7b8304248f33ed1882d` |
| `phase_d_cache_smoke_v1/logs/phase_d.log` | `97cdb718877a2c58a0a11352102d874b4c1b670b38ed090e90d60f91e5412d84` |
| `phase_d_cache_smoke_v1/phase_d_inventory_manifest.json` | `a53735d5ac62738284a22a5b6422beb7edb5290d04d92f4ea7057986a6c01b9a` |
| `cache/.../cache_plan_manifest.json` | `c4ab6c4e3b4c205f5366de034ee5f6c202a420d07a058bad1d0d3eb86731eac9` |
| `cache/.../cache_manifest.json` | `1a0d73b1e4e6a4b12ac367b50f3a49f04a81a4ed6f692d957129d3b9d2f75816` |
| `cache/.../real_cache_build_report.json` | `221f32039df792a3b4d64dbe35bcedf7d99741f70bb6381e374fac903027f8c5` |

权威目录为：

```text
outputs/thought3/phase_d_cache_smoke_v1/
outputs/thought3/cache/phase_d_libero_goal_task0_v1/
```

两个目录受 `.gitignore` 保护；上表 hash 是文档与机器工件之间的只读锚点。

## 9. 复核命令

新目录首次运行：

```bash
CONFIRM_THOUGHT3_PHASE_D=YES \
THOUGHT3_GPU_ID=1 \
bash scripts/run_thought3_phase_d_cache_smoke.sh
```

只有在首次 build 中断且尚未产生最终 `gate_d_result.json` 时，使用同一配置：

```bash
CONFIRM_THOUGHT3_PHASE_D=YES \
THOUGHT3_GPU_ID=1 \
bash scripts/run_thought3_phase_d_cache_smoke.sh --resume
```

已完成 cache 的只读独立校验：

```bash
fastwam-ood thought3-validate-cache \
  --config configs/thought3/phase_d_cache_smoke.yaml
```

不要删除 shard 后在原 Run ID 上补写；任何协议、数据、K、seed 或 checkpoint
变化都必须使用新的 output/cache 目录与 fingerprint。

## 10. 下一门 Gate E

Gate E 只做 100–500 step 的真实 Adapter 工程 smoke，不看 OOD success：

1. 先实现 identity join，从 28 个 train episode 读取当前观测、action target 和
   对应 K cache；4 个 development episode 只计算 development action loss；
2. 先跑 A0 与 A1，microbatch 1、单卡、固定 seed，保持 backbone 全冻结；
3. 第 1 个 optimizer step 只允许 zero-init gate 打开；
4. 从第 2 个有效 step 起，必须观察 projector/attention 等非 gate 参数出现
   finite、nonzero gradient，而不只是 gate 有梯度；
5. 记录 loss、gate、分模块 grad norm、显存、step time、sample cursor；
6. 中断恢复与不中断训练的 Adapter semantic hash 必须一致；
7. frozen backbone hash 前后不变，checkpoint 不包含 backbone；
8. Gate E 通过后再扩 A2/A4，不启动正式 ID/OOD rollout。

