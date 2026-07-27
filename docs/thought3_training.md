# Thought3 训练与恢复手册

状态：Phase B CPU/mock 链路完成；真实 Fast-WAM backward 尚未运行
更新时间：2026-07-27

## 1. 当前能做什么

Phase B 已实现：

- 单次 action-encoder-output 注入；
- zero-gated Future-to-Action cross-attention；
- A0/A1/A2/A4 同构模型；
- action velocity MSE 的 CPU mock；
- Adapter-only safetensors checkpoint；
- optimizer、global step、sample cursor resume；
- train/development loss、grad norm、gate、attention norm、step time、NaN/Inf；
- 中断+恢复与不中断训练的最终 Adapter tensor hash 一致测试。

当前不能做：

- 加载官方 12 GB checkpoint 做训练；
- 生成真实 Video DiT cache；
- 声称 mock loss 下降代表机器人控制有效；
- 自动启动 3 GPU 长训练。

## 2. Adapter 结构

```text
future [B,48,2,14,28]
  → Conv3d 48→256, kernel/stride [1,2,2]
  → [B,256,2,7,14]
  → LayerNorm + factorized T/H/W position
  → [B,196,256] K/V

action_encoder output [B,32,1024]
  → LayerNorm + Q projection
  → 8-head cross-attention, attention dim 512
  → output projection
  → x + tanh(gate) × residual
```

默认参数量精确为 `1,371,137`：

| 模块 | 参数 |
| --- | ---: |
| Conv3d projector | 49,408 |
| future LayerNorm | 512 |
| factorized position | 5,888 |
| action query LayerNorm | 2,048 |
| Q/K/V/out projections | 1,313,280 |
| scalar gate | 1 |
| 合计 | **1,371,137** |

gate 初始为 0，因此 A0 wrapper 与同一 B0 backbone 在固定输入上的 action 输出
逐元素完全一致。Adapter 不原地修改 action hidden。

## 3. 冻结与 hook

- backbone 全部 `requires_grad=false` 并保持 eval mode；
- wrapper 只允许 `adapter.*` trainable；
- optimizer 只接收 Adapter parameters；
- future context 用 context manager 激活；
- 每次 action prediction 检查 `action_encoder` hook 恰好调用一次；
- 异常退出必须清理 context，防止 future 泄漏到下一 batch。

真实 Phase C 还要对 VAE、Video DiT、Action DiT、MoT 和 proprio encoder
做 forward/backward 前后 hash 与 `.grad is None` 审计。

## 4. Phase B 可复现流程

先生成一次共享 mock cache：

```bash
source scripts/activate_env.sh

fastwam-ood thought3-plan-cache \
  --config configs/thought3/cache_smoke.yaml

fastwam-ood thought3-build-cache \
  --config configs/thought3/cache_smoke.yaml

fastwam-ood thought3-validate-cache \
  --config configs/thought3/cache_smoke.yaml
```

然后分别运行小型 mock：

```bash
fastwam-ood thought3-train \
  --config configs/thought3/train_a0_smoke.yaml

fastwam-ood thought3-train \
  --config configs/thought3/train_a1_smoke.yaml

fastwam-ood thought3-train \
  --config configs/thought3/train_a2_smoke.yaml

fastwam-ood thought3-train \
  --config configs/thought3/train_a4_smoke.yaml
```

中断后使用相同 config：

```bash
fastwam-ood thought3-train \
  --config configs/thought3/train_a1_smoke.yaml \
  --resume
```

identity 不一致时没有 `force` 或 `strict=false` 逃生口，必须新建 run。

## 5. 训练输入

Adapter 输入只有模型生成的 `future_latent` 与 mask。action flow batch 包含：

- current RGB/proprio/context；
- noisy action；
- action target 和 pad mask；
- model-generated future latent；
- identity/provenance。

action target 用于 loss，不能进入 future sampler。真实 future observation 不存在于
Trainer API。

## 6. 变体纪律

| 变体 | active K | Adapter 输入 | 是否训练 |
| --- | ---: | --- | --- |
| B0 | 0 | 无 | 不训练 |
| A0 | 0 | 全零 `[48,2,14,28]` | 独立训练 |
| A1 | 1 | K1 cache | 独立训练 |
| A2 | 2 | K2 cache | 独立训练 |
| A4 | 4 | K4 cache | 独立训练 |
| A-shuffle | K | 推理替换 | 不另训 |

所有 A0/A1/A2/A4 使用同一 Adapter class、参数量和 initialization seed。
动作噪声 seed 的推导不含 K/variant。

## 7. Checkpoint 内容

每个 checkpoint 目录只含：

```text
adapter.safetensors
optimizer.pt
manifest.json
```

不复制 6B backbone。manifest 绑定：

- official checkpoint/stats SHA-256；
- Fast-WAM commit；
- Adapter/config/split/cache fingerprint；
- variant/K/train seed；
- global step、sample cursor、world size；
- trainable name/count allowlist；
- frozen parameter hash；
- Adapter semantic-state SHA-256；
- 每个 checkpoint 文件 SHA-256；
- `contains_backbone=false`。

加载时逐项精确匹配，tensor key/shape 采用 strict load。

## 8. 记录指标

每 step 的 `train_metrics.jsonl`：

- action flow loss；
- gradient norm；
- raw gate 与 `tanh(gate)`；
- attention residual norm；
- trainable parameter count；
- step time；
- peak memory（Phase B CPU 为 0；真实运行记录 CUDA）；
- NaN/Inf；
- sample cursor。

每 checkpoint 原子刷新 metrics；因此 crash 后可从最近 commit 恢复。

## 9. Phase C 单卡顺序

Phase C 不直接开始训练，按以下顺序：

1. 获取一条标准 LIBERO train sample，确认没有 OOD/test 来源；
2. 只加载一个官方 Fast-WAM；
3. 生成 K=1/2/4 native future，核对 shape/dtype/device/schedule/noise；
4. 与 upstream joint/video path 做 same-seed parity；
5. gate=0 比较 B0/A0 action hash；
6. 执行一次 action-only loss backward；
7. 检查只有 Adapter 有 finite gradient；
8. 对 frozen 参数做 pre/post hash；
9. 记录 CUDA allocated/reserved peak，硬上限 43 GiB；
10. 完成真实 future mutation invariance。

任一项失败就停在 Phase C，不进入 cache 或训练。

## 10. Phase E 真实 smoke

真实小训练先只做 A0/A1：

- 一个 suite、少量 task；
- 100–500 steps；
- microbatch 1；
- bf16；
- gradient accumulation；
- 一个 train seed；
- checkpoint selection 只看 development action loss；
- 不看正式 OOD success。

只有 loss/gate/gradient 可诊断、resume 一致、frozen 不变且单卡无 OOM，才扩到
A2/A4 和三卡 DDP。
