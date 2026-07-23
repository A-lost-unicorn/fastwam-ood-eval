# 阶段三手册：轻量 Future-to-Action Adapter

状态：设计阶段，尚无训练或结果

## 1. 要回答的因果问题

阶段三才回答：

> 在 checkpoint、训练数据、Action DiT 去噪步数和评测 episode 都匹配时，给动作分支增加不同精度的 future latent，是否能提高 OOD 成功率？需要多少视频去噪步，延迟代价是多少？

阶段一提供冻结基线，阶段二提供一致性证据；二者都不能替代这里的训练对照。

## 2. 最小结构

```text
current observation ──→ Fast-WAM current/action representation ───────┐
                                                                    │
current observation ──→ frozen Video DiT, K updates ─→ future latent ├─→ gated adapter
                                                                    │       │
                                                                    └───────┘
                                                                            ↓
                                                                  frozen Action DiT
                                                                            ↓
                                                                       action chunk
```

建议第一版：

1. 丢弃固定的 frame-0 latent，只使用 future tokens。
2. 对 future latent 做 LayerNorm + 小型时空 pooling/projection。
3. 用低秩 cross-attention 或 FiLM/gated residual 注入 1–2 个 Action DiT block。
4. gate 零初始化，使 adapter 初始状态尽量接近原 Fast-WAM。
5. 冻结 Video DiT、VAE 和主体 Action DiT；只训练 adapter，必要时再增加少量 Action DiT LoRA。

第一版不要解码 RGB 视频；adapter 直接消费 latent，部署延迟只包含 K 次视频去噪和 adapter，不包含 VAE decode。

## 3. K 的精确定义

| 版本 | 视频分支 | 动作分支 |
| --- | --- | --- |
| `B0-base` | 不运行视频去噪 | 固定 N 步 |
| `A0-null` | 不运行视频去噪；adapter 输入训练过的 null token | 固定 N 步 |
| `A1` | 同一 scheduler 的 1 次视频更新 | 固定 N 步 |
| `A2` | 同一 scheduler 的 2 次视频更新 | 固定 N 步 |
| `A4` | 同一 scheduler 的 4 次视频更新 | 固定 N 步 |

原路线中的 K=0 还需要拆成两个控制：

- `B0-base` 控制“原 Fast-WAM”。
- `A0-null` 控制“新增参数、训练和注入位置本身”。

否则 `A1/A2/A4` 优于原模型时，无法区分收益来自未来信息还是额外参数/训练。

所有 K 使用同一初始视频噪声 seed、scheduler、sigma shift、latent shape 和 future horizon。K 表示完成的 scheduler update 数，不得用“换一个 timestep”含糊代替。

## 4. 离线 cache

训练时先冻结视频模型生成 cache：

```text
training sample
  └── checkpoint/config/scheduler/seed 固定
        ├── K=1 future latent
        ├── K=2 future latent
        └── K=4 future latent
```

每条 cache 至少保存：

- 稳定 `sample_id`、dataset/trajectory/frame index；
- observation、task text、proprio 的来源 hash；
- Fast-WAM checkpoint、Video DiT/VAE commit 与 config hash；
- K、总 scheduler 配方、实际 timestep/sigma 序列；
- initial noise seed、dtype、shape、normalization；
- latent checksum 和 cache schema version。

目录建议：

```text
outputs/thought3/cache/<cache_fingerprint>/k_1/
outputs/thought3/cache/<cache_fingerprint>/k_2/
outputs/thought3/cache/<cache_fingerprint>/k_4/
```

cache fingerprint 不同不得静默混用。训练只读 cache；重新生成时写新目录。

## 5. 数据泄漏控制

- 训练/验证按完整 trajectory 切分，不能把同一轨迹相邻帧分到两侧。
- 如果研究声称 task-level holdout，必须按 task 切分并证明 release 训练集边界；否则只称环境 OOD。
- LIBERO-Plus 正式 OOD episode 不得用于 adapter 训练、阈值选择或 early stopping。
- 超参数先在训练/验证集和小型 ID pilot 固定，再解锁正式 OOD。
- 阶段二人工失败标签可以用于提出假设，不能直接用于挑选正式测试样本后再报告总体增益。

## 6. 训练协议

第一版建议仅训练 adapter：

- loss：保持 Fast-WAM 原 action diffusion/noise-prediction 目标；
- optimizer、LR、batch、训练 step、augmentation 对 A0/A1/A2/A4 完全一致；
- 每个 K 至少 3 个训练 seed；
- checkpoint 选择只看预注册 validation metric；
- 保存 trainable parameter count、吞吐、峰值显存和总 GPU-hours；
- 记录 frozen 参数 hash，验证训练前后没有变化。

若 adapter-only 没有信号，再建立单独 LoRA 实验；不要把 LoRA 与 adapter-only 结果混成同一 K 曲线。

## 7. 分阶段计算门禁

### Gate 1：形状与梯度

- 一个 batch 前向/反向通过。
- 只有 adapter 参数有非零梯度。
- frozen Video/Action 参数 hash 不变。
- A0 在 gate=0 时数值接近 B0。

### Gate 2：cache 正确性

- 随机样本在线生成与 cache latent 在容差内一致。
- K=1/2/4 的 scheduler metadata 可重建。
- 跨 K 的 sample ID 完全配对，无缺失或重复。

### Gate 3：小型 overfit

- 先在极小训练集确认 loss 能下降。
- 不把 overfit 成功率作为研究结果。

### Gate 4：ID pilot

- A0 不应因工程错误显著劣于 B0。
- A1/A2/A4 action 输出 finite，控制频率和 horizon 不变。
- 记录实时 K-generation latency，而不是 cache 读取时间。

### Gate 5：OOD 正式评测

前四个 gate 全通过后才启动。

## 8. 评测矩阵

主表至少包含：

| Variant | K | Train seed | Trainable params | ID success | OOD success | Absolute drop | Action latency | Future latency | Total latency | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

一致性表补充：

- future latent L1/cosine；
- motion-direction cosine；
- 人工 future-goal-progress / future-actual-agreement；
- 指标与成功率的相关性。

所有版本使用完全相同的阶段一任务/seed/variant manifest。动作去噪 N 固定。延迟必须在线测量：

```text
total policy latency = current/action path + K-step future latent + adapter
```

训练 cache 节省的是训练成本，不得从部署延迟中扣除。

## 9. 统计

- 成功率比较使用同 episode 配对。
- 报告 paired success difference、cluster bootstrap 95% CI 和 exact McNemar。
- 多训练 seed 先分别报告，再给 seed-level 均值/方差。
- 按 suite、扰动、difficulty 分层，但主假设和多重比较处理需预注册。
- 任何不匹配 checkpoint、训练 recipe 或任务 manifest 的结果只能作相关性对照。

## 10. 关键消融

优先级从高到低：

1. `B0-base` vs `A0-null`：额外参数/训练效应。
2. `A0-null` vs `A1/A2/A4`：future 信息效应。
3. 同 K 下 shuffled-future latent：检查 adapter 是否真的使用样本对应未来。
4. 同 K 下 zero/noise latent：检查是否只利用幅值或 shortcut。
5. 注入层位置和 adapter 容量。
6. adapter-only vs adapter+LoRA。

`oracle actual-future latent` 可作为上界诊断，但它使用未来观测，不能作为可部署模型或主结果。

## 11. 预期主图

以 K 为横轴，同时画：

- ID/OOD success；
- Clean→OOD absolute drop；
- future/action consistency；
- future generation latency；
- total latency；
- peak memory。

论文真正需要寻找的是 Pareto 点，而不是单独最大成功率：

> 最少的 K 是否已经获得大部分 OOD 收益，同时保持接近 Fast-WAM 的实时性？

## 12. 停止条件

出现以下任一情况先停：

- frozen 参数发生变化；
- A0 与 B0 的差异无法解释；
- cache 与在线 latent 不一致；
- 不同 K 的样本、训练 step 或 action 去噪步数不匹配；
- 评测读取了 OOD 测试信息做选模；
- 显存或延迟使目标硬件不可部署；
- 只有单一训练 seed 出现增益且配对 CI 跨零。

## 13. 进入阶段三前必须具备

1. 阶段一正式 Clean/OOD 表。
2. 阶段二 20-step pilot 与 static 阈值校准。
3. 明确选用的 native future latent 层、形状和 K scheduler 语义。
4. 数据 split 与 cache schema 冻结。
5. B0/A0/A1/A2/A4 预注册配置。

