# Thought3 Phase 1：K=1 在线动作反事实协议与运行手册

状态：`COMPLETED / VALID ENGINEERING SMOKE / BRANCH A`

配置：`configs/thought3/online/phase1_k1_action_counterfactual.yaml`

输出：`outputs/thought3/phase1_k1_online_counterfactual_v1/`

结果：
[thought3_phase1_k1_online_counterfactual_report.md](thought3_phase1_k1_online_counterfactual_report.md)

> 2026-07-30 已按下述唯一命令完成真实单卡运行。B0 replay 与 B0-null parity
> 均逐位一致；correct-null、correct-shuffle 和 action-hash 三项均为 `8/8`，
> 冻结分类为 `future_content_sensitivity_observed`（分支 A）。这只支持技术
> action sensitivity，不支持 success/OOD/K 排序结论。

## 1. 研究问题

本实验直接回答一个技术因果问题：

> 在相同当前观测、language、proprio、Action DiT noise 和 20-step action
> denoising 下，在线生成的 K=1 future latent 的具体内容是否改变动作块？

它不是 rollout，不读取 success 或 OOD，也不回答 future 是否改善闭环成功率。

## 2. 冻结来源

主 checkpoint 唯一固定为：

```text
outputs/thought3/phase_e6_fresh_cohort_replication_v1/
  tracks/a1/checkpoints/step_00000200
```

| 项目 | 冻结值 |
| --- | --- |
| Adapter file SHA | `aa55622c03aafea05c1bfedcb8548df398b0912dcecba397741c190c6b01b78f` |
| Adapter semantic SHA | `19f62cf45ba36c72da8dbfd752165cc5ef5678d4212b5ab7bf07635fdc7825d9` |
| Adapter fingerprint | `7c636482574a42165eb752b18a637b81668282d21c48f6533e47f0b1884ab2fd` |
| Fast-WAM checkpoint SHA | `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579` |
| Fast-WAM source commit | `45d8e1458921d83f8ad6cf9ce993d371208dabd0` |
| Frozen Fast-WAM parameter SHA | `ac0dd59de78495dc8ed5c5601e24951b78f3fd6ad40843d76b83106eccceb4f8` |
| Config fingerprint | `e343fc73a7cfb6bbdd85a146d32ee79ea30246b839c8be8d2d91f74d739ee544` |
| Cohort fingerprint | `89ce6b84358c6891b1566ef6051201558a59aaeb3e4c28981bc9847dc9af6f72` |
| Shuffle mapping SHA | `55782357b348ef62620efe73939a0e9f3638d56e080f8ceba8a73257aea7f874` |
| 单卡 peak allocated hard bound | `<23.8 GiB` |

这是 post-training engineering checkpoint，不是论文正式模型。禁止同时尝试多个
checkpoint 后选择动作差异最大的一个。

## 3. Cohort 与数据边界

使用 E6 已消耗的 8 条 `libero_goal/task_0` train demonstration：

```text
episode 14, 10, 11, 30, 19, 38, 0, 12；均为 frame 0
```

runner 在 HuggingFace Dataset 上先执行 `select_columns`，只保留：

```text
episode_index, frame_index, task_index, timestamp, observation.state
```

双相机当前 RGB 通过当前 timestamp 解码。没有选择 action 列；没有读取真实
future RGB、Phase D training future cache、reserve 17–28、development、OOD、
rollout 或 success。

## 4. 四条件数据流

```text
同一 target：current RGB + language + proprio + action seed
                    │
          ┌─────────┼───────────┬─────────────────────┐
          │         │           │                     │
          ▼         ▼           ▼                     ▼
 B0 infer_action  null       correct K=1          shuffle K=1
 原始 Fast-WAM   无 tensor    target Video DiT     donor Video DiT
          │       无 Video DiT  native latent       native latent
          │         │           │                     │
          │         └──────┬────┴──────────┬──────────┘
          │                ▼               │
          │       同一 target current cache│
          │       同一 20-step Action DiT  │
          │       同一 initial action noise│
          └────────────────┴───────────────┘
                           ▼
                    四个 action chunk
```

### B0

调用未修改的上游 `FastWAM.infer_action()`。每条样本重复两次，用于 replay
determinism、replay floor 与 null parity。runner 在 B0 replay 全部通过后才
注册 future injection hook，避免基线计时包含任何 Adapter hook 开销。

### correct

target 当前 latent 在线经过 frozen Video DiT 的严格 K=1 scheduler update。
保留 native `[1,48,2,14,28]` future latent，不 decode RGB，不写 training cache。
Adapter 在 Action DiT 的 20 次 `action_encoder` 调用中各注入一次。

### null

这是 request-scoped、parameter-free 的正式 null mask/bypass：

- 不构造零 tensor；
- 不运行 Video DiT；
- injection boundary 原样返回 action hidden；
- Adapter forward 次数必须为 0；
- hook 次数仍必须精确为 20。

null 与 B0 的最大绝对差若超过 `1e-5`，实验 fail closed，不生成 sensitivity
分类。

### shuffle

使用确定性一一 derangement；donor 必须来自其他 episode。只用 donor
current/context 在线生成 future，随后将该 latent 注入 target 的 Action DiT。
target RGB、language、proprio、current cache、action seed 和 initial action
noise均不变。correct/shuffle 复用 recipient future-noise seed。

## 5. Replay floor 与决策规则

先完成全部 8 条 B0×2 replay，再看任何 intervention：

```text
hard replay pass:
    max B0-repeat L∞ <= 1e-5

material L2 floor:
    max(1e-7, 10 × B0-repeat L2 p95)
```

若 hard replay 失败，runner 只写失败工件，不输出 A/B/C 分类。

| 分支 | 冻结规则 | 后续 |
| --- | --- | --- |
| A `future_content_sensitivity_observed` | correct-null 与 correct-shuffle 各至少 6/8 超过 replay floor，且 correct/shuffle action hash 至少 6/8 改变 | 允许进入 Phase 2；runner 不自动启动 |
| B `latent_presence_sensitivity_only` | correct-null 至少 6/8 超过 floor，但未满足内容敏感性 | 最多一次单变量结构修复，再用相同 cohort 重复一次 |
| C `no_material_online_action_sensitivity` | correct-null 未达到 6/8 | 停止 Adapter-only full training、A2/A4 和 OOD rollout |

不得根据结果降低 `6/8`、hash 或 replay 阈值。

## 6. 输出指标

每条 sample 保存：

- B0/correct/null/shuffle action tensor、semantic SHA 与 safetensors file SHA；
- correct/null/shuffle native future metadata；correct/shuffle latent safetensors；
- action seed、future seed、donor-target mapping；
- target/donor current RGB、proprio、context/mask 与 current latent 的 tensor SHA，
  用于证明 shuffle 没有替换 target 输入；
- 四组 pair 的 L1、L2、L∞、action cosine；
- translation、rotation、gripper difference；
- 32 个 timestep 的 action L2；
- EEF trajectory status；无可靠 FK 时明确写 `unavailable`；
- correct-null 与 correct-shuffle delta direction cosine；
- finite/NaN/Inf、source access telemetry。

聚合保存 mean、median、p50、p95、action hash 改变数、超过 replay floor 的样本
数及原始 sample JSONL。

CUDA synchronize 分别计时：

- preprocessing；
- context construction；
- current encoding；
- K=1 Video DiT；
- action current-cache；
- Adapter；
- Action DiT；
- condition total 与 policy total；
- peak allocated / reserved。

warmup 单独落盘，不进入正式计时。B0 公共 API 不暴露内部拆分，因此其
`action_dit_ms` 是含 VAE encode/current cache 的 inclusive time；null 提供与
B0 parity 的可拆分 current-only timing。shuffle 的 `policy_total_ms` 额外包含
donor preprocessing/context/current encoding，避免把控制 latent 的构造成本
藏在预处理阶段；`future_video_dit_ms` 仍只记录真实 K=1 sampler。

## 7. Fail-closed 与恢复

正式运行要求：

- 项目与 Fast-WAM worktree clean；
- Fast-WAM 与 E6 checkpoint/config/file SHA 全部匹配；
- 只暴露一个逻辑 GPU；
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`；
- deterministic algorithms、TF32/flash/mem-efficient SDP 关闭；
- frozen Fast-WAM 与 Adapter semantic SHA 前后不变；
- 0 backward、0 optimizer、0 parameter gradients；
- config override、device override 和多 rank 全部拒绝。

JSON/JSONL 与 tensor 文件原子写入。`--resume` 只接受相同 protocol lock、按
cohort 顺序的前缀记录和 checksum-valid tensor artifacts。已完成运行的
`--resume` 只做验证，不重载模型。

## 8. Dry-run

已执行：

```bash
PYTHONPATH=src python -m fastwam_ood_eval.cli \
  thought3-k1-online-counterfactual \
  --config configs/thought3/online/phase1_k1_action_counterfactual.yaml \
  --dry-run
```

结果确认：8 samples、四条件、固定 mapping SHA；不导入 Torch/Safetensors，
不加载 checkpoint/Fast-WAM，不写工件，不启动 rollout。

## 9. 已执行的唯一真实运行命令

2026-07-30 在 GPU 1 只运行了一次：

```bash
CONFIRM_THOUGHT3_K1_ONLINE_CF=YES \
THOUGHT3_GPU_ID=1 \
bash scripts/run_thought3_k1_online_counterfactual.sh
```

脚本拒绝 `THOUGHT3_GPU_ID=1,2`；该实验是严格单卡，因为四条件需要共享同一个
live model、相同确定性状态和逐样本配对。预计热缓存约 12–25 分钟；首次冷
加载或 HF cache 重建可预留 20–35 分钟。模型加载与前后参数 SHA 通常比 8 条
K=1 update 更耗时。

本次实测 wall time 为 `14.96 min`，其中模型加载 `552.82 s`。最终
`run_status.status=completed`、`phase2_started=false`；输出 manifest 的 62 个
文件已逐项重算 SHA 通过。

运行后查看：

```bash
cat outputs/thought3/phase1_k1_online_counterfactual_v1/run_status.json
cat outputs/thought3/phase1_k1_online_counterfactual_v1/decision.json
cat outputs/thought3/phase1_k1_online_counterfactual_v1/phase1_decision_report.md
```

本命令不会启动 Phase 2。
