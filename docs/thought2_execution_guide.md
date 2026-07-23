# 阶段二执行手册：未来—动作—实际变化一致性

更新日期：2026-07-23

## 1. 阶段二拆成 2A 与 2B

| 子阶段 | 输入到视频分支 | 当前 checkpoint 是否支持 | 能回答什么 |
| --- | --- | --- | --- |
| 2A `unconditional_future` | 当前图像、语言、proprio；**不输入 policy action** | 支持，真实 smoke 已通过 | 模型的任务条件未来先验，是否与独立动作造成的实际变化相容 |
| 2B `action_conditioned_future` | 当前图像、语言、proprio、将执行动作 | 不支持 | 给定动作后的动力学未来是否正确 |

2A 是原研究路线中“同一 checkpoint 离线生成/恢复未来”的可执行版本。它只测相关性：动作分支不读取生成视频，视频分支也不读取受保护动作。2B 保留为更强的可选诊断，不得用 2A 结果冒充。

当前实现是**独立 shadow rerun**，不是从阶段一视频文件做纯离线 replay：

1. 只读阶段一 manifest，复用同一 job、seed、checkpoint 和控制协议。
2. 在新的进程/输出目录重新执行 episode。
3. 先由原 `FastWAMAdapter.act()` 确定动作并复制、哈希。
4. Shadow future probe 在 RNG 隔离区生成未来。
5. 仿真器执行受保护的原动作；生成延迟不推进仿真时间。
6. 收集实际 observation，写入阶段二独立工件。

这不会改变阶段一历史结果，但阶段二的 success 是这次 deterministic rerun 的结果；正式分析前仍应抽检它与 source episode 的动作/结果可复现性。

## 2. 已实现的隔离门禁

- 普通 `plan/evaluate/distributed-evaluate` 拒绝 `diagnostics.enabled=true`。
- `diagnose-future` 输出与 source output 必须是互不包含的目录。
- source checkpoint hash、Fast-WAM commit、控制 horizon、相机尺寸、seed 和 policy identity 必须匹配。
- probe 前后动作 hash 必须一致，否则拒绝执行。
- Python、NumPy、Torch CPU/CUDA RNG 在 probe 后恢复。
- 2A 必须验证 `video_expert.action_conditioned=false`。
- 2B 必须验证 `action_conditioned=true`、动作依赖覆盖、checkpoint 参数真实加载及训练 provenance；当前 allowlist 为空。

## 3. 运行顺序

### 3.1 只读 dry-run

```bash
source scripts/activate_env.sh
fastwam-ood diagnose-future \
  --config configs/studies/thought2_unconditional_smoke.yaml \
  --device cuda:0 \
  --dry-run
```

预期：只打印 assigned/pending，不加载模型、环境，也不写 diagnostics。

### 3.2 真实 2-step smoke

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  fastwam-ood diagnose-future \
  --config configs/studies/thought2_unconditional_smoke.yaml \
  --device cuda:0
```

这个配置只有 1 job、1 probe、`max_steps=10`、2 个视频去噪步。它只验收模型、显存、媒体、动作隔离和时间对齐；它产生的 success/failure 与一致性数值都不是正式结果。

### 3.3 20-step Clean/OOD 小 pilot

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  fastwam-ood diagnose-future \
  --config configs/studies/thought2_unconditional_clean.yaml \
  --device cuda:0
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2 MUJOCO_GL=egl \
torchrun --standalone --nproc_per_node=3 \
  -m fastwam_ood_eval.cli distributed-diagnose-future \
  --config configs/studies/thought2_unconditional_ood.yaml
```

Clean 配置复用 `outputs/clean_smoke` 的 2 条 episode；OOD 配置复用 `outputs/ood_pilot` 中 task 0/4/9 的 camera/easy 各 1 条。二者仍是 pilot，不是论文样本。

### 3.4 聚合

```bash
fastwam-ood aggregate-diagnostics \
  --experiment-dir outputs/thought2_unconditional_smoke
fastwam-ood report-diagnostics \
  --experiment-dir outputs/thought2_unconditional_smoke
```

正式 Clean/OOD 分目录时，另建 comparison output，并将二者作为 `--input-dir`；只有 protocol signature 相同才允许合并。

## 4. 每个 probe 必须保存什么

| 内容 | 字段/工件 |
| --- | --- |
| 当前帧 | `current_frame_path`，双相机官方拼接 PNG |
| 预测未来 | `predicted_video_path`，完整 9 帧 |
| 实际未来 | `actual_video_path`，仅可精确对齐的帧 |
| 并排对照 | `side_by_side_video_path` |
| 完整预测动作 | `predicted_actions`；名称指原 policy 输出，不是从 future 反推动作 |
| 实际执行动作 | `executed_actions` |
| 动作隔离 | `action_hash_before/after`、`action_unchanged` |
| episode 结果 | `success`、`termination_reason` |
| provenance | checkpoint/upstream/source hash、protocol fingerprint、seed |
| 资源 | generation latency、完整 diagnostic latency、峰值显存 |

预测视频 9 帧对应动作 offset `0,4,8,...,32`。阶段一 `control_horizon=10` 时，每次重规划最多得到 0/4/8 三个实际对照帧，即两个 future frame。

## 5. 自动指标

当前指标都在同一冻结 VAE 的**逐帧重编码 embedding**中计算：

- `future_latent_l1`：排除当前帧后的平均 L1。
- `future_latent_cosine_distance`：排除当前帧后的整体 cosine distance。
- `predicted_motion_energy` / `actual_motion_energy`：末帧减首帧的平均绝对变化。
- `motion_energy_ratio`：预测变化量 / 实际变化量。
- `motion_direction_cosine`：预测与实际 representation delta 的方向 cosine。

这些不是 native diffusion latent likelihood、物理光流或 7-DoF action cosine。尤其不能把 `motion_direction_cosine` 写成“动作向量与视频方向余弦”；它只表示预测视觉变化与动作执行后视觉变化的相容程度。

## 6. 静止阈值校准

默认 `static_motion_threshold=1.0` 只是 schema 初值，当前 smoke 已证明它可能过高：有明显动作时预测/实际 energy 约 0.2，仍被标成 static。正式实验前必须：

1. 建立独立 calibration set，不进入后续成功/OOD统计。
2. 对相同帧重复编码，测编码数值噪声。
3. 在 Clean 和代表性 OOD 条件下执行 no-op，采集 offset 0/4/8 的实际视觉变化。
4. 以 no-op motion energy 的高分位数（预注册 95% 或 99%）设阈值，并做敏感性分析。
5. 固定阈值和 calibration manifest 后再运行正式诊断。

阈值改变必须使用新输出目录和 protocol fingerprint。

## 7. 人工盲审

模板：[templates/future_case_annotations.csv](templates/future_case_annotations.csv)

推荐两轮：

1. 第一轮隐藏 Clean/OOD、success/failure 和自动指标，只看任务文本、当前帧、预测/实际并排视频。
2. 第二轮解盲结果，填写失败假设和备注。

字段允许值：

- `video_validity`：`valid/corrupt/unaligned/uncertain`
- `future_goal_progress`：`correct/partial/wrong_object/wrong_direction/static/uncertain`
- `future_physical_plausibility`：`plausible/minor_artifact/unphysical/uncertain`
- `future_actual_agreement`：`aligned/partial/conflict/static/uncertain`
- `action_execution_quality`：`realized/stalled/collision/oscillation/uncertain`
- `primary_failure_hypothesis`：`not_failure/future_hypothesis/action_selection/action_execution/perception/environment/compound/ambiguous`
- `confidence`：`low/medium/high`

至少对主表样本的一个预注册子集做双人独立标注，报告一致率或 Cohen's kappa。分歧经讨论产生 adjudicated 标签，但原始两份标签必须保留。

## 8. “未来错误还是动作错误”的表述边界

自动指标只能产生候选案例，不能自动给出因果归因：

| 观察 | 允许的描述 |
| --- | --- |
| 预测未来与实际变化冲突，动作被正常执行 | future hypothesis 与 realized outcome 不一致 |
| 预测目标进展合理，但实际动作停滞/碰撞 | 更符合 action execution/selection 假设 |
| 预测和实际都朝错误目标变化 | 可能是共同 perception/task-understanding 错误 |
| 视频不可判或多因素同时出现 | 保持 `ambiguous/compound` |

即使 2A 的预测与成功强相关，也不能写“动作依赖未来”或“未来错误导致动作失败”。

## 9. 正式抽样设计

阶段二不必对阶段一全部 7,571 条 episode 生成视频。建议预注册两个 cohort：

1. **Outcome-blind cohort**：在看 success 前，按 suite、扰动、difficulty 分层随机抽样，用于估计 ID/OOD 一致性差异。
2. **Matched case-control cohort**：阶段一完成后，按 task/seed/扰动匹配成功与失败案例，用于失败机制分析；该 cohort 不能用于估计总体失败率。

一对多 OOD variant 不应假装成独立 Clean 配对。主分析应在 task/seed 层聚类，或为每个 category/level 预先固定一个 OOD variant 做一对一配对。

## 10. 正式完成门槛

- 20-step Clean/OOD pilot 无 error，媒体和时间对齐抽检通过。
- static threshold 使用独立 calibration set 固定。
- outcome-blind 抽样 manifest 在查看结果前冻结。
- 项目与三个上游 checkout 的 manifest dirty 状态全部为 `false`。
- 每个条件报告 episode/probe/aligned-frame 分母。
- 主统计先在 episode 内聚合 probe，再按 episode 等权；clip-weighted 仅作诊断。
- 人工标注完成并保留原始/裁决版本。
- 报告明确区分 2A 与 2B，`causal_interpretation_allowed=false`。

## 11. 当前 real smoke

`P2A-CLEAN-SMOKE-v1` 已于 2026-07-23 完成：

- 1 job / 1 probe / 0 error。
- 当前帧、预测、实际和并排媒体齐全。
- 2 个 aligned future frames；20 Hz 与 offset 0/4/8 精确验证。
- 动作 hash 未改变。
- 2-step generation `1,223.53 ms`；完整 diagnostic `4,616.06 ms`。
- 峰值 `24,841.09 MB`。

详细机器结果位于 `outputs/thought2_unconditional_smoke/summary/`，证据解释见 [experiment_ledger.md](experiment_ledger.md)。
