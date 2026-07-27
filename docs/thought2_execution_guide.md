# 阶段二执行手册：未来—动作—实际变化一致性

更新日期：2026-07-27

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

若只暴露物理 GPU 1，robosuite 的 EGL 检查要求
`CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1`，但 torch 参数仍写
`--device cuda:0`，因为 torch 会把唯一可见卡重映射为逻辑 0。

### 3.4 聚合

```bash
fastwam-ood aggregate-diagnostics \
  --experiment-dir outputs/thought2_unconditional_smoke
fastwam-ood report-diagnostics \
  --experiment-dir outputs/thought2_unconditional_smoke
```

正式 Clean/OOD 分目录时，另建 comparison output，并将二者作为 `--input-dir`；只有 protocol signature 相同才允许合并。

当前 pilot 的独立 comparison 命令：

```bash
fastwam-ood aggregate-diagnostics \
  --experiment-dir outputs/thought2_unconditional_pilot_comparison \
  --input-dir outputs/thought2_unconditional_clean \
  --input-dir outputs/thought2_unconditional_ood
fastwam-ood report-diagnostics \
  --experiment-dir outputs/thought2_unconditional_pilot_comparison
```

多输入聚合会生成独立 comparison manifest，记录 mode、共同 provenance、输入
fingerprint 和 source manifest hash；禁止把 comparison summary 写回任一输入目录。

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

独立 `calibrate-static` 子系统已经实现并完成 7 条真实 PILOT：

- 不调用 `policy.act()`，只执行标准
  `[0,0,0,0,0,0,-1]`；
- 复用标准 30-step settle，并保存 offset `0/4/8` 的官方双相机帧；
- 同一帧重复编码 3 次测编码噪声；
- Clean 2 条 + 五类 OOD 各 1 条，7/7 eligible、0 error；
- 同帧噪声全部为 0，8-step no-op energy 中位数/最大值为
  `0.00661479/0.01322303`。

2026-07-23 的 7-job PILOT 只得到
`candidate_static_motion_threshold=0.01322303`，资格仍是
`candidate_only`。2026-07-26 正式 runner 随后完成独立的 200-job null set：
Clean/OOD 各 100、五类 OOD 各 20、200/200 eligible、0 error；在任何
diagnostic future 指标生成前，全部 freeze gate 通过并由 runner 显式接受：

```text
formal_static_motion_threshold = 0.016742116587908088
```

PILOT 候选值仍只作为历史记录，不能与正式阈值混用或回写旧 diagnostics。

只读敏感性分析把旧阈值下 predicted/actual static 的 `7/7` / `7/7` 重分类为
候选阈值下 `0/7` / `0/7`，原 diagnostics JSONL 未改写。它证明旧阈值量纲
明显不合理，但不能把候选二值标签写成模型能力结论。

完整协议、命令、逐类别数值、freeze gate 与恢复规则见
[静态/无动作校准手册](thought2_static_calibration.md)。

## 7. 人工盲审

现已实现公开 packet 与私有 unblinding key 的物理分离。第一轮 packet 只暴露
不透明 case ID、任务文本和 current/predicted/actual/comparison 媒体，不包含
condition、outcome、metric、动作、seed 或 source identifier。命令：

```bash
fastwam-ood prepare-blind-review \
  --packet-dir <fresh-public-dir> \
  --key-dir <fresh-private-dir> \
  --input-dir <clean-diagnostic-dir> \
  --input-dir <ood-diagnostic-dir> \
  --seed <pre-registered-seed> \
  --max-cases-per-job 1
```

第一轮模板是
[templates/future_blind_annotations.csv](templates/future_blind_annotations.csv)，
有意不含 `primary_failure_hypothesis`。第二轮解盲后才使用
[templates/future_case_annotations.csv](templates/future_case_annotations.csv)
填写失败假设。

至少对主表预注册子集做双人独立标注，先报告一致率或 Cohen's kappa，再讨论产生
adjudicated 标签；原始两份标签不可覆盖。这里是 label-blind 而非 perceptually
blind：reviewer 仍可能从视频推断扰动或结果。

解盲前用 `analyze-blind-review` 校验完整 case set、reviewer identity 与枚举，
并分别报告 nonmissing 和排除 `uncertain` 后的 decisive 分母。边际分布退化时
κ 保持 `undefined`；程序不会把表面 100% agreement 伪装成 κ=1。

当前 7-probe PILOT packet 已生成并通过 28 个媒体的完整解码与 public/private
hash 审计，但尚无人为标注。详细命令、字段、packet ID 和保管规则见
[盲审与 outcome-blind 抽样手册](thought2_blind_review_and_sampling.md)。

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

阶段二不必对阶段一全部 7,571 条 episode 生成视频。使用两个资格完全不同的
cohort：

1. **Outcome-blind cohort**：在看 success 前，按 suite、扰动、difficulty 分层随机抽样，用于估计 ID/OOD 一致性差异。
2. **Matched case-control cohort**：阶段一完成后，按 task/seed/扰动匹配成功与失败案例，用于失败机制分析；该 cohort 不能用于估计总体失败率。

一对多 OOD variant 不应假装成独立 Clean 配对。主分析应在 task/seed 层聚类，或为每个 category/level 预先固定一个 OOD variant 做一对一配对。

outcome-blind v2 草案使用 seed `20260724`：每个 suite/task 抽 5 个
Clean，并强制包含 episode index 0；每个可运行
suite/task/category/difficulty cell 抽 1 个 OOD。共 200 Clean + 532 OOD =
732，另有 68 个 skipped-only cell，supported shortfall 为 0。八份 manifest
最初均是 `draft_not_frozen`，不能直接通过正式 runner 的安全门。

由于这些草案在阶段一 outcome JSONL 产生前已经固定 exact job ID，现在采用
`ratified_before_diagnostic_outcomes`：不重新抽样，只在 clean commit 上把原
job ID 锁定到新的 manifest，同时明确披露阶段一 outcome 已经存在。这不是阶段一
outcome 前的 preregistration，但可以防止根据尚未观察的阶段二 future 指标改样本。

2026-07-26 正式 runner 选择五类版本，在任何 Phase 2 future metric 产生前
exact-ratify 八份 manifest，并完成全部 732 条。若后续论文另做四类分析，
612/622 只能作为明确标注的 sensitivity 或新实验，不能事后冒充本次 formal
cohort。
完整计数、cohort ID、freeze 命令和废弃 v1 记录见
[盲审与 outcome-blind 抽样手册](thought2_blind_review_and_sampling.md)。
Episode 内先聚合 probe、task 内再聚合 cell、suite-stratified task bootstrap
及 outcome 最小分母见
[统计分析计划](thought2_statistical_analysis_plan.md)。该计划在正式
collection 前仍是 DRAFT；本次 v1 分析与其方法一致，但只能称
protocol-consistent post-run analysis，不能称 preregistered confirmatory。

### 9.1 五类后台 runner

完整、可续跑的入口已于 2026-07-26—27 成功执行：

```bash
CONFIRM_PHASE2_FIVE_CATEGORY=YES \
ACCEPT_STATIC_THRESHOLD=YES \
GPU_IDS=0,1,2 \
bash scripts/run_thought2_five_category_full.sh --background all
```

它依次执行：

1. 精确 ratify 八份 cohort，验证总计 732；
2. 100 Clean + 100 OOD static/no-op calibration；
3. 自动聚合，且仅在全部 freeze-eligibility gate 通过时读取阈值；
4. 依次运行四个 suite 的 Clean 与 OOD shadow diagnostics；
5. 生成八份分组报告和一份四 suite combined 报告。

默认输出是 `outputs/thought2/five_category_formal_v1/`。脚本要求项目和三个
上游 clean、GPU 空闲显存至少 24,000 MiB，并使用 incomplete-only resume。
纯 workload 约 17.7 GPU-hours；3 卡理想 5.9 h，包含 10 个进程组冷启动、I/O
和尾部不均衡后建议预留 8–12 h。4 卡可通过 `GPU_IDS=0,1,2,3` 启动，但应先
确认第四张卡和 EGL 均正常。

本次实际状态：

```text
status=completed
exit_code=0
started_at=2026-07-26 18:18:15
finished_at=2026-07-27 00:41:46
```

完成后使用只读分析命令；它要求全新 output，不修改 raw JSONL/CSV/video：

```bash
source scripts/activate_env.sh
fastwam-ood analyze-thought2-formal \
  --experiment-dir outputs/thought2/five_category_formal_v1 \
  --thought1-summary outputs/thought1/fastwam/combined/summary/episode_results.csv \
  --source-trace-root outputs/thought1/fastwam \
  --output-dir outputs/thought2/five_category_formal_v1/formal_analysis_v1 \
  --bootstrap-replicates 10000 \
  --bootstrap-seed 20260725 \
  --verify-media
```

人工解释见 [thought2_formal_results.md](thought2_formal_results.md)，机器表位于
`outputs/thought2/five_category_formal_v1/formal_analysis_v1/`。

## 10. 正式完成门槛

- **正式 static 已完成**：200/200 eligible，Clean/OOD 各 100、五类各 20；
  阈值 `0.0167421166` 在 diagnostic 启动前通过全部 freeze gate。
- **正式 cohort 已完成**：五类 200 Clean + 532 OOD exact-ratify，732/732
  episodes、1,010 probes、2,020 aligned future frames，0 error。
- **已实现**：正式配置可设 `require_frozen_cohort=true`，草案会在模型加载和
  environment reset 前被拒绝。
- **正式 provenance 已通过**：raw collection 使用 clean commit `0fb8350`，
  项目与三个上游 checkout 均 clean；checkpoint 与阶段一相同。
- **已实现**：每个条件报告 episode/probe/aligned-frame 分母。
- **正式统计已运行**：先 probe→episode→task，40 task 等权，进行 10,000 次
  suite-stratified task bootstrap；另报告 first-probe-only sensitivity。
- **已通过动作/媒体审计**：1,010/1,010 probe 的同次运行 action hash 不变；
  4,040 个媒体全量解码 0 error。
- **待完成**：人工标注完成并保留原始/裁决版本。
- **已实现**：报告明确区分 2A 与 2B，`causal_interpretation_allowed=false`。
- **资格边界**：raw collection 为正式数据；统计计划未预冻结，v1 推断必须标为
  post-run、not preregistered。

## 11. 五类正式结果

- Clean/OOD task-equal cosine distance 为 `0.1025/0.1341`，
  OOD−Clean `+0.0316 [0.0254, 0.0381]`。
- Motion-direction cosine 为 `0.7416/0.5518`，
  OOD−Clean `−0.1898 [−0.2134, −0.1664]`。
- 排除两个 Phase 1/2 outcome mismatch 后，OOD 为 255 success/275 failure；
  failure−success cosine 为 `+0.0249 [0.0166, 0.0328]`，仅首 probe
  sensitivity 为 `+0.0197 [0.0116, 0.0282]`。
- 首 probe 最低/最高 cosine-error 四分位 failure rate 分别为
  `41.67%/65.41%`。一致性与失败相关，但不是成功的充分或必要条件。
- 这些结果回答“自动 future–realized proxy 是否在 OOD/失败下降”；它们不回答
  语义 goal correctness，也不能把失败归因为 future error。

## 12. 历史 real smoke 与 20-step pilot

`P2A-CLEAN-SMOKE-v1` 已于 2026-07-23 完成：

- 1 job / 1 probe / 0 error。
- 当前帧、预测、实际和并排媒体齐全。
- 2 个 aligned future frames；20 Hz 与 offset 0/4/8 精确验证。
- 动作 hash 未改变。
- 2-step generation `1,223.53 ms`；完整 diagnostic `4,616.06 ms`。
- 峰值 `24,841.09 MB`。

详细机器结果位于 `outputs/thought2_unconditional_smoke/summary/`，证据解释见 [experiment_ledger.md](experiment_ledger.md)。

`P2A-CLEAN-PILOT-v1` 与 `P2A-OOD-CAMERA-PILOT-v1` 也已于
2026-07-23 完成：

- Clean 2 episodes / 2 probes / 4 aligned future frames / 0 error。
- OOD camera/easy 3 episodes / 5 probes / 10 aligned future frames / 0 error。
- 7/7 probe 的动作与阶段一 trace 逐元素一致，5/5 outcome 一致。
- 全部 21 个 MP4 和 7 个 current PNG 可解码，抽检无损坏。
- episode-weighted Clean/OOD generation latency 为
  `4,108.12/4,563.88 ms`，完整 diagnostic 为 `7,214.25/8,200.65 ms`。
- pilot 中 L1 为 `0.1512/0.2002`，cosine distance 为
  `0.1168/0.1942`，motion-direction cosine 为 `0.7697/0.5283`。
- 已将 7 个 probe 转为公开/私有分离的 blind-review packet：7 cases /
  28 media，public sensitive key 为 0，全部媒体可解码；**human annotation
  仍为 0/7**。

最后一组数值只登记为 camera/easy 小样本趋势；严格 ID/OOD pair 只有 1，
static 阈值只有 7 条 null candidate、尚未冻结，也未完成盲审，不能写成论文结论。联合报告位于
`outputs/thought2_unconditional_pilot_comparison/summary/`。
