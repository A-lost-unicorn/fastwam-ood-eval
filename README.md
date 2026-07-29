# fastwam-ood-eval

## 1. 这个项目研究什么

这个项目第一阶段严格回答一个问题：**同一份 Fast-WAM 权重在标准 LIBERO 和环境发生分布外变化的 LIBERO-Plus 中，成功率会下降多少？** 同时为“测试时未来想象是否改善 unseen 泛化”保留可审计的跨策略配对统计，但不会把不匹配的 checkpoint 比较包装成因果结论。

标准评测的心智模型是：

```text
Fast-WAM policy → 标准 LIBERO observation → action chunk → LIBERO environment → success / failure
```

OOD 评测的心智模型是：

```text
Fast-WAM policy → LIBERO-Plus 扰动后的 observation/environment → action chunk → 扰动环境 → success / failure
```

Clean 和 OOD 使用同一个 checkpoint，并为同一个基础任务/episode 分配相同 seed。模型权重不变，唯一改变的是测试环境；这样观察到的差距才尽量可以归因于扰动。本阶段测量的是**鲁棒性**，不是重新训练后的“泛化提升”。

## 2. Fast-WAM、ID、OOD 分别是什么

- Fast-WAM 是被评测的策略。适配器复用官方 checkpoint loader、图像/本体状态预处理和 action 后处理，不复制模型内部实现。
- ID（in-distribution）表示测试条件接近训练或标准评测条件；这里用原版 LIBERO 作为 clean/ID 基线。
- OOD（out-of-distribution）表示相机、光照、背景、机器人初始姿态或物体布局发生变化；这里使用 LIBERO-Plus 官方预生成任务变体。

绝对成功率下降：

```text
clean_success_rate - ood_success_rate
```

相对性能下降：

```text
(clean_success_rate - ood_success_rate) / max(clean_success_rate, epsilon)
```

`P1-FORMAL-v1` 已完成 800 个 Clean 和 6,771 个 runnable OOD rollout：
Clean `97.25%`、OOD `47.70%`，绝对下降 `49.55` 个百分点，0 exception。
完整分层结果与结论边界见 [思考点一正式报告](docs/thought1_report.md)。

## 3. 当前阶段不做什么

阶段一、二仍冻结，不重训或改写其 Fast-WAM 基线，也不把仿真 OOD 结论外推成
真机结论。阶段三已在独立 namespace 完成 Phase A–D 与 E.1–E.8 的真实工程
诊断；Gate E.5 六条轨迹完整，但没有 A0/A1 共同 eligible LR。E.6 已在
未使用 train cohort 完成 A0/A1@3e-4 序贯复验：A1 信号复现，但 A0 仅 4/8
样本不变差，故总 Gate 有效失败。E.7 已完成只读 step-50/100/150/200
trajectory 诊断：primary flow 不支持预注册的实质晚期退化模式，且没有任何
step 同时通过 A0、A1 absolute 与 paired 三组门槛。E.8 已完成 A0-only
的 64 个全新 flow replication：工程 Gate 通过，step 200 pooled mean 下降
3.728%，但 full/两个 block 的 sample stability 均未全部通过；只有 1/3
预识别 target 被确认恶化，故冻结分类为 `mixed_or_inconclusive`。因此
E.9 matched sample-tail mitigation 已预注册但尚未运行；完整 Gate E、A2/A4
和在线成功率评测仍未启动。未运行的实验不会填入虚构结果。

当前 pinned Fast-WAM 代码与 release 权重带来四个结论边界：

- 跨环境：可评测。训练配置只列标准 LIBERO 数据，LIBERO-Plus 官方变体可作为 unseen environment shift。
- 跨物体、跨任务：可按 `libero_object` 和各 suite 报告性能，但 release 训练配置已经包含全部四个 suite，不能称为 unseen-object 或 unseen-task。
- 跨平台：当前不可评测。同一策略没有同时兼容 LIBERO 与 RoboTwin 的 observation/action/权重。
- 未来想象：release `libero_uncond` 的动作仅读取当前首帧 token。保存预测视频不等于启用未来想象；因果对照需要训练配方匹配的 `joint`/`idm` checkpoint。

完整可识别性审计见 [思考点 1 协议](docs/thought1_generalization.md)。
当前实现与实验完成度逐项清单见 [思考点 1 readiness audit](docs/thought1_readiness.md)。

### 文档导航

- [研究总控与阶段状态](docs/research_index.md)：论文主线、证据等级、阶段隔离和当前优先级。
- [实验、卡点与结论台账](docs/experiment_ledger.md)：已运行数字、失败尝试、可写与不可写结论。
- [思考点一正式报告](docs/thought1_report.md)：全量结果、完整性审计、suite/category/difficulty 分层与结论边界。
- [思考点 1 实施与验收手册](docs/thought1_execution_guide.md)：checkpoint/assets、单卡 smoke、三卡 pilot 与正式运行门禁。
- [思考点二上游审计](docs/thought2_upstream_audit.md)：`infer_joint()`、官方预处理、动作语义、VAE、时间对齐和 release 能力门禁。
- [思考点二概念说明](docs/thought2_concepts.md)：Shadow Diagnostics 的研究问题、旁观语义和因果边界。
- [思考点二执行手册](docs/thought2_execution_guide.md)：2A/2B、真实命令、指标、阈值校准与人工盲审。
- [思考点二五类正式结果](docs/thought2_formal_results.md)：732 episodes 的 ID/OOD、outcome association、资源与结论边界。
- [思考点二盲审与抽样](docs/thought2_blind_review_and_sampling.md)：public/private 盲审包、outcome-blind cohort、anchor 与 pre-outcome freeze。
- [思考点二统计分析计划](docs/thought2_statistical_analysis_plan.md)：episode/task 层级、primary estimand、cluster bootstrap、缺失与停止规则。
- [思考点二 static/no-op 校准](docs/thought2_static_calibration.md)：独立 null set、候选阈值、freeze gate、真实数据与恢复规则。
- [思考点三 Phase B 验收](docs/thought3_phase_b_report.md)：1.37M Adapter、cache、mock trainer、checkpoint、测试与 Phase C 门禁。
- [思考点三 Phase C/D 验收](docs/thought3_phase_c_report.md)、[Phase D cache 验收](docs/thought3_phase_d_report.md)：真实单样本 backward 与 32-sample K1/K2/K4 cache。
- [思考点三 Gate E.5 结果](docs/thought3_phase_e5_report.md)：full-cohort 六轨迹结果、有效失败原因、工件哈希和下一复验边界。
- [思考点三 Gate E.6 预注册](docs/thought3_phase_e6_protocol.md)：未使用 cohort、全新 flow slots、post-selection 披露和冻结门槛。
- [思考点三 Gate E.6 结果](docs/thought3_phase_e6_report.md)：两轨迹完整结果、唯一失败门槛、逐样本对照和工件哈希。
- [思考点三 Gate E.7 协议与结果](docs/thought3_phase_e7_protocol.md)、[结果报告](docs/thought3_phase_e7_report.md)：只读 checkpoint trajectory、primary/continuity 分离、冻结分类与无候选结论。
- [思考点三 Gate E.8 协议与结果](docs/thought3_phase_e8_protocol.md)、[结果报告](docs/thought3_phase_e8_report.md)：A0 step 100/200、64 个全新 flow、双 32-flow block、配对 bootstrap，以及 mixed 诊断结论。
- [思考点三 Gate E.9 预注册](docs/thought3_phase_e9_protocol.md)：raw/normalized × A0/A1 单变量配方、固定权重、FWER tail Gate，以及 train 排序 17–28 的一次性复验边界。
- [思考点三设计与数据协议](docs/thought3_design.md)、[训练手册](docs/thought3_training.md)、[在线评测手册](docs/thought3_evaluation.md)。
- [思考点三最终目标完成度](docs/thought3_completion_audit.md)：已证明、未证明与通往正式 OOD 对照的依赖链。
- [工程亮点、难点与阻碍台账](docs/engineering_highlights.md)：工程复盘、未解决风险和简历素材。
- [环境配置](docs/environment_setup.md)、[实验协议](docs/experiment_protocol.md)、[上游勘察](docs/upstream_notes.md)。

## 4. 项目架构图

```text
官方 Fast-WAM checkpoint
            │
            ▼
    FastWAMAdapter
            │
      ┌─────┴─────┐
      ▼           ▼
Clean LIBERO   LIBERO-Plus OOD
      │           │
      └─────┬─────┘
            ▼
       Episode Results
            │
            ▼
       Aggregate & Report
```

原版 LIBERO 与 LIBERO-Plus 都导出名为 `libero` 的 Python 包，不能同时 import。本项目在一个兼容环境中按**进程**选择一个 backend：Clean 命令加载 `third_party/LIBERO`，OOD 命令加载 `third_party/LIBERO-plus`。模型无需通过网络 server/client 通信。

正式并行是 episode-level data parallel：每张 GPU 一个独立进程，`job_id % world_size` 稳定分片；不是模型 DDP。

## 5. 安装

先获取代码并查看已锁定的提交：

```bash
bash scripts/fetch_upstreams.sh
git -C third_party/FastWAM rev-parse HEAD
git -C third_party/LIBERO rev-parse HEAD
git -C third_party/LIBERO-plus rev-parse HEAD
```

当前 checkout 使用项目本地 Conda 环境时，只需：

```bash
source scripts/activate_env.sh
```

从零创建普通的命名 Conda 环境时使用：

```bash
bash scripts/create_env.sh fastwam-ood
conda activate fastwam-ood
```

LIBERO-Plus 还需要单独下载 assets；无显示服务器需要 `MUJOCO_GL=egl`。精确步骤、兼容性理由和 assets 目录见 [环境文档](docs/environment_setup.md)。不要同时 `pip install -e third_party/LIBERO` 和 `pip install -e third_party/LIBERO-plus`。

## 6. 下载 checkpoint

```bash
bash scripts/download_checkpoints.sh
```

checkpoint、配套 stats 与 Plus assets 的分阶段准备和验收命令见 [实施手册](docs/thought1_execution_guide.md)。

默认配置期望：

```text
checkpoints/fastwam_release/libero_uncond_2cam224.pt
checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json
```

先检查，不启动实验：

```bash
fastwam-ood doctor --config configs/eval_clean_smoke.yaml
```

没有 checkpoint/GPU 时，可先验证完整 CPU mock 数据链路：

```bash
fastwam-ood plan --config configs/eval_mock_smoke.yaml
fastwam-ood evaluate --config configs/eval_mock_smoke.yaml
fastwam-ood aggregate --experiment-dir outputs/mock_smoke
```

## 7. 单 GPU smoke test

`plan` 只写 job manifest，不加载 checkpoint、模型或环境：

```bash
fastwam-ood plan --config configs/eval_clean_smoke.yaml
fastwam-ood evaluate --config configs/eval_clean_smoke.yaml --device cuda:0 --dry-run
```

检查计划后再执行真实 smoke test（1 task × 2 episodes）：

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  fastwam-ood evaluate --config configs/eval_clean_smoke.yaml --device cuda:0
```

为了肉眼验收成功 episode，smoke 时还应将 `experiment.save_failure_video_only` 覆盖为 `false`；完整命令和 action/robot-state 检查见 [实施手册](docs/thought1_execution_guide.md)。

## 8. Clean baseline

单卡 Clean/OOD smoke、三卡 pilot 和正式 800 个 Clean baseline 均已完成。
现有结果位于 `outputs/thought1/fastwam/*/clean/`；下面命令只用于新目录复现或
真正的 incomplete resume，不要对已完成正式结果使用 `--overwrite`：

```bash
fastwam-ood plan --config configs/eval_clean_full.yaml
fastwam-ood evaluate --config configs/eval_clean_full.yaml --device cuda:0
```

这会使用原版 LIBERO 的官方 task suite、固定 init states、稀疏成功判定和 Fast-WAM 官方预/后处理。

## 9. OOD 测试

第一版选取 LIBERO-Plus 的真实类别 `Camera Viewpoints`、`Light Conditions`、`Background Textures`、`Robot Initial States`、`Objects Layout`。统一等级映射为 easy=官方 1–2、medium=3、hard=4–5；每个 job 都保留真实官方难度、classification ID 和变体名。

```bash
fastwam-ood plan --config configs/eval_ood_smoke.yaml
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  fastwam-ood evaluate --config configs/eval_ood_smoke.yaml --device cuda:0
```

LIBERO-Plus 的扰动来自其 BDDL、场景 XML、robot class、init state 和 observation wrapper，不在本项目中伪造图像。

## 10. 3 GPU 正式评测

`P1-FORMAL-v1` 已用三卡入口顺序完成四个 suite 的 800 Clean 和 6,771 OOD，
并生成 combined report。该入口保留用于严格复现或 incomplete resume；运行前
仍须确认 checkpoint/stats hash、suite、最大步数、五类扰动、三个等级和新输出
策略。LIBERO-Plus 必须保持每个官方 task variant 1 次，不能复制 Clean 的
20 次口径：

```bash
CONFIRM_FULL_EVAL=YES GPU_IDS=0,1,2 \
  bash scripts/run_thought1_3gpu_full.sh all
```

单个配置/单个 suite 的底层入口仍为：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 CONFIRM_FULL_EVAL=YES \
  bash scripts/run_3gpu_eval.sh configs/eval_ood_full.yaml
```

等价核心命令：

```bash
torchrun --standalone --nproc_per_node=3 \
  -m fastwam_ood_eval.cli distributed-evaluate \
  --config configs/eval_ood_full.yaml
```

每个 rank 写入 `outputs/<experiment>/workers/rank_N/`。默认 resume 不重复已落盘 job；`--rerun failed` 只重跑 exception/max_steps，`--overwrite` 重跑全部已分配 job。

单张 RTX 4090 顺序完成四个 suite 时使用项目脚本；它会自动激活本地环境并支持跨已有 `rank_*` 结果断点续跑：

```bash
CONFIRM_FULL_EVAL=YES GPU_ID=0 \
  bash scripts/run_thought1_single_gpu_full.sh all
```

`all/clean/ood` 分别表示完整 Clean+OOD、仅 Clean、仅 OOD。单卡后台运行、显存门禁和日志监控见 [实施与验收手册](docs/thought1_execution_guide.md#81-单卡-rtx-4090-全量脚本)。

## 11. 聚合结果

```bash
fastwam-ood aggregate --experiment-dir outputs/ood_full
fastwam-ood report --experiment-dir outputs/ood_full
```

输出位于 `summary/`：JSONL、episode CSV、按策略/任务/扰动/等级 CSV、failures、metrics 和 `report.md`。成功率 CI 使用固定随机种子的 95% bootstrap；若同时聚合 Clean 与 OOD 记录，还会给出配对 seed 的四格计数。Clean 与 OOD 分在两个目录时，建立一个比较输出目录并显式传入两者：

```bash
fastwam-ood aggregate --experiment-dir outputs/clean_vs_ood \
  --input-dir outputs/clean_full \
  --input-dir outputs/ood_full
```

## 12. 思考点二：Shadow Future Diagnostics

思考点二是独立、显式启用的旁观诊断链路。它先调用原 `FastWAMAdapter.act()` 得到将要执行的 action chunk，再尝试用同一模型的视频分支预测 future；预测结果从不反馈给策略或环境。实际环境始终执行原 action，诊断结果只写入新的 `outputs/thought2_*` 目录。普通的 `plan`、`evaluate`、`distributed-evaluate`、`aggregate` 和 `report` 不会隐式开启它。

当前明确分为两种模式：

- `unconditional_future`（2A）：适配官方 `libero_uncond` release；视频不读取受保护 action，只测未来先验与实际动作结果的一致性。
- `action_conditioned_future`（2B）：要求可信 action-conditioned checkpoint、完整参数 provenance 和动作依赖覆盖；当前 release 必须被门禁拒绝。

先确保对应的思考点一 source experiment 已真实执行过，且 manifest 中记录了非空的 checkpoint hash 与 Fast-WAM commit；仅运行 `plan` 得到的 `checkpoint_hash=null` 不足以开始真实诊断。随后可做不加载模型和环境的只读检查：

```bash
fastwam-ood diagnose-future \
  --config configs/studies/thought2_unconditional_smoke.yaml \
  --device cuda:0 \
  --dry-run
```

单卡 2-step 模型/episode smoke 的精确命令是：

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  fastwam-ood diagnose-future \
  --config configs/studies/thought2_unconditional_smoke.yaml \
  --device cuda:0
```

该真实 smoke 已于 2026-07-23 完成 1 job / 1 probe / 0 error，保存了当前帧、9 帧预测、3 帧实际对照、并排视频和完整动作；它只证明链路，不能当作成功率或未来质量结论。

随后完成的 20-step PILOT 包含 Clean 2 episodes/2 probes 与 camera-easy OOD
3 episodes/5 probes，共 14 个 aligned future frames、0 error。7/7 probe 的执行
动作与阶段一 trace 逐元素一致，5/5 outcome 一致；所有媒体可解码。描述性
Clean→OOD 数值为 L1 `0.1512→0.2002`、cosine distance
`0.1168→0.1942`、motion-direction cosine `0.7697→0.5283`。由于只有
5 个 episode、1 个严格 pair 且 static 阈值只有小样本 candidate、尚未冻结，
这些只能作为预实验假设。

五类正式 2A runner 已于 2026-07-26—27 完成：200 Clean + 532 OOD =
732 episodes、1,010 probes、2,020 aligned future frames，0 error；4,040 个
媒体工件全部通过解码审计。同次运行 1,010/1,010 probe 的 action hash
before/after 一致。40-task 等权结果为：

- cosine distance `0.1025→0.1341`，OOD−Clean
  `+0.0316 [0.0254, 0.0381]`；
- motion-direction cosine `0.7416→0.5518`，OOD−Clean
  `−0.1898 [−0.2134, −0.1664]`；
- OOD failure−success cosine 为 `+0.0249 [0.0166, 0.0328]`，仅首 probe
  sensitivity 为 `+0.0197 [0.0116, 0.0282]`。

这支持“OOD/失败与较差的自动 future–realized consistency proxy 相关”，不支持
“future error 导致动作失败”。统计方法与运行前 DRAFT 一致，但 DRAFT 没有预先
冻结，因此应标为正式 raw collection 上的 protocol-consistent post-run
analysis，而不是 preregistered confirmatory result。

三卡 20-step episode-level pilot 的精确命令是：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 MUJOCO_GL=egl \
torchrun --standalone --nproc_per_node=3 \
  -m fastwam_ood_eval.cli distributed-diagnose-future \
  --config configs/studies/thought2_unconditional_ood.yaml
```

该 OOD 配置只读复用已真实执行的 `outputs/ood_pilot`，选取 task 0/4/9 的三个 camera-viewpoint job；按既有 `job_id` 分片时三个 rank 各分到一个 episode。

2A 不能声称动作条件动力学或因果依赖。2B 仍使用 `thought2_shadow_*.yaml`：当前 release 的 `video_expert.action_conditioned=false`；即使未来提供新结构，pinned `first_frame_causal` 还会使 future frame 的传递依赖闭包覆盖完整 32-action horizon，超过阶段一固定的 `control_horizon=10`。上游 `strict=False` loader 也要求验证 action-embedding 实值和训练 provenance。当前 allowlist 为空，因此 2B 真实命令应在 reset/action 前失败，不能自动降级。

诊断聚合与报告使用独立命令：

```bash
fastwam-ood aggregate-diagnostics \
  --experiment-dir outputs/thought2_unconditional_smoke
fastwam-ood report-diagnostics \
  --experiment-dir outputs/thought2_unconditional_smoke
```

Clean/OOD 联合结果必须写到新的 comparison 目录；聚合器会生成独立 manifest，
记录共同 mode/provenance 和两份 source hash，禁止覆盖任一输入实验。

独立 static/no-op PILOT 也已完成：2 条 Clean + 五类 OOD 各 1 条，共
7/7 eligible、0 error；同帧编码噪声全为 0，8-step no-op energy 最大为
`0.013223`。这批历史 PILOT 只产生 `candidate_only` 阈值。后续正式 runner
已独立完成 200/200 eligible null jobs（Clean/OOD 各 100、五类各 20），并在
diagnostics 启动前接受正式阈值 `0.0167421166`。聚合和只读重分类 PILOT 的
命令为：

```bash
fastwam-ood aggregate-static-calibration \
  --experiment-dir outputs/thought2_static_calibration_pilot_comparison \
  --input-dir outputs/thought2_static_calibration_clean \
  --input-dir outputs/thought2_static_calibration_ood \
  --diagnostic-dir outputs/thought2_unconditional_clean \
  --diagnostic-dir outputs/thought2_unconditional_ood
fastwam-ood report-static-calibration \
  --experiment-dir outputs/thought2_static_calibration_pilot_comparison
```

旧阈值 1.0 下 predicted/actual static 均为 7/7；候选敏感性下均为 0/7，
源 diagnostics JSONL 不改写。完整执行、自动指标和人工失败归因见
[阶段二手册](docs/thought2_execution_guide.md)，校准细节见
[static/no-op 手册](docs/thought2_static_calibration.md)。

7 个真实 20-step probe 已另行生成 label-blind workflow packet：7 cases /
28 media，公开 manifest/HTML/CSV 不含 condition、outcome、metric 或 source
identifier，私有 key 独立保存映射；全部媒体完成解码与 hash 审计。当前 human
annotation 为 0/7，所以这仍不是 future 质量结论。

正式阶段二抽样也已实现并执行。v2 草案按 suite/task 选 200 条 Clean，
并强制每个 task 包含 episode index 0；按每个 supported
suite/task/category/difficulty cell 选 532 条 OOD，另记录 68 个 unsupported
cell。正式五类 runner 在任何 Phase 2 future metric 产生前把 exact job ID
复制并锁定到新的 `ratified_before_diagnostic_outcomes` manifest，随后完成
732/732。正式配置必须设置：

```yaml
diagnostics:
  cohort_manifest_path: <frozen-manifest.json>
  require_frozen_cohort: true
```

阶段一正式 outcome 现已出现，而 v2 草案此前没有通过 clean-tree freeze。
因此不能在事后把它改名为“阶段一 outcome 前的预注册 cohort”。新的 ratification
只证明 Phase 2 future 指标产生前 exact selection 未改变，并显式保留这个协议
偏差。五类后台入口为：

```bash
CONFIRM_PHASE2_FIVE_CATEGORY=YES \
ACCEPT_STATIC_THRESHOLD=YES \
GPU_IDS=0,1,2 \
bash scripts/run_thought2_five_category_full.sh --background all
```

完整命令和审计边界见
[盲审与 outcome-blind 抽样手册](docs/thought2_blind_review_and_sampling.md)。
正式派生分析命令为：

```bash
fastwam-ood analyze-thought2-formal \
  --experiment-dir outputs/thought2/five_category_formal_v1 \
  --thought1-summary outputs/thought1/fastwam/combined/summary/episode_results.csv \
  --source-trace-root outputs/thought1/fastwam \
  --output-dir outputs/thought2/five_category_formal_v1/formal_analysis_v1 \
  --bootstrap-replicates 10000 \
  --bootstrap-seed 20260725 \
  --verify-media
```

完整数值和论文/简历措辞见
[思考点二五类正式结果](docs/thought2_formal_results.md)。

## 13. 思考点三：Partial-Future Adapter

Phase B 已实现独立的 `thought3-*` CPU/mock 链路，不会读取或修改
Thought1/Thought2 输出。只看计划且不 import torch/加载 checkpoint：

```bash
fastwam-ood thought3-train \
  --config configs/thought3/train_a1_smoke.yaml \
  --dry-run
```

完整 mock 工程演练按顺序使用：

```bash
fastwam-ood thought3-plan-cache --config configs/thought3/cache_smoke.yaml
fastwam-ood thought3-build-cache --config configs/thought3/cache_smoke.yaml
fastwam-ood thought3-validate-cache --config configs/thought3/cache_smoke.yaml
fastwam-ood thought3-train --config configs/thought3/train_a1_smoke.yaml
fastwam-ood thought3-counterfactual --config configs/thought3/train_a1_smoke.yaml
```

以上命令是 Phase B 的 `TEST` 工程链路，不能解释为 OOD 提升。真实 Phase C/D
现已通过；E.6 也已在未使用 cohort 完成 400 updates、3,200 train objectives
和 160 held-out objectives。A1 下降 14.842%/7-of-8，但 A0 仅 4/8 不变差，
所以总 Gate 有效失败。E.7 的 800-objective 只读诊断工程 Gate 通过；primary
flow 上 A0 在 step 50/100 稳定、150/200 不稳定，但 pooled mean 继续下降，
故不支持实质晚期退化，且没有 joint checkpoint candidate。E.8 已完成
1,536-objective 只读诊断：step 200 full/双 block 的 pooled mean 都改善，
但一个预识别 target 和一个非 target 经校正确认恶化，主要分类为
`mixed_or_inconclusive`。见
[Gate E.6 报告](docs/thought3_phase_e6_report.md)与
[Gate E.7 报告](docs/thought3_phase_e7_report.md)、
[Gate E.8 协议](docs/thought3_phase_e8_protocol.md)和
[Gate E.8 结果](docs/thought3_phase_e8_report.md)。下一步 E.9a 的单变量
raw/normalized 四轨协议已冻结，但没有运行结果，见
[Gate E.9 预注册](docs/thought3_phase_e9_protocol.md)。

## 14. 查看失败视频

```bash
fastwam-ood review-failures --experiment-dir outputs/ood_full
# 浏览器打开 outputs/ood_full/failure_review/index.html
```

页面不需要后端；标注保存在浏览器 localStorage，并可导出 `annotations.json`。默认只保留失败视频。

## 15. 如何理解结果

报告能够说明 Fast-WAM 对已测扰动是否敏感、哪类/哪个强度下降最大，以及标准分布与 OOD 分布的实测差距。它不能说明显式未来想象一定能修复 OOD、Fast-WAM 完全没有世界建模能力、所有 WAM 都不需要未来想象，或仿真与真机 OOD 等价。详细统计口径见 [实验协议](docs/experiment_protocol.md)。

## 16. 常见报错

- `checkpoint ... missing`：运行下载脚本，或覆盖 `checkpoint.path` 和 `checkpoint.dataset_stats_path`。
- `A different libero package is already loaded`：Clean/OOD 必须分别启动新进程，不要在 notebook 内切换 backend。
- MuJoCo/EGL 初始化失败：设置 `MUJOCO_GL=egl` 与当前 worker 对应的 `MUJOCO_EGL_DEVICE_ID`。
- LIBERO-Plus asset not found：按环境文档将官方 `assets.zip` 解压到正确目录。
- CUDA OOM：保持每 GPU 一个 worker，确认 4090 实际可用显存，并降低并发；不要设置 `workers_per_gpu > 1`。

更多见 [故障排查](docs/troubleshooting.md)。

## 17. 下一阶段路线

阶段一与阶段二正式计算已完成；阶段三 Phase A–D 与 E.1–E.8 已完成。E.6
为有效负 Gate；E.7 的主要分类为
`not_supported_no_material_late_degradation`，且无 joint candidate；E.8 为
`mixed_or_inconclusive`。当前
路线是：

1. 保留 E.5 为有效负总 Gate，不覆盖 Run ID、不放宽共同门槛；
2. 保留 E.6 failed Run ID，不 resume、不把 A0 的 6/8 门槛降为 4/8；
3. 保留 E.7 primary 与 continuity 的证据等级，不事后合并或挑选 step 100/200；
4. 冻结 E.8 mixed 结果：不继续在同八条 sample 上堆 flow、改 target 或降低
   `2/3`/`6-of-8` 门槛；
5. 已预注册 matched raw/normalized × A0/A1 单变量 sample-tail mitigation；
   E.9a 尚未运行；
6. train 排序 17–28 已按身份与新 flow namespace 冻结，只能在 E.9a 产生
   candidate 后做一次性 demonstration-level 独立复验；
7. 只有形成并独立验证稳定配方后才冻结新的完整 28-train/4-development Gate E；
8. 完整 Gate E 通过后才训练 A2/A4，并实现真实 online no-cache evaluator；
9. Phase F 技术 pilot 后冻结分析协议，再解锁正式 ID/OOD 六组对照。

最近结果、停止条件和完整依赖链见
[Gate E.8 结果](docs/thought3_phase_e8_report.md)、
[Gate E.8 协议](docs/thought3_phase_e8_protocol.md)、
[Gate E.9 预注册](docs/thought3_phase_e9_protocol.md)、
[Gate E.7 结果](docs/thought3_phase_e7_report.md)、
[Gate E.7 协议](docs/thought3_phase_e7_protocol.md)、
[Gate E.6 结果](docs/thought3_phase_e6_report.md)、
[Gate E.6 预注册](docs/thought3_phase_e6_protocol.md)、
[Gate E.5 报告](docs/thought3_phase_e5_report.md) 与
[阶段三完成度审计](docs/thought3_completion_audit.md)。

实施过程中的工程决策、阻碍和可量化简历素材持续记录在 [工程台账](docs/engineering_highlights.md)。
