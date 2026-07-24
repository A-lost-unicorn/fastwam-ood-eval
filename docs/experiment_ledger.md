# 实验、卡点与结论台账

更新日期：2026-07-23

本台账只记录可追溯事实。机器工件是权威来源，本文是便于论文、周报、简历和面试使用的索引。

## 1. 已运行实验

| Run ID | 日期 | 阶段/等级 | 配置或来源 | 分母与结果 | 结论资格 |
| --- | --- | --- | --- | --- | --- |
| `P1-CLEAN-SMOKE-v1` | 2026-07-22 | 1 / SMOKE | `configs/eval_clean_smoke.yaml` | 2 completed，2 success，0 exception | 只证明 Clean 链路 |
| `P1-OOD-SMOKE-v1` | 2026-07-22 | 1 / SMOKE | `configs/eval_ood_smoke.yaml` | 4 completed，4 success，0 exception | 只证明 camera/light 链路 |
| `P1-OOD-PILOT-v1` | 2026-07-22 | 1 / PILOT | `configs/eval_ood_pilot.yaml` | 9 planned，8 attempted，2 success，1 skipped，0 exception | 不得作为正式 OOD 成功率 |
| `P1-FORMAL-PLAN-v1` | 2026-07-22 | 1 / PLAN | `outputs/thought1/.../job_manifest.jsonl` | 800 Clean；6,771 OOD runnable；68 skipped | 分母已审计，rollout 未运行 |
| `P2A-CLEAN-SMOKE-v1` | 2026-07-23 | 2A / SMOKE | `configs/studies/thought2_unconditional_smoke.yaml`；只读 `outputs/clean_smoke` | 1 job，1 probe，2 aligned future frames，0 error | 只证明真实 future 工件与指标链路 |
| `P2A-CLEAN-PILOT-v1` | 2026-07-23 | 2A / PILOT | `configs/studies/thought2_unconditional_clean.yaml`；只读 `outputs/clean_smoke` | 2 episodes，2 probes，4 aligned frames，2 success，0 error | 20-step ID 小样本 |
| `P2A-OOD-CAMERA-PILOT-v1` | 2026-07-23 | 2A / PILOT | `configs/studies/thought2_unconditional_ood.yaml`；只读 `outputs/ood_pilot` | 3 episodes，5 probes，10 aligned frames，1 success/2 max_steps，0 error | 只覆盖 camera/easy |
| `P2A-ID-OOD-COMP-v1` | 2026-07-23 | 2A / PILOT | 独立 multi-input comparison | 5 episodes，7 probes，14 aligned frames；严格 ID/OOD pair 仅 1 | 只用于形成假设 |
| `P2-STATIC-CLEAN-PILOT-v1` | 2026-07-23 | 2 / CALIBRATION PILOT | `outputs/thought2_static_calibration_clean` | 2 planned/completed/eligible，0 error | 独立 Clean null，小样本 |
| `P2-STATIC-OOD-PILOT-v1` | 2026-07-23 | 2 / CALIBRATION PILOT | `outputs/thought2_static_calibration_ood` | 五类 OOD 各 1；5/5 eligible，0 error | 类别覆盖 smoke，不是分布估计 |
| `P2-STATIC-COMP-v1` | 2026-07-23 | 2 / CALIBRATION PILOT | 独立 calibration comparison + 只读 pilot sensitivity | 候选阈值 `0.013223`；旧/候选 predicted-static `7/7→0/7` | `candidate_only`，不得冻结 |
| `P2-STATIC-FORMAL-PLAN-v2` | 2026-07-23 | 2 / CALIBRATION PLAN | `thought2_static_calibration_formal_{clean,ood}.yaml` | Clean 100 + OOD 100；五类 OOD 各 20；dry-run 0 skipped | 只读计划已审计，尚未运行 |
| `P2-BLIND-PACKET-PILOT-v1` | 2026-07-23 | 2 / WORKFLOW PILOT | 20-step Clean/OOD diagnostic 输入；public packet/private key 分离 | 7 cases / 28 media；0 sensitive public key；全媒体可解码；human labels 0/7 | 只证明盲审链路，不是人工 future 质量结果 |
| `P2-COHORT-FORMAL-DRAFT-v2` | 2026-07-23 | 2 / PLAN | 八份 outcome-blind manifests；seed `20260724` | Clean 200 + OOD 532；68 unsupported；0 supported shortfall；Clean 含 episode-0 anchor | `draft_not_frozen`；类别方案和 clean commit 待定 |
| `P2-SAP-DRAFT-v1` | 2026-07-23 | 2 / ANALYSIS PLAN | `thought2_statistical_analysis_plan.md` | 先 episode 后 task 聚合；suite-stratified task bootstrap；human/outcome/missing gate | 未冻结；不得据此声称已预注册或已有正式效应 |

### `P2A-CLEAN-SMOKE-v1` 机器证据

- checkpoint SHA-256：`1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579`
- Fast-WAM commit：`45d8e1458921d83f8ad6cf9ce993d371208dabd0`
- source manifest SHA-256：`8e9231615887e4a58053cec1ea7454247982b5fac1c73c4d30cd28c4429c4628`
- protocol fingerprint：`22f2ddbe80b18b07c6345e5dad4823a5e6e02842111ad8071c6c07a0714796da`
- action hash 前后完全一致：`42d23114...bbcad`
- 时间对齐：预测帧 0/1/2 对应环境 offset 0/4/8，即 0/0.2/0.4 s；运行时 control frequency 已验证为 20 Hz
- 生成：9 帧，2 个视频去噪步；generation latency `1,223.53 ms`
- 完整诊断：`4,616.06 ms`，不包含环境 step
- probe 峰值显存：`24,841.09 MB`；相对 probe 前增量约 `1,152.88 MB`
- 工件：当前帧 PNG、预测 9 帧 MP4、实际 3 帧 MP4、并排 3 帧 MP4，均已真实写入并抽检
- episode 在 `max_steps=10` 被 smoke 人为截断，`success=false` **不能解释为模型任务失败**
- 该 smoke 运行时阶段二实现尚未提交；旧 manifest 只记录 HEAD `9dfc254`，
  未记录 dirty 状态。因此它保持 SMOKE 资格，不作为可复现实验主结果。后续
  manifest 已增加 `git_dirty` 与三个上游 `*_dirty` 字段。

当前自动一致性数值仅留作管线检查：latent L1 `0.1437`、latent cosine distance `0.1025`、motion-direction cosine `0.7961`。不得将其写成模型能力结论。

### 20-step Clean/OOD pilot 机器证据

两组都使用 20 个视频去噪步、同一 checkpoint、相同 probe/metric 协议。下表是
**先在 episode 内聚合，再对 episode 等权**的均值：

| Group | Episodes / probes / aligned frames | Success / max_steps | Latent L1 | Cosine distance | Motion-direction cosine | Motion-energy ratio | Generation latency | Full diagnostic latency | Peak memory |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean | 2 / 2 / 4 | 2 / 0 | 0.1512 | 0.1168 | 0.7697 | 1.0573 | 4,108.12 ms | 7,214.25 ms | 24,841.09 MB |
| OOD camera/easy | 3 / 5 / 10 | 1 / 2 | 0.2002 | 0.1942 | 0.5283 | 1.4176 | 4,563.88 ms | 8,200.65 ms | 24,841.09 MB |

可复现性与工件验收：

- 项目 commit `b3c1be8`，项目与三个上游 source dirty 状态均为 `false`；
  checkpoint/Fast-WAM commit 与阶段一完全一致。
- Clean/OOD protocol fingerprint 分别为
  `56f2a1973d4074cc38f403ca803509e34a7804eb2fcd2ba7f96da487d3002c55`
  与
  `998601c443725de147ce2be899fe87a5a10fd2810e4b3a44454826f8959755b3`；
  source manifest SHA-256 分别为
  `8e9231615887e4a58053cec1ea7454247982b5fac1c73c4d30cd28c4429c4628`
  与
  `16b89cfcf22179604a6c2f38ccaba7bcbd088ca01e2410d46d19b3ac03f185e5`。
- 7/7 probe 的 10 条实际执行动作均与阶段一对应 trace **逐元素完全相同**，
  最大绝对差为 0；5/5 episode 的 success/termination 也完全复现。
- 7 张 current PNG、7 个 9-frame predicted MP4、7 个 3-frame actual MP4 和
  7 个 3-frame side-by-side MP4 均可解码；抽检 Clean success、OOD success、
  OOD failure 无黑帧、错位或损坏。
- 7/7 probe 都有 2 个精确对齐的 future frame；20 Hz、offset 0/4/8，
  approximate/unavailable/error 均为 0。
- Clean/OOD comparison manifest 显式记录两份 source manifest hash、两个输入
  protocol fingerprint 和 `mode=unconditional_future`，不会再把未知模式误报为
  action-conditioned。
- 两个输入实验是在 clean `b3c1be8` 上生成；本轮新增 comparison
  manifest/延迟聚合代码尚未提交，因此当前 comparison 的
  `aggregation_provenance.git_dirty=true`。它本来就是 PILOT；正式分析必须在
  提交后重新聚合，使输入与聚合 provenance 都可复现。

只允许作为预实验假设：

- 本 pilot 中 OOD 的 L1/cosine distance 较高，motion-direction cosine 较低；
  success 3 条对 failure 2 条也呈相同方向。
- 唯一严格配对的 task-0 episode 中，OOD-ID 为：L1 `+0.05984`、cosine
  distance `+0.10413`、motion-direction cosine `-0.20439`。
- 这些差异不能推断总体效应：Clean/OOD 只有 2/3 个 episode，仅 camera/easy，
  任务与 outcome 混杂、probe 数不等、严格 pair 只有 1，无法产生有意义的 CI。
- `static_motion_threshold=1.0` 把 7/7 明显运动 probe 都标成 static；
  predicted/actual energy 约为 `0.22–0.26` / `0.10–0.22`。static flag 继续禁用，
  必须先做独立 no-op 校准。

### Static/no-op calibration PILOT-v1 机器证据

- checkpoint SHA-256 与阶段一/二完全相同：
  `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579`。
- Clean/OOD protocol fingerprint：
  `d0a3b3db6ae8a1b3f80db0c5fe51c078d2e24db959db18bd76cd99bc01298ea7` /
  `6ec8d51045f0a14df3a18347c8231d5eceb92d5b3a00502a97f8ab8228a3db55`；
  compatibility fingerprint 都是
  `9981dd18a609fc0e28899916b3ee74f9b25495bd81013f7fe60ef10ef17bf072`。
- 2 Clean + camera/light/background/robot-init/object-layout 各 1 OOD；
  7/7 completed、eligible，0 exception/excluded/skipped。
- 运行时均为 20 Hz，官方双相机 model frame 均为 `224×448×3`；21 张
  offset `0/4/8` PNG 全部解码并做 contact-sheet 目检，无损坏。
- 同一帧重复编码的每样本最大噪声 7/7 为 0。no-op energy 的
  offset-4 中位数/最大值为 `0.00488984/0.01134326`，offset-8 为
  `0.00661479/0.01322303`。
- 8-step 逐条 energy：Clean task 1/6 为 `0.01148672/0.00829949`；
  OOD camera/light/background/robot-init/layout 为
  `0.00642072/0.00661479/0.00580508/0.01322303/0.00404918`。
- 99% `higher` 敏感性候选为 `0.0132230342`。旧阈值 1.0 下
  predicted/actual static 均为 7/7；候选下均为 0/7。最小
  predicted/actual energy 仍为候选值的 `16.41×/7.70×`。
- 候选状态必须保持 `candidate_only`：只有 7/200，Clean/OOD 仅
  2/100、5/100，五类各 1/20；而且 v1 source manifest 未预先记录
  `higher` 插值法。当前协议已补齐，任何新运行必须使用新目录。
- 项目为 `b3c1be8 + git_dirty=true`，三个上游 clean；因此即使链路和原始
  数值有效，也不能升级为 FORMAL。聚合器已将此编码为
  `all_source_trees_explicitly_clean=0/2` 的失败门禁；20 Hz 和
  `224×448×3` 一致性门禁均通过。
- 权威报告：
  `outputs/thought2_static_calibration_pilot_comparison/summary/static_calibration_report.md`。

### Blind-review packet PILOT-v1 机器证据

- public packet：
  `outputs/thought2_future_blind_pilot_packet`；private key：
  `outputs/thought2_future_blind_pilot_key`。
- packet ID `16a1dbc38c93c5367e665aef`；public manifest SHA-256
  `273c4b67b8a642c4b724289c6c56854952322c0ebb90d57bc11b74f942587b7f`。
- 7 个不透明 case、28 个复制媒体；public manifest/HTML/CSV 的敏感 key 和
  private source identifier 泄漏检查为 0，private mapping/hash 验证通过。
- 全量解码：7 PNG；predicted 为 7×9 帧 `224×448×3`；actual 为 7×3
  帧 `224×448×3`；comparison 为 7×3 帧 `224×896×3`。
- 第一轮 schema 有意不含 `primary_failure_hypothesis`。当前 CSV 是空模板，
  **0/7 case 有 human annotation**；不能写成 blind-review 质量结论。
- 已实现 blind-only annotation validator/agreement：要求两份完整、不同 reviewer
  的合法 CSV/JSON，分别输出 nonmissing 与 decisive 分母、exact agreement、
  pairwise Cohen's κ 和退化状态；只用合成标签通过回归测试，尚无真实 agreement
  数字。

### Outcome-blind cohort FORMAL-DRAFT-v2

- planner 只读取 source `job_manifest.jsonl` 和 planning-time `skip_reason`；
  manifest 明确记录 `outcome_fields_read=false`、
  `episode_result_files_read=false`；八个 formal source 当前均没有非空
  episode-result JSONL。
- Clean：4 suite × 10 task × 5 jobs = 200，并在每个 task 强制包含
  `episode_index=0`；OOD：每个 supported
  suite/task/category/difficulty cell 取 1，共 532。
- suite 分母：spatial `50+126`、object `50+137`、goal `50+139`、
  libero_10 `50+130`。68 个 skipped-only cell 保持 unsupported，supported
  shortfall 为 0。
- OOD 类别分母：background 103、camera 104、light 95、layout 110、
  robot-init 120。
- 八份 v2 manifest 均可根据 source hash 和 seed 精确重放，但当前项目 tree
  dirty，故 `frozen=false/status=draft_not_frozen`。正式 runner 新增
  `require_frozen_cohort=true` fail-closed 门禁。
- 原始路线说第四类采用“layout 或 robot-init”，而现有阶段一计划覆盖五类。
  因此 732 仍是覆盖草案：排除 robot-init 后为 612，排除 layout 后为 622；
  必须在查看正式 outcome 前选择并生成新 ID。

## 2. 失败尝试与解决记录

| 日期/尝试 | 现象 | 根因 | 修复与证据 | 是否污染实验 |
| --- | --- | --- | --- | --- |
| 2026-07-23 / P2A smoke attempt 1 | policy 导入时报 Fast-WAM dependencies unavailable | 官方 Fast-WAM evaluator 在 policy 构造时 import `libero`，但 backend 路径此前只在环境构造时设置 | 抽出无仿真副作用的 `configure_libero_package()`，policy 和 environment 复用 | 否；未加载模型、未 reset |
| attempt 2 | checkpoint 加载后报不同 LIBERO package | 同一工作区有 `/home/...` 与 `/data/...` 路径别名，字符串比较误判 | 改为 `Path.resolve()` 后的父目录身份判断 | 否；未 reset、无 diagnostic row |
| attempt 3 | 仍报不同 LIBERO package | 顶层 `libero` 是 namespace package，`__file__=None`，真实来源在 `__path__` | 同时验证 `__file__` 与全部 `__path__`；新增 symlink/namespace 回归测试 | 否；未 reset、无 diagnostic row |
| attempt 4 | 完成 | 完整链路通过 | 1 job / 1 probe / 0 error；142 tests passed | 产生有效 SMOKE 工件 |
| 2026-07-23 / Clean pilot attempt 1 | robosuite 在 import 时触发 EGL assertion | `CUDA_VISIBLE_DEVICES=1` 时，robosuite 要求 `MUJOCO_EGL_DEVICE_ID` 使用物理可见 ID `1`，而不是 torch 重映射后的 `0` | 保持 `--device cuda:0`，将 EGL ID 改为 `1` 后完成 2/2 episode | 否；模型未加载、环境未 reset、无 diagnostic row |
| 2026-07-23 / static calibration v1 协议复核 | 99% 分位数已固定，但 source manifest 未写插值方法；线性小样本分位数会低于观测最大 null | 把正式方法锁为保守 `higher`，增加自动 freeze check；协议 hash 变化时 dry-run/real 均拒绝复用旧目录 | v1 raw energy 不变，聚合只给 candidate；v2/FORMAL 必须新目录 | 否；没有改写 raw calibration 或 diagnostic JSONL |
| 2026-07-23 / outcome-blind cohort draft v1 | 随机 Clean 子集没有保证包含 OOD 共用的 episode index 0，削弱预先配对 | v1 只按 selection hash 抽样，没有 anchor 约束 | 新增 `anchor_episode_indices`，v2 每个 Clean task 固定 index 0；v1 目录只保留审计并标为 superseded | 否；只有 manifest 草案，未执行 episode、未读取 outcome |

冷启动观测：2-step smoke 中 Wan 组件装载约 `336–433 s`；20-step OOD
三进程观测到约 `604.37 s`，Clean 单进程为 `521.74 s`。这不是单次 future
latency；正式运行必须一 worker 多 episode 复用模型。

Provenance 补充核对：Fast-WAM 和 LIBERO checkout clean；LIBERO-Plus
只有非源码下载缓存 `.downloads/assets.zip`，无 tracked diff。后续 manifest
会显式记录 `.downloads/` 排除项，其他未跟踪文件和任何 tracked 修改仍会令
`*_dirty=true`。

## 3. 当前结论账

### 已支持

1. 阶段一评测工程链路和正式 manifest 已准备好。
2. 官方 `libero_uncond` release 在本机能够真实生成 unconditional future。
3. 阶段二 A 能在不改变动作哈希的前提下保存当前帧、预测未来、实际未来、动作、结果和资源指标。
4. 预测与实际帧可按官方 4 action/frame 比例精确对齐到控制步。
5. 在当前 5-episode pilot 上，shadow rerun 的执行动作和 outcome 与阶段一完全复现。
6. 20-step future 在当前硬件上约需 4.1–4.6 s，probe 峰值约 24.84 GB；这是一项明确的部署成本信号。
7. 独立 no-op calibration 的工程链路已真实覆盖 Clean 与五类 OOD；旧 static
   阈值 1.0 明显不在当前 embedding energy 的合理数量级。
8. 标签盲化 packet 能把 7 个真实 probe 的 condition/outcome/metric/source
   identity 留在私有 key 中，并对 28 个公开媒体做 hash/解码审计。
9. Outcome-blind planner 能在不读取 episode result 的条件下固定精确 job ID、
   unsupported cell 和 Clean index-0 anchor，并拒绝正式 runner 使用未冻结草案。

### 尚未支持

1. Fast-WAM 正式 Clean→OOD 成功率下降。
2. 哪类扰动最敏感。
3. unconditional future consistency 与成功/OOD 的稳定相关性；当前只有方向一致的 PILOT 趋势。
4. “失败来自未来错误还是动作错误”的自动因果分类。
5. 显式未来能改善 OOD，或 K=1/2/4 中哪个最好。
6. 正式 static threshold；当前 `0.013223` 只有 candidate 资格。
7. 人工盲审结论；当前只有空模板，human labels 为 0/7。
8. 732 是否是最终正式 cohort；五类/四类取舍、power justification 和 clean-tree
   freeze 均未完成。

## 4. 待填论文主表

以下表格是论文数字的唯一人工汇总入口。先从机器报告复制分母和估计值，再由
第二人核对 artifact；禁止手算后直接覆盖机器结果。

### 4.1 阶段一：ID→OOD 鲁棒性

| Suite | Perturbation | Level | ID n / SR / 95% CI | OOD n / SR / 95% CI | Absolute drop (pp) | Relative drop | Action latency p50 / p95 | Failure videos reviewed | FORMAL Run ID |
| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |
| 待正式运行 | — | — | — | — | — | — | — | — | — |

主文至少给总体和四类目标扰动；附录再展开 suite、difficulty 和变体。必须同时
报告 exception/skipped，并区分 episode-weighted 与 variant-weighted 口径。

### 4.2 阶段二：未来一致性

| Cohort | Mode | Condition/outcome | Episodes / probes / aligned frames | Video steps | L1 | Cosine distance | Motion-direction cosine | Human agreement | Generation / diagnostic latency | FORMAL Run ID |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 待正式阈值冻结与正式抽样 | 2A | — | — | 20 | — | — | — | — | — | — |

自动指标与盲审标签分开保存。success/failure matched cohort 不能用于估计总体失败率；
2A 结果始终标记 `causal_interpretation_allowed=false`。

### 4.3 阶段三：部分未来 Adapter

| Variant | K | Train seed | Trainable params | ID SR | OOD SR | Absolute drop | Future latency | Action latency | Total latency | Peak memory | FORMAL Run ID |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `B0-base` | 0 | — | 0 | — | — | — | 0 | — | — | — | — |
| `A0-null` | 0 | — | — | — | — | — | 0 | — | — | — | — |
| `A1` | 1 | — | — | — | — | — | — | — | — | — | — |
| `A2` | 2 | — | — | — | — | — | — | — | — | — | — |
| `A4` | 4 | — | — | — | — | — | — | — | — | — | — |

所有行必须使用相同 episode manifest、Action DiT 去噪步数、训练预算和选模规则。
`B0-base` 对 `A0-null` 控制额外参数/训练效应；`A0-null` 对 A1/A2/A4 才隔离
future 信息效应。

### 4.4 结论—证据登记

| Claim ID | 拟写结论 | 所属阶段 | 必须满足的证据 | 当前状态 | Artifact/Run ID |
| --- | --- | --- | --- | --- | --- |
| `C1` | Fast-WAM 对特定环境 shift 敏感 | 1 | 完整 ID/OOD 分母、配对 drop、CI、0 未解释 exception | 未支持 | — |
| `C2` | future consistency 在 OOD 或失败样本下降 | 2A | 冻结 cohort、校准阈值、episode-level 统计与盲审 | 未支持 | — |
| `C3` | 显式部分未来提高 OOD 成功率 | 3 | B0/A0/A1/A2/A4 配方匹配、跨 seed 配对 CI | 未支持 | — |
| `C4` | 某个 K 位于效果—延迟 Pareto 前沿 | 3 | 在线 latency/memory 与同 manifest ID/OOD 结果 | 未支持 | — |

只有满足“必须证据”后才能把状态改为“支持/不支持”；负结果同样保留，不按预期
方向筛选。

## 5. 更新纪律

- 新实验先登记 Run ID、配置、输出目录、checkpoint/commit、证据等级和停止条件。
- 失败尝试也登记；不得只保留最终成功路径。
- 结果表必须写分母、exception/skipped、CI 和适用范围。
- 修改协议后使用新 output directory 和 protocol fingerprint。
- `FORMAL` run 必须在项目和上游 checkout 均为 clean tree 时启动；否则降级为 PILOT。
- 简历数字必须能回指 manifest、JSONL、聚合报告或测试输出。
