# 实验、卡点与结论台账

更新日期：2026-07-27

本台账只记录可追溯事实。机器工件是权威来源，本文是便于论文、周报、简历和面试使用的索引。

## 1. 已运行实验

| Run ID | 日期 | 阶段/等级 | 配置或来源 | 分母与结果 | 结论资格 |
| --- | --- | --- | --- | --- | --- |
| `P1-CLEAN-SMOKE-v1` | 2026-07-22 | 1 / SMOKE | `configs/eval_clean_smoke.yaml` | 2 completed，2 success，0 exception | 只证明 Clean 链路 |
| `P1-OOD-SMOKE-v1` | 2026-07-22 | 1 / SMOKE | `configs/eval_ood_smoke.yaml` | 4 completed，4 success，0 exception | 只证明 camera/light 链路 |
| `P1-OOD-PILOT-v1` | 2026-07-22 | 1 / PILOT | `configs/eval_ood_pilot.yaml` | 9 planned，8 attempted，2 success，1 skipped，0 exception | 不得作为正式 OOD 成功率 |
| `P1-FORMAL-PLAN-v1` | 2026-07-22 | 1 / PLAN | `outputs/thought1/.../job_manifest.jsonl` | 800 Clean；6,771 OOD runnable；68 skipped | 分母已审计，rollout 未运行 |
| `P1-FORMAL-v1` | 2026-07-24–26 | 1 / FORMAL | 8 个 `outputs/thought1/fastwam/<suite>/<condition>` source | Clean 778/800=97.25%；OOD 3,230/6,771=47.70%；drop 49.55 pp；68 skipped；0 exception | 支持当前五类 LIBERO-Plus 环境 shift 的正式鲁棒性结论 |
| `P2A-CLEAN-SMOKE-v1` | 2026-07-23 | 2A / SMOKE | `configs/studies/thought2_unconditional_smoke.yaml`；只读 `outputs/clean_smoke` | 1 job，1 probe，2 aligned future frames，0 error | 只证明真实 future 工件与指标链路 |
| `P2A-CLEAN-PILOT-v1` | 2026-07-23 | 2A / PILOT | `configs/studies/thought2_unconditional_clean.yaml`；只读 `outputs/clean_smoke` | 2 episodes，2 probes，4 aligned frames，2 success，0 error | 20-step ID 小样本 |
| `P2A-OOD-CAMERA-PILOT-v1` | 2026-07-23 | 2A / PILOT | `configs/studies/thought2_unconditional_ood.yaml`；只读 `outputs/ood_pilot` | 3 episodes，5 probes，10 aligned frames，1 success/2 max_steps，0 error | 只覆盖 camera/easy |
| `P2A-ID-OOD-COMP-v1` | 2026-07-23 | 2A / PILOT | 独立 multi-input comparison | 5 episodes，7 probes，14 aligned frames；严格 ID/OOD pair 仅 1 | 只用于形成假设 |
| `P2-STATIC-CLEAN-PILOT-v1` | 2026-07-23 | 2 / CALIBRATION PILOT | `outputs/thought2_static_calibration_clean` | 2 planned/completed/eligible，0 error | 独立 Clean null，小样本 |
| `P2-STATIC-OOD-PILOT-v1` | 2026-07-23 | 2 / CALIBRATION PILOT | `outputs/thought2_static_calibration_ood` | 五类 OOD 各 1；5/5 eligible，0 error | 类别覆盖 smoke，不是分布估计 |
| `P2-STATIC-COMP-v1` | 2026-07-23 | 2 / CALIBRATION PILOT | 独立 calibration comparison + 只读 pilot sensitivity | 候选阈值 `0.013223`；旧/候选 predicted-static `7/7→0/7` | `candidate_only`，不得冻结 |
| `P2-STATIC-FORMAL-PLAN-v2` | 2026-07-23 | 2 / CALIBRATION PLAN | `thought2_static_calibration_formal_{clean,ood}.yaml` | Clean 100 + OOD 100；五类 OOD 各 20；dry-run 0 skipped | 历史计划；已由 `P2-STATIC-FORMAL-v1` 执行完成 |
| `P2-BLIND-PACKET-PILOT-v1` | 2026-07-23 | 2 / WORKFLOW PILOT | 20-step Clean/OOD diagnostic 输入；public packet/private key 分离 | 7 cases / 28 media；0 sensitive public key；全媒体可解码；human labels 0/7 | 只证明盲审链路，不是人工 future 质量结果 |
| `P2-COHORT-FORMAL-DRAFT-v2` | 2026-07-23 | 2 / PLAN | 八份 outcome-blind manifests；seed `20260724` | Clean 200 + OOD 532；68 unsupported；0 supported shortfall；Clean 含 episode-0 anchor | `draft_not_frozen`；类别方案和 clean commit 待定 |
| `P2-SAP-DRAFT-v1` | 2026-07-23 | 2 / ANALYSIS PLAN | `thought2_statistical_analysis_plan.md` | 先 episode 后 task 聚合；suite-stratified task bootstrap；human/outcome/missing gate | 未冻结；不得据此声称已预注册或已有正式效应 |
| `P2A-FIVE-CATEGORY-FULL-PLAN-v1` | 2026-07-26 | 2A / FORMAL DATA-COLLECTION PLAN | `run_thought2_five_category_full.sh` + formal five-category template | static 100+100；diagnostic 200 Clean + 532 OOD；8 suite×condition 组；20 video steps、≤2 probes/episode | 历史计划；已由后续 collection 执行完成；ratification 只覆盖 Phase 2 指标前锁定 |
| `P2-STATIC-FORMAL-v1` | 2026-07-26 | 2 / FORMAL CALIBRATION | `outputs/thought2/five_category_formal_v1/static` | 200/200 eligible；Clean/OOD 各 100；五类 OOD 各 20；阈值 `0.0167421166` | diagnostic 启动前通过全部 freeze gate 并由 runner 显式接受 |
| `P2A-FIVE-CATEGORY-COLLECTION-v1` | 2026-07-26–27 | 2A / FORMAL DATA COLLECTION | commit `0fb8350`；`run_thought2_five_category_full.sh` | 200 Clean + 532 OOD；732 episodes / 1,010 probes / 2,020 aligned frames；0 error | 正式 raw collection 完成；项目与三个上游 clean |
| `P2A-FIVE-CATEGORY-ANALYSIS-v1` | 2026-07-27 | 2A / POST-RUN ANALYSIS | `formal_analysis_v1`；10,000 次 suite-stratified task bootstrap | cosine OOD−Clean `+0.0316`；direction `−0.1898`；730/732 outcome match；4,040 media 0 decode error | 与运行前 DRAFT 方法一致，但 DRAFT 未冻结；不得称 preregistered confirmatory |

### `P1-FORMAL-v1` 机器证据

- 项目 commit `575ba8fcd89f6baf801190fcb8127142ba0406c5`；项目和三个
  上游 checkout 在运行时均为 clean。
- checkpoint SHA-256：
  `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579`；
  7,639 条结果中无第二种 checkpoint、策略或 commit。
- 7,639 planned/result rows = 800 Clean + 6,771 runnable OOD + 68 expected
  skipped；7,639 个 job ID 唯一，manifest/raw result 无缺失、多余或重复。
- Clean `778/800=97.25%`，row-bootstrap CI `[96.00%, 98.38%]`；
  OOD `3,230/6,771=47.70%`，CI `[46.55%, 48.90%]`；
  绝对/相对下降 `49.55 pp / 50.95%`。
- 扰动 SR：camera `15.13%`、robot-init `42.84%`、background `51.49%`、
  layout `61.25%`、light `81.88%`。Camera 在四个 suite 内均为最低类别。
- easy/medium/hard SR 为 `59.82% / 49.51% / 35.07%`；五类扰动内部均保持
  粗粒度 `easy ≥ medium ≥ hard`。
- 40-task 等权敏感性：Clean `97.25%`、OOD `48.03%`、drop `49.22 pp`，
  task-bootstrap CI `[42.14, 56.39] pp`，与 variant-weighted 主结论一致。
- 7,571 条 trace、2,399,314 个 action step：0 非有限/空/全零运动 episode；
  末端首末位移最小 `0.0385 m`。3,563 个失败均为 `max_steps`，对应
  3,563 个非空失败视频；0 exception。
- 首条到末条结果跨度 `44.26 h`，合计 `123.20 episode GPU-hours`；
  action-chunk latency p50/p95 `969.51/978.18 ms`，峰值显存
  `23,814.42 MB/worker`。
- 权威工件：
  `outputs/thought1/fastwam/combined/{experiment_manifest.json,summary/}`；
  详细解释见 [thought1_report.md](thought1_report.md)。

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

### Outcome-blind cohort DRAFT-v2

- planner 在 2026-07-23 只读取 source `job_manifest.jsonl` 和 planning-time
  `skip_reason`；manifest 明确记录 `outcome_fields_read=false`、
  `episode_result_files_read=false`。这是生成当时的历史事实；2026-07-26
  `P1-FORMAL-v1` 已产生 outcome JSONL。
- Clean：4 suite × 10 task × 5 jobs = 200，并在每个 task 强制包含
  `episode_index=0`；OOD：每个 supported
  suite/task/category/difficulty cell 取 1，共 532。
- suite 分母：spatial `50+126`、object `50+137`、goal `50+139`、
  libero_10 `50+130`。68 个 skipped-only cell 保持 unsupported，supported
  shortfall 为 0。
- OOD 类别分母：background 103、camera 104、light 95、layout 110、
  robot-init 120。
- 八份 v2 manifest 均可根据 source hash 和 seed 精确重放，但当时项目 tree
  dirty，故 `frozen=false/status=draft_not_frozen`。现在 outcome 已存在，
  不能把它们追溯称为阶段一 outcome 前预注册。新增
  `ratified_before_diagnostic_outcomes` 只复制并锁定原 exact job ID，记录 draft
  hash 和 source outcome 已存在；未 ratify 的 draft 仍由
  `require_frozen_cohort=true` 拒绝。
- 原始路线说第四类采用“layout 或 robot-init”，而现有阶段一计划覆盖五类。
  2026-07-26 正式 runner 明确选择并锁定五类 732-job 版本；因此本轮结果回答
  五类扰动，不可事后删掉某一类再把 612/622 当成同一 formal cohort。

### `P2A-FIVE-CATEGORY-COLLECTION/ANALYSIS-v1` 机器证据

- 运行状态 `completed`、exit code 0；时间
  `2026-07-26 18:18:15—2026-07-27 00:41:46`，三卡墙钟约
  `6 h 23 min 31 s`。
- 正式 raw collection 使用项目 clean commit `0fb8350`；项目与 Fast-WAM、
  LIBERO、LIBERO-Plus 均 clean。checkpoint SHA-256 与阶段一相同：
  `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579`。
- static/no-op calibration 为 `200/200` eligible：Clean/OOD 各 100、
  五类 OOD 各 20、0 error；runner 在 diagnostics 前显式接受正式阈值
  `0.016742116587908088`。
- diagnostics 完成 `200 Clean + 532 OOD = 732 episodes`、
  `1,010 probes`、`2,020 aligned future frames`，0 diagnostic error。
  Clean 为 198 success/2 failure；OOD 为 256 success/276 failure。
- 每个成功 episode 有 1 个 probe，每个失败 episode 有 2 个 probe。因此主表
  同时给全部可用 probe 与仅首 probe sensitivity，避免把失败轨迹的额外 probe
  当作独立证据。
- 同一次 rerun 内 `1,010/1,010` probe 的 action hash before/after 完全一致。
  与阶段一历史 trace 跨运行比较为 `996/1,010` exact、13 mismatch、1 unavailable；
  最大绝对差 2.0。Phase 1/2 outcome 为 `730/732` 一致；两个不一致 episode
  从 outcome association 排除，但保留在 ID/OOD consistency 分析。
- 1,010 张 current PNG，加 predicted/actual/side-by-side 各 1,010 个 MP4，
  共 4,040 个媒体工件；全量解码、帧数、分辨率检查均为 0 error。
- 40-task 等权主 contrast：
  cosine distance Clean/OOD `0.1025/0.1341`，OOD−Clean
  `+0.0316`，95% CI `[0.0254, 0.0381]`；latent L1 为
  `+0.0277 [0.0238, 0.0317]`；motion-direction cosine 为
  `−0.1898 [−0.2134, −0.1664]`。
- outcome-matched OOD 分母为 255 success/275 failure、40/40 mixed tasks。
  failure−success cosine 为 `+0.0249 [0.0166, 0.0328]`；仅首 probe 为
  `+0.0197 [0.0116, 0.0282]`。Direction contrast 分别为
  `−0.2127 [−0.2328, −0.1923]` 与
  `−0.0784 [−0.1046, −0.0541]`。
- 仅首 probe 的 cosine 最低误差四分位仍有 `55/132=41.67%` failure，
  最高误差四分位为 `87/133=65.41%` failure。未来一致性与失败相关，但既非
  成功充分条件，也非必要条件。
- 20-step generation mean/p50/p95 为
  `3,354.66/3,316.96/3,564.12 ms`；完整 diagnostic 为
  `5,816.77/5,762.95/6,271.52 ms`；峰值显存 `24,841.09 MB`。
  这是 control-loop 外 shadow 成本，不是 base policy action latency。
- 权威人工解读见 [thought2_formal_results.md](thought2_formal_results.md)；
  机器工件位于
  `outputs/thought2/five_category_formal_v1/formal_analysis_v1/`。
- 统计方法与运行前 DRAFT 一致，但 DRAFT 未冻结。因此资格是
  `formal data collection + protocol-consistent post-run analysis`，
  不能表述为 preregistered confirmatory analysis。

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
| 2026-07-24 / 三卡 full 启动前预检 | runner 立即返回，`nvidia-smi` 报 NVML driver/library mismatch | 内核仍加载 580.159.03，用户态 NVML/DKMS 已升级到 580.173.02；原脚本在 command substitution 中被 `set -e` 静默截断 | 显式捕获并打印 `nvidia-smi -i` 错误，重启加载 580.173.02；commit `575ba8f` 后正式运行完成 | 否；模型未加载、无 rollout/result |

冷启动观测：2-step smoke 中 Wan 组件装载约 `336–433 s`；20-step OOD
三进程观测到约 `604.37 s`，Clean 单进程为 `521.74 s`。这不是单次 future
latency；正式运行必须一 worker 多 episode 复用模型。

Provenance 补充核对：Fast-WAM 和 LIBERO checkout clean；LIBERO-Plus
只有非源码下载缓存 `.downloads/assets.zip`，无 tracked diff。后续 manifest
会显式记录 `.downloads/` 排除项，其他未跟踪文件和任何 tracked 修改仍会令
`*_dirty=true`。

## 3. 当前结论账

### 已支持

1. 阶段一正式 7,571 个 runnable rollout 已完成：Clean 97.25%、OOD 47.70%，
   绝对下降 49.55 pp，0 exception。
2. Camera viewpoints 是当前五类中最敏感且跨四个 suite 稳定最低的扰动；
   coarse difficulty 在五类内均呈 easy→medium→hard 下降。
3. 官方 `libero_uncond` release 在本机能够真实生成 unconditional future。
4. 阶段二 A 能在不改变动作哈希的前提下保存当前帧、预测未来、实际未来、动作、结果和资源指标。
5. 预测与实际帧可按官方 4 action/frame 比例精确对齐到控制步。
6. 在当前 5-episode pilot 上，shadow rerun 的执行动作和 outcome 与阶段一完全复现。
7. 20-step future 在当前硬件上约需 4.1–4.6 s，probe 峰值约 24.84 GB；这是一项明确的部署成本信号。
8. 独立 no-op calibration 的工程链路已真实覆盖 Clean 与五类 OOD；旧 static
   阈值 1.0 明显不在当前 embedding energy 的合理数量级。
9. 标签盲化 packet 能把 7 个真实 probe 的 condition/outcome/metric/source
   identity 留在私有 key 中，并对 28 个公开媒体做 hash/解码审计。
10. Outcome-blind planner 能在不读取 episode result 的条件下固定精确 job ID、
   unsupported cell 和 Clean index-0 anchor，并拒绝正式 runner 使用未冻结草案。
11. 五类正式 static/no-op calibration 已完成：200/200 eligible，正式阈值
    `0.0167421166` 在 diagnostic 启动前通过全部 freeze gate。
12. 五类 Phase 2 raw collection 已完成：732 episodes、1,010 probes、
    2,020 aligned future frames、0 error；4,040 个媒体全部通过解码审计。
13. 自动 future–realized proxy 在 OOD 下稳定变差：task-equal cosine distance
    增加 `0.0316 [0.0254, 0.0381]`，motion-direction cosine 下降
    `0.1898 [0.1664, 0.2134]`。
14. OOD failure 的 future inconsistency 高于 success，且仅首 probe sensitivity
    仍保持同方向；这是同轨迹关联证据，不是 future 对 action 的因果依赖。
15. 同一次 Phase 2 执行中 1,010/1,010 probe 的动作哈希保持不变；该 shadow
    诊断没有把 future 反馈给控制动作。

### 尚未支持

1. 自动 proxy 是否等价于语义上的“未来预测正确”；goal progress、物理合理性、
   wrong-object/wrong-goal 仍缺标签盲化人工评审。
2. “失败来自未来错误还是动作错误”的自动或因果分类。基础模型的动作不读取
   shadow future，因此本阶段不能把 future error 写成失败原因。
3. 显式未来能否改善 OOD，或 K=1/2/4 中哪个最好；必须由阶段三
   B0/A0/A1/A2/A4 对照回答。
4. 正式 human-review 结论与 reviewer agreement；当前只有 7-case pilot 空模板。
5. 把本次自动统计称为 preregistered confirmatory analysis；统计 DRAFT 在
   formal future 指标生成前没有冻结。
6. 20-step shadow RGB 生成成本是否等于阶段三 cached latent 或在线 Adapter
   成本；阶段三必须单独计时。

## 4. 待填论文主表

以下表格是论文数字的唯一人工汇总入口。先从机器报告复制分母和估计值，再由
第二人核对 artifact；禁止手算后直接覆盖机器结果。

### 4.1 阶段一：ID→OOD 鲁棒性

| Suite | Perturbation | Level | ID n / SR / 95% CI | OOD n / SR / 95% CI | Absolute drop (pp) | Relative drop | Action latency p50 / p95 | Failure videos reviewed | FORMAL Run ID |
| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |
| Overall | all five | all | 800 / 97.25% / [96.00%, 98.38%] | 6,771 / 47.70% / [46.55%, 48.90%] | 49.55 | 50.95% | 969.51 / 978.18 ms | 0（3,563 saved） | `P1-FORMAL-v1` |
| Overall | camera | all | 800 / 97.25% / [96.00%, 98.38%] | 1,599 / 15.13% / [13.38%, 16.95%] | 82.12 | 84.44% | — | 0 | `P1-FORMAL-v1` |
| Overall | robot init | all | 800 / 97.25% / [96.00%, 98.38%] | 1,550 / 42.84% / [40.39%, 45.23%] | 54.41 | 55.95% | — | 0 | `P1-FORMAL-v1` |
| Overall | background | all | 800 / 97.25% / [96.00%, 98.38%] | 1,076 / 51.49% / [48.51%, 54.46%] | 45.76 | 47.06% | — | 0 | `P1-FORMAL-v1` |
| Overall | layout | all | 800 / 97.25% / [96.00%, 98.38%] | 1,525 / 61.25% / [58.75%, 63.67%] | 36.00 | 37.02% | — | 0 | `P1-FORMAL-v1` |
| Overall | light | all | 800 / 97.25% / [96.00%, 98.38%] | 1,021 / 81.88% / [79.43%, 84.04%] | 15.37 | 15.80% | — | 0 | `P1-FORMAL-v1` |

主文至少给总体和四类目标扰动；附录再展开 suite、difficulty 和变体。必须同时
报告 exception/skipped，并区分 episode-weighted 与 variant-weighted 口径。

### 4.2 阶段二：未来一致性

| Cohort | Mode | Condition/outcome | Episodes / probes / aligned frames | Video steps | L1 | Cosine distance | Motion-direction cosine | Human agreement | Generation / diagnostic latency | FORMAL Run ID |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 五类正式 cohort | 2A | Clean | 200 / 202 / 404 | 20 | 0.1431 | 0.1025 | 0.7416 | 待人工盲审 | 3,336.63 / 5,785.64 ms（mean） | `P2A-FIVE-CATEGORY-ANALYSIS-v1` |
| 五类正式 cohort | 2A | OOD | 532 / 808 / 1,616 | 20 | 0.1708 | 0.1341 | 0.5518 | 待人工盲审 | 3,361.44 / 5,828.48 ms（mean） | `P2A-FIVE-CATEGORY-ANALYSIS-v1` |
| OOD outcome association | 2A | 255 success / 275 failure；排除 2 mismatch | 530 / 805 / 1,610 | 20 | failure−success +0.0196 | failure−success +0.0249 | failure−success −0.2127 | 待人工盲审 | 同上 | `P2A-FIVE-CATEGORY-ANALYSIS-v1` |

自动指标与盲审标签分开保存。success/failure matched cohort 不能用于估计总体失败率；
2A 结果始终标记 `causal_interpretation_allowed=false`。前两行是全部可用 probe
的 task-equal 均值；outcome 行 L1 的 `+0.0196`、cosine 的 `+0.0249` 和
direction 的 `−0.2127` 也是全部可用 probe 口径，首 probe sensitivity 另见
正式结果文档。

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
| `C1` | Fast-WAM 对特定环境 shift 敏感 | 1 | 完整 ID/OOD 分母、配对 drop、CI、0 未解释 exception | **支持** | `P1-FORMAL-v1`；`outputs/thought1/fastwam/combined/summary/` |
| `C2a` | 自动 future–realized consistency proxy 在 OOD/失败样本下降 | 2A | 固定 cohort、正式阈值、episode→task 统计、probe-count sensitivity | **支持（post-run analysis，非 preregistered）** | `P2A-FIVE-CATEGORY-ANALYSIS-v1`；`formal_analysis_v1/` |
| `C2b` | 语义 future 正确性与“未来错误/动作错误”失败机制 | 2A | 双人标签盲审、agreement、解盲后机制标签；因果措辞仍禁止 | 未支持 | 仅 blind packet PILOT |
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
