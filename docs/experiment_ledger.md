# 实验、卡点与结论台账

更新日期：2026-07-29

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
| `P3-PHASE-A-v1` | 2026-07-27 | 3 / PLAN-AUDIT | `thought3_{upstream_audit,design,risk_register}.md` | 上游/shape/scheduler/injection/显存/隔离审计；用户确认 10 项默认值 | 允许进入 Phase B；无模型结果 |
| `P3-PHASE-B-v1` | 2026-07-27 | 3 / TEST | `src/fastwam_ood_eval/thought3/`、`configs/thought3/`、Thought3 tests | 1.371M Adapter、paired cache、mock trainer、Adapter-only resume、counterfactual、online-no-cache 与旧 CLI 回归通过 | 只证明 CPU/mock 工程 contract；无真实 latent/训练/OOD 结论 |
| `P3-PHASE-C-v1` | 2026-07-28 | 3 / SMOKE | `configs/thought3/phase_c_single_sample.yaml`；GPU 1；真实 `libero_goal` sample 0 | K1/K2/K4、upstream parity、zero-gate、1 backward、0 backbone grad、12.964 GiB peak；0 cache/optimizer step | Gate C 通过，允许小型真实 cache；不支持训练收敛、OOD 增益或 K 排序 |
| `P3-PHASE-D-v1` | 2026-07-28 | 3 / SMOKE | commit `02a010e`；`phase_d_cache_smoke.yaml`；GPU 1；`libero_goal` task 0 | 32 base samples × K1/K2/K4 = 96 entries；12 shards；paired/checksum/resume/leakage 通过；0.806 sample/s；12.677 GiB execution peak | Gate D 通过，允许 100–500 step 单卡训练 smoke；不支持收敛、OOD 增益、K 排序或在线 latency |
| `P3-PHASE-E-v1` | 2026-07-28 | 3 / FAILED-SMOKE | commits `eb5ec8a..2b42964`；GPU 1 | A0 resumed/uninterrupted 各 100 step；默认 CUDA backward 从 step 2 出现微小非确定性，最终 semantic SHA 不同 | invalid diagnosis；促成确定性 CUDA gate，不得用于模型结果 |
| `P3-PHASE-E-v2` | 2026-07-28 | 3 / FAILED-SMOKE | commit `c4fcadb`；A0/A1；确定性 CUDA | 两组各 50→100 + uninterrupted 100；第 2 step 非 gate grad；最终 SHA 完全一致；A1 development 未低于初始 | 工程恢复/梯度子门禁通过；总 Gate 拒绝，无 future 效果结论 |
| `P3-PHASE-E-v3` | 2026-07-28 | 3 / FAILED-SMOKE | commit `dc77bd2`；固定 4-sample train probe | A0 两条 100-step 轨迹完全重放；fixed train probe `0.0015776→0.0015993`，未下降，停止于 A0 | Gate E 未通过；进入单样本 overfit/优化诊断，不扩 A2/A4 |
| `P3-PHASE-E1-v1` | 2026-07-28 | 3 / ENGINEERING DIAGNOSTIC | prereg commit `30ffc93`；GPU 1；单 train sample、固定 noise/timestep；A0/A1 各 200 step | A0 loss `0.0358901→0.0025362`（−92.93%）；A1 `→0.0001490`（−99.58%）；first non-gate step=2；frozen SHA before=after | Gate E.1 通过，只证明单目标可拟合；发现 BF16 delta/action-hidden 为 A0 1.91×、A1 0.70×；Gate E、A2/A4、OOD 仍锁定 |
| `P3-PHASE-E2-v1` | 2026-07-28 | 3 / FAILED ENGINEERING DIAGNOSTIC | prereg commit `e104328`；8 train samples；A0/A1 × LR `1e-4/3e-4/1e-3`；六轨迹共 1,200 step | 六轨迹 execution/pairing/frozen/checkpoint/memory 全通过；无共同 eligible LR；A1 mean reduction 为 24.19%/40.01%/−13.97%，但 non-worsened 仅 4/8、4/8、0/8 | Gate E.2 按预注册 6/8 门槛失败；不得回改阈值或扩 A2/A4；单固定 flow draw 的初始 loss 跨度 94.28× |
| `P3-PHASE-E3-v1` | 2026-07-28 | 3 / INVALID ENGINEERING RUN | commit `330fe15`；config fingerprint `f2313eec...5652`；GPU 1；只读 E.2 checkpoint | model/data/A0/A1 initial probe 完成；A0/A1 mean loss 均为 `0.005565503754223755`；一个 `t=1000` objective 的官方 weight/loss 为 0；非门控 ratio 汇总抛错；frozen SHA 前后相同 | 未生成 `gate_e3_result.json`，无 Gate pass/fail 或 LR 结论；v1 工件按 SHA 冻结，不得覆盖 |
| `P3-PHASE-E3-v2` | 2026-07-28 | 3 / FAILED ENGINEERING DIAGNOSTIC | prereg commit `139742f`；8 samples、6 E.2 checkpoints、held-out flow `1..5`；320 forward、0 optimizer/backward | 全部 execution/provenance/zero-weight checks 通过；A0 最好下降 1.35%/2-of-8，A1 最好下降 0.025%/2-of-8；三个 LR 均不 eligible | 有效负 Gate；E.2 的 fixed-flow 降幅未迁移到 held-out flow；不能解释为 future 无效，不扩 A2/A4/Phase F |
| `P3-PHASE-E4-v1` | 2026-07-28 | 3 / FAILED ENGINEERING DIAGNOSTIC | prereg commit `07d949d`；8 samples；A0/A1 × 原 LR grid；唯一 paired slot `10001..10200`；held-out flow `1..5` | 六轨迹 1,200 step、480 held-out objectives 和 108 execution checks 完整；六条 reduction 均为正但仅 0.997%–1.948%；只有 A1@3e-4 达 7/8，三个 LR 均不 eligible | 有效负 Gate；diversified flow 缓解 fixed-flow 退化但不足 10%；不进入 full E、A2/A4 或 Phase F |
| `P3-PHASE-E5-v1` | 2026-07-28 | 3 / FAILED ENGINEERING DIAGNOSTIC | prereg commit `a8245d1`；config fingerprint `c4c6815...122fc`；matched-update；A0/A1 × 原 LR grid；每 update 8-sample mean；slots `20001..21600` | 六轨迹完成 1,200 updates、9,600 train objectives、480 held-out objectives；120/120 execution checks 通过；A1@3e-4 下降 19.668%/8-of-8，A0@3e-4 仅 2.638%；三个 LR 均不 eligible | 有效负总 Gate；full-cohort aggregation 产生待新 cohort 复验的探索性 A1 信号，但不得事后选择 3e-4、扩 A2/A4 或作 future/OOD 结论 |
| `P3-PHASE-E6-v1` | 2026-07-29 | 3 / FAILED ENGINEERING DIAGNOSTIC | prereg commit `cb6f311`；新 train cohort 9–16；A0/A1@3e-4；slots `31001..32600` | 两轨迹完成 400 updates、3,200 train objectives、160 held-out objectives；A1 下降 14.842%/7-of-8；A0 下降 1.191%/4-of-8 | 有效负 Gate；A1 absolute/paired superiority 复现，但 A0 stability 未过门；不解锁 full E/A2/A4/OOD |
| `P3-PHASE-E7-PREREG-v1` | 2026-07-29 | 3 / PRE-REGISTERED DIAGNOSTIC | `phase_e7_checkpoint_trajectory.yaml`；E.6 step 50/100/150/200；primary flow `6..10` | 预注册冻结时预算为 8 checkpoints、800 forward objectives、0 backward/optimizer/新训练，尚未运行 | 只诊断 A0 instability 是否符合晚期退化；运行结果另见 `P3-PHASE-E7-v1` |
| `P3-PHASE-E7-v1` | 2026-07-29 | 3 / COMPLETED READ-ONLY DIAGNOSTIC | prereg commit `703b57fd`；primary flow `6..10`、continuity `1..5`；8 checkpoints | 800/800 forward objectives、0 backward/optimizer/write；primary A0 step 50/100 pass、150/200 为 5/8 fail；step-200 mean 比 step 50 低 5.651%；13.98 分钟 | 工程 Gate 通过；冻结分类为 `not_supported_no_material_late_degradation`；A1 随 step 增强但无 joint candidate，不解锁 full E/A2/A4/OOD |
| `P3-PHASE-E8-PREREG-v1` | 2026-07-29 | 3 / PRE-REGISTERED DIAGNOSTIC | A0 step 100/200；flows `11..74`；target 为 E.7 step-200 三条 worsened sample | 预算 1,536 forward、0 backward/optimizer/训练；双 32-flow block；20k paired bootstrap、16-comparison Bonferroni、20k five-flow resampling；**尚未运行** | 只区分 persistent target tail risk、five-flow variance 或 mixed；同一 cohort flow-level 精度诊断，不产生 checkpoint selection |

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

### `P3-PHASE-B-v1` 机器证据

- 独立分支 `feature/thought3-partial-future-adapter`；所有新 writer 强制
  `thought3` path component，并拒绝 Thought1/Thought2/third_party。
- 默认 Adapter 参数精确为 `1,371,137`；A0/A1/A2/A4 structure/count 相同，
  zero-gate A0/B0 action 逐元素一致。
- K1/K2/K4 cache 使用相同 base-sample seed 与 initial-noise hash；native schema
  `[48,2,14,28]`，支持原子 safetensors shard、resume 和多级 checksum。
- Adapter-only checkpoint 不包含 backbone；固定输入 round-trip 输出一致，
  mid-run resume 与 uninterrupted 的最终 Adapter semantic hash 一致。
- 60 个 Thought3 tests、235 个仓库全量 tests 全部通过；5 条 warning 均为测试
  环境 NVML 初始化限制。
- Thought1/Thought2 八个冻结 sentinel SHA-256 与 Phase A 前完全一致；
  `third_party/FastWAM` 无修改。
- 未加载官方大 checkpoint、未使用 GPU、未生成真实 cache、未训练真实模型。
  因此该 run 只能登记为 `TEST`，不能填写阶段三成功率表。

### `P3-PHASE-C-v1` 机器证据

- 实现 commit：
  `5c7d9a84a1058f1ca1d01641d02810eae102ea2a`；官方 checkpoint、
  stats、Fast-WAM commit 与 Phase A/B 冻结值一致。
- 数据 revision `117413dc0ca99c7cd64036c4eaa4a316c537d692`，archive
  SHA-256 `a21ae10171535585fb43e6405d9efa09ff38ef34689e4176428ca005af3a39ea`；
  本次只读取 `libero_goal` episode 0/frame 0。
- current latent `[1,48,1,14,28]`；K1/K2/K4 future 均为
  `[1,48,2,14,28]` BF16，且 initial-state SHA-256 完全相同。
- K1/K2/K4 单样本 sampler latency 为
  `120.34/165.62/325.30 ms`；这不是正式 latency 分布。
- video-only 对 upstream joint video max/mean diff `0/0`；
  zero-gate 对 current-only action bitwise equal。
- action loss `0.0002746765`，一次 backward；Adapter 1,371,137 参数，
  backbone gradient count 0，MoT hash 前后相同。zero-init 首步只有 gate
  非零梯度，Phase E 必须验证 gate 打开后其余参数获得非零梯度。
- 最高执行阶段 12.964 GiB；模型加载峰值 23.125 GiB；均低于 43 GiB。
- `future_frames` schema 拒绝、future-RGB mutation hash invariant；
  optimizer 未创建、optimizer step 0、真实 cache 0。
- 权威工件 SHA-256：status
  `581de5813e11fd19c8d7a1433c511c1a32e896900f62677ac3e47330d3f3bc33`；
  result
  `ccac9ac39fd7920dc89726313b89a3ae16ab71b5494b072d0b6c6ba6778d3f02`；
  log
  `f09670e9e5bd8bdb9ddd51653d71f7f5759c8f51cc3cb079a1c95993c5e648d2`。
- 完整解释见 [thought3_phase_c_report.md](thought3_phase_c_report.md)。

### `P3-PHASE-D-v1` 机器证据

- Phase C 收口 commit
  `f37a66bd43399bff637e2d2ffb1b9fd4103bd942`；Phase D 实现和运行 commit
  `02a010eb63897a97c911fb5f68e0bb209fe654ec`。运行前 Phase C 三个冻结 SHA
  全部匹配。
- 数据只来自标准 `libero_goal` task 0
  `open the middle drawer of the cabinet`。42 个 episode 先按 identity 分为
  37 train / 5 development，再选 32 个不同 episode；cache 内为 28/4。
- cache fingerprint
  `63a70e1af38f68bc894fc11d03c84f212e6c6328a5051256c9d045741156d9c5`；
  32 base samples、96 entries、12 safetensors shards、41 files、
  7,687,316 bytes。
- 所有 shard tensor shape 均为 latent `[8,48,2,14,28]` BF16、mask
  `[8,2,14,28]` bool；96 metadata rows、32 base sample identity 全部通过。
- 32/32 base sample 的 K1/K2/K4 共享 seed 与 initial-state hash；三个 K 的
  latent hash 对每个 base sample 都不同。
- no-op resume 为 built `0`、skipped `12/12`、`model_loaded=false`。临时副本
  单字节损坏触发 checksum mismatch，正式 shard hash 未改变。
- source audit 为 current camera frames `64`、future RGB `0`、actual future
  read `false`、action target read `false`、ground-truth future `false`。
- K1/K2/K4 的 32-sample sampling mean 为
  `127.54/186.62/362.99 ms`；generation loop `39.70 s`，不含模型加载吞吐
  `0.806 base sample/s`。模型加载 `888.44 s`，执行/加载峰值分别
  `12.677/23.125 GiB`。
- status/result/log SHA-256 分别为
  `d302cd63d3fd18161775f92ac3aa9d18e84842ee97b3316fe0f427df2e819baa`、
  `a636d649491ad9df67a1ea2cb91d8e9bf708784a410ba7b8304248f33ed1882d`、
  `97cdb718877a2c58a0a11352102d874b4c1b670b38ed090e90d60f91e5412d84`。
- optimizer 未创建、backward/optimizer step 为 0、训练未启动。完整解释见
  [thought3_phase_d_report.md](thought3_phase_d_report.md)。

### `P3-PHASE-E-v1/v2/v3` 机器证据

- 数据仍为 Phase D 的 32 个 base sample；训练 join 为 28 train / 4
  development、1024 行 action target、64 张当前相机帧、0 future RGB。
- A0/A1 初始化 Adapter semantic SHA 都是
  `77974a49c3d14fac142322244cc3613dccf0a329a25faa6e7053d99345ae627f`，
  trainable 参数都为 1,371,137。
- 第 1 step 只有 gate 非零 gradient；第 2 step 的 projector、attention 和
  non-gate 路径均出现 finite、nonzero gradient。
- v1 发现默认 CUDA backward 存在微小非确定性；v2 固定 cuBLAS workspace、
  deterministic algorithms、math SDP 并关闭 TF32/Flash SDP 后，两次独立
  2-step 重放逐位一致。
- v2 A0/A1 的 resumed 50→100 与 uninterrupted 0→100 最终 semantic SHA
  分别为 `67d0735f...ae00`、`f327127e...a8fb1`，每组内部完全相同。
- v2 A0 development `0.0184834→0.0183179`；A1
  `0.0184834→0.0185307`。v3 A0 fixed train probe
  `0.0015776→0.0015993`，未下降，故总 Gate fail-closed。
- model load / optimizer step peak 为 23,679.51 / 13,273.17 MiB；确定性
  step mean 约 662–666 ms。
- v2 status/log SHA 为
  `0631de121b683d0c78a2154476c2768d5a33ef1ae87f28c856716f95845fd56f` /
  `73f227a7db560521261e5985424383194d421b14bb277f9c30d0a76583cb29c1`；
  v3 为
  `ab19301eeaf572b6750389f4de0862d1641669da3b8c02089e3fa7ea6b65bc53` /
  `248fde0261029083ee5bcbca4e91ccd7ede96bb981eb2e56498f19421c679678`。
- Gate 在 frozen-after hash 前停止，因此该项仍未闭环。完整失败边界与下一步见
  [thought3_phase_e_report.md](thought3_phase_e_report.md)。

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
| 2026-07-28 / Phase E v1 preflight | split fingerprint lookup 与 progress callback 接口报错 | orchestration 误读 validator 字段；callback 签名与 trainer contract 不同 | commits `9b51179`、`2b42964`；257 tests 通过 | 否；均在 optimizer step 0 前停止 |
| 2026-07-28 / Phase E v1 A0 replay | resumed/uninterrupted 最终 Adapter SHA 不同 | CUDA backward reduction 从 step 2 产生极小非确定性 | v2 固定 cuBLAS/deterministic/math-SDP；两步 preflight 和 100-step replay SHA 完全一致 | 否；v1 独立目录保留为 invalid diagnosis |
| 2026-07-28 / Phase E v2/v3 loss gates | A1 development 未下降；固定 A0 train probe 也未下降 | 当前 100-step zero-gated 配方没有表现出稳定 loss 改善；不是 gradient 断链 | 总 Gate 保持 failed；下一步做单样本 fixed-noise overfit 和注入尺度诊断 | 否；没有 OOD/rollout，未扩 A2/A4 |
| 2026-07-28 / Gate E.3 v1 | A0/A1 initial probe 后，在首个 final checkpoint outcome 汇总报 `initial objective loss must be positive` | 一个 held-out draw 经 BF16 得到 `timestep=1000`；官方 scheduler weight 精确为 0，官方加权 loss 合法为 0；v1 非门控 `final/initial` ratio 错误假定严格正分母 | 冻结 v1 四个工件和 frozen-backbone SHA；v2 新 Run ID 显式记录 weight，零 loss 保留在 sample mean，只排除未定义 ratio；新增端点回归测试 | 否；0 optimizer/backward/rollout，frozen SHA 前后相同；但 v1 无 Gate 结论 |
| 2026-07-28 / Gate E.4 | 六条 diversified-flow 轨迹完整后命令返回非零 | 不是工程异常；全部 execution checks 通过，但六条 held-out reduction 仅 0.997%–1.948%，无 LR 达到共同 10%+6/8 | 保存 valid failed root result；不挑 A1@3e-4、不放宽门槛；下一诊断收缩到 objective aggregation/effective batch | 否；无 dev/OOD/success/rollout/future RGB，Fast-WAM SHA 前后相同 |
| 2026-07-28 / Gate E.5 | 六条 full-cohort 轨迹完整后命令返回非零 | 不是执行异常；120/120 execution、全部 paired/cross/frozen checks 通过，但 A0 在三档 LR 都未达 10%，所以无 A0/A1 共同 eligible LR | 冻结 valid failed root result；保留 A1@3e-4 的 19.668%/8-of-8 为探索性复验信号，不回改门槛或 selected LR | 否；0 dev/OOD/success/rollout/future RGB，Fast-WAM SHA 前后相同 |
| 2026-07-29 / Gate E.6 预注册 | E.5 后只复验 A0/A1@3e-4，存在结果后选择污染风险 | 明确登记为 post-selection sequential replication；冻结未使用 train 排序 9–16、slots 31001–32600、A1 绝对/A0 稳定/A1-vs-A0 三组门槛 | 配置、编排器、resume provenance、CLI、单卡 runner 和测试已实现；尚未运行、无 E.6 outcome | 否；预注册阶段只读 Phase D/E.5，运行仍锁定 dev/OOD/success/rollout/future RGB |
| 2026-07-29 / Gate E.6 结果 | 命令在两条轨迹完整后以 hard checks failed 返回 | 有效负门禁而非工程崩溃；唯一 false check 是 A0 non-worsened 4/8 < 6/8。A1 降 14.842%/7-of-8，A1 final mean 比 A0 低 13.815%/6-of-8 | 冻结 failed Run ID，不 resume、不放宽门槛；下一步先做只读 intermediate-checkpoint trajectory 诊断 | 否；400 updates/3,200 objectives 完整，全部 execution/paired/frozen/leakage checks 通过 |
| 2026-07-29 / Gate E.7 预注册 | 已知 E.6 step-200 A0 不稳定，若复用同一 flow 后挑中间 checkpoint 会产生结果后选择 | 将旧 flow `1..5` 降为 continuity-only；新 primary flow 固定为 `6..10`，冻结四种 A0 分类、最早稳定点比较和 post-run-only joint candidate | 8 个 checkpoint 的三文件 SHA、两套 probe identity SHA、只读/clean-repo gate、CLI、单卡 runner 和测试已实现；尚未加载模型或查看中间 outcome | 否；复用 E.6 cohort 但不训练、不消耗剩余 cohort，0 dev/OOD/success/rollout/future RGB |
| 2026-07-29 / Gate E.7 结果 | Primary A0 step 50/100 稳定、150/200 不稳定，但 pooled mean 继续下降；continuity 描述性 panel 则满足 late-overtraining rule | 不存在预注册要求的“non-worsened 至少下降 2 且 endpoint mean 变差”；五个 flow draw 对 8-sample stability/materiality 判定仍敏感 | 保留 primary 分类 `not_supported_no_material_late_degradation`，continuity 不覆盖主结论；不挑 step 100/200。下一步先预注册更大的全新 flow panel | 否；800 objectives 完整，0 backward/optimizer/checkpoint write，全部 frozen/provenance/RNG/leakage checks 通过 |
| 2026-07-29 / Gate E.8 预注册 | E.7 已知三条 worsened target 与 step-100/200 outcome，继续扩大 panel 存在新的结果后分析自由度 | 明确登记为 E.7 后序贯诊断；冻结 target、A0-only step 100/200、flows 11–74、两个 32-flow block、20k bootstrap/FWER 与三种互斥分类 | config、CLI、单卡 runner、父工件/checkpoint SHA、RNG/zero-weight、无训练/leakage gate 和测试已实现；尚未加载模型 | 否；不读取剩余 train cohort、dev/OOD/success/rollout/future RGB，0 optimizer/backward |

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
16. Phase D 已证明同一官方 checkpoint 能在单卡、当前观测唯一输入的条件下，
    为 32 个真实 training sample 生成可配对、可校验、可恢复的 K1/K2/K4
    future latent cache。
17. Phase D 的 12-shard cache 可检测单字节损坏并在 no-op resume 时避免模型
    重载；这是 cache 工程可靠性证据，不是 Adapter 收敛或 OOD 效果证据。
18. Phase E 真实训练证明 zero gate 第 1 step 后，第 2 step 的
    projector/attention 等非 gate 参数会获得 finite、nonzero gradient。
19. 在确定性 CUDA 协议下，A0/A1 的 50→100 resume 与独立 0→100 训练最终
    Adapter semantic SHA 完全一致；该结论只覆盖工程恢复，不代表模型有效。
20. Gate E.3 v2 的 320/320 held-out multi-flow forward 已证明：E.2 当前
    fixed-flow checkpoint 的 loss 改善没有达到跨 flow 的 `10% + 6/8` 稳定门槛；
    六条 hidden-scale/catastrophic/provenance 检查均通过。该结论只否定当前训练
    配方已足够稳定，不否定 future latent。
21. Gate E.4 在保持样本、LR、Adapter、预算、probe 和门槛不变时，将 optimizer
    改为 200 个唯一 paired flow slots；六条 held-out reduction 均转为正值，但
    只有 `0.997%–1.948%`，仍无共同 eligible LR。该结果说明 flow diversification
    有工程改善但不足以进入完整训练，不构成 future 模型效应。
22. Gate E.5 完成 1,200 次 full-cohort update 和 9,600 个 train objective，
    其中 `A1@3e-4` held-out loss 下降 19.668%、8/8 sample 不变差；但同 LR
    A0 仅下降 2.638%，故预注册共同门有效失败。该配对差只构成新 cohort
    序贯复验的工程依据，不构成 future 或 OOD 效果。
23. Gate E.6 在未使用的八条 train demonstration 上完成匹配 A0/A1@3e-4
    复验：A1 held-out loss 下降 14.842%、7/8 不变差，final mean 比 A0 低
    13.815%；但 A0 只有 4/8 不变差，预注册 Gate 有效失败。该结果证明审计链路
    能保留“信号复现但总门禁失败”的细粒度结果，不构成 OOD 因果结论。
24. Gate E.7 只读评估八个既有 checkpoint 的 800 个 objective：primary
    flow 上 A0 step 50/100 通过、150/200 因 5/8 失败，但 endpoint mean 相对
    step 50 仍低 5.651%，故不支持预注册的实质晚期退化模式；A1 在 150/200
    通过 absolute gate，但没有任何 joint candidate。该结论是 train-cohort
    optimization diagnosis，不是 future 或 OOD 效果。

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
7. E.7 预识别的 A0 step-200 三条 worsened sample 是 persistent tail risk
   还是 five-flow panel 方差。E.8 已预注册但未运行；无论结果为何，中间
   checkpoint 仍不是正式选择，A2/A4 和阶段三成功率继续锁定。

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
| `A0-null` | 0 | — | 1,371,137（设计/TEST） | — | — | — | 0 | — | — | — | — |
| `A1` | 1 | — | 1,371,137（设计/TEST） | — | — | — | — | — | — | — | — |
| `A2` | 2 | — | 1,371,137（设计/TEST） | — | — | — | — | — | — | — | — |
| `A4` | 4 | — | 1,371,137（设计/TEST） | — | — | — | — | — | — | — | — |

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
