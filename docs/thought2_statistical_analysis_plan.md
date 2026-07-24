# 阶段二统计分析计划（DRAFT，尚未冻结）

更新日期：2026-07-23

## 1. 资格与目的

本文件把“未来—动作—实际变化一致性”转成可执行 estimand。当前状态是
`DRAFT_NOT_FROZEN`：类别范围、human-review 预算和正式 static threshold 尚未
确定，不能把本文件称为 preregistration。

阶段二只回答关联问题：

> 同一官方 Fast-WAM checkpoint 产生的 unconditional future，与受保护原动作实际
> 造成的视觉变化有多一致；这种一致性是否在 OOD 或失败 episode 中下降？

它不回答“动作读取了未来”或“future 错误导致失败”。显式 future 的因果增益只由
阶段三 B0/A0/A1/A2/A4 配方匹配实验回答。

## 2. 数据来源与不可替代关系

| 数据 | 权威用途 | 不能替代 |
| --- | --- | --- |
| 阶段一 800 Clean + 6,771 OOD | ID/OOD success rate、drop、action latency、failure prevalence | 阶段二 sampled cohort |
| 阶段二 outcome-blind cohort | future consistency 的 ID/OOD 估计 | 阶段一总体成功率 |
| 200-job static/no-op calibration | 冻结 representation motion threshold | 已观察 future pilot |
| Blind human subset | 视觉质量与局部执行标签 | 自动指标或全部 episode |
| Outcome-aware matched case-control | 失败机制解释 | prevalence/总体效应 |

正式分析必须固定 checkpoint、Fast-WAM/upstream commits、source job-manifest
hash、cohort ID、diagnostic protocol fingerprint、static threshold version 和
blind packet ID。

## 3. 实验单位与层级

数据不是 532 个独立 OOD 点：

```text
suite
  └── task（40 个独立 cluster：4 suite × 10 task）
       ├── Clean episode（5 个，含 index 0 anchor）
       └── OOD category × difficulty cell（每 supported cell 1 个）
            └── episode
                 └── probe（最多 2 个）
                      └── aligned future frame（offset 4/8）
```

主分析遵循三次聚合：

1. 同一 probe 的 aligned frame 按现有 metric 定义聚合。
2. 同一 episode 的有效 probe 先平均，episode 权重恒为 1。
3. 同一 task 的 Clean episode 或 OOD cell 再平均，最后 40 个 task 等权。

禁止把 aligned frame、probe 或 OOD variant 直接当作独立统计样本。Clip-weighted
结果只能作为诊断附录。

## 4. 主要 endpoint

### 4.1 自动主要 endpoint

推荐冻结：

```text
episode-level future_latent_cosine_distance
```

它是冻结 VAE 对每个精确对齐 predicted/actual frame 独立重编码后的 cosine
distance，越低表示越一致。它不是 diffusion likelihood、像素误差、光流，也不是
动作向量与视频向量的 cosine。

主 estimand：

```text
对每个 task：
  mean(OOD supported-cell episode cosine distance)
  - mean(5 个 Clean episode cosine distance)

再对 40 个 task 等权平均
```

正值表示 OOD 一致性更差。总体主 contrast 只检验一次，不因观察方向而换 endpoint。

### 4.2 关键 secondary endpoints

- `future_latent_l1`：OOD−ID，正值为更差。
- `motion_direction_cosine`：ID−OOD，正值为 OOD 更差；只有 predicted 与
  actual motion energy 都高于**正式冻结** static threshold 时才进入
  decisive 分母。
- `predicted_motion_energy`、`actual_motion_energy` 和 ratio：只描述动态范围，
  不单独定义“future 正确”。
- Human `future_actual_agreement`、`future_goal_progress`、
  `future_physical_plausibility`：报告完整类别比例，不把 `static/uncertain`
  强行塞进好/坏二元序。
- Generation latency 与 full diagnostic latency：分开报告 episode-level
  p50/p95；模型冷启动另表，不混进单 probe latency。

连续值始终保留。Static threshold 只能增加资格/敏感性列，不允许覆盖原始
diagnostics JSONL。

## 5. “动作—未来一致性”的操作化边界

7-DoF action delta 与视觉 latent 不在同一向量空间，当前项目没有把二者直接做
cosine。当前可识别的量是：

```text
预测视觉变化
    vs
受保护原动作执行后实际实现的视觉变化
```

因此 `motion_direction_cosine` 应写成
“predicted-vs-realized visual change direction consistency”，不能写成
“action-future cosine”。Human `action_execution_quality` 也只覆盖 offset
0/4/8 的 0–0.4 s 可见窗口，不能区分 action selection 与低层 controller 故障。

若论文必须声称直接 action-future direction，需要另加有校准依据的 inverse
dynamics、robot keypoint projection 或共享 action-effect encoder；在此之前该
claim 保持未实现。

## 6. ID/OOD contrast 与配对敏感性

### 6.1 主要总体 contrast

- 每个 task 内先平均所有选中的 OOD supported cells。
- 每个 task 内 Clean 使用 5 个 outcome-blind episode 的平均。
- 40 个 task 等权；suite 不按可用 cell 数加权。
- 95% CI 使用 suite-stratified task cluster bootstrap：每个 suite 内有放回抽
  10 个 task，再合并四个 suite。
- Bootstrap 次数建议 `10,000`，seed 建议 `20260725`，必须在 formal analysis
  前进入 manifest。

### 6.2 精确 anchor sensitivity

另报告 OOD episode index 0 与同 suite/task 的 Clean index 0：

```text
OOD metric - Clean-anchor metric
```

同一个 Clean anchor 会被多个 OOD cell 复用，因此仍按 task cluster 聚合/重采样，
不能把每个 cell 当独立 pair。该结果是 sensitivity，不替代主要 5-Clean baseline。

### 6.3 类别和难度

category、difficulty、suite contrast 均为 secondary：

- category 内仍先 task-level 聚合；
- unsupported-only cell 不计失败，也不做零填充；
- 每张表同时报告 eligible tasks/cells/episodes/probes/aligned frames；
- 对预先声明的 category×metric family 使用 Benjamini–Hochberg FDR
  `q=0.05`，同时保留 raw p/CI；不按显著性筛表。

## 7. Success/failure 关联

阶段二 outcome 分析使用**同一次 diagnostic rerun**的 success，因为 future、
actual frames 和 outcome 必须来自同一轨迹；同时必须报告它与阶段一 source 的：

- action exact-match 分母和最大绝对差；
- success/termination match 分母；
- 不一致 episode 的完整列表。

建议冻结以下最低资格：

- 每个比较组至少 20 success、20 failure；
- 至少 10 个 task 同时贡献两类 outcome；
- 否则只报告分母和描述值，不给 inferential p-value。

可报告 success−failure 的 task-clustered standardized consistency difference，
但必须写 `causal_interpretation_allowed=false`。阶段一完成后建立的 matched
case-control 只能解释失败机制，不能估计 success/failure prevalence。

## 8. Human-review 分析

正式 blind subset 必须在 label/metric 解盲前固定：

- 独立 review seed；
- `max_cases_per_job=1`；
- human case budget；
- packet manifest/hash；
- 两名或以上 reviewer ID 规则。

流程：

1. 两人独立完成 public packet。
2. `analyze-blind-review` 在不读取 private key 的条件下验证完整 case set。
3. 对每个字段报告 nonmissing 和 decisive 分母、exact agreement、pairwise
   Cohen's κ；退化边际保持 `undefined`。
4. 保存两份 raw、blind agreement、解盲映射、讨论记录和 adjudicated 文件；
   任一版本不得覆盖上一版本。
5. 解盲后的条件/outcome 表以 adjudicated 标签为主，同时附两名 raw reviewer
   sensitivity。

Reviewer 能从画面推断相机/光照或明显成败，因此论文用
“label-blind visual review”，不能声称 fully blinded。

## 9. 缺失、异常和资格规则

| 情形 | 处理 |
| --- | --- |
| 上游没有 category/level variant | structural unsupported；不进入分母 |
| Diagnostic technical exception | 记录并按冻结 retry 规则重跑；不能填成差一致性 |
| Episode 只有 1 个有效 probe | 用该 probe，显式报告 |
| Episode 无有效 aligned probe | 对该 endpoint 缺失，不插补 |
| Static threshold 未冻结 | direction/static decisive 主表禁止生成 |
| Action hash 改变 | protocol violation；episode 不进入正式一致性表 |
| Source outcome mismatch | 保留 consistency；从 outcome-association 主分析排除并单列 |
| Human missing | 从 nonmissing/decisive pair 排除，保留 missing 分母 |
| Human `uncertain` | nonmissing 保留；decisive 排除 |

所有 retry attempt 保留；只允许因技术异常重试，不允许因 success/failure 或 metric
方向重跑。不得基于中间效应大小提前停止 frozen cohort。

## 10. 样本量与 power 状态

当前 200 Clean + 532 OOD 是**coverage-driven design**，不是已经完成的 power
analysis。真正的独立 cluster 主要是 40 个 task，不能用 `n=732` 的独立样本公式
夸大 power。

冻结前应使用不进入主分析的 20–50 条跨 suite technical/power pilot，估计：

- task-level contrast variance；
- task 内/episode 内相关性；
- technical missing rate；
- success/failure imbalance；
- human label prevalence 与 review 时长。

随后对“40 task、当前 cell layout、suite-stratified cluster bootstrap”做 simulation
power 或报告 minimum detectable effect。当前 5-episode camera/easy pilot 太小，
且 task/outcome 混杂，只能估算链路和成本，不能提供可信 power。

## 11. 停止与冻结门禁

正式 analysis protocol 至少固定：

- 五类 732，或四类 612/622；
- primary endpoint 与 sign；
- bootstrap seed/次数；
- static threshold artifact/hash；
- outcome 最小分母；
- human case budget/reviewer 数；
- secondary family 与 multiplicity rule；
- retry 上限和技术失败定义；
- code commit、所有 input manifest/hash、output namespace。

只有在项目与上游 clean、outcome-blind manifests `frozen=true`、static threshold
人工冻结后，才能把状态改为 `FROZEN_BEFORE_ANALYSIS`。冻结后不因结果方向修改；
必要变更必须产生新版本并在论文中列出 deviation。

## 12. 论文表图映射

| Artifact | 主内容 |
| --- | --- |
| Table 1 | 阶段一完整 ID/OOD success/drop/latency 分母 |
| Table 2 | 阶段二 task-equal ID/OOD consistency contrast 与 CI |
| Figure 1 | category/difficulty 的 task-level consistency forest plot |
| Figure 2 | 连续 consistency 与 outcome 的关联；明确非因果 |
| Figure 3 | Blind human label 分布与 reviewer agreement |
| Appendix | anchor sensitivity、clip-weighted、static-threshold sensitivity、全部 missing/unsupported |

阶段三 K=0/1/2/4 的成功率—延迟曲线使用另一套训练/评测 manifest，绝不追加到本
阶段二表中。

## 13. 待用户冻结的决策

| 决策 | 推荐草案 | 当前状态 |
| --- | --- | --- |
| 扰动范围 | 五类 732；若严格四类，优先保留 object-layout 得到 612 | 待确认 |
| 自动 primary | `future_latent_cosine_distance` | 待确认 |
| Bootstrap | suite-stratified task bootstrap，10,000 次，seed `20260725` | 待确认 |
| Human budget | 先做跨 suite power/timing pilot，再固定；不从结果反推 | 待确认 |
| Outcome inferential gate | ≥20 success、≥20 failure、≥10 mixed-outcome tasks | 待确认 |

