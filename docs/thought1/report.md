# 思考点一正式报告：Fast-WAM 的 LIBERO-Plus 环境 OOD 鲁棒性

- 报告日期：2026-07-26
- 正式 Run ID：`P1-FORMAL-v1`
- 实验状态：**800 Clean + 6,771 OOD runnable rollout 已全部完成并通过完整性审计**

## 摘要

本实验冻结同一份官方 Fast-WAM `libero_uncond_2cam224` checkpoint，在标准
LIBERO 与 LIBERO-Plus 官方环境扰动变体上评测 4 个 suite、40 个基础任务。

主结果是：

- Clean：`778/800 = 97.25%`，95% row-bootstrap CI
  `[96.00%, 98.38%]`。
- OOD：`3,230/6,771 = 47.70%`，95% row-bootstrap CI
  `[46.55%, 48.90%]`。
- Clean→OOD 绝对下降：**49.55 个百分点**。
- 相对下降：**50.95%**，即 OOD 条件下只保留约 49.05% 的 Clean 成功率。
- 7,571 个真实 rollout 均正常结束，`0 exception`；另有 68 条官方空分层
  `skipped` 审计记录，不进入成功率分母。

最稳定的科学结论是：**该 release checkpoint 在标准 LIBERO 上接近饱和，但对
LIBERO-Plus 环境 shift 明显不鲁棒；相机视角是最强且跨 suite 稳定的脆弱因素，
光照变化整体影响最小。**

这个结果不能证明未来想象能够修复失败，也不能称为 unseen-object、unseen-task、
跨平台或真机泛化结论。

## 1. 研究问题与正式协议

思考点一回答的是：

> 在 checkpoint、dataset stats、动作接口和基础任务保持固定时，从标准 LIBERO
> 切换到 LIBERO-Plus 官方环境扰动变体，Fast-WAM 成功率下降多少，且哪些扰动、
> 难度和任务最敏感？

协议如下：

- Clean：4 suite × 10 task × 20 个初始化 index/seed，共 800 次。
- OOD：五类官方预生成 variant 使用 `all_once`，每个 runnable variant 运行
  1 次，共 6,771 次；不执行 `variant × 20` 的重复采样。
- 五类扰动：
  `camera_viewpoints`、`light_conditions`、`background_textures`、
  `robot_initial_states`、`objects_layout`。
- 难度映射：官方 1–2 为 easy、3 为 medium、4–5 为 hard。
- `libero_spatial/object/goal` 最多 400 个 policy step，
  `libero_10` 最多 700 个 policy step；每个 episode 前有 30 个 settle step。
- success 使用环境官方成功判定；`max_steps` 是策略失败，exception 也计失败，
  skipped 不进入分母。

主成功率是 **variant/episode-weighted** 描述统计。由于每个官方 OOD variant
只运行一次，CI 描述的是已评测行的重采样不确定性，不等同于跨任务、跨世界分布
或真机泛化的不确定性。第 7 节另给任务聚类敏感性分析。

## 2. 可复现性与完整性审计

### 2.1 固定环境

| 项目 | 正式记录 |
| --- | --- |
| 项目 commit | `575ba8fcd89f6baf801190fcb8127142ba0406c5`，clean |
| Fast-WAM commit | `45d8e1458921d83f8ad6cf9ce993d371208dabd0`，clean |
| LIBERO commit | `8f1084e3132a39270c3a13ebe37270a43ece2a01`，clean |
| LIBERO-Plus commit | `4976dc30028e805ff8094b55501d532c48fec182`，clean |
| checkpoint SHA-256 | `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579` |
| dataset stats SHA-256 | `30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638` |
| Python / PyTorch / CUDA | 3.10.20 / 2.7.1+cu128 / CUDA 12.8 |
| GPU / driver | 3 × NVIDIA GeForce RTX 4090 / 580.173.02 |
| 推理 | bf16、TF32 enabled、每 GPU 1 worker |
| future imagination | `false`；策略为官方 uncond Fast-WAM |

八个 source manifest 和 7,639 条结果全部记录同一 checkpoint、项目 commit 与
三个上游 commit，没有混合策略或 dirty source。

### 2.2 作业、轨迹和媒体

| 审计项 | 结果 |
| --- | ---: |
| 磁盘结果行 | 7,639 |
| 真实 attempted / completed | 7,571 / 7,571 |
| Clean / OOD attempted | 800 / 6,771 |
| 预期 skipped | 68 |
| success / max_steps / exception | 4,008 / 3,563 / 0 |
| 唯一 job ID / 重复 ID | 7,639 / 0 |
| manifest 缺失结果 / 多余结果 | 0 / 0 |
| action trace | 7,571，和 attempted job 一一对应 |
| action step | 2,399,314 |
| failure video | 3,563，和失败 job 一一对应 |
| 空视频 / 缺失视频 | 0 / 0 |

三张卡在八个阶段都产生结果；每个 source 的 manifest、raw worker JSONL 和聚合
结果 job ID 集合完全相等，没有分片遗漏或 resume 重复。

全量 trace 流式审计还确认：

- 0 个 JSON 解析错误、空 trace、错误动作维度或非有限动作；
- 0 个 episode 的前 6 个运动动作全为零；
- trace 行数全部等于 `steps - 30`；
- 末端执行器首末位移范围 `0.0385–1.0377 m`，中位数 `0.3130 m`；
- 12 个平移动作分量轻微超出名义 `[-1, 1]`，最大绝对值 `1.0181`，
  占 2,399,314 个 action step 的约 0.0005%；robosuite controller 在执行前
  会裁剪到输入范围。它们没有造成 exception，也不是 OOD 特有现象。

因此，3,563 个失败都是环境在最大步数内未判定成功，而不是 CUDA、EGL、NaN、
空动作、静止机器人或结果落盘故障。

## 3. 总体结果

| Condition | 磁盘行 | Attempted | Success | Failure | Success rate | 95% row-bootstrap CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Clean | 800 | 800 | 778 | 22 | **97.25%** | [96.00%, 98.38%] |
| OOD | 6,839 | 6,771 | 3,230 | 3,541 | **47.70%** | [46.55%, 48.90%] |

主估计的绝对下降为 `97.25% - 47.70% = 49.55 pp`，相对下降为
`49.55 / 97.25 = 50.95%`。独立 row bootstrap 对绝对下降给出的补充区间为
`[47.87, 51.22] pp`。

这说明模型并非“任务本身不会做”：Clean 已达到 97.25%，但同一 checkpoint
面对官方环境变化后失败率从 2.75% 增至 52.30%。

## 4. 按 suite 分层

| Suite | Clean success / N | Clean SR (95% CI) | OOD success / N | OOD SR (95% CI) | Absolute drop |
| --- | ---: | --- | ---: | --- | ---: |
| `libero_spatial` | 197 / 200 | 98.50% [96.50%, 100.00%] | 926 / 1,661 | 55.75% [53.22%, 58.16%] | 42.75 pp |
| `libero_object` | 197 / 200 | 98.50% [96.50%, 100.00%] | 1,132 / 1,742 | 64.98% [62.86%, 67.16%] | 33.52 pp |
| `libero_goal` | 193 / 200 | 96.50% [94.00%, 99.00%] | 531 / 1,681 | 31.59% [29.27%, 33.85%] | 64.91 pp |
| `libero_10` | 191 / 200 | 95.50% [92.50%, 98.00%] | 641 / 1,687 | 38.00% [35.80%, 40.37%] | 57.50 pp |

`libero_object` 的 OOD 保持率最高，`libero_goal` 的绝对下降最大，
`libero_10` 次之。suite 差异不能简单解释成对象、目标或长时序的单一因果效应：
任务语义、variant 组成和最大步数同时不同。它更适合说明**基础任务与环境 shift
存在很强交互**。

## 5. 按扰动类别分层

### 5.1 主统计与 task-macro 敏感性分析

| Perturbation | Success / N | Variant-weighted SR (95% CI) | Task-macro SR (task bootstrap CI) | Task-mix adjusted Clean→OOD drop |
| --- | ---: | --- | --- | ---: |
| Camera viewpoints | 242 / 1,599 | **15.13%** [13.38%, 16.95%] | 18.79% [12.16%, 27.04%] | **81.97 pp** |
| Robot initial states | 664 / 1,550 | 42.84% [40.39%, 45.23%] | 44.33% [35.15%, 52.72%] | 54.31 pp |
| Background textures | 554 / 1,076 | 51.49% [48.51%, 54.46%] | 53.61% [42.36%, 64.66%] | 45.00 pp |
| Objects layout | 934 / 1,525 | 61.25% [58.75%, 63.67%] | 61.72% [53.23%, 70.05%] | 36.02 pp |
| Light conditions | 836 / 1,021 | **81.88%** [79.43%, 84.04%] | 84.19% [75.47%, 91.95%] | **15.20 pp** |

`Task-mix adjusted` 先按每个 OOD 类别在 40 个基础任务上的 variant 数量，对相应
Clean task success rate 加权，再计算下降，避免类别的任务组成不同直接造成偏差。
它与直接使用总体 Clean 97.25% 得到的排序一致。

最可靠的类别结论是：

1. **Camera viewpoints 是首要脆弱点。** 它在四个 suite 内都是最低成功率类别。
   任务配对的 camera−robot-init 差为 `-25.53 pp`，task-bootstrap CI
   `[-34.16, -16.92] pp`。
2. **Light conditions 整体最容易。** 任务配对的 light−layout 差为
   `+22.47 pp`，CI `[14.35, 30.60] pp`。
3. 中间三类的精确名次应谨慎：layout−background 的 task-bootstrap CI
   `[-0.61, 17.49] pp` 跨 0；不能仅凭 micro point estimate 声称 layout
   在所有任务上必然优于 background。

### 5.2 类别与 suite 的交互

| Suite | Background | Camera | Light | Layout | Robot init |
| --- | ---: | ---: | ---: | ---: | ---: |
| `libero_spatial` | 75.58% | **14.63%** | 93.15% | 66.75% | 42.00% |
| `libero_object` | 76.61% | **23.48%** | 97.64% | 75.68% | 63.82% |
| `libero_goal` | 34.52% | **7.35%** | 82.28% | 41.65% | 23.72% |
| `libero_10` | 24.91% | **15.27%** | 52.55% | 62.50% | 42.24% |

相机脆弱性跨 suite 稳定；其他类别明显依赖任务。例如背景纹理在
`libero_object` 为 76.61%，在 `libero_10` 只有 24.91%；光照在前三个 suite
相对稳健，但在 `libero_10` 降至 52.55%。因此不能把“光照鲁棒”外推为所有任务
都不受光照影响。

## 6. 难度效应

### 6.1 预注册 easy/medium/hard

| Difficulty | Success / N | Success rate | 95% row-bootstrap CI |
| --- | ---: | ---: | --- |
| Easy | 1,532 / 2,561 | 59.82% | [57.83%, 61.73%] |
| Medium | 760 / 1,535 | 49.51% | [47.04%, 52.05%] |
| Hard | 938 / 2,675 | 35.07% | [33.23%, 36.90%] |

从 easy 到 hard 下降 `24.75 pp`。在同时拥有三个等级的 146 个
`task × category` cell 上做等权配对敏感性分析后，macro SR 为
easy `57.45%`、medium `50.41%`、hard `37.03%`；easy−hard 差
`20.42 pp`，task×category bootstrap CI `[14.92, 25.59] pp`。

### 6.2 每类扰动内的难度趋势

| Perturbation | Easy | Medium | Hard |
| --- | ---: | ---: | ---: |
| Background textures | 69.64% | 51.33% | 22.01% |
| Camera viewpoints | 26.83% | 11.38% | 7.69% |
| Light conditions | 85.71% | 85.25% | 77.61% |
| Objects layout | 68.92% | 61.40% | 52.01% |
| Robot initial states | 59.07% | 47.59% | 26.26% |

五类扰动在粗粒度上均满足 `easy ≥ medium ≥ hard`，所以总体难度趋势不是只由
类别比例造成。但官方 1–5 级并非每一类都严格单调：camera 的 level 5 略高于
level 4，light/layout 的 level 4 也略高于 level 3。difficulty 更适合作为
**粗分层标签**，不应解释成精确等距的连续强度。

## 7. 任务异质性

40 个基础任务的 OOD task-macro SR 为 48.03%，中位数 51.71%，范围从
4.57% 到 92.63%。其中 5 个任务低于 10%，8 个低于 25%，6 个达到或超过 75%。

最低的五个任务如下：

| Suite / task | Clean SR | OOD success / N | OOD SR | Drop |
| --- | ---: | ---: | ---: | ---: |
| `libero_10/4` 两个杯子分别放到左右盘子 | 85.00% | 9 / 197 | 4.57% | 80.43 pp |
| `libero_goal/1` 把碗放到炉子上 | 100.00% | 6 / 108 | 5.56% | 94.44 pp |
| `libero_10/6` 杯子放盘子且布丁放右侧 | 85.00% | 16 / 193 | 8.29% | 76.71 pp |
| `libero_goal/3` 打开顶层抽屉并放入碗 | 90.00% | 19 / 212 | 8.96% | 81.04 pp |
| `libero_spatial/1` 拿起 ramekin 旁黑碗并放盘子 | 100.00% | 12 / 132 | 9.09% | 90.91 pp |

最高的任务是 `libero_object/1` 的 cream-cheese-to-basket：
`88/95 = 92.63%`。同一 suite 内也同时存在高低极端，说明一个总体 OOD 数字无法
替代 task-level 报告。

这些结果支持“任务与扰动有强交互”，但尚不能自动回答失败究竟来自目标检测、
相机几何、抓取、接触、长时序误差还是成功判定。所有失败视频已保存，因果化的
失败 taxonomy 仍需按预定义抽样进行人工复核。

## 8. 配对结果应该怎样解释

机器报告给出：

- Clean-success / OOD-failure：3,541
- Both-success：3,230
- Clean-failure / OOD-success：0
- Both-failure：0

这 6,771 个展开比较只对应 **40 个唯一 Clean anchor**：每个基础任务的
`episode_index=0`。这 40 个 anchor 恰好全部成功，而每个 anchor 对应
95–225 个 OOD variants。

因此该四格表可以解释为：

> 在标准环境中成功的同一任务/初态 anchor，切换到各官方 OOD variant 后，
> 3,541 个 variant 失败、3,230 个仍成功。

它不能解释为 6,771 组相互独立的 Clean/OOD trial，也不适合直接做普通
McNemar 推断。主结论仍使用 800 条 Clean 基线与 6,771 条 OOD variant；
补充的 40-task 等权分析为：

- Clean task-macro：97.25%，CI `[95.88%, 98.50%]`
- OOD task-macro：48.03%，CI `[40.38%, 55.65%]`
- 配对 task-macro drop：49.22 pp，CI `[42.14, 56.39] pp`

任务聚类区间更宽，但结论方向和效应量与主统计一致。

## 9. 运行成本

| 指标 | 实测值 |
| --- | ---: |
| 首条到末条结果跨度 | 44.26 h |
| Clean / OOD episode GPU-hours | 6.95 / 116.25 h |
| 合计 episode GPU-hours | 123.20 h |
| 三卡窗口内近似 episode 利用率 | 92.8% |
| Clean / OOD 平均 episode 时长 | 31.27 / 61.81 s |
| Clean / OOD 平均总步数 | 183.05 / 366.27 |
| action-chunk 推理延迟 mean | 970.07 ms |
| action-chunk 推理延迟 p50 / p95 | 969.51 / 978.18 ms |
| 每 worker 最大记录显存 | 23,814.42 MB |
| 正式输出占用 | 约 1.8 GiB |

推理延迟是一次 action-chunk 规划调用，不是单个控制 action 的独立延迟。
三卡实际在约 44.3 小时内完成，低于 pilot 阶段保守预留的 60–72 小时。

## 10. 可以写与不能写的结论

### 已由 `P1-FORMAL-v1` 支持

1. 官方 Fast-WAM release 在标准 LIBERO 上成功率为 97.25%，但在所评
   LIBERO-Plus variant 上降至 47.70%，绝对下降 49.55 pp。
2. 相机视角是最严重且跨 suite 稳定的脆弱因素；光照整体最轻。
3. easy→medium→hard 在五类扰动内均呈粗粒度下降，说明 severity 标签具有
   描述性区分度。
4. suite、任务与扰动之间存在强交互，任务 OOD SR 跨越 4.57%–92.63%。
5. 全量计算链路没有 exception、遗漏、重复、NaN 或静止策略；失败是
   max-steps 任务失败，而不是评测基础设施故障。

### 本实验仍不支持

1. “未来想象能够/不能够改善 OOD”：当前 uncond action 不读取预测未来。
2. unseen-object 或 unseen-task：release 训练配置已包含四个评测 suite。
3. cross-platform 或真机结论：当前只有 LIBERO 系仿真。
4. 自动化失败机制归因：尚无正式人工视频 taxonomy。
5. 将 row-bootstrap CI 解释为对所有任务、所有环境或现实世界的置信区间。

## 11. 后续工作

1. 对 3,563 个失败视频做预定义的 suite/category/level/task 分层抽样和双人
   标注，报告实际 reviewed 分母与一致性，不能凭少量印象外推。
2. 阶段二 future-consistency 分析必须单独记录为 observational diagnostics；
   阶段一失败本身不能证明未来有用。
3. 若要做 future/no-future 因果比较，需要相同数据、初始化、训练预算和多个
   seed 的 recipe-matched checkpoint。
4. 若要证明 unseen object/task/platform，需要重新定义互斥训练/测试 split
   或跨平台接口，不能复用本次四个已见 suite 的结果。

## 12. 权威工件与哈希

- 机器生成报告：
  [combined/summary/report.md](../../outputs/thought1/fastwam/combined/summary/report.md)
- 机器指标：
  [combined/summary/metrics.json](../../outputs/thought1/fastwam/combined/summary/metrics.json)
- 逐条结果：
  [combined/summary/episode_results.jsonl](../../outputs/thought1/fastwam/combined/summary/episode_results.jsonl)
- 正式 manifest：
  [combined/experiment_manifest.json](../../outputs/thought1/fastwam/combined/experiment_manifest.json)

| Artifact | SHA-256 |
| --- | --- |
| `combined/experiment_manifest.json` | `57dd93f51a2491423f1b14f0d90523f219218698e231a133dcef114caca132ee` |
| `combined/summary/metrics.json` | `0aa1173038a1c37d37123570a83ff9f08667490e3f94276345c802151897dbb5` |
| `combined/summary/report.md` | `889d567e4882b9982fb2121788dbbacdf983e1556faf8e4f9bb5a29768f8e137` |
| `combined/summary/episode_results.jsonl` | `2f478526ab66a3eacc42e14196a5dbaf13cec6e282230915c2c74973e62cf5e9` |

机器报告是原始主统计的权威来源；本文增加了 task-macro、task-mix 和
task×category 聚类敏感性分析，并明确标注其补充性质。
