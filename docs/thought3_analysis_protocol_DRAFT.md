# Thought3 正式分析协议（DRAFT）

状态：**DRAFT，未冻结，不是预注册**
更新时间：2026-07-27
冻结时机：Phase F 技术 pilot 结束后、任何 Phase G 正式结果产生前

## 1. 研究问题与允许结论

主问题：

> 相同 backbone、训练预算、动作去噪和评测 episode 下，正确的低成本
> future latent 相对 null-future Adapter 是否提高 task-equal OOD success？

允许结果包括正提升、无提升、负提升和 Adapter 忽略 future。本协议不预设方向。

## 2. 实验组

| 组 | K | 是否独立训练 | future |
| --- | ---: | --- | --- |
| B0 | 0 | 否 | 官方 Fast-WAM，无 Adapter |
| A0 | 0 | 是 | 全零 native-shape future，经同一 Adapter |
| A1 | 1 | 是 | 正确 K=1 |
| A2 | 2 | 是 | 正确 K=2 |
| A4 | 4 | 是 | 正确 K=4 |
| A-shuffle | 与 A-K 相同 | 否 | 同一 A-K checkpoint，在线错误 donor future |

A0/A1/A2/A4 必须具有相同 Adapter structure/init、optimizer、训练 step、
sample order、action flow seed 和 checkpoint selection rule。A-shuffle 不另训。

## 3. Estimand

### 3.1 主 estimand

```text
ΔOOD(K) =
task_equal_OOD_success(AK) - task_equal_OOD_success(A0)
```

主 K 集为 `{1,2,4}`。在冻结版本中必须预先指定一个 primary K 或对三个 K
做 multiplicity correction；当前尚未选择。

### 3.2 重要对照

- `A0 − B0`：额外参数与重新训练效应；
- `AK-correct − AK-shuffle`：样本对应 future 的增量；
- `AK-correct − AK-null/random`：动作反事实敏感性；
- `Clean − OOD`：鲁棒性下降；
- K 增加带来的 latency/memory Pareto trade-off。

## 4. 主、次指标

主指标：

- OOD task-equal success rate；
- `AK − A0` paired task-equal success difference。

次指标：

- Clean task-equal success；
- camera、robot-init、background、layout、lighting 分层 success；
- Clean→OOD absolute/relative drop；
- future sampling、Adapter、action、total latency 的 P50/P95；
- peak GPU memory；
- action chunk L1/L2、direction cosine、gripper change、EEF trajectory change；
- correct/null/shuffle/random/different-K action hash；
- future consistency 与 failure taxonomy。

Phase B mock success/latency 不进入任何上述结果。

## 5. 统计单位

1. episode 是原始 outcome 单位，但不能被当作完全独立统计样本；
2. 先在 task 内聚合 episode；
3. task 等权；
4. suite-stratified task bootstrap；
5. 默认至少 10,000 bootstrap replicates；
6. 所有 variant 使用相同 episode seed/job manifest 做 paired comparison；
7. 多 train seed 时先在每个 train seed 内聚合，再报告 seed 间分布。

二元 paired outcome 可补充 exact McNemar，但不能替代 task-cluster interval。

## 6. Counterfactual 判读

固定 current observation、language、proprio、action diffusion seed 和 checkpoint，
只替换 future：

1. correct；
2. null；
3. cross-task/episode shuffle；
4. random；
5. K=1/2/4。

判读：

- future 替换后动作接近 numerical replay floor：Adapter 未使用 future；
- shuffle 明显改变动作但 correct 不提高 success：模型使用 future，但信息质量或
  训练方式没有控制收益；
- correct 优于 shuffle 且相对 A0 提升：支持样本对应 future 有增量信息；
- A0 优于 B0、AK 不优于 A0：收益更可能来自 Adapter/重训，而非 future。

任务 success change 只能由在线 paired rollout 得出，不能从离线 mock action 推断。

## 7. Latency 与显存

每次 policy call 分开记录：

```text
preprocessing
current-state encoding
K-step future sampling
Adapter
20-step action denoising
total policy
```

future 保持 latent，不做 VAE decode。正式在线评测不能读取训练 cache。
correct/shuffle 均各自在线运行相同 K；不能让 shuffle 少算一次 future。

## 8. 模型选择纪律

允许用于选择：

- train loss；
-独立 development action loss；
- gate/gradient/NaN；
- ID 技术 pilot；
- Phase F 预注册的小型技术 cohort。

禁止用于选择：

- Phase G 正式 OOD outcome；
- Thought1/2 正式 rollout；
- LIBERO-Plus 训练；
- 看完某类别效果后修改 primary K、排除规则或停止时间。

checkpoint 选择规则、training seeds、formal job manifest、bootstrap seed、
primary metric 和 missing/exclusion rules 必须在 Phase G 前写入
`thought3_analysis_protocol_FROZEN.md`。

## 9. Missing 与无效 run

以下 run 整体无效，不能仅删掉异常 episode：

- backbone/stats/config hash 不同；
- frozen 参数有 gradient 或 hash 改变；
- action steps 不等于 20；
- online 读取训练 cache；
- K/seed/sample 对不齐；
- A-shuffle donor 同 task/episode；
- NaN/Inf；
- train/dev/test 泄漏；
- 正式结果产生后才冻结协议。

环境明确标为 unsupported 的 job 保留在 manifest 分母说明中，不伪装为失败或成功。
其他 runtime exception 单独报告，不能无声排除。

## 10. Multiplicity（冻结前待定）

冻结前必须从以下二选一：

1. 指定一个 primary K，其他 K 为 secondary；
2. 三个 `AK−A0` 同为 primary，使用 Holm correction。

类别×difficulty 分层默认探索性，除非在 frozen protocol 中预先列出有限的主 contrast。

## 11. 报告表

主表至少包含：

| Variant | K | Train seed | Params | Clean SR | OOD SR | AK−A0 | Correct−Shuffle | Future P50/P95 | Total P50/P95 | Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

主图以 K 为横轴，同时画 OOD/ID success、future-action sensitivity、future
latency、total latency 和 peak memory，不只展示最优 K。

## 12. 冻结前未决项

- primary K 或 Holm 策略；
- formal train seed 数和具体 seed；
- Phase G episode/job manifest；
- checkpoint selection metric 与 patience；
- numerical replay floor；
- smallest effect of interest；
- failure video 人工标注预算；
- bootstrap seed；
- official real-data inventory hash；
- Phase F→G 技术门槛的量化阈值。

这些项未确定前，本文只能保留 `DRAFT` 文件名。
