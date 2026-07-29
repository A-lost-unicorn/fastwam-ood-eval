# Thought3 Gate E.8：A0 大规模 Flow-Variance Replication 预注册

状态：**PRE-REGISTERED PROTOCOL / RUN COMPLETED**

冻结日期：2026-07-29

运行完成：2026-07-29。工程 Gate 通过，预注册主要分类为
`mixed_or_inconclusive`；本文件继续保留运行前规则，实际数字、工件 SHA 和
结论边界见 [结果报告](thought3_phase_e8_report.md)。

> 本协议在查看 E.7 全部结果后建立。E.7 的 primary/continuity trajectory、
> step 100/200 outcome 和三条 step-200 worsened sample 均已知。E.8 是明确
> result-conditioned 的序贯工程诊断，不是独立确认性实验，也不是 checkpoint
> 选择实验。

> 预注册权威版本是首次同时包含本文、冻结配置、编排器、runner 和测试的 clean
> git commit。正式运行会记录精确 project HEAD，并在 project 或 FastWAM
> worktree dirty 时拒绝启动。预注册阶段不加载模型、不运行 GPU probe。

## 1. 研究问题

E.7 的两个五-flow panel 都显示：

- A0 step 50/100 通过原稳定性门槛；
- A0 step 150/200 未通过逐样本稳定性门槛；
- primary step 200 为 `5/8` sample 不变差，但 pooled mean 继续改善；
- continuity step 200 为 `4/8`，pooled mean 相对 step 50 仅恶化约
  `0.154%`。

两个 panel 对“是否达到实质晚期退化”给出不同分类，说明每条 sample 只有五个
flow draw 时，sample mean 与 `6/8` Gate 可能受 Monte Carlo panel 方差影响。

E.8 只回答：

> E.7 primary step 200 中预先识别的三条 A0 worsened sample，是否在规模更大、
> 完全未使用的 flow panel 上持续恶化；还是原来的 `5/8` 主要来自五-flow
> 小 panel 的统计方差？

这里的“尾部风险”专指：**同一八条 demonstration 中，E.7 预识别的少数样本在
大量新 action noise/timestep draw 上仍有可重复的 mean action-loss 恶化**。
它不是 OOD failure tail，也不是对 LIBERO demonstration 总体的外推。

## 2. 不能回答的问题

E.8 不能回答：

- A1/future 是否改善动作或 OOD success；
- step 100 或 step 200 哪个可以进入论文主实验；
- A0 风险是否能推广到未使用的 demonstration cohort；
- 风险来自 optimizer、表示、动作尺度还是任务语义；
- A2/A4 是否应解锁；
- ID/OOD latency–success Pareto。

E.8 只读取标准 LIBERO train cohort 的当前观测、动作 target 和既有 A0
Adapter checkpoint。它不读取 development、OOD、success、rollout outcome 或
真实 future RGB。

## 3. 已知信息与结果后选择披露

在冻结 E.8 前已知：

- E.7 工程 Gate 完整通过；
- E.7 primary 分类为
  `not_supported_no_material_late_degradation`；
- E.7 没有 joint diagnostic candidate；
- E.7 primary A0 step 100 通过原 Gate，step 200 因 `5/8` 失败；
- continuity A0 step 100 通过，step 200 因 `4/8` 失败；
- step 100 是在看到 E.7 后选作 early comparator；
- step 200 是预先明确的目标 endpoint；
- 三条 E.7 primary step-200 worsened sample 已知。

三条 target 的完整机器 ID 冻结为：

| E.7 对应 episode | Base sample ID | E.7 primary step-200 ratio |
| --- | --- | ---: |
| `episode_000010` | `75359438f810e6921754de327beda8bd974343f5e89fb54d7ac8852f79c89c9b` | 1.1292× |
| `episode_000011` | `5f82a5db9be7a61f969fd32f5bca19dbb19a65106fb49d5357705be2d03def44` | 1.0194× |
| `episode_000012` | `81363feff988d3f3faaeeb66191e7ff9c4fd40c85d7b3b7cd0bda84cd41e3b9b` | 1.1814× |

因此，E.8 不是对“任意最差样本”的重新搜索。主要 tail-risk 判据只针对以上三条
结果前已固定 target；其余五条仍完整报告，但不能替换 target 以获得有利结论。

## 4. 冻结父证据

### 4.1 E.7 工件

输出根：`outputs/thought3/phase_e7_checkpoint_trajectory_v1/`

| 工件 | SHA-256 |
| --- | --- |
| `gate_e7_result.json` | `9b242a3a38638cf2f67c31dd343af0e0d1ec39941d3e784dcd3e167bf14baa4b` |
| `run_status.json` | `207dc70a5a83bd67787f038559a4262708b9fb4e355f628cbc6cca90a162e125` |
| `pre_validation_result.json` | `cbe4bf697c07307bca3f9708fefd235160ccb6bcf355920c85913ac979616b5f` |
| `data_preparation.json` | `f6635c8d0e80d052ad06ce5848bbd2d2ee14635fd0594d44095ccc3461a57fc4` |
| `logs/phase_e7.log` | `e32a9bbbd74582f39d4593f851235e29c6145dd01b6c4cd3188f77ac8a78d899` |

运行前后必须重算这些 SHA。E.8 同时复核 E.6 root、八个 checkpoint 的 24 个
文件 SHA、checkpoint manifest/cursor/schedule 和 frozen Fast-WAM commit。

### 4.2 只读 checkpoint

E.8 只加载：

- A0 step 100：E.7 两个 panel 都通过的 early comparator；
- A0 step 200：E.7 不稳定的目标 endpoint。

不加载 A1，不评估 step 50/150，不训练新 checkpoint。step 100 的选择发生在
E.7 结果已知之后，只用于 onset 描述，不能包装成独立早停验证。

## 5. Cohort 与隔离

E.8 复用 E.6/E.7 的八条 train demonstration，即 Phase D 冻结 train 排序
1-based 位置 `9–16`：

- 不读取排序 `17–28` 的剩余 train cohort；
- 不消耗以后用于 recipe replication 的独立样本；
- 不读取四条 development sample；
- sample 顺序、payload SHA 和 Phase D 28/4 split 必须与 E.6/E.7 一致。

因此 E.8 能增加 flow-level 精度，但不能增加 demonstration-level 样本量。

## 6. 全新 flow panel

### 6.1 Full panel

- flow slots：`11..74`
- 每个 probe：`8 samples × 64 flows = 512 objectives`
- pre-outcome RNG identity SHA：
  `710b809614aeb502c944275c4c43759d2383b00e52fd9d5216898fb949b5772a`

这些 slots 与以下 namespace 硬不相交：

- E.1/E.2 fixed flow：`0`
- E.3–E.6 held-out flows：`1..5`
- E.7 primary flows：`6..10`
- E.4 train slots：`10001..10200`
- E.5 train slots：`20001..21600`
- E.6 train slots：`31001..32600`

### 6.2 两个冻结 replication block

- Block A：`11..42`，32 flows/sample；
- Block B：`43..74`，32 flows/sample。

两个 block 不重复。每个 block 都是 E.7 五-flow panel 的 6.4 倍；full panel
为 12.8 倍。block 只由 slot 范围决定，不允许运行后重新分组。

### 6.3 预知 zero-weight endpoints

按 `sample_index, flow_step` 冻结为：

```text
(2,47), (2,59), (2,68), (4,69), (5,58), (6,70), (7,11)
```

这些 objective 的官方 scheduler weight 和 loss 必须精确为 0；它们保留在
sample mean 中，仅在逐 objective ratio 分母为 0 时不计算 ratio。

## 7. 固定计算预算

| 项目 | 数量 |
| --- | ---: |
| Track | 1（A0 only） |
| Checkpoints | 2（step 100、200） |
| Flows/sample | 64 |
| Initial objectives | 512 |
| Step-100 objectives | 512 |
| Step-200 objectives | 512 |
| 总 forward objectives | 1,536 |
| Backward | 0 |
| Optimizer / optimizer step | 0 / 0 |
| 新 checkpoint | 0 |
| 新 training objective | 0 |
| development/OOD/success/rollout | 0 |
| future RGB decoded | 0 |

E.7 的 800 objectives 总耗时 13.98 分钟，其中模型加载 6.67 分钟。按 probe
部分线性估算，E.8 单卡预计约 `19–25` 分钟；硬显存上限仍为 `<43 GiB`。
耗时估计不是停止或科学判据。

## 8. 冻结指标

对 step 100/200，分别相对同一 zero-gate A0 initial probe 计算：

- full 64-flow mean loss reduction；
- Block A/B mean loss reduction；
- full 和两个 block 的 `8` 条 sample non-worsened 数；
- catastrophic sample 数；
- median/max gated-delta-to-action-hidden ratio；
- 每条 sample 的 full/Block A/Block B relative change；
- 每条 sample 中 `final > initial` 的 flow 比例；
- 逐 objective loss ratio 与 zero-weight 计数。

full 和两个 block 都复用原 A0 stability Gate：

1. mean loss reduction `>=0%`；
2. `>=6/8` sample 不变差；
3. catastrophic sample `=0`；
4. median delta/hidden `<=0.5`；
5. max objective delta/hidden `<=1.0`。

## 9. 配对 bootstrap 与多重比较

### 9.1 Estimand

每条 sample、每个 checkpoint 的 flow-level relative mean change 为：

```text
(mean(final_loss over flows) - mean(initial_loss over flows))
/ mean(initial_loss over flows)
```

正值表示 A0 checkpoint 相对 zero-gate initial 变差。

### 9.2 冻结方法

- resampling unit：同一 sample 内的 paired flow；
- 每次同步重采 initial/final 的同一 flow index；
- bootstrap replicates：`20,000`；
- seed：`20260729080`；
- percentile method：NumPy `linear`；
- family-wise alpha：`0.05`；
- 比较数：`8 samples × 2 checkpoints = 16`；
- one-sided lower quantile：`0.05/16 = 0.003125`。

一条 sample 定义为 `confirmed_worsened`，当且仅当：

1. full 64-flow relative change `>0`；
2. Block A relative change `>0`；
3. Block B relative change `>0`；
4. Bonferroni one-sided bootstrap lower bound `>0`。

`relative change >=2%` 另记为 material descriptive flag，但 **2% 不参与主要
分类**，避免根据 E.7 的已知 ratio 追溯设置有利阈值。

该 bootstrap 量化 pseudo-random flow panel 的 Monte Carlo 不确定性，不是
demonstration-population CI。

## 10. 五-flow panel sensitivity

为直接量化“五个 flow draw 会有多不稳定”，从 64 个新 slots 中执行：

- `20,000` 次无放回抽取 5 slots；
- 每次八条 sample 共用同一组 slot 编号，与 E.7 panel 构造一致；
- seed：`20260729081`；
- 每次重新计算原 A0 Gate。

固定报告：

- five-flow Gate pass/fail rate；
- 因 `<6/8` sample 导致的 stability fail rate；
- non-worsened count `0..8` 分布；
- pooled mean worsening rate；
- reduction 的 p05/p50/p95。

这是 full 64-flow 工件上的 sensitivity analysis，不是 20,000 组独立新模型
forward，也不进入主要二元分类。

## 11. 主要互斥分类

只允许以下三种结果：

### 11.1 `persistent_target_tail_risk_supported`

E.7 三条预识别 target 中，至少 `2/3` 在 step 200 满足
`confirmed_worsened`。

解释边界：A0 的少数 demonstration-level loss harm 能跨大量新 flow 复现；
不表示其能推广到新 demonstration，也不表示 A1/future 有用。

### 11.2 `five_flow_panel_variance_supported`

必须同时满足：

1. step 200 的八条 sample 中 `0` 条 `confirmed_worsened`；
2. step 200 full 64-flow panel 通过原 A0 Gate；
3. step 200 Block A 通过原 A0 Gate；
4. step 200 Block B 通过原 A0 Gate。

解释边界：E.7 的 `5/8` 没有在更大新 panel 上复现，更符合五-flow Monte Carlo
方差；不证明 A0 对所有数据稳定。

### 11.3 `mixed_or_inconclusive`

其余全部情况，包括：

- 只有 `1/3` target 被确认；
- 风险转移到非 target sample；
- full 与两个 block 给出不同 Gate；
- sample mean 恶化但校正后下界跨 0；
- Gate 仍失败但证据不足以确认两条 target。

不得在 mixed 结果后降低 `2/3`、改 block 或换 target。

## 12. Step-100 onset 子分类

主要 tail-vs-variance 分类只由 step 200 决定。step 100 只描述已经确认的
step-200 target 风险何时出现：

- `late_emergent_after_step100`：step-200 confirmed targets 在 step 100
  均未确认；
- `partly_present_by_step100`：部分已在 step 100 确认；
- `already_present_by_step100`：全部已在 step 100 确认；
- `not_applicable_no_confirmed_endpoint_target`：step 200 无 target 确认。

该子分类不能把 step 100 升级为 checkpoint candidate。

## 13. 工程有效性

以下任一失败使 E.8 为 `engineering invalid`，不得给科学分类：

- E.7 五个冻结工件 SHA 改变；
- E.6 任一 checkpoint/root 工件 SHA 或 manifest 改变；
- project/FastWAM repository dirty 或 FastWAM commit 不符；
- sample 顺序、payload、split 或 target ID 改变；
- flow grid 缺失、重复或与 `11..74` 不同；
- RNG identity SHA 或任一 objective seed 不符；
- zero-weight 位置/loss 不符；
- initial/final timestep 或 action weight 不配对；
- 出现 NaN/Inf；
- 读取 development、OOD、success、rollout 或 future RGB；
- Adapter/Fast-WAM 出现 grad；
- Fast-WAM parameter SHA 前后改变或变为 trainable；
- checkpoint 或 E.7 工件在运行中改变；
- 显存达到 43 GiB 硬上限。

三种有效分类都表示工程运行正常，命令 exit code 为 `0`。
`gate_e8_passed=true` 只表示工程 Gate 通过，不等于 tail-risk hypothesis
被支持。

## 14. 停止、恢复与结果边界

- 正式运行前必须 clean commit；
- 同一 output 已存在有效结果时，只允许 `--resume` 只读返回；
- partial/invalid run 不允许在同一 Run ID 续算；
- 修复需新 config fingerprint、新输出目录和新协议记录；
- E.8 后不得事后合并 E.7/E.8 flow 伪装成预注册主 panel；
- 不论分类为何，都不直接解锁 full E、A2/A4 或 OOD。

若支持 tail risk，下一步应设计 matched A0/A1 的单变量 tail-risk mitigation，
并在未使用 train cohort 独立复验。若支持 five-flow variance，也只能把 A0
配方作为候选，仍须未使用 cohort 复验。mixed 则需报告不确定性，不能继续堆叠
结果后阈值。

## 15. 配置、命令与输出

- config：
  `configs/thought3/phase_e8_a0_flow_variance_replication.yaml`
- config fingerprint：
  `ed587c61cec3e386e5b44af11fca646dab527acbe46cce34d6badfd34ff09f7f`
- output：
  `outputs/thought3/phase_e8_a0_flow_variance_replication_v1/`
- schema：
  `thought3.phase_e8.a0_flow_variance_replication.v1`

无写入 dry-run：

```bash
fastwam-ood thought3-replicate-a0-flow-variance \
  --config configs/thought3/phase_e8_a0_flow_variance_replication.yaml \
  --dry-run
```

正式运行必须使用一张空闲 GPU：

```bash
CONFIRM_THOUGHT3_PHASE_E8=YES \
THOUGHT3_GPU_ID=1 \
bash scripts/run_thought3_phase_e8_a0_flow_variance_replication.sh
```

`THOUGHT3_GPU_ID=1,2` 会被拒绝。可使用卡 1 或卡 2，但一次只能指定一张。

监控：

```bash
tail -f \
  outputs/thought3/phase_e8_a0_flow_variance_replication_v1/logs/phase_e8.log
```

权威输出：

- `run_status.json`
- `data_preparation.json`
- `pre_validation_result.json`
- `gate_e8_result.json`
- `logs/phase_e8.log`

E.8 已按本协议完成；本协议的 flow、target、bootstrap、分类和证据边界未追溯
修改。结果报告见 [thought3_phase_e8_report.md](thought3_phase_e8_report.md)。
