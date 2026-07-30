# Thought3 Gate E.7：只读 Checkpoint-Trajectory 诊断预注册

状态：**已按协议运行；工程 Gate 通过；主要分类不支持实质晚期退化**

冻结日期：2026-07-29

> 本协议在查看 Gate E.6 的 step-200 结果后建立。E.6 的 A0 step-200
> 结果已知；step 50/100/150 的 A0/A1 probe outcome 在本协议冻结前没有读取。
> E.7 是结果后的机制诊断，不是独立确认性实验，也不是新的训练或 checkpoint
> 选择实验。

> 预注册权威版本是首次包含本文、冻结配置、编排器、runner 和测试的 clean git
> commit。正式运行会记录精确 project HEAD，并在 project 或 FastWAM worktree
> dirty 时拒绝启动。预注册本身不加载模型、不运行 GPU probe。

> 2026-07-29 结果更新：primary flow 上 A0 step 50/100 stable、150/200
> unstable，但 step-200 mean 比 step 50 更低且 non-worsened 只下降 1，故冻结
> 分类为 `not_supported_no_material_late_degradation`。没有 joint diagnostic
> candidate；全部工程检查通过。原协议和门槛不追溯修改。详见
> [thought3_phase_e7_report.md](phase_e7_report.md)。

## 1. 研究问题与结论边界

Gate E.6 中，匹配的 `A0/A1@3e-4` 在未使用 train cohort 上完成 200 updates：

- A0 held-out mean loss 下降 `1.191%`，但只有 `4/8` sample 不变差；
- A1 held-out mean loss 下降 `14.842%`，`7/8` sample 不变差；
- A1 final mean 比 A0 低 `13.815%`，逐样本为 `6/8`。

E.6 因 A0 稳定性门槛失败，结论保持有效负 Gate。E.7 只问：

> A0 在 step 200 的样本级不稳定，是否更符合“较早 checkpoint 稳定、训练后期
> 退化”，而不是从较早训练阶段起就没有稳定 checkpoint？

E.7 不能回答：

- future 是否改善 OOD 或在线成功率；
- A1 相对 A0 的差异是否由 future 因果产生；
- 哪个 checkpoint 已可用于论文主结果；
- A2/A4 是否应解锁；
- 训练后期退化的优化器机制、过拟合对象或统计显著性。

本诊断只使用 train-cohort 上固定 action-flow objective 的离线 action loss。
无 development、OOD、success、rollout 或真实 future RGB 输入。

## 2. 已知信息、未知信息与污染披露

在冻结 E.7 前已知：

- E.6 是执行完整的有效负 Gate；
- E.6 step-200 continuity panel 使用 flow `1..5`；
- A0 step-200 reduction 为 `1.1911835351551193%`，`4/8` sample
  不变差；
- A1 step-200 reduction 为 `14.841688817723778%`，`7/8` sample
  不变差；
- A1 相对 A0 的 step-200 paired contrast 已通过；
- checkpoint 在 step `50/100/150/200` 保存。

在冻结 E.7 前未知且不得提前查看：

- step 50/100/150 在任一 panel 的 A0/A1 loss；
- 新 primary flow `6..10` 上所有 checkpoint 的结果；
- 哪个 step 满足稳定性或 joint diagnostic candidate 规则；
- A0 trajectory 的最终分类。

因此，旧 flow `1..5` 只作为 continuity/reproduction panel；它不能决定主要
trajectory 分类或候选 step。主要判断只由新 flow `6..10` 决定。

## 3. 冻结父证据

### 3.1 Gate E.6 身份

- E.6 预注册 commit：
  `cb6f311fe1154722eaaeaf1f02f26cfde4922d56`
- E.6 结果 commit：
  `e5eeb3b6763a100bb371f1f78a548e11f1e1205a`
- E.6 config fingerprint：
  `8cb2ab718eed2cc226491038423c92f1c59128246d966a2a9c3700d505f292d9`
- cohort SHA：
  `6a354151d6d3e93335b66743f16be1908abc8d0fe835ee3811562b2eeb63d7c3`
- train identity schedule SHA：
  `419b09a2ec30ce7bffc99c95aff1a343f77d39e83e77a752fc67bc984508febc`
- sample payload SHA：
  `f5e61fd99d68244d7fa3cca6cc1ff59aabc12317840e4832ff2595f9ff78252f`
- frozen Fast-WAM parameter SHA：
  `ac0dd59de78495dc8ed5c5601e24951b78f3fd6ad40843d76b83106eccceb4f8`

父结果根目录的冻结 SHA：

| 工件 | SHA-256 |
| --- | --- |
| `gate_e6_result.json` | `464d9d3e02c52c2b1f2838ce59fe71a9b35716884d4d1da4b3d0e2ad78b42af6` |
| `run_status.json` | `b6dd1edf41375e4ecd5d6495976298b6246307eafe16946be6662a99cb3b9adc` |
| `pre_validation_result.json` | `3639032aa3d8faed5fd20d9f5da313ee51fb7605cf81aac95465a78390d83ec2` |
| `data_preparation.json` | `4f8c6d02c06a4f6a80bc01ec54e88c13a39d4bab4be9f04b7cb547347af552df` |
| `logs/phase_e6.log` | `b888d48f3b45dedc7577f616a6910400d950d38aa38be75ebbacf4f8d90eb81d` |

运行前必须重算这些 SHA，并确认 E.6 只有
`A0.performance.at_least_6_of_8_samples_non_worsened=false`；其他
execution、performance、paired 和 frozen checks 均为真。

### 3.2 八个只读 checkpoint

所有目录必须恰好包含 `adapter.safetensors`、`manifest.json` 和
`optimizer.pt`。E.7 会校验三类文件，但只把 Adapter state 加载到内存；
optimizer state 不会反序列化，也不会创建 optimizer。

| Track | Step | Adapter SHA | Manifest SHA | Optimizer SHA |
| --- | ---: | --- | --- | --- |
| A0 | 50 | `36ec9e9e0394...ba3a` | `c2f6d4793038...4c58` | `2c60f67b18cf...4a69` |
| A0 | 100 | `b42e683e5463...70cf6` | `d8730f48c926...92c5` | `4278f9691f10...0869` |
| A0 | 150 | `64efdff45f3f...393f` | `e052ccb80f10...bc06` | `082e5167cd89...17d0` |
| A0 | 200 | `c8cdef567f0b...a292` | `1d233ae7720a...8dcf` | `87c8680ac795...1049a` |
| A1 | 50 | `62437b947a0b...40b32` | `0af7df855588...3b80` | `3393fe9837fc...229a` |
| A1 | 100 | `18021db7419b...ca7f` | `7c3e01e3ddf5...f9a3` | `73689d942bff...88ed` |
| A1 | 150 | `04f3af7c7be3...e743` | `e8cd038be429...10f0` | `66f8771d80c7...bcff` |
| A1 | 200 | `aa55622c03aa...b78f` | `82cfa32891dc...48f3` | `72daa2fc60ba...4fe1` |

完整 64-character SHA 冻结在
`phase_e7_checkpoint_trajectory.py::PHASE_E6_CHECKPOINT_FILE_SHA256`，
测试会逐文件重算，不以上表的缩写代替机器校验。

## 4. 冻结 cohort 与 probe 身份

E.7 复用 E.6 的八条 train 样本，即确定性 train 排序的 1-based 位置
`9–16`。不读取排序 `17–28` 的剩余 train cohort，不新增训练 sample，也不改变
Phase D 的 28/4 split。

### 4.1 Primary panel

- flow steps：`6,7,8,9,10`
- 每 checkpoint：`8 samples × 5 flows = 40 objectives`
- pre-outcome RNG identity SHA：
  `3361f17069cb79bea7a330181fc97ecc3adfa9f3473d55b60640ad4249752f68`
- 预知 `t=1000 / weight=0` 位置：无
- 角色：唯一可决定 A0 trajectory 分类和 joint diagnostic candidate 的 panel

这些 flow 没有用于 E.6 的 held-out panel，也不与 E.4/E.5/E.6 的训练
namespace `10001+ / 20001+ / 31001+` 重叠。

### 4.2 Continuity panel

- flow steps：`1,2,3,4,5`
- 每 checkpoint：40 objectives
- pre-outcome RNG identity SHA：
  `94f54e530b7cf9ea4a8f178f8fa47afe3cab8769e652d65a5c0a25dcf085d739`
- 预知唯一 zero-weight 位置：`sample_index=8, flow=5`
- 角色：复现 E.6 step-200，并描述同一旧 panel 的轨迹
- 不用于主要分类，不用于候选 step

continuity step-200 outcome 必须与冻结 E.6 JSON 完全一致，否则 E.7 为工程无效。

## 5. 固定计算预算

每个 track 和 panel 都先用相同 zero-gate 初始化 Adapter 建立 step-0
baseline，再依次只读评估 step 50/100/150/200：

| 项目 | 数量 |
| --- | ---: |
| Tracks | 2（A0、A1） |
| Checkpoints | 8 |
| Initial probe panels | 4 |
| Checkpoint probe panels | 16 |
| Objectives/panel | 40 |
| 总 forward objectives | 800 |
| Backward | 0 |
| Optimizer / optimizer step | 0 / 0 |
| 新 checkpoint 写入 | 0 |
| 新训练 objective | 0 |
| development/OOD/success/rollout | 0 |
| future RGB decoded | 0 |

单张空闲 GPU 运行；预计总耗时约 `12–18` 分钟，其中模型加载约 6 分钟。
继承硬显存上限 `<43 GiB`；E.6 同模型加载峰值为 `23,679.513 MiB`。该估计
不是结果门槛。

## 6. 冻结指标与 A0 稳定性规则

每个 checkpoint 相对同 track、同 panel 的 zero-gate step-0 baseline 计算：

- mean action loss reduction；
- `8` 条 sample 中 non-worsened 数；
- catastrophic sample 数，即 final loss `>2×` initial；
- median gated-delta/action-hidden ratio；
- 最大 objective gated-delta/action-hidden ratio；
- 逐 sample initial/final loss 和逐 objective loss ratio。

A0 checkpoint 定义为稳定，当且仅当：

1. mean loss reduction `>=0%`；
2. `>=6/8` sample 不变差；
3. catastrophic sample `=0`；
4. median delta/hidden `<=0.5`；
5. 最大 objective delta/hidden `<=1.0`。

这些门槛完全复用 E.6 的 A0 stability gate，不因 trajectory 结果改变。

## 7. 主要 trajectory 分类

早期 checkpoint 固定为 `50/100/150`，endpoint 固定为 `200`。为避免结果后
挑选“最好看的早期点”，比较基准固定为**最早满足 A0 稳定性规则的 checkpoint**。

`late_overtraining_supported` 当且仅当 primary panel 同时满足：

1. step 50/100/150 中至少一个 A0 checkpoint 稳定；
2. step 200 A0 不稳定；
3. 从最早稳定 checkpoint 到 step 200，non-worsened sample 数下降至少 `2`；
4. step-200 final mean action loss 高于最早稳定 checkpoint。

否则使用以下互斥分类：

- `not_supported_endpoint_stable`：primary step 200 仍稳定；
- `not_supported_no_earlier_stable_checkpoint`：没有早期稳定 checkpoint；
- `not_supported_no_material_late_degradation`：早期稳定且 endpoint 不稳定，
  但未同时达到“下降至少 2 条 sample + mean loss 增加”。

这些是诊断分类，不是显著性检验。“not supported”不等于证明没有 late
over-training；它只表示固定的 8-sample/5-flow 诊断没有满足预注册模式。

## 8. A1 与 joint diagnostic candidate

A1 的绝对门槛保持：

- mean reduction `>=10%`；
- `>=6/8` sample 不变差；
- catastrophic sample `=0`；
- median ratio `<=0.5`；
- 最大 objective ratio `<=1.0`。

同 step 的 A1-vs-A0 paired superiority 保持：

- A1 final mean 至少比 A0 低 `10%`；
- 至少 `6/8` sample 的 A1 final loss 不高于 A0；
- sample IDs 完全一致。

若 primary panel 某 step 同时满足 A0 stability、A1 absolute 和 paired
superiority，则列为 diagnostic candidate；只报告最早满足者以及所有满足 step。
该结果必须标记：

```text
post_run_diagnostic_candidate_only
```

它不是正式 checkpoint selection，不允许直接进入 A2/A4、full E 或 OOD。若要
采用，必须在 E.7 未使用的新 cohort 上另行预注册复验。

## 9. 工程有效性与停止规则

以下任一失败都使 E.7 为 `engineering invalid`，不得给 trajectory 科学分类：

- 父 E.6 root 工件或任一 checkpoint 文件 SHA 改变；
- checkpoint manifest 的 variant/step/cursor/schedule/provenance 不符；
- project/FastWAM repository dirty 或 FastWAM commit 不符；
- E.6 sample payload、顺序、split 或 probe RNG identity 不符；
- 读取 development、future RGB、OOD、success 或 rollout outcome；
- probe 网格缺失/重复、NaN/Inf、zero-weight 位置不符；
- continuity step-200 不能 exact reproduce E.6；
- Fast-WAM 参数 hash 前后改变、出现 grad 或变为 trainable；
- Adapter probe 产生 grad；
- checkpoint 文件在运行前后改变；
- 显存达到硬上限。

有效的四种 trajectory 分类都表示 E.7 正常完成，因此命令 exit code 为 `0`。
只有工程无效或执行异常返回非零。不能把 `gate_e7_passed=true` 误读为
late-overtraining hypothesis 被支持；该字段只表示工程 Gate 通过。

若中途失败：

- 保留 `run_status.json`、log 和已有 pre-validation；
- 不覆盖同一 Run ID；
- 不对 partial evidence 使用 `--resume`；
- 修复只能使用新 config fingerprint 和新输出目录。

## 10. 配置、输出与运行命令

- config：
  `configs/thought3/phase_e7_checkpoint_trajectory.yaml`
- config fingerprint：
  `3823a3403e2d94c4690cf210209e1b530388722446fe64220a79560c18209af2`
- output：
  `outputs/thought3/phase_e7_checkpoint_trajectory_v1/`
- schema：
  `thought3.phase_e7.checkpoint_trajectory.v1`

预注册完成后的无写入 dry-run：

```bash
fastwam-ood thought3-diagnose-checkpoint-trajectory \
  --config configs/thought3/phase_e7_checkpoint_trajectory.yaml \
  --dry-run
```

正式运行必须使用一张空闲卡；卡 1 或卡 2 均可，但不能写成 `1,2`：

```bash
CONFIRM_THOUGHT3_PHASE_E7=YES \
THOUGHT3_GPU_ID=1 \
bash scripts/run_thought3_phase_e7_checkpoint_trajectory.sh
```

监控：

```bash
tail -f outputs/thought3/phase_e7_checkpoint_trajectory_v1/logs/phase_e7.log
```

权威输出包括：

- `run_status.json`
- `data_preparation.json`
- `pre_validation_result.json`
- `gate_e7_result.json`
- `logs/phase_e7.log`

运行完成后才允许撰写 E.7 结果报告；原协议、阈值、已知/未知披露和 primary/
continuity 角色不得追溯修改。
