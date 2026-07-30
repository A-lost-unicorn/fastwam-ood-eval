# Thought3 Gate E.9a-v1：Matched Sample-Tail Mitigation 归档协议

状态：**INVALID ENGINEERING RUN / SUPERSEDED**

冻结日期：2026-07-29

> 2026-07-29 的首次真实启动在 raw/A0 initial probe 前因 evaluator
> 错误地硬编码 `flow_steps=1..5` 而停止；协议要求的是 `75..106`。该次运行
> 完成模型加载与八条样本准备，但完成 `0` 个训练 objective、`0` 次
> optimizer update，没有写出科学结果，冻结 Fast-WAM SHA 前后相同。v1
> 输出已只读归档，不得 resume 或覆盖。证据见
> [v1 失败报告](thought3_phase_e9_v1_failure_report.md)；修复后的唯一入口和
> 不变科学边界见 [E.9a-v2 协议](thought3_phase_e9_v2_protocol.md)，实际
> 四轨状态见 [E.9a-v2 结果](thought3_phase_e9_v2_report.md)。

> 本协议在查看 E.8 全部结果后建立。已知 E.8 为
> `mixed_or_inconclusive`，并已知每条样本的 zero-gate initial loss、A0
> step-200 结果和 confirmed-harm 身份。因此 E.9 是明确的 result-conditioned
> 序贯工程实验，不是独立确认性检验。

> E.9 不选择 step 100，不降低 E.6–E.8 的任何门槛，不继续用“增加 flow
> 数量”解释同八条样本。E.9a 的新 flow 仅用于四条 matched 训练/评估轨迹，
> 以隔离一个优化变量；E.8 结论不重算、不覆盖。

> 预注册权威版本是首次同时包含本文、冻结配置、实现、runner 和测试的 clean
> git commit。预注册阶段不加载模型、不启动 GPU、不写 E.9 运行结果。

## 1. 研究问题与唯一处理变量

E.8 说明 pooled loss 改善与少数 sample harm 可以同时存在。E.9a 检验：

> 用 zero-gate initial loss 对八条样本的训练 objective 做固定尺度归一化，能否
> 在保持 A0/A1 absolute 和 paired 门槛的同时，减少 FWER-confirmed sample
> harm？

四条轨迹为：

| Recipe | Variant | Future | 训练聚合 |
| --- | --- | --- | --- |
| raw | A0 | K=0 | 八条 raw loss 的 arithmetic mean |
| raw | A1 | K=1 | 八条 raw loss 的 arithmetic mean |
| normalized | A0 | K=0 | 八条固定加权 loss 的 mean |
| normalized | A1 | K=1 | 八条固定加权 loss 的 mean |

raw 与 normalized 使用同一批新 train-flow slots、同一 held-out panel、同一
初始化、LR、200 update、每 update 八条样本各一个 objective、checkpoint
interval 和 Adapter 结构。唯一处理变量是 `sample_loss_weight`。

本实验不改变：

- checkpoint endpoint：只判定 step 200；
- LR：`3e-4`；
- action denoise steps：20；
- A0/A1 参数量、初始化和 optimizer；
- 原 `6/8`、A1 `10%`、尺度与 catastrophic 门槛；
- 标准 LIBERO 当前观测唯一输入和 zero future-RGB leakage。

## 2. 已知 E.8 结果与结果后设计披露

冻结父结果：

| 工件 | SHA-256 |
| --- | --- |
| `gate_e8_result.json` | `e3809eedaadc4eb7ce4c681151214f01304e08b0a45cd3bccf926ed003c989e1` |
| `run_status.json` | `03e9039b078ef5cd34c2a97d55b5d25fec29937959aff29c4dd322956ce8f53a` |
| `pre_validation_result.json` | `1a46e92af902e1613a87a4644912326f184b1517c289ea04d0d0becab8d6bc04` |
| `data_preparation.json` | `abdb800855e3bdedc5f8e9e267e5c7e1cef030050b88a32cd58ecdf81c983828` |
| `logs/phase_e8.log` | `68eda4a7b131a9cb82209df2c56ac67877ffc3dff564682877026d0abdc9743c` |

冻结 E.8 事实：

- 工程 Gate 通过；
- 分类为 `mixed_or_inconclusive`；
- A0 step 200 full 32-flow×2 panel pooled loss 改善 `3.7283%`；
- full sample stability 只有 `4/8`；
- 两条 sample 为 FWER-confirmed harm，其中一条是预识别 target；
- E.8 没有产生 checkpoint 或配方候选。

选择 sample normalization 是查看上述结果后的工程决策。其证据等级必须写成
`POST-RUN SEQUENTIAL ENGINEERING`；即使 E.9a 通过，也不能称为独立确认。

## 3. 固定 sample normalization

### 3.1 Calibration 数据

只使用 E.8 zero-gate initial probe 的 64-flow sample mean，不使用 E.8
step-100/200 final loss、harm 标签、A1 outcome、development、OOD 或 success：

| Episode | Base sample ID | `L_i` | `w_i` |
| --- | --- | ---: | ---: |
| 14 | `9610d2aed3a6ddf382c514715ead977c9f9a25b56265b2705a9146ac28f6c0cc` | 0.005343552826 | 0.768689729381 |
| 10 | `75359438f810e6921754de327beda8bd974343f5e89fb54d7ac8852f79c89c9b` | 0.003564936602 | 1.152203989644 |
| 11 | `5f82a5db9be7a61f969fd32f5bca19dbb19a65106fb49d5357705be2d03def44` | 0.002473537945 | 1.660590727897 |
| 30 | `8f34793be5e051e0d62c0397b83cc341f17b626bd73660968f48ff1f6339d1b9` | 0.005682245364 | 0.722871666543 |
| 19 | `8c00174e915504c49a3c69057f9c199af1654a6ecef414070c1657316b1e4418` | 0.004042625069 | 1.016056177703 |
| 38 | `461a673f2745ab243d99d617f4514a737644d44ba2fc5fdece8b45f347e51564` | 0.007539614805 | 0.544793637655 |
| 0 | `739baab482230ba4ee1ae9c0cccf5886268db9ee37c895435af6c6891d22c3b0` | 0.004545348814 | 0.903678539127 |
| 12 | `81363feff988d3f3faaeeb66191e7ff9c4fd40c85d7b3b7cd0bda84cd41e3b9b` | 0.003336432747 | 1.231115532050 |

公式冻结为：

```text
w_i = (1 / L_i) / mean_j(1 / L_j)
weighted_update_loss = (1 / 8) * sum_i(w_i * loss_i)
```

因此 `sum(w_i)=8`，平均权重为 1，不改变 nominal gradient scale。没有 clipping、
temperature、epsilon、learned weight 或运行后再归一化。

- calibration payload SHA：
  `edfb31e3fe1d6a8067a607ed20803ded33ba98f860c2a679067e70aa21105d70`
- weight-only SHA：
  `3e65b4f76f6cdee7176c49c9befd12bcd416fe9f60f2f719446a2896b05719f6`

### 3.2 为什么保留 raw 对照

若只运行 normalized A0/A1，新的 train/held-out flow draw 与 weighting 会同时
变化，无法判断稳定性变化来自处理还是随机 schedule。四轨设计让 raw 与
normalized 在同一新 schedule 上比较，避免把新 flow draw 冒充 mitigation。

raw 控制是否通过不是 E.9a engineering validity 的条件；它必须完整报告，作为
tail-contrast 参照。

## 4. E.9a cohort 与 flow 隔离

### 4.1 开发 cohort

E.9a 只使用 E.6–E.8 的 train 排序位置 `9–16`，共八条 demonstration：

- 不读取四条 development sample；
- 不读取排序 `17–28`；
- 不读取 Thought1/2、OOD、success 或 rollout outcome；
- 不解码真实 future RGB；
- 训练数据仍为当前双相机帧、proprio 和 action target。

### 4.2 新训练 namespace

- 每轨 200 updates；
- 每 update 8 objectives；
- train slots：`40001..41600`；
- 每轨 1,600 train objectives；
- 四轨总计 6,400 train objectives；
- pre-outcome train identity SHA：
  `4c5c66f977e6f75dfaf3bb9db398a13c8a2807d6c065ae19307b19435440d64e`。

预知 22 个 zero-weight `(update, micro, slot)`：

```text
(16,4,40124), (16,8,40128), (27,4,40212), (39,7,40311),
(57,5,40453), (58,4,40460), (62,5,40493), (69,8,40552),
(82,5,40653), (89,1,40705), (91,8,40728), (104,3,40827),
(107,5,40853), (108,3,40859), (109,1,40865), (119,8,40952),
(124,7,40991), (135,5,41077), (142,5,41133), (167,4,41332),
(183,1,41457), (183,6,41462)
```

### 4.3 新 held-out panel

- Block A：`75..90`；
- Block B：`91..106`；
- full：32 flows/sample；
- 每次 probe：`8×32=256 objectives`；
- 四轨 initial+final：`4×2×256=2,048 objectives`；
- identity SHA 使用 0-based sample index：
  `76e96cb5be832908aff1510256bc058fa5023c8b71e51b57dfe6b3f277d899fb`；
- zero-weight `(0-based sample index, flow)`：
  `(1,80), (7,93)`。

这些 slots 与 E.1–E.8 所有 train/probe namespace 不相交。E.9a 不是再做一个
“多加 flow 的 A0 诊断”；这些 flow 是四轨单变量实验本身的冻结随机化。

## 5. 固定工程预算

| 项目 | 数量 |
| --- | ---: |
| Track | 4 |
| Optimizer updates | 800 |
| Train objectives | 6,400 |
| Held-out objectives | 2,048 |
| Checkpoint endpoint | 200 only |
| 每轨 checkpoint | 50/100/150/200，科学判定只用 200 |
| Development/OOD/success/rollout | 0 |
| Reserved 17–28 sample decode/train | 0 |
| Future RGB decoded | 0 |

按 E.6 双轨 43.89 分钟和本次更大 held-out panel 估算，单卡预计约
`90–120` 分钟；估算不属于停止或科学判据。峰值显存必须 `<43 GiB`。

## 6. 原门槛保持不变

每个 recipe 的 A0/A1 都使用 full 32-flow outcome。

A0 absolute：

1. pooled mean loss reduction `>=0%`；
2. `>=6/8` sample 不变差；
3. catastrophic sample `=0`；
4. median gated-delta/hidden `<=0.5`；
5. max objective gated-delta/hidden `<=1.0`。

A1 absolute：

1. pooled mean loss reduction `>=10%`；
2. `>=6/8` sample 不变差；
3. 同一 catastrophic 与尺度门槛。

normalized A1 vs normalized A0：

1. A1 final mean 至少低 `10%`；
2. A1 在至少 `6/8` sample 上不高于 A0；
3. sample IDs、初始化、flow schedule、budget 完全匹配。

raw A1 vs raw A0 完整报告，但不替代 normalized candidate Gate。

禁止：

- 把 `6/8` 改成 `5/8`；
- 把 A1 `10%` 改为只要求正改善；
- 选择 step 100、50 或 150；
- 根据 E.9a outcome clipping/reweight；
- 只保留 A1 或只汇报 pooled mean。

## 7. FWER sample-tail 判据

每条轨迹、每条 sample 在 full 32-flow panel 上执行 paired-flow bootstrap：

- resampling unit：同 sample 内 initial/final paired flow；
- replicates：20,000；
- seed：`20260729090`；同 sample 在四轨使用相同 bootstrap index；
- family-wise alpha：0.05；
- comparisons：`4 tracks × 8 samples = 32`；
- one-sided lower quantile：`0.05/32 = 0.0015625`；
- NumPy percentile method：`linear`。

`confirmed_worsened` 当且仅当：

1. full relative mean change `>0`；
2. Block A change `>0`；
3. Block B change `>0`；
4. Bonferroni one-sided lower bound `>0`。

`>=2%` 只记 descriptive material flag，不是主要门槛。normalized A0 和 A1
都要求 confirmed harm `=0`。

## 8. E.9a 互斥分类

### `tail_mitigation_candidate_supported`

当且仅当：

- normalized A0/A1 absolute Gate 全过；
- normalized A1-vs-A0 paired Gate 全过；
- normalized A0/A1 confirmed harm 总数为 0；
- normalized confirmed harm 总数严格少于 raw 两轨总数。

### `stable_normalized_candidate_without_tail_contrast`

normalized candidate 三组 Gate 全过，但 raw confirmed harm 同样为 0，因此没有
直接证据证明 weighting 减少了 tail。它仍可作为稳定工程候选进入独立复验，但
结论必须写“稳定候选、未证明 tail mitigation”。

### `sample_tail_mitigation_not_supported`

其余全部结果，包括 normalized absolute/paired 任一失败或任一 normalized track
存在 confirmed harm。

以上三种科学分类只要工程检查完整，命令都正常退出；不能把负分类冒充 crash。

## 9. 一次性独立复验 E.9b（现只冻结，不运行）

E.9b 只在 E.9a 的
`independent_replication_candidate=true` 时解锁。E.9a runner 不解码、不训练、
不 probe 下列样本，也不会自动启动 E.9b。

冻结 train 排序 `17–28`：

| 位置 | Episode | Base sample ID |
| ---: | --- | --- |
| 17 | 5 | `1fc95daceda870a85bb86922ab9616fbafbe855cf8ba4087a9e24a4fba0ff15c` |
| 18 | 8 | `8905a37f8ae459be86fc1b32038978b31e2e76705c61d8376b6197047eb0650e` |
| 19 | 15 | `a10b86b1ab484588bd9dc3123b453bba8e32d2e1a299ec70119b6eafc96d6d63` |
| 20 | 9 | `0f4df424468f65f5d811a534f66667239f6e5491a54e9b4dfbe3d4155fa54456` |
| 21 | 16 | `3adec4471f56081985baeb57d428088594897e574c9dc932cad3950a909ab702` |
| 22 | 22 | `8f192e8bb4efccda60df55a6144aec7c4be8d4a1a3486757de88c7f094a69361` |
| 23 | 6 | `011cd4c8d0b8733b64f3bb6972d3e9cf729624fd86410ed93e4beadb3782f7f5` |
| 24 | 39 | `7b6f6128910d00fa642e1558255a6d109870cb691ea056ba1bae08537bd3a6ab` |
| 25 | 20 | `a57634c75a6ff93a7c9c403cac92a165dfd17c989f3c8af796c487b360717bb9` |
| 26 | 17 | `79f40b100893a2f47bc0fc20dfef740e60732710e2b2de8cc366b01ec41c6835` |
| 27 | 18 | `12bbc8a48340d1ea1d4f144c34c5cd1896321587038259cbf231824fa4bc4255` |
| 28 | 40 | `201476e51f22ba7a3cd26d3eb56013f4fa1fefe87a7c956e7a7bfdc820072613` |

- cohort SHA：
  `0218d90eb6455d3297857423bfd34109469f308db9f69d5adeee02146ee42324`；
- flows：Block A `107..122`、Block B `123..138`；
- identity SHA（0-based sample index）：
  `d5aeb3df50bbf11940ba545318327fd08df7f1e83dc27d7e3026ff6ed70b4f64`；
- zero-weight positions：
  `(1,113), (2,113), (3,120), (3,121), (8,133), (9,131)`。

E.9b 只读 E.9a normalized A0/A1 step-200 checkpoint 和共同 zero-gate initial：

- 不训练；
- 不读取 raw tracks；
- `12×32×3=1,152` forward objectives；
- FWER comparisons：`12×2=24`；
- one-sided quantile：`0.05/24`；
- A0 reduction `>=0%`；
- A1 reduction `>=10%`；
- 每轨 `>=9/12` sample 不变差，保持原 75% 比例，不能降为 8/12；
- normalized A1 vs A0：mean 至少低 10%，且至少 9/12 sample 不高；
- 两轨 confirmed harm 都为 0；
- catastrophic 与尺度门槛不变。

只有 E.9b 全过才允许把候选写成“一次性独立 demonstration-level 复验通过”。
若 E.9b 失败，保留负结果，不再从这 12 条挑子集、增 flow 或改门槛。

## 10. 后续阶段保持锁定

E.9a 和 E.9b 都不是完整 Gate E。以下继续锁定：

- 完整 28 train / 4 development Gate E；
- A2/A4；
- Phase F；
- ID/OOD rollout；
- K 选择与论文 OOD claim。

只有 E.9b 通过后，才能另行冻结新的完整 Gate E 配方、Run ID、train/dev
selection 和停止规则。完整 Gate E 通过后才讨论 A2/A4。

## 11. 工程拒绝条件

任一项出现即 invalid：

1. project 或 FastWAM worktree dirty；
2. E.8 冻结工件 SHA 改变；
3. E.6 cohort、Phase D split/payload 或 cache identity 改变；
4. raw/normalized 不是同一新 train/held-out schedule；
5. A0/A1 配方除 active K 外不匹配；
6. normalized weights、和、SHA 或公式改变；
7. reserved 17–28 被 E.9a 解码、训练或 probe；
8. frozen Fast-WAM requires-grad、gradient 或参数 SHA 改变；
9. future RGB、development、OOD、success 或 rollout outcome 被读取；
10. NaN/Inf、zero-weight loss 非零、checkpoint/resume provenance 不一致；
11. 单卡显存达到 43 GiB；
12. 科学判定使用 step 100 或任何非 200 endpoint。

## 12. 归档运行边界

以下 v1 命令已经永久停用：

```bash
bash scripts/run_thought3_phase_e9_sample_tail_mitigation.sh
```

归档 runner 现在只返回 exit code 2，避免 v1 被误 resume 或覆盖。不得把 v1
模型加载、数据准备或 failure log 当作有效 E.9a 结果。v2 已从全新输出目录
执行，其状态以 [E.9a-v2 结果](thought3_phase_e9_v2_report.md)为准。
