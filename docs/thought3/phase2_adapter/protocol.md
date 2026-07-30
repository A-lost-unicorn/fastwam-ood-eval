# Thought3 Phase 2：完整 28/4 A0/A1 单配方训练协议

状态：`COMPLETED / VALID OFFLINE NEGATIVE RESULT / PHASE 3 LOCKED`

冻结日期：2026-07-30
配置：`configs/thought3/phase2_full_28_4_a0_a1.yaml`
配置 fingerprint：
`fabb96a97b7e137ca39a5477c2090deab1844909887b10cd22fa92ebbee66468`

## 0. 2026-07-30 calibration 中断与恢复边界

首次真实启动使用冻结项目 commit `96ace8b`。所有 calibration objective 已经
完成并原子落盘：

- train：`28 × 32 = 896` rows，文件 SHA
  `0f812c75...4b997`；
- development：`4 × 32 = 128` rows，文件 SHA
  `07f29ba3...3a5d`；
- normalized weight SHA：`4c36dece...1dc22`；
- train/development initial mean loss：
  `0.0049083332 / 0.0042341036`；
- frozen Fast-WAM SHA before/after 相同；
- future RGB、OOD 与 success 均未读取。

中断发生在 `calibration.json` 已写完之后、`artifact_manifest.json` 写入之前。
原因是 config 保留 repository-relative output path，而 safety helper 返回 resolved
absolute root；旧代码直接对二者调用 `Path.relative_to()`。这只是工件路径表示
错误，不是 model、objective、weight、显存或数据错误。

中断边界：

- calibration `run_status=failed`；
- A0/A1 track 均未创建；
- optimizer update、backward 和新 checkpoint 均为 0；
- 已保存的 1,024 rows 保留，禁止删除或另建 outcome-conditioned run。

恢复补丁只在 manifest bookkeeping 前统一 resolve path/root，并将相同 helper
用于 calibration、track 与 checkpoint 工件；不改变 config fingerprint、数据、
flow、loss、weight、LR、训练预算、endpoint 或门槛。恢复必须在原目录执行
`--resume`，先复验并复用已完成 rows，再启动 matched A0/A1。

恢复运行随后完整完成。固定 step-200 结果为 A0 reduction `+1.845%`、A1
reduction `−1.712%`；A1 final mean 比 A0 高 `3.624%`，且 4/4 development
sample 的 A1 loss 更高。冻结分类为
`training_valid_dev_direction_not_observed`，Phase 3 保持锁定。完整审计见
[thought3_phase2_full_28_4_report.md](report.md)。

## 1. 研究问题与进入条件

Phase 1 已在固定 E6 A1@3e-4 step-200 checkpoint 上得到
`future_content_sensitivity_observed`：

- B0 replay 与 formal null 精确相同；
- correct-null 超过 replay floor：8/8；
- correct-shuffle 超过 replay floor：8/8；
- correct/shuffle action hash 改变：8/8。

这证明 K=1 future 的具体内容进入了该工程 checkpoint 的动作计算，但没有证明
轨迹或成功率受益。Phase 2 只回答下一层问题：

> 该 K=1 离线信号能否从已消耗的 8-sample cohort 扩展到 Phase D 的完整
> 28 train / 4 development subset，并形成可进入小规模 rollout pilot 的候选？

Phase 2 不测 OOD、success 或 rollout，不比较 K=2/K=4，也不重新打开 E5–E9
surrogate Gate。

## 2. Recipe 选择披露

E9a-v2.1 audit 为 `audit_valid_scientific_failed`，因此按目标文件中运行前冻结的
规则选择 normalized sample-loss recipe：

```text
每条 train sample：
  在 zero-gate A0 初始状态上跨 32 个 calibration flows 求 mean loss L_i

权重：
  w_i = (1 / L_i) / mean_j(1 / L_j)
```

权重必须：

- 覆盖精确 28 条 train sample；
- finite、严格大于 0；
- 和为 28，即单位均值；
- 只计算一次，A0/A1 读取同一个冻结 SHA；
- 不 clipping、不加 epsilon、不做第二种权重或 sweep。

选择该 recipe 的原因仅是 E9 normalized A0/A1 均无 confirmed harm。必须同时
披露：

- 这是 post-E8 engineering development 后选择的 recipe；
- E9 normalized A1-vs-A0 paired advantage 为 8.274%，低于冻结 10% 门槛；
- E9 没有科学通过，E9b 仍锁定；
- Phase 1 action difference 的大小没有参与 recipe 选择。

## 3. 数据与泄漏边界

数据固定为 Phase D cache 的同一个 `libero_goal/task_0`：

| 项目 | 冻结值 |
| --- | --- |
| task | `open the middle drawer of the cabinet` |
| base sample | 32 个不同 demonstration episode 的 frame 0 |
| train | 28 |
| development | 4 |
| split fingerprint | `ea540295...951eb` |
| cache fingerprint | `63a70e1...d9c5` |
| future source | frozen Video DiT 从当前观测生成的 K=1 latent |
| future RGB | 0 |

允许读取：

- train/development 当前双相机 RGB 与 proprio；
- train/development action target 和 pad mask；
- Phase D 中 model-generated K=1 latent；
- development action loss。

禁止读取：

- 真实 future RGB 或 next observation；
- LIBERO-Plus/OOD；
- rollout success、failure video 或阶段一 outcome；
- Thought1/Thought2 输出作为训练输入；
- A2/A4；
- development 用于 checkpoint 选择、LR、权重、结构或预算调整。

该 28/4 subset 只有一个 task，属于 exploratory full-subset training，不代表完整
LIBERO 多任务训练分布。

## 4. 唯一训练配方

| 项目 | 冻结值 |
| --- | --- |
| 变体 | A0、A1 |
| Adapter | 1,371,137 参数，同一结构和 initialization seed |
| backbone | Video DiT/VAE/Action DiT 全冻结 |
| optimizer | AdamW |
| LR | 3e-4 |
| weight decay | 1e-2 |
| train seed | 3407 |
| optimizer updates | 200 |
| objectives/update | 28，每条 train sample 各一个 |
| objectives/track | 5,600 |
| checkpoint | 50/100/150/200，仅 step 200 是主终点 |
| checkpoint rule | `fixed_step_200_no_selection_no_fallback` |
| action denoise steps | 20，保持不变 |

A0/A1 必须共享：

- Adapter 初始 semantic SHA；
- 28 条 sample 的顺序；
- 每条 sample 的 normalized weight；
- 5,600 个 action noise/timestep identity；
- optimizer update 数和每 update 的 objective 数；
- LR、weight decay、seed、checkpoint interval；
- development sample 与 flow panel。

唯一处理变量是 future 输入：

- A0：正式 zero-future control；
- A1：对应样本的 K=1 model-generated latent。

## 5. Flow namespace

旧 E9a 与被锁定 E9b 的 flows `75..138` 不复用。

| 用途 | 冻结 flow |
| --- | --- |
| train-only weight calibration | 139..170，共 32 |
| fixed development endpoint | 171..202，共 32 |
| training | 50001..55600，共 5,600 |

训练 slot 映射：

```text
slot = 50000 + (optimizer_update - 1) × 28 + micro_index
```

`micro_index=1..28` 对应冻结 sample 顺序。A0/A1 的完整 identity schedule SHA
必须相同；任何错位、重复或遗漏均 fail closed。

## 6. 固定开发集规则

development 只在两个固定端点评估：

- step 0：calibration stage 的 exact zero-gate 基线；
- step 200：每条轨迹的唯一最终 checkpoint。

不评估 step 50/100/150 来挑 checkpoint，不允许 fallback。冻结方向规则为：

```text
A1 final mean loss < A0 final mean loss
且
A1 final mean loss < shared step-0 mean loss
```

这里没有新增可调百分比阈值。若方向不成立，登记
`training_valid_dev_direction_not_observed` 并在 Phase 3 前停止，而不是降低门槛、
换 checkpoint 或试第二个 LR。

## 7. 三段执行与双卡隔离

执行顺序固定：

```text
calibrate：GPU 1
    ↓ 冻结 28 个 weights + SHA
A0：GPU 1 ─────────┐
A1：GPU 2 ─────────┤ 并行、各自只见 logical cuda:0
                    ↓
finalize：CPU-only matched audit
```

不使用 DDP。两张卡分别加载一份 frozen Fast-WAM，A0/A1 写入不同 track
目录；这样没有 rank shard、梯度同步或重复 sample 的额外混淆。

calibration 额外加载一次模型，是为了让两条轨迹读取完全相同的一份权重，而不是
在两张卡上各自重算后再容忍浮点差异。

## 8. 工程硬检查

calibration：

- Phase D、Phase 1、E9 audit 和 62 个 Phase 1 工件 SHA 全部复核；
- 28/4 identity、cache/split fingerprint 精确；
- zero gate 对所有 calibration/development objective 的 gated delta 精确为 0；
- 权重 finite/positive/unit mean；
- future RGB=0；
- Fast-WAM frozen SHA 前后相同。

每条训练轨迹：

- optimizer 只含 Adapter parameters；
- 第 1 update gate gradient 非零、non-gate gradient 为 0；
- 第 2 update projector/attention/non-gate gradient 非零且 finite；
- backbone 无 gradient；
- loss/gradient finite；
- checkpoint 与 objective/update metric prefix SHA 绑定；
- Adapter/optimizer checkpoint round-trip；
- 200×28 完整；
- step-200 fixed development 完整；
- frozen Fast-WAM SHA 前后相同；
- 峰值低于 23.8 GiB。

finalize：

- A0/A1 sample、weight、identity schedule、observed flow schedule 精确匹配；
- 两轨均固定 step 200；
- development 不超过初始 mean 的 10× catastrophic safety bound；
- 不读取 OOD/success/future RGB；
- 不直接解锁 Phase 3。

## 9. Phase 2 分类与后续锁

只允许：

1. `phase2_engineering_invalid`
   - 修复唯一失败的工程 invariant；
   - 不根据 loss 结果改变配方。
2. `training_valid_dev_direction_not_observed`
   - 登记负方向；
   - Phase 3 保持锁定。
3. `training_valid_pending_full_checkpoint_online_sensitivity`
   - 仅说明训练与 development 方向满足；
   - Phase 3 仍锁定；
   - 下一步必须用完整 A1 checkpoint 复验一次 online
     correct/null/shuffle。

只有完整 checkpoint 的 action sensitivity 仍存在，且 latency/memory 可运行，
才把 Phase 2 登记为可进入最小 Clean/OOD pilot 的工程候选。

## 10. Dry-run

以下命令不会 import torch/safetensors，不加载 checkpoint/Fast-WAM，也不写输出：

```bash
.conda/envs/fastwam-ood/bin/python -m fastwam_ood_eval.cli \
  thought3-train-phase2-full \
  --config configs/thought3/phase2_full_28_4_a0_a1.yaml \
  --stage calibrate \
  --dry-run
```

`--stage` 也可为 `A0`、`A1`、`finalize`。四个 stage 不会隐式串联。

## 11. 唯一真实运行命令

要求两张空闲 RTX 4090；示例使用物理卡 1、2：

```bash
CONFIRM_THOUGHT3_PHASE2_FULL=YES \
THOUGHT3_GPU_IDS=1,2 \
bash scripts/run_thought3_phase2_full_28_4.sh
```

若 calibration 或任一 track 中断，保留原目录，使用完全相同的 config：

```bash
CONFIRM_THOUGHT3_PHASE2_FULL=YES \
THOUGHT3_GPU_IDS=1,2 \
bash scripts/run_thought3_phase2_full_28_4.sh --resume
```

resume 会：

- 验证已完成 calibration 的所有文件 SHA；
- 从最近的 50-step Adapter+optimizer checkpoint 恢复；
- 验证 checkpoint 与 metric prefix、sample/flow/weight identity；
- 已完成的另一条 track 只读验证后跳过；
- 不提供 `force`、替换 checkpoint 或忽略 checksum 的选项。

## 12. 时间、显存与磁盘预估

已有 E9 实测为 8 objectives/update、约 4.95 s/update。线性外推：

- calibration：1,024 objectives + 一次约 9 分钟模型加载，约 18–25 分钟；
- 单 track：5,600 train objectives + 128 dev objectives + 模型加载，约
  65–80 分钟；
- A0/A1 双卡并行；
- 总墙钟预计约 1.5–2 小时。

已知峰值：

- Fast-WAM load：约 23,679.5 MiB；
- Adapter training：约 13,277.4 MiB。

因此每个进程只暴露一张空闲 4090，并设 23.8 GiB fail-closed ceiling。预计输出
主要来自两轨 4 个 Adapter/optimizer checkpoint、5,600-row metrics 和审计工件，
约数百 MiB，远小于 future RGB/video 数据规模。

## 13. 运行后必须记录

- calibration/result/artifact SHA；
- 28 个 initial loss、weight 与 weight SHA；
- A0/A1 初始/最终 Adapter SHA；
- 5,600-flow identity/observed schedule SHA；
- train/update/development metric SHA；
- step 0、step 200 development mean 与逐 sample 表；
- A1−A0 final mean；
- gate、首次 non-gate gradient update；
- update time、peak allocated/reserved；
- frozen Fast-WAM SHA before/after；
- checkpoint round-trip；
- classification 与下一步；
- negative result 也完整登记。

## 14. 当前可证明与不可证明

协议、代码与完整运行当前可以证明：

- Phase 2 的数据、配方、schedule、endpoint 和停止规则已在 outcome 前冻结；
- 双卡并行不会改变 A0/A1 的 sample/flow/weight 配对；
- checkpoint 与 objective/update metric prefix SHA 绑定；
- Adapter/optimizer checkpoint round-trip；
- 两轨均完成 200×28，step-200 development 完整；
- frozen Fast-WAM SHA 前后相同；
- 12/12 hard checks、32/32 manifest descriptor 与 8/8 checkpoint provenance
  通过；
- A1 未满足冻结 direction，Phase 3 已按规则锁定。

仍不能证明：

- 完整 checkpoint 仍有 future-content action sensitivity；
- A1 提高 Clean/OOD success；
- K=1 优于 K=2/K=4；
- 本次一个 task、一个 seed、4 条 development sample 可推广到其他设置；
- Fast-WAM 在 OOD 中普遍不需要 future。
