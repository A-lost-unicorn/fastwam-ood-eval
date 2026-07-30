# Thought3 研究加速路线：从在线动作敏感性到 directional OOD

更新日期：2026-07-30

当前节点：Phase 0 完成；Phase 1 进入分支 A；Phase 2 双卡完整完成但
development direction 为负；路线按预注册规则停止在 Phase 3 之前

## 1. 为什么改线

E5–E9 已经证明 Adapter、冻结训练、checkpoint、paired objective 与离线 loss
信号可以工作，但连续增加 flow、checkpoint 与 sample-weight surrogate Gate
仍没有回答论文核心问题。现在主线只保留能更接近下游结论的实验：

```text
Phase 0：E9 工件账本修复（已完成，不阻塞）
        │
        └──────────────┐
                       ▼
Phase 1：K=1 correct/null/shuffle 在线动作反事实
        │
        ├─ A 内容敏感（已观测）→ Phase 2：28/4 A0/A1（已完成）
        │                    ↓
        │              direction 未观察到（实测）
        │                    ↓
        │              停止 Phase 3/A2/A4/OOD 扩展
        │
        ├─ B 仅 presence ─→ 一次单变量结构修复 → 只复验一次 Phase 1
        │
        └─ C 无实质响应 ─→ 停止 Adapter-only 路线
```

## 2. 不再阻塞主线的旧 Gate

以下结果继续保留在证据账本，但不再要求“全部通过”才允许 Phase 1：

- E5/E6 的 A0 stability 失败；
- E7 checkpoint trajectory；
- E8 flow variance panel；
- E9 normalized paired 门槛；
- E9b reserve replication。

Phase 1 已于 2026-07-30 得到分支 A，Phase 2 随后按唯一配方完成并得到有效
离线负结果。仍禁止新增 flow-variance、checkpoint-trajectory、sample-weight、
LR、K 或 surrogate-threshold Gate，也不训练 A2/A4。

## 3. Phase 2 已完成：完整 28/4 A0/A1

Phase 1 满足分支 A后，本阶段按冻结 config、单配方、flow namespace、update
budget 和 step-200 rule 完整运行。A0/A1 各完成 200×28 objectives，12/12
hard checks 全部通过。

### 配方

因为 E9a-v2.1 audit valid，默认采用 normalized sample-loss recipe，理由仅是：

- raw A0 confirmed harm 为 2；
- normalized A0/A1 confirmed harm 均为 0。

必须披露该 recipe 来自 post-E8 engineering development；E9 科学 Gate 并未
通过。禁止根据 Phase 1 动作差异大小重新选择 raw/normalized。

### 固定实验

- 完整 `libero_goal/task_0` split：28 train / 4 development；
- 只训练 A0 和 A1；
- 唯一 LR `3e-4`、Adapter 结构、train seed `3407`；
- 200 optimizer updates，每 update 28 个完整 cohort objective；
- calibration flows `139..170`，development flows `171..202`；
- training slots `50001..55600`；
- A0/A1 完全匹配 sample/flow schedule；
- development 只评估 step 0/200，主 checkpoint 固定 step 200、无 fallback；
- 不读 OOD，不根据 rollout success 调参；
- 不训练 A2/A4，不做 LR/sample-weight sweep。

### 冻结结果

- A0 development reduction：`+1.845%`；
- A1 development reduction：`−1.712%`；
- A1 final mean 比 A0 高 `3.624%`；
- 4/4 development sample 的 A1 loss 高于 A0；
- 分类：`training_valid_dev_direction_not_observed`；
- `phase3_unlocked=false`。

为使用两张空闲卡而不引入 DDP 混淆，先在卡 1 生成唯一 normalized weight
artifact，再让 A0/A1 分别在卡 1/2 并行。结果说明工程链路有效，但没有产生
进入 directional pilot 的候选。完整协议与结果见
[thought3_phase2_full_28_4_protocol.md](thought3_phase2_full_28_4_protocol.md)、
[thought3_phase2_full_28_4_report.md](thought3_phase2_full_28_4_report.md)。

## 4. Phase 3 草案：未解锁、仅保留历史设计

该草案原本只在 Phase 2 direction 正向时执行。本次 direction 为负，因此不得
生成 manifest 或运行下列 240-rollout 预算：

```text
4 groups：B0 / A0 / A1 / A-shuffle
× 3 environments：Clean / camera-view / robot-init
× 5 个预先固定代表任务
× 4 个 paired episode seeds
= 240 rollouts
```

选择 camera-view 是因为阶段一已知其为最脆弱类别；robot-init 是在查看 A1 pilot
结果前由阶段一总体排序选出的非相机代表扰动。任务和 episode seeds 必须在任何
pilot outcome 前冻结，并与未来正式 Phase G seed namespace 分离。

### 配对约束

- 四组共享 task、environment、episode seed、初始状态与控制预算；
- A-shuffle 使用冻结的一一 donor mapping；
- 多 GPU shard 不重复且可 resume；
- 不覆盖阶段一或未来 Phase G output；
- 每条失败保存视频与精确 action/future latency。

### 指标

- Clean/OOD 分别报告 success；
- A1−A0、A1−A-shuffle exact paired difference；
- OOD gain 是否不弱于 Clean gain；
- task-cluster bootstrap CI；
- action/future/total latency、peak memory；
- failure video manifest。

它只能登记为 `DIRECTIONAL PILOT`，不能冒充正式多 seed 论文结论。

### Pilot 决策

- **正向**：A1>A0、A1>A-shuffle、OOD gain 不弱于 Clean、latency 可接受；
  才允许 Phase 4 训练 A2/A4。
- **动作有效但 success 无收益**：登记“future 进入动作，但 K=1 未转化为闭环
  收益”，停止 A2/A4，除非出现明确 K=1 quality 机制证据。
- **负向**：A1 不优于 A0/shuffle 或 latency 不可部署；停止扩展。

## 5. 冻结停止记录

1. ~~单卡运行一次 Phase 1，获得 A/B/C。~~ 已完成，结果为 A。
2. ~~冻结并运行唯一 Phase 2。~~ 已完成，结果为 negative direction。
3. ~~在正向时复验完整 checkpoint。~~ 条件未满足，不执行。
4. ~~在复验通过时冻结 240-rollout manifest。~~ 条件未满足，不执行。
5. 登记并保留负结果，停止 E9b、新 LR/weight/K/flow Gate、A2/A4 与 OOD
   Adapter pilot。

## 6. 当前论文边界

当前已经支持：

- Fast-WAM 在 Clean→LIBERO-Plus OOD 出现显著成功率下降；
- OOD 下 future–realized 自动一致性 proxy 变差；
- K=1 A1 离线 action-loss 信号在 E5/E6 两个 cohort 出现；
- E9 normalization 有 tail-stabilization signal，但未过 paired 门槛。
- 固定 E6 A1 checkpoint 在八条同 task train sample 上对 K=1 future 内容具有
  技术动作敏感性：correct-null、correct-shuffle 与 action-hash 均为 `8/8`。
- 完整 28/4 offline ablation 中，A0 改善 `1.845%`、A1 恶化 `1.712%`，
  A1 比 A0 高 `3.624%`；future sensitivity 未转化为该配方的 held-out utility。

当前尚不支持：

- Phase 1 的小幅动作变化会转化成闭环轨迹或成功率差异；
- K=1 提高 Clean/OOD success；
- K=1 在其他 task、seed、结构或 K 设置中也无效；
- future 对失败的因果解释；
- K=1 优于 K=2/K=4；
- 任何 A2/A4 或正式多 seed 结论。
