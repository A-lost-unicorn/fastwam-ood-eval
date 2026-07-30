# Thought3 研究加速路线：从在线动作敏感性到 directional OOD

更新日期：2026-07-30

当前节点：Phase 0 完成；Phase 1 实现完成、真实 GPU 待运行

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
        ├─ A 内容敏感 ──→ Phase 2：28/4 A0/A1
        │                    ↓
        │              Phase 3：240 paired Clean/OOD pilot
        │                    ↓（仅正向）
        │              Phase 4：A2/A4 + 正式多 seed
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

在首次 Phase 1 结果前禁止新增 flow-variance、checkpoint-trajectory、
sample-weight、LR 或 surrogate-threshold Gate，也不训练 A2/A4。

## 3. Phase 2 草案：完整 28/4 A0/A1

仅当 Phase 1 分支 A 时允许执行。运行前另行冻结正式配置与 commit；本节只是
设计草案，不是启动授权。

### 配方

因为 E9a-v2.1 audit valid，默认采用 normalized sample-loss recipe，理由仅是：

- raw A0 confirmed harm 为 2；
- normalized A0/A1 confirmed harm 均为 0。

必须披露该 recipe 来自 post-E8 engineering development；E9 科学 Gate 并未
通过。禁止根据 Phase 1 动作差异大小重新选择 raw/normalized。

### 固定实验

- 完整 `libero_goal/task_0` split：28 train / 4 development；
- 只训练 A0 和 A1；
- 唯一 LR、Adapter 结构、train seed 与更新预算；
- A0/A1 完全匹配 sample/flow schedule；
- development 只用于运行前冻结的 checkpoint rule；
- 不读 OOD，不根据 rollout success 调参；
- 不训练 A2/A4，不做 LR/sample-weight sweep。

### 通过边界

- train/dev loss finite；
- A1 dev 保持冻结方向，或明显优于 A0；
- 无 catastrophic 数值异常；
- frozen Fast-WAM SHA 不变；
- 完整 checkpoint 上再次保留 Phase 1 内容敏感性；
- latency/memory 可运行。

目标是产生进入 directional pilot 的工程候选，不重建 E5–E9 式“所有 surrogate
样本必须完美”的总门禁。

## 4. Phase 3 草案：最小 Clean/OOD paired pilot

仅在 Phase 2 完成后执行。建议在 manifest 冻结前采用以下 240-rollout 预算：

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

## 5. 从现在到 directional OOD 的最短路径

1. 单卡运行一次 Phase 1，获得 A/B/C。
2. 只有 A：冻结并训练一次 28/4 A0/A1。
3. 在完整 checkpoint 上复验一次 K=1 内容敏感性。
4. 冻结 240-rollout paired manifest。
5. 运行 B0/A0/A1/A-shuffle 的 Clean/camera/robot-init pilot。
6. 登记正向、动作但无 success、或负向三类结论。

最短路径中没有 E9b、新 LR sweep、新 flow Gate、A2 或 A4。

## 6. 当前论文边界

当前已经支持：

- Fast-WAM 在 Clean→LIBERO-Plus OOD 出现显著成功率下降；
- OOD 下 future–realized 自动一致性 proxy 变差；
- K=1 A1 离线 action-loss 信号在 E5/E6 两个 cohort 出现；
- E9 normalization 有 tail-stabilization signal，但未过 paired 门槛。

当前尚不支持：

- K=1 future 的内容确实进入动作决策（等待 Phase 1 GPU）；
- K=1 提高 Clean/OOD success；
- future 对失败的因果解释；
- K=1 优于 K=2/K=4；
- 任何 A2/A4 或正式多 seed 结论。
