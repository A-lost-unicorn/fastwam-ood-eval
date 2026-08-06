# Phase 6 审计与实现说明

## 唯一 Adapter

- Checkpoint：`outputs/thought3/phase2_full_28_4_a0_a1_v1/tracks/a1/checkpoints/step_00000200`
- `adapter.safetensors` SHA-256：`0ebff4705039c4ca0a1e77330a9480f0ed4b6bc0b21235b447153417b64730b0`
- Adapter state SHA-256：`cf8c7d4c2aa7bafef37e4e52719481f3da5ecf6909372f74feb4e4de0e159dd7`
- 参数数：1,371,137；Phase 6 中全部 `requires_grad=false`。
- 注入位置：`model.action_expert.action_encoder` 输出之后、Action DiT blocks 之前。

选择规则是固定 A1/K=1/step 200，不按 loss 或文件新旧选择。Backbone checkpoint SHA-256 为 `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579`。

## Scheduler

公式为 `phi(u,s)=s*u/(1+(s-1)u)`，`s=5`。官方实现位于 `third_party/FastWAM/src/fastwam/models/wan22/schedulers/scheduler_continuous.py`。

20 个 BF16 timestep 为：

`1000, 988, 980, 964, 952, 936, 920, 904, 884, 860, 832, 804, 768, 728, 680, 624, 556, 468, 358, 208`。

因此 Fsigma 的 Adapter 计划调用数为 17/20；F0 为 20/20；B0 为 0/20。

## 冻结任务选择

规则是排除 Thought3/4/5 使用的 `libero_goal/0`，其余按 canonical task ID 升序取每套前两个。选择结果为：

- `libero_spatial/0, 1`
- `libero_object/0, 1`
- `libero_goal/1, 2`
- `libero_10/0, 1`

选择过程不读取 baseline success、future utility、难度或 rollout outcome。每个任务的 Camera variant 由 BDDL stem 和 LIBERO-Plus `Camera Viewpoints` 分类匹配。

## 当前数据 blocker

本地只有 `libero_goal_no_noops_lerobot`，缺少另外三套 demonstrations。任务和 Camera variants 可唯一定位，但三个 suite 的正式 episode IDs 无法冻结。因此 v1 审计允许使用未使用的 `libero_goal` task 做 Phase 6A 技术 smoke，却拒绝 Phase 6B 与后续 rollout。

这不是负向科学结果，也不是代码崩溃。安装三套数据后应使用新的不可变协议 namespace 重新冻结完整 episode provenance，不能覆盖 v1 manifest。

