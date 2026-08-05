# Thought5：Camera-Equivariant Geometry Alignment

更新日期：2026-08-05

Thought5 是独立于 Thought1–4 的机制干预阶段。它不重新解释或覆盖既有
正式结果，而是针对 Thought4 formal v6 唯一解锁的
`camera_equivariance_gap` 分支，实现 **Fast-WAM-GeoEq**：训练期
Geo-REPA 几何表征对齐，加推理期 relative-pose/camera-ray conditioning。

当前状态：代码审计、CPU/mock contract dry-run、GPU smoke v3/v4/v5 和三卡
pilot v4 已完成。pilot v3 是权重加载前的 launcher 失败；修复后的 pilot v4 是
有效的单 task 方向性实验。其 training、representation、future geometry、future
utility 和 rollout 五项方向门禁全部为 false，`formal_unlocked=false`，因此
formal v2 保持 **NOT RUN**。这是一项有效的 Pilot 停止结论，但不是 H1/H2/H3
的正式多任务否证。
当前可执行协议为 v2；v1 在任何真实 GPU 运行前因 Phase 5-B evaluator 公平性
复核而废止，仅保留 `NOT RUN` scaffold。

## 文档入口

- [代码与方法审计](audit_and_method.md)：真实特征路径、标签能力、模型结构和参数边界。
- [冻结实验协议](protocol.md)：cohort、对照、统计规则、判定与停止规则。
- [三卡执行调度预注册](three_gpu_execution_preregistration.md)：等价性、固定波次和 ETA。
- [子进程启动修复预注册](worker_launcher_hotfix_preregistration.md)：pilot v3 失败审计、修复边界和新 namespace。
- [Pilot v4 结果登记](pilot_v4_results.md)：五项 Gate、关键数值、论文边界和停止决定。
- [运行手册](runbook.md)：audit、单卡 smoke、2/3 卡 pilot、3/4 卡 formal。
- [论文证据链与结果模板](report.md)：Thought1→5 叙事、15 个最终问题及可写/不可写结论。

机器权威工件位于：

```text
outputs/thought5/phase5_camera_equivariant_geo_repa_v2/
```

Pilot v4 权威工件位于：

```text
outputs/thought5/phase5_camera_equivariant_geo_repa_pilot_v4/
```

审计别名位于：

```text
outputs/thought5/phase5_audit_report_v2.md
```

## 研究问题

```text
Camera-equivariant geometry alignment
        ↓ H1
Camera OOD geometry gap 缩小
        ↓ Phase 5-B
K=1 future geometry 改善
        ↓ H2
future sensitivity 转化为 held-out utility
        ↓ H3
Camera OOD success 提高且 Clean 不劣
```

只有 H1、H2、H3 同时通过，并且 B1 matched fine-tuning 与 G4 shuffled
geometry 均不能解释收益，才允许登记 `full_mechanism_support`。即使通过，
也只能说这是重要机制之一，不能声称唯一或充分原因。
