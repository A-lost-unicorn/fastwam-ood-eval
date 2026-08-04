# Thought5：Camera-Equivariant Geometry Alignment

更新日期：2026-08-04

Thought5 是独立于 Thought1–4 的机制干预阶段。它不重新解释或覆盖既有
正式结果，而是针对 Thought4 formal v6 唯一解锁的
`camera_equivariance_gap` 分支，实现 **Fast-WAM-GeoEq**：训练期
Geo-REPA 几何表征对齐，加推理期 relative-pose/camera-ray conditioning。

当前状态：代码审计、CPU/mock contract dry-run 和真实 GPU smoke v3 已完成；
smoke v2 是 dtype 失败运行，pilot v2 只渲染约 6 秒后被人工中断，二者均为无效
技术尝试。三卡执行已预注册为 smoke v4 → pilot v3 → formal v2；这三个新阶段
均为 **NOT RUN**。因此 H1/H2/H3
以及最终机制分类尚未产生，不得把本目录中的方法设计写成效果结论。
当前可执行协议为 v2；v1 在任何真实 GPU 运行前因 Phase 5-B evaluator 公平性
复核而废止，仅保留 `NOT RUN` scaffold。

## 文档入口

- [代码与方法审计](audit_and_method.md)：真实特征路径、标签能力、模型结构和参数边界。
- [冻结实验协议](protocol.md)：cohort、对照、统计规则、判定与停止规则。
- [三卡执行调度预注册](three_gpu_execution_preregistration.md)：等价性、固定波次和 ETA。
- [运行手册](runbook.md)：audit、单卡 smoke、2/3 卡 pilot、3/4 卡 formal。
- [论文证据链与结果模板](report.md)：Thought1→5 叙事、15 个最终问题及可写/不可写结论。

机器权威工件位于：

```text
outputs/thought5/phase5_camera_equivariant_geo_repa_v2/
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
