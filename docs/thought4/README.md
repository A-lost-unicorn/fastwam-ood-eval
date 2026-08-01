# Thought4：Geometry–Action Gap Diagnosis

Thought4 是冻结官方 Fast-WAM 的诊断阶段。它承接 Thought1–3 的
Future Sensitivity–Utility Gap，但不训练新策略、不运行成功率 rollout，也不把
probe 结果写成 OOD improvement。

## 当前状态

| 项目 | 状态 |
| --- | --- |
| 10 项代码/数据审计 | COMPLETE |
| hooks、labels、probe、intervention、decision | IMPLEMENTED |
| Thought4 CPU/mock 单测 | COMPLETE（41 passed） |
| 全项目回归 | COMPLETE（438 passed；5 条 NVML 环境 warning） |
| smoke/formal dry-run | COMPLETE（严格零写入） |
| 真实单卡 smoke | v1/v2 工程失败；v3 **PASSED（未覆盖 Robot-init）**；v4 **ENGINEERING FAILED（observation path）**；v5 **INTERRUPTED / RESUME BUG（无科学结果）**；v6 **NOT RUN** |
| 正式 64-state diagnosis | v1 工程失败；v2/v3 未运行且由新代码身份取代；v4 **NOT RUN** |
| Geo-REPA / SE(3)-Align | **NOT IMPLEMENTED（按协议锁定）** |

## 文档入口

- [代码与数据审计](code_data_audit.md)
- [冻结研究协议](protocol.md)
- [实现与数据流](implementation.md)
- [运行手册](runbook.md)
- [实验记录与论文表格](experiment_record.md)
- [卡点与排错](troubleshooting.md)

## 研究链路

```text
Camera OOD failure
  → OOD future consistency degradation
  → future-content action sensitivity
  → no held-out future utility
  → geometry/action gap localization          ← Thought4
  → one targeted repair
  → future preregistered Camera OOD validation
```

Phase 4 只允许输出以下一个分类：

1. `video_geometry_representation_gap` → Geo-REPA；
2. `world_action_interface_gap` → SE(3)-Align；
3. `camera_equivariance_gap` → Geo-REPA + relative pose/ray equivariance；
4. `geometry_hypothesis_not_supported` → 返回预处理/坐标/shortcut 审计。

正式结果产生前不能预先选择方法。
