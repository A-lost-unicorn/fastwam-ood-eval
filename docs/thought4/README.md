# Thought4：Geometry–Action Gap Diagnosis

Thought4 是冻结官方 Fast-WAM 的诊断阶段。它承接 Thought1–3 的
Future Sensitivity–Utility Gap，但不训练新策略、不运行成功率 rollout，也不把
probe 结果写成 OOD improvement。

## 当前状态

| 项目 | 状态 |
| --- | --- |
| 10 项代码/数据审计 | COMPLETE |
| hooks、labels、probe、intervention、decision | IMPLEMENTED |
| Thought4 CPU/mock 单测 | COMPLETE（43 passed） |
| 全项目回归 | COMPLETE（440 passed；5 条 NVML 环境 warning） |
| smoke/formal dry-run | COMPLETE（严格零写入） |
| 真实单卡 smoke | v1/v2/v4 工程失败；v3 **PASSED（无 Robot-init）**；v5 中断；v6 **PASSED / NON-SCIENTIFIC**；simulator-replay v7 **NOT RUN** |
| 正式 64-state diagnosis | v1 工程失败；v2/v3 未运行；v4 **ENGINEERING FAILED（pre-model alignment）**；simulator-replay v5 **PRE-REGISTERED / NOT RUN** |
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

## simulator-replay v5 冻结说明

formal v4 在第 2 个排序状态因 Clean prefix 与 parquet EEF 超出 3 cm 阈值而
停止。对原 64 个冻结状态的只读审计显示 56 通过、8 失败；因此这不是单条偶然
误差，也不能靠略微放宽阈值处理。v5 保留原 64 个 state identity 和原
3 cm / 15° 阈值，把对齐改为完整 QC 披露；动作—运动标签则从每个真实 simulator
输入状态 `t` 重放冻结 demonstration actions 得到。它不读取 future RGB、success、
OOD 或 policy outcome，也不筛除/替换任何状态。

新执行顺序严格为 `smoke v7 → formal v5`。旧 smoke v6 虽已通过，但不能为改过
代码身份的 formal v5 解锁。
