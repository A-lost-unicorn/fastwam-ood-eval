# Thought4：Geometry–Action Gap Diagnosis

Thought4 是冻结官方 Fast-WAM 的诊断阶段。它承接 Thought1–3 的
Future Sensitivity–Utility Gap，但不训练新策略、不运行成功率 rollout，也不把
probe 结果写成 OOD improvement。

## 当前状态

| 项目 | 状态 |
| --- | --- |
| 10 项代码/数据审计 | COMPLETE |
| hooks、labels、probe、intervention、decision | IMPLEMENTED |
| Thought4 CPU/mock 单测 | COMPLETE（46 passed） |
| 全项目回归 | COMPLETE（443 passed；5 条 NVML 环境 warning） |
| smoke/formal dry-run | COMPLETE（严格零写入） |
| 真实单卡 smoke | FP32 arithmetic v8 **PASSED / NON-SCIENTIFIC**；真实 BF16 bitwise correct |
| 正式 64-state diagnosis | v6 **FORMAL COMPLETE**；分类 `camera_equivariance_gap` |
| 下一方法 | 只解锁 **Geo-REPA + relative pose / camera-ray equivariance**；尚未实现或评测 |

## 文档入口

- [代码与数据审计](code_data_audit.md)
- [冻结研究协议](protocol.md)
- [formal v6 FP32 修复预注册](fp32_subspace_v6_preregistration.md)
- [formal v6 正式结果](formal_v6_results.md)
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

formal v6 已按上述冻结规则输出第三类；这只决定下一研究分支，不是方法效果。

## formal v6 FP32 修复冻结说明

formal v4 在第 2 个排序状态因 Clean prefix 与 parquet EEF 超出 3 cm 阈值而
停止。对原 64 个冻结状态的只读审计显示 56 通过、8 失败；因此这不是单条偶然
误差，也不能靠略微放宽阈值处理。v5 保留原 64 个 state identity 和原
3 cm / 15° 阈值，把对齐改为完整 QC 披露；动作—运动标签则从每个真实 simulator
输入状态 `t` 重放冻结 demonstration actions 得到。它不读取 future RGB、success、
OOD 或 policy outcome，也不筛除/替换任何状态。

simulator-replay smoke v7 随后通过，但 formal v5 在真实 BF16 cache 上执行
geometry reconstruction 时触发旧 `5e-4` hard check。根因是旧实现把 FP32 basis
降为 BF16 后完成整段 subspace arithmetic；v7 只验证 raw identity replacement，
没有覆盖真实 projection/reconstruction，所以它不能为修复后的 formal 解锁。

已预注册单变量工程修复：coordinates、residual 和 reconstruction 全部用 FP32，
只在 consumer 边界 cast 一次 BF16；correct control 必须 `torch.equal` 且
input/output SHA 相同，绝不放宽阈值。新执行顺序严格为 `smoke v8 → formal v6`。
formal v5 原目录冻结，不覆盖、不 resume；v6 还会在 intervention 前先落盘 probe
结果。该顺序已经完整执行：36/36 geometry shuffle 超过 replay floor，Camera
paired geometry gap 在三个 seed 均显著且大于 Lighting，冻结分类为
`camera_equivariance_gap`。完整数字、SHA 与完整性 caveat 见
[formal v6 正式结果](formal_v6_results.md)。
