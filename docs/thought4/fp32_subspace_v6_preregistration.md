# Formal v6：FP32 Subspace Arithmetic 工程修复预注册

登记时间：2026-08-03（formal v5 工程失败之后、smoke v8 与 formal v6 真实运行之前）
状态：`PRE-REGISTERED / IMPLEMENTED / NOT RUN`

本次变更只修复 Phase 4-C 的数值实现和工件提交顺序，不修改 Phase 4 的科学问题、
cohort、probe、层选择、干预 seed、统计门槛或方法分类规则。smoke v8 通过之前，
formal v6 保持锁定。

## 1. 触发事件与历史工件

simulator-replay smoke v7 已在 commit
`229a0f383d7638b1919aa6a08f5aa3ea999a5cfe`、物理 GPU 2 上通过；它验证了真实
BF16 cache capture 与原 tensor identity replacement，但没有执行 subspace
projection/reconstruction。其结果 SHA 为
`9d81d79afa9f3efcadf1a015f596f33e60414198b72f7c1e6cfa5a1322a1fbf9`。

formal v5 使用同一 commit，于 2026-08-03 03:03:08–04:13:40 UTC 运行。它已经
完成 64 个 base states、256 个四条件 render/label、12,544 条 feature，以及内存中
的 Video/Action probe panel；进入第一条 geometry-subspace intervention 时报告：

```text
InterventionRuntimeError: correct geometry reconstruction exceeded BF16 tolerance
```

formal v5 固定保留在
`outputs/thought4/phase4_geometry_action_diagnosis_v5/`，状态为 `error`。禁止删除、
覆盖、手工补结果或 `--resume`。该运行没有 intervention、method selection 或科学
classification；旧编排也尚未把已经算完的 probe panel 落盘，因此不能从 v5
恢复科学 probe 数字。

## 2. 根因与修复假设

旧代码把 FP32 正交 basis 转成 captured hidden 的 BF16 dtype，再执行 projection、
residual 和 reconstruction。该转换破坏了 basis 的数值正交性，使
`(h - P(h)) + P(h)` 在 BF16 中不再恢复输入。旧实现用 `5e-4` 阈值检查，因此在
真实 intervention 前 fail closed。

修复假设是纯数值工程假设：保持 hidden 与 basis 的投影、残差、坐标替换和重构
全部为 FP32，只在 Action consumer replacement 边界做一次 BF16 cast。correct
control 必须在这次输出 cast 后与原 captured BF16 tensor 逐位相同；不得通过提高
`5e-4`、动作 replay 或任何其他容差来替代该检查。

冻结协议字符串为：

```text
fp32_coordinates_residual_single_output_cast_bitwise_correct
```

## 3. smoke v8 技术 Gate

smoke v8 使用真实 `mot.video_kv_cache.15.v`，必须新增并同时通过：

1. capture tensor dtype 必须是 `torch.bfloat16`；
2. 从固定 seed 的 FP32 technical linear weight 构造 FP32 SVD basis；
3. 使用生产路径执行 BF16 capture → FP32 coordinates/residual/reconstruction →
   单次 BF16 replacement；
4. reconstruction 的 shape、dtype、device 必须保持不变；
5. `torch.equal(reconstructed, captured)` 必须为 true；
6. input/output tensor SHA 必须完全相同，cast 后 max-abs 必须为 `0.0`；
7. reconstructed tensor 必须真正送入 Action consumer，动作差异不能超过同一次
   smoke 实测 replay floor；
8. 所有检查和内部 SHA 写入 `smoke_result.json`，formal gate 会重算而不是只信
   `formal_unlocked=true`。

这里的动作 replay tolerance 只检查底层推理复现；它不能替代第 5–6 项的 tensor
逐位恢复硬门禁。

## 4. formal v6 工件顺序

formal v6 在 Phase 4-A/B probe panel 与 development-only intervention selection
完成后，必须在进入 Phase 4-C 之前原子写出并校验：

- `video_probe_results.json`；
- `action_probe_results.json`；
- `layer_summary.json`；
- `probe_stage_result.json`。

`probe_stage_result.json` 记录两个 panel、summary 和冻结 selection 的 canonical
SHA，并明确 `test_used_for_selection=false`、`future_rgb_read=false`、
`success_outcome_read=false`。后续 intervention 即使工程失败，已经完成的 probe
证据也保留；失败仍不得产生或补写 method classification。

## 5. 新身份与未改变项

| 字段 | 冻结值 |
| --- | --- |
| smoke config/output | `phase4_geometry_action_smoke_v8.yaml` / `phase4_geometry_action_smoke_v8/` |
| smoke config fingerprint | `81d3885ccb5b58806c1a729e509c039f6e1cb33a34ff242f8fa16785796149d7` |
| smoke planned cohort SHA | `a67ff85321dc684a80b853b58ab133905232a275e6d71255fd1c966b9a3d6c12` |
| formal config/output | `phase4_geometry_action_diagnosis_v6.yaml` / `phase4_geometry_action_diagnosis_v6/` |
| formal config fingerprint | `3b14a7d7fd09deda9253bb1cd9950d9c4b5bd0cdf9f124a4dfede22add5c24f6` |
| formal planned cohort SHA | `9af7cf7c1933fb1e5574099361f6d7dcc7500727480ecb4bbf010089f28d8f04` |
| formal underlying 64-state identity SHA | `9916e0444ccfca08bd3d87ead73c344974f733f9b0c5dcc5cd9996a2618f8e8b`（与 v5 相同） |
| subspace arithmetic | `fp32_coordinates_residual_single_output_cast_bitwise_correct` |

planned cohort SHA 包含 config fingerprint，所以 namespace 改变后会变化；上表的
underlying identity SHA 只对 task/episode/frame/split/sample identity 计算，证明
v5→v6 没有重新抽样。

以下项目全部保持 formal v5 不变：64-state 40/12/12 episode-safe cohort、四条件、
simulator-replay 标签、alignment disclosure、五个 Video layer、四个 Action hook、
20-step action denoising、probe model/seeds/optimizer/bootstrap、development-only
层选择、SVD energy/rank、donor mapping、action seeds、replay rule、统计阈值和四选一
method classification。仍不读取 future RGB、success、reward 或 OOD rollout。

## 6. 执行与判定边界

执行顺序只能是：

```text
clean preregistration commit
  → smoke v8
  → smoke v8 bitwise Gate PASS
  → 同一 project commit 的 formal v6
```

formal v6 使用全新 namespace，不能复制 v5 文件，也不能对 v5 执行 resume。若
smoke v8 的真实 BF16 bitwise check 失败，则登记新的技术负 Gate 并停止 formal；
不得放宽阈值。若 formal v6 在 probe 落盘后 intervention 再失败，probe 只能登记为
已完成的诊断中间证据，不能据此选择或宣称修复方法。
