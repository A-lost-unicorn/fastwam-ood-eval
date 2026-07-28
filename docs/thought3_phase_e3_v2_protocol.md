# Thought3 Gate E.3 v2：Held-out Multi-flow 修复版协议

状态：`EXECUTED / VALID FAILED GATE`
证据等级：`ENGINEERING DIAGNOSTIC / NOT MODEL EFFECT`

> 2026-07-28：v2 已完整执行 320/320 forward；实现与审计检查全部通过，但没有
> A0/A1 共同 eligible LR，故 Gate E.3 有效失败。结果、边界与冻结工件见
> [thought3_phase_e3_v2_report.md](thought3_phase_e3_v2_report.md)。

## 1. 目的与边界

v1 因官方 scheduler 的合法零权重端点触发非门控 ratio 遥测错误，未生成 Gate
结果。v2 重新执行完整 E.3，仍只回答：

> Gate E.2 的 8-sample 稳定性失败是否主要被单个 action-flow draw 混淆？

v2 不训练、不读取 development/OOD/success/rollout，也不能回答 future 是否改善
任务成功率。Gate E.2 的失败结论保持不变。

## 2. 相对 v1 的唯一实现修复

每个 objective 新增官方 scheduler `action_weight`。对于
`initial_action_loss=0`：

- 该 objective 仍完整保留在 per-sample 五 draw 的 mean action loss；
- 仍进入 mean reduction、6/8 non-worsened 和 catastrophic sample 门槛；
- 仍进入 hidden-scale、finite、memory 和 provenance 检查；
- 只从非门控 `final_loss / initial_loss` objective-level 最大值中排除，因为
  除以零未定义；
- 记录：
  - `objective_loss_ratio_count`；
  - `zero_initial_loss_objective_count`；
  - `zero_weight_objective_count`；
  - `zero_initial_loss_with_positive_weight_count`；
  - `positive_final_from_zero_initial_loss_count`；
  - `max_final_loss_from_zero_initial_loss`。

若 `action_weight=0` 而官方加权 `action_loss!=0`，或 initial/final 的 timestep/
weight 不同，立即 fail-closed。

这不是 eligibility 条件变化。v1 预注册的五个门槛保持原样：

1. sample-equal mean loss 至少下降 10%；
2. 至少 6/8 sample 不变差；
3. 0/8 sample 超过初始 mean loss 的 2 倍；
4. median sample mean `delta/action-hidden ≤ 0.50`；
5. max sample mean `delta/action-hidden ≤ 1.00`。

仍选择 `1e-4 → 3e-4 → 1e-3` 中第一个 A0/A1 共同 eligible 的 LR。

## 3. 冻结输入与预算

完全继承 v1：

- Gate E.2 四个 root artifact SHA；
- 六个 step-200 Adapter-only checkpoint 及 semantic SHA；
- 同一八条 train-only sample；
- `flow_step=1,2,3,4,5`；
- 同一 action noise/timestep seed namespace；
- 同一官方 action Flow Matching / velocity MSE；
- 同一 BF16、scheduler、shift、action normalization；
- 320 个 action-loss forward；
- 0 optimizer、0 backward、0 checkpoint 写入；
- 0 future RGB、0 development/OOD/success/rollout。

v1 原始失败目录必须保持不变。v2 不读取 v1 的部分 probe 作为结果输入，只将
v1 failure report 作为修复依据。

## 4. 冻结运行身份

| 字段 | v2 值 |
| --- | --- |
| config | `configs/thought3/phase_e3_multiflow_diagnostic_v2.yaml` |
| config fingerprint | `eeab3e38c1fd7ce15afc0852c1cac1007455a5551758c37d068ad6ea470b392e` |
| schema | `thought3.phase_e3.multiflow.v2` |
| output | `outputs/thought3/phase_e3_multiflow_v2/` |
| flow steps | `[1, 2, 3, 4, 5]` |
| checkpoints | 6 |
| forward objectives | 320 |

任何对 output/name、样本、flow steps、LR grid、训练 checkpoint 或门槛的 override
均拒绝。旧 v1 config 在写任何状态文件前被拒绝，以防覆盖失败证据。

## 5. 输出与判读

成功完成计算时，无论 gate pass 或 fail，都先写：

```text
outputs/thought3/phase_e3_multiflow_v2/
├── data_preparation.json
├── pre_validation_result.json
├── gate_e3_result.json
├── run_status.json
└── logs/phase_e3.log
```

- `gate_e3_passed=true`：只冻结候选 LR，随后以新 Run ID 重跑完整 28/4 Gate E；
- `gate_e3_passed=false`：不放宽门槛，不扩 A2/A4，不启动 Phase F；
- 进程/实现错误且无 `gate_e3_result.json`：分类为无效工程运行，不作模型结论。

## 6. 显式运行授权

测试和 dry-run 不加载大模型。真实运行预计约 12–18 分钟；按 v1 实测，model load
约 6.4 分钟、load peak `23,679.513 MiB`，43 GiB hard limit 不变。

必须由用户重新显式确认：

```bash
CONFIRM_THOUGHT3_PHASE_E3_V2=YES \
THOUGHT3_GPU_ID=1 \
bash scripts/run_thought3_phase_e3_multiflow_v2.sh
```

代码、测试或文档完成后不得自动运行。
