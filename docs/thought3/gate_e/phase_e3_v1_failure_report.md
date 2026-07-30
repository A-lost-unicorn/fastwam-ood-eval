# Thought3 Gate E.3 v1 无效运行报告

状态：`INVALID ENGINEERING RUN / NO GATE CONCLUSION`
运行日期：2026-07-28
证据等级：`ENGINEERING DIAGNOSTIC`

## 1. 结论

Gate E.3 v1 **没有跑完，也没有通过或失败的科学门控结论**。运行成功完成模型、
数据和初始 A0/A1 probe，并完成冻结 Fast-WAM 的事后哈希；随后在第一个
step-200 checkpoint 的非门控 objective-level ratio 汇总中触发实现错误。

该错误不表示 Adapter checkpoint 训练失败，也不表示 held-out loss 未达到门槛。
`gate_e3_result.json` 不存在，故不得把本次运行写成 Gate E.3 passed/failed，更不得
据此选择 learning rate。

## 2. 冻结运行身份

| 字段 | 值 |
| --- | --- |
| 源码 commit | `330fe15e5bfcc6a42710fd6564125ec5fc49d66e` |
| v1 config fingerprint | `f2313eec175f26d7d0bc61a89c77127344f76012e234d9792fc3850131075652` |
| 输出目录 | `outputs/thought3/phase_e3_multiflow_v1/` |
| 开始时间 | `2026-07-28T07:59:48.960348+00:00` |
| 失败状态写入时间 | `2026-07-28T08:09:10.334591+00:00` |
| 物理 GPU / 逻辑设备 | GPU 1 / `cuda:0` |
| model load peak | `23,679.513 MiB` |
| Gate E.2 root SHA | `40f66bc50acd8e175ecb61ec150a04ef9ed5c55bf1fa9090802cc529104214bb` |

v1 原始工件被冻结且不得覆盖：

| 工件 | SHA-256 |
| --- | --- |
| `run_status.json` | `0ae9f21dfa9e1306c878e0188c91398464ca23d50104bf3e5dbb3455e2b99605` |
| `pre_validation_result.json` | `f5a1ec7800ad3f2940b51c3fac89215481e672262e9d7deb1d922853db9e3ff8` |
| `data_preparation.json` | `fddc57bf73f6d4dd996f5e959ed667a7db98e370c499a7e17a0e7e7fb8ee65d2` |
| `logs/phase_e3.log` | `7dccb25bc644aea1256705dafaa9e8d2cfa4c210875eaad256e70c7305742f7c` |

## 3. 已完成且可保留的工程证据

- 八条标准 LIBERO train-only 样本准备完成；
- `future_rgb_frames_decoded=0`，未把真实未来作为 Adapter 输入；
- A0/A1 初始 probe 各覆盖 `8 × 5 = 40` 个 held-out objective；
- A0/A1 初始 mean action loss 精确相同：
  `0.005565503754223755`；
- 冻结 Fast-WAM 参数 SHA 前后均为
  `ac0dd59de78495dc8ed5c5601e24951b78f3fd6ad40843d76b83106eccceb4f8`；
- 没有 optimizer、backward、checkpoint 写入或 rollout。

首个 final checkpoint 的 forward 已执行，但 outcome 在写入
`final_results` 前抛错，因此不能从 v1 恢复该结果，也不能将空的
`final_results.lr_1e_04` 解释为 checkpoint 表现。

## 4. 根因

初始 A0 和 A1 的 40 个 objective 中各有且只有一行：

```text
base_sample_id =
0643a55f92bbe4fd0da1abb9ab756422ca564f2f921b8f8b0b1f9d50514b5c69
flow_step       = 5
timestep        = 1000.0
action_loss     = 0.0
```

Fast-WAM 官方 continuous flow scheduler 在 `timestep=1000` 时：

```text
training_weight(1000.0) = 0.0
```

官方 action objective 是 token MSE 乘该 scheduler weight，所以加权 loss 为
零是合法端点，不是 NaN、数据损坏或完美预测。v1 的 sample-equal 主统计允许
non-negative loss，但后续一个只用于诊断的 `max_objective_loss_ratio` 又要求每个
initial objective loss 严格大于零。由于 `final / 0` 未定义，代码抛出：

```text
RealTrainingError:
Gate E.3 initial objective loss must be positive
```

因此根因分类为：

```text
non-gating telemetry implementation bug
```

而不是：

```text
model failure / checkpoint corruption / frozen-backbone drift
```

## 5. 修复纪律

v2 只修正未定义的 objective-level ratio 遥测：

1. 每个 objective 显式保存官方 `action_weight`；
2. scheduler weight 为零时，仍将该 objective 的零 loss 保留在五次
   draw 的 sample mean 和全部原门槛中；
3. 只从非门控的 `final/initial` objective ratio 最大值中排除
   `initial_loss=0` 的未定义项；
4. 显式记录排除数量、zero-weight 数量，以及零 initial loss 后 final 是否变正；
5. 样本、flow step、checkpoint、LR、loss、阈值和选择规则全部不变；
6. 使用新的 schema、config fingerprint、输出目录和用户确认变量重新运行全部
   320 个 forward，不复用 v1 的部分结果。

v2 协议见
[thought3_phase_e3_v2_protocol.md](phase_e3_v2_protocol.md)。
