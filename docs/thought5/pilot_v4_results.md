# Thought5 Pilot v4 结果登记

运行日期：2026-08-05

科学角色：**单 task 方向性 Pilot，非正式多任务结果**

## 1. 总结判定

Pilot v4 有效完成，但预注册方向门禁为负：

```text
training direction          false
representation direction    false
future geometry direction   false
future utility direction    false
rollout direction           false
shuffled control excluded   false
                              ↓
formal_unlocked             false
```

这不是运行故障。所有 track、collector、utility 和 rollout worker 均为
`status=complete`；负 Gate 来自结果本身。`formal_protocol_frozen.json` 没有生成，
因此 formal v2 必须保持锁定。

## 2. 完整性与身份

- Project commit：`c7ab21537e0a46ed1455812b2a6741628cec6db6`
- Config fingerprint：
  `ff143479357dce6d1cfe93f99948b40d4af43d25f34939dd2762c646be8e186d`
- Cohort semantic SHA：
  `16ee9c53ff9542da6ac9f46b26e3d17fd81c9e7ab14f6638b23b9fad5d5fdac0`
- Execution schedule SHA：
  `f9170eee2faab1773efbf86340a1ff8a8e48571fe8a8a34444bc8b24a237e6b4`
- Cohort：单个 `libero_goal` task，互斥 8 train / 4 development / 4 pilot-test
  episode；B1/G3/G4 matched 三卡执行。
- 总 wall-clock：约 2 小时 29 分钟；单 track 约 19–20 分钟；峰值显存
  24,508.1 MiB。
- 三个 trainable track 参数预算一致，均为 1,335,320；Action DiT 冻结。

权威文件 SHA-256：

| 工件 | SHA-256 |
| --- | --- |
| `run_status.json` | `165f2657317b49b71e74c720096292d82236d174cb5a2a0226e4fd5afe46d3b8` |
| `pilot_direction.json` | `1bb2944d196253b7002daaa87f340bbad61c86e4e7e2a3e05f0c0fbe46a98c3d` |
| `training_results.json` | `2c46a539bb4c2fc27c138f593694d29fa26d755bd069dece16ad3f0ed6313b36` |
| `representation_results.json` | `5604d8ff8f6de52ae1744feb856f0678dcddb19e3f5e953fdfef6be52b8a5efa` |
| `future_geometry_results.json` | `582d836ff2b28eb2febef5627007975c460be7a83c73bbc2f5ebe7b1bbdaa4a2` |
| `future_utility_results.json` | `612c8f8cecdf9d3ade8e033aa6d996de4dd083d676c8a03bcd382a543d998bba` |
| `rollout_results.json` | `fac48a4b92a59820c77efcd6ca1e64bdc24ca0640fa26202b0782a2da943aa3e` |

## 3. Gate 分解

### 3.1 Training/development

三个版本都按冻结规则选择 step 100：

| Variant | Development selection objective | 原始 Fast-WAM loss |
| --- | ---: | ---: |
| B1 | 0.035865 | 0.035865 |
| G3 | 0.048300 | 0.035921 |
| G4 | 0.048728 | 0.035904 |

G3 既没有优于 B1，也没有优于 G4，所以 `training_direction_observed=false`。
G3/G4 较高的 selection objective 包含冻结协议中的辅助项；即使只看原始动作 loss，
两者也未优于 B1。

### 3.2 Current representation（H1 方向）

| 指标 | B1 | G3 | G4 |
| --- | ---: | ---: | ---: |
| Camera representation gap | 0.002246 | 0.001776 | 0.001666 |
| Clean error | 0.056578 | 0.056892 | 0.057013 |

G3 相对 B1 的 Camera gap 缩小 **20.94%**，低于预注册的 25% 门槛；G3−B1
camera difference 为 −0.000470，grouped bootstrap 95% interval
[−0.001146, 0.000175]，跨过 0。Clean non-degradation 与 lighting specificity
通过，但 shuffled-geometry G4 的 gap 反而更小，因此 specificity control 未排除。
结论：存在弱方向信号，但不足以登记 H1 方向通过。

### 3.3 K=1 future geometry

主 Camera geometry RMSE 为 B1 0.341277、G3 0.341320、G4 0.341331；G3 相对
B1 轻微变差 +0.000043，interval [−0.001099, 0.000916]。四个预注册 Camera
指标中，latent L1、depth relation 与 EEF-object position 有小幅改善，但主
camera-geometry RMSE 没有改善。Clean→Camera gap 从 0.011111 缩至 0.010125
不能替代主误差改善，因为 G3 的 Clean error 同时上升。方向 Gate 为 false。

### 3.4 Future utility（H2 方向）

utility 定义为 `loss(A0/null) - loss(A1/correct)`；正数才表示 correct future
优于 null。

| Backbone | A0 loss | A1 loss | AS loss | Correct utility | Specificity `AS-A1` |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 | 0.130078 | 0.145728 | 0.148330 | −0.015649 | +0.002602 |
| G3 | 0.130148 | 0.135378 | 0.143135 | −0.005231 | +0.007757 |
| G4 | 0.130187 | 0.141489 | 0.145699 | −0.011302 | +0.004209 |

G3 相对 B1 的 utility 改善为 +0.010419，interval
[0.007896, 0.012978]；这是本 Pilot 最清晰的正向信号。但 G3 的绝对 utility
仍为 −0.005231，interval [−0.008987, −0.001763]：correct future 仍显著差于
null。Correct 优于 shuffle 的 specificity 为 +0.007757，interval
[0.004558, 0.011340]。因此可写“G3 缓解了 future conditioning 的伤害并保留
内容 specificity”，不能写“future 已变得有用”；H2 方向仍为 false。

### 3.5 Paired rollout（H3 方向）

| Condition | B1 | G3 | G4 |
| --- | ---: | ---: | ---: |
| Clean | 1/4 | 1/4 | 1/4 |
| Camera | 1/4 | 1/4 | 0/4 |
| Lighting | 4/4 | 4/4 | 4/4 |
| Robot-init | 4/4 | 4/4 | 4/4 |

G3−B1 在四个条件上均为 0；Clean non-inferiority 通过，但 Camera improvement
与 camera specificity 均未出现。由于 G3 没有 rollout gain，shuffled control
也无法被该 endpoint 排除。H3 方向为 false。

## 4. 可写与不可写结论

可以写：

- Fast-WAM-GeoEq 在单 task Pilot 中产生了弱且非特异的 Camera representation
  gap 缩小，但没有达到预注册门槛。
- G3 显著缩小了 K=1 correct-future 相对 null 的负 utility，却没有把它变成正
  utility；future sensitivity 与 future usefulness 仍然不同。
- 表征小幅变化没有转化为 paired rollout success 改善。
- 该结果反对直接把当前 G3 recipe 扩展到 28–45 小时 formal，而不否定 Thought4
  已确认的 Camera Equivariance Gap 诊断。

不可写：

- 不能把单 task Pilot 当作 H1/H2/H3 的正式多任务否证。
- 不能声称 Geo-REPA 普遍无效、未来想象普遍无用，或 Camera gap 不是因果机制。
- 不能运行 formal v2、降低 25% 门槛、选择性忽略 G4，或依据本 Pilot 调参后继续
  使用同一批 Pilot endpoint 作确认性证据。

## 5. 冻结的下一步

当前 recipe 到此停止。现有工件的 condition/flow/action、RayPose gate/LoRA 与
训练轨迹[只读失败分解](pilot_v4_readonly_failure_analysis.md)已经完成；它将下一
假设收窄到 Clean/低 sigma 伤害与 RayPose/共享正则化识别，但不改变本 Pilot 判定。
若要尝试 condition-aware future fusion 或其他新 recipe，必须建立新预注册、新
namespace 和未使用的独立 Pilot cohort，再决定是否创建新的 formal 候选。
