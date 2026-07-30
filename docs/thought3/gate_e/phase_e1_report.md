# Thought3 Gate E.1：单样本固定目标 overfit 诊断报告

状态：**Gate E.1 通过；Gate E 总门禁仍未通过**
验收日期：2026-07-28
证据等级：`ENGINEERING DIAGNOSTIC / NOT MODEL EFFECT`
Run ID：`P3-PHASE-E1-v1`

## 1. 结论

Gate E.1 已在一张 RTX 4090 上完成真实 Fast-WAM 的 A0/A1 单样本固定目标
诊断。预注册的全部硬检查通过：

- A0/A1 使用同一条标准 LIBERO train sample、相同 Adapter 初始化、相同
  action noise/timestep 和相同 200-step 训练预算；
- zero gate 下初始 action loss 精确相同，均为 `0.0358901434`；
- step 1 只有 scalar gate 获得非零梯度；
- step 2 起 future projector、attention 和 non-gate 路径均获得 finite
  nonzero gradient；
- A0 最终固定 loss 为 `0.0025361809`，相对下降 `92.93%`；
- A1 最终固定 loss 为 `0.0001489980`，相对下降 `99.58%`；
- 两者都超过运行前冻结的 `≥50%` loss reduction 门槛；
- 实际写回 BF16 action hidden 的 delta 非零，不是仅存在于 fp32 residual；
- Adapter-only checkpoint round-trip 通过；
- Fast-WAM 全参数 SHA 在训练前后精确相同；
- 无 NaN/Inf、无 OOM、无 future RGB、无 ground-truth future、无
  development/OOD/success outcome、无 rollout。

因此可以排除“当前注入图或 optimizer 完全断开，连一条固定真实目标都不能拟合”
这一解释。Gate E v2/v3 的固定 probe 不下降，更可能来自多样本/多噪声优化、尺度、
训练配方或泛化问题，而不是单纯的梯度断链。

但 Gate E.1 **不能**证明：

- A1 泛化优于 A0；
- future latent 改善任务成功率或 OOD；
- K=1 优于 K=0；
- 当前 `lr=1e-3` 配方适合多样本训练；
- Gate E 已通过，或可以开始 A2/A4/rollout。

## 2. 冻结协议与运行身份

Gate E.1 的代码、配置、50% loss 门槛和解释边界在查看本次结果前冻结于：

```text
30ffc9343e0f5936e97fbc4e7f629805b438329e
```

| 项目 | 值 |
| --- | --- |
| 分支 | `feature/thought3-partial-future-adapter` |
| 配置 | `configs/thought3/phase_e1_overfit_diagnostic.yaml` |
| config fingerprint | `206b847ba00354d9278c390d8bfc6ffda44ba13f1e169cc3b71998f66ef29164` |
| Fast-WAM commit | `45d8e1458921d83f8ad6cf9ce993d371208dabd0` |
| checkpoint SHA-256 | `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579` |
| cache fingerprint | `63a70e1af38f68bc894fc11d03c84f212e6c6328a5051256c9d045741156d9c5` |
| split fingerprint | `ea5402955023ccd48d790d821a73f98549b31d1ace8af035a90ceae2ad3951eb` |
| physical/logical GPU | GPU 1 / `cuda:0`，visible device count = 1 |
| trainable parameters | 1,371,137，Adapter-only |
| optimizer | AdamW，LR `1e-3`，weight decay `1e-2` |
| budget | A0 200 + A1 200 optimizer step |
| checkpoint | 每 50 step |
| seed | 3407 |

数据仍是 Phase D 的 `libero_goal` task 0。32 个 cache base sample 的 split 为
28 train / 4 development；本诊断只选择固定排序后的第一条 train sample：

```text
base_sample_id =
4a0a595342e32200b9f7dc1266b0a110ef9c062370b524c6c5808102eade8bfb
task = open the middle drawer of the cabinet
```

development split 只作为已冻结 provenance 存在，未计算 development loss。

## 3. 主结果

| Variant | Future input | Initial loss | Final fixed loss | Reduction | Final gate | Final BF16 delta norm | Nonzero fraction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | zero/null latent | 0.03589014 | 0.00253618 | 92.93% | 0.0166272 | 20.1428 | 99.954% |
| A1 | Phase D K=1 latent | 0.03589014 | 0.00014900 | 99.58% | 0.0146621 | 7.3923 | 99.615% |

最终 loss 是 step 200 optimizer update 后对同一固定目标重新 forward 的值，所以
可与 step 200 更新前日志中的 `0.00225582 / 0.000130185` 略有不同。

A1 在这一个独立过拟合目标上取得更低 loss，只能作为后续稳定性诊断的线索。
样本数为 1、两个变体分别训练、A0 是 null-latent Adapter 而不是官方无 Adapter
基线，因此不能将该差异写成 future 有效或因果证据。

## 4. Zero-gate 与梯度链

| Variant | Step | Gate before→after | Gate grad | Projector grad/param | Attention grad/param | Non-gate nonzero |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| A0 | 1 | 0→0.0010000 | −0.0209961 | 0 | 0 | 0 |
| A0 | 2 | 0.0010000→0.00199999 | −0.0209961 | 4.83e−5 | 2.17e−5 | 1,321,979 |
| A1 | 1 | 0→0.000999932 | −0.000146866 | 0 | 0 | 0 |
| A1 | 2 | 0.000999932→0.00199968 | −0.000145912 | 1.25e−5 | 6.12e−6 | 1,371,077 |

两组首次 non-gate nonzero gradient 都严格出现在 step 2。A0 的 future latent
全零，因此 projector weight 没有输入梯度，主要通过 256 个 bias 元素传播；
A1 的真实 K1 latent 使 projector weight/bias 都获得梯度。这是预期的 matched
null-latent 工程对照行为。

## 5. 注入尺度：新发现的下一门卡点

固定 action hidden norm 始终约为 `10.5326`。step 200 更新前：

| Variant | Attention residual norm | Actual BF16 delta norm | Delta / action hidden |
| --- | ---: | ---: | ---: |
| A0 | 1,214.69 | 20.1612 | 1.914 |
| A1 | 505.93 | 7.4103 | 0.704 |

scalar gate 虽然只有约 `0.015`，但 residual 已放大到数百至上千。A0 的实际
修正甚至达到原 action hidden norm 的 1.91 倍。该结果说明：

1. BF16 并未把注入完全量化为零；
2. 单样本 overfit 是通过较大的 hidden correction 实现的；
3. “能 overfit”不能直接升级为“训练稳定”或“会泛化”；
4. 下一轮 8-sample train-only 诊断必须同时监控 loss 与
   `gated_delta_norm / action_hidden_norm`，不能只按最低 loss 选 LR。

是否需要 residual normalization、独立 gate LR、gate/delta regularization 或更低
全局 LR，必须在新协议中一次只改一个变量后再判断。本 run 不回看结果后修改配方。

## 6. Frozen、checkpoint、显存与时间

Fast-WAM 全参数哈希：

```text
before = ac0dd59de78495dc8ed5c5601e24951b78f3fd6ad40843d76b83106eccceb4f8
after  = ac0dd59de78495dc8ed5c5601e24951b78f3fd6ad40843d76b83106eccceb4f8
```

| 指标 | A0 | A1 |
| --- | ---: | ---: |
| mean optimizer step | 811.71 ms | 795.70 ms |
| peak optimizer-step memory | 13,273.17 MiB | 13,273.17 MiB |
| final Adapter semantic SHA | `6e6790e3…b6c030c` | `5b8f2edf…641c78` |
| checkpoint optimizer entries | 18 | 18 |
| checkpoint state equal after round-trip | true | true |

其他时间：

- 模型加载：531.46 s，峰值 23,679.51 MiB；
- 32-sample current-only 数据准备：64.83 s；
- A0 训练调用：168.84 s；
- A1 训练调用：165.82 s；
- 整个 Gate：1,346.35 s（22 分 26.35 秒，包含两次全模型 SHA）。

该 step time 是 fixed-target forward/backward/update，不是在线 action denoising 或
机器人 rollout latency。

## 7. 数据泄漏与阶段隔离

本次数据访问审计：

```text
base samples                    32
train / development             28 / 4
current camera frames decoded   64
action chunks / rows            32 / 1024
current state rows              32
future RGB frames decoded       0
actual future read              false
uses ground-truth future        false
development outcomes read       false
OOD/success outcomes read       false
rollout started                 false
```

只读取标准 LIBERO training demonstration 和冻结 Phase D cache。没有读取或写入
Thought1/Thought2 实验结果，也没有修改 `third_party/FastWAM`。

## 8. 冻结机器工件

| 工件 | SHA-256 |
| --- | --- |
| `run_status.json` | `260bf19aa3de329883f5ccf90016a2742b6112b8328e9f695d84b3d480e27c11` |
| `gate_e1_result.json` | `862e37bb7cb80f6fa66118de16f543c2c72a3dbe70fa6a7f3a4d5705b2aac425` |
| `pre_validation_result.json` | `3fdf583306e80b83ebfc8d9ceccd1e52eae694a4fb00802aa31b97605a900983` |
| `data_preparation.json` | `cd05e4a8a7248edfa1c46060ce686cd2b2c0695c7b0bfabb6b00e9f0152b9cc5` |
| `logs/phase_e1.log` | `0766826a62058d3aa9068850387976782dae6b032dad69fa8e778b4300cf6d67` |
| A0 metrics | `a849b078bd2b02947ad737cbdc90eb79c8fa7a4d1c3dbe06261b7c039d57c8d1` |
| A0 manifest | `1ef8280ba52cf22e8d7f3fdae679911d619207246a1496f94bc7aa59e127afe5` |
| A0 final `adapter.safetensors` | `b7bc48a131af29601af744ae08bb1f5f5734eb858612c6753f3c0953254dcf96` |
| A0 final checkpoint manifest | `f52f48ce7dabf19b24f52b695cc78cbcc9272f845ec09e84d01df2c5602586b8` |
| A0 final `optimizer.pt` | `53e0a3eb6a05f6414018340f002c0d5479dcedcd54b0ddcb805612fc174b215b` |
| A1 metrics | `4d6054bdae139c5527de9922313125c0e0b74235ba0899665f90a2a79a863271` |
| A1 manifest | `25d5a3a1879b056625f190477dabf09d14a0925a3aa0bf40b05d51753e856170` |
| A1 final `adapter.safetensors` | `a16ac1ed8e835abd9402f6c310155c186715b3f7d9b29540f388dd19943235ad` |
| A1 final checkpoint manifest | `e21007e1706885f7befc14384fe9070ddba6f41a9993aaeed516080821c580e9` |
| A1 final `optimizer.pt` | `b8b77a7794e116dbf01860f357de9f96d7146e61ee2ffff68a5a6102c7f50d0e` |

权威目录为：

```text
outputs/thought3/phase_e1_overfit_v1/
```

目录大小约 127 MiB。不要删除 checkpoint 后复用同一 Run ID；任何协议或配方变化
必须使用新 output directory。

## 9. 对 Gate E 的影响与下一步

Gate E.1 关闭了两个工程问题：

- 单样本固定目标可以显著下降；
- frozen-before/after 已完整闭环。

它不追溯改判 Gate E v2/v3。多样本固定 probe 仍未改善，所以 A2/A4 和 Phase F
继续锁定。

下一步应先冻结 Gate E.2：

1. 只取 8 条 train sample，不读取 development/OOD/success outcome；
2. 比较预先列出的少量 LR，例如 `1e-4 / 3e-4 / 1e-3`；
3. 每个 sample 使用固定且可复现的 noise/timestep；
4. 同时报 sample-equal fixed loss、gate、分模块 gradient 和
   `delta/action-hidden` 比率；
5. 禁止只按最低 train loss 选择会产生过大 hidden correction 的配方；
6. 冻结一个稳定配方后，重新运行完整 28/4 Gate E；
7. 只有新的 Gate E 完整通过，才解锁 A2/A4。

这一路径仍是训练工程诊断，不构成论文里的 future 效果实验。
