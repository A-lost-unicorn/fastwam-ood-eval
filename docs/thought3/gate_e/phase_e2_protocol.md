# Thought3 Gate E.2：八样本 Train-only LR/尺度诊断协议

状态：`PREREGISTERED DRAFT / NOT RUN`
证据等级：`ENGINEERING DIAGNOSTIC / NOT MODEL EFFECT`

Gate E.1 已证明当前注入图能拟合一个固定真实目标，但 A0/A1 的实际 BF16
hidden correction 分别达到原 action hidden norm 的 1.91×/0.70×。Gate E.2
用于回答：在不读取 development/OOD/success outcome 的条件下，是否存在一个
共同 LR，使 A0 和 A1 都能在 8 条固定 train sample 上降低 loss，同时控制实际
hidden correction 尺度。

本门禁不会训练 A2/A4，不运行 simulator，不做模型效果或 future 因果结论。

## 1. 冻结数据与轨迹

- 来源：Phase D 冻结的标准 `libero_goal` task 0 cache；
- 完整 cache split：28 train / 4 development；
- Gate E.2 sample：按既有 seed `3407` 排序后的前 8 条 train sample；
- development 只保留为 split provenance，不计算 loss；
- source loader 只读取这 8 条 train sample：16 张当前相机帧、8 行 state、
  8 个 action chunk / 256 行 action target；不加载 4 条 development action；
- 每个 sample 的 action noise/timestep 由 sample ID、seed 3407 和固定
  `flow_step=0` 生成；
- 训练以固定 sample order 做 8-sample round-robin；
- A0 使用 zero/null latent；
- A1 使用对应 sample 的 Phase D K=1 model-generated latent；
- 两者结构、初始化、sample order、训练预算、checkpoint 频率完全相同。

禁止：

- future RGB 或真实后续 observation；
- ground-truth future latent；
- development loss、OOD outcome、success、rollout；
- Thought1/Thought2 正式轨迹；
- 在线 future 生成；
- A2/A4/A-shuffle；
- 修改冻结 Fast-WAM 或 `third_party/FastWAM`。

## 2. 冻结 LR 网格与预算

按从小到大的顺序执行：

```text
1e-4
3e-4
1e-3
```

每个 LR 都执行 A0 和 A1，共 6 条轨迹：

```text
3 learning rates × 2 variants × 200 optimizer steps
= 1,200 optimizer steps
```

共同配置：

- AdamW；
- weight decay `1e-2`；
- microbatch 1；
- gradient accumulation 1；
- Adapter-only 1,371,137 trainable parameters；
- checkpoint interval 50；
- deterministic CUDA math SDP；
- 单张物理 GPU，逻辑 `cuda:0`；
- 43 GiB hard memory limit。

配置：

```text
configs/thought3/phase_e2_eight_sample_diagnostic.yaml
```

## 3. 执行硬门槛

六条轨迹都必须满足：

1. metrics 严格覆盖 step 1–200；
2. sample 严格按同一 8-sample round-robin；
3. probe schedule 严格为 step `0/50/100/150/200`；
4. step 1 只有 scalar gate 获得非零 gradient；
5. step 2 起 projector、attention、non-gate gradient finite 且非零；
6. loss、gradient、gate、hidden delta 全程 finite；
7. optimizer 只包含 Adapter；
8. Fast-WAM 不获得 gradient；
9. Adapter/optimizer checkpoint round-trip 通过；
10. 峰值显存小于 43 GiB；
11. 不读取 development/OOD/success/ground-truth future；
12. A0/A1 在每个 LR 下使用相同初始化、sample、初始 action loss 和预算。

若任一轨迹出现 NaN、OOM、backbone gradient 或 provenance 错配，立即停止后续
optimizer step；但在抛错前仍计算 frozen-after SHA，并保存已经完成的 partial
track 和 traceback。

## 4. 配方合格门槛

一个 LR 只有在 A0 和 A1 **分别**都满足以下条件时才 eligible：

1. 8-sample sample-equal mean fixed loss 相对初始化至少下降 10%；
2. 至少 6/8 sample 的最终 fixed loss 不高于自身初始化；
3. 0/8 sample 的最终 loss 超过自身初始化的 2 倍；
4. 8-sample median `gated_delta_norm/action_hidden_norm ≤ 0.50`；
5. 8-sample max `gated_delta_norm/action_hidden_norm ≤ 1.00`。

这些是运行前冻结的工程稳定性阈值，不是论文中的统计显著性或成功率门槛。

总 Gate E.2 通过还要求：

- 六条轨迹的执行硬门槛全部通过；
- 所有 LR 的 A0/A1 配对检查全部通过；
- Fast-WAM frozen parameter SHA before/after 精确相同；
- 至少存在一个 A0/A1 共同 eligible LR。

## 5. 固定选择规则

如果多个 LR eligible，固定选择 **最小的 eligible LR**：

```text
1e-4 → 3e-4 → 1e-3
```

不按最低 A1 loss、A1−A0 差值、development loss 或 OOD outcome 选择。该规则
优先减小尺度风险，并避免看到结果后挑选最漂亮的 future 差异。

若没有共同 eligible LR：

- Gate E.2 失败；
- 不自动修改 gate、normalization、regularization 或 Adapter 结构；
- 不扩 A2/A4；
- 根据完整六轨迹 telemetry 再冻结一个单变量诊断协议。

## 6. 必须保存的遥测

每个 optimizer step：

- base sample ID 与 sample cursor；
- 固定 action loss；
- gate 更新前后值、gradient 与符号；
- action hidden、attention residual、future token norm；
- 实际 BF16 hidden delta norm/nonzero fraction；
- `gated_delta/action_hidden`；
- gate/projector/future-token/attention/non-gate/all 的 gradient L2、
  parameter L2、gradient-to-parameter ratio；
- step time、peak memory、NaN/Inf。

每 50 step 的 8-sample probe：

- 每 sample initial/final action loss 与 ratio；
- sample-equal mean action loss；
- non-worsened/catastrophic sample count；
- median/max `delta/action-hidden`；
- gate；
- checkpoint 与 Adapter semantic SHA。

根目录还保存：

- Phase D frozen SHA；
- current/action 数据访问审计；
- frozen-before/after SHA；
- 六轨迹结果；
- 每个 LR 的 eligibility；
- 固定选择结果；
- pre-validation partial evidence；
- final status 和 traceback。

## 7. 输出、恢复和预计成本

输出：

```text
outputs/thought3/phase_e2_eight_sample_v1/
├── data_preparation.json
├── pre_validation_result.json
├── gate_e2_result.json
├── run_status.json
├── logs/phase_e2.log
└── tracks/
    ├── lr_1e_04/{a0,a1}/
    ├── lr_3e_04/{a0,a1}/
    └── lr_1e_03/{a0,a1}/
```

每条 track 保存 step 50/100/150/200 的 Adapter-only checkpoint。根据 Gate E.1
实测估计：

- 模型加载约 9 分钟；
- 两次全模型 SHA 约 6 分钟；
- 六条训练与 fixed probe 约 20–25 分钟；
- 总 wall time 约 35–45 分钟；
- optimizer-step 峰值预计约 13 GiB；
- 输出预计约 380–450 MiB。

首次运行命令必须由用户显式确认：

```bash
CONFIRM_THOUGHT3_PHASE_E2=YES \
THOUGHT3_GPU_ID=1 \
bash scripts/run_thought3_phase_e2_eight_sample.sh
```

只有真实中断、且已有 checksum-valid checkpoint 时才使用：

```bash
CONFIRM_THOUGHT3_PHASE_E2=YES \
THOUGHT3_GPU_ID=1 \
bash scripts/run_thought3_phase_e2_eight_sample.sh --resume
```

禁止删除失败 track 后复用同一 Run ID。

## 8. 解释边界与下一门禁

Gate E.2 通过只说明：存在一个共同的、train-only 工程候选 LR，能在 8 条固定
目标上同时满足 loss 与尺度约束。

它不能说明：

- A1 泛化优于 A0；
- future 改善 ID/OOD success；
- K=1 优于 K=0；
- 当前 Adapter 已真正依赖 future；
- 可以直接进入正式实验。

Gate E.2 通过后：

1. 用冻结的共同 LR 创建新 Run ID；
2. 重新运行完整 28 train / 4 development Gate E；
3. 保留固定 train probe、development-only checkpoint selection、resume、
   frozen hash、loss 和尺度门槛；
4. 只有完整 Gate E 通过，才实现并训练 A2/A4；
5. 随后进入包含 B0/A0/A1/A2/A4/A-shuffle 的 Phase F 在线技术 pilot。
