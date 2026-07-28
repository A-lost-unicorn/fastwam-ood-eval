# Thought3 Gate E.1：单样本固定目标 overfit 诊断协议

状态：`PREREGISTERED / NOT YET RUN`

本门禁用于解释 Gate E 的负面训练结果。它不改变 Phase D cache，不读取
development/OOD/success outcome，也不评价 K=1 是否优于 K=0。唯一问题是：
在冻结 Fast-WAM、只训练 Adapter 时，当前注入图和优化器能否拟合一条完全固定的
真实 LIBERO flow-matching 目标。

## 1. 冻结范围

- 数据：Phase D 的 `libero_goal` task 0、32 个 base sample、28/4 split；
- 训练样本：按既有 seed `3407` 排序后的第一条 `train` sample；
- 对照：A0 使用 zero/null future latent，A1 使用同一 sample 的 Phase D K=1
  cached latent；
- 两个变体使用相同 Adapter 初始化、action noise、action timestep、LR、
  weight decay、step 数和单卡环境；
- Fast-WAM 全参数冻结，optimizer 只能包含 1,371,137 个 Adapter 参数；
- 每个变体 200 optimizer step，LR `1e-3`，weight decay `1e-2`，batch
  和 accumulation 均为 1，每 50 step 保存 checkpoint；
- 禁止在线 future 生成、future RGB、ground-truth future、rollout、
  development/OOD/success outcome。

配置：
`configs/thought3/phase_e1_overfit_diagnostic.yaml`

执行入口：

```bash
CONFIRM_THOUGHT3_PHASE_E1=YES \
THOUGHT3_GPU_ID=1 \
bash scripts/run_thought3_phase_e1_overfit.sh
```

断点恢复：

```bash
CONFIRM_THOUGHT3_PHASE_E1=YES \
THOUGHT3_GPU_ID=1 \
bash scripts/run_thought3_phase_e1_overfit.sh --resume
```

## 2. 运行前预注册硬门槛

A0、A1 必须分别满足：

1. 指标严格覆盖 step 1–200，始终是同一 base sample 和固定 flow step；
2. step 1 只有 scalar gate 获得非零梯度；
3. step 2 起 future projector、attention 和整体 non-gate 路径获得 finite
   nonzero gradient，首次 non-gate step 必须为 2；
4. 最终实际 BF16 action hidden delta 的 norm 和 nonzero fraction 均大于
   0，避免只看到 fp32 residual 却没有真正写入 Fast-WAM hidden；
5. 最终固定目标 loss 相对初始化至少下降 **50%**；
6. 全程无 NaN/Inf，峰值显存小于 43 GiB；
7. Adapter checkpoint 可完整 round-trip；
8. 不读取 ground-truth future RGB，optimizer scope 为 Adapter-only。

成对门槛：

1. A0/A1 使用同一个 base sample；
2. action noise/timestep seed 完全相同；
3. Adapter 初始 semantic SHA 完全相同；
4. zero gate 下初始 action loss 精确相同；
5. trainable parameter count 相同；
6. Fast-WAM frozen parameter SHA 在训练前后完全相同。

任一项失败，Gate E.1 总状态为 failed，且不解锁 A2/A4。失败前仍须先保存
`pre_validation_result.json` 和 frozen-before/after，以免重现 Gate E
“loss 门槛先报错、冻结哈希未闭环”的证据缺口。

## 3. 必须记录的诊断量

每一步保存：

- 固定目标 action loss；
- gate 更新前后值、gate gradient 及符号；
- action hidden norm、attention residual norm；
- 实际 BF16 gated hidden delta norm 与 nonzero fraction；
- gate、future projector、future token path、attention、non-gate、all 的
  gradient L2、parameter L2、gradient-to-parameter L2 ratio；
- step time 和峰值显存。

输出目录：

```text
outputs/thought3/phase_e1_overfit_v1/
├── data_preparation.json
├── pre_validation_result.json
├── gate_e1_result.json
├── run_status.json
├── logs/phase_e1.log
└── variants/
    ├── a0/
    └── a1/
```

## 4. 解释边界

- 若 A0/A1 均通过，只能说明注入图和 Adapter-only optimizer 能拟合一个
  固定真实目标；不能说明泛化、任务成功率或 future 的因果价值。
- 若单样本通过而 28-sample Gate E 不通过，下一步进入预注册的 8-sample
  train-only LR/batch 诊断，不读取 development/OOD outcome 来挑配方。
- 若任一单样本轨迹仍不能下降，优先检查 gate optimizer group、BF16 注入
  量化、residual 尺度和 loss 连接，不扩 K、不跑 rollout。
- A0 在本诊断中是 matched null-latent Adapter 工程对照，不等同于最终论文中
  “完全无 Adapter 的官方 Fast-WAM”科学基线。
