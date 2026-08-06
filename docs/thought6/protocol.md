# Phase 6 冻结协议

## 假说

Thought5 的只读失败分解显示，future utility 在低 sigma 分桶明显为负，在高 sigma 分桶略为正。Phase 6 的可证伪假说是：无需训练主干或 Adapter，只在 `effective_sigma >= 0.5` 时注入 correct K=1 future，可以消除低噪声阶段的负迁移，同时保留高噪声方向信息。

阈值 `sigma_0=0.5` 写死在代码常量中，配置和 CLI 均不能覆盖。任何 Phase 6 development、utility 或 rollout 结果都不能修改该阈值。

## 冻结模型边界

- Backbone：原始 Fast-WAM release checkpoint，完全冻结。
- Adapter：Thought3 Phase 2 A1、K=1、固定 step 200 checkpoint，完全冻结。
- 禁止 Thought5 G3、Video LoRA、RayPoseEncoder、GeoProjector。
- 不训练任何参数，不构造 optimizer，不按结果选 checkpoint。

## 六种模式

| 模式 | 规则 | 角色 |
| --- | --- | --- |
| B0 | 不读取 future、不调用 Adapter | no-future baseline |
| F0 | 全部 20 个 Action Flow step 注入 correct future | full-stage control |
| Fsigma | 仅 `sigma >= 0.5` 注入 correct future | 唯一主方法 |
| Label-Oracle | 仅 Camera condition 全阶段注入 | 诊断上界，不可部署 |
| Label-Oracle+Fsigma | Camera 且高 sigma 才注入 | 诊断上界 |
| Shuffle+Fsigma | 与 Fsigma 同门控，future 内容打乱 | 内容特异性对照 |

官方 BF16 20-step scheduler 在当前实现下有 17 个 step 满足阈值，最后 3 个 step 退化为严格 identity/B0 注入边界。门控读取实际 scheduler sigma，不按 step 编号硬编码。

## 离线与在线 sigma

- 离线 utility：CPU FP32 uniform 经 `phi(u,5)` 得到 sigma，乘 1000 后转换 BF16 timestep，再按 BF16 路径除以 1000 得 effective sigma。
- 在线 rollout：读取 Fast-WAM 20-step scheduler 返回的实际 BF16 timestep，以 `t/1000` 得当前 effective sigma。

二者共用固定阈值，但采样过程不同；flow slot 也不等于在线 denoising step。

## Gate 与停止规则

Phase 6B 必须依次通过 Clean non-inferiority、Camera positive utility、correct-vs-shuffle specificity、Fsigma-vs-F0 timing benefit、null/B0 无人工退化五个 Gate，才解锁 Phase 6C。

Stage 1 仅比较 B0/F0/Fsigma。只有 Camera 方向为正但 CI 尚含 0，同时 Clean 非劣且 Fsigma 方向优于 F0，才允许用全新的 state 10–19 扩展 Stage 2。Stage 2 不自动启动。

任一主 Gate 失败即停止当前 recipe，不搜索 0.4/0.6，不换任务，不重训 Adapter。

