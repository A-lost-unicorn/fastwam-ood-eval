# Fast-WAM 项目经历｜简略版

## 可直接粘贴到一页简历

**机器人世界动作模型的 OOD 泛化与 Future Utility 机制证伪**｜独立研究与工程实现｜2026.07–08

技术栈：Python、PyTorch、CUDA/EGL、MuJoCo、Fast-WAM、LIBERO/LIBERO-Plus、torchrun、pytest、safetensors

- 从 0 到 1 搭建配置驱动的机器人 OOD 评测系统，通过进程级适配隔离两个同名 `libero` backend，并实现确定性任务清单、episode 级多 GPU 分片、逐条持久化与断点续跑；全量回归测试 `504 passed`。
- 在 3 张 GPU 上完成 4 个任务套件、5 类环境扰动下的 `7,571` 次真实 rollout（`2,399,314` 个 action step），实现 `0 exception`、`0` 重复/遗漏；测得同一 Fast-WAM checkpoint 从 Clean `97.25%` 降至 OOD `47.70%`，定位相机视角为最敏感扰动（成功率 `15.13%`）。
- 设计不反馈控制动作的 shadow-future 诊断，在 `732` 个 episode 上完成 `1,010` 次 probe、`4,040` 个媒体工件审计和 `10,000` 次 task-cluster bootstrap；发现 OOD 下 future consistency distance 增加 `0.0316`、视觉运动方向一致性下降 `0.1898`，并将结论严格限定为非因果关联。
- 设计 `1.371M` 参数、zero-gated 的 Future-to-Action Adapter，用 correct/null/shuffle 反事实证明 future 内容在 `8/8` 样本上改变动作；随后执行双 GPU、预注册 K=0/K=1 matched 训练，K=1 held-out loss 比 K=0 高 `3.624%`、`4/4` 样本更差，证明 action sensitivity 不等于 future utility。
- 在 `64` 个 base state、`256` 个配对样本和 `12,544` 条冻结特征上定位 Camera geometry gap（`+0.020273 m`）高于 Lighting（`+0.011660 m`）；rank-3 correct/shuffle 干预以 `36/36` 逐位恢复/超过 replay floor 将下一假设收敛为 camera equivariance，不将离线动作变化写成任务收益。
- 基于原假设“Camera Equivariance Gap 可由 Geo-REPA + Pose/Ray 修复并恢复 future utility”，实现 `1.335M` 参数 Fast-WAM-GeoEq，在 `3×RTX 4090` 上完成 B1/G3/G4 matched Pilot；G3 gap 缩小 `20.94%<25%`、future utility 仍为 `−0.005231`、Camera success 与基线同为 `1/4`，且 G4 gap 降幅更大，完整机制链未被支持。
- 未继续调参追正结果，而是冻结五项失败 Gate、停止预计 `28–45 h` 的 formal，对既有工件做 CPU-only 失败分解；将问题从“几何不够好”收敛为“future utility 具有 condition/noise-stage dependence”，形成完整的 hypothesis–intervention–falsification 闭环。

## 30 秒项目叙述

我把原问题拆成 Failure→Representation→Sensitivity→Geometry→Intervention→Failure analysis 六层证据。前四阶段确认了 OOD 缺口、future 关联、动作敏感性和 Camera Equivariance Gap；但针对性 Geo-REPA + Pose/Ray 干预没有恢复 aggregate future utility 或 Camera success。我没有继续调参，而是冻结负结果并做只读分解，最终把下一问题收窄到 condition/noise-stage dependence。方法没有涨点，但原假设已被回答，因此这是一个完整的证伪闭环。
