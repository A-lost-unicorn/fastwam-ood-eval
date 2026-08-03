# Fast-WAM 项目经历｜简略版

## 可直接粘贴到一页简历

**机器人世界动作模型的 OOD 泛化评测与未来效用审计**｜独立研究与工程实现｜2026.07–08

技术栈：Python、PyTorch、CUDA/EGL、MuJoCo、Fast-WAM、LIBERO/LIBERO-Plus、torchrun、pytest、safetensors

- 从 0 到 1 搭建配置驱动的机器人 OOD 评测系统，通过进程级适配隔离两个同名 `libero` backend，并实现确定性任务清单、episode 级多 GPU 分片、逐条持久化与断点续跑；全量回归测试 `443 passed`。
- 在 3 张 GPU 上完成 4 个任务套件、5 类环境扰动下的 `7,571` 次真实 rollout（`2,399,314` 个 action step），实现 `0 exception`、`0` 重复/遗漏；测得同一 Fast-WAM checkpoint 从 Clean `97.25%` 降至 OOD `47.70%`，定位相机视角为最敏感扰动（成功率 `15.13%`）。
- 设计不反馈控制动作的 shadow-future 诊断，在 `732` 个 episode 上完成 `1,010` 次 probe、`4,040` 个媒体工件审计和 `10,000` 次 task-cluster bootstrap；发现 OOD 下 future consistency distance 增加 `0.0316`、视觉运动方向一致性下降 `0.1898`，并将结论严格限定为非因果关联。
- 设计 `1.371M` 参数、zero-gated 的 Future-to-Action Adapter，以 parameter-free null 和跨 episode shuffle 构造 K=1 动作反事实，验证 future 内容在 `8/8` 固定样本上改变动作；随后执行双 GPU、预注册 K=0/K=1 matched 训练，发现 K=1 未改善 held-out objective，并按冻结停止规则终止 OOD 扩展，保留有效负结果。
- 在 `64` 个 base state、`256` 个配对样本和 `12,544` 条冻结特征上定位 Camera geometry gap（`+0.020273 m`）高于 Lighting（`+0.011660 m`）；rank-3 correct/shuffle 干预以 `36/36` 逐位恢复/超过 replay floor 将下一假设收敛为 camera equivariance，不宣称未评测的新方法收益。

## 30 秒项目叙述

我没有直接把“能生成未来”当成“未来对控制有用”，而是先验证 Clean/OOD 行为缺口，再依次检查 shadow-future 关联、future-content 动作敏感性和 K=0/K=1 held-out utility。K=1 没有得到正向收益后，我按冻结规则停止 OOD 扩展；随后用 exact-state geometry probe 与可逆 subspace 干预把最严重的 Camera 缺口定位为等变性问题，但仍把新方法效果留给独立实验。
