# Fast-WAM 项目经历｜简略版

## 可直接粘贴到一页简历

**机器人世界动作模型的 OOD 泛化评测与未来效用审计**｜独立研究与工程实现｜2026.07

技术栈：Python、PyTorch、CUDA/EGL、MuJoCo、Fast-WAM、LIBERO/LIBERO-Plus、torchrun、pytest、safetensors

- 从 0 到 1 搭建配置驱动的机器人 OOD 评测系统，通过进程级适配隔离两个同名 `libero` backend，并实现确定性任务清单、episode 级多 GPU 分片、逐条持久化与断点续跑；全量回归测试 `397 passed`。
- 在 3 张 GPU 上完成 4 个任务套件、5 类环境扰动下的 `7,571` 次真实 rollout（`2,399,314` 个 action step），实现 `0 exception`、`0` 重复/遗漏；测得同一 Fast-WAM checkpoint 从 Clean `97.25%` 降至 OOD `47.70%`，定位相机视角为最敏感扰动（成功率 `15.13%`）。
- 设计不反馈控制动作的 shadow-future 诊断，在 `732` 个 episode 上完成 `1,010` 次 probe、`4,040` 个媒体工件审计和 `10,000` 次 task-cluster bootstrap；发现 OOD 下 future consistency distance 增加 `0.0316`、视觉运动方向一致性下降 `0.1898`，并将结论严格限定为非因果关联。
- 设计 `1.371M` 参数、zero-gated 的 Future-to-Action Adapter，以 parameter-free null 和跨 episode shuffle 构造 K=1 动作反事实，验证 future 内容在 `8/8` 固定样本上改变动作；随后执行双 GPU、预注册 K=0/K=1 matched 训练，发现 K=1 未改善 held-out objective，并按冻结停止规则终止 OOD 扩展，保留有效负结果。

## 30 秒项目叙述

我没有直接把“能生成未来”当成“未来对控制有用”，而是把问题拆成四层：先测同一模型在 Clean/OOD 下是否真的掉点，再用不改变动作的 shadow probe 看未来预测与失败是否相关，然后用 correct/null/shuffle 反事实确认 future 内容是否进入动作，最后用 K=0/K=1 matched 训练检验这种敏感性是否形成 held-out 收益。前三层分别观察到行为缺口、相关性和动作敏感性，但第四层没有得到正向收益，因此我按预先冻结的停止规则保留负结果，没有用 OOD outcome 继续反向调参。
