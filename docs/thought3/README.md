# Thought 3：部分未来 Adapter

研究问题：在冻结 Fast-WAM 主体的前提下，future latent 是否真的进入动作，
以及这种影响能否转化为 held-out 控制目标收益？

当前证据形成两步：

1. [Phase 1 在线动作反事实](phase1_action/report.md)：correct/null/shuffle
   中 future 内容在 8/8 样本上改变动作，证明技术敏感性。
2. [Phase 2 matched 训练](phase2_adapter/report.md)：A0(K=0) 与 A1(K=1)
   各完成 200×28 objectives；A1 的 development loss 比 A0 高 3.624%，
   4/4 样本更差，按预注册规则停止。

目录说明：

| 目录 | 内容 |
| --- | --- |
| [foundations/](foundations/) | Adapter 设计、数据、训练、评测、风险和完成度 |
| [phase_b_d/](phase_b_d/) | 从 mock 到真实 backward，再到 32-sample cache smoke |
| [gate_e/](gate_e/) | E.1–E.9 诊断、预注册、失败与只读审计 |
| [phase1_action/](phase1_action/) | K=1 correct/null/shuffle 技术反事实 |
| [phase2_adapter/](phase2_adapter/) | 完整 28/4 A0/A1 单配方训练 |

完整 Gate E、A2/A4 和 OOD rollout 仍锁定。当前结果不能写成“Fast-WAM 在
OOD 中不需要未来”。

