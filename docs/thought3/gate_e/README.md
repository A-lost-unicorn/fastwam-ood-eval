# Thought 3 Gate E 索引

Gate E 是 Adapter 训练可用性与统计稳定性的开发证据链，不是 OOD 成功率主表。
主论文只概括其 fail-closed 作用，细节保留在此以复现实验从 0 到 1 的过程。

| Gate | 目的 | 结果入口 |
| --- | --- | --- |
| E 总门禁 | 真实训练入口与最初失败诊断 | [总报告](phase_e_report.md) |
| E.1 | 单样本 fixed-noise 可拟合性 | [协议](phase_e1_protocol.md) / [结果](phase_e1_report.md) |
| E.2 | 八样本 LR/尺度诊断 | [协议](phase_e2_protocol.md) / [结果](phase_e2_report.md) |
| E.3 | held-out multi-flow | [v1 失败](phase_e3_v1_failure_report.md) / [v2 协议](phase_e3_v2_protocol.md) / [v2 结果](phase_e3_v2_report.md) |
| E.4 | diversified train-flow | [协议](phase_e4_protocol.md) / [结果](phase_e4_report.md) |
| E.5 | full-cohort objective aggregation | [协议](phase_e5_protocol.md) / [结果](phase_e5_report.md) |
| E.6 | fresh-cohort replication | [协议](phase_e6_protocol.md) / [结果](phase_e6_report.md) |
| E.7 | checkpoint trajectory 只读诊断 | [协议](phase_e7_protocol.md) / [结果](phase_e7_report.md) |
| E.8 | A0 flow-variance panel | [协议](phase_e8_protocol.md) / [结果](phase_e8_report.md) |
| E.9a | sample-tail mitigation | [v1 失败](phase_e9_v1_failure_report.md) / [v2 协议](phase_e9_v2_protocol.md) / [v2 结果](phase_e9_v2_report.md) |
| E.9a-v2.1 | 对 E.9 的只读有效性审计 | [协议](phase_e9_v2_1_readonly_audit_protocol.md) / [报告](phase_e9_v2_1_readonly_audit_report.md) |

解释规则：工程失败不登记为科学负结果；有效但未过冻结阈值的运行登记为负
Gate；不得从同一小 cohort 反复增加 flow、挑 checkpoint 或降低门槛。

