# Thought 6：Sigma-Aware Selective Future Fusion

本目录记录一个独立、零训练的新阶段：保留原始冻结 Fast-WAM 和 Thought3 固定的 K=1 Adapter，仅用 Action Flow 当前的 effective sigma 决定是否注入 future。

当前状态：协议与工程契约已实现，CPU/mock dry-run 通过；真实 Phase 6A/6B/6C 均为 **NOT RUN**。本地缺少 `libero_spatial`、`libero_object`、`libero_10` 三套 LeRobot demonstrations，因此 Phase 6B 按协议 fail-closed，Phase 6C 继续锁定。

## 文档入口

- [冻结协议](protocol.md)：研究假说、六种对照、统计 Gate 与停止规则。
- [运行手册](runbook.md)：审计、dry-run、Phase 6A/6B/6C 命令和三卡时间估计。
- [实现与审计说明](audit_and_implementation.md)：checkpoint、scheduler、任务选择和当前 blocker。

## 当前可写结论

Phase 6 目前只能写“提出并预注册了 sigma-aware selective future fusion，并完成技术实现与测试”。不能写任何 Clean utility、Camera utility 或 LIBERO-Plus success 提升。

