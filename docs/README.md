# 文档中心

本目录按研究问题分层，入口保持简短，详细协议、运行记录和论文材料分别归档。
实验输出以 `outputs/` 为机器权威来源；`docs/` 负责解释证据，不覆盖原始结果。

## 快速入口

| 目录 | 内容 | 当前结论 |
| --- | --- | --- |
| [paper/](paper/) | 完整论文、证据链、图表与复现索引 | Future sensitivity 不等于 future utility |
| [thought1/](thought1/) | 标准 LIBERO→LIBERO-Plus 只评测基线 | 97.25%→47.70%，下降 49.55 pp |
| [thought2/](thought2/) | 不改动作的离线 future shadow diagnostics | OOD 一致性代理变差且与失败相关，非因果 |
| [thought3/](thought3/) | Future-to-Action Adapter、技术反事实与 matched 训练 | future 改变动作；K=1 未改善 held-out objective |
| [thought4/](thought4/) | 冻结几何表征—动作接口诊断 | formal v6 完成：支持 `camera_equivariance_gap`；下一分支为 Geo-REPA + camera equivariance |
| [shared/](shared/) | 环境、架构、总控、实验台账和通用协议 | 跨阶段工程与结论边界 |

## 论文主线

```text
Thought 1：OOD 行为缺口
       ↓
Thought 2：future–realized 一致性关联
       ↓
Thought 3 Phase 1：future 内容对动作的技术因果影响
       ↓
Thought 3 Phase 2：K=1 held-out 效用负结果
       ↓
当前结论：能影响动作，不代表能改善控制；OOD success 因果问题仍未回答
       ↓
Thought 4：定位 Video geometry / Action interface / camera equivariance gap
       ↓
诊断结论：Camera shift 对几何表征破坏大于 Lighting；尚未证明修复能提高 success
```

论文级数字、证据强度和不可写结论见
[论文证据链](paper/evidence_chain.md)；可直接修改投稿的正文见
[完整论文草稿](paper/manuscript.md)。

## 维护规则

1. 新实验先写协议，再运行，再写结果；协议与结果不混写。
2. `outputs/` 下冻结工件不因文档整理而修改。
3. 论文图由 `scripts/build_paper_figures.py` 从权威工件生成，不手抄数字。
4. 失败运行保留为工程证据，但不得混入科学主结果。
5. “观察关联”“技术因果”“任务效用”必须使用不同证据标签。
