# Thought5 论文证据链与正式结果模板

## 冻结前史

| 阶段 | 冻结证据 | 能支持什么 |
| --- | --- | --- |
| Thought1 | Clean 97.25%，总体 OOD 47.70%，Camera 15.13% | Camera 是最严重 OOD 类别 |
| Thought2 | distance +0.0316；motion cosine −0.1898 | OOD future-consistency proxy 变差，非因果 |
| Thought3 Phase 1 | correct-null/shuffle 均 8/8 超 replay floor | future 内容对动作技术敏感 |
| Thought3 Phase 2 | A1 比 A0 held-out loss 高 3.624%，4/4 更差 | K=1 sensitivity 未转化为 utility |
| Thought4 formal v6 | Camera gap 0.020273 m > Lighting 0.011660 m；shuffle 36/36 改变动作 | action-consumed geometry 存在 Camera Equivariance Gap |
| Thought5 | Audit/CPU contract 完成；GPU 实验 NOT RUN | 方法和判定协议可执行，尚无效果结论 |

核心叙事是 Failure → Representation → Intervention → Utility，而不是因为外部
方法使用 depth/pose 就堆模块。

## 正式结果占位

以下内容只能从 sealed JSON 自动回填。当前全部为 **NOT RUN**。

| 问题 | 权威工件 | 当前答案 |
| --- | --- | --- |
| H1 Camera gap 是否缩小 ≥25% | `representation_results.json` | NOT RUN |
| future geometry 是否改善 | `future_geometry_results.json` | NOT RUN |
| H2 correct future 是否有 held-out utility | `future_utility_results.json` | NOT RUN |
| H3 Camera success 是否提高 | `rollout_results.json` | NOT RUN |
| 最终机制分类 | `mechanism_classification.json` | NOT RUN |

Phase 5-B 的 geometry 数字必须来自 v2 matched frozen probe（train fit、development
选 alpha、formal 只读），不能直接比较 G3 训练过的 GeoProjector 与 B1 inactive
head。正式 finalizer 会在所有工件 complete 后自动生成下列 15 问的逐项数值回答；
缺任一 collector、episode/task CI 或 G4 pilot specificity 都不能生成 complete report。

## 正式报告必须逐项回答

1. Geo-REPA + Pose/Ray 是否缩小 Camera geometry gap？
2. 改善发生在 Video 哪一层？
3. Action current geometry 和 future SE(3) 是否改善？
4. K=1 future geometry 是否更接近真实未来？
5. A1 是否由 action-sensitive 变为 action-useful？
6. correct future 是否优于 null 和 shuffle？
7. Camera OOD success 是否提高？
8. Clean performance 是否保持？
9. Lighting 与 Robot-init 是否呈现不同模式？
10. B1 matched fine-tuning 能否解释结果？
11. G1/G2 哪个组件贡献更大？
12. G4 是否排除一般正则化解释？
13. 最终属于五种预注册分类中的哪一种？
14. 当前证据能写什么、不能写什么？
15. success 改善是否经由 future utility，还是仅来自 current representation？

## 结论边界模板

- 若 H1/H2/H3 全通过且 controls 排除：支持“Camera Equivariance Gap 是该
  checkpoint future utility 缺失与 Camera OOD failure 的重要机制之一”。
- 若 H1 通过但 H2/H3 不完整：只写 representation repair，不写任务收益。
- 若 H1/H3 通过、H2 失败：写 current-geometry route，不写 future mediation。
- 若 H1 失败、B1 解释收益或 G4 同等有效：方法干预不支持当前机制假说。

无论结果如何，都不得写“唯一原因”“充分原因”或“对所有 WAM 普遍成立”。
