# Thought 2：Future Shadow Diagnostics

研究问题：在保持 Fast-WAM 原动作不变时，离线生成的 unconditional future
与实际执行后的视觉变化是否一致，这种一致性是否与成败相关？

结论：732 个 episode、1,010 个 probe 中，OOD 相比 Clean 的
future-latent cosine distance 增加 0.0316，motion-direction cosine 降低
0.1898；OOD failure 的一致性也更差。由于 future 位于 control loop 外，
这是关联证据，不是“future error 导致失败”的因果证据。

建议阅读顺序：

1. [正式结果](formal_results.md)
2. [概念与因果边界](concepts.md)
3. [统计分析计划](statistical_analysis_plan.md)
4. [执行手册](execution_guide.md)
5. [static/no-op 校准](static_calibration.md)
6. [盲审与抽样](blind_review_and_sampling.md)
7. [上游接口审计](upstream_audit.md)

标注模板位于 [templates/](templates/)；机器权威分析位于
`outputs/thought2/five_category_formal_v1/formal_analysis_v1/`。

