# Thought 1：环境 OOD 鲁棒性评测

研究问题：冻结官方 Fast-WAM 权重后，从标准 LIBERO 切换到
LIBERO-Plus 环境扰动，成功率下降多少？

结论：800 个 Clean rollout 成功 778 个（97.25%）；6,771 个 runnable OOD
rollout 成功 3,230 个（47.70%），绝对下降 49.55 个百分点。相机视角最敏感，
成功率仅 15.13%。本阶段证明环境鲁棒性缺口，不证明未来想象能修复该缺口。

建议阅读顺序：

1. [正式结果](report.md)
2. [完成度审计](readiness.md)
3. [研究结论边界](generalization.md)
4. [执行与验收手册](execution_guide.md)

机器权威结果位于
`outputs/thought1/fastwam/combined/summary/`。

