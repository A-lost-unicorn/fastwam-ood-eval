# Paper package

本目录是一套可直接继续修改的论文材料：

| 文件 | 内容 |
| --- | --- |
| [manuscript.md](manuscript.md) | 传统论文格式的完整中英文摘要与中文正文 |
| [evidence_chain.md](evidence_chain.md) | Phase 1→Phase 2 及 Thought1→Thought5 的证据链 |
| [reproducibility.md](reproducibility.md) | 原始工件、SHA、复现实验与重绘命令 |
| [figures/](figures/) | 六张由冻结结果自动生成的 SVG 图 |
| [tables/](tables/) | cohort 规模、阶段发现、Phase 5 对照、证据边界及完整诊断 CSV |
| [最近工作冻结档案](../shared/recent_work_2026-07-27_to_2026-08-06.md) | 07-27—08-06 时间线、价值点与简历口径 |

Thought4 formal v6 的冻结结论为 `camera_equivariance_gap`。Thought5 已实现
Geo-REPA + RayPose/relative-pose 的最小 G3 配方并完成三卡单 task Pilot，但
Camera gap 缩小 20.94% 未过 25% 门槛、future utility 仍为负、Camera success
无提升，故 formal 保持锁定。这形成了完整的
hypothesis–intervention–falsification 闭环；只读分解将下一假设收窄到
condition/noise-stage dependence。该 Pilot 不能写成方法的正式多任务效果或普遍否证；
完整边界见 [Thought4 报告](../thought4/formal_v6_results.md)、
[Thought5 Pilot](../thought5/pilot_v4_results.md)与
[只读失败分解](../thought5/pilot_v4_readonly_failure_analysis.md)。

重绘全部图表：

```bash
MPLCONFIGDIR=/tmp/fastwam-paper-mpl \
  .conda/envs/fastwam-ood/bin/python scripts/build_paper_figures.py
```

脚本只读 `outputs/`，并将来源 SHA 和实际作图值写入
[figures/data_manifest.json](figures/data_manifest.json)。
