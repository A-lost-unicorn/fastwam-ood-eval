# Paper package

本目录是一套可直接继续修改的论文材料：

| 文件 | 内容 |
| --- | --- |
| [manuscript.md](manuscript.md) | 传统论文格式的完整中英文摘要与中文正文 |
| [evidence_chain.md](evidence_chain.md) | Phase 1→Phase 2 及 Thought1→Thought4 的证据链 |
| [reproducibility.md](reproducibility.md) | 原始工件、SHA、复现实验与重绘命令 |
| [figures/](figures/) | 五张由冻结结果自动生成的 SVG 图 |
| [tables/](tables/) | 核心数字、Phase 2 逐样本与 Thought4 诊断 CSV |

Thought4 formal v6 的正文结论为 `camera_equivariance_gap`；推荐的
Geo-REPA、relative pose 与 camera-ray equivariance 尚未实现或评测，不能写成
方法效果。完整审计见
[Thought4 formal v6 报告](../thought4/formal_v6_results.md)。

重绘全部图表：

```bash
MPLCONFIGDIR=/tmp/fastwam-paper-mpl \
  .conda/envs/fastwam-ood/bin/python scripts/build_paper_figures.py
```

脚本只读 `outputs/`，并将来源 SHA 和实际作图值写入
[figures/data_manifest.json](figures/data_manifest.json)。
