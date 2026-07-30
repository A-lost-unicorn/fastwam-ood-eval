# Paper package

本目录是一套可直接继续修改的论文材料：

| 文件 | 内容 |
| --- | --- |
| [manuscript.md](manuscript.md) | 传统论文格式的完整中英文摘要与中文正文 |
| [evidence_chain.md](evidence_chain.md) | Phase 1→Phase 2 及 Thought1→Thought3 的证据链 |
| [reproducibility.md](reproducibility.md) | 原始工件、SHA、复现实验与重绘命令 |
| [figures/](figures/) | 五张由冻结结果自动生成的 SVG 图 |
| [tables/](tables/) | 核心数字和 Phase 2 逐样本 CSV |

重绘全部图表：

```bash
MPLCONFIGDIR=/tmp/fastwam-paper-mpl \
  .conda/envs/fastwam-ood/bin/python scripts/build_paper_figures.py
```

脚本只读 `outputs/`，并将来源 SHA 和实际作图值写入
[figures/data_manifest.json](figures/data_manifest.json)。

