# Fast-WAM 项目简历文档

本目录包含两份可直接使用的项目经历，每份同时提供 Markdown 源文档和 Word 版本：

- 简略版：[Markdown](FastWAM项目经历_简略版.md) / [Word](FastWAM项目经历_简略版.docx)。适合一页中文简历，保留从 OOD 评测到 Thought5 负向 Pilot 的核心成果和 30 秒项目叙述。
- 详细版：[Markdown](FastWAM项目经历_详细版.md) / [Word](FastWAM项目经历_详细版.docx)。包含完整项目描述、决策历程、重难点、面试伏笔和回答口径。

文档中的时间按仓库提交记录写为 `2026.07`，角色按单一 Git 作者记录写为“独立研究与工程实现”。如果实际存在团队协作，请在投递前把角色改成真实职责。

所有量化结果均来自仓库中的正式报告或本地测试；其中 Thought 2 是非因果关联分析，Thought 3 的 `8/8` 是动作敏感性而非成功率，Thought4 的 `36/36` 是离线 tensor 干预，Thought5 的 `1/4` 是单 task Pilot。详细使用边界见详细版附录，统一时间线见[最近工作冻结档案](../shared/recent_work_2026-07-27_to_2026-08-06.md)。

Markdown 是可编辑事实源；更新后用项目环境重建两份 Word：

```bash
.conda/envs/fastwam-ood/bin/python scripts/build_resume_docx.py
```

脚本只使用本地 Markdown、字体和 LibreOffice，不访问网络。
