# 原文与证据状态

本目录保存本轮实际阅读的开放论文原文、逐页文本抽取和少量用于版面核验的页面渲染图。正文研究笔记以论文 PDF 为主证据，项目网页/代码仓库只用于核对当前开源与维护状态。

| 论文 | 本地原文 | 证据状态 |
|---|---|---|
| SimAI | `simai.pdf` | USENIX NSDI 2025 正式 PDF，完整 |
| ASTRA-sim 2.0 | `astra-sim2.pdf` | arXiv 开放全文，完整；正式发表于 ISPASS 2023 |
| Proteus | `proteus.pdf` | arXiv v1 开放全文，完整；后正式发表于 IEEE TPDS 2024 |
| FlexFlow | `flexflow.pdf` | arXiv 开放全文，完整；正式发表于 SysML 2019 |
| Multiverse | `multiverse.pdf` | USENIX NSDI 2025 正式 PDF，完整 |
| ParallelSim | `parallelsim_official_page.pdf` | Springer 正式全文受订阅限制；该文件是官方落地页打印快照，只用于摘要和元数据取证，不是论文 PDF |

ParallelSim 检索记录：正式题名为 *Parallelsim: an accurate, generic, and efficient simulator for distributed deep learning*，DOI `10.1007/s42514-025-00271-w`；Springer 页面显示 2026-03-16 发布、卷 8、页 221–236，ResearchGate 显示 `No full-text available`。`parallelsim_official_page.pdf` 的网页打印页 1–2 可复核题名、作者、出版信息和摘要末尾的“16 DGX A100 / 平均误差 1.83%”，但页面包含 cookie/订阅遮罩。在未取得合法全文前，不能核验正文小节、图表、算法、误差口径、开源仓库或更细实现。

文本文件由 `pypdf` 逐页抽取，页间加入页码标记，便于定位；它们不是新的原始来源。`render/` 中 PNG 仅用于检查复杂版式、图表和公式。
