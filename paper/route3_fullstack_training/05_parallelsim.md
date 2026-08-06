# ParallelSim：仅基于官方摘要的受限证据笔记

## 1. 论文身份与证据等级

- 正式题名：*Parallelsim: an accurate, generic, and efficient simulator for distributed deep learning*。
- Springer 落地页：[DOI 10.1007/s42514-025-00271-w](https://link.springer.com/article/10.1007/s42514-025-00271-w)。
- 期刊：*CCF Transactions on High Performance Computing*，2026-03-16 在线发布，卷 8，印刷页 221–236。DOI 字符串中的 `2025` 不是正式发布日期。
- **证据等级**：本轮未取得合法公开全文。Springer 需要订阅，ResearchGate 标为 `No full-text available`；因此以下“原文事实”只来自官方落地页的未分页 Abstract/metadata，不能提供正文 PDF 页、小节、图、表、算法或段落定位。
- 本地 `sources/` 不保存任何假 PDF；访问拒绝返回的 3038B HTML 和伪图片已删除。

## 2. 摘要可确认的贡献

**原文事实（官方摘要，未分页）**：ParallelSim 把 Python/PyTorch 程序转换为 intermediate representation（IR）subgraphs，目标是兼顾通用性。

**原文事实（官方摘要，未分页）**：作者从 GPU hardware design 与 profiling logic 分析 profiling errors 和 computation-communication overlap，并据此做适配。

**原文事实（官方摘要，未分页）**：系统采用 hierarchical simulation engine，将 inter-stage 与 intra-stage simulation 解耦。

**原文事实（官方摘要，未分页）**：在 16 个 DGX A100 节点上评估，报告平均模拟误差 1.83%，并称可用于选择最优并行策略。

## 3. 目前不能从摘要确认的内容

以下均为**证据缺口**，而非论文没有实现：

1. Python/PyTorch 转 IR 的方式：静态分析、FX/JIT、运行插桩、trace 或多者组合。
2. IR schema：是否包含 shape、dtype、动态控制流、optimizer、通信 group/ordinal、stream、内存和依赖。
3. “inter-stage/intra-stage”具体指 pipeline stage、层次调度还是引擎内部划分。
4. 算子/kernel 成本来自实测表、解析式、学习模型，还是 GPU 微架构模拟。
5. collective、NCCL、网络拓扑、拥塞和通信算法的模型粒度。
6. overlap 是事件资源竞争自然产生，还是经验 correction factor。
7. 1.83% 的聚合方式、样本数量、模型、GPU 数、最大/分位误差、训练时间测量口径。
8. 16 节点是校准/验证集群规模还是最大模拟规模。
9. OOM/内存生命周期、pipeline bubble、recompute、DP/TP/PP/EP 支持范围。
10. 代码仓库、许可证、artifact、profile 数据和可复现实验脚本。

**方法纪律**：不能从 Springer 页面列有 Fig. 1–10/Listing 1 就推断图表内容；也不能把标题中的 accurate/generic/efficient 当作独立验证结论。

## 4. 暂定分类，而非完整方法复述

**归纳（低置信）**：摘要显示它更接近“execution-driven frontend/IR + hierarchical simulation”的统一仿真框架，而不是单纯查表或纯网络模拟。但因缺全文，无法判断 IR 是录制所得的源 execution trace、由程序推导的目标 Recipe，还是运行时图。

**归纳（低置信）**：1.83% 与 16 DGX A100 节点是有吸引力的落地指标，但在误差定义、校准边界和 workload 分布未公开前，不宜直接与 SimAI 的 <4%、Proteus 的 3.0% 或 Multiverse 的 <3% 排名。

## 5. 与录制回放五层架构的暂定映射

| 五层 | 摘要可见信息 | 状态 |
|---|---|---|
| Execution Recipe | Python/PyTorch → IR subgraphs | 有候选入口，语义字段未知 |
| Physical Binding | DGX A100、并行策略 | 只知评估平台与用途，绑定细节未知 |
| Observation Ledger | profiling logic/error 分析 | 确有 profiling，但 schema/provenance 未知 |
| Cost Model | 未披露 | 未知 |
| Event Runtime | hierarchical engine，stage 间/内解耦 | 有框架描述，算法与资源模型未知 |

**归纳**：在获得全文前，ParallelSim 只能作为“IR 前端 + 分层仿真 + profiling/overlap 适配”的待验证设计线索，不能成为架构选型或页码级证据的主支撑。

## 6. 对 Ascend/CANN/HCCL 的有限启示

1. Python/PyTorch 自动转 IR 值得作为前端方向：在 Ascend PyTorch 上需要验证 graph break、动态 shape、自定义 op、CANN fusion 和 HCCL 调用能否保留。
2. “按 GPU hardware/profiling logic 修正误差”提示不能盲信 profiler duration；Ascend 侧也要区分 host launch、device task、kernel、HCCL wait 与时间同步误差。
3. stage 间/内解耦可能适合 PP，但在看到 IR 和调度算法前不做工程承诺。
4. 若后续取得全文，应优先核验 IR schema、overlap 方程/算法、通信层、误差样本与代码开放状态，再决定是否吸收到 V0.8 架构。

## 7. 后续补证清单

- 请用户/作者提供合法 PDF 或机构访问副本。
- 补齐每项贡献的 PDF 页、正式印刷页、小节、图表/算法和段落 locator。
- 核验 1.83% 指标的分母、平均方式、训练配置与误差尾部。
- 检索并验证官方代码，而非根据同名仓库猜测。
- 与 SimAI/Proteus 在同一“固定 Recipe / 系统 capacity”口径下重新比较。
