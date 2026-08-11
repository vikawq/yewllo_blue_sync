# DNN 性能预测论文笔记库

面向已有后训练/强化学习经验、但尚未系统学习 GPU 编译、分布式执行和 serving runtime 的读者。笔记以论文一手资料为主，并把原论文结果与后续论文复测分开标注。

## 建议入口

1. [AI Infra 基础教程：shape、kernel、Roofline、rank、DP/TP/PP/EP、L1/L2/L3](00_ai_infra_primer.md)
2. [10 分钟速查：概念、公式、并行方式和每篇一句话](00_one_page_cheatsheet.md)
3. [11 篇论文快速阅读地图：路线、演进和数字口径](00_reading_guide.md)
4. [11 篇论文横向矩阵：输入、输出、测量依赖、外推和理论类型](00_cross_paper_matrix.md)
5. 再按下面的系统轴、component 轴或编译器轴进入逐篇笔记。

## 系统级图与事件模拟

| 序号 | 论文 | 核心问题 |
| --- | --- | --- |
| 01 | [Daydream：profile、依赖图变换与回放](papers/01_daydream.md) | 已经观察过一次，如何预测优化后的迭代时间？ |
| 04 | [dPRO：跨设备全局 DFG 与组合优化](papers/04_dpro.md) | 如何对齐、诊断和优化分布式训练关键路径？ |
| 05 | [Proteus：并行策略编译与 HTAE 模拟](papers/05_proteus.md) | 如何在部署前比较 DP/TP/PP 组合？ |
| 06 | [Vidur：LLM serving 的 profile、插值与事件模拟](papers/06_vidur.md) | 如何搜索推理部署和调度配置？ |

## Component/kernel 成本模型

| 序号 | 论文 | 核心问题 |
| --- | --- | --- |
| 02 | [Habitat：wave scaling 与跨 GPU 预测](papers/02_habitat.md) | 同一 kernel 换 GPU 后会多久？ |
| 03 | [nn-Meter：融合感知的移动端 kernel 回归](papers/03_nn_meter.md) | 如何低成本预测大量移动模型的设备时延？ |
| 07 | [NeuSight：tile/wave、Roofline 与利用率学习](papers/07_neusight.md) | 如何外推未见 GPU/模型的 component 成本？ |

## 编译器与自动调优 cost model

| 序号 | 论文 | 核心问题 |
| --- | --- | --- |
| 08 | [TPU learned cost model：XLA 图、tile 与 fusion](papers/08_tpu_cost_model.md) | 编译器如何给 tile/fusion 候选排序？ |
| 09 | [TLP/MTL-TLP：schedule primitive 序列与跨硬件迁移](papers/09_tlp_mtl_tlp.md) | 如何用较少目标数据加速 tensor program 搜索？ |
| 10 | [Ansor：搜索空间生成与学习型 cost model](papers/10_ansor.md) | 自动调优器如何生成并筛选高性能 schedule？ |
| 11 | [TenSet：大规模 tensor program 数据与 ranking](papers/11_tenset.md) | 如何训练、迁移和公平评估 cost model？ |

## 证据标注约定

- **原文结果**：来自该论文的正文、表格或附录；
- **后续复测**：来自另一篇论文在其代码/数据设定下重训或比较；
- **作者 artifact 对齐**：使用作者发布的 profile、权重、opgraph 或 label 重跑；
- **本地探索实验**：在公开测量数据上做的新对照，不等同于论文复现；
- **工程推论**：根据多个来源形成的设计建议，不冒充论文原话。

## 与已有项目文档的关系

- [统一调研、灰盒方案与本地验证](../README.md)：当前方案和实验的主报告；
- [实验复现手册](../experiments/README.md)：Vidur、nn-Meter、NeuSight 和灰盒校准的运行方式；
- 本目录：专注背景教学和逐篇精读，不重复维护实验数字。

## 阅读时始终追问

> 它的输入从哪里来？它预测的是绝对毫秒还是候选排序？shape/route 是否已知？它负责 L1、L2、L3 的哪一层？没有负责的部分由谁提供？
