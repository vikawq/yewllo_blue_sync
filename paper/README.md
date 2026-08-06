# 录制回放论文调研索引

本目录汇总分布式 DNN/LLM 训练与推理性能建模、Trace 回放、查表/拟合和全栈仿真的论文精读结果。调研前已对齐 `survey/` 下 V0.5–V0.8 的既有录制回放方法论。

当前覆盖 19 篇不同论文/系统；Vidur 同时从“成本预测”和“Serving 仿真”两个视角分析，因此共有 20 份单篇笔记。每个方向另有横向总览。

## 推荐阅读顺序

1. [TPDS 2024 分布式 DNN 性能建模综述精读](00_TPDS2024_Distributed_DNN_Performance_Modeling_Survey.md)：先建立方法谱系，并理解论文的 analytical / graph-based / execution-driven 与项目三条路线为什么不能机械一一对应。
2. [路线一总览：Trace 采集、因果建图与离散事件回放](route1_trace_replay/00_route1_summary.md)：Daydream、dPRO、Echo、Lumos。
3. [路线二总览：Profiling → 查表/拟合 → 系统级预测](route2_profile_prediction/00_route2_summary.md)：Habitat、Vidur、NeuSight、精度感知训练时间预测器。
4. [路线三总览：训练侧全栈/统一仿真](route3_fullstack_training/00_route3_summary.md)：SimAI、ASTRA-sim 2.0、Proteus、FlexFlow、ParallelSim、Multiverse。
5. [Serving 仿真总览](route4_serving_simulation/00_serving_summary.md)：Vidur、LLMServingSim、APEX、Frontier、Charon。

## 分篇入口

| 方向 | 论文笔记 |
|---|---|
| Trace 回放 | [Daydream](route1_trace_replay/01_daydream.md) · [dPRO](route1_trace_replay/02_dpro.md) · [Echo](route1_trace_replay/03_echo.md) · [Lumos](route1_trace_replay/04_lumos.md) |
| 查表/拟合 | [Habitat](route2_profile_prediction/01_habitat.md) · [Vidur](route2_profile_prediction/02_vidur.md) · [NeuSight / GPU Forecasting](route2_profile_prediction/03_neusight_gpu_forecasting.md) · [精度感知训练预测器](route2_profile_prediction/04_precision_aware_training_predictor.md) |
| 训练全栈仿真 | [SimAI](route3_fullstack_training/01_simai.md) · [ASTRA-sim 2.0](route3_fullstack_training/02_astra_sim_2.md) · [Proteus](route3_fullstack_training/03_proteus.md) · [FlexFlow](route3_fullstack_training/04_flexflow.md) · [ParallelSim](route3_fullstack_training/05_parallelsim.md) · [Multiverse](route3_fullstack_training/06_multiverse.md) |
| Serving 仿真 | [Vidur（Serving 视角）](route4_serving_simulation/01_vidur_serving.md) · [LLMServingSim](route4_serving_simulation/02_llmservingsim.md) · [APEX](route4_serving_simulation/03_apex.md) · [Frontier](route4_serving_simulation/04_frontier.md) · [Charon](route4_serving_simulation/05_charon.md) |

## 统一分析框架

三条技术路线不是互斥分类。更准确的比较方式是同时检查：

1. **执行语义表示**：公式、静态图、实测 Trace 图、可执行 IR 或事件状态机；
2. **节点代价来源**：解析式、Profile 查表、ML、硬件/网络模拟或目标机实测；
3. **建模栈与状态范围**：框架调度、算子/kernel、内存、集合通信、网络、请求/KV 状态；
4. **回放层级**：功能、路径、工作量、性能或 capacity/what-if。

工程上统一映射为：

```text
Execution Recipe
  + Physical Binding
  + Observation Ledger
  + Cost Model
  + Event Runtime
  + Serving State（推理场景）
```

核心原则是：源端观测时长只能进入 Observation Ledger / Cost Model，不能被当作目标 Execution Recipe 的固有语义；换模型、shape、并行策略、硬件或拓扑时，应先重建合法的工作量、状态和因果，再重新估算节点成本。

## 证据口径与已知缺口

- 单篇笔记均说明 PDF/印刷页码口径，并尽量用章节、小节、图、表、算法、公式或段落定位词交叉定位。
- 论文事实、跨论文归纳和面向 Ascend/CANN/HCCL 的工程推断分开标记；未披露信息不补猜。
- 路线一、路线三和 Serving 方向保留了可合法获取的原始 PDF 或来源说明，便于复核版本与页码。
- **ParallelSim** 合法全文受订阅限制，当前仅为官方摘要级证据，不能与全文精读论文直接比较；其未知项和补证清单见单篇笔记。
- **Frontier、Charon** 是 2026 年新论文，笔记将论文能力、公开仓库能力、同行评议/外部复现成熟度分别评价。
- 论文中“模拟到 N 张卡”不自动等于在 N 张卡上取得真机 ground truth；各篇均尽量区分校准规模、验证规模和纯模拟规模。

## 原文与来源说明

- [TPDS 综述来源说明](sources/README.md)
- [路线一原始 PDF](route1_trace_replay/sources/)
- [路线三原文与访问状态](route3_fullstack_training/sources/README.md)
- [Serving 原始 PDF、版本与 SHA-256](route4_serving_simulation/sources/README.md)

