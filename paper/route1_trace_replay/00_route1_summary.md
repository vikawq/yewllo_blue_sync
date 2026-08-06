# 路线一总览：Trace 采集、因果建图与离散事件回放

## 0. 范围、来源与阅读口径

本文综合四篇原论文：

1. [Daydream](01_daydream.md)：USENIX ATC 2020，kernel 依赖图 + 图变换 + what-if。
2. [dPRO](02_dpro.md)：MLSys 2022，全局 DFG + 细粒度通信 + 自动优化。
3. [Echo](03_echo.md)：arXiv 2024，单 GPU ex-situ tracing + 白盒 collective + overlap slowdown。
4. [Lumos](04_lumos.md)：MLSys 2025，Kineto 全栈依赖 + 现代 LLM 3D 并行 trace replay。

原始 PDF 保存在 [sources](sources/)；每篇笔记说明了各自 PDF 版本和 1-based 页码换算，并用“页码 + 节/子节 + Figure/Table/Algorithm”定位证据。

开始阅读论文前，已完整阅读本项目 `survey/` 下 V0.5 详细版、V0.6 实验验证、V0.7 系统化方法论与工程架构、V0.8 系统化设计与源码证据。本文沿用其中的核心区分：**Execution Recipe、Physical Binding、Observation Ledger**，以及逻辑计划、rank-local 物理计划、cross-rank 因果计划三层。

## 1. 路线的共同本质

四篇工作都遵循同一骨架：

```text
真实执行/脱机执行
  → trace 采集与语义补标
  → task/op/kernel 图
  → 数据依赖 + 资源序 + 同步/消息因果
  → duration/cost 注入
  → 离散事件回放
  → 新时间线、step time、breakdown、what-if
```

典型模拟规则可统一写成：

```text
start(v) = max(
    resource_available(resource(v)),
    max(finish(p) for p in predecessors(v)),
    rank_or_request_arrival(v)
)
finish(v) = start(v) + duration(v, context)
```

其中：

- Daydream 主要解决 `predecessors + resource`；
- dPRO 补 `cross-rank message predecessors + link queue + clock alignment`；
- Lumos 补 `CPU thread/CUDA stream/event/sync runtime dependency`；
- Echo 补 `ex-situ rank workload + collective rendezvous + context-dependent overlap slowdown`。

关键判断：**duration 是观测/成本输入，不是 replay 语义本身。** 如果目标 DP/TP/PP/EP、batch、sequence、专家路由或推理请求发生变化，必须先重建合法 workload、依赖和 arrival，再查表/拟合/解析得到 duration；不能只复制旧 duration。

## 2. 三层回放语义

### 2.1 第一层：Logical Execution Recipe（逻辑执行配方）

回答“应该做什么”：

- 模型/算子语义与阶段：forward、backward、optimizer、prefill、decode；
- 输入与输出的逻辑 shape/dtype；
- `raw/valid/padded/storage extent`；
- DP/TP/PP/EP/CP/DCP 的逻辑 group 与 shard 规则；
- collective/P2P 的逻辑消息、字节数和发生条件；
- 动态路径决策：top-k、expert assignment、index、count、branch；
- 状态对象及版本：parameter、optimizer、RNG、KV cache。

四篇论文都只覆盖其子集。Echo 通过框架真实执行获得较强的 op/shape recipe；Lumos 有 PyTorch op 元数据；Daydream 有 layer mapping；dPRO 有 tensor/transaction identity。但没有一篇完整记录数值、决策和状态版本。

### 2.2 第二层：Rank-local Physical Execution Plan（rank 内物理计划）

回答“一个 rank 在本机如何执行”：

- host thread 与 framework/runtime task；
- device stream、kernel、memcpy；
- launch correlation；
- event record/wait；
- device→host sync；
- kernel implementation 与资源绑定。

覆盖强度：**Lumos > Daydream > Echo > dPRO**。

- Lumos 的四类依赖和 fixed/runtime dependency 最完整；
- Daydream 有 CUPTI kernel/CPU/gap，但并发 kernel 采集和现代 stream 语义较弱；
- Echo 侧重 framework op workload 与整体 composer，物理依赖细节不如 Lumos；
- dPRO 的 compute 主要是 framework op，强项不在 kernel/stream。

### 2.3 第三层：Cross-rank Causal Execution Plan（跨 rank 因果计划）

回答“多个 rank 为什么在这个时刻发生通信/等待”：

- communicator/group membership；
- logical rank 与 physical rank/device/NIC binding；
- collective ordinal；
- P2P message match；
- tensor/chunk/step/peer 身份；
- 每个 rank 的 arrival/predecessors；
- collective rendezvous、链路/通道资源和调度序；
- PP stage/microbatch 的发送—接收因果。

覆盖强度：**dPRO 最强的身份化 trace，Echo 最强的规则化 LLM group/rendezvous；Lumos 与 Daydream 需要补全。**

- dPRO 用 transaction ID、chunk/step 和 Middleman 拼全局 DFG，并做时钟对齐；
- Echo 从 group/消息占位符和预定义 schedule 重建跨 rank 时间线，并让 collective 等最后 rank 到达；
- Daydream 可插入 bucket/all-reduce 等通信 task，但内部和跨 rank 身份粗；
- Lumos 在 rank-local stream/event 上很强，论文对全局消息 identity 展开不足。

### 2.4 正交层：Observation Ledger（观测账本）

Observation Ledger 不是第四种“执行语义”，而是对三层计划所需事实与成本的可追溯记录：

- duration 来源：实测/均值/解析/查表/ML/人工倍率；
- 原始时间戳、时钟域、校正偏移；
- shape、bytes、count、index、branch outcome；
- kernel/collective algorithm/protocol；
- 软件/固件/驱动/芯片/频率/拓扑版本；
- 采样次数、均值/方差/分位数；
- 变换 provenance 与置信区间。

Daydream 的 task duration、dPRO 的 10-iteration 均值/时钟偏移、Echo 的 collective/slowdown profile DB、Lumos 的 Kineto 字段都应被纳入 Ledger，但不能替代 Execution Recipe。

## 3. 方法对比矩阵

### 3.1 Trace、图和回放

| 维度 | Daydream | dPRO | Echo | Lumos |
|---|---|---|---|---|
| 目标 | 预测训练优化收益 | 分布式训练诊断与自动优化 | 低成本模拟超大规模 LLM 训练 | 高精度回放/变换 LLM 训练 trace |
| 基线获取 | 真实目标/相近机器 CUPTI | 多 worker 实际运行 trace | 单 GPU 逐 rank ex-situ 执行 | 真实配置 PyTorch Kineto trace |
| 计算粒度 | 单 GPU kernel + CPU API/gap | framework computation op | framework op/shape/time | PyTorch op + CUDA runtime + GPU kernel |
| 通信粒度 | bucket/collective 或切片 task | transaction/chunk/hop SEND/RECV | collective/P2P placeholder + 白盒模型 | trace 中 NCCL/runtime/kernel；变换时补通信 task |
| layer/model 语义 | instrumentation 映射 layer | framework graph/tensor | framework-specific rank workload | operator trace + layer grouping |
| 本地依赖 | thread/stream、launch、sync | framework op 数据依赖 | tracer 图与框架规则 | 四类依赖：CPU/CPU、CPU/GPU、GPU/CPU、GPU/GPU |
| 跨 rank 因果 | 较粗，按梯度/bucket 插入 | 最细：transaction ID、chunk、step、时钟对齐 | group/message + collective rendezvous + schedule rule | 论文展开较少，强项在 rank-local runtime |
| 资源模型 | CPU thread/GPU stream/通信资源 | worker/PS/link FIFO device | compute/comm/memcpy timelines | CPU thread/GPU stream processors |
| runtime dependency | 基本依赖图 | 资源序动态加入 | event-driven composer | 显式 fixed/runtime dependency |
| duration | 实测 + 人工倍率/解析通信 | 10 次实测均值 + fusion cost | 实测 compute + 白盒 comm + XGBoost slowdown | 实测 kernel + 内部 fleet performance model |
| 输出 | 新迭代时间线/收益 | 全局回放、关键路径、优化方案 | 大规模 step time/分解 | 模拟 trace、时间分解、1ms utilization |

### 3.2 what-if、实现和验证

| 维度 | Daydream | dPRO | Echo | Lumos |
|---|---|---|---|---|
| 图变换 | 缩放、增删、选择、重排，最通用原语化 | op/tensor fusion、partition、调度、内存 pass | 配置驱动生成 DP/TP/PP workload | DP/PP、layer 数、hidden/FFN size |
| 新 kernel/shape | 需人工估时，弱 | 离线 profile/cost model | profiler/模型数据库 | 内部 fleet model；论文称 kernel 预测 out of scope |
| overlap contention | 主要沿用 trace，论文揭示 NCCL contention | link/compute 并行，但无通用 slowdown 模型 | 显式 XGBoost slowdown | 原 trace 依赖很细；新并发下无显式 slowdown predictor |
| 最大实证规模 | 4 台×4 RTX 2080Ti | 128×V100 | ground truth 至 96×H800；模拟至 8,192 GPU | 512×H100 |
| 代表误差 | 多项约 <10–13%，P3 最大 16.2% | 多数 <5%，128 GPU 最大约 5.6% | GPT-175B/96 GPU 约 8% | 平均 3.3%，多数 <5% |
| 模拟成本 | 未给统一大规模数字 | 无加速搜索可 >24h；优化后 BERT 0.49h | 128 GPU 83.4s；8,192 GPU 1.38h | 一次流程数秒到数分钟 |
| 代码 | 论文原型；官方公开仓库未确认 | 有 GitHub AE release/DOI artifact | 论文称计划开源；仓库未确认 | 生产原型；仓库和 fleet model 未确认 |
| 现代 LLM | 无，旧 CNN/BERT/GNMT | BERT Base，无现代 3D LLM | 强，13B–175B Megatron/DeepSpeed | 强，15B–175B、3D、H100 |

误差不可横向直接排名：硬件、模型、配置、ground truth、指标和 what-if 距离不同。尤其 Lumos 的 3.3% 是原配置 replay 平均值；新配置需要内部 duration model。Echo 的 8,192 GPU 是模拟规模而非 ground-truth 验证规模。

## 4. 四篇工作的能力演进

### 4.1 Daydream：从 profiler 到“可变换 DAG”

核心贡献是把低层 trace 变成可修改的 kernel DAG，证明 what-if 可以不先实现整个优化。其接口思想至今仍适用，但跨 rank、动态 workload 和现代 LLM runtime 不够。

证据：Daydream §4–§6，PDF pp.5–12。

### 4.2 dPRO：从单机 DAG 到全局 DFG

核心贡献是通信事务身份、跨机时钟对齐、link queue、关键路径和自动优化闭环。它告诉我们：跨 rank replay 的基础不是把多个 Chrome trace 按时间拼在一起，而是可验证的消息匹配与因果边。

证据：dPRO §4–§6，本地 arXiv PDF pp.4–11。

### 4.3 Echo：从“大集群 trace 前置条件”到 ex-situ recipe

核心贡献是无需先拥有目标规模集群，也能逐 rank 生成 framework-specific workload；同时用分层 collective 模型和 overlap slowdown 把模拟成本控制在可用范围。弱点是通信占位执行对数据依赖路径不安全，跨 rank 因果更多来自规则而非同次运行观测。

证据：Echo §4–§8，PDF pp.5–12。

### 4.4 Lumos：从框架 op 图到 CUDA runtime/stream/event 完整物理计划

核心贡献是证明 inter-stream dependency 的缺失会在 LLM 上造成系统性过度重叠，并用 Kineto 内置 trace 低侵入恢复四类依赖。弱点是跨 rank 消息身份、新配置时长、内存和动态 workload 仍弱。

证据：Lumos §3–§5，PDF pp.3–10。

## 5. 共同假设与失效条件

### 5.1 “一个/少量 iteration 可代表未来”

Daydream、dPRO、Lumos 都以稳定迭代模板为主；Echo 的 ex-situ step 也假设占位通信不会改变后续路径。以下场景会失效：

- 动态 sequence/request batch；
- MoE top-k 与 expert load 随输入变化；
- 自适应 kernel/autotuning/recompilation；
- optimizer、checkpoint、evaluation 等低频阶段；
- straggler、故障、后台流量和热/频率变化；
- 推理 continuous batching、prefill/decode 混合、KV eviction。

### 5.2 “图变换后原依赖模式仍成立”

Daydream 的手工 mutation、Lumos 的 layer duplication/PP 规则、Echo 的 schedule rules 都依赖此假设。并行度改变可能改变 fusion、bucket、stream、allocator、collective algorithm 和 kernel family，因此必须区分：

- **preserve**：可原样保留；
- **recompute**：必须从目标 recipe 重算；
- **derive**：可由其他字段确定性推导；
- **constrain**：保留但需要校验；
- **rebind**：只改变物理映射；
- **reject**：缺证据时拒绝生成不可信 replay。

### 5.3 “duration 可移植”

四篇都需要不同形式的时长校准。换 GPU/NPU、shape、并发、通信 topology 或软件版本后，旧 duration 不再是事实。应给每个 duration 标注：

`source + hardware/software fingerprint + shape/workload extent + concurrency context + sample distribution + confidence`。

## 6. 它们与真正录制回放的边界

### 6.1 已经能回放什么

- task 级偏序；
- CPU/GPU/通信资源占用；
- kernel/op/collective duration；
- 部分 framework/layer/tensor/group 语义；
- 原配置或受约束变换后的端到端时间线。

### 6.2 不能据此声称什么

- 不能声称同样的输入得到相同输出；
- 不能声称相同随机数、参数/优化器状态、KV cache；
- 不能声称数据依赖分支、top-k、expert routing 相同；
- 不能声称换并行度后每个 rank 工作量和通信 bytes 自动正确；
- 不能声称目标配置不会 OOM/死锁；
- 不能用相近 step time 证明算子级、通信级和 workload 级都忠实。

### 6.3 推荐的回放目标分级

| 等级 | 目标 | 四篇覆盖 |
|---|---|---|
| L0 数值回放 | tensor/输出/状态等价 | 基本不覆盖 |
| L1 路径回放 | 分支、top-k、index/count、路由等价 | 基本不覆盖 |
| L2 workload 回放 | shape、有效元素、bytes、op 数、collective 序列等价 | 部分覆盖，Echo/Lumos 较强但不完整 |
| L3 性能回放 | 因果、资源、重叠、端到端时间线 | 四篇主战场 |
| L4 operator/kernel 回放 | 真实 device kernel 级物理计划 | Lumos/Daydream 较强，dPRO 较粗 |

## 7. 对 Ascend 训练录制回放的组合方案

### 7.1 Capture

至少采集：

- PyTorch/框架 op、模块、phase、逻辑 tensor；
- ACL/Ascend runtime API、device task/kernel、stream/event、memcpy；
- host→device correlation 与 device→host sync；
- HCCL group、logical/physical rank、collective ordinal、root/peer、message bytes、algorithm/transport/chunk；
- DP/TP/PP/EP/CP/DCP 坐标和 stage/microbatch；
- dynamic shape、valid/padded count、MoE expert assignment、dispatch/combine index/count；
- parameter/optimizer/RNG 状态版本；
- duration 分布与软硬件 fingerprint。

设计来源：Lumos 的四类依赖 + dPRO 的消息 identity + Echo 的 group/workload metadata + 前置 survey 的 Observation Ledger。

### 7.2 Normalize / Build graph

按三层产物拆开：

1. `logical_recipe.json`：与物理 rank/stream 解耦的 workload 和决策；
2. `rank_local_plan/{logical_rank}.json`：host/device task、stream/event、kernel 计划；
3. `global_causal_graph.json`：collective/P2P 匹配、rank arrival、PP/EP 因果；
4. `observations.parquet`：时长、shape、bytes、计数器、版本、置信度。

先以稳定 ID 连接对象，再用时间戳校验；不要反过来靠时间接近猜消息身份。

### 7.3 Target transformation

改变 `world_size/DP/TP/PP/EP/CP/DCP` 时：

1. 从 logical recipe 重算 shard、stage、microbatch 和 communicator；
2. 对动态决策选择 preserve/recompute/derive/constrain/reject；
3. 生成新的 rank-local workload 和消息 bytes；
4. rebind 到目标 NPU/NIC/topology；
5. 由 duration provider 赋时；
6. 校验 shape、collective sequence、内存与状态可行性；
7. 才进入 DES。

### 7.4 Duration provider

应是可插拔接口：

- 原 shape/原设备：实测 trace；
- 相近 shape：查表 + 插值/拟合；
- 新 kernel：微基准/编译器估计/ML 模型；
- HCCL：分层解析模型或全栈通信 simulator；
- overlap：基于并发上下文的 slowdown model；
- 未覆盖：返回低置信度或 reject，不静默复用旧值。

### 7.5 Replay engine

建议合并：

- Daydream 的图变换原语；
- dPRO 的 transaction ID、link queue、partial replay、symmetry；
- Lumos 的 stream/event 和 fixed/runtime dependency；
- Echo 的 collective rendezvous、ex-situ workload generation、context slowdown。

## 8. 对 Ascend 推理/Serving 的额外要求

四篇都以训练为主；Echo 明确只实现训练，Lumos只口头声称可扩推理。因此 serving 不能直接套训练 iteration DAG。必须增加：

- request arrival trace/生成模型；
- scheduler 决策与 admission/preemption；
- dynamic/continuous batching 的 batch membership；
- prefill/decode phase 与每请求 token position；
- KV cache block、allocate/free/evict/swap、版本和归属；
- speculative decoding 的 draft/verify/accept 决策；
- TP/PP/EP 下每 token 的通信 bytes 与 rank arrival；
- TTFT、TPOT、goodput、SLO violation 和 tail latency，而非只看 step time。

## 9. 验证矩阵

不能只用端到端误差。建议逐层验证：

| 层 | 检查项 | 典型指标 |
|---|---|---|
| Recipe | op/phase/shape/count/bytes/decision/state version | exact match、coverage、mismatch count |
| Rank-local plan | thread/stream、launch、event/wait、sync、kernel sequence | edge precision/recall、拓扑合法性 |
| Cross-rank plan | group、ordinal、send/recv、collective sequence、arrival | unmatched message、sequence divergence、deadlock |
| Cost | compute/comm/memcpy/host duration 分布 | MAPE、P50/P95/P99、bias |
| Timeline | exposed compute/comm、overlap、idle、critical path | component error、critical-edge recall |
| End-to-end | step time/throughput 或 TTFT/TPOT | relative error、rank imbalance |
| What-if | 新配置真实小规模对照 | extrapolation error + confidence calibration |

## 10. 落地优先级

### P0：先保证语义可用

- 定义三层 schema 与稳定 ID；
- 采齐 rank-local stream/event/correlation；
- 采齐 HCCL group/ordinal/message identity；
- 记录 dynamic workload 与状态版本；
- 做原配置结构一致性验证。

### P1：实现性能 replay 闭环

- 资源约束 DES；
- compute/communication duration provider；
- overlap/arrival；
- 输出兼容 profiler 的模拟 trace；
- breakdown 与关键路径验证。

### P2：受约束 what-if

- 先支持 DP/PP 或静态 batch/sequence 的安全变换；
- 再扩 TP/EP/CP/DCP 与 MoE；
- 每种变换声明字段策略和 reject 条件；
- 用小规模实机闭环校准。

### P3：规模与生产化

- graph template、coarsening、symmetry、partial replay；
- calibration cache 与软硬件版本治理；
- 不确定性/置信区间；
- 多 iteration、尾延迟、straggler 和背景流量。

## 11. 最终结论

路线一已经充分证明：从 trace 恢复依赖图并用离散事件回放，可以比“算子耗时相加”更准确地重建端到端训练时间，也能支持很多反事实优化。但四篇论文拼在一起仍只形成 L3/L4 性能回放的大部分能力：

- Daydream 提供可变换 kernel DAG；
- dPRO 提供跨 rank 消息因果和全局 DFG；
- Echo 提供 ex-situ workload、快速 collective 与 overlap 校正；
- Lumos 提供现代 LLM 的 thread/stream/event 完整物理依赖。

真正可迁移的 Ascend 录制回放系统必须在此基础上补齐 L0/L1/L2：逻辑 workload extent、动态决策、状态版本、通信身份和目标变换策略。最重要的架构原则是：**先重建“做什么、由谁做、为何等待”，最后才预测“做多久”。**
