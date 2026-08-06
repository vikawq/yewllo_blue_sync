# Frontier：面向全面、准确 LLM 推理仿真的高保真离散事件仿真器

> 论文正式标题：*Frontier: Towards Comprehensive and Accurate LLM Inference Simulation*，arXiv:2605.21312v2，2026-06-13。本文页码均为 PDF 物理页码；当前日期为 2026-08-06，论文与公开仓库均较新。
>
> 标题核验：arXiv v1（2026-05-20）与本地 v2 PDF 首页/元数据均使用上述正式标题；调研初稿中的 *A High-Fidelity Simulator for Modern LLM Serving* 是描述性简称，并非 2605.21312 的正式 v1/v2 标题。另外，arXiv:2508.03148 *Frontier: Simulating the Next Generation of LLM Inference Systems* 是同团队更早的另一条 arXiv 记录，不能当作 2605.21312 的 v1。
>
> 证据标记：**[论文事实]**、**[综合判断]**、**[迁移推断]**、**[未知]**。论文能力与公开版本可直接运行能力分开陈述。

## 1. 为什么需要 Frontier

**[论文事实]** Frontier 认为旧仿真器的主要误差不再只来自算子时延，而来自现代 runtime 状态：co-location、prefill/decode disaggregation（PDD）、attention/FFN disaggregation（AFD）、CUDA Graph、speculative/MTP decoding、prefix cache、chunked prefill、MoE/EP、分层 KV cache，以及 reasoning/agent/RL 的多轮状态（PDF 第 1–2 页，Abstract、§1，Table 1）。

论文给出的直接例子包括（PDF 第 3–4 页，§2，Figure 1–5、Table 2）：

- CUDA Graph capture bucket 常为 {1,2,4,8,16,32,64}，batch=33 会 pad 到 64；
- 不建模 padding 会造成 co-located 22.6–45.8%、PDD 40–57.2% 的相关偏差；
- CUDA Graph 带来的速度变化在 co-located 为 32.3–46.5%、PDD 为 37.1–59.7%；
- Vidur 式 token-only attention/MoE 特征平均误差可达 32.6%/21.0%；
- 解析 KV 估算可高估预算 8.1%/27.2%，进而把吞吐低估 23.6%/32.4%。

这些例子说明：若 Serving State 没有进入 simulator，即使单算子模型看似准确，配置排序也可能漂移。

## 2. 总体架构与事件模型

**[论文事实]** Frontier 包含四层（PDF 第 4–5 页，§3.1，Figure 6）：

1. Workload & Configuration：模型、硬件、请求、架构与 runtime 特性；
2. Fidelity Plane：compute operator library、memory capacity model、communication backend；
3. Control Plane：scheduler、admission、batching、routing、Serving State；
4. Execution Plane：DES 事件循环、跨 cluster 因果关系与指标。

每个 cluster 有独立 event queue 和一个执行线程；跨 cluster 通过队列传递事件。运行时跟踪 per-request history、batch trace、TTFT/TPOT、吞吐、E2E 与内存（PDF 第 5 页，§3.1 “Discrete-event execution” 段）。

### 2.1 显式依赖

**[论文事实]** Frontier 不把解耦架构简单折成一个总时延：

- PP 用严格 stage chain；
- PDD 发出 KVTransferStart/End；
- AFD 在 attention 与 FFN clusters 间进行 activation ping-pong；
- MoE EP 要等待所有 rank ready，最慢 rank 到达后才触发 combine collective。

见 PDF 第 5–6 页，§3.2 “Parallelism and dependencies” 延续段；更完整伪代码见 PDF 第 17 页，Appendix A.2，Algorithm 1。

**[综合判断]** 这是五篇中最接近“全局因果图 + 离散事件回放”的设计。尤其最慢 MoE rank 决定 combine ready，能自然产生一部分等待；但它仍使用预测代价，并不是从原始 profiler trace 恢复每个 rank 的真实到达。

## 3. Workload、路由与 Serving State

### 3.1 请求、路由与多轮状态

**[论文事实]** 常规请求携带 arrival、prompt/decode 计划；解耦架构会在角色 worker 间传递 KV 或 activation。对 reasoning/agent workload，论文增加 stateful request abstraction：每个请求包含 round 数、每轮 prefill/decode token 计划、工具执行延迟；每轮完成后发 ThinkingRequeueEvent，并通过 session affinity 回到同一 replica 以复用 KV/prefix（PDF 第 6 页，§3.2 “Agentic Reasoning”）。

这比只给 input/output length 的 trace 更接近真实 workload recipe，但仍是“计划 token 数与延迟”，不保存 token 内容、工具返回值或模型分支。

### 3.2 Runtime invariants

**[论文事实]** 在 batch 形成前，Control Plane 固定 capacity envelope 与运行时契约：Graph capture sizes、prefix eligibility、speculative decoding token allowances、watermark 与 preemption。KV manager 以 block 数检查容量，超 watermark 触发 preemption（PDF 第 6 页，§3.2 “Runtime invariants”）。

### 3.3 Runtime Adapters

**[论文事实]** Frontier 把现代优化写成适配 scheduler-batch-engine loop 的 Runtime Adapters（PDF 第 6 页，§3.3）：

- CUDA Graph：pad 到最近 capture size；Graph 路径查询 kernel-only cost，非 Graph 路径查询 launch-inclusive cost；
- MTP/speculative decoding：每请求维护 planned、verified、accepted、committed token 状态；verify 是 prefill-like workload，并允许不同请求接受率不同；
- Prefix cache：用 block hash 标记命中，并在整块完成时更新 cache；
- Chunked prefill：限制每轮长 prompt 的 token budget；
- Scheduler：镜像 vLLM/SGLang 的生产调度逻辑，只把 I/O 替换为仿真事件。

**[综合判断]** 它准确抓住了 serving replay 的本质：优化不仅改变 cost，还改变 batch shape、请求进度和后续调度。局限是论文明确说 prefix cache 避免 token-level KV bookkeeping；因此它仍不能验证逐 token/block 的内容正确性。

## 4. Cost Model、KV 内存与通信

### 4.1 Compute Operator Library

**[论文事实]**（PDF 第 6–7 页，§3.4）：

- token operators：按 token 数与 shard 配 RF/linear model；
- attention：RF 特征包含 batch size、prefill/decode 长度的 total/min/max/percentiles；
- MoE：RF 特征包含 expert load 的 variance/max、selection ratio、expert count 与模型维度；
- 在真实 GPU 上 profile 单 shard，collective 可 stub；
- 分别采集 kernel-only 与 launch-inclusive family，以支持 CUDA Graph 路径。

相比“总 token”特征，这些统计量保留了 batch 内部分布和 MoE skew，但依然不保留每个序列/专家的完整向量。

### 4.2 Memory Capacity Model

**[论文事实]** Frontier 复用 vLLM dummy profile：加载权重、记录 torch peak 和 non-torch residency，再计算可用 KV blocks；运行中由 scheduler 依据 block watermark 做 admission/preemption（PDF 第 6–7 页，§3.4 “Memory-Capacity Model”）。

### 4.3 Communication Backend

**[论文事实]** 根据 domain 与规模选择 ASTRA-sim 或 HTSim，支持 collective 与 point-to-point；PDD/AFD 的 transfer 通过显式事件连接不同 cluster（PDF 第 7 页，§3.4 “Communication Backend”）。

**[综合判断]** 这比单纯查表更能表达依赖和拓扑，但论文没有把 profiler 的 message/wait/transit 分量与真实 peer arrival 全量对齐。Observation Ledger 的重点是校准预测器，而不是保存原始实机执行证据。

## 5. 指标与验证

### 5.1 平台和 workload

**[论文事实]** 实机为 2 台服务器、每台 8×H800 SXM，机内 NVLink 400GB/s，机间 400Gb NDR IB/GPU；vLLM 0.10.2 V1。模型包括 Qwen3-30B MoE、Step3-316B、Llama3.1-8B；workload 包括 prefill-heavy 2048/256、decode-heavy 256/2048、balanced 1024/1024 与 ShareGPT（PDF 第 7–8 页，§5.1，Table 3）。AFD ground truth 来自作者 in-house、未公开实现（PDF 第 8 页脚注）。

### 5.2 微观模型

**[论文事实]** H800 BF16 的 p50/p95 相对误差：attention 3.5%/14.2%，linear 3.3%/6.4%，GMM 1.4%/5.3%；对比 Vidur 特征，attention 达 55.4%/376.1%。FP8 attention p95 为 8.8%（PDF 第 8 页，§5.2，Figure 7）。

KV 初始预算误差在 1.89% 内，而解析模型为 14.1–39.73%；ShareGPT 运行中最大差 294 blocks，即 115.6MB；makespan 在 7.6% 内（PDF 第 8 页，§5.3，Figure 8、Table 4）。

### 5.3 Runtime feature 与端到端

**[论文事实]**（PDF 第 9–10 页，§5.4–§5.5）：

- CUDA Graph speedup 误差：co-located 1.7% 内、PDD 6.1% 内；
- prefix cache 最终 hit rate：模拟 36.98%，实机 37.11%；
- MTP 的 p95 指标最大误差 11.28%；
- co-located 32 个 case 全部在 9.37% 内；
- PDD 32 个 case 全部在 10.99% 内，29/32 小于 9%，20/32 小于 3%；
- AFD 的 TPOT/throughput 误差小于 6.4%/7%；
- Abstract 汇总：16×H800 平均 throughput error <4%，E2E error 从旧模型 co-located 44.9% 降到 6.4%，disaggregated 51.7% 降到 2.6%。

## 6. What-if 案例

**[论文事实]**

- 256×H800 搜索 483,536 个候选，65,190 个 OOM，496 个满足 SLA；输出 Pareto frontier（PDF 第 10–11 页，§6.1，Figure 12）；
- 宽松 TTFT 下 PDD 可达 137.4K，AFD 116.2K、co-located 27.7K；TTFT 收紧到 500ms 后 AFD 116.2K，PDD 100.7K（PDF 第 10–11 页，§6.1）；
- 1,024 个异构 H800/H20 可搜索角色分配与成本效率（PDF 第 11–12 页，§6.2）；
- reasoning 场景的 phase-aware scheduler 把 answer-visible TTFT p95 降低 30.4%，planning throughput 提升 23.2%；trace 为 4K–32K prompt、每轮约 0.2K decode（PDF 第 12 页，§6.3，Figure 14）；
- RL 训练/rollout 根据 active requests 从 DP32/PP16/TP2 切到 DP8/PP16/TP8，重配置 4.52s，makespan 从 528.8s 降到 259.1s，约 2.04×（PDF 第 12–13 页，§6.4，Figure 15）。

这些比例都是特定配置下的 what-if，不应脱离硬件、SLO 和 workload 引用。

## 7. 落地、开源与成熟度

**[论文事实]** 论文称实现约 70K Python LoC，并基于/重构 Vidur；支持 HuggingFace config、PyTorch/vLLM/FlashInfer operator library（PDF 第 7 页，§4）。官方仓库为 https://github.com/NetX-lab/Frontier ，MIT 许可。

**[公开实现现状]** 截至 2026-08-06，README 显示：PDD 与 sequential PD-AF 已开放，但公开 PDD/PD-AF 要求 no-enable_parallel_clusters；并行 disaggregated execution 仍有保护限制；smoke run 默认可能使用 formula analytical backend；只有 h800 与 rtx_pro_6000 提供 full-feature profiles，并建议在目标机重新 profile；scheduler 目前主要对齐 vLLM，其他 engine 仍在计划中。

**[综合判断]** 论文方法和验证很强，但仓库刚发布、提交与使用样本仍少；“论文实现支持”不等于“公共版本默认配置可复现全部实验”。AFD ground truth 还是内部系统。成熟度应写成“前沿公开研究系统，设计最完整，外部复现实验仍需版本锁定与 profile/后端核验”。

## 8. 优缺点与失效边界

### 优点

- 五篇中 Serving State 和 DES 最完整；
- runtime optimization 既修改 cost，也修改 batch shape/状态；
- 显式支持 PDD/AFD/MoE 依赖、CUDA Graph、MTP、prefix、chunking；
- cost 特征覆盖 batch 分布与 expert skew；
- 同时验证微观算子、KV 容量、runtime feature 与端到端指标；
- 能扩到 1K+ GPU 做策略与异构资源搜索。

### 缺点与边界

- 仍是预测仿真，不是原始 kernel trace 或数值回放；
- prefix/KV 采用块计数/哈希抽象，非逐 token/slot 内容状态；
- 主要校准 vLLM，SGLang/TensorRT-LLM 尚未同等验证（PDF 第 13 页，§7）；
- CPU overhead 模型可能不稳，论文假设大规模场景约 90% 时间在 GPU，并把 CUDA API interception 列为未来工作（PDF 第 13 页，§7）；
- 公共 disaggregation 并行模式和 profile 覆盖仍有限；
- AFD 生产 ground truth 不公开。

## 9. 与录制回放分层的对应

| 模块 | Frontier 对应 | 完整度 |
|---|---|---|
| Execution Recipe | workload、架构、runtime feature、stateful rounds | 高 |
| Physical Binding | cluster/role、TP/DP/EP/PP、设备与通信 backend | 高；仍缺目标 runtime 的低层 kernel/graph binding |
| Observation Ledger | operator profile、KV dummy profile、实机验证 | 中高；provenance/原始 profiler 账本仍可增强 |
| Cost Model | 分布感知 RF/linear + kernel/launch families | 高 |
| Event Runtime | per-cluster DES、跨 cluster/collective 因果事件 | 高 |
| Serving State | graph bucket、MTP、prefix、chunk、KV block、preemption、agent rounds | 高；非数值/逐 slot 级 |

结论：Frontier 是当前最接近“**高保真 serving 性能回放内核**”的论文，但仍属于模型驱动仿真。要成为录制回放系统，还需把输入从手工 workload/config 升级为可审计的 Execution Recipe + Physical Binding + Observation Ledger，并提供 preserve/recompute/rebind/reject 规则。

## 10. Ascend / vLLM / SGLang / CANN / HCCL 迁移建议

**[迁移推断]**

1. 优先复用 Frontier 的 Control/Event Runtime 与 Adapter 抽象；为 vLLM/SGLang Ascend backend 各写 scheduler adapter，避免假设二者 admission、prefix 与 chunk 规则一致。
2. Compute Operator Library 重新 profile Ascend：MatMul/FA/GMM、Graph/非 Graph、动态 shape bucket、dtype/format/tiling、workspace 与 launch overhead 分开建 family。
3. CUDA Graph adapter 应抽象成 Graph Capture Adapter，映射到 CANN Graph/ACL Graph 的真实 capture/replay、动态 shape 与 padding 规则，不能只改名称。
4. Memory model 通过实际 CANN/torch_npu allocator 与 vLLM/SGLang block manager 校准；记录权重、静态 workspace、非框架驻留、KV block size、watermark、swap/recompute。
5. HCCL backend 显式保存 communicator/rank mapping、collective algorithm、message/storage extent、ready/wait/transit、链路域和带宽共享；把 profiler 的通信证据作为 Observation Ledger 对账。
6. PDD/AFD 在 Ascend 上应把 KV/activation transfer、格式转换、host staging、stream/event 同步列成独立事件，而非只给一条传输代价。
7. 每次结果都输出证据类型：实测命中、同路径插值、跨 shape 外推、跨硬件解析估算；超出 profile 支持域时降级或拒绝，不静默预测。

## 11. 一句话评价

Frontier 把现代 LLM serving 的“状态改变时间线”做成了真正的 DES，是本组最值得借鉴的 Event Runtime/Serving State 蓝本；它距离可审计、跨平台的录制回放，还差低层绑定、原始观测账本与功能状态。
