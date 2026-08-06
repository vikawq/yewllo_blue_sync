# 推理 / Serving 仿真路线综述：Vidur、LLMServingSim、APEX、Frontier、Charon

> 调研对象：Vidur（MLSys 2024）、LLMServingSim（IISWC 2024）、APEX（arXiv v2，2025）、Frontier（arXiv v2，2026）、Charon（MLSys 2026）。调研日期：2026-08-06。
>
> 标题/版本核验：APEX 从 arXiv v1 的 *Toward High-Performance LLM Serving: A Simulation-Based Approach for Identifying Optimal Parallelism* 改为 v2 的 *APEX: An Extensible and Dynamism-Aware Simulator for Automated Parallel Execution in LLM Serving*，本文全部数字按 v2。Frontier 2605.21312 的 v1/v2 正式标题均为 *Frontier: Towards Comprehensive and Accurate LLM Inference Simulation*；*A High-Fidelity Simulator for Modern LLM Serving* 只是调研初稿中的描述性简称。Charon 2605.17164 的 v1/v2 与 MLSys proceedings 正式标题均为 *Charon: A Unified and Fine-Grained Simulator for Large-Scale LLM Training and Inference*；省略 “and Fine-Grained” 的写法仅是简称。
>
> 本文基于五篇原始 PDF，并结合既有 V0.5–V0.8 录制回放方法论。所有页码均指各自 PDF 物理页码；逐篇证据见 [Vidur](01_vidur_serving.md)、[LLMServingSim](02_llmservingsim.md)、[APEX](03_apex.md)、[Frontier](04_frontier.md)、[Charon](05_charon.md)，原始文件版本与 SHA-256 见 [来源清单](sources/README.md)。Vidur 的 profile-prediction 侧还可参阅相邻路线已有笔记 [route2 Vidur](../route2_profile_prediction/02_vidur.md)。

## 1. 核心结论

五篇论文不是同一种“回放”：

- **Vidur**：用少量 operator/communication profile + RF 预测单轮代价，再在层次化 scheduler 中推进动态 batching。最佳定位是“工作负载回放 + 预测型性能回放”。
- **LLMServingSim**：请求调度、硬件编译/模拟、Chakra execution graph、ASTRA-sim 组成闭环。最佳定位是“全栈硬件/系统 co-simulation”。
- **APEX**：Transformer IR + Parallel Templates + Device Mapper 自动搜索细粒度并行方案。最佳定位是“并行方案生成器 + 相对性能/能耗搜索器”。
- **Frontier**：把 CUDA Graph、prefix、MTP、chunk、KV block、preemption、PDD/AFD/MoE、agent rounds 作为状态与事件。最佳定位是“现代 serving 的高保真 DES”，也是录制回放 Event Runtime/Serving State 的首选参考。
- **Charon**：原生模型 tracing、compiler passes、per-device operator graph 与 profiling/RF/roofline/通信多后端。最佳定位是“图编译驱动的 operator-level 性能预测器”。

若要构建面向 Ascend 的录制回放系统，不应整套照搬任何一篇。更合适的组合是：

1. 用 **Charon** 思路构造编译前/后图和 Physical Binding；
2. 用 **Frontier** 的 adapters + DES 表达请求生命周期和现代 Serving State；
3. 用 **Vidur** 的分型回归作为未命中 shape 的低成本 fallback；
4. 用 **LLMServingSim** 的 execution graph/网络 backend 处理新硬件与跨层 what-if；
5. 用 **APEX** 的并行模板和 logical-to-physical mapping 做可执行策略搜索；
6. 再补上五篇均不充分的 **Observation Ledger、证据分级、低层绑定合法性和功能状态**。

## 2. 统一对比表

| 维度 | Vidur | LLMServingSim | APEX | Frontier | Charon |
|---|---|---|---|---|---|
| 主要问题 | 大规模 serving 配置搜索 | 硬件/软件协同仿真 | 自动并行方案搜索 | 现代 runtime 高保真 DES | 训练/推理统一图仿真 |
| 时间粒度 | iteration | operator graph + system events | iteration/stage | request/batch/op/transfer/cluster events | operator/communication timeline |
| 代价来源 | profile + Random Forest | compiler/cycle simulator + cache reuse | profile + linear interpolation | profile + RF/linear + comm backend | profile + RF + roofline + comm analytical |
| 到达/长度 | 真实 trace；动态验证 Poisson | TSV：input/output/arrival；Poisson | 三类长度分布；Poisson | ShareGPT/合成长短/多轮计划 | 正文未完整披露动态 arrival |
| Continuous batching | 支持多种 replica scheduler | Orca selective/iteration batching | contiguous iteration batching | 镜像 vLLM/SGLang loop | 未完整描述 |
| Chunked prefill | Sarathi 策略/搜索 | 原论文未明确 | 扩展示例 | Runtime Adapter，显式 token budget | 未明确 |
| Preemption | 记录 preempt/restart | 页换出 host、后续 reload | 最近请求临时移除 | watermark + block capacity + preemption | 未明确 |
| KV cache | 容量级 | page 数 + swap/reload transfer | 内存/admission 级 | block 预算、prefix hash、命中与 preemption | 未给出 serving KV manager |
| Prefix cache | 无 | 无 | 无 | 有 | 无 |
| Spec/MTP | 无 | 无 | 无 | planned/verified/accepted/committed | 无 |
| CUDA Graph | 无 | 无 | 无 | bucket padding + kernel/launch 双代价 | 未说明 |
| PD/AF disaggregation | 无 | 原论文无 | 无 | PDD/AFD 显式 transfer/dependency | 仅称 prefill/decode 可分图 |
| TP/PP/DP/EP | TP/同步 PP、replica | TP/PP/hybrid | TP/PP/DP/EP/cell-DP | TP/PP/DP/EP/attention-DP/roles | TP/PP/DP/EP/SP/ZeRO/DualPipe |
| 通信/网络 | profile cost，较粗 | Chakra + ASTRA-sim | collective 查表 | ASTRA/HTSim + 跨 cluster 事件 | 分层解析模型 + overlap |
| 主要指标 | TTFT/TBT/latency/throughput/MFU/memory/SLO | throughput/cycle/request latency | TTFT/TPOT/P95/MFU/MBU/energy/SLO | TTFT/TPOT/E2E/throughput/memory/state history | TTFT/TPOT/TPS/GPU、memory、timeline |
| 最适合 | 容量/配置搜索 | 新硬件/异构 co-design | 并行搜索 | runtime 策略和 serving 架构 | 图/编译/算子/通信策略 |
| 主要缺口 | 现代 state、精细通信 | 成本高、状态版本较旧 | 绝对时延与 state 粗 | 仍非数值/路径回放，公开版较新 | serving loop 弱、未开源 |

## 3. 指标口径：不可只看一个“误差”

### 3.1 TTFT / TBT / ITL / TPOT / E2E

- **TTFT** 应拆成 queueing + prefill service + 必要的 route/transfer；PDD 中还可能包含 KV transfer。
- **TBT/ITL** 是相邻输出 token 的间隔，可能是每 token 序列或 percentile；**TPOT** 常是 output phase 平均每 token 时间。不同论文口径未必等价。
- **E2E latency** 包含排队、prefill、所有 decode iteration、transfer 和可能的工具调用；只用单 iteration error 不能推导 E2E error。
- **Throughput** 需标明 request/s、token/s、goodput 还是满足 SLO 的 token/s。
- **SLO** 必须标明 percentile、阈值、统计对象和是否含 queueing。

Vidur 明确使用 TTFT/TBT 并做 P90/P99 SLO（PDF 第 7、9 页，§5.2、§7.3）；APEX 用 TTFT/TPOT 与能耗（PDF 第 6–8 页，§3.4、§4）；Frontier 用 TTFT/TPOT/E2E/throughput，并验证现代 runtime feature（PDF 第 5、9–10 页，§3.1、§5）；Charon 用 TTFT/TPOT/TPS，但动态到达口径不足（PDF 第 9–13 页，§4.2、§5.2）；LLMServingSim 正文偏 throughput/cycle/request latency，TTFT/ITL/SLO 体系不完整（PDF 第 8–10、15 页，§6、Artifact Appendix）。

### 3.2 验证结果不能横向直接排名

| 论文 | 论文报告的代表性准确度 | 必须同时保留的条件/边界 |
|---|---|---|
| Vidur | 静态 P95 最大 3.33%；85% capacity 动态多数组合 <5% | 95% capacity、小模型最大 12.65%；A100/H100、优化 vLLM fork（PDF 第 8–9、15 页） |
| LLMServingSim | 摘要 <14.7%；NPU-PIM 几何平均 8.88% | RTX3090/vLLM 与模拟 NPU/PIM；不同 backend 的 speedup 口径不同（PDF 第 1、8–9 页） |
| APEX | 相对 speedup 平均误差 10.7% | 绝对 TPOT 系统性偏低；Mixtral EP 误差最高 28%；unconstrained 方案可能不可实现（PDF 第 7–9 页） |
| Frontier | 16×H800 平均 throughput <4%；多数 E2E case <10% | vLLM 0.10.2/H800 为主；AFD ground truth 不公开；公共版 profile/并行解耦有限（PDF 第 7–10、13 页） |
| Charon | inference error <5.35%；RF operator MAE 1.12–2.22% | decode 短算子相对误差更大；请求调度口径不足；无公开 artifact（PDF 第 9–11 页） |

因此建议分别报告：operator error、iteration error、request percentile error、throughput error、SLO classification error、配置排序正确率；同时区分 fixed-workload replay 与 capacity-search。高负载下一个小 cost error 会改变 batch/排队，不能用静态算子精度代替系统精度。

## 4. Arrival、长度、内容与会话：四种 workload fidelity

这些论文主要覆盖前两级：

1. **长度级**：input/output length、arrival time；五篇大多如此。
2. **阶段/轮次级**：prefill/decode、agent rounds、tool delay；Frontier 覆盖最好。
3. **内容/路径级**：token、prefix 相等关系、MoE route、spec draft/accept、stop condition；只有 Frontier 用哈希/接受率等摘要模拟一部分。
4. **功能状态级**：真实 KV 内容、page/block/slot/version、采样 RNG、工具返回、会话一致性；五篇均不完整。

对 workload replay，必须记录 arrival inter-arrival、burst、prompt/decode 联合分布及 tenant/session correlation。简单 Poisson 可用于可控实验，但不等于生产 workload。对 prefix、MoE、spec、agent 场景，仅长度分布无法保持原始执行路径。

## 5. Scheduler 与 Serving State 的关键差异

### 5.1 Scheduler 不是一个策略名

要可重放，至少需要：route/replica 选择、admission、batch token budget、prefill/decode 优先级、chunking、preemption victim/recovery、KV watermark、prefix eligibility、Graph bucket、spec token transition、PD/AF transfer 和完成事件。

- Vidur 的 global/replica/stage 分层非常适合接口设计，但 state 较旧（PDF 第 6 页，§4.4）。
- LLMServingSim 把页换出/换入转为显式 memory transfer，这是重要借鉴（PDF 第 5 页，§4.1）。
- APEX 的 active list/admission 足够做搜索，不足以复现具体 engine（PDF 第 5–6 页，§3.3）。
- Frontier 的 Runtime Adapters 最符合“一个优化同时改变状态和 cost”的现实（PDF 第 6 页，§3.3）。
- Charon 的图 scheduler 解决设备执行顺序，但没有充分定义请求 scheduler（PDF 第 6 页，§3.2）。

### 5.2 建议的 Serving State 最小集合

用于性能回放：

- request_id、tenant/session、arrival、priority/deadline；
- prompt/decode/round/tool 进度；
- replica/role/route；
- batch membership 与每轮 token budget；
- KV block 数、block table、slot mapping、watermark、swap/recompute；
- prefix hash/hit/source、引用计数；
- graph capture bucket/padded batch；
- speculative planned/verified/accepted/committed；
- preemption 原因、victim、恢复路径；
- PDD KV transfer、AFD activation transfer 与完成依赖。

用于功能/路径回放，还要增加 token IDs、RNG/sampling state、真实 KV 或可验证摘要、kernel/graph dispatch key、allocator/version 与错误/超时分支。

## 6. Operator/kernel Cost Model 的三类路线

### 6.1 Profile + 回归/插值

Vidur 用 operator triaging + RF；APEX 用 profile + linear interpolation；Frontier 用更丰富的 batch distribution/MoE skew 特征；Charon 用 per-op RF。优点是快，缺点是跨硬件、backend、Graph/非 Graph、dtype/layout/tiling 时不具备天然可迁移性。

推荐将 cost key 规范为：

model-op semantic + effective/raw/padded/storage shape + dtype/format + shard + kernel/backend + graph mode/bucket + device/CANN version + workspace + topology/group。

### 6.2 Compiler / cycle-level simulation

LLMServingSim 用 compiler/hardware simulator，能探索不存在的硬件；但大规模仍昂贵，GPT-3 175B、2048 NPU 的一个 iteration 需 4.13 小时（PDF 第 10 页，§6.5）。适合离线建表和稀疏高价值候选，不适合所有事件在线调用。

### 6.3 多保真组合

Charon 的 profiling/prediction/analytical/fused 和 Frontier 的 kernel-only/launch-inclusive family 提供了方向。实际系统应按以下优先级：

1. 同硬件、同版本、同路径精确 profile；
2. 同路径邻域插值/回归；
3. 解析模型/硬件 simulator；
4. 超出支持域则 reject 或明确低置信度，不静默外推。

每个 cost 必须回链 Observation Ledger，并记录 evidence source、sample count、分布、profile 环境、外推距离和置信度。

## 7. 通信：duration 不是固有常数

五篇中：Vidur/APEX 更接近通信查表；Charon 加入拓扑/算法/拥塞和 overlap；LLMServingSim/Frontier 用系统网络 backend 与显式依赖，Frontier 还让 MoE combine 等待最慢 rank。

录制回放必须把通信至少拆成：

- message：payload/storage extent、collective/P2P、group/ranks；
- readiness：每个 rank 何时到达；
- wait：因 peer/stream/dependency 未就绪的时间；
- transit/service：网络实际占用时间；
- contention：并发 collective/P2P 的链路资源共享；
- overlap：与 compute/其他 communication 的可重叠区间。

否则把 profiler 中一次 all-reduce duration 原样复制，会把源系统的 wait 与目标拓扑混在一起。Frontier/LLMServingSim 的事件图可以作为 runtime 基础，Observation Ledger 则必须来自真实 profiler 的 message/wait/transit 与 rank mapping。

## 8. 六层录制回放框架映射

| 论文 | Execution Recipe | Physical Binding | Observation Ledger | Cost Model | Event Runtime | Serving State |
|---|---:|---:|---:|---:|---:|---:|
| Vidur | 中高 | 中 | 中 | 高 | 高（迭代级） | 中低 |
| LLMServingSim | 中高 | 高 | 中 | 高但昂贵 | 高 | 中 |
| APEX | 中高 | 中 | 中 | 中高 | 中低 | 低 |
| Frontier | 高 | 高 | 中高 | 高 | 高 | 高 |
| Charon | 高 | 中高 | 中 | 高 | 中高（算子级） | 低 |

### 8.1 Execution Recipe

Frontier 的 workload/runtime configuration 与 Charon 的原生模型/compiler passes 最有参考价值。需要补 token/content、session、采样、异常路径与版本冻结。

### 8.2 Physical Binding

LLMServingSim 的 device/operator graph、APEX 的 logical-to-physical mapper、Charon 的 per-device graph 可组合。仍须记录真实 rank/device/stream/kernel/graph/format/storage/group，而不是只有“TP=8”。

### 8.3 Observation Ledger

这是五篇共同薄弱处。它们有 profile database，但通常不把原始 profiler 事件、调用栈、环境、采样覆盖和事实/推断等级设计为可审计账本。录制回放系统必须新增这一层。

### 8.4 Cost Model

五篇贡献最集中：Vidur 低成本分型、LLMServingSim 硬件模拟、APEX 搜索友好插值、Frontier runtime-aware 特征、Charon 多后端。可采用分层 fallback。

### 8.5 Event Runtime

Frontier 最完整；LLMServingSim 的 Chakra/ASTRA 适合硬件网络；Charon 适合 operator graph；Vidur 适合轻量 iteration simulation；APEX 更适合搜索粗筛。

### 8.6 Serving State

Frontier 是唯一覆盖 Graph/MTP/prefix/chunk/解耦/agent 的系统性蓝本；LLMServingSim 的 KV swap 节点也值得吸收。但五篇都没有完整数值/KV slot 状态。

## 9. 与五种“回放”含义的关系

| 回放目标 | 要求 | 五篇覆盖 |
|---|---|---|
| Functional replay | 输出/状态正确，token/KV/RNG 可验证 | 基本不覆盖 |
| Path replay | operator/kernel/branch/collective 路径一致 | Charon/LLMServingSim 部分生成相似图，但非原 trace 保真 |
| Workload replay | 到达、长度、batch、路由、状态转换一致 | Vidur/Frontier 较强；Frontier 最完整 |
| Performance replay | 时间线、排队、通信、资源争用与指标可信 | Frontier/LLMServingSim 较强；其余适合特定层级 |
| Capacity/what-if simulation | 搜索未运行配置与硬件 | 五篇均覆盖，APEX/Frontier 最强调搜索 |

因此推荐用明确标签：

- **L2-W**：保持工作负载/调度语义，cost 可重估；
- **L2-P**：保持目标平台物理绑定与依赖，重建时间线；
- **L1-F/L1-Path**：功能/路径等价，需要额外状态与低层 trace。

不要把 L2-W 仿真准确率写成“复现了原执行路径”。

## 10. 开源、实现和复现成熟度

| 论文 | 论文/版本 | 官方实现（截至 2026-08-06） | 成熟度判断 |
|---|---|---|---|
| Vidur | MLSys 2024，arXiv v2 | https://github.com/microsoft/vidur ，MIT | 公开研究原型，易二次开发 |
| LLMServingSim | IISWC 2024 | https://github.com/casys-kaist/LLMServingSim ，MIT/Artifact/Zenodo | 可复现 co-simulator；仓库已演进 2.0，需锁版本 |
| APEX | arXiv v2 2025；PDF venue 占位 | https://github.com/microsoft/apex_plus ，MIT/Zenodo | 公开研究原型；不可把占位模板当录用证明 |
| Frontier（正式标题 *Towards Comprehensive and Accurate LLM Inference Simulation*） | arXiv:2605.21312v2，2026；v1/v2 标题相同 | https://github.com/NetX-lab/Frontier ，MIT | 最新、能力强；公开并行解耦/profile 覆盖仍有限 |
| Charon（正式标题含 *Unified and Fine-Grained*） | arXiv:2605.17164v2；MLSys 2026；v1/v2/proceedings 标题相同 | 未发现作者官方代码 | 内部落地证据强，外部复现性低 |

复现时应保存：paper version、repo commit、profile bundle hash、engine/CUDA/CANN version、模型 revision、trace hash、命令行、随机种子和硬件拓扑。尤其不能用 2026 仓库的新功能替 2024 论文背书。

## 11. 面向 Ascend / CANN / HCCL 的组合落地方案

### 11.1 录制侧

- 从 Ascend Profiler/CANN runtime 采集 framework op、kernel/task、stream/event、memory、HCCL、host launch 与调用栈；
- 记录原始、有效、padding、storage extent，避免把 padding workload 与有效 token 混合；
- 冻结模型、tokenizer、vLLM/SGLang Ascend backend、torch_npu/CANN/HCCL、Graph 与 kernel 版本；
- 记录 request/session/token recipe、scheduler decisions、batch composition、KV block/slot/prefix/spec/preemption/PD transfer；
- 构造 stable IDs，将 request→batch→op→kernel→communication→rank/device 串起来。

### 11.2 建模侧

- Compute：同路径 profile 优先，RF/MLP 插值其次，roofline/硬件 simulator 兜底；
- Communication：HCCL topology/group/algorithm-aware DES，显式 ready/wait/transit/contention；
- Memory：权重、workspace、allocator/non-framework residency、KV block、swap/recompute；
- CPU/control：Python/C++ scheduler、launch、Graph build/replay、tokenizer 与 IPC 单独建模，避免小模型/高负载时级联误差；
- Runtime adapters：CANN Graph、chunked prefill、prefix、spec/MTP、PDD/AFD、MoE/EP、agent rounds。

### 11.3 回放侧

建议使用 Frontier 风格 per-cluster event queues，但节点来自 Charon 风格编译/执行图，并由 Observation Ledger 绑定实测 evidence。每个节点执行以下策略之一：preserve、recompute、derive、constrain、rebind、reject。地址、Graph executable、communicator 等不可移植状态不能静默复用。

### 11.4 验证侧

分层对账：

1. operator/kernel：时延分布、shape/path/format；
2. communication：message/wait/transit、collective ready、overlap；
3. iteration/batch：batch composition、token budget、KV blocks、stage time；
4. request：queue、TTFT、ITL/TBT、TPOT、E2E、preemption；
5. system：throughput/goodput/SLO/memory/MFU/MBU；
6. causal drift：第一处 scheduler/state/path 分叉，而不只看最终平均误差。

验证集要覆盖：静态低负载、动态 50/85/95% capacity、长 prompt/长 decode、burst、OOM 水位、prefix 命中、Graph bucket 边界、spec 接受率、MoE skew、PD/AF transfer 与多机 HCCL contention。

## 12. 推荐技术选型

### 如果目标是快速做容量规划

以 Vidur 为最小基线；加入 Ascend profile、vLLM/SGLang scheduler adapter 和 KV block state。优先保证 TTFT/TBT/SLO 口径和 85%/95% load 验证。

### 如果目标是新 NPU/网络架构探索

采用 LLMServingSim 式 hardware execution stack + ASTRA/自研网络仿真；用缓存/多保真代理降低 cycle simulation 调用次数。

### 如果目标是自动并行搜索

采用 APEX templates/mapper，但设计空间分成 executable 与 unconstrained；候选先粗模筛选，再由 Frontier/实机复核。

### 如果目标是现代 serving 策略复现

以 Frontier 为 Event Runtime/Serving State 蓝本，优先补 public repo 与目标 backend 的差距：并行 PDD/AFD、SGLang、Ascend Graph、KV allocator 与 HCCL。

### 如果目标是算子/编译路径 what-if

采用 Charon 式原生 graph + compiler passes + fused multi-engine；同时保留编译前后 stable mapping 和真实 CANN kernel binding。

## 13. 最终判断

这组论文表明，serving 性能预测已经从“算子时间相加”演进到“状态驱动的因果仿真”。真正决定高保真度的不是 cost model 单点误差，而是三件事能否同时成立：

1. **Execution Recipe/Serving State 足够完整**，不会在 batch、KV、Graph、spec、route 上提前分叉；
2. **Physical Binding/Observation Ledger 可审计**，知道哪个时间来自何种硬件、版本、路径与证据；
3. **Event Runtime 让等待和重叠从依赖与资源竞争中产生**，而不是把源 trace 的 duration 当常数复制。

Frontier 最接近第 1、3 点；Charon/LLMServingSim/APEX 补图、硬件和并行；Vidur补低成本代价预测。第 2 点仍需我们的录制回放体系自行补齐，也是从“仿真器”走向“可解释、可迁移、可拒绝的性能回放系统”的关键差异。
