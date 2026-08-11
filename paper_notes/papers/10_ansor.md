# Ansor：自动生成高性能 Tensor Programs（OSDI 2020）

## 元信息与一手资料

- 论文：Lianmin Zheng 等，*Ansor: Generating High-Performance Tensor Programs for Deep Learning*，OSDI 2020。
- 一手资料：[USENIX 论文页](https://www.usenix.org/conference/osdi20/presentation/zheng)；[USENIX PDF](https://www.usenix.org/system/files/osdi20-zheng.pdf)；[arXiv extended version](https://arxiv.org/abs/2006.06762)。
- 开源落地：论文已集成 Apache TVM；后来 TVM 将新一代机制演进为 MetaSchedule，但本文笔记以论文当时的 Ansor 实现为准。
- 论文核心：不是单独发明一个时延预测模型，而是一个 **search-space construction + evolutionary tuning + learned cost model + task scheduling** 的 tensor program 生成系统。

## 30 秒总结

一句话类比：**Ansor 先自动画出几类“房屋骨架”（sketch），再给每个骨架随机填楼层尺寸、材料和施工细节（annotation），最后用成本模型指导遗传搜索，只把少量好设计真正建出来验收。**

在 Ansor 前，AutoTVM 依赖专家为每个 operator/hardware 写模板；Halide 类 sequential construction 又会在 program 尚未完整时用 cost model 早剪枝，容易把“前期难看、最终很快”的方案扔掉。Ansor 用两层搜索空间避免这两个问题：

1. 用通用 derivation rules 枚举少量高层 sketches；
2. 随机填 tile size、parallel、unroll 等低层 annotations，始终得到完整程序；
3. 用 evolutionary search + learned cost model 细化完整程序；
4. 用目标机实测回填 cost model；
5. 用 task scheduler 把测量预算优先给最可能改善端到端模型的 subgraphs。

最容易被误写的一点：**Ansor 原论文的 cost model 是 GBDT（梯度提升树）+ 加权平方误差，标签是每个 DAG 内归一化 throughput；不是 MLP + RankLoss。** MLP/LambdaRank 的系统比较来自后续 TenSet。

## 背景：模型图为什么还要变成 tensor program

### 从模型 operator 到机器代码

PyTorch 中的一次 `matmul` 是“做什么”的数学语义，不是“怎么在某硬件上做”。部署时，编译器要把它变成循环、缓存、线程和指令：

```text
模型 DAG（matmul / conv / relu ...）
  ↓ graph partition / fusion
subgraphs
  ↓ schedule / tensor program generation
loop nests + memory scopes + parallel/vector/thread mapping
  ↓ LLVM / NVCC 等后端
机器代码
```

同一个 `C[i,j] = Σk A[i,k]B[k,j]` 可以有大量合法实现：多层 tiling、loop reorder、parallelization、vectorization、unrolling、producer-consumer fusion、cache stage、reduction factorization。最优组合依赖 shape 和硬件。

### 三种路径

1. **vendor library**：cuDNN/oneDNN 等由专家长期优化，常见算子很强，但新算子、特殊 shape 和跨 operator fusion 覆盖有限。
2. **template-guided tuning**：专家写结构模板，只搜索 tile/unroll 参数。质量不错，但模板工程量大、搜索空间被模板边界锁死。
3. **auto-scheduling/program generation**：从计算定义自动构造并搜索实现。Ansor 属于第三类。

### 为什么 sequential construction 容易误剪枝

如果按固定顺序一步步构造 program，搜索算法需要在“只完成一半 schedule”的候选间做 top-k 剪枝。但真实标签只能来自**完整可编译程序**。用在完整程序上训练的 cost model 去预测不完整程序的最终性能，会有 distribution mismatch。

Ansor 的策略是：先采样完整 program，再评分、mutation/crossover。这样 cost model 始终看到与训练标签同类的对象。

## 输入、输出与系统边界

| 项目 | Ansor 中的含义 |
| --- | --- |
| 输入 | 一个或多个 DNN；论文支持 ONNX、TensorFlow PB 等，经 Relay fusion/partition 得到 subgraphs |
| 每个 task | 为一个确定 shape 的 subgraph + target hardware 搜索高性能 tensor program |
| 搜索输出 | 每个 subgraph 的低层实现及其目标机实测 runtime |
| cost model 输出 | 候选的归一化 throughput score/fitness，主要服务排序 |
| 端到端输出 | 编译后的模型执行程序，不是一个通用 SLA 时延预测 API |

关键假设与范围：

- shape 静态、编译前已知；
- 以 dense operator 为主；
- 论文主要评估 inference、float32；
- 每个具体 shape 是不同 task，最佳 program 也可能不同；
- 最终质量仍依赖目标硬件测量闭环。

## 方法总览

```text
DNNs
  ↓ Relay graph fusion / partition
subgraph tasks
  ↓
Program Sampler
  ├─ sketch generation：枚举高层结构
  └─ random annotation：填 tile/parallel/unroll 等细节
  ↓ complete programs
Performance Tuner
  ├─ cost model 预测 fitness
  ├─ mutation / crossover
  └─ top candidates 上目标硬件实测，回填 cost model
  ↑
Task Scheduler：决定下一批预算给哪个 subgraph
```

## 1. Hierarchical search space：Sketch + Annotation

### Sketch 是什么

Sketch 只决定高层结构，例如：

- compute-heavy、有 reuse 的 matmul/conv 要多层 tiling；
- 是否把 consumer（bias/ReLU）融合进 tiled producer；
- 是否增加 cache write stage；
- reduction parallelism 不够时是否做 `rfactor`；
- 简单 elementwise node 是否 inline。

典型 subgraph 的 sketches 少于 10 个，但每个 sketch 可对应海量低层参数组合。

### Derivation rules

Ansor 从 DAG 输出节点逆拓扑地应用规则。论文 CPU 规则包括：

1. Skip；
2. Always Inline；
3. Multi-level Tiling；
4. Multi-level Tiling with Fusion；
5. Add Cache Stage；
6. Reduction Factorization；
7. 允许用户扩展特殊规则。

谓词如 `HasDataReuse`、`HasFusibleConsumer` 由计算定义的 read/write pattern 静态分析得到。

### Annotation 是什么

在 sketch 中尚未确定的低层量包括：

- 各层 tile factors；
- 哪些 loop parallel/vectorize/unroll；
- unroll pragma；
- producer 的 compute location；
- GPU block/virtual-thread/thread binding。

CPU compute-heavy loop 使用 `SSRSRS` 多层结构：S 表示 spatial loop tile，R 表示 reduction loop tile。对 matmul 的 `i,j,k`，它可展开成类似：

```text
i0, j0, i1, j1, k0, i2, j2, k1, i3, j3
```

GPU 使用 `SSSRRSRS`，前三层 spatial tile 分别映射到 BlockIdx、virtual thread、ThreadIdx，并增加 shared-memory cache 与 cross-thread reduction 规则。

### 为什么“两层”有用

- 若全靠模板：高层结构种类太少。
- 若把每个低层参数都显式枚举：组合爆炸。
- Sketch 把“结构”与“参数”分开：少量结构可枚举，海量细节可随机采样。

随机采样不是最终优化器，它的作用是给所有区域被发现的机会；之后 evolutionary tuner 再把随机点推向高质量区域。

## 2. Performance tuning：完整程序上的 evolutionary search

每轮过程是：

1. 取随机新样本 + 之前实测的优秀 programs 作为 population；
2. cost model 快速预测数万候选的 throughput fitness；
3. 按 fitness 概率选择 parents；
4. mutation/crossover 生成后代；
5. 多代后挑高分候选；
6. 编译并在目标硬件测量；
7. 将真实 runtime/throughput 加入数据，重新训练 cost model。

### 专门为 tensor program 设计的 mutation

- **Tile-size mutation**：把某层 tile factor 除以一个因子，再乘到另一层，保持各层乘积等于原 loop extent，确保合法。
- **Parallel mutation**：融合或重新 split 邻近 loop，改变并行粒度。
- **Pragma mutation**：改变如 `auto_unroll_max_step=N`。
- **Computation-location mutation**：把灵活 producer 移到另一合法 attach point。
- **Node-based crossover**：组合两个 parent 的 rewrite histories；以 DAG node 为粒度降低依赖冲突，并做合法性验证。

这比对固定参数网格做通用 GA 更强，因为 mutation 理解 program transformation 的合法约束。

## 3. Learned cost model：原文到底怎么建模

### 输入特征与聚合

Tensor program 由多个交错 loop nests 和 innermost non-loop assignment statements 组成。Ansor：

1. 对每个最内层 statement，在完整 program 上下文中提取算术和内存访问特征；
2. 模型 \(f(s)\) 给每个 statement 一个 score；
3. 对 statement scores 求和得到 program score。

论文正文把详细特征列表放到 extended version；后续 TLP 论文概括为来自计算、访存、算术强度等五方面的 164 个特征。

### 标签与损失

对同一个 DAG 的所有候选，把 throughput 归一到 `[0,1]`。设 program \(P\) 的真实归一 throughput 为 \(y\)，最内层 statements 集为 \(S(P)\)，预测为 \(\sum_{s\in S(P)}f(s)\)。损失为：

\[
L(P)=y\left(\sum_{s\in S(P)}f(s)-y\right)^2.
\]

由于 weight 直接取 \(y\)，快程序权重更高。这是“为 top candidates 倾斜的回归”，而不是 RankLoss。

底层模型是 **gradient boosted decision tree（GBDT/XGBoost 类）**。优化一个 DNN 时通常少于 3 万个已测 programs，树模型重训很快，所以每轮直接从头训练，不做增量更新。

### 为什么输出不适合当绝对时延

- throughput 在每个 DAG 内独立归一；跨 DAG 的 0.8 不代表相同 FLOP/s 或毫秒；
- program score 可为负；
- 目标是把好候选排前，不是校准物理时间；
- 真 winner 仍由目标机 measurements 决定。

所以 Ansor cost model 很适合 autotuning，不适合直接做容量规划/SLA。

## 4. Task scheduler：把测量预算花在瓶颈 subgraph

一个模型会有很多不同 shape 的 subgraphs。ResNet-50 在论文中有 29 个 unique subgraphs。若每个 task 都给相同的 1000 trials，会浪费预算：有些不是瓶颈，有些已难再优化。

设 \(t_i\) 是给 task \(i\) 的测量批次数，\(g_i(t)\) 是当前找到的最小 subgraph latency，端到端目标为：

\[
\min_t f(g_1(t),\ldots,g_n(t)).
\]

单模型 latency 可近似为：

\[
f=\sum_i w_i g_i(t),
\]

其中 \(w_i\) 是该 subgraph 在模型中出现次数。Ansor 以历史改善率和相似 task 的性能作 optimistic estimate，近似 \(\partial f/\partial t_i\)，每轮把资源给绝对梯度最大的 task，并用 \(\epsilon\)-greedy 保留探索。

直觉：先调最慢、看起来最有潜力的 subgraph；如果多轮不再下降，它的预计边际收益变小，预算转向其他 task。

这和 RL 很像一个 non-stationary bandit/resource allocation，但论文实现用 gradient-like heuristic，不是训练一个 RL policy。

## Worked example：Matmul + ReLU

输入 subgraph：

\[
C_{ij}=\sum_k A_{ik}B_{kj},\qquad D_{ij}=\max(C_{ij},0).
\]

### Sketch 级选择

- Sketch A：matmul 多层 tiling，ReLU consumer 融到 tiled producer。
- Sketch B：matmul tiling，但独立 cache write stage，再写回 D。
- Sketch C：当输出维很小、reduction 很大时用 rfactor 引入 reduction parallelism。

### Annotation 级选择

对 Sketch A，可能采样：

```text
i factors = [4, 8, 2, 8]
j factors = [2, 4, 8, 8]
k factors = [32, 16]
parallel = fuse(i0, j0)
vectorize = j3
unroll = k1
```

随机配置未必快。cost model 从实测过的其他完整 programs 学到，大 tile 可能 reuse 更好，但 parallel 粒度和 vector width 也要匹配。GA 通过移动 tile factors、改变 parallel/unroll 来搜索邻域。最后 top candidates 才上 CPU/GPU 实测。

若 `M=N=K=512` 改成 `M=8,N=4,K=512`，最优 sketch 可能从空间并行变为 rfactor reduction parallelism。**这就是 shape 变化为何不能只复用旧时延或旧 schedule。**

## 实验设置与原文结果

### 平台与工作负载

- Intel CPU：18-core Platinum 8124M@3.0GHz（部分图注使用 20-core 8269CY 环境）；
- NVIDIA V100；
- ARM Cortex-A53（Raspberry Pi 3B+）；
- 所有评估为 float32。

单算子覆盖 1D/2D/3D conv、matmul、group/dilated/depthwise/transposed/capsule conv、matrix 2-norm；每算子 4 种 shape × batch 1/16，共 80 cases。端到端模型包含 ResNet-50、MobileNet-V2、3D-ResNet-18、DCGAN、BERT。

### 代表结果

- 相对最佳替代方案，端到端 DNN execution performance 最大提升：Intel CPU 3.8×、ARM CPU 2.6×、NVIDIA GPU 1.7×。
- 与 AutoTVM 比，端到端整体为持平到约 1.8×；优势尤其来自 uncommon shapes、小 batch 和模板未覆盖结构。
- 两个 multi-op subgraph（Conv+BN+ReLU；transpose+batch-matmul+softmax）上，Ansor 相对库/搜索 baseline 约 1.1–4.2×。
- ResNet-50 的 weight layout rewrite 带来约 40% 改善；这说明端到端收益不只来自 cost model。

### 搜索时间

达到 AutoTVM 在 Intel CPU、batch=1 的性能时：

| 网络 | AutoTVM measurements | Ansor measurements | 节省倍数 | AutoTVM wall time(s) | Ansor wall time(s) | 节省倍数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet-50 | 21,220 | 6,403 | 3.3× | 39,250 | 4,540 | 8.6× |
| MobileNet-V2 | 31,272 | 1,892 | 16.5× | 58,468 | 660 | 88.6× |
| 3D-ResNet | 5,158 | 1,927 | 2.7× | 7,594 | 2,296 | 3.3× |
| DCGAN | 3,003 | 298 | 10.1× | 4,914 | 420 | 11.7× |
| BERT | 6,220 | 496 | 12.5× | 12,007 | 266 | 45.1× |

通常在单机上为一个 DNN 生成充分优化程序仍需数小时；作者认为 inference 部署前一次性成本可接受。

### Cost model 本身

用 ResNet-50/Intel CPU tuning 中的 25,000 programs，20,000 train / 5,000 test：

- RMSE：0.079（归一 throughput 空间）；
- \(R^2\)：0.958；
- pairwise comparison accuracy：0.851；
- top-30 recall@30：0.624。

这些数字不能转译成“绝对时延误差 7.9%”。

## 论文真正解决了什么，没解决什么

### 解决

- 减少每个 operator/hardware 手写模板；
- 自动构造比模板更大的合法 schedule space；
- 避免在不完整程序上早剪枝；
- 把 learned ranking 与目标机 measurement 做闭环；
- 多 subgraph 联合调优时动态分配预算。

### 没解决

- 不预测任意新硬件上的绝对 kernel latency；换硬件仍要目标机 measurements 训练/适配 cost model。
- 不模拟多个 kernels、通信和服务调度的系统级 overlap。
- 不处理动态 shape；每个静态 shape 基本是新 task。
- 原论文不支持 sparse operators。
- 依赖 LLVM/NVCC 做 instruction selection；当时对 Intel VNNI、NVIDIA Tensor Core、ARM Dot 等特殊低精度指令利用不足。
- 不保证全局最优；规则定义的 search space 外仍不可达，搜索有随机性。

## 与相关工作的关系

| 工作 | 搜索空间 | 搜索/成本模型 | 相比 Ansor |
| --- | --- | --- | --- |
| Vendor libraries | 专家手写 kernels | 离线人工调优/heuristics | 常见算子强，但新算子/shape/fusion 覆盖慢 |
| AutoTVM | 手写 operator template 的参数网格 | XGBoost/SA 等模板内搜索 | Ansor 自动生成 sketch，空间更大、少写模板 |
| Halide auto-scheduler | sequential construction | 对 incomplete program beam search/early pruning | Ansor 只在 complete program 上评分和 fine-tune |
| FlexTensor | 更通用但仍是单算子模板 | 参数搜索 | Ansor 支持 multi-op fusion/更复杂空间 |
| ProTuner | Halide 上 MCTS | 缓解不完整程序估计 | 并行工作；重点和 workload 不同 |
| TenSet | Ansor 生成的 5200 万真实记录 | 预训练 MLP + ranking 等 | 用离线数据给 Ansor warm start，减少在线冷启动 |
| TLP | primitive sequence | attention + LambdaRank、MTL | 继续替换 Ansor 手工 program features 与 GBDT |

## 优点

- 系统设计完整：空间、搜索、cost model、measurement、跨 task scheduler 相互配合。
- 通过 complete-program sampling 解决明确的训练/推理对象错位。
- derivation rules 有程序语义，生成候选合法性高于盲目参数搜索。
- target measurement 持续回填，能适应具体硬件而非只信 offline model。
- 单 operator、subgraph、end-to-end 与 cost-model 多层实验齐全。

## 关键短板/不适用场景

- **动态 seqLen/batch**：原文明确不支持 dynamic shapes；需要 bucket/specialization 或新方法。
- **训练性能规划**：评估集中于 inference，不含 backward、optimizer、activation checkpoint、collective。
- **分布式并行**：不处理 DP/TP/PP/EP/rank/topology。
- **SLA/容量规划**：cost score 非绝对时间，无法直接输出 P95/P99 或吞吐。
- **MoE/sparsity/data-dependent routing**：原文 dense only，搜索空间需要重设计。
- **新硬件零样本**：模型设计可重用，但标签与高效 rule/annotation policy 仍需硬件适配。
- **编译成本**：充分调优仍要数小时；线上 shape 爆炸会造成 specialization/cache 管理问题。

## 映射到“输入 → L1/L2/L3 → 输出”

```text
输入：DNN + static shapes + target hardware
  ↓
L1：Ansor/Relay 的核心能力之一
    graph fusion/partition → subgraphs
    derivation rules → sketches → complete tensor programs
  ↓
L2：Ansor learned cost model + target measurements
    program features → GBDT normalized-throughput score
    → top candidates 实测、在线重训
  ↓
L3：只做 tuning-budget scheduler，不是运行时事件模拟器
    不模拟 kernel overlap、通信、服务排队
  ↓
输出：每个 subgraph 的优化实现与最终可执行模型
```

它给灰盒路线的启示不是“用 Ansor 做整机预测”，而是：L1 必须显式生成具体 shape 的实现候选；L2 的 learned model 应与实测闭环；排序型输出只用于选路，不能冒充绝对 runtime。

## 快速记忆 5 点

1. Ansor 的大招是 **sketch + annotation 的层次搜索空间**。
2. 它对完整程序评分，避免 incomplete-program early pruning 的错配。
3. fine-tuning 是 cost-model-guided evolutionary search + 目标机测量闭环。
4. 原始 cost model 是 GBDT + throughput-weighted squared error，不是 RankLoss。
5. 原文局限：static shape、dense op、特殊低精度指令不足；不做分布式/L3 模拟。

## 自测问题

1. Sketch 与 template 的本质区别是什么？
2. 为什么完整程序上训练的 cost model 不适合给半成品 program 打分？
3. 加权 MSE 怎样偏向快候选？它为何仍不是 RankLoss？
4. task scheduler 优化的是运行时 schedule，还是调优预算？
5. seqLen 从 512 变到 513 时，为什么可能需要新 task 或新 bucket？

## 术语表

- **compute definition**：operator/subgraph 的数学语义。
- **tensor program**：该语义的低层 loop/memory/thread 实现。
- **sketch**：只固定高层 schedule 结构的程序骨架。
- **annotation**：tile factors、parallel、vectorize、unroll 等低层具体选择。
- **rfactor**：把 reduction 拆分/重组以暴露更多并行性。
- **compute_at**：改变 producer 在 consumer loop nest 中的计算位置。
- **population / mutation / crossover**：遗传搜索中的候选群体、变异和重组。
- **GBDT**：梯度提升决策树。
- **normalized throughput**：同一 DAG 内将候选性能缩放到 `[0,1]`。
- **measurement trial**：编译并在目标硬件运行一个候选的昂贵评测单位。

## 证据索引

- 背景、模板与 sequential construction 问题：论文 §1–2。
- 总体架构：论文 §3 与 Figure 4。
- sketch rules、CPU/GPU tiling：论文 §4、Table 1、Figure 5。
- evolutionary operations 与 cost model：论文 §5；损失公式在 §5.2。
- task scheduler 目标与梯度近似：论文 §6。
- 三平台/多层评估、搜索时间和 cost-model metrics：论文 §7、Table 3、Figure 6–11。
- dynamic shape、sparse、special instructions 限制：论文 §9。

