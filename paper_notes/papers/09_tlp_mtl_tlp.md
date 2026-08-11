# TLP / MTL-TLP：把调度原语当“张量语言”（ASPLOS 2023）

## 元信息与一手资料

- 论文：Yi Zhai 等，*TLP: A Deep Learning-based Cost Model for Tensor Program Tuning*，ASPLOS 2023。
- 一手资料：[arXiv PDF](https://arxiv.org/pdf/2211.03578)；[ACM DOI](https://doi.org/10.1145/3575693.3575737)；[作者开源实现](https://github.com/zhaiyi000/tlp)（仓库已归档，README 指向后续 TLM）。
- TLP：Tensor Language Processing，直接编码 schedule primitive 序列。
- MTL-TLP：在 TLP 的共享 backbone 上，为不同硬件设置不同 task head，以目标硬件少量标签配合其他硬件标签训练。

## 30 秒总结

一句话类比：**一个 tensor program 像一份已经做完的菜，TLP 不再拆成分子分析成品，而是读“切块、换序、融合、并行、展开”的烹饪步骤。**

Ansor/TenSet 的传统 cost model 先生成低层 loop program，再从程序中抽大量计算、访存和并行特征。TLP 观察到：程序是由一串 schedule primitives 变换出来的，于是直接把 primitive type、数字参数、名称参数编码成类似“词序列”的特征，用 self-attention/LSTM 预测候选 score。这样既省去部分生成/特征提取时间，也减少 CPU/GPU 专用 feature engineering。

MTL-TLP 再把“不同硬件”当多任务：共享层学习共同的 schedule 语义，每个 head 学各自硬件的偏好。它减轻跨硬件数据需求，但没有消灭目标机测量：论文的“7% 数据”约为目标硬件 50 万条有标签测量，作者明确说采集仍需几十小时。

## 先建立直觉：什么是 tensor program tuning

### 数学定义与程序实现是两件事

一个 matmul 的数学定义是：

\[
C_{ij}=\sum_k A_{ik}B_{kj}.
\]

但程序可按很多方式执行：

- `i,j,k` 循环先后顺序不同；
- 把 `i` 拆成 `i_outer × i_inner`；
- 外层分给 CPU threads 或 GPU blocks；
- 内层向量化、unroll；
- 把 bias/ReLU inline 或 fusion；
- 数据放寄存器、shared memory、cache 的方式不同。

这些程序数学等价，性能却不同。**Tensor program** 指某个确定的低层实现；**tuning** 指在大量等价实现里寻找目标硬件上最快的一个。

### Schedule primitive 是什么

Schedule primitive 是对 loop/tensor program 的一个语义明确的变换动作，例如：

- `split(i, factors=[...])`：把循环 `i` 分层；
- `reorder(i0, j0, k0, i1, ...)`：换循环顺序；
- `fuse(i0, j0)`：合并循环或 stage；
- `compute_at(A, loop=j0)`：改变中间结果的计算位置；
- `parallel(i0)`、`vectorize(i1)`、`unroll(k1)`：映射硬件并行与指令级展开。

在现代编译栈里，它类似一串可重放的 transformation trace：

```text
naive loop program
  --split--> tiled loops
  --reorder--> cache-friendly order
  --fuse/compute_at--> changed producer-consumer locality
  --parallel/vectorize/unroll--> hardware mapping
  = final tensor program
```

### Cost model 在搜索中做什么

搜索空间在 CPU 上可达百万量级、GPU 上可达十亿量级。每个候选都经历编译、远程加载、执行、重复计时，代价高。cost model 先给数万候选打分，只把 top-k 编译/上机测量。

对后训练/RL 的类比：

- schedule primitive 序列 ≈ action sequence；
- 最终 tensor program ≈ rollout 后到达的 state/trajectory；
- 目标机 runtime ≈ 真 reward；
- cost model ≈ learned value/ranker；
- genetic search ≈ 用 value model 加速 proposal selection；
- 上机测量 ≈ 稀缺但不可省掉的 environment interaction。

## 论文要解决的两个问题

### 1. 传统特征工程重

论文称 Ansor 为最内层 assignment 从计算、访存、算术强度等五方面提取 164 个特征；TIRAMISU cost model 提取 2534 个特征并利用 AST。这样的特征：

- 需要编译/体系结构专家；
- CPU 与 GPU 往往要分开设计；
- 从最终 loop program 抽取，可能要先完成低层程序生成；
- 程序是变长、嵌套树，批处理困难。

### 2. Offline cost model 跨硬件失效

相同 schedule 在不同 CPU/GPU 上延迟不同；cache、SIMD、core count、memory hierarchy 和执行模型会改变相对名次。在某硬件采集的海量标签不能直接等价为另一硬件标签。论文称之为 **cross-hardware unavailability**。

## 输入、输出与关键假设

| 项目 | TLP | MTL-TLP |
| --- | --- | --- |
| 输入 | 一个候选的 schedule primitive sequence | 同样的 primitive sequence |
| 标签 | 同一 subgraph 内归一化性能 `min_latency / latency` | 每个硬件一个标签；缺失硬件标签可为 `None` |
| 模型 | 线性升维 + self-attention/LSTM + residual blocks + head | 共享 backbone + 每硬件独立 head |
| 输出 | 候选 performance score，越高越好 | 对指定硬件的 score |
| 主要用途 | 同一 task 内 top-k 排序 | 用少量目标硬件标签做跨硬件辅助学习 |

这里的 label 是：

\[
y(P\mid G,H)=\frac{T_{min}(G,H)}{T(P,G,H)}\in(0,1],
\]

其中 \(G\) 是 subgraph/task，\(H\) 是硬件，\(P\) 是候选 program。最快候选为 1，其余小于 1。

这不是原始毫秒标签。因此 TLP 虽把任务称为 regression，最佳配置实际使用 LambdaRank，输出不应直接当 SLA 所需的绝对时延。

关键假设：

- autotuner 的 schedule primitive 基本完整地描述了如何从 subgraph 得到程序；
- 不同程序拥有相同 primitive 序列的碰撞率很低；
- 对一个候选的优劣，primitive 序列中已包含足够多的 shape/loop 参数；
- 训练与推理使用兼容的 primitive vocabulary/语义。

## 方法拆解

### 1. 把 primitive 拆成三类元素

论文定义：

\[
S ::= p^*,\qquad
p ::= \tau\,(id\mid num)^*,
\]

- \(\tau\)：primitive type，如 split/reorder/fuse；
- \(num\)：切分因子、loop extent 等数字参数；
- \(id\)：loop/stage 名称等字符参数。

编码函数：

\[
F(p)=F_1(\tau)\,\Vert\,F_2(id)\,\Vert\,F_3(num),
\]

其中 type 变 one-hot，name 变 token，数字保留数值，最后按原始顺序拼接，再 crop、pad、normalize。

它不是把整行 schedule 字符串丢给一个大语言模型。作者刻意保留 primitive 类型与参数结构，以免相同 `split` 但参数略不同的动作被编码成两个毫无关系的 token。

### 2. 为什么可以把 primitive sequence 当作 program 的代理

论文在 TenSet CPU 数据上统计：865 万 tensor programs 中有 856 万个不同 primitive sequences，重复率约 1.043%。即使把特征裁到 `25×22`，重复率约 1.4034%。作者据此把 primitive sequence 近似看成 program 的唯一描述。

这是一个**经验论证，不是普适定理**：

- 同一 primitive trace 在不同 compiler version/codegen 下仍可能生成不同机器码；
- primitive 没有显式覆盖的硬件后端细节可能影响 runtime；
- 新 DSL 或自定义 schedule primitive 需要重建 vocabulary 与预处理器。

### 3. 模型结构

TLP 先用多层 linear 把每个 primitive embedding 升到 256 维以上，然后：

1. self-attention 或 LSTM 捕获动作间上下文；
2. 两个 residual blocks；
3. 多个 linear layers；
4. 对序列维求和得到一个 score。

消融选择为：一层 self-attention、8 heads、升维到 256、两个 residual blocks。论文没有使用 BERT 级的大模型，因为输入只有约 `25×22`，大模型既易过拟合又拖慢编译搜索。

### 4. 损失：MSE 或 LambdaRank

论文比较 normalized score 上的 MSE 与 LambdaRank。最佳组合是 self-attention + ranking loss：

| Backbone + loss | Top-1 score | Top-5 score |
| --- | ---: | ---: |
| Attention + Rank | 0.9194 | 0.9710 |
| Attention + MSE | 0.9128 | 0.9542 |
| LSTM + Rank | 0.9119 | 0.9509 |
| LSTM + MSE | 0.9061 | 0.9540 |

LambdaRank 的核心不是准确回归每个 score，而是对“交换两个候选名次会损失多少最终排序质量”赋权，从而把学习容量集中到 top candidates。

### 5. MTL-TLP：共享 backbone，多硬件 head

对 \(n\) 个硬件任务，输入相同，但标签是：

\[
(x,[y_1,y_2,\ldots,y_n]).
\]

若某条 program 只在硬件 2 测过，硬件 1 标签为 `None`，计算 loss 时跳过缺失项：

\[
L=\sum_{i:y_i\ne None}L_i(\hat y_i,y_i).
\]

- 共享 backbone：学习 split/reorder/fuse 等相对通用的 schedule 语义。
- 专用 head：学习特定硬件的 cache、SIMD、core、memory 等偏好。

论文不讨论 CPU 与 GPU 之间的 MTL，因为两者 tensor program 与执行模型差异过大；它主要在 CPU 之间、GPU 之间迁移。

## 一个 worked example

设有 fused dense + ReLU：

\[
Y=\max(XW+b,0).
\]

两个候选：

```text
P1: split(i, 8) → split(j, 16) → reorder(i0,j0,k,i1,j1)
    → vectorize(j1) → parallel(i0)

P2: split(i, 32) → split(j, 4) → reorder(i0,k,j0,i1,j1)
    → unroll(k) → parallel(i0)
```

传统 Ansor cost model 会先把 P1/P2 变成完整 loop IR，再抽“每层访存字节、reuse、向量化、并行度”等特征。TLP 直接编码上述动作和参数序列。

在 8-core Intel CPU 上，P1 的 vector width/parallel granularity 可能更合适；换到另一个 AVX-512 CPU，P2 或新的切分可能更快。MTL-TLP 让两个硬件共享“split + reorder 会改变 locality”这类表示，但各自 head 学不同排名。

注意：若 seqLen、batch 或 hidden size 改变，schedule 参数与最佳切分也会改变。没有新 shape 对应的 primitive candidates 和目标机标签，TLP 不能凭模型名称自动给出可靠时延。

## 实验设置与原文结果

### 数据

- 主要使用 TenSet：120 个网络配置、2308 个 subgraphs、6 个原始硬件平台，每硬件约 859 万 program/latency 记录。
- 作者另在 Intel i7-10510U 上耗时 50 多天采集约 865 万条 TenSet-TLP 数据。
- dataset-based test 持出 ResNet-50、MobileNet-V2、ResNeXt-50、BERT-tiny、BERT-base；batch=1，图像 224 或 sequence length=128。
- 端到端搜索平台：Intel i7-10510U CPU 与 NVIDIA Tesla T4 GPU；目标工作负载仍为上述 5 个模型。

### TLP 与 TenSet MLP 的离线 top-k

在五个 CPU 与两个 GPU 数据集上，TLP 的 CPU top-k 普遍高于 TenSet MLP；GPU 结果各有胜负。例如：

- Intel Platinum 8272：TLP top-1/top-5 为 0.9194/0.9710，TenSet MLP 为 0.8748/0.9527。
- Tesla K80：TLP top-1 略低（0.9059 vs 0.9083），top-5 更高（0.9741 vs 0.9629）。
- Tesla T4：TLP top-1 略高（0.8847 vs 0.8757），top-5 更低（0.9250 vs 0.9528）。

这比一句“全面精度更好”更准确。

### MTL 的数据效率

- 目标 Intel E5-2673 只用 500K：single-task top-1 0.6647；加 Platinum-8272 全量辅助任务后为 0.8741；再加 EPYC 为 0.8901；加到 4 个 CPU task 反而降到 0.8753。
- 目标 Tesla T4 只用 500K：top-1 0.7971；加 K80 全量辅助任务后为 0.8876。
- 同 ISA 的辅助硬件效果更好；任务太多会出现 negative transfer。
- 作者建议 2–3 个任务，辅助任务最好来自相同指令集架构，目标硬件至少 500K 标签。

### 端到端搜索结果：倍率到底是什么

Ansor 一轮先用 cost model/遗传算法筛候选，再测 10 个；论文跑 200 rounds，共 2000 次目标机测量。

- TLP 自身特征提取/预测使相同 2000 次 tuning 的执行过程，比 TenSet MLP 平均快 1.7×（CPU）、1.8×（GPU）。
- 达到 **TenSet MLP 调优 2000 次后得到的同等整网运行性能** 时，TLP 所需搜索时间平均少 9.1×（CPU）、3.0×（GPU）。
- MTL-TLP-500K 达到同一整网运行性能时，搜索时间平均少 4.7×（CPU）、2.9×（GPU）。
- 相对 **Ansor 调优 2000 次后得到的整网运行性能**，TLP 报 16.7×/16.0×，MTL-TLP 报 10.0×/15.8×（CPU/GPU）。

这些都是“达到同等搜索结果所需时间”的加速，不是：

- 模型绝对时延 MAPE；
- 最终 kernel 固定快 9.1×；
- 跨硬件零样本预测快 4.7×。

“仅 7% 目标数据”是约 500K 条目标硬件有标签 program measurements；原文限制部分明确说，采集仍需几十小时。

### 运行开销

TLP 把部分 CPU 特征生成工作移到 GPU inference：batch size 2048 时，GPU memory 从 TenSet MLP 的 882 MB 增到 1634 MB；五轮遗传算法总时间约从 20 秒降到 6 秒。

## 与相关工作的关系

| 工作 | 输入表示 | 学习/搜索重点 | TLP 的变化 |
| --- | --- | --- | --- |
| AutoTVM | 手工模板参数 + 手工特征 | 模板内调参 | TLP 面向自动搜索框架生成的 primitive trace |
| Ansor | 完整 program 的算术/访存等特征；GBDT | 在线测量闭环 + evolutionary search | 保留 Ansor 搜索框架，替换/增强 cost model 特征与网络 |
| TenSet MLP | Ansor program 特征 + 高层图特征；大离线数据 | MLP + ranking/MSE、预训练、local residual transfer | 直接读 primitive，避免先生成部分低层 program；MTL 替代单 local residual 思路 |
| TIRAMISU cost model | AST + 大量手工特征 | 递归/序列建模 program tree | TLP 用规则 sequence，批处理更容易 |
| TPU learned cost model | XLA kernel graph + GNN | XLA tile/fusion | TPU 模型读编译 IR 图；TLP 读 schedule action sequence |
| NeuSight | tile/wave + Roofline + utilization | 跨 GPU component 绝对时延 | TLP 主要优化候选排序，不提供物理边界或 SLA 绝对时间 |

## 优点

- 把 feature extraction 前移到 schedule trace，工程接口清晰、推理更快。
- primitive type/参数结构保留语义，比原始字符串 token 化更有 inductive bias。
- 同一编码机制适配 CPU/GPU primitive，减少手写两套硬件特征。
- MTL 明确分离共享表示与硬件专用 head，是一种可落地迁移机制。
- 同时报告离线 top-k 和端到端 search time，避免只看 MSE。

## 短板与不适用场景

- **不是绝对时延预测器**：最佳配置用归一化 score + LambdaRank，不能直接输出 SLA 所需毫秒。
- **仍需目标域标签**：7% 约 50 万测量，采集几十小时；不是 zero-shot。
- **表示绑定 schedule DSL**：primitive 种类、参数语义、compiler version 改变后需适配/重训。
- **碰撞率结论只对所测 TenSet/Ansor 成立**：不能当成 program 与 primitive trace 永远一一对应的定理。
- **跨架构迁移有限**：论文不做 CPU↔GPU MTL；相同 ISA 的辅助 task 才更有效。
- **静态 dense 工作负载为主**：动态 shape、sparse/MoE routing、数据依赖分支不在其证据范围。
- **不做系统级模拟**：没有通信、rank、DP/TP/PP/EP、排队或 compute-communication overlap。
- **离线数据有选择偏差**：训练标签来自特定 Ansor 搜索空间和采样策略，不覆盖的 schedule 不能被模型凭空发现。

## 映射到“输入 → L1/L2/L3 → 输出”

```text
输入：subgraph + shape + target hardware
  ↓
L1：Ansor 构造搜索空间，产生多条 schedule primitive sequences
  ↓
L2：本文核心
    primitive sequence → TLP/MTL-TLP → hardware-specific candidate score
    → top-k 上目标机实测并回填
  ↓
L3：不覆盖；没有分布式/服务事件模拟
  ↓
输出：更快找到高性能 tensor program
```

在灰盒方案中，TLP 最适合作为 L2 的“candidate ranker”或数据稀缺时的 warm start，不宜担当最终绝对时延接口。容量规划仍需真实 runtime/cache/机制模型，并由 L3 模拟组合。

## 快速记忆 5 点

1. TLP 读的是 **schedule primitive 序列**，不是最终源代码文本。
2. primitive 拆成 type one-hot、name token、numeric parameter。
3. 最佳模型是轻量 self-attention + LambdaRank；标签是 task 内归一化 score。
4. MTL-TLP 是共享 backbone + 每硬件 head，不是硬件 ID 拼接后零样本外推。
5. 9.1×/3.0× 和 4.7×/2.9× 都是达到同等调优结果的**搜索时间**加速；7% 仍是 500K 目标机标签。

## 自测问题

1. 为什么直接读 primitive sequence 可以省时间？
2. primitive sequence 与最终机器码为什么只是近似一一对应？
3. normalized score 能否直接转换成微秒？为什么？
4. MTL 为什么在同 ISA 硬件间更有效？任务太多为什么可能变差？
5. 若把 TLP 接入 LLM 推理容量规划，还缺哪两层能力？

## 术语表

- **tensor program**：某个 tensor subgraph 的确定低层循环/并行实现。
- **schedule primitive**：split、reorder、fuse、compute_at 等程序变换原语。
- **TIR**：低层 tensor IR，显式表达 loop、buffer、thread binding 等。
- **cost model**：用便宜预测替代大部分候选上机测量的评分器。
- **LambdaRank**：把排序指标变化转为梯度权重的 learning-to-rank 方法。
- **top-k score**：预测前 k 个候选中实际最好候选与全局最好候选的性能比型指标。
- **multi-task learning**：相关任务共享一部分参数，同时保留任务专用输出层。
- **domain gap**：不同硬件上的标签函数不同导致的分布/条件关系变化。
- **negative transfer**：辅助任务干扰目标任务，使效果变差。

## 证据索引

- tuning 测量成本、传统特征数量、两项贡献：论文 §1。
- search compiler、TenSet 与 primitive 背景：论文 §2。
- TLP 训练/推理管线：论文 §3。
- primitive grammar、碰撞率、模型与 label：论文 §4。
- MTL 结构与 loss：论文 §5。
- top-k、模型消融、跨硬件数据效率和端到端 search time：论文 §6。
- 500K 仍需几十小时：论文 §8 Limitation。
- 开源状态与实现来源：[作者仓库](https://github.com/zhaiyi000/tlp)。
