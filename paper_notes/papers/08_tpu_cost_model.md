# A Learned Performance Model for Tensor Processing Units（MLSys 2021）

## 元信息与一手资料

- 论文：Samuel J. Kaufman 等，*A Learned Performance Model for Tensor Processing Units*，MLSys 2021。
- 一手资料：[MLSys 论文 PDF](https://proceedings.mlsys.org/paper_files/paper/2021/file/6bcfac823d40046dca25ef6d6d59cc3f-Paper.pdf)；[Google Research 论文页](https://research.google/pubs/a-learned-performance-model-for-tensor-processing-units/)。
- 论文对象：XLA 为 TPU 生成的 tensor program；重点研究两个编译决策——**tile-size selection** 和 **operator fusion**。
- 论文不是：端到端训练/推理服务模拟器，也不是 GPU/NPU 的通用零样本时延模型。

## 30 秒总结

一句话类比：**把一段 XLA kernel 看成一张“小计算图”，用 GNN 读懂图中算子与 tensor 流，再根据任务选择“只排候选名次”或“回归绝对 kernel 时间”。**

它解决的是编译器内部的决策成本：候选 tile 或 fusion 方案可能极多，全部编译并上 TPU 实测太贵。作者用 XLA 图、shape/layout、tile size 以及少量可选静态性能量作为输入，训练 GraphSAGE 等模型。tile 选择只关心同一个 kernel 内谁更快，使用 pairwise ranking；fusion 要把多个 kernel 时间相加成程序时间，因此使用绝对时延回归。

最值得记住的不是“GNN 一定胜过解析模型”，而是：

1. **目标函数要匹配编译决策**：tile 选路用排序，fusion 用绝对回归。
2. **随机同分布切分很好看，刻意 OOD 切分会反转结论**：随机 split 的 tile APE 为 3.7%，解析模型为 6.1%；manual split 下 learned 为 6.3%，反而差于解析模型的 2.3%。
3. 模型服务于“缩小上机测量集合”，不是替代所有实测。

## 论文要解决什么问题

### 问题背景

编译器接到一个数学等价的计算图后，仍有很多实现自由度：

- 要不要把 producer 与 consumer 融在一起；
- 一个大 tensor 每次搬多大的 tile 到片上 scratchpad；
- tile 的各维怎样切分；
- 后端如何排指令、分配寄存器、安排数据搬运与计算重叠。

候选之间数学结果相同，但运行时间可能差很多。传统办法是工程师手写解析/启发式模型；另一办法是把每个候选都编译后上硬件跑。前者维护贵且容易漏掉复杂效应，后者测量代价太高。论文希望训练一个可重定向的 learned cost model，给 autotuner 提供便宜的 reward/cost 信号。

论文列出的设计目标是：能处理复杂多层 loop nest；泛化到未见过的 program；尽量少依赖手工特征；同一框架可用于不同编译任务。

### 为什么难

即便 TPU 没有乱序执行和多 kernel 并发，性能仍由多种耦合因素决定：

- systolic array、vector unit、VLIW、寄存器和 scratchpad 的使用方式；
- tile 大小同时改变 HBM 搬运次数、单次搬运带宽、计算/拷贝重叠和寄存器压力；
- fusion 一方面省去中间 tensor 的 HBM 写回/读回，另一方面可能扩大 live range、增加资源压力；
- 真正指令排程与寄存器分配发生在更低层后端，而 cost model 做判断时未必已经看到最终机器码。

所以“FLOPs ÷ 峰值算力”或“bytes ÷ 峰值带宽”只能给粗边界，不能稳定区分相邻候选。

## 必要背景知识

### XLA、HLO 与 kernel

XLA 是面向机器学习计算的编译器。可以把编译过程粗略理解为：

```text
TensorFlow/JAX 计算图
    ↓ 高层优化、代数化简、layout/fusion 等
XLA 高层 tensor IR（通常称 HLO）
    ↓ 把一个或多个 primitive op 组织成 kernel
低层 IR、指令调度、寄存器分配
    ↓
TPU 可执行代码
```

- **primitive operation**：add、multiply、reduce、reshape、convolution 等图节点。
- **fusion**：把多个 primitive operation 合成一个 kernel。例如 `matmul → bias_add → relu` 可以融合，避免 `matmul` 结果先落 HBM 再读回。
- **kernel**：一次由设备执行的编译单元；一个 kernel 内部仍可能包含多节点的小图。

注意：论文使用的是 2020 年前后的 XLA/TPU 表示与工具链。今天 OpenXLA 的 IR 名称和工程实现已有演进，但“高层图 → fusion/kernel → 低层代码”的思路仍然成立。

### TPU、HBM、scratchpad 与 tiling

TPU 通过矩阵乘 systolic array 获得高吞吐，同时有容量较小但更快、由软件管理的片上 scratchpad。大 tensor 放不下时，要分块计算：

```text
大矩阵 / 大输出
  ├─ tile 0：搬进片上 → 计算 → 搬出
  ├─ tile 1：搬进片上 → 计算 → 搬出
  └─ ...
```

tile 太小，循环和搬运次数多，单次传输也难跑满带宽；tile 太大，可能放不下、增加寄存器压力，或让流水重叠变差。因此 tile size 是离散且有明显非线性的编译决策。

### 解析 cost model 与 learned cost model

XLA 原解析模型估算数据搬运时间和计算时间，再近似取两者最大值，类似局部 Roofline：

$$
T_{analytic}\approx \max(T_{memory},T_{compute}).
$$

但它还要用启发式近似双向传输、指令调度、寄存器使用、stall 等。learned model 则从大量真实测量标签中学习这些难写全的关系。

两者不是非黑即白：本文的 learned model也可输入 FLOPs、读写 bytes 等静态分析结果；这其实已是轻度灰盒。

## 输入、输出与关键假设

| 项目 | Tile-size 任务 | Fusion 任务 |
| --- | --- | --- |
| 候选单位 | 同一 kernel 的不同 tile size | 同一程序的不同 fusion 配置所产生的 kernels |
| 主要输入 | kernel 数据流图、op 属性、shape/layout、tile-size；可选静态性能量 | kernel 数据流图、op 属性、shape/layout；可选静态性能量 |
| 学习目标 | 同一 kernel 内候选相对排序 | 单个 kernel 的绝对 runtime |
| 聚合到程序 | 选每个 kernel 最优 tile | 将各 kernel 预测 runtime 求和 |
| 最终用途 | 直接替换启发式，或筛 top-10 再实测 | 指导 fusion autotuner，再对 promising 配置实测 |

关键系统假设：TPU 当时一次执行一个 kernel；kernel 间没有执行重叠，且假设没有显著跨 kernel cache 影响。因此：

$$
\hat T_{program}=\sum_{k\in K}\hat T_k.
$$

这正是它不能直接迁移成现代多 stream GPU、分布式训练或在线推理系统模型的原因：那些场景需要 L3 事件模拟处理并发、通信、排队和重叠。

## 方法拆解

### 1. 把 kernel 表示成带特征的数据流图

每个 kernel 的输入包含三部分：

1. **邻接矩阵**：谁的 tensor 输出流向谁。
2. **node features**：opcode、输出 tensor shape、layout、stride、padding、卷积 filter size 等。
3. **kernel features**：tile size；以及可选的 FLOPs、读取 bytes、写入 bytes、特殊功能单元指令数。

变长维度使用 padding/truncation 变成定长向量，并加入维度列表的 sum/product，以尽量保留规模信息。

### 2. GNN 生成节点表示

作者主要使用 GraphSAGE。对节点 `i` 的第 `k` 层表示，可直观写成：

$$
h_i^{(k)}=\operatorname{norm}\left(
W^{(k)}\left[h_i^{(k-1)}\,\Vert\,
\operatorname{AGG}_{j\in\mathcal N(i)}g^{(k)}(h_j^{(k-1)})
\right]\right).
$$

含义不是“神奇地理解代码”，而是让节点同时看到邻居。例如一个 op 的 node feature 只直接写了输出 shape，它可以从 producer 邻居接收输入 shape 信息。

### 3. 从节点表示聚合为 kernel 预测

论文比较了多种 reduction：

- 对节点 embedding 做 column-wise mean/max/sum；
- 对拓扑排序后的节点序列跑 LSTM；
- Transformer encoder；
- 每个节点各自产生 cost，再求和。

长程序列模型理论上能看到跨图依赖，但更复杂并不自动更好。实验中 GraphSAGE + LSTM 在 tile 任务最好，GraphSAGE + Transformer 在 fusion 任务最好；较简单的 GraphSAGE + column-wise reduction 也有不错的速度/精度权衡。

### 4. 为两类任务选择不同损失

#### Tile：pairwise rank loss

tile 任务只需回答“候选 A 是否优于 B”。论文对同一批样本的预测差 `\hat y_i-\hat y_j` 和真实次序做 pairwise loss，可用 hinge 或 logistic 形式：

$$
L_{rank}=\frac{1}{n(n-1)/2}\sum_{i,j}
\mathbf 1[y_i>y_j]\,\phi(\hat y_i-\hat y_j).
$$

因此输出是用于排序的 score，不要求具有“微秒”单位。消融中，将 rank loss 换成 MSE，Tile-Size APE 均值从约 6.8% 恶化到 17.7%。

#### Fusion：对数时延 MSE

fusion 需要将 kernel 预测相加，所以必须学习绝对时间。由于标签从纳秒到秒且右偏，作者对 runtime 做 log transform，再最小化 squared error：

$$
L_{fusion}=\left(\log \hat T_k-\log T_k\right)^2.
$$

这也说明“该论文只做排序”是不准确的：**tile 分支做排序，fusion 分支做绝对回归。**

## 一个通俗 worked example

假设 XLA 看到：

```text
X --matmul(W)--> M --add(bias)--> A --relu--> Y
```

编译器要做两类选择。

### 选择 A：是否 fusion

- 不融合：3 个 kernel；中间张量 `M`、`A` 可能写回 HBM 再读回。
- 全融合：1 个 kernel；中间结果留在片上，但 kernel 更复杂，可能增加寄存器/片上内存压力。

模型分别预测两种配置产生的各 kernel 时间并求和。它不是只看到“少两个 launch 就一定快”，而是用融合后的小图结构、shape/layout 和历史测量学习资源效应。

### 选择 B：matmul kernel 的 tile

候选为 `64×64`、`128×64`、`128×128`。模型无需声称三者分别是 8.1、7.3、7.6 µs；只要稳定把真实最快的 `128×64` 排在前面即可。autotuner 可以取预测 top-10 上 TPU 真测，最终 winner 仍由硬件决定。

对后训练/RL 的类比：这更像给候选 action 学一个 value/ranking model，用它减少昂贵 environment rollout；不是拿 value score 直接当真实 reward 的物理单位。

## 数据与实验：原文到底证明了什么

### 数据规模

- 104 个生产或研究 XLA programs。
- kernel 平均 41 个节点，范围 1–1000。
- Tile 数据约 2500 万样本：每个 kernel 的候选最多 50 万，使用 50 台带 accelerator 的 host、每 kernel 最长采集 30 分钟；标签取 3 次运行的最小值。
- Fusion 数据去重后约 2.08 亿 kernel 样本：每程序随机探索最多 5 万 fusion 配置，或使用 50 台机器跑满 4 小时。
- 测量覆盖 TPU v2、v3；主要实验默认使用 v2，论文另报 v3 结果。

### 关键结果

| 任务/切分 | Learned | 解析 baseline | 应怎样解释 |
| --- | ---: | ---: | --- |
| Tile，随机 program split，mean Tile-Size APE | 3.7% | 6.1% | 同分布附近 learned 更好 |
| Tile，manual dissimilar split，mean APE | 6.3% | 2.3% | 更强 OOD 下解析模型反而更好 |
| Fusion，随机 split，kernel ≥5µs，MAPE | 4.5% | 31.1% | learned 填补了解析模型薄弱的 fusion 绝对成本 |
| Fusion，kernel <5µs，MAPE | 5.0% | 22.7% | 趋势相同，但小 kernel 对总时间贡献较少 |

对应的随机 split Kendall τ：tile learned 0.80、解析 0.74；fusion learned 0.92、解析 0.80。TPU v3 上 tile mean APE 3.8%，fusion（≥5µs）MAPE 4.9%。

### 工具链集成结果

- tile autotuner 用模型选 top-10 再上硬件，和解析模型 top-10 在测试中大致相当，通常相差 1%–3%。
- fusion autotuner 先在 CPU 上用 cost model 搜 1 小时，再用 TPU 验证 1 或 10 分钟；与只在 TPU 上搜 10 分钟相比，cost model + hardware 找到的配置平均快 1.5%。
- 它把真实目标硬件评测时间从 10 分钟降到 1 分钟而未观察到性能退化；从随机初始配置出发时，使用模型平均找到快约 10% 的配置。

### 必须避免的误读

1. **没有证明 TPU v2 → v3 零样本迁移。** 原文说在 v2、v3 上评估并获得相似结果，但没有给出“只用 v2 标签训练，直接预测 v3”的跨代 zero-shot protocol。
2. 3.7% 是 tile 选择后的程序级 regret-like APE，不是逐 kernel 绝对时延 MAPE。
3. “96.3%/95.5% accuracy”是作者摘要对两任务结果的概括；工程比较应回到具体的 Tile-Size APE、MAPE、Kendall τ 和 autotuning 结果。
4. manual split 的反转说明：训练语料覆盖度仍是 learned cost model 的硬约束。

## 与相关工作的关系

| 工作 | 表示/机理 | 与本文差异 |
| --- | --- | --- |
| XLA 手写解析 tile model | 搬运、计算、流水与启发式 | 物理可解释、OOD 可能更稳；开发维护成本高，fusion 绝对成本能力弱 |
| Ithemal | x86 basic block 指令序列 | 对象是短、无 loop 的 basic block；本文处理最多上千 op、隐含多层 loop 的 tensor kernel |
| Halide learned model | 静态分析产生的专家特征 | 本文更直接读 XLA 图，减少专门 feature engineering |
| AutoTVM | 单个/少量 kernel 的手工调度模板和 learned ranking | 本文训练跨众多 XLA programs 的图模型，并覆盖 program-level fusion |
| Ansor / TenSet | TVM tensor program 搜索；程序特征、GBDT/MLP 与 ranking | 面向 TVM 调度候选；本文绑定 XLA/TPU 图与 tile/fusion 决策 |
| NeuSight | tile/wave + Roofline 约束的跨 GPU component 预测 | NeuSight 更明确加入物理上界/下界与硬件描述；本文更偏编译器内候选学习 |

## 优点

- 明确展示“同一模型骨架、不同任务目标”：排序与绝对回归不混为一谈。
- 图表示天然支持不同 kernel 大小和拓扑，且能直接利用编译器已有 IR。
- 大规模真实生产/研究语料，包含工具链集成而非只报离线 test error。
- 学习模型与真机 top-k 验证结合，实际减少稀缺 TPU 占用。
- manual OOD split 没被隐藏，给出了 learned model 的真实边界。

## 短板与不适用场景

- **生态绑定**：特征、opcode、layout 与标注语料来自 XLA/TPU；迁移到 CUDA、Ascend 或其他编译器需要重做 lowering 适配与标签库。
- **目标硬件标签昂贵**：论文数据采集使用大量 TPU hosts；“少手工特征”不等于“少数据”。
- **OOD 仍弱**：manual split 已显示 learned model 可能输给解析模型。
- **串行 kernel 假设**：不处理 GPU 多 stream、通信并发、服务排队、跨 kernel cache 等系统效应。
- **不是动态 shape 模型**：不同 shape 会改变图特征和候选空间；训练分布未覆盖时不能安全外推。
- **不是分布式性能模型**：没有 DP/TP/PP/EP、collective、rank 拓扑或跨机带宽。
- **fusion 预测依赖固定编译栈**：后端 codegen、运行时、硬件 stepping 变化会造成 concept drift，需要重新测量或校准。

## 映射到“输入 → L1/L2/L3 → 输出”

```text
输入：XLA program + 候选 tile/fusion configuration + 目标 TPU
  ↓
L1：XLA 已产生/变换计算图与候选 kernel 边界
  ↓
L2：本文核心
    kernel 图特征 → GNN/sequence reduction
    tile：相对 score；fusion：绝对 kernel runtime
  ↓
L3：非常简化
    TPU 串行 kernel，直接 Σ kernel runtime；不做通用事件模拟
  ↓
输出：tile/fusion 候选排序、估计成本、供 autotuner 选 top-k 上机
```

它对拟议灰盒架构的启示是：保留 shape/layout/FLOPs/bytes 等可解析量，让 ML 学习编译器后端难写清的残差或利用率；但在系统级必须接另一个 L3 模拟器，不能拿 kernel 求和替代通信/调度模型。

## 快速记忆 5 点

1. 对象是 **XLA/TPU 编译候选**，不是通用端到端 DNN 预测。
2. kernel 用数据流图表示，GraphSAGE 聚合局部 tensor 流信息。
3. tile 选路用 pairwise ranking；fusion 用 log-runtime MSE。
4. 随机 split：3.7% vs 6.1%；manual OOD：6.3% vs 2.3%，结论反转。
5. v2/v3 都有实验，但没有 v2→v3 zero-shot 证据。

## 自测问题

1. 为什么 tile-size 任务不需要输出微秒，而 fusion 任务需要？
2. fusion 为什么既可能加速，也可能因资源压力而变慢？
3. manual split 为什么比 random split 更接近“新模型外推”？
4. 为什么在 TPU 上求和 kernel 时间尚可，在多 stream GPU 服务上却危险？
5. 若迁移到 Ascend NPU，哪些部分可以复用，哪些数据必须重建？

## 术语表

- **HLO**：XLA 的高层 tensor IR 家族；表达 tensor op 与依赖。
- **kernel**：一次设备执行的编译单元，内部可含多个 fused primitive ops。
- **fusion**：把多个 op 合成一个 kernel，通常减少中间内存流量和 launch。
- **tile**：把大 tensor/loop nest 切出的计算块。
- **scratchpad**：由软件显式管理的快速片上存储。
- **systolic array**：以规则数据流执行矩阵乘累加的硬件阵列。
- **GraphSAGE**：通过邻域聚合学习节点表示的 GNN。
- **Kendall’s τ**：衡量两组排序一致性的统计量。
- **APE/MAPE**：绝对百分比误差/其均值；要确认统计单位和聚合方式。
- **autotuner**：自动生成、评分并测量编译候选的搜索系统。

## 证据索引

- 目标硬件、tile/fusion 定义与串行 kernel 假设：论文 §2。
- 图输入、GraphSAGE、reduction 与损失：论文 §3。
- 104 programs、2500 万 tile 样本、2.08 亿 fusion kernel 样本：论文 §4。
- 3.7%/6.1%、6.3%/2.3%、4.5%/31.1%：论文 Table 2 与 §5。
- GNN/损失消融：论文 Table 3、Table 4 与 §6。
- compiler/autotuner 集成：论文 §7。
- OOD 限制的原文措辞：论文 §9 Conclusion。
