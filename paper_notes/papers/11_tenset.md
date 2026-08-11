# TenSet：面向 Learned Tensor Compiler 的大规模真机数据集（NeurIPS 2021）

## 元信息与一手资料

- 论文：Lianmin Zheng 等，*TenSet: A Large-scale Program Performance Dataset for Learned Tensor Compilers*，NeurIPS 2021 Datasets and Benchmarks Track。
- 一手资料：[论文页](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/a684eceee76fc522773286a895bc8436-Abstract-round1.html)；[论文 PDF](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/a684eceee76fc522773286a895bc8436-Paper-round1.pdf)；[官方数据/代码仓库](https://github.com/tlc-pack/tenset)。
- 核心贡献：发布跨 6 种硬件的约 5200 万条 tensor program **真实测量记录**，并系统比较 cost-model 架构、loss、metrics、数据采样、预训练与迁移。
- 与 Ansor 的关系：TenSet 用 Ansor 生成搜索空间/programs，并把预训练 cost model 接回 Ansor；它不是 Ansor 原论文 cost model 的别名。

## 30 秒总结

一句话类比：**Ansor 像一个边做题边积累错题的学生，TenSet 则提前整理了一套跨 CPU/GPU 的超大题库，让学生不必每到一台新机器都从随机猜测开始。**

此前 tensor compiler 的 learned cost model 各用自己的小数据、特征和指标，难以复现，也难以做大规模 offline pretraining。TenSet 从 120 个模型/输入配置中切出 2308 个 subgraphs，在 6 个硬件平台上形成 13,848 个 tasks；每 task 用 Ansor 随机采样最多 4000 个低层 programs，真实编译、warm-up、重复执行并保存 runtime 或 error。

论文最关键的研究结论有三条：

1. **这是有真实 runtime 的数据集，不是“只有排序标签”。** 排序模型只是其中一种用法。
2. 对 autotuning 来说，RMSE/MAPE 未必对应最终搜索质量；作者提出/采用更直接的 top-k score，并发现 MLP + LambdaRank 总体最好。
3. 预训练模型让 Ansor 达到相同 search quality 的时间最多缩短 10×；但跨硬件直接迁移仍会退化，在线 local residual calibration 能改善结果。

## 必要背景：一条 TenSet record 表示什么

### 四层对象

```text
Network：例如 ResNet-50，固定 batch/image/seq shape
  ↓ graph partition
Subgraph：例如 conv2d + bias + ReLU
  × Hardware platform：例如 Intel 8272 / Tesla T4
  = Task
  ↓ schedule search space
Program candidates：不同 tile/reorder/vectorize/parallel/fusion 实现
  ↓ compile + execute repeatedly
Measurement records：program + runtime samples / error code
```

论文术语：

- **Network**：带具体输入 shape 的完整计算图；`B=1, S=128` 与 `B=16, S=128` 是不同配置。
- **Subgraph**：编译器 graph partition 后的最小编译/搜索单元。
- **Task**：`subgraph × hardware platform`。同一 subgraph 在两台硬件上是两个 tasks。
- **Search space**：该 task 的所有合法低层实现，CPU 通常百万级，GPU 可达十亿级。
- **Program**：一个具体低层实现。
- **Measurement record**：task、program 与多次实际 execution time，或编译/运行错误。

### 为什么真时延也常被转成排序学习

每条 record 有秒/毫秒 runtime，但编译器的决策通常是：在同一个 task 内，把最快候选放进 top-k。于是可以：

- 用 MSE 回归 normalized throughput/latency；
- 用 pairwise/listwise ranking 直接学相对次序；
- 无论模型怎样训练，都用真实 runtime 计算 top-k 搜索质量。

因此应区分：

1. **数据标签有绝对时间**；
2. **ranking model 的输出 score 未必有时间单位**；
3. **最终 top-k 仍可在设备上测绝对时间**。

这和 RLHF 很像：preference model 的 score 不是人类效用的可校准绝对单位，但训练集仍可包含明确的 pairwise outcomes；TenSet 更进一步保留了产生次序的原始 runtime。

## 论文要解决的问题

### 1. Offline pretraining 缺数据

Ansor 原始流程从随机探索开始，在目标机在线测量并训练 GBDT。一个 DNN 可能要数小时。若能从历史大数据预训练 cost model，新 task 搜索就能有较好的初始排序。

### 2. 缺统一 benchmark

早期研究各自使用不同：

- compiler/search space；
- feature level；
- hardware；
- train/test split；
- RMSE、MAPE、pairwise accuracy 或最终 latency。

即使两个模型 RMSE 接近，也不知道谁能更快找到好 program。TenSet 希望提供公开、多平台、足够大的标准数据。

### 3. 新硬件与新 task 的迁移

新硬件没有完整大数据时，能否复用旧硬件模型？新网络/shape 是否能利用其他 tasks 的知识？论文用多平台数据和 online local model 做初步探索。

## 数据是怎样构造的

### 网络与 shape

模型来自 PyTorch vision model zoo 与 Hugging Face transformer model zoo，覆盖 CV/NLP。作者改变 batch size、image size/sequence length，形成 120 个 network configurations。数据偏向小 batch，因为论文目标是 trained-model inference 的 tensor compiler tuning。

### Graph partition

每个 network 被切成 unique subgraphs。典型 subgraph 含一个 heavy op：

- conv2d/conv3d/transposed conv/depthwise conv；
- matmul；
- softmax；

并融合 lightweight elementwise ops，例如 bias add、ReLU。该 partition algorithm 本身会决定数据里“有哪些 task”，因此也是数据集边界。

### Program sampling 与测量

对每个 subgraph/hardware task：

1. Ansor 构造 search space；
2. 随机抽最多 4000 个 programs；
3. 在 AWS/Azure 真实实例上编译；
4. warm-up；
5. 多次执行并保存全部 time costs；
6. 失败则保存 error code。

官方仓库示例里，一条 program record 包含 8 次左右的秒级计时数组，以及完整 schedule/loop program。这对研究噪声、聚合方式和错误候选很重要。

### 数据规模与平台

| 平台 | 实例/特征 |
| --- | --- |
| Intel Platinum 8272CL，16 cores | Azure D32s_v4，AVX-512 |
| Intel E5-2673 v4，8 cores | Azure F16s，AVX2 |
| AMD EPYC 7452，4 cores | Azure D16as_v4，AVX2 |
| ARM Graviton2，16 cores | AWS c6g.4xlarge，NEON |
| NVIDIA Tesla K80 | AWS p2.xlarge，Kepler |
| NVIDIA Tesla T4 | AWS g4dn.xlarge，Turing |

- Networks：120。
- Hardware platforms：6。
- Subgraphs：2308。
- Tasks：`2308 × 6 = 13,848`。
- Measurement records：官方仓库与 §3.4 为 51,577,248，摘要四舍五入为 52M。

论文 Table 1 打印为 51,532,994，与 §3.4/官方仓库相差约 4.4 万；工程引用宜写“约 5158 万/5200 万”，并注明版本统计差异，而不要制造伪精确。

## Cost model：输入、输出和训练目标

### 特征可以从哪一层抽

论文给出一个通用设计空间：

| 特征层 | 例子 | 优点 | 代价/风险 |
| --- | --- | --- | --- |
| 高层 task/graph | shape、access pattern、hardware cache/vector width | 提取快、跨 program 共享 | 看不到低层 codegen 细节 |
| Optimization/schedule | tile、loop transformations、parallel/vectorize | 直接表达候选差异 | 绑定 schedule DSL |
| Lowered IR/machine code | 访存、指令、实际低层结构 | 更接近硬件执行 | 要先 lowering/compile，特征慢且生态绑定 |

TenSet 实验沿用/扩展 Ansor 的 program features：对最内层 statements 抽计算与访存等特征，再用 padding/sum aggregation 形成定长输入；MLP 还加入约 10 个高层 graph features。论文也比较 XGBoost 与 LSTM。

### 回归与排序

回归型：

$$
L_{MSE}=\frac{1}{N}\sum_i(\hat y_i-y_i)^2,
$$

其中 `y` 常是 task 内 normalized throughput/latency。

排序型：论文使用源自 LambdaRank/LambdaLoss 的 probabilistic ranking cost，对交换候选名次影响较大的 pair 赋更大梯度。最终目标是让好候选进入 top-k，而不是让每个 score 都等于某个绝对时间。

### Top-k score

对单 task 的直观形式是：

$$
TopK(G)=\frac{T^*(G)}{\min_{P\in \widehat C_k(G)}T(P,G)}\in(0,1],
$$

其中 `T^*` 是数据集中最快 runtime，`\widehat C_k` 是模型预测的前 k 个候选。完整模型评估再按 subgraph 在 network 中出现次数加权聚合。

- 1 表示预测 top-k 中包含真实最佳候选；
- 0.9 可直观理解为选出的最好候选性能约为数据集中最优的 90%（具体聚合是按 latency/权重形成的比值）；
- 它比 RMSE 更贴近 compiler 只会实际测 top-k 的行为。

## 为什么 RMSE 很好，搜索仍可能差

论文训练三个示例模型。Model #1 的 RMSE/R² 看起来不错，但最终搜索得到 7.89 ms；ranking 模型 Model #3 的 RMSE 7.27、R² 为 -1818.41——回归指标看似灾难——最终却得到最好 6.39 ms。

原因：ranking score 可以做任意单调变换，甚至尺度完全不同；拿它和 normalized runtime 做 RMSE 本来就没有意义。真正重要的是：

- top candidates 的顺序；
- 在有限 measurement budget 下找到的最佳真实 runtime；
- 达到相同质量需要多少搜索时间。

这与 reward model 类似：只要 score ordering 能把优质 response 排前，它的绝对数值未必校准；但如果任务是 SLA，就必须另训/校准绝对值模型。

## 模型比较：不要把结论写得过满

测试集持出五个完整网络相关 tasks：ResNet-50、MobileNet-V2、ResNeXt-50、BERT-tiny、BERT-base；batch=1，image=224 或 seqLen=128。

| 模型 | ResNet-50 Top-1 | MobileNet-V2 | ResNeXt-50 | BERT-tiny | BERT-base |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP + ranking | 0.8823 | **0.7446** | **0.8584** | **0.8041** | **0.9143** |
| MLP + MSE | **0.8873** | 0.7026 | 0.6772 | 0.8001 | 0.8535 |
| XGBoost + MSE | 0.8535 | 0.7259 | 0.8411 | 0.6534 | 0.7621 |
| LSTM + MSE | 0.8637 | 0.7145 | 0.7653 | 0.7972 | 0.8693 |

表中 MobileNet-V2 的严格最高其实是 MLP+ranking 的 0.7446（上表 XGBoost 0.7259）；总体上 MLP + ranking 表现最好，但不是每个 network、每个 top-k 指标都必胜。例如 ResNet-50 Top-1 是 MLP+MSE 略高；部分 Top-5 最优也可能属于其他模型。

因此准确表述是：**在该持出协议的整体比较中，MLP + ranking 是最强通用选择，而非“所有数据点绝对最好”。**

## 数据采样结论

固定总 records 数量时，增加 task diversity 往往比给少数 tasks 采更多 programs 更有效。例如总量 12 万时，`600 tasks × 200 programs` 通常优于 `300×400` 或 `200×600`。

直觉：模型的主要泛化难点是新 subgraph/shape，而不是把一个旧 task 的局部候选曲面采得极密。这对灰盒校准也很有启示：先覆盖更多机制区域/shape buckets，再在高非线性边界加密。

## 预训练如何加速 Ansor

作者分别用 Intel Platinum 8272、Intel E5-2673、AMD EPYC 7452 和 NVIDIA K80 的数据训练四个 MLP + ranking 模型，再接入 Ansor；这项实验并没有为数据集中的六个平台各训练一个模型：

- Ansor default 没有 offline pretrained model，初期随机测量、在线训练，冷启动慢；
- TenSet pretrained model 一开始就能筛较好的候选；
- 在 Intel Platinum CPU 与 K80 GPU 的 ResNet-50、MobileNet-V2、BERT-base 上，达到相同 search quality 的时间最多缩短 10×。

“10×”指 **search time to same result quality**，不是：

- program 最终 latency 一定缩短 10×；
- cost model 预测误差缩短 10×；
- 新硬件零测量 10×。

论文 Figure 6 的具体条形数据还显示收益按 network/hardware 差异很大，CPU/GPU 都并非统一 10×。

## Transfer learning：论文实际做法

### Cross-platform直接复用

论文在三类 CPU 上用各自模型互测，也训练一个混合平台模型。源硬件模型可直接用于另一 CPU，但通常不如用目标硬件数据训练；混合模型在三平台上大致第二好，说明共享表示有潜力，不等于解决了新硬件问题。

### Online local residual model

先用 offline pretrained model 给 score，再用新 task 的少量在线 measurements 拟合局部修正：

$$
\hat T_{adapted}(P)=\hat T_{pretrained}(P)+\hat\Delta_{local}(P).
$$

在 ResNet-50、每 task 50 trials 的实验里：

- Setting 1：Intel 8272 为 6.22 ms vs 不迁移 6.43；E5-2673 为 27.26 vs 29.94。
- Setting 2（只看 local model 训练后最后 10 个 measurements）：Intel 8272 为 6.44 vs 7.15；E5-2673 为 28.92 vs 32.03。

作者只做 CPU 类型之间的 transfer，明确把 CPU↔GPU 留作未来工作，因为二者 memory hierarchy 与 execution model 差异大。

这与用户提出的“基于实测数据做分区段选择性校准”非常接近：offline 模型给初值，目标机少量数据学局部 gap，而不是全量重训或盲信 zero-shot。

## Additional random sampling：L2 最优之和未必是 E2E 最优

作者还发现，对每个 subgraph 独立选真实最快 program，组合成整网时未必全局最快。可能原因包括跨 task/runtime 的 cache、layout、测量噪声或组合效应。于是从每 task top-3 中随机组合，做 80 次端到端测量，能继续找到更低整网 latency。

这点很重要：即使 L2 component ranking 很准，仍需要 L3/端到端验证；`Σ 每个局部最优` 不保证系统级最优。

## 与相关工作的关系

| 工作 | 贡献 | TenSet 相比它多了什么 |
| --- | --- | --- |
| AutoTVM | 模板化 tensor tuning + learned cost model | 大规模跨平台公开离线数据与统一比较 |
| Ansor | 自动搜索空间、GBDT 在线闭环、task scheduler | 为 Ansor 提供 offline warm start；系统比较 MLP/ranking/MSE |
| TPU learned model | XLA/TPU 图语料、tile/fusion cost | TenSet 是 TVM/CPU/GPU program 语料；覆盖 6 平台但表示不同 |
| TIRAMISU learned model | AST + 大量特征 | TenSet 提供公开、规模更大、多平台数据 |
| TLP/MTL-TLP | primitive sequence + attention、多任务迁移 | TLP 直接在 TenSet 上继续研究表示与跨硬件数据效率 |
| NeuSight | 物理约束 tile/wave 跨 GPU 时延 | TenSet 偏数据/排序，不含 Roofline 上下界和新 GPU 硬件特征 |

## 优点

- 真实编译/执行测量，且保留重复 runtime 与 error，不是合成标签。
- 多平台、公开、数据/代码/教程齐全，显著提升 cost model 研究可复现性。
- 研究问题完整：模型、loss、metric、数据规模、搜索、迁移和 E2E 组合。
- 指出 top-k/search-based metric 比盲看 RMSE 更契合 autotuning。
- online local residual 给出了灰盒选择性校准的早期实证模板。

## 短板与不适用场景

- **生态/版本绑定**：programs 来自特定版本 TVM/Ansor 搜索空间和 graph partition；编译器升级后可能漂移。
- **硬件老且范围有限**：K80/T4 与几类 CPU，未覆盖现代 A100/H100/B200、Ascend NPU、TPU、Trainium。
- **偏 inference、小 batch、float/dense**：论文明确不含 sparse，训练/backward/optimizer/MoE 也不在范围。
- **静态 shape**：测试只覆盖固定 shape；动态 seqLen 需 buckets 或重新生成 tasks/programs。
- **排序不是绝对预测**：最佳 MLP+ranking score 不能直接用于 SLA、容量规划或成本核算。
- **跨硬件不是真 zero-shot 解法**：直接迁移退化，local correction 仍需目标机测量；未做 CPU↔GPU transfer。
- **采样偏差**：每 task 最多 4000 点，而搜索空间可十亿级；数据只代表 Ansor sampler 可到达的区域。
- **系统级缺失**：单 device programs 无 DP/TP/PP/EP、rank/topology、通信、排队与 overlap。
- **聚合风险**：局部 top-1 组合不保证 E2E 最优，论文自己用额外全网随机组合测量补救。

## 映射到“输入 → L1/L2/L3 → 输出”

```text
输入：带固定 shape 的 network + target hardware
  ↓
L1：TVM graph partition + Ansor search-space/program generation
  ↓
L2：TenSet 的核心位置
    offline real-runtime records
      → MLP/XGBoost/LSTM + MSE/ranking
      → pretrained candidate score
      → online target measurements + local residual correction
  ↓
L3：未建模；仅通过整网实测/随机组合发现局部之和问题
  ↓
输出：更少搜索时间找到高性能 tensor programs
```

对灰盒架构的直接启示：

1. 实测库必须记录完整 fingerprint：shape、program/schedule、硬件、compiler/runtime version。
2. 排序模型与绝对 runtime 模型应分开服务不同目标。
3. 新域采样应优先扩 task/机制覆盖，再在非线性区加密。
4. offline 预训练后仍需目标域 local residual/calibration。
5. L2 component 最优必须经 L3/E2E 组合验证。

## 快速记忆 5 点

1. TenSet 是 **约 5200 万条真实 program runtime records**，不是纯排名数据。
2. 120 networks/configs、2308 subgraphs、6 hardware、13,848 tasks。
3. MLP + LambdaRank 总体最好；Ansor 原模型则是 GBDT + weighted MSE。
4. top-k score 比 RMSE 更贴合 autotuning；但 ranking score 不可直接当毫秒。
5. 预训练最多减少 10×搜索时间；跨硬件仍需实测 local correction。

## 自测问题

1. 为什么同一 subgraph 在两台 CPU 上算两个 tasks？
2. 为什么 ranking 模型的 R² 可以极差，最终搜索却更好？
3. 数据总量固定时，为什么 task diversity 比每 task 更密采样常更重要？
4. local residual transfer 与灰盒“选择性校准”是什么关系？
5. 为什么每个 subgraph 的局部最优 program 组合后不保证整网最优？

## 术语表

- **record**：一个 program 在一个 task 上的重复真机时间或错误。
- **task**：subgraph 与 hardware platform 的配对。
- **offline model**：搜索前用历史大数据训练的模型。
- **online data**：当前搜索过程中在目标机新测的数据。
- **cold start**：没有预训练时，从随机候选开始收集标签的慢阶段。
- **LambdaRank/LambdaLoss**：面向排序指标优化的 loss/梯度构造。
- **top-k score**：模型预测前 k 中最好真实候选与数据最优的接近程度。
- **local residual model**：用少量目标域数据学习 pretrained prediction gap 的局部模型。
- **measurement budget**：允许上目标机编译/执行候选的次数预算。

## 证据索引

- 数据目的、规模与 up-to-10×：论文 Abstract、§1。
- search-based compiler 与 cost model 背景：论文 §2、Figure 1。
- 术语、硬件、采集管线：论文 §3、Table 1–2、Figure 2–4；另见[官方仓库](https://github.com/tlc-pack/tenset)。
- 特征、架构、loss、top-k 与集成：论文 §4。
- MLP/ranking 比较、数据规模、搜索和 transfer：论文 §5、Table 3–5、Figure 5–7。
- float/dense 与 partition 限制：论文 §7 Discussion。
