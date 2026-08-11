# 11 篇论文横向对比矩阵

## 1. 它们究竟预测什么

| 论文 | 最小预测单位 | 最终输出 | 主要决策目标 | 是否要求绝对值 |
| --- | --- | --- | --- | --- |
| Daydream | profile task/kernel | 优化后 iteration time、speedup | 优化 what-if | 是 |
| Habitat | PyTorch op / CUDA kernel | 目标 GPU iteration time、吞吐、租金效率 | GPU 选型 | 是 |
| nn-Meter | 设备实际 kernel/fused kernel | 模型 inference latency | 移动端部署/NAS | 是 |
| dPRO | 全局 DFG task/communication sub-op | iteration time、瓶颈、组合优化收益 | 分布式训练优化 | 是 |
| Proteus | strategy-compiled op/communication task | throughput、memory、OOM | DP/TP/PP 策略选择 | 是 |
| Vidur | batch-level model execution + request event | TTFT、E2E、throughput | LLM serving 配置 | 是 |
| NeuSight | kernel tile/wave | kernel/opgraph latency | 新模型/新 GPU 外推 | 是 |
| TPU cost model | XLA kernel candidate | tile rank 或 kernel time | tile/fusion autotuning | tile 否；fusion 是 |
| TLP/MTL-TLP | TVM schedule candidate | candidate score/rank | tensor program 搜索 | 主要否 |
| Ansor | tensor program state/schedule | normalized throughput score | 自动调优搜索 | 主要否 |
| TenSet | tensor program candidate | runtime 标签与 learned score | cost-model 训练/评估 | 数据有绝对值；ranking score 未校准 |

## 2. 图、shape 和 route 从哪里来

| 论文 | 执行图来源 | shape 来源 | route/tactic 是否已知 | 全新 route 怎么办 |
| --- | --- | --- | --- | --- |
| Daydream | 基线 trace 恢复，再人工变图 | 基线实际执行 | 基线 task 已知 | 新 task 时间须外部给定 |
| Habitat | 源 GPU 运行时 op 序列 | 源端固定 shape | kernel-alike/varying 分类 | 交给预训练特定 op MLP；仍可能 OOD |
| nn-Meter | 模型 IR + 设备 kernel/fusion 检测 | IR/采样配置 | 先探测设备规则 | 新设备重新采样建模 |
| dPRO | 目标集群多 rank trace | 目标实测 | 目标 trace 已包含 | 非离线新硬件外推工具 |
| Proteus | Strategy Tree 编译生成 | 按 shard/并行策略显式推导 | op 类型已知，具体成本靠 profile | 新 shape/hardware 做 microbenchmark/profile |
| Vidur | 声明式模型与 scheduler | 按 TP、batch、token 显式推导 | profile 采样实际 backend | 采样范围外需补 profile |
| NeuSight | 发布/提取的 opgraph | opgraph + kernel metadata | 依赖 kernel name/grid/tile | 完全新机制外推无保证 |
| TPU cost model | XLA compiler IR | IR node feature | tile/fusion 候选明确 | 训练分布外会退化，仍需测量 |
| TLP/MTL-TLP | TVM schedule primitive 序列 | task/schedule 给定 | schedule 候选明确 | 目标硬件微调 |
| Ansor | 自动生成搜索状态 | workload/task 给定 | schedule 候选明确 | 上机测量反馈更新 |
| TenSet | 收集的 task/schedule | 数据集给定 | 候选明确 | 目标 task/hardware 数据迁移 |

## 3. 真实测量依赖与外推范围

| 论文 | 预测前要测什么 | 新 shape | 新 GPU/TPU | 分布式/动态系统 |
| --- | --- | --- | --- | --- |
| Daydream | 基线 workload trace | 弱，不适合直接复用旧 task 时间 | 弱 | 可表达部分 DP；无现代 TP/PP/MoE 系统验证 |
| Habitat | 同模型/shape 的源 GPU iteration | 不是目标问题 | 主要目标，但受 route 一致性限制 | 不建模通信 |
| nn-Meter | 每种设备的自适应 kernel 数据 | 设备内采样支持域较好 | 换设备重建 | 聚焦串行移动 inference |
| dPRO | 目标集群全局 trace | trace 覆盖处准确 | 不做纯离线换硬件 | 强项是目标集群全局 DFG；实证偏 DP/PS/AllReduce |
| Proteus | 目标硬件 op profile、重叠系数等 | 重新推导后查/profile | 需目标 profile | DP/TP/PP 是强项；动态 MoE/变长有限 |
| Vidur | 目标 GPU 单卡 profile | 插值范围内 | 需各目标 GPU profile | LLM inference request/scheduler 是强项 |
| NeuSight | 源 GPU/model kernel 数据；预测需 metadata | 论文目标之一 | 论文目标之一 | 通信较粗，超大规模仅模拟 |
| TPU cost model | 大规模 TPU 编译候选真测 | 同分布强、人工 OOD 会退化 | v2/v3 分别有数据，非零样本证明 | 不负责 |
| TLP/MTL-TLP | tensor program 运行时间；迁移仍需目标标签 | 搜索任务内 | 少量目标域微调 | 不负责 |
| Ansor | 在线上机测量作为搜索反馈 | 自动生成搜索空间 | 每个 target 都调优 | 不负责 |
| TenSet | 离线大规模测量集 | 依赖 split 与迁移 | 支持跨 task/device 研究但非万能零样本 | 不负责 |

## 4. 它们的“理论”是哪一种理论

不要把所有公式都理解为有形式化误差保证的理论。

| 类型 | 代表 | 作用 | 没有保证什么 |
| --- | --- | --- | --- |
| 依赖图与调度语义 | Daydream、dPRO、Proteus、Vidur | 保证模拟按指定依赖/事件规则推进 | 输入图和 component time 不一定正确 |
| 机理解析近似 | Habitat、NeuSight | 用 wave、FLOPs、bytes、Roofline 规定结构 | cache、route、新架构和争用仍可能错 |
| 监督学习损失 | TPU、TLP、Ansor、TenSet | 让候选成本/排序匹配测量标签 | OOD 泛化和绝对值校准不自动成立 |
| 采样/搜索算法 | nn-Meter、Ansor、TLP | 用更少实测找到有信息或高性能候选 | 不等于端到端性能模型完整 |

多数系统论文提供的是**模型结构 + 经验假设 + 实证准确度**，而不是对任意硬件/模型的误差上界证明。

## 5. 相关工作演进的精确说法

### Daydream → dPRO

dPRO 不是简单“把 Daydream 换成更大图”，而是针对分布式 trace 增加跨设备时钟对齐、细粒度通信和自动组合优化。Daydream 更轻量地表达人工 what-if；dPRO 更依赖目标集群观测。

### Habitat → NeuSight

二者都尝试跨 GPU，但 Habitat 从源 GPU runtime 锚点缩放 kernel-alike 操作，并对会换 kernel 的 op 直接回归；NeuSight 更系统地分解 tile/wave，用 Roofline 限制 learned utilization，并把“新 GPU + 新模型”作为核心评测。NeuSight 对 Habitat 的大误差是后续重训复测，不能倒写成 Habitat 自己报告的结果。

### nn-Meter 与 Habitat/NeuSight

nn-Meter 的重点不是跨 GPU 零样本，而是先探测每种移动设备的融合/kernel 规则，并用自适应采样把该设备建准。它牺牲换设备即用的能力，换取大规模移动模型/NAS 的设备内准确率。

### Proteus 与 Daydream/dPRO

Daydream/dPRO 从运行 trace 观察图；Proteus 从模型和并行策略主动生成候选图。这使 Proteus 更适合未部署策略搜索，但它仍要从目标硬件 profiling 获得 op 成本和重叠参数。

### Vidur 与 Proteus

两者都显式推导目标配置 shape 并进行模拟。Proteus 重点是训练的 DP/TP/PP 策略；Vidur 重点是 LLM inference 的请求、prefill/decode、KV cache、batch 和 scheduler。

### Ansor/TenSet、TPU model 与 TLP

- Ansor 把 cost model 放进生成式自动调优循环；
- TenSet 提供更大、更统一的数据并验证 ranking 训练；
- TPU model 直接编码 XLA kernel 图、tile/fusion 决策，服务 TPU 编译器；
- TLP 将 TVM schedule primitive 视为序列，并通过多任务学习减少目标硬件数据。

它们共同点是服务编译候选筛选，表示绑定各自 compiler IR/schedule，不能无代价迁移成任意 GPU/NPU 的端到端绝对模型。

## 6. 对当前灰盒方案的直接启示

| 设计选择 | 来自哪些证据 | 仍需验证什么 |
| --- | --- | --- |
| L1 显式生成 per-rank shape/通信 | Proteus、Vidur | 真正模型代码和 runtime 对齐 |
| route 先分类再回归 | nn-Meter、Habitat、NeuSight 反例 | pre-kernel route classifier 精度 |
| Roofline/tile/wave 约束残差 | Habitat、NeuSight、工业 cost model | 多 dtype、attention、fusion、NPU |
| 精确实测缓存 + OOD fallback | 各 autotuner 的 shortlist→measure 模式 | gate 的 precision/coverage 与测量预算 |
| L2 与 L3 解耦 | Daydream、dPRO、Proteus、Vidur | 争用、重叠和 scheduler fidelity |
| 搜索头与绝对值头分开 | TPU、TenSet、TLP | 共享表征是否值得，绝对值校准怎样维护 |
| 分区段选择性校准 | nn-Meter 分 kernel、当前公开数据实验 | 业务真机上是否优于 global/always-segment |

