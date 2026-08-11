# DNN 性能预测 10 分钟速查

## 一句话总纲

端到端性能不是 `模型 FLOPs ÷ GPU 峰值算力`，而是：

```text
目标模型与负载
→ 生成每个 rank 的真实执行图和 shape（L1）
→ 估计每个 kernel/通信 component 的成本（L2）
→ 按依赖、重叠、争用、batch 和队列推进时间（L3）
→ 输出 latency/throughput/瓶颈/置信度
```

## 最容易混的 12 个概念

| 概念 | 只记这一句 |
| --- | --- |
| op | 数学/框架语义，如 matmul |
| kernel | GPU/NPU 实际启动的设备程序 |
| tactic/route | 同一 op 的某个具体实现、tile 和算法路径 |
| shape | 不只决定 FLOPs，也会触发 padding、尾波和算法切换 |
| tile | kernel 将大问题切成的局部计算块 |
| wave | 一批能同时铺到 SM 的 block；最后一批常利用不足 |
| fusion | 少 launch/中间流量，但会生成全新 kernel |
| Roofline | 乐观下界，不是实际 latency |
| rank | 通信组中的逻辑参与者编号；PyTorch 常见为进程，不是 GPU 性能等级 |
| collective | 多 rank 的协同通信，如 AllReduce/AllToAll |
| cost model | 预测局部成本或候选优劣，不一定输出毫秒 |
| event simulator | 沿依赖和资源状态组合局部成本 |

## 两条公式

$$
AI=FLOPs/Bytes
$$

$$
T_{roof}=\max(FLOPs/PeakFLOPs, Bytes/PeakBW)
$$

真实时间通常大于 `T_roof`。灰盒模型学习的是利用率、slowdown 或有界残差，而不是重新猜 FLOPs。

## rank 的最短解释

2 台机器×8 卡、采用 PyTorch 常见的一进程一卡：global ranks 为 0..15，每台 local ranks 都为 0..7。`rank=9` 可以映射到 node 1 的 GPU 1。数字 9 本身不让它更慢；真正影响性能的是它负责的 shard/stage/expert、通信伙伴和跨机链路。

## 四种并行分别改变什么

| 并行 | 切分对象 | 主要新增成本 | 最关键动态因素 |
| --- | --- | --- | --- |
| DP | batch/样本 | 梯度同步 | global/local batch 口径 |
| TP | 层内 tensor/权重 | AllGather/AllReduce/ReduceScatter | 每 rank GEMM shape 变小 |
| PP | 网络层和执行阶段 | stage send/recv、bubble | microbatch 与最慢 stage |
| EP | MoE experts/tokens | AllToAll | token 路由不均衡 |

## 11 篇论文各记一句

| 论文 | 一句话 |
| --- | --- |
| Daydream | 从真实 profile 构图，改图后回放，适合已观察配置附近的优化 what-if。 |
| Habitat | 用 wave 与硬件比率跨 GPU 缩放，算法会切换的 kernel 再用 MLP。 |
| nn-Meter | 先探测设备融合规则，再对真实 kernel 分段回归并求和。 |
| dPRO | 对齐多设备时间，构建全局 DFG，诊断和组合分布式优化。 |
| Proteus | 把 DP/TP/PP 策略编译成目标执行图，再模拟重叠与共享。 |
| Vidur | 用目标 GPU profile/插值给 component 定价，再模拟 LLM 请求和 scheduler。 |
| NeuSight | 将 kernel 分为 tile/wave，用 Roofline 约束并学习利用率。 |
| TPU cost model | 用 GNN 编码 XLA kernel 图和编译决策，为 tile/fusion 候选估成本。 |
| TLP/MTL-TLP | 把 TVM schedule primitive 当序列学习，并用少量目标数据迁移。 |
| Ansor | 自动生成 schedule 搜索空间，用 GBDT cost model 减少昂贵测量。 |
| TenSet | 提供大规模 tensor program 数据，并系统比较回归与 ranking cost model。 |

## 选论文的最短规则

- 想看“优化后会快多少”：Daydream、dPRO；
- 想看“并行策略还没部署怎么比较”：Proteus；
- 想看“LLM serving 容量规划”：Vidur；
- 想看“跨 GPU component 外推”：Habitat、NeuSight；
- 想看“移动端/NAS”：nn-Meter；
- 想看“编译器为何用 ranking”：Ansor、TenSet、TPU、TLP。

## 五个禁止跳步的检查

1. 没有模型/运行时语义，就没有唯一的 L1 图；
2. 不知道 tactic/route，L2 仍存在离散不确定性；
3. L2 很准，也不代表重叠和排队后的 L3 很准；
4. 排序分数不能直接当毫秒；
5. 没有目标 GPU/NPU 真测，只能形成方法级结论，不能形成业务 SLA 结论。
