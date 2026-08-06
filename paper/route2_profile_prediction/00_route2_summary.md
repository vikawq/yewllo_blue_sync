# 路线二总览：Profiling → 查表/拟合 → 系统级性能预测

> 调研对象：Habitat、Vidur、NeuSight（用户简称 GPU Forecasting）、Training Time Prediction for Mixed Precision-based Distributed Training（用户简称精度感知预测器）。  
> 调研日期：2026-08-06。  
> 阅读前置：`survey/录制回放_整体技术路线与实现设计_V0.5-详细版.md`、`V0.6` 实验验证、`V0.7` 系统化架构、`V0.8` 字段与源码证据。  
> 证据原则：论文方法/实验标“原文事实”；跨论文评价标“归纳”；昇腾方案标“设计推断”。页码细节见各篇笔记。

## 1. 结论先行

四篇论文共同证明：只用 FLOPs 或 `op+shape` 粗表不能稳定预测现代 DNN/LLM 性能；需要把实际执行得到的 kernel/runtime 信息、operator workload特征和硬件规格联合建模。但它们解决的是**成本预测**，不是本项目完整的语义录制回放：

```text
Execution Recipe（路径/工作量/状态/依赖）
  不能由 latency 表替代

Physical Binding（kernel/layout/tiling/collective/graph）
  是 cost-model 的输入

Observation Ledger（实测 duration/counters/evidence）
  是训练样本和验证 ground truth

Cost Model（查表/解析/ML）
  输出目标 DEG 节点 service time

Event-driven Replay/Simulator
  用 DAG、stream、scheduler、peer arrival 组合端到端时间
```

最重要的横向判断：

1. **Habitat** 适合“有一张源 GPU、预测另一张 GPU”：源 trace 揭示真实 kernel；相似 kernel 用 wave scaling，少数架构特化 operation 用直接 latency MLP。优点是混合解析/实测，缺点是直接 latency MLP 对远 OOD 可能崩溃，且单卡迭代直接求和。
2. **Vidur** 最接近本路线完整闭环：operator triaging → RF 插值 → lookup table → scheduler/KV/queue 事件仿真 → deployment search。它把 workload 动态性建进系统，但 attention sufficient statistics、通信与 overlap 仍较粗。
3. **NeuSight** 最适合跨未见 GPU 外推：MLP 只学物理上限下的 utilization，tile/wave 与 roofline 负责结构约束；OOD 精度明显优于直接 latency MLP。其关键隐性依赖是 tile/implementation database。
4. **精度感知短文** 只足以说明 dtype 必须是 per-operator 一等特征；3 页 v1 的 profile/schema/实现/通信模型证据不足，不能作为成熟系统基线。

## 2. 论文身份与路线纠偏

| 用户简称 | 正式身份 | 纠偏 |
|---|---|---|
| Habitat | USENIX ATC’21，正式会议论文，训练 | 不是 serving/边缘 inference predictor；主体为单 GPU 训练迭代 |
| Vidur | MLSys’24，arXiv 2405.05465v2，LLM inference | 同时属于路线二和 serving 事件模拟；不是简单静态查表求和 |
| GPU Forecasting | NeuSight，正式题名 *Forecasting GPU Performance for Deep Learning Training and Inference*，ASPLOS’25，arXiv 2407.13853v3 | 早期 metadata/简称不统一；笔记以 ASPLOS 正式题名为准 |
| 精度感知预测器 | *Training Time Prediction for Mixed Precision-based Distributed Training*，arXiv 2604.16145v1 | 仅 3 页、无系统名/代码/venue，不应过度解读 |

## 3. 统一对比表

| 维度 | Habitat | Vidur | NeuSight | Precision-aware v1 |
|---|---|---|---|---|
| 目标 | 跨 GPU 单卡训练迭代 | LLM serving 配置/负载仿真 | 新模型×未见 GPU 的训练/推理 | mixed precision 分布式单 iteration |
| 必须访问目标 GPU | 否；需源 GPU | 新 SKU 通常需初始 profile | 否；需公开规格与 tile 估计 | 是；实验/文字按给定 H100 job profile |
| 录制粒度 | PyTorch operation + kernel | token/sequence/comm operator | operator/kernel + tile/wave | Torch.fx op + actual AMP dtype |
| Profile key 核心 | op参数、源 kernel/launch/counters、GPU规格 | shard、tokens、prefill `Σp²`、KV bytes、message size/topology | op/kernel、dims、tile、GPU per-SM 特征 | local shape、F/B、precision、DP/TP/PP |
| 拟合 | wave解析 + 4个直接 latency MLP | Random Forest → lookup table | 最近邻 tile 表 + bounded-utilization MLP + roofline | 未给插值；按 job config profile并求和 |
| OOD 机制 | 无拒绝/置信度 | 无拒绝/置信度 | 物理上界改善 OOD；仍无显式 reject | 未讨论 |
| 单卡组合 | operation 时间求和 | event-driven batch execution | kernel 时间求和 | subgraph op 时间求和 |
| Scheduler/queue | 无 | 有，三层 scheduler + arrival + KV capacity | 无 | 无 |
| 通信 | 主体无 | profile all-reduce/all-gather/send-recv | link utilization缩放，插网络 op | `V/B_link` |
| overlap | 主体无 | 论文版 synchronous PP；async future | 未建模真实 async overlap | 无，直接加法 |
| PP | future/需新模型 | stage scheduler + microbatch（同步） | GPipe schedule/bubble | `T_comp×(PP-1)` 粗式 |
| 状态 | 无 | KV capacity/preempt/restart，不含数值 KV | 无 | 无 |
| 开源 | 是，Apache-2.0，源码构建 | 是，MIT，完整 simulator | 是，MIT，有 artifact/data | 未核实到代码 |
| 代表准确率 | 11.8% avg | 静态 P95 ≤3.33%；85%容量动态几乎<5% | 9.7% infer/7.3% train；OOD GPU avg 8.1% | mixed 9.8%、FP16 10.64% |
| 成熟度 | 可复现实验原型 | 研究系统，中高 | artifact 完整的实验原型 | 概念验证/短文 |

## 4. Profiling 与表 schema 的演进

### 4.1 Habitat：从 operation 到 kernel

Habitat 的 schema 隐含两层：operation 输入/参数用于 MLP；底层 kernel name、launch configuration、duration 和 CUPTI 算术强度用于 wave scaling。其经验说明：framework op 相同并不意味着不同 GPU 上的 kernel implementation 相同，必须先分类 kernel-alike/kernel-varying。

关键证据：`01_habitat.md` §3–§4；原论文 PDF 6–8/印刷 507–509。

### 4.2 Vidur：按 runtime sufficient statistic 做 triaging

Vidur 把表 key 压缩为：token-level 看本轮总 token；prefill attention 看 `Σp_i²`；decode attention 看总 KV read；communication 看 bytes+topology。这比全 shape 笛卡尔积节省大量样本，并使 RF 小数据插值可行。

关键证据：`02_vidur.md` §3–§4；原论文 PDF 5–6，§4.3–§4.4。

### 4.3 NeuSight：表不只存 latency，还存 tile regime

NeuSight 使用 database 最近匹配 tile size，再计算 tile/wave，MLP 学 utilization。它说明成本表应保存“为什么进入这个性能 regime”的结构字段，而非只存 duration。

关键证据：`03_neusight_gpu_forecasting.md` §3–§4；原论文 PDF 6–10，Eq. (2)–(8)、表 3、§6.1 `Tile size`。

### 4.4 Precision-aware：dtype 是 operation 级字段

Mixed precision 不是 run-level 标签。每个 operand/accumulation/communication tensor 的实际 dtype 必须进入 key；否则 compute kernel 和 message bytes 都会错。

关键证据：`04_precision_aware_training_predictor.md` §3–§5；原论文 PDF 2，§III。

## 5. 推荐给昇腾录制回放的统一 CostModelRecord

以下是四篇论文与 V0.8 结合后的**设计推断**：

```yaml
cost_model_record:
  identity:
    semantic_op_id: "step42.layer17.moe.gmm0"
    op_type: "grouped_gemm"
    phase: "prefill|decode|training_fwd|training_bwd"
    implementation_family: "cann-op/kernel/fusion/graph-mode"
  workload:
    global_shape: []
    local_logical_shape: []
    storage_shape: []
    valid_extent: []
    m_n_k: []
    lengths_summary: {}
    expert_counts: []
    rank_splits: []
    effective_kv_bytes: 0
    locality_summary: {}
  numeric:
    input_dtypes: []
    output_dtypes: []
    accumulation_dtype: "..."
    quantization: {}
  binding:
    soc: "..."
    cann: "..."
    torch_npu: "..."
    npu_format: "ND|NZ|..."
    block_dim: null
    tiling_key: null
    workspace_bytes: null
    graph_descriptor: null
  distributed:
    group_id: null
    ordinal: null
    ordered_ranks: []
    split_sizes: []
    message_bytes: 0
    rank_placement: {}
  observation:
    samples_ns: []
    median_ns: 0
    p90_ns: 0
    source_locator: "..."
    evidence_level: "fact"
  predictor:
    kind: "exact|lookup|rf|bounded_mlp|analytic|fallback"
    model_version: "..."
    training_domain: {}
    distance_or_confidence: null
    ood: false
```

### 为什么这些字段不可删

- Habitat：`implementation_family/launch/counters` 防止同 op 跨设备误缩放。
- Vidur：`valid_extent/length/expert/KV/bytes` 避免 padded shape 代替 workload。
- NeuSight：`format/tiling/block_dim/硬件特征` 区分 tile/wave regime。
- Precision-aware：`dtype/accumulation/quant` 同时决定 compute 与 comm。
- V0.6/V0.8：`group/ordinal/placement/predecessor` 防止把 peer wait 写成固有通信成本。

## 6. 插值、外推与 fallback 策略

建议按风险从低到高分层，而非一个模型包打天下：

1. **Exact lookup：** 完整 key 命中同版本、同 binding 的重复观测，输出 median/P90。
2. **Local interpolation：** 同 implementation/tiling regime 内用 RF/GBDT 对相邻 shape/valid extent 插值，类似 Vidur。
3. **Analytic scaling：** kernel family 稳定时用硬件资源比/roofline/Cube-Vector 上界缩放，类似 Habitat wave scaling。
4. **Bounded ML：** 学 relative utilization，不直接学 latency；用物理上界和工作块/波次恢复，类似 NeuSight。
5. **Constrained synthetic calibration：** 对 MoE/attention/collective 合成相同 count/split/locality 的 microbenchmark。
6. **Reject/OOD：** implementation、dtype、layout、tiling、版本或 feature 距离越界时要求实测；不能静默用最近点。

论文均没有完善的 OOD reject/置信度，这是本项目必须补的工程能力。

## 7. 系统级组合规则

### 7.1 不允许的组合

```text
end_to_end = Σ source_measured_duration
```

这会同时犯三种错误：把源 binding 当目标 binding、把 source wait 当 intrinsic cost、把 scheduler/queue/overlap 丢掉。

### 7.2 正确组合

```text
Source semantic trace
  -> Target Recipe Transformer
  -> target local ABI / decision / state / collective intent
  -> CostModel predicts node service time
  -> event-driven runtime replays stream/DAG/peer dependencies
  -> target scheduler/queue/KV capacity advances system state
  -> new Observation Ledger
```

Vidur 已示范 scheduler/queue 事件驱动；本项目还要加入 V0.6 所需的 per-rank predecessor/arrival DAG 和 V0.8 的 state/decision/graph binding。

### 7.3 Compute、communication、overlap、queue 分开

| 成本 | 建模输入 | 组合方式 |
|---|---|---|
| compute/kernel | local ABI、valid extent、dtype/layout/tiling/SoC | device stream/DAG |
| communication transit | kind/group/splits/bytes/placement/HCCL binding | 所有 peer 到达后按网络模型推进 |
| arrival/wait | peer predecessor completion、host enqueue、stream queue | 由跨-rank DAG自然产生 |
| scheduler queue | arrivals、request state、token/KV budget、policy | scheduler event loop |
| state transition | KV/block/slot/router/spec versions | Recipe 的 state edge，不是 latency table |

## 8. 冷启动与采样计划

### 8.1 模型 onboarding

1. 由 `ModelAdapterSpec` 提取 key sinks、global→local partition、decision producer和 state scope。
2. 先按 Vidur triaging 找最小 sufficient statistics，而不是全 shape 网格。
3. 对 table 中出现 implementation/tiling regime 边界处主动加密采样。
4. dense building blocks 跨模型复用；attention/MoE/KV/quant/graph 按 implementation family 单独建库。

### 8.2 目标硬件 onboarding

1. 采 SoC/HBM/片上存储/理论 compute/bandwidth等公开规格。
2. 用小型 microbenchmark 校准实际 HBM、Cube/Vector、collective link 和 launch overhead。
3. 对 CANN 不透明 tiling，只保存可观测的 kernel/format/blockDim/workspace 与经验 regime；不可见字段为 unknown。
4. 新 SoC 初期使用 bounded analytical/ML，并提高 OOD 比例；有真机后增量回填 exact/RF 表。

### 8.3 动态 workload 采样

至少覆盖：prefill/decode/mixed；raw/valid/padded token；length skew；KV locality；MoE expert/rank histogram；TP/EP group与跨机 placement；50/85/95% capacity。Vidur 的结果显示接近容量时成本误差会被 queue 级联放大。

## 9. 验证矩阵与误差口径

### 9.1 分层指标

| 层 | 指标 |
|---|---|
| 单样本 | exact hit rate、table coverage、OOD rate |
| 单 op | MAPE/sMAPE、median/P90 absolute error、regime classification accuracy |
| rank-local | phase latency、kernel sequence、busy/idle、local ABI digest |
| distributed | collective sequence/bytes、arrival/completion skew、transit/wait 分解 |
| serving | TTFT/TBT/E2E/throughput/queue delay，按 50/85/95% capacity |
| compatibility | V0.8 M/L/W/S/D/P/N 各维 exact/transformed/comparable/failed/unknown |

### 9.2 不能只报平均 MAPE

Habitat/NeuSight 表明高误差小 op 可能不影响端到端；Vidur 表明小 op error 在高负载却可能经 queue 级联。建议同时报：

- op importance-weighted error；
- worst-regime/OOD error；
- tail latency error；
- capacity point偏移；
- 模型选择/配置排序是否正确；
- 置信度覆盖曲线（拒绝多少样本后误差降到何水平）。

## 10. 四篇论文的主要证据定位

| 事实 | 原论文定位 |
|---|---|
| Habitat wave scaling / Eq. (1)–(3) | Habitat PDF 6–7 / 印刷 507–508，§3.3、§4.2 |
| Habitat MLP schema与采样范围 | Habitat PDF 6–8 / 印刷 507–509，§3.4、§4.3、表 1 |
| Habitat 11.8% | Habitat PDF 9–10 / 印刷 510–511，§5.2、图 3 |
| Vidur operator triaging | Vidur PDF 5，§4.3 |
| Vidur RF→lookup | Vidur PDF 5–6，§4.2、§4.4 |
| Vidur scheduler/KV/queue | Vidur PDF 6，§4.5；PDF 8–9，§7.2 |
| Vidur 静态/动态 fidelity | Vidur PDF 8–9，图 3/4、§7.2；附录 A.1 |
| NeuSight tile/wave/roofline/utilization | NeuSight PDF 6–8，§4.1–§4.3、Eq. (1)–(8) |
| NeuSight tile database 最近匹配 | NeuSight PDF 10，§6.1 `Tile size` |
| NeuSight OOD 8.1% | NeuSight PDF 10，§6.2、图 7/8 |
| Precision-aware op dtype/profile | 2604.16145 PDF 2，§III、Algorithm 1 |
| Precision-aware 9.8%/10.64% | 2604.16145 PDF 3，§IV、图 2 |

## 11. 独立论文笔记

- [Habitat](01_habitat.md)
- [Vidur](02_vidur.md)
- [NeuSight / GPU Forecasting](03_neusight_gpu_forecasting.md)
- [Precision-aware Distributed Training Predictor](04_precision_aware_training_predictor.md)

## 12. 推荐落地顺序

1. 先做同 SoC/同 topology 的 exact lookup + RF 插值，打通 schema、evidence、OOD 与 per-op 验证。
2. 按 Vidur 思路加 LLM workload triaging，但把 MoE/KV/index/locality 纳入 sufficient statistics。
3. 按 NeuSight 思路增加 Ascend-specific physical bound/utilization model，服务跨 shape/SoC 外推。
4. 将 precision/format/quant/accumulation 设为硬 key，禁止跨 dtype 混表。
5. cost model 接入 V0.8 DEG/event identity，先同 topology 回放，再做 TP/EP/placement 变换。
6. 最后扩展 HCCL transit+arrival、scheduler/queue 与大规模 what-if；任何未验证多节点预测明确标 simulation。

## 13. 最终判断

路线二适合成为本项目的“物理成本层”，可显著降低每个 shape/配置都上机测量的成本；但它无法独立决定目标执行路径。正确产品形态不是一个万能 latency predictor，而是：

```text
有版本的 profiling 数据库
+ 分 regime 的 exact/RF/analytic/bounded-ML predictor
+ OOD/reject 与证据账本
+ Target Recipe/DEG
+ 分布式事件模拟器
+ 分层验证报告
```

其中 Habitat 提供 runtime cross-device calibration，Vidur 提供 workload-aware table 与系统事件组合，NeuSight 提供受物理约束的 OOD 外推，precision-aware 短文提醒 dtype/communication bytes 必须进入 key；V0.8 则负责把这些成本模型放在正确的 Recipe/Binding/Observation 边界内。
