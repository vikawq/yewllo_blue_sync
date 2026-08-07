# NeuSight：以 tile/波次和物理上界约束的跨 GPU 性能预测

> 证据截图说明：正文中的 `原文截图 E###` 可跳转到文末证据卡片。截图按 PDF 物理页码生成；原有章节、图表、算法和段落定位保持不变。


> 论文：Seonho Lee, Amar Phanishayee, Divya Mahajan, **Forecasting GPU Performance for Deep Learning Training and Inference**, ASPLOS 2025, pp. 493–508。框架名 **NeuSight**。  
> 原文：[arXiv PDF（2407.13853v3）](https://arxiv.org/pdf/2407.13853)；[arXiv 页面](https://arxiv.org/abs/2407.13853)；[ACM DOI](https://doi.org/10.1145/3669940.3707265)；[开源仓库（已迁移到 scai-tech）](https://github.com/scai-tech/NeuSight)。  
> 名称核对：用户所称“GPU Forecasting”不是正式论文题名；arXiv 搜索结果有时显示早期元数据 `Data-driven Forecasting of Deep Learning Performance on GPUs`，正式 ASPLOS’25 题名如上。PDF 标 `arXiv:2407.13853v3, 12 Dec 2024`。  
> 页码口径：PDF 共 16 页，论文正文 PDF 1–13，artifact appendix PDF 14，参考文献 PDF 14–16。下文页码以 PDF 文件页从 1 起算；段落号按小节正文自然段计。

## 1. 一句话结论与路线位置

NeuSight 不让 MLP 直接回归整个 kernel latency，而是把 kernel 分成由 GPU library 实际 tiling 策略定义的 tile，计算 tile 数和跨 SM wave 数；MLP 只预测“一个 tile 在该 GPU 上能达到 roofline 上界的利用率”，并用 sigmoid 将其约束在 1 以下，再用峰值 FLOPS/内存带宽物理上界恢复 tile、kernel 和模型 latency。预测时还通过一个 tile-size database 做最近匹配，最后沿 Torch.fx dataflow 图顺序求和，并可加单机多 GPU 的网络算子。

它属于“profiling → ML+解析约束 → 查 tile 表 → 图级组合”，核心目标是**新模型/新 shape 在不可访问的新 GPU 上的 forecasting**，不是 replay 原 trace。

定位：PDF 1–2，Abstract 与 §1 第 5–9 段；PDF 5–8，§4、图 3–6、Eq. (1)–(8)。 〔[原文截图 E001](#evidence-e001)〕

## 2. 问题、输入与输出

### 2.1 问题

新 GPU 贵且难获得，而 kernel latency 受 SM、cache、HBM、warp/tiling 和 cuDNN/CUTLASS 实现共同影响。cycle-accurate simulator 对每代 GPU 维护成本高且很慢；纯 roofline 太粗；直接 latency 的线性回归/MLP 对训练分布外的新 shape/GPU 泛化差。NeuSight 要在不执行“目标模型×目标 GPU”的情况下，预测训练与推理性能。

定位：PDF 2，§1 第 1–8 段；PDF 3–4，§3.1–§3.3、图 2。 〔[原文截图 E002](#evidence-e002)〕

### 2.2 输入

1. PyTorch/TensorFlow 模型或 Hugging Face 风格 model config；Torch.fx 提取 operator/kernel graph 与 tensor dimensions。
2. 目标 GPU 可公开获得的规格：memory size/bandwidth、SM 数、L2 cache size、peak FLOPS；仓库 device config 还含 cores/SM 和 frequency。
3. 分布式可选输入：DP width、PP depth/schedule、Megatron tensor-model-parallel width、目标 network link bandwidth。
4. 离线训练资产：operation latency 数据、kernel name、thread-block 数、推断的 tile size、GPU features、预训练五类 MLP。

定位：PDF 5，§4 第 1 段；PDF 7–9，图 6、§5、§6.1；artifact appendix PDF 14；仓库 README `Tool Inputs`。 〔[原文截图 E003](#evidence-e003)〕

### 2.3 输出

- per-tile utilization / latency；
- per-kernel latency；
- 单 GPU 训练 iteration 或 inference latency；
- 单机多 GPU 的 DP/TP/PP 总时间；
- 预测误差报表。

论文的文本生成 inference 指标是 first-token latency，不是在线 serving 的 TTFT（包含排队）或连续 decode TPOT/TBT。

定位：PDF 8，§5；PDF 9，§6.1 `DNN workloads evaluated`；PDF 10–13，§6.2–§6.3。 〔[原文截图 E004](#evidence-e004)〕

## 3. Profiling 数据与查表 schema

### 3.1 采样数据集

五类 predictor 的训练数据：

| operator | 数据点 | 采样维度范围 |
|---|---:|---|
| BMM | 87,627 | batch 与各矩阵维 1–1024 |
| Fully connected | 32,256 | batch 1–8192，input/output 1–65,536 |
| Element-wise | 26,066 | batch 512–16,384，vector 512–4096；add/div/mul/GELU/ReLU/Tanh |
| Softmax | 1,807 | batch 4096–16,384，vector 512–4096 |
| LayerNorm | 1,501 | batch 4096–16,384，vector 512–4096 |

每个 operation 跑 25 次取平均；输入 tensor 正态随机；主体数据全为 FP32。NVIDIA 侧 PyTorch 2.1/CUDA 12.1，AMD 侧 PyTorch 2.4.1/ROCm 6.1；20% 留作 validation。

定位：PDF 9–10，§6.1 `Generating the training dataset` 及列表。 〔[原文截图 E005](#evidence-e005)〕

### 3.2 GPU 训练/测试划分

论文表 4 给出 P4、P100、V100、T4、A100-40GB、A100-80GB、L4、H100，以及 AMD MI100/MI210/MI250。NVIDIA 的 OOD 测试包括 A100-80GB、L4、H100；A100-40GB 在训练集而 80GB 版进入测试。AMD 用 MI100/MI210 训练、MI250 测试。

定位：PDF 9，§6.1 `Hardware`、表 4。 〔[原文截图 E006](#evidence-e006)〕

### 3.3 Tile database

PyTorch Profiler 提取 kernel name 与 thread-block 数：GEMM 从 kernel-name metadata 推断 tile size，其他 kernel 用 thread blocks 反推。训练时数据库记录：

```text
kernel_name
input_dimensions
GPU_features
tile_sizes
```

预测时按 kernel name、input dimensions 和 GPU features 找“closest match”估计 tile size。因此 NeuSight 不是完全无查表：MLP 预测利用率，但 tile regime 由 nearest-neighbor database 提供。

定位：PDF 10，§6.1 `Tile size` 段（定位词 `closest match in the database`）；PDF 8，图 6 的 `NeuSight Tile Database`。 〔[原文截图 E007](#evidence-e007)〕

论文没有披露 closest-distance 公式、类别特征编码、同距处理、库未命中/OOD 拒绝和 tile table 的覆盖诊断。这是复现和跨 NPU 迁移的关键缺口。

### 3.4 推荐的统一 schema

以下是对论文机制的**工程归纳**：

```text
TileProfileKey:
  op_type
  kernel_name / implementation_family
  input/output dimensions
  dtype
  device_features:
    num_sms, peak_flops, memory_bw, memory_size, l2_size

TileObservation:
  tile_dimensions
  thread_blocks
  measured_kernel_latency_mean_of_25
  derived_num_tiles / num_waves

PredictorFeatures:
  flops_per_tile / peak_flops_per_sm
  memory_per_tile / memory_bw_per_sm
  (num_waves * memory_per_tile) / l2_per_sm
  (num_waves * memory_per_tile) / memory_size_per_sm
  arithmetic_intensity / machine_balance
```

最后五项来自表 3；原表 PDF 排版把分子/分母拆成多行，语义分别是 per-SM 归一化的 compute、memory、L2/容量压力与 roofline 比。

定位：PDF 7–8，§4.3 最后两段、表 3、图 6。 〔[原文截图 E008](#evidence-e008)〕

## 4. 预测方法

### 4.1 Tile、wave 与 kernel latency

若 output 第 `i` 维为 `x_i`，tile 第 `i` 维为 `t_i`：

```text
num_tiles = Π ceil(x_i / t_i)                         Eq. (2)
num_waves = ceil(num_tiles / num_sms)                 Eq. (3)
PerOpLatency = PerTileLatency × num_waves             Eq. (4)
```

这里假定每个 SM 一次执行一个 tile，wave 顺序执行；同一 wave 内并行和线程 stall hiding 被后面的 utilization 模型吸收。

定位：PDF 6–7，§4.2 `Tile-granularity prediction`、Eq. (2)–(4)、图 4。 〔[原文截图 E009](#evidence-e009)〕

### 4.2 Roofline 物理边界

先计算 kernel 算术强度 `K=flops_k/mem_k`，`rooflineBW=min(K×memoryBW_peak, FLOPS_peak)`。然后：

```text
PerTileLatency = flops_tile / achievedBW              Eq. (5)
achievedBW = rooflineBW × utilization                 Eq. (6)
```

这样模型不能宣称超过目标 GPU 的峰值 compute/memory bound。

定位：PDF 6，§4.1 `Fundamental performance laws`、Eq. (1)；PDF 7，§4.2 Eq. (5)–(6)。 〔[原文截图 E010](#evidence-e010)〕

### 4.3 MLP 不预测 latency，只预测利用率曲线

NeuSight 用 MLP 输出 `alpha,beta`，经 sigmoid 限制到 0–1，再令：

```text
utilization = alpha - beta / num_waves                Eq. (7)
alpha, beta = sigmoid(MLP(features))                   Eq. (8)
```

即 wave 增多时利用率上升、渐近 `alpha`，并受物理上限约束。五个独立 MLP 对应 BMM、FC、element-wise、softmax、layernorm；每个 8 hidden layers×512 ReLU。未知 operator 默认 memory-bound，latency=`memory requirement / memory bandwidth`。

定位：PDF 7，§4.2 `Imposing performance laws`、Eq. (7)–(8)；§4.3 第 1–4 段。 〔[原文截图 E011](#evidence-e011)〕

### 4.4 训练设置与误差函数

MLP 用 AdamW+L2，100 epoch，batch 16–128，各 predictor LR 在 `1e-6`–`5e-3`；NeuSight loss 是 symmetric MAPE，基线 Habitat 用 MAPE。正文没有给出每类 predictor 的精确 LR、hidden dropout、early stop 或 alpha/beta label 构造细节。

定位：PDF 10，§6.1 `Training the NeuSight predictor`。 〔[原文截图 E012](#evidence-e012)〕

### 4.5 Operator fusion

连续 vector kernel fusion：累加 FLOPs，但删除中间结果的 memory traffic，用第一个 op 的 tile metadata 和相应 predictor；GEMM+activation 则用 BMM/FC predictor并修正 compute/memory count。它建模的是已知 fusion pattern 的合成 ABI，不是自动预测编译器会不会 fusion。

定位：PDF 8，§4.4 全部两段。 〔[原文截图 E013](#evidence-e013)〕

## 5. 模型级与分布式组合

### 5.1 单卡

Torch.fx 提取 graph，给每个 kernel 标注 op type、tensor dimensions、tile 预测；论文假定主流 framework 中 kernel 在一个 GPU 上顺序执行，因此 per-device latency 是各 kernel latency 之和。

定位：PDF 8，§5 第 1–2 段、图 6；PDF 3，§2.2 `Per-device execution`。 〔[原文截图 E014](#evidence-e014)〕

这忽略多 stream overlap、CUDA graph launch/host overhead、异步 copy 与动态 scheduler；小模型在 H100 上误差较高，作者归因于 library overhead。

### 5.2 单机多 GPU

NeuSight 按用户提供的 parallelism 往图中插：PP 的 send/recv 和 GPipe bubble；DP/TMP 的 ring all-reduce。网络 latency 通过现有机器实测 link utilization，再结合目标 peak link bandwidth缩放。论文版支持单 server NVLink/DGX、GPipe schedule、Megatron TMP，并把计算预测和 network operator 求和。

定位：PDF 8–9，§5.1 全部四段。 〔[原文截图 E015](#evidence-e015)〕

### 5.3 计算通信重叠与排队

| 能力 | 结论 |
|---|---|
| 计算 kernel overlap | 未建模；单卡顺序求和 |
| collective overlap | 论文没有展示 async overlap timeline；网络 op 插入 graph 后聚合 |
| PP bubble | 支持 GPipe 规则，按 microbatch/GPU/send-recv 估计 |
| 在线 serving queue | 不支持；输入是模型图与 batch，不模拟请求到达/scheduler/KV allocator |
| 多节点 | 可接 ASTRA-Sim/ns-3；论文给解析外推示例，但无大集群实测验证 |

定位：PDF 8–9，§5；PDF 13，§6.3 `Multi-node distributed execution`。 〔[原文截图 E016](#evidence-e016)〕

## 6. 冷启动与泛化

### 6.1 新 shape/新模型

新模型只要可由支持的五类 op 图覆盖，输入维度可超出训练范围：GPT3-2.7B 的某 BMM 维为 2048，而训练最大 1024，被定义为 OOD model。泛化来自 tile/波次分解与上界，不是模型级 embedding。

定位：PDF 10，§6.2 第 1 段；表 5。 〔[原文截图 E017](#evidence-e017)〕

### 6.2 新 GPU

不需在目标 GPU 上跑目标模型，但需要公开 GPU 规格和一个 tile-size estimate。论文声称 memory/L2/peak 信息通常在新品发布附近公开；H100/L4/A100-80GB 被作为 OOD GPU。

定位：PDF 7，§4.3 `GPU features` 段；PDF 10，§6.2。 〔[原文截图 E018](#evidence-e018)〕

### 6.3 新 vendor / 新数值类型

- AMD：MI100/MI210 训练、MI250 测试，平均 inference 误差 8.8%、training 15.7%。
- FP16 Tensor Core：通过调整 memory requirement 与 peak FLOPS 输入，不用 FP32 规格；H100 BMM 平均误差 13%。

定位：PDF 11–12，§6.2 `GPU across vendors`、`New numerical type and hardware unit`、图 9/10。 〔[原文截图 E019](#evidence-e019)〕

需要强调：主体训练数据是 FP32。FP16 只做 BMM/Tensor Core 特例，不等于全面支持 mixed precision 图、quantization、routing 数值漂移。

## 7. 误差定义与关键结果

### 7.1 误差

正文把“percentage error”定义为相对实测 latency 的 MAPE；NeuSight 训练 loss 为 sMAPE。正文主要报平均与少量最大误差，没有置信区间或预测不确定度。

定位：PDF 2，§1 最后一段；PDF 10，§6.1 training 段。 〔[原文截图 E020](#evidence-e020)〕

### 7.2 单卡结果

- 6 个模型×8 NVIDIA GPU×多个 batch：inference MAPE 9.7%，training 7.3%；roofline 为 31.2%/31.9%，复训后的 Habitat 为 220.9%/725.8%，线性回归为 61.2%/58.3%。
- OOD GPU 平均误差 8.1%、最大 28.2%；同训练集下 Habitat 平均 724.3%、最大 4529.9%，线性回归平均 94.0%、最大 435.9%。
- OOD BMM/FC：NeuSight 13.8%/13.9%；Habitat 123.2%/799.3%；线性方法 30.0%/152.6%。
- GPT3 on H100 的训练+推理摘要：NeuSight 2.3%，对比 prior 121.4% 与 30.8%。
- fusion：BERT-Large/GPT2-Large 跨 L4/A100/H100 的 fused model 平均 15.7%。

定位：PDF 1 Abstract；PDF 10–12，§6.2、图 7/8、表 6/7。 〔[原文截图 E021](#evidence-e021)〕

### 7.3 分布式结果

4×A100 NVLink 与 4×H100 DGX，GPT2-Large/GPT3-XL，分别单独使用 DP/TP/PP：总体平均误差 7.7%，H100 6.7%，A100 10.5%。摘要另给出 4-GPU 平均 5.4%，与正文 7.7% 口径不完全相同；笔记以正文表 8 总结为主，并标记摘要口径差异。

多节点 GPT-3 从 1 到 3840 nodes 只给预测，论文明确因资源限制未真机验证。

定位：PDF 2 §1 最后一段；PDF 12–13，§6.3、表 8/9。 〔[原文截图 E022](#evidence-e022)〕

## 8. 实现、开源与落地成熟度

**原文事实：** 论文开放 NeuSight，artifact 含源码、脚本、训练/ground-truth 数据、预训练 predictor；约 50GB，使用提供数据约 1 小时复现实验，从头采集约 10 小时；论文给出安装与 `gpt3_inference_h100.sh` basic test。

**现状核验（2026-08-06）：** 原 `sitar-lab/NeuSight` 已重定向到 `scai-tech/NeuSight`；公共仓库 MIT，约 8 次提交，含 predictor、Torch.fx tracing、op graph、dataset、ASPLOS 脚本、device/model JSON、标签与结果，无正式 releases。

**成熟度判断：可复现实验原型（中）。** artifact 完整度是四篇中最高之一，适合拿来建立第一版 cost-model pipeline；但代码覆盖的是论文定义的 op 集/模型图和规格文件，在线 serving、现代 fused attention/MoE、动态图与大规模通信需要重新工程化。

定位：PDF 13–14，Conclusion、Artifact Appendix A.1–A.6；GitHub README。 〔[原文截图 E023](#evidence-e023)〕

## 9. 优点、缺点与边界

### 优点

1. 把 ML 任务从直接 latency 回归改为有物理语义的 utilization 回归，OOD 泛化显著强于 Habitat 类 MLP。
2. tile/wave 显式反映 GPU software library 与 SM 并行量化现象。
3. roofline/sigmoid 保证不超过 compute/memory 上界，结果更可解释。
4. 只依赖公开硬件特征，能对暂时拿不到的目标 GPU forecasting。
5. 覆盖 NVIDIA/AMD、训练/推理、FP32/有限 FP16、fusion 和单机多 GPU，并有 artifact。

### 缺点

1. tile size 的最近匹配仍是核心经验表；新 kernel family/编译器/CANN 可能无可比邻居。
2. `一个 SM 一次一个 tile`、wave 线性叠加、单卡 kernel 顺序执行都是粗化假设。
3. 只支持五类 predictor，unknown op 一律 memory-bound；attention、MoE GMM、稀疏/分页 KV、quantized kernel 可能不满足。
4. 主体 FP32；mixed precision、量化和 dtype 导致的 kernel path/数值决策未系统建模。
5. 网络模型简单，需先在现有系统测 link utilization；没有 collective arrival、拥塞、重叠和多节点实测。
6. 不模拟在线请求、scheduler、queue、KV state；“inference”主要是静态 first-token graph。
7. MAPE/sMAPE 无不确定度输出，最近邻远距时没有 reject。

## 10. 与录制回放的关系

### 10.1 最适合放在 Physical Binding 的成本预测器

NeuSight 应被视为：

```text
Target Operator ABI + target hardware features
  -> tile/tiling regime
  -> bounded utilization
  -> target node service time
```

它不能生成 `branch/index/state/collective ordinal`，也不能从源 shape 自动恢复目标 execution recipe。先由本项目 Transformer 生成正确目标 ABI，再用 NeuSight 类模型估 cost。

### 10.2 对昇腾迁移的主要障碍

CUDA 的 SM/thread block/tile、kernel-name metadata 与 CUTLASS 规律不能直接映射为 Ascend AICore。昇腾存在 Cube/Vector pipeline、blockDim、tilingData、ND/NZ、L1/UB/L2、workspace、CANN autotune/fusion 和不同 SoC 约束；部分 tiling 信息可能不公开。

### 10.3 昇腾版 NeuSight 方案

1. 建 `AscendTileProfile`：semantic op、CANN op/kernel、logical/storage shape、dtype/quant、format、blockDim、可见 tiling key、workspace、SoC/CANN/torch_npu version。
2. 将 tile 概念抽象为“独立 AICore 工作块”，由 CANN profiler/算子 dump/重复 microbenchmark 反推，而不是依赖 kernel name 字符串。
3. 对 MatMul/Cube、Vector、GMM、attention、quant/fusion 分 predictor；输出不是 raw latency，而是相对对应 compute/HBM/片上带宽上界的 utilization。
4. 表 key 加 `implementation_family + layout + tiling regime`；最近邻距离超过训练覆盖时输出 OOD，降级为实测/保守 roofline，而不是静默外推。
5. 对动态 LLM 加 `valid_extent`、expert/rank counts、effective KV bytes、locality、graph bucket；capacity/padded shape 与有效工作量分开。
6. 将 predicted service time 注入 Distributed Execution Graph；用 stream、collective intent、peer arrival 和依赖做离散事件组合，不能沿静态 op 图简单求和。
7. 用昇腾 source/target 两套 profile 做分层验证：tile/op MAPE、regime accuracy、kernel path、rank-local ABI、collective DAG、端到端 phase latency。

以上为**本项目推断/设计**，不是 NeuSight 已有 NPU 支持。

## 11. 最终评价

NeuSight 是四篇里对“跨未见硬件外推”最有启发的工作：与其让黑盒模型记忆 latency，不如学习物理上限下的利用率，并显式建模并行波次。但其精度仍依赖目标 stack 的 tile/implementation schema。对昇腾录制回放，应该移植“分解问题+物理约束+OOD 检测”的思想，而不是移植 CUDA 方程和 kernel-name 最近邻本身。

<!-- EVIDENCE_SCREENSHOTS:BEGIN -->

## 原文证据截图附录

正文中的 `原文截图 E###` 与本节一一对应。卡片保留原笔记行号和原有页码/章节定位；图片按 PDF 物理页生成。截图用于快速核读，正式引用仍以原论文为准。

<a id="evidence-e001"></a>

<details>
<summary><strong>E001</strong> - 原笔记第 17 行 - PDF p.1, 2, 5, 6, 7, 8</summary>

<p><strong>原定位：</strong> <code>定位：PDF 1–2，Abstract 与 §1 第 5–9 段；PDF 5–8，§4、图 3–6、Eq. (1)–(8)。</code></p>

![E001 - PDF p.1, 2, 5, 6, 7, 8](../evidence_pages/neusight/p001.png)

![E001 - PDF p.1, 2, 5, 6, 7, 8](../evidence_pages/neusight/p002.png)

![E001 - PDF p.1, 2, 5, 6, 7, 8](../evidence_pages/neusight/p005.png)

![E001 - PDF p.1, 2, 5, 6, 7, 8](../evidence_pages/neusight/p006.png)

![E001 - PDF p.1, 2, 5, 6, 7, 8](../evidence_pages/neusight/p007.png)

![E001 - PDF p.1, 2, 5, 6, 7, 8](../evidence_pages/neusight/p008.png)

</details>

<a id="evidence-e002"></a>

<details>
<summary><strong>E002</strong> - 原笔记第 25 行 - PDF p.2, 3, 4</summary>

<p><strong>原定位：</strong> <code>定位：PDF 2，§1 第 1–8 段；PDF 3–4，§3.1–§3.3、图 2。</code></p>

![E002 - PDF p.2, 3, 4](../evidence_pages/neusight/p002.png)

![E002 - PDF p.2, 3, 4](../evidence_pages/neusight/p003.png)

![E002 - PDF p.2, 3, 4](../evidence_pages/neusight/p004.png)

</details>

<a id="evidence-e003"></a>

<details>
<summary><strong>E003</strong> - 原笔记第 34 行 - PDF p.5, 7, 8, 9, 14</summary>

<p><strong>原定位：</strong> <code>定位：PDF 5，§4 第 1 段；PDF 7–9，图 6、§5、§6.1；artifact appendix PDF 14；仓库 README `Tool Inputs`。</code></p>

![E003 - PDF p.5, 7, 8, 9, 14](../evidence_pages/neusight/p005.png)

![E003 - PDF p.5, 7, 8, 9, 14](../evidence_pages/neusight/p007.png)

![E003 - PDF p.5, 7, 8, 9, 14](../evidence_pages/neusight/p008.png)

![E003 - PDF p.5, 7, 8, 9, 14](../evidence_pages/neusight/p009.png)

![E003 - PDF p.5, 7, 8, 9, 14](../evidence_pages/neusight/p014.png)

</details>

<a id="evidence-e004"></a>

<details>
<summary><strong>E004</strong> - 原笔记第 46 行 - PDF p.8, 9, 10, 11, 12, 13</summary>

<p><strong>原定位：</strong> <code>定位：PDF 8，§5；PDF 9，§6.1 `DNN workloads evaluated`；PDF 10–13，§6.2–§6.3。</code></p>

![E004 - PDF p.8, 9, 10, 11, 12, 13](../evidence_pages/neusight/p008.png)

![E004 - PDF p.8, 9, 10, 11, 12, 13](../evidence_pages/neusight/p009.png)

![E004 - PDF p.8, 9, 10, 11, 12, 13](../evidence_pages/neusight/p010.png)

![E004 - PDF p.8, 9, 10, 11, 12, 13](../evidence_pages/neusight/p011.png)

![E004 - PDF p.8, 9, 10, 11, 12, 13](../evidence_pages/neusight/p012.png)

![E004 - PDF p.8, 9, 10, 11, 12, 13](../evidence_pages/neusight/p013.png)

</details>

<a id="evidence-e005"></a>

<details>
<summary><strong>E005</strong> - 原笔记第 64 行 - PDF p.9, 10</summary>

<p><strong>原定位：</strong> <code>定位：PDF 9–10，§6.1 `Generating the training dataset` 及列表。</code></p>

![E005 - PDF p.9, 10](../evidence_pages/neusight/p009.png)

![E005 - PDF p.9, 10](../evidence_pages/neusight/p010.png)

</details>

<a id="evidence-e006"></a>

<details>
<summary><strong>E006</strong> - 原笔记第 70 行 - PDF p.9</summary>

<p><strong>原定位：</strong> <code>定位：PDF 9，§6.1 `Hardware`、表 4。</code></p>

![E006 - PDF p.9](../evidence_pages/neusight/p009.png)

</details>

<a id="evidence-e007"></a>

<details>
<summary><strong>E007</strong> - 原笔记第 85 行 - PDF p.8, 10</summary>

<p><strong>原定位：</strong> <code>定位：PDF 10，§6.1 `Tile size` 段（定位词 `closest match in the database`）；PDF 8，图 6 的 `NeuSight Tile Database`。</code></p>

![E007 - PDF p.8, 10](../evidence_pages/neusight/p008.png)

![E007 - PDF p.8, 10](../evidence_pages/neusight/p010.png)

</details>

<a id="evidence-e008"></a>

<details>
<summary><strong>E008</strong> - 原笔记第 118 行 - PDF p.7, 8</summary>

<p><strong>原定位：</strong> <code>定位：PDF 7–8，§4.3 最后两段、表 3、图 6。</code></p>

![E008 - PDF p.7, 8](../evidence_pages/neusight/p007.png)

![E008 - PDF p.7, 8](../evidence_pages/neusight/p008.png)

</details>

<a id="evidence-e009"></a>

<details>
<summary><strong>E009</strong> - 原笔记第 134 行 - PDF p.6, 7</summary>

<p><strong>原定位：</strong> <code>定位：PDF 6–7，§4.2 `Tile-granularity prediction`、Eq. (2)–(4)、图 4。</code></p>

![E009 - PDF p.6, 7](../evidence_pages/neusight/p006.png)

![E009 - PDF p.6, 7](../evidence_pages/neusight/p007.png)

</details>

<a id="evidence-e010"></a>

<details>
<summary><strong>E010</strong> - 原笔记第 147 行 - PDF p.6, 7</summary>

<p><strong>原定位：</strong> <code>定位：PDF 6，§4.1 `Fundamental performance laws`、Eq. (1)；PDF 7，§4.2 Eq. (5)–(6)。</code></p>

![E010 - PDF p.6, 7](../evidence_pages/neusight/p006.png)

![E010 - PDF p.6, 7](../evidence_pages/neusight/p007.png)

</details>

<a id="evidence-e011"></a>

<details>
<summary><strong>E011</strong> - 原笔记第 160 行 - PDF p.7</summary>

<p><strong>原定位：</strong> <code>定位：PDF 7，§4.2 `Imposing performance laws`、Eq. (7)–(8)；§4.3 第 1–4 段。</code></p>

![E011 - PDF p.7](../evidence_pages/neusight/p007.png)

</details>

<a id="evidence-e012"></a>

<details>
<summary><strong>E012</strong> - 原笔记第 166 行 - PDF p.10</summary>

<p><strong>原定位：</strong> <code>定位：PDF 10，§6.1 `Training the NeuSight predictor`。</code></p>

![E012 - PDF p.10](../evidence_pages/neusight/p010.png)

</details>

<a id="evidence-e013"></a>

<details>
<summary><strong>E013</strong> - 原笔记第 172 行 - PDF p.8</summary>

<p><strong>原定位：</strong> <code>定位：PDF 8，§4.4 全部两段。</code></p>

![E013 - PDF p.8](../evidence_pages/neusight/p008.png)

</details>

<a id="evidence-e014"></a>

<details>
<summary><strong>E014</strong> - 原笔记第 180 行 - PDF p.3, 8</summary>

<p><strong>原定位：</strong> <code>定位：PDF 8，§5 第 1–2 段、图 6；PDF 3，§2.2 `Per-device execution`。</code></p>

![E014 - PDF p.3, 8](../evidence_pages/neusight/p003.png)

![E014 - PDF p.3, 8](../evidence_pages/neusight/p008.png)

</details>

<a id="evidence-e015"></a>

<details>
<summary><strong>E015</strong> - 原笔记第 188 行 - PDF p.8, 9</summary>

<p><strong>原定位：</strong> <code>定位：PDF 8–9，§5.1 全部四段。</code></p>

![E015 - PDF p.8, 9](../evidence_pages/neusight/p008.png)

![E015 - PDF p.8, 9](../evidence_pages/neusight/p009.png)

</details>

<a id="evidence-e016"></a>

<details>
<summary><strong>E016</strong> - 原笔记第 200 行 - PDF p.8, 9, 13</summary>

<p><strong>原定位：</strong> <code>定位：PDF 8–9，§5；PDF 13，§6.3 `Multi-node distributed execution`。</code></p>

![E016 - PDF p.8, 9, 13](../evidence_pages/neusight/p008.png)

![E016 - PDF p.8, 9, 13](../evidence_pages/neusight/p009.png)

![E016 - PDF p.8, 9, 13](../evidence_pages/neusight/p013.png)

</details>

<a id="evidence-e017"></a>

<details>
<summary><strong>E017</strong> - 原笔记第 208 行 - PDF p.10</summary>

<p><strong>原定位：</strong> <code>定位：PDF 10，§6.2 第 1 段；表 5。</code></p>

![E017 - PDF p.10](../evidence_pages/neusight/p010.png)

</details>

<a id="evidence-e018"></a>

<details>
<summary><strong>E018</strong> - 原笔记第 214 行 - PDF p.7, 10</summary>

<p><strong>原定位：</strong> <code>定位：PDF 7，§4.3 `GPU features` 段；PDF 10，§6.2。</code></p>

![E018 - PDF p.7, 10](../evidence_pages/neusight/p007.png)

![E018 - PDF p.7, 10](../evidence_pages/neusight/p010.png)

</details>

<a id="evidence-e019"></a>

<details>
<summary><strong>E019</strong> - 原笔记第 221 行 - PDF p.11, 12</summary>

<p><strong>原定位：</strong> <code>定位：PDF 11–12，§6.2 `GPU across vendors`、`New numerical type and hardware unit`、图 9/10。</code></p>

![E019 - PDF p.11, 12](../evidence_pages/neusight/p011.png)

![E019 - PDF p.11, 12](../evidence_pages/neusight/p012.png)

</details>

<a id="evidence-e020"></a>

<details>
<summary><strong>E020</strong> - 原笔记第 231 行 - PDF p.2, 10</summary>

<p><strong>原定位：</strong> <code>定位：PDF 2，§1 最后一段；PDF 10，§6.1 training 段。</code></p>

![E020 - PDF p.2, 10](../evidence_pages/neusight/p002.png)

![E020 - PDF p.2, 10](../evidence_pages/neusight/p010.png)

</details>

<a id="evidence-e021"></a>

<details>
<summary><strong>E021</strong> - 原笔记第 241 行 - PDF p.1, 10, 11, 12</summary>

<p><strong>原定位：</strong> <code>定位：PDF 1 Abstract；PDF 10–12，§6.2、图 7/8、表 6/7。</code></p>

![E021 - PDF p.1, 10, 11, 12](../evidence_pages/neusight/p001.png)

![E021 - PDF p.1, 10, 11, 12](../evidence_pages/neusight/p010.png)

![E021 - PDF p.1, 10, 11, 12](../evidence_pages/neusight/p011.png)

![E021 - PDF p.1, 10, 11, 12](../evidence_pages/neusight/p012.png)

</details>

<a id="evidence-e022"></a>

<details>
<summary><strong>E022</strong> - 原笔记第 249 行 - PDF p.2, 12, 13</summary>

<p><strong>原定位：</strong> <code>定位：PDF 2 §1 最后一段；PDF 12–13，§6.3、表 8/9。</code></p>

![E022 - PDF p.2, 12, 13](../evidence_pages/neusight/p002.png)

![E022 - PDF p.2, 12, 13](../evidence_pages/neusight/p012.png)

![E022 - PDF p.2, 12, 13](../evidence_pages/neusight/p013.png)

</details>

<a id="evidence-e023"></a>

<details>
<summary><strong>E023</strong> - 原笔记第 259 行 - PDF p.13, 14</summary>

<p><strong>原定位：</strong> <code>定位：PDF 13–14，Conclusion、Artifact Appendix A.1–A.6；GitHub README。</code></p>

![E023 - PDF p.13, 14](../evidence_pages/neusight/p013.png)

![E023 - PDF p.13, 14](../evidence_pages/neusight/p014.png)

</details>

<!-- EVIDENCE_SCREENSHOTS:END -->
