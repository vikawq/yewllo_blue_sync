# SimAI：训练工作负载、计算与通信网络的统一仿真

> 证据截图说明：正文中的 `原文截图 E###` 可跳转到文末证据卡片。截图按 PDF 物理页码生成；原有章节、图表、算法和段落定位保持不变。


## 1. 论文身份与页码约定

- 正式题名：*SimAI: Unifying Architecture Design and Performance Tuning for Large-Scale Large Language Model Training with Scalability and Precision*。
- 作者：Xizheng Wang 等；USENIX NSDI 2025，正式印刷页 541–558。
- 原文：[USENIX 论文页](https://www.usenix.org/conference/nsdi25/presentation/wang-xizheng-simai)、[正式 PDF](https://www.usenix.org/system/files/nsdi25-wang-xizheng-simai.pdf)、[代码仓库](https://github.com/aliyun/SimAI)。
- 页码：封面为 PDF p.1；正文从 PDF p.2/印刷 p.541 开始，故本文同时写作“PDF p.X / 印刷 p.Y”。段落序号按对应小节内自然段计。 〔[原文截图 E001](#evidence-e001)〕

## 2. 一句话结论

**原文事实**：SimAI 把框架工作负载生成、GPU/模块计算成本、NCCL 集合通信逻辑、包/RDMA 网络和离散事件执行放到一条仿真链中，报告端到端平均偏差 1.9%。它是本路线中覆盖层次最完整、生产落地证据最强的一篇，但并不执行训练数值，也没有还原数据依赖状态；计算仍以实测查表/粗粒度外推为主，专家路由还假定均衡。（PDF p.2/印刷 p.541，Abstract；PDF p.5–9/印刷 p.544–548，§3） 〔[原文截图 E002](#evidence-e002)〕

## 3. 问题、目标与假设

**原文事实**：作者认为单独的框架分析器、GPU 计算估计器或网络模拟器无法覆盖大模型训练中工作负载、计算和通信之间的相互作用，因而提出四个目标：workload、computation、communication、simulation speed。（PDF p.3–4/印刷 p.542–543，§1 第3–6段及 §2.3） 〔[原文截图 E003](#evidence-e003)〕

**原文事实**：其“统一”边界是性能时序，而非训练功能正确性。SimCCL 不验证传输数据；MoE/EP 的 gating 结果被假定为在专家间均匀分布。（PDF p.8/印刷 p.547，§3.4 “Supporting expert parallelism”及其后段落） 〔[原文截图 E004](#evidence-e004)〕

**归纳**：这是一套“执行驱动生成工作负载 + 测量/外推计算成本 + 通信库复刻 + 网络 DES”的性能仿真器，不是录下源机真实 trace 后逐事件重放，也不是 GPU 指令级或训练数值仿真。

## 4. 方法与框架

### 4.1 总体架构

**原文事实**：图 1 将系统分为 SimAI-WG（workload generator）、Execution Engine、SimAI-CP（computation profiler/model）和 SimAI-CM（communication model）。WG 生成模型子模块以及 collective/P2P 与依赖；CP 提供模块或 kernel 计算时间；CM 把集合通信展开成 P2P 流；Execution Engine 组合时间线。（PDF p.5/印刷 p.544，§3.1 第1–4段，Fig. 1） 〔[原文截图 E005](#evidence-e005)〕

### 4.2 工作负载生成：在单机上“劫持”训练框架

**原文事实**：SimAI-WG 在单个 host 上运行 Megatron-LM/DeepSpeed，通过伪造 world size、rank 和拓扑等运行环境让框架走到目标并行路径，同时跳过真实 NCCL 数据通信；流水并行需要逐 rank 配置。输出保留算法子模块、collective/P2P 操作及其依赖。（PDF p.5–6/印刷 p.544–545，§3.2.1 第1–4段，Fig. 2） 〔[原文截图 E006](#evidence-e006)〕

**原文事实**：作者用 1024 GPU 集群上的 Nsight 结果核对生成依赖。对 TP、PP、EP，论文认为通信模式和流量在给定配置下相对固定；对 DP 则随规模变化。小于 1 KB 的元数据通信与 barrier 被视为可忽略。（PDF p.6/印刷 p.545，§3.2.1 “Accuracy validation”及其后两段） 〔[原文截图 E007](#evidence-e007)〕

**设计推断**：WG 最接近“目标框架执行路径提取器”。它可以生成一份目标 Recipe，却没有记录真实样本、随机数、动态 MoE 路由、内存地址或 kernel 选择；不能把它等同于 Observation Ledger 或源机 trace replay。

### 4.3 计算成本：实测数据库、模块拆解与未知 GPU 外推

**原文事实**：对于已有 GPU，SimAI-CP 在 GPU 上测量子模块执行时间，写入按 GPU/配置组织的 operation database；论文还给出 module-to-kernel converter，把子模块拆成更细 kernel 后分别测量。（PDF p.6–7/印刷 p.545–546，§3.3 “Supporting existing GPUs”，Table 4） 〔[原文截图 E008](#evidence-e008)〕

**原文事实**：对未发布 GPU，作者把 kernel 分成 compute-bound 与 memory-bandwidth-bound，按已知与新 GPU 的 FLOPS 或带宽比缩放旧时间，并建议选择相近架构作基线。论文同时承认朴素跨 GPU 缩放偏差可达 25.1%。（PDF p.7/印刷 p.546，§3.3 “Supporting unreleased GPUs”，两条公式） 〔[原文截图 E009](#evidence-e009)〕

**证据提示**：版面核查确认论文的 compute-bound 公式原样写成 `Time_Comp_New = FLOPS_Comp_New / FLOPS_Comp_Known × Time_Comp_Known`，memory 公式亦为新/旧带宽比。若把 FLOPS/BW 理解为峰值性能，方向与通常的“性能越高、时间越短”直觉相反；本文不替作者改式，工程采用前必须查代码和实验定义。

### 4.4 通信：复刻 NCCL 决策并进入包/RDMA 网络

**原文事实**：SimCCL 是修改版 NCCL，拦截初始化、拓扑发现、channel、算法选择和 collective API，提取 sender、receiver、message size、route 等，把 collective 转成 P2P flow 交给网络模拟。（PDF p.7–8/印刷 p.546–547，§3.4 前4段，Fig. 3） 〔[原文截图 E010](#evidence-e010)〕

**原文事实**：作者称关键修改为 572 行、总修改超过 10K 行；适配新 NCCL/其他 CCL 需要重复这项工作。附录 Table 6 列出支持的 NCCL 环境变量，并明确有 4 个与 adaptive routing/SHARP 有关的变量未支持。（PDF p.8/印刷 p.547，§3.4 “Reproducing NCCL’s key procedures”；PDF p.17–19/印刷 p.556–558，Appendix A，Table 6） 〔[原文截图 E011](#evidence-e011)〕

**原文事实**：网络层使用 NS-3；Execution Engine 是 DES，并以 UNISON 的并行离散事件技术、多线程和 lock-free global variables 加速。lock-free 方案相对单线程最高 23×，相对已有多线程约 15%。（PDF p.8–9/印刷 p.547–548，§3.5，Fig. 4） 〔[原文截图 E012](#evidence-e012)〕

### 4.5 调度、依赖、重叠、内存

- **依赖/调度（原文事实）**：依赖来自框架路径并进入 DES；网络事件通过通信回调完成。论文没有给出一个独立、可配置的训练 runtime scheduler 模型。（PDF p.5–9/印刷 p.544–548，§3.1–3.5） 〔[原文截图 E013](#evidence-e013)〕
- **计算通信重叠（归纳）**：由依赖、不同资源时间线及网络完成事件自然形成；论文未提供独立的经验 overlap ratio 参数。
- **内存（证据缺口）**：正文没有形成类似 ASTRA-sim 2.0/Proteus 的显式内存层、峰值内存生命周期或 OOM 判定模型，不能因“全栈”一词推定已覆盖。

## 5. 校准、精度、规模与实验

**原文事实**：真实平台为最多 128 台服务器、每台 8 GPU，即 1024 GPU；包括 A100+4×ConnectX-6 和 H100+8×ConnectX-7，多 rail RoCEv2。（PDF p.9/印刷 p.548，§4.1 第1–2段） 〔[原文截图 E014](#evidence-e014)〕

**原文事实**：通信微基准中，SimAI 在 A100/H100 平台的平均偏差分别为 3.9%/2.3%，ASTRA-sim 对照为 74.8%/51.7%；作者把小消息误差归因于未模拟 libibverbs/NIC pipeline 等。（PDF p.9/印刷 p.548，§4.2，Fig. 5–7） 〔[原文截图 E015](#evidence-e015)〕

**原文事实**：已有 GPU 的 SimAI-CP 误差为 0.5%–3.1%；跨模型的 GPU 外推误差约 13%–15%。对照 ASTRA-sim 在 H100/A100/H20 上的相关误差为 49.8%/117.9%/224%。（PDF p.9–10/印刷 p.548–549，§4.3，Fig. 8） 〔[原文截图 E016](#evidence-e016)〕

**原文事实**：端到端验证扩展至 1024 GPU，SimAI 的偏差均低于 4%，文中概括为最高 3.9%、相对 ASTRA-sim 精度提高 36.1×。（PDF p.10/印刷 p.549，§4.4，Fig. 9） 〔[原文截图 E017](#evidence-e017)〕

**边界**：这些结果是在作者选定的模型、框架版本、网络栈和内部/商用集群上校准；不自动证明换 CCL、MoE 负载失衡、不同 NIC pipeline 或新 GPU 架构仍有同样误差。

## 6. What-if 与生产落地

**原文事实**：论文以 SimAI 比较主机网络带宽：H100 从 200 提升到 400 Gbps 得到 19% 性能提升，H20 从 100 到 200 Gbps 为 11%、200 到 400 Gbps 为 6%；作者称相关设计被生产采用。（PDF p.10–11/印刷 p.549–550，§5.1，Fig. 10） 〔[原文截图 E018](#evidence-e018)〕

**原文事实**：并行策略案例中，GPT-3 13B、LLaMA 65B、GPT-3 175B 在 8-GPU host 的最优 TP 分别为 4、8、8。（PDF p.11–12/印刷 p.550–551，§5.2，Fig. 11） 〔[原文截图 E019](#evidence-e019)〕

**原文事实**：作者还描述了基于 Kubernetes 的 simulation-as-a-service；工作负载与 GPU stack 解耦后，任务可部署在无 GPU 的服务节点。（PDF p.12–13/印刷 p.551–552，§6.2） 〔[原文截图 E020](#evidence-e020)〕

**归纳**：在六篇中，SimAI 的“硬件采购/网络设计已被采用”和线上服务是最强的落地证据；但公开代码能否独立重现论文内部 benchmark suite、未发布 GPU 参数与 1024 GPU 实验，仍需逐项核验。

## 7. 优点、缺点与适用边界

### 优点

1. 跨框架、计算、CCL 和网络，避免把 collective 当作固定 `size/bw`。
2. 框架驱动生成依赖，比手写静态 workload 更接近真实控制路径。
3. 包/RDMA 网络、NCCL 决策与 1024 GPU 校准形成较完整证据链。
4. 支持硬件、网络、并行策略 what-if，并有明确生产服务与设计采用案例。

### 缺点/边界

1. 不做数值执行和数据正确性；EP 均衡假设直接排除了重要动态行为。
2. 已有 GPU 仍依赖 profiling；未知 GPU 只用 FLOPS/BW 分类缩放，且论文公式方向值得复核。
3. 复刻 NCCL 的工程耦合重，迁移至 HCCL 并非换一个配置文件。
4. 小消息 NIC/software pipeline、SHARP/自适应路由和显式内存生命周期没有完整覆盖。
5. 工作负载生成是目标执行路径，不是源 trace 的语义无损回放。

## 8. 对录制回放五层架构的映射

| 五层 | SimAI 对应物 | 判断 |
|---|---|---|
| Execution Recipe | WG 输出的子模块、通信与依赖 | 较强，但缺数据依赖状态、随机性、动态路由 |
| Physical Binding | rank/world/topology、NCCL channel/算法/route | 强，尤其通信；kernel/layout/tiling 绑定仍弱 |
| Observation Ledger | benchmark/Nsight/operation DB 的测量 | 有测量，无统一的事件来源、置信度和版本 ledger |
| Cost Model | module/kernel DB 与 GPU 外推，SimCCL/网络 | 强，但计算 OOD 外推粗糙 |
| Event Runtime | DES + NS-3 + UNISON PDES | 强；显式训练调度/内存语义不足 |

**归纳**：SimAI 最适合作为目标侧仿真执行器和通信 binding 的参考，而不是把整套系统直接定义成“录制回放”。V0.8 的关键约束仍成立：源端 duration 只能进 Observation/Cost Model，不能变成目标 Recipe；CCL ordinal、group、split、route 必须有可解释绑定。

## 9. Ascend/CANN/HCCL 落地启示

1. **Recipe 生成**：可借鉴单机 mock 世界规模的办法，让目标训练框架走到目标 TP/PP/DP/EP 分支；必须额外记录动态 shape、随机种子、MoE expert/token 路由和 micro-batch 决策。
2. **SimHCCL 而非换名**：要获得 SimCCL 等级的精度，需记录/复现 HCCL communicator group、collective ordinal、算法/协议、rank pair、chunk/split 和拓扑路由；若 HCCL 内部不可插桩，应保留“实测黑盒 CCL 成本模型”和“显式 flow 后端”两档保真度。
3. **CANN 计算键**：operation DB 至少按 SoC、CANN/算子 ABI、shape、dtype、layout、融合、tiling、stream、编译选项和频率状态分层；不能只用模型子模块名。
4. **Observation Ledger**：msprof/Ascend PyTorch Profiler 事实必须带 rank/device/stream/task/op/kernel、时间基准、版本和来源，不让实测 duration 污染可移植 Recipe。
5. **先验证失衡**：EP、动态序列长度和拥塞是最容易击穿 SimAI 假设的场景，应作为 Ascend 原型首批反例，而不是沿用均衡假设。

## 10. 实现与复现成熟度

- **原文事实**：论文明确描述生产服务和设计采用，并提供公开仓库。
- **当前核验**：仓库公开可访问；本轮未复跑构建、未验证论文内部数据与 1024 GPU 校准集是否全部公开。
- **成熟度判断（归纳）**：研究原型以上、生产内部使用证据强；公开可复现实验仍受内部 benchmark、硬件和网络环境约束。

<!-- EVIDENCE_SCREENSHOTS:BEGIN -->

## 原文证据截图附录

正文中的 `原文截图 E###` 与本节证据卡片一一对应。卡片保留原笔记行号和原有页码/章节定位，并跳转到后面的页图；每个物理页在本篇笔记中只展示一次。截图用于快速核读，正式引用仍以原论文为准。

<a id="evidence-e001"></a>

<details>
<summary><strong>E001</strong> - 原笔记第 11 行 - PDF p.1, 2</summary>

<p><strong>原定位：</strong> <code>- 页码：封面为 PDF p.1；正文从 PDF p.2/印刷 p.541 开始，故本文同时写作“PDF p.X / 印刷 p.Y”。段落序号按对应小节内自然段计。</code></p>

<p><strong>页图：</strong> <a href="#source-page-p001">PDF p.1</a> · <a href="#source-page-p002">PDF p.2</a></p>

</details>

<a id="evidence-e002"></a>

<details>
<summary><strong>E002</strong> - 原笔记第 15 行 - PDF p.2, 5, 6, 7, 8, 9</summary>

<p><strong>原定位：</strong> <code>**原文事实**：SimAI 把框架工作负载生成、GPU/模块计算成本、NCCL 集合通信逻辑、包/RDMA 网络和离散事件执行放到一条仿真链中，报告端到端平均偏差 1.9%。它是本路线中覆盖层次最完整、生产落地证据最强的一篇，但并不执行训练数值，也没有还原数据依赖状态；计算仍以实测查表/粗粒度外推为主，专家路由还假定均衡。（PDF p.2/印刷 p.541，Abstract；PDF p.5–9/印刷 p.544–548，§3）</code></p>

<p><strong>页图：</strong> <a href="#source-page-p002">PDF p.2</a> · <a href="#source-page-p005">PDF p.5</a> · <a href="#source-page-p006">PDF p.6</a> · <a href="#source-page-p007">PDF p.7</a> · <a href="#source-page-p008">PDF p.8</a> · <a href="#source-page-p009">PDF p.9</a></p>

</details>

<a id="evidence-e003"></a>

<details>
<summary><strong>E003</strong> - 原笔记第 19 行 - PDF p.3, 4</summary>

<p><strong>原定位：</strong> <code>**原文事实**：作者认为单独的框架分析器、GPU 计算估计器或网络模拟器无法覆盖大模型训练中工作负载、计算和通信之间的相互作用，因而提出四个目标：workload、computation、communication、simulation speed。（PDF p.3–4/印刷 p.542–543，§1 第3–6段及 §2.3）</code></p>

<p><strong>页图：</strong> <a href="#source-page-p003">PDF p.3</a> · <a href="#source-page-p004">PDF p.4</a></p>

</details>

<a id="evidence-e004"></a>

<details>
<summary><strong>E004</strong> - 原笔记第 21 行 - PDF p.8</summary>

<p><strong>原定位：</strong> <code>**原文事实**：其“统一”边界是性能时序，而非训练功能正确性。SimCCL 不验证传输数据；MoE/EP 的 gating 结果被假定为在专家间均匀分布。（PDF p.8/印刷 p.547，§3.4 “Supporting expert parallelism”及其后段落）</code></p>

<p><strong>页图：</strong> <a href="#source-page-p008">PDF p.8</a></p>

</details>

<a id="evidence-e005"></a>

<details>
<summary><strong>E005</strong> - 原笔记第 29 行 - PDF p.5</summary>

<p><strong>原定位：</strong> <code>**原文事实**：图 1 将系统分为 SimAI-WG（workload generator）、Execution Engine、SimAI-CP（computation profiler/model）和 SimAI-CM（communication model）。WG 生成模型子模块以及 collective/P2P 与依赖；CP 提供模块或 kernel 计算时间；CM 把集合通信展开成 P2P 流；Execution Engine 组合时间线。（PDF p.5/印刷 p.544，§3.1 第1–4段，Fig. 1）</code></p>

<p><strong>页图：</strong> <a href="#source-page-p005">PDF p.5</a></p>

</details>

<a id="evidence-e006"></a>

<details>
<summary><strong>E006</strong> - 原笔记第 33 行 - PDF p.5, 6</summary>

<p><strong>原定位：</strong> <code>**原文事实**：SimAI-WG 在单个 host 上运行 Megatron-LM/DeepSpeed，通过伪造 world size、rank 和拓扑等运行环境让框架走到目标并行路径，同时跳过真实 NCCL 数据通信；流水并行需要逐 rank 配置。输出保留算法子模块、collective/P2P 操作及其依赖。（PDF p.5–6/印刷 p.544–545，§3.2.1 第1–4段，Fig. 2）</code></p>

<p><strong>页图：</strong> <a href="#source-page-p005">PDF p.5</a> · <a href="#source-page-p006">PDF p.6</a></p>

</details>

<a id="evidence-e007"></a>

<details>
<summary><strong>E007</strong> - 原笔记第 35 行 - PDF p.6</summary>

<p><strong>原定位：</strong> <code>**原文事实**：作者用 1024 GPU 集群上的 Nsight 结果核对生成依赖。对 TP、PP、EP，论文认为通信模式和流量在给定配置下相对固定；对 DP 则随规模变化。小于 1 KB 的元数据通信与 barrier 被视为可忽略。（PDF p.6/印刷 p.545，§3.2.1 “Accuracy validation”及其后两段）</code></p>

<p><strong>页图：</strong> <a href="#source-page-p006">PDF p.6</a></p>

</details>

<a id="evidence-e008"></a>

<details>
<summary><strong>E008</strong> - 原笔记第 41 行 - PDF p.6, 7</summary>

<p><strong>原定位：</strong> <code>**原文事实**：对于已有 GPU，SimAI-CP 在 GPU 上测量子模块执行时间，写入按 GPU/配置组织的 operation database；论文还给出 module-to-kernel converter，把子模块拆成更细 kernel 后分别测量。（PDF p.6–7/印刷 p.545–546，§3.3 “Supporting existing GPUs”，Table 4）</code></p>

<p><strong>页图：</strong> <a href="#source-page-p006">PDF p.6</a> · <a href="#source-page-p007">PDF p.7</a></p>

</details>

<a id="evidence-e009"></a>

<details>
<summary><strong>E009</strong> - 原笔记第 43 行 - PDF p.7</summary>

<p><strong>原定位：</strong> <code>**原文事实**：对未发布 GPU，作者把 kernel 分成 compute-bound 与 memory-bandwidth-bound，按已知与新 GPU 的 FLOPS 或带宽比缩放旧时间，并建议选择相近架构作基线。论文同时承认朴素跨 GPU 缩放偏差可达 25.1%。（PDF p.7/印刷 p.546，§3.3 “Supporting unreleased GPUs”，两条公式）</code></p>

<p><strong>页图：</strong> <a href="#source-page-p007">PDF p.7</a></p>

</details>

<a id="evidence-e010"></a>

<details>
<summary><strong>E010</strong> - 原笔记第 49 行 - PDF p.7, 8</summary>

<p><strong>原定位：</strong> <code>**原文事实**：SimCCL 是修改版 NCCL，拦截初始化、拓扑发现、channel、算法选择和 collective API，提取 sender、receiver、message size、route 等，把 collective 转成 P2P flow 交给网络模拟。（PDF p.7–8/印刷 p.546–547，§3.4 前4段，Fig. 3）</code></p>

<p><strong>页图：</strong> <a href="#source-page-p007">PDF p.7</a> · <a href="#source-page-p008">PDF p.8</a></p>

</details>

<a id="evidence-e011"></a>

<details>
<summary><strong>E011</strong> - 原笔记第 51 行 - PDF p.8, 17, 18, 19</summary>

<p><strong>原定位：</strong> <code>**原文事实**：作者称关键修改为 572 行、总修改超过 10K 行；适配新 NCCL/其他 CCL 需要重复这项工作。附录 Table 6 列出支持的 NCCL 环境变量，并明确有 4 个与 adaptive routing/SHARP 有关的变量未支持。（PDF p.8/印刷 p.547，§3.4 “Reproducing NCCL’s key procedures”；PDF p.17–19/印刷 p.556–558，Appendix A，Table 6）</code></p>

<p><strong>页图：</strong> <a href="#source-page-p008">PDF p.8</a> · <a href="#source-page-p017">PDF p.17</a> · <a href="#source-page-p018">PDF p.18</a> · <a href="#source-page-p019">PDF p.19</a></p>

</details>

<a id="evidence-e012"></a>

<details>
<summary><strong>E012</strong> - 原笔记第 53 行 - PDF p.8, 9</summary>

<p><strong>原定位：</strong> <code>**原文事实**：网络层使用 NS-3；Execution Engine 是 DES，并以 UNISON 的并行离散事件技术、多线程和 lock-free global variables 加速。lock-free 方案相对单线程最高 23×，相对已有多线程约 15%。（PDF p.8–9/印刷 p.547–548，§3.5，Fig. 4）</code></p>

<p><strong>页图：</strong> <a href="#source-page-p008">PDF p.8</a> · <a href="#source-page-p009">PDF p.9</a></p>

</details>

<a id="evidence-e013"></a>

<details>
<summary><strong>E013</strong> - 原笔记第 57 行 - PDF p.5, 6, 7, 8, 9</summary>

<p><strong>原定位：</strong> <code>- **依赖/调度（原文事实）**：依赖来自框架路径并进入 DES；网络事件通过通信回调完成。论文没有给出一个独立、可配置的训练 runtime scheduler 模型。（PDF p.5–9/印刷 p.544–548，§3.1–3.5）</code></p>

<p><strong>页图：</strong> <a href="#source-page-p005">PDF p.5</a> · <a href="#source-page-p006">PDF p.6</a> · <a href="#source-page-p007">PDF p.7</a> · <a href="#source-page-p008">PDF p.8</a> · <a href="#source-page-p009">PDF p.9</a></p>

</details>

<a id="evidence-e014"></a>

<details>
<summary><strong>E014</strong> - 原笔记第 63 行 - PDF p.9</summary>

<p><strong>原定位：</strong> <code>**原文事实**：真实平台为最多 128 台服务器、每台 8 GPU，即 1024 GPU；包括 A100+4×ConnectX-6 和 H100+8×ConnectX-7，多 rail RoCEv2。（PDF p.9/印刷 p.548，§4.1 第1–2段）</code></p>

<p><strong>页图：</strong> <a href="#source-page-p009">PDF p.9</a></p>

</details>

<a id="evidence-e015"></a>

<details>
<summary><strong>E015</strong> - 原笔记第 65 行 - PDF p.9</summary>

<p><strong>原定位：</strong> <code>**原文事实**：通信微基准中，SimAI 在 A100/H100 平台的平均偏差分别为 3.9%/2.3%，ASTRA-sim 对照为 74.8%/51.7%；作者把小消息误差归因于未模拟 libibverbs/NIC pipeline 等。（PDF p.9/印刷 p.548，§4.2，Fig. 5–7）</code></p>

<p><strong>页图：</strong> <a href="#source-page-p009">PDF p.9</a></p>

</details>

<a id="evidence-e016"></a>

<details>
<summary><strong>E016</strong> - 原笔记第 67 行 - PDF p.9, 10</summary>

<p><strong>原定位：</strong> <code>**原文事实**：已有 GPU 的 SimAI-CP 误差为 0.5%–3.1%；跨模型的 GPU 外推误差约 13%–15%。对照 ASTRA-sim 在 H100/A100/H20 上的相关误差为 49.8%/117.9%/224%。（PDF p.9–10/印刷 p.548–549，§4.3，Fig. 8）</code></p>

<p><strong>页图：</strong> <a href="#source-page-p009">PDF p.9</a> · <a href="#source-page-p010">PDF p.10</a></p>

</details>

<a id="evidence-e017"></a>

<details>
<summary><strong>E017</strong> - 原笔记第 69 行 - PDF p.10</summary>

<p><strong>原定位：</strong> <code>**原文事实**：端到端验证扩展至 1024 GPU，SimAI 的偏差均低于 4%，文中概括为最高 3.9%、相对 ASTRA-sim 精度提高 36.1×。（PDF p.10/印刷 p.549，§4.4，Fig. 9）</code></p>

<p><strong>页图：</strong> <a href="#source-page-p010">PDF p.10</a></p>

</details>

<a id="evidence-e018"></a>

<details>
<summary><strong>E018</strong> - 原笔记第 75 行 - PDF p.10, 11</summary>

<p><strong>原定位：</strong> <code>**原文事实**：论文以 SimAI 比较主机网络带宽：H100 从 200 提升到 400 Gbps 得到 19% 性能提升，H20 从 100 到 200 Gbps 为 11%、200 到 400 Gbps 为 6%；作者称相关设计被生产采用。（PDF p.10–11/印刷 p.549–550，§5.1，Fig. 10）</code></p>

<p><strong>页图：</strong> <a href="#source-page-p010">PDF p.10</a> · <a href="#source-page-p011">PDF p.11</a></p>

</details>

<a id="evidence-e019"></a>

<details>
<summary><strong>E019</strong> - 原笔记第 77 行 - PDF p.11, 12</summary>

<p><strong>原定位：</strong> <code>**原文事实**：并行策略案例中，GPT-3 13B、LLaMA 65B、GPT-3 175B 在 8-GPU host 的最优 TP 分别为 4、8、8。（PDF p.11–12/印刷 p.550–551，§5.2，Fig. 11）</code></p>

<p><strong>页图：</strong> <a href="#source-page-p011">PDF p.11</a> · <a href="#source-page-p012">PDF p.12</a></p>

</details>

<a id="evidence-e020"></a>

<details>
<summary><strong>E020</strong> - 原笔记第 79 行 - PDF p.12, 13</summary>

<p><strong>原定位：</strong> <code>**原文事实**：作者还描述了基于 Kubernetes 的 simulation-as-a-service；工作负载与 GPU stack 解耦后，任务可部署在无 GPU 的服务节点。（PDF p.12–13/印刷 p.551–552，§6.2）</code></p>

<p><strong>页图：</strong> <a href="#source-page-p012">PDF p.12</a> · <a href="#source-page-p013">PDF p.13</a></p>

</details>

## 原文页面图库（按页去重）

同一页可能支撑多个证据点；下面按物理页集中展示，每个截图文件只嵌入一次。

<a id="source-page-p001"></a>

<details>
<summary><strong>PDF p.1</strong> - 被 E001 引用</summary>

![PDF p.1](../evidence_pages/simai/p001.png)

</details>

<a id="source-page-p002"></a>

<details>
<summary><strong>PDF p.2</strong> - 被 E001、E002 引用</summary>

![PDF p.2](../evidence_pages/simai/p002.png)

</details>

<a id="source-page-p003"></a>

<details>
<summary><strong>PDF p.3</strong> - 被 E003 引用</summary>

![PDF p.3](../evidence_pages/simai/p003.png)

</details>

<a id="source-page-p004"></a>

<details>
<summary><strong>PDF p.4</strong> - 被 E003 引用</summary>

![PDF p.4](../evidence_pages/simai/p004.png)

</details>

<a id="source-page-p005"></a>

<details>
<summary><strong>PDF p.5</strong> - 被 E002、E005、E006、E013 引用</summary>

![PDF p.5](../evidence_pages/simai/p005.png)

</details>

<a id="source-page-p006"></a>

<details>
<summary><strong>PDF p.6</strong> - 被 E002、E006、E007、E008、E013 引用</summary>

![PDF p.6](../evidence_pages/simai/p006.png)

</details>

<a id="source-page-p007"></a>

<details>
<summary><strong>PDF p.7</strong> - 被 E002、E008、E009、E010、E013 引用</summary>

![PDF p.7](../evidence_pages/simai/p007.png)

</details>

<a id="source-page-p008"></a>

<details>
<summary><strong>PDF p.8</strong> - 被 E002、E004、E010、E011、E012、E013 引用</summary>

![PDF p.8](../evidence_pages/simai/p008.png)

</details>

<a id="source-page-p009"></a>

<details>
<summary><strong>PDF p.9</strong> - 被 E002、E012、E013、E014、E015、E016 引用</summary>

![PDF p.9](../evidence_pages/simai/p009.png)

</details>

<a id="source-page-p010"></a>

<details>
<summary><strong>PDF p.10</strong> - 被 E016、E017、E018 引用</summary>

![PDF p.10](../evidence_pages/simai/p010.png)

</details>

<a id="source-page-p011"></a>

<details>
<summary><strong>PDF p.11</strong> - 被 E018、E019 引用</summary>

![PDF p.11](../evidence_pages/simai/p011.png)

</details>

<a id="source-page-p012"></a>

<details>
<summary><strong>PDF p.12</strong> - 被 E019、E020 引用</summary>

![PDF p.12](../evidence_pages/simai/p012.png)

</details>

<a id="source-page-p013"></a>

<details>
<summary><strong>PDF p.13</strong> - 被 E020 引用</summary>

![PDF p.13](../evidence_pages/simai/p013.png)

</details>

<a id="source-page-p017"></a>

<details>
<summary><strong>PDF p.17</strong> - 被 E011 引用</summary>

![PDF p.17](../evidence_pages/simai/p017.png)

</details>

<a id="source-page-p018"></a>

<details>
<summary><strong>PDF p.18</strong> - 被 E011 引用</summary>

![PDF p.18](../evidence_pages/simai/p018.png)

</details>

<a id="source-page-p019"></a>

<details>
<summary><strong>PDF p.19</strong> - 被 E011 引用</summary>

![PDF p.19](../evidence_pages/simai/p019.png)

</details>

<!-- EVIDENCE_SCREENSHOTS:END -->
