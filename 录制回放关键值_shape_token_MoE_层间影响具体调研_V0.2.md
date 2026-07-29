# 录制回放中的关键 index、shape、token、MoE 与层间影响（V0.2）

> 日期：2026-07-29  
> 目标：把 V0.1 中较抽象的泛化框架，落到可以观察、录制、回放和验证的具体张量上。  
> 重点模型：DeepSeek-V4 Pro/Flash、GLM-5.2、MiniMax-M3、Qwen3.7 Max/Plus。Qwen3.7 采用官方公开的服务参数；未公开的内部结构和权重 shape 不做猜测。  
> 范围：以推理为主，包括 Prefill、Decode、KV Cache、稀疏注意力、MoE、MTP、量化和多机多卡部署。

---

## 0. 先直接回答目前关心的五个问题

### 0.1 推理过程中计算出来的 index 会不会影响计算路径？

**会，但要区分四种影响。**

| 类型 | 例子 | shape 是否变化 | 算子集合是否变化 | 性能是否可能变化 |
|---|---|---:|---:|---:|
| 只改变读写位置 | `input_ids`、正常的 `block_table` | 通常不变 | 通常不变 | 会，取决于 cache/locality，但一般较弱 |
| 改变动态工作量 | MoE `topk_indices`、每个专家的 token 数 | 内部 ragged shape 会变 | 算子类型可不变 | 会，且可能很大 |
| 直接决定是否执行 | 无效 `slot_mapping`、EOS、MTP 接受数、DeepSeek 压缩边界 | 可能变化 | 可能增删分支 | 会 |
| 跨层复用状态 | GLM IndexShare、MiniMax IndexCache | 表面 shape 可不变 | 某些层跳过 indexer | 会 |

最典型的两条链路是：

```text
position_ids 的值
  → 是否到达 KV 压缩边界
  → 是否写压缩 KV
  → cache 写入次数和后续可读取位置变化
```

```text
hidden_states 的值
  → router logits
  → MoE topk expert index
  → 每个 expert 收到的 token 数 n_e
  → AllToAllV send/recv counts
  → grouped GEMM 的实际 M 维
  → 时延和后续 hidden_states 变化
```

因此，录制时不能只记 index 的 `shape/dtype/min/max`。对影响路径的离散 index，**要么保存完整内容，要么保存能够逐元素确定性重建它的输入、状态和算法版本**。

### 0.2 所有涉及到的 shape，包括权重 shape，是否都要考虑？

**要考虑，但不能只维护一张“模型逻辑 shape 表”。同一个权重至少可能有四种 shape：**

1. checkpoint 中的全局逻辑 shape；
2. TP/EP/PP 分片后的单 rank shape；
3. INT4/FP8/MXFP8 等量化后的 packed weight 与 scale shape；
4. 算子执行前重新排布、补齐后的 kernel shape。

运行时也不只有 `[token, hidden]`，还包括：

- 输入、位置、mask、请求边界；
- Q/K/V、KV Cache、block table、slot mapping；
- MoE router、TopK、per-expert ragged token；
- 稀疏注意力的 token/block index；
- 通信 send/recv counts 和通信 buffer；
- 图 bucket 补齐后的 token 数；
- 量化 scale、workspace、临时排序/反排序 buffer。

所以第一阶段不应靠人工穷举每一个临时 tensor，而应：

- 人工维护“必须理解语义”的 shape 表；
- 从 checkpoint 自动导出全部权重 manifest；
- 从真实运行自动抓取全部算子输入/输出描述符；
- 把逻辑、局部、物理、执行四种 shape 分开保存。

### 0.3 token 的非零值、零值、重复值会不会影响？

结论分四种情况：

1. **token id 的数值是不是 0，本身一般不会让 Dense MatMul 少算。**Embedding 是查表，`0` 只是第 0 行，不代表数值稀疏。
2. **0 可能是特殊 token。**例如 DeepSeek-V4 的 `bos_token_id=0`；GLM-5.2 的 padding id 是 `154820`，不是 0。因此绝不能写成“token id 为 0 就跳过”。
3. **attention mask、slot mapping 中的无效值有控制意义。**被 mask 的 token、负 slot、padding token 可能被跳过或不写 cache。
4. **重复 token 不等于重复计算可复用。**同一个 token 出现在不同 position，经 RoPE 和前序上下文后 hidden state 通常不同；只有启用了 prefix cache，且多个请求具有完全相同的前缀及匹配的缓存状态时，才可能跳过共享前缀的 Prefill。

### 0.4 MoE 是否可能与 token 值相关？

**是，而且是本课题中最重要的 value-dependent shape 来源之一。**

MoE 不是直接拿 `input_id` 做路由，而是使用当前层的 `hidden_states`。但 `hidden_states` 来自 token、position、前序层、量化误差和 cache 状态，因此最终仍然与 token 内容强相关。

即使两个请求的：

```text
T、H、E、TopK、dtype
```

完全相同，只要路由结果不同，每个专家的 `n_e` 就不同，通信量和 expert GEMM 的实际 shape 就可能不同。

### 0.5 layers 之间会不会相互影响？

**会，至少有四条依赖链。**

1. 当前层输出是下一层输入，数值误差会逐层传播；
2. 每层有自己的 KV Cache、MoE 路由和稀疏注意力 index；
3. 某些模型会跨层共享稀疏 index，例如 GLM-5.2 IndexShare；
4. 层并不一定同构：Dense/MoE、全注意力/稀疏注意力、不同压缩比可能交错出现。

因此，“录一个代表层，然后乘以层数”只适用于已经证明以下内容都相同的层：

```text
层类型 + 输入/执行 shape + 离散 index 分布 + kernel/tiling + 通信模式
```

未证明前，必须逐层记录。

---

## 1. 本文统一符号

| 符号 | 含义 |
|---|---|
| `B` | 当前批次中的请求数 |
| `T` | 本次 forward 的真实 token 总数，通常为各请求 query token 数之和 |
| `T_exec` | graph bucket 或 kernel padding 后实际执行的 token 数，`T_exec >= T` |
| `S_i` | 第 `i` 个请求当前总上下文长度 |
| `Q_i` | 第 `i` 个请求本轮 query token 数 |
| `H` | hidden size |
| `V` | vocabulary size |
| `L` | Transformer 层数 |
| `Nq` | query head 数 |
| `Nkv` | KV head 数 |
| `Dh` | head dimension |
| `I` | Dense MLP intermediate size |
| `E` | routed expert 总数 |
| `K` | 每个 token 选择的专家数 |
| `Ie` | 每个 routed expert 的 intermediate size |
| `n_e` | 某一层中被路由到专家 `e` 的 token assignment 数 |
| `TP/EP/PP/DP` | Tensor/Expert/Pipeline/Data Parallel size |
| `P` | KV cache page/block 中的 token 数 |
| `Nb` | 当前分配的 KV blocks 数 |

注意：

```text
Σ_e n_e = T × K
```

这是 assignment 数，不是去重后的 token 数。一个 token 选择 `K` 个专家，会被计算 `K` 次再加权合并。

---

## 2. 哪些 index 必须录制：逐项清单

### 2.1 输入和调度 index

| index/元数据 | 典型逻辑 shape | 值改变后发生什么 | 建议录制 |
|---|---|---|---|
| `input_ids` | `[T]` 或 `[B,S]` | 改变 embedding 读取行；通过 hidden state 间接改变所有后续路由和稀疏 index | 完整值、tokenizer revision、special token 定义 |
| `position_ids` | `[T]` 或 `[B,S]` | 改变 RoPE；可能改变滑窗/压缩边界和 cache 写入 | 完整值 |
| `attention_mask` | `[B,S]`、`[B,1,Q,S]` 或压缩表示 | 决定 padding/因果可见性；某些后端直接改变有效工作量 | 完整值或可无损重建表示 |
| `seq_lens` | `[B]` | 决定每个请求可见的 KV 长度 | 完整值 |
| `context_lens` | `[B]` | 区分已缓存上下文和本轮新 token | 完整值 |
| `cu_seqlens_q/k` | `[B+1]` | 定义 ragged segment 边界 | 完整值 |
| `token_to_req` | `[T]` | 把扁平 token 映射到请求，影响 cache 地址计算 | 完整值 |
| Prefill/Decode 划分 | 标量和 index 列表 | 决定进入 Prefill kernel 还是 Decode kernel | 请求顺序、数量、边界 |

只记录 `T` 不够。例如以下两个 batch 都有 `T=8`：

```text
A: Q = [8]          # 一个长 Prefill
B: Q = [1,1,1,1,1,1,1,1]  # 八个 Decode
```

它们的主激活 shape 都可能先展平为 `[8,H]`，但 attention metadata、KV 访问、kernel 和性能完全不同。

### 2.2 KV Cache 地址 index

| index | 典型逻辑 shape | 影响 |
|---|---|---|
| `slot_mapping` | `[T]` | 指定每个 token 写入哪个物理 cache slot；负值通常表示无效/不写 |
| `block_table` | `[B,max_blocks]` | 将逻辑 block 映射到物理 block |
| `block_numbers` | kernel 内部动态读取 | 决定实际访问哪个 KV page |
| `block_offsets` | `[T]` 或 kernel 内计算 | 决定 page 内偏移 |
| prefix-cache block hash/命中结果 | 每个逻辑 block 一个 key/状态 | 命中时跳过共享前缀的 Prefill |
| allocate/evict/copy index | 动态长度 | 改变 cache 状态和复制工作量 |

这类 index 常见的特点是：**tensor shape 相同，但读写地址完全不同**。如果只回放 attention 算子而没有恢复同一份 cache 初始状态，结果没有可比性。

### 2.3 DeepSeek-V4 中 position 直接控制 KV 压缩写入

vLLM 的 DeepSeek-V4 压缩实现包含明确的提前返回条件：

```python
if (position + 1) % COMPRESS_RATIO != 0:
    return
```

因此对于 C4 层：

```text
position = 0,1,2   → 不产生一个完整 C4 压缩项
position = 3       → 产生压缩项
position = 4,5,6   → 不产生新的完整压缩项
position = 7       → 再产生一个压缩项
```

C128 同理，只是边界变成 `127、255、383...`。

同一实现还会：

```python
slot_id = slot_mapping[token_idx]
if slot_id < 0:
    return
```

这说明至少有两个“值控制的提前退出”：

1. `slot_mapping` 是否有效；
2. `position_ids` 是否到压缩边界。

这两个值都不能只保存 shape，必须保存具体内容。

DeepSeek-V4 的层类型也不同：

| 模型 | 主干层数 | 主干 `compress_ratio` 分布 | 直接含义 |
|---|---:|---|---|
| Pro | 61 | C128：31 层；C4：30 层 | C4 层有稀疏 indexer；C128 层压缩但无 C4 indexer |
| Flash | 43 | SWA-only：2 层；C4：21 层；C128：20 层 | 三种 attention 分支都存在 |

当前 vLLM 实现中：

```text
compress_ratio == 4
  → compressor + sparse indexer

compress_ratio == 128
  → compressor，无 C4 sparse indexer

compress_ratio == 0
  → SWA-only，无 compressor
```

所以层号本身也是一个静态 selector，`position_ids` 则是层内动态 selector。

### 2.4 稀疏注意力 index

#### GLM-5.2

GLM-5.2 indexer 的主要结果是：

```text
topk_indices: int32 [B, Q, min(2048, S)]
```

这些 index 指向当前 query 需要读取的历史 token。不同 token 内容会改变 indexer score，从而改变被读取的 KV 位置。

更关键的是 GLM-5.2 使用跨层共享：

```text
full 层：重新计算 topk_indices
shared 层：复用前一个 full 层产生的 topk_indices
```

官方配置中 78 层共有：

```text
full: 21 层
shared: 57 层
```

因此 shared 层不能被当作“本层根据本层 hidden state 独立得到 index”。回放 shared 层时必须同时提供它所依赖的前序 full 层 index。

#### MiniMax-M3

MiniMax-M3 选择的不是 2048 个单 token，而是稀疏 block：

```text
block_size = 128 tokens
topk_blocks = 16
```

一个 query 的稀疏读取上限可粗略理解为：

```text
16 × 128 = 2048 token positions
```

实际还会叠加 local/init block、因果边界和最后一个不完整 block，不能简单假设永远正好读 2048 个有效 token。

vLLM 的 MiniMax-M3 metadata 中明确包含：

```text
seq_lens
max_seq_len
slot_mapping
num_actual_tokens
num_decodes / num_prefills
cu_seqlens_q / cu_seqlens_k
block_table
total_kv_blocks
```

可选的 IndexCache 还允许若干稀疏层共用同一个 `topk_indices_buffer`。这同样是层间状态。

#### DeepSeek-V4

DeepSeek-V4 C4 层的逻辑稀疏选择可表示为：

```text
sparse_topk_indices: [Q, K_index]
```

其中：

```text
Pro   K_index = 1024
Flash K_index = 512
```

这里指向的是压缩后的 KV 位置；其物理 slot、有效长度、Prefill/Decode buffer shape 还取决于实现和 cache 状态。

### 2.5 MoE 路由 index

每个 MoE 层至少需要关注：

| tensor/元数据 | 逻辑 shape | 作用 |
|---|---|---|
| `router_logits` | `[T,E]` | 每个 token 对所有专家的原始分数 |
| `topk_indices` | `[T,K]` | 每个 token 选中的专家 |
| `topk_weights` | `[T,K]` | 合并专家输出时的权重 |
| `expert_counts` | `[E]` | 每个专家的 `n_e` |
| `permutation` | `[T×K]` | 按专家重排 token assignment |
| `expert_offsets` | `[E+1]` | 每个专家在重排后 buffer 中的区间 |
| `expert_to_rank` | `[E]` 或版本化 map | 每个专家当前放在哪个 rank |
| `send_counts/recv_counts` | `[EP]` 或 `[EP,EP]` | AllToAllV 的实际通信长度 |
| `unpermutation` | `[T×K]` | 恢复 token 顺序 |

如果系统有 EPLB/冗余专家，`expert_to_rank` 不是永远固定的；要保存映射版本或映射 hash。

### 2.6 采样、EOS 和 MTP index

| 决策 | 影响 |
|---|---|
| Top-k/Top-p 候选 token index | 改变下一步输入 token |
| sampled token id | 改变后续所有层的 hidden、MoE 路由和稀疏 index |
| EOS 命中 | 请求结束，不再进入后续 Decode |
| MTP draft token | 一次 verification 的输入 token 数变化 |
| MTP accepted count/positions | 下一轮真正推进几个 token，改变 `T`、position 和 cache 写入 |

做端到端回放时，必须记录随机种子还不够；还要固定 sampler 实现、并行归约顺序和离散采样结果。做算子性能回放时，则可直接保存 sampler/MTP 的离散输出作为输入。

### 2.7 index 的建议分级

| 级别 | 判定 | 保存方式 |
|---|---|---|
| P0 | 会增删算子、改变循环次数、改变通信量或跨层状态 | 完整保存，逐元素校验 |
| P1 | 决定 gather/scatter/cache 地址 | 完整保存或无损重建 |
| P2 | 只用于诊断分布 | histogram/摘要可作为补充，但不能替代 P0/P1 原值 |
| P3 | 纯日志 id，不参与计算 | 可只存映射关系 |

对体积较大的 P0/P1 index，可使用：

```text
独立二进制 blob + dtype + shape + endianness + 内容 hash
```

hash 只能确认相等，不能用于真正回放；回放仍需 blob 或确定性重建。

---

## 3. 所有需要考虑的 shape

## 3.1 四个 shape 层次

以逻辑权重 `W: [O,I]` 为例：

| 层次 | 示例 | 为什么需要单独记录 |
|---|---|---|
| 全局逻辑 shape | `[O,I]` | 表达模型语义和 checkpoint |
| rank-local shape | Column Parallel `[O/TP,I]`；Row Parallel `[O,I/TP]` | 决定单卡 GEMM 和通信 |
| 量化物理 shape | INT4 packed、FP8 weight + scale | 决定真实内存和 kernel |
| 执行 shape/layout | padding、转置、C8/NZ、tile 后 shape | 决定实际算子性能 |

以激活 `hidden_states: [T,H]` 为例：

```text
逻辑:       [T, H]
图执行:     [T_exec, H]，T_exec >= T
TP 局部:    可能仍是 [T_exec,H]，也可能某个维度被切分
MoE 重排后: [n_e,H]，不同 expert 的 n_e 不同
```

### 3.2 权重的公共逻辑 shape

下表采用 PyTorch `Linear(in_features, out_features)` 的权重方向 `[out,in]`。

| 模块 | 逻辑权重 shape |
|---|---|
| Token Embedding | `[V,H]` |
| LM Head | `[V,H]`；如果 tied embedding，可能共享同一权重 |
| RMSNorm/LayerNorm weight | `[H]` |
| Q projection | `[Nq×Dh,H]` |
| K projection | `[Nkv×Dh,H]` |
| V projection | `[Nkv×Dh,H]` |
| fused QKV | `[(Nq+2Nkv)×Dh,H]` |
| O projection | `[H,Nq×Dh]` |
| Dense gate/up | 各 `[I,H]`；fused 时 `[2I,H]` |
| Dense down | `[H,I]` |
| MoE router | `[E,H]` |
| routed expert gate/up | 分开时各 `[E,Ie,H]`；fused 时 `[E,2Ie,H]` |
| routed expert down | `[E,H,Ie]` |
| shared expert | 与 Dense MLP 相同，只是 intermediate size 取 shared expert 配置 |

这张表是语义公式，不代表 checkpoint 一定使用同样的 tensor 名称或融合方式。

### 3.3 目标模型的结构维度

| 模型 | `H` | `L` | attention | MoE |
|---|---:|---:|---|---|
| DeepSeek-V4 Pro | 7168 | 61 | 128 heads，`Dh=512`，C4/C128 | `E=384,K=6,Ie=3072` |
| DeepSeek-V4 Flash | 4096 | 43 | 64 heads，`Dh=512`，SWA/C4/C128 | `E=256,K=6,Ie=2048` |
| GLM-5.2 | 6144 | 78 | MLA/DSA，64 heads，QK=256，V=256 | 前 3 层 Dense，后 75 层 `E=256,K=8,Ie=2048` |
| MiniMax-M3 | 6144 | 60 | GQA，`Nq=64,Nkv=4,Dh=128`，block sparse | 前 3 层 Dense，后 57 层 `E=128,K=4,Ie=3072` |
| Qwen3.7 Max/Plus | 未公开 | 未公开 | 官方公开 1M context、64K 最大输出；内部 attention 结构未公开 | 内部 MoE/Dense 结构未公开 |

### 3.4 DeepSeek-V4 具体权重 shape 示例

当前 vLLM 实现将 attention 输入投影构造为：

```text
fused_wqa_wkv: [Rq + Dh, H]
wq_b:          [Nq × Dh, Rq]
wo_a:          [G × Ro, Nq × Dh / G]
wo_b:          [H, G × Ro]
```

其中 `Rq=q_lora_rank`，`Ro=o_lora_rank`，`G=o_groups`。

#### Pro

```text
fused_wqa_wkv = [1536+512, 7168] = [2048,7168]
wq_b          = [128×512,1536]   = [65536,1536]
wo_a          = [16×1024,128×512/16] = [16384,4096]
wo_b          = [7168,16×1024]       = [7168,16384]
```

MoE 语义权重：

```text
router         [384,7168]
expert gate_up [384,6144,7168]
expert down    [384,7168,3072]
```

#### Flash

```text
fused_wqa_wkv = [1024+512,4096] = [1536,4096]
wq_b          = [64×512,1024]   = [32768,1024]
wo_a          = [8×1024,64×512/8] = [8192,4096]
wo_b          = [4096,8×1024]     = [4096,8192]
```

MoE 语义权重：

```text
router         [256,4096]
expert gate_up [256,4096,4096]
expert down    [256,4096,2048]
```

注意：DeepSeek-V4 checkpoint 中 routed expert 为 FP4、其他权重主要为 FP8。上面是反量化后的逻辑 shape；真实文件和昇腾执行时还会有 packed weight、scale、对齐和重排 shape，必须从目标 checkpoint revision 与后端加载结果导出。

### 3.5 GLM-5.2 具体权重 shape

根据官方配置和 Transformers 的 `GlmMoeDsaAttention` 实现：

```text
embedding             [154880,6144]
q_a_proj              [2048,6144]
q_b_proj              [64×256,2048] = [16384,2048]
kv_a_proj_with_mqa     [512+64,6144] = [576,6144]
kv_b_proj              [64×(192+256),512] = [28672,512]
o_proj                 [6144,64×256] = [6144,16384]
```

DSA indexer 权重：

```text
indexer.wq_b           [32×128,2048] = [4096,2048]
indexer.wk             [128,6144]
indexer.weights_proj   [32,6144]
```

Dense MLP：

```text
gate_proj              [12288,6144]
up_proj                [12288,6144]
down_proj              [6144,12288]
```

MoE：

```text
router                 [256,6144]
expert gate_up         [256,4096,6144]
expert down            [256,6144,2048]
shared expert gate/up  各 [2048,6144]
shared expert down     [6144,2048]
```

其中 router 在官方配置中使用 FP32 计算，这是为了降低离散 TopK 路由对数值误差的敏感性；录制时仍需核对目标推理后端是否保持该精度。

### 3.6 MiniMax-M3 具体权重 shape

文本模型主要维度：

```text
embedding       [200064,6144]
Q projection    [64×128,6144] = [8192,6144]
K projection    [4×128,6144]  = [512,6144]
V projection    [4×128,6144]  = [512,6144]
O projection    [6144,64×128] = [6144,8192]
```

具体 checkpoint/后端可能将 Q/K/V 与 indexer 投影融合成一个更大的权重；上面保留的是容易对照的语义 shape。

前 3 个 Dense 层：

```text
gate_proj       [12288,6144]
up_proj         [12288,6144]
down_proj       [6144,12288]
```

后 57 个 MoE 层：

```text
router          [128,6144]
expert gate_up  [128,6144,6144]   # 2×Ie = 6144
expert down     [128,6144,3072]
shared expert gate/up 各 [3072,6144]
shared expert down    [6144,3072]
```

MiniMax-M3 还是多模态模型。出现 `image_token_index=200025` 或 `video_token_index=200026` 时，会额外进入视觉编码、patch merge 和 projector 路径。文本回放和多模态回放必须拆成不同用例族。

### 3.7 Qwen3.7 替代基线：公开参数和可观测 shape

Qwen3.7 当前可使用的官方公开参数如下：

| 参数 | `qwen3.7-max` | `qwen3.7-plus` |
|---|---|---|
| 输入模态 | Text | Text、Image、Video |
| 输出模态 | Text | Text |
| context window | 1M tokens | 1M tokens |
| 最大输出 | 64K tokens | 64K tokens |
| Thinking | 支持，可按请求启停/调节 | 支持，可按请求启停/调节 |
| Function Calling | 支持 | 支持 |
| Built-in tools | 支持 | 支持 |
| Structured output | 支持 | 支持 |
| 单请求最大图片数 | 不适用 | 2048 |
| 单请求最大视频数 | 不适用 | 64 |
| 单图最大像素 | 不适用 | 16M |
| 单视频上限 | 不适用 | 2 小时、2 GB |

Qwen3.7 Plus 官方给出的图像 token 估算关系是：

```text
image_tokens ≈ h × w / (32 × 32) + 2
```

因此即使文本 token 数相同，只要图像分辨率不同，视觉 token 数、Prefill 总长度和多模态编码工作量就会变化。录制时至少需要保存：

```text
model_id / snapshot
input modality
text token count
image/video 数量
每个媒体对象的原始和预处理后分辨率
估算/返回的媒体 token 数
thinking mode / reasoning effort
tool schema token 数
实际输出 token 数
stop/EOS 原因
```

#### Qwen3.7 当前能确定的 shape

在服务接口层，可以确定或观测：

```text
文本输入:       [B, S_text]
图片列表:       [N_image, C, H_i, W_i]，各图片尺寸可不同
视频帧/片段:    [N_video, F_i, C, H_i, W_i]
媒体 token 数:  Σ_i tokens(image_i/video_i)
总上下文长度:   S_text + S_media + S_tool + S_history <= 1M
生成长度:       S_output <= 64K
```

这里的 shape 是请求/工作负载 shape，不是模型内部算子 shape。

#### Qwen3.7 当前不能可靠填写的内部 shape

官方公开资料目前没有给出：

```text
V、H、L、Nq、Nkv、Dh
Dense/MoE 层排布
E、K、Ie
router 和 sparse-attention index
embedding/QKV/MLP/expert 权重 shape
量化格式、TP/EP 切分和 KV cache 内部布局
```

因此 Qwen3.7 在本报告中承担两个作用：

1. 作为 1M 长上下文、Thinking、Tool Calling 和多模态 workload 的请求级泛化基线；
2. 等获得可本地部署的 checkpoint 或内部部署 manifest 后，使用本文的 weight manifest、离散 index 探针和逐层差异定位方法补齐算子级信息。

不能把 Qwen3、Qwen3.5 或 Qwen3.6 开放模型的权重 shape 直接写成 Qwen3.7 的实际 shape。若工程验证暂时使用旧版开放 Qwen 作为代理，实验结果必须标记为 `architecture_proxy`，不能标记为 Qwen3.7 同路径回放。

### 3.8 MiniMax-M3 的 TP 本地 Q/K/V shape 示例

MiniMax-M3 有：

```text
Nq = 64
Nkv = 4
Dh = 128
```

当 `TP=8` 时：

```text
local Q heads  = 64 / 8 = 8
local Q        = [T,8,128]
```

因为 `Nkv=4 < TP=8`，每个 rank 至少保留 1 个 KV head，KV head 会在 rank 之间复制：

```text
local K = [T,1,128]
local V = [T,1,128]
```

这说明“卡数翻倍”不一定继续把 KV 维度除以 2；当 TP 大于 KV head 数时，局部 KV shape 会进入复制区间。

### 3.9 运行时 activation 和 metadata shape

| 阶段/模块 | 逻辑 shape | 备注 |
|---|---|---|
| hidden input/output | `[T,H]` | 图执行可能为 `[T_exec,H]` |
| norm output | `[T,H]` | shape 不变，数值影响后续路由 |
| Q | `[T,Nq_local,Dh]` | 可能 padded heads |
| K/V | `[T,Nkv_local,Dh]` | TP>Nkv 时可能复制 |
| attention output | `[T,Nq_local,Dv]` 或 `[T,H_local]` | 随 backend |
| logits | `[T,V_local]` 或只计算采样位置 | Decode 常不必对所有 Prefill token 计算 LM Head |
| router logits | `[T,E]` 或 `[T,E_local]` | 取决于 router 是否复制/切分 |
| MoE topk | `[T,K]` | shape 固定，内容导致 `n_e` 变化 |
| expert input | `[n_e,H]` | 每层、每个 expert 动态变化 |
| sparse token index | `[B,Q,min(K_idx,S)]` | 长度不足 `K_idx` 时 TopK 变短 |
| sparse block index | `[B,Q,K_blocks]` | 还需有效 block mask |
| KV cache | 常见语义 `[Nb,P,Nkv_local,Dh]` | DeepSeek 压缩 cache 有专用布局 |
| block table | `[B,max_blocks_per_req]` | 值决定物理页 |
| slot mapping | `[T]` | 负值可触发跳过 |
| communication buffer | `[sum(send_counts),H]` | 由路由结果决定 |
| graph input | `[T_exec,...]` | `T_exec` 由 bucket 决定 |

### 3.10 量化后的 shape

对于逻辑权重：

```text
W: [O,I]
```

常见但非唯一的物理表示包括：

```text
INT4 packed bytes: [O,ceil(I/2)]
per-group scale:   [O,ceil(I/group_size)]
FP8 block scale:   [ceil(O/B_o),ceil(I/B_i)]
```

DeepSeek-V4 config 中普通 FP8 权重块为 `128×128`，其 scale 网格可按：

```text
[ceil(O/128), ceil(I/128)]
```

理解。实际 checkpoint 还可能转置 scale、添加 padding 或使用专用 FP4 expert 格式。

因此录制系统必须把下面四个字段分开：

```text
logical_shape
stored_shape
runtime_layout_shape
scale_shape
```

不能只把 dtype 从 BF16 改成 INT4/FP8 后继续复用原来的权重 shape。

### 3.11 64 卡、128 卡和显存大小如何影响 shape

卡数本身不是直接代入所有 shape 的除数。先要知道并行方案：

```text
world_size = DP × TP × PP（EP 与 TP/DP 的组合关系取决于框架）
```

#### 只增加 DP

例如从 64 卡变 128 卡，如果只是 DP 从 8 变 16：

```text
单 rank 权重 shape：不变
单请求算子 shape：可不变
模型副本数：翻倍
全局吞吐与调度：变化
```

这种情况不能把单卡 GEMM shape 直接除以 2。

#### 增加 TP

Column/Row Parallel 权重局部 shape 会变化，attention local heads 和 collective group 也会变化。

#### 增加 EP

每个 rank 放置的专家数、token dispatch 的目的 rank、AllToAllV `send_counts` 都会变化。即使专家权重本身的单专家 shape 不变，单 rank 持有的专家集合会变。

#### 64G 与 96G/128G 单卡

显存大小通常先影响：

- 能否装下某种量化权重；
- TP/PP/EP 方案选择；
- KV cache 可分配 block 数；
- `max_model_len`、`max_num_seqs`、`max_num_batched_tokens`；
- graph bucket 上限；
- 是否 offload。

然后才间接改变本地 shape 和执行路径。**显存容量不能作为一个连续比例直接缩放算子 shape。**

---

## 4. token 零值、非零值和重复值的具体影响

### 4.1 需要先分清“token id 为 0”和“tensor 元素为 0”

| 情况 | 是否天然稀疏/少算 |
|---|---|
| `input_ids[i] == 0` | 否，只是读取 embedding 第 0 行；还可能是 BOS |
| hidden activation 中很多 0 | 普通 Dense GEMM 通常仍完整计算 |
| `attention_mask == 0/False` | 有语义，可能屏蔽 token |
| `slot_mapping < 0` | 有控制语义，可能跳过 cache 写入 |
| 特殊 padding id | 有语义，但 id 不一定等于 0 |
| value-sparse 专用 kernel | 可能少算，但必须确认真实后端启用 |

DeepSeek-V4 的 BOS 是 0；GLM-5.2 的 padding id 是 154820。由此可见，通用录制器不能通过 `token_id==0` 判断 padding。

### 4.2 重复 token 为什么通常不能复用中间结果

例如输入：

```text
[42,42,42,42]
```

Embedding 查表得到的四个初始向量相同，但它们的：

```text
position_ids = [0,1,2,3]
```

不同。经过 RoPE、因果 attention 和残差后，各位置 hidden states 通常不再相同。下一层的 MoE router 和稀疏 index 也可能不同。

所以：

```text
重复 token id
≠ 重复 hidden state
≠ 相同 MoE expert
≠ 可以跳过算子
```

### 4.3 重复前缀为什么可能改变执行路径

如果请求 A 和 B 具有完全相同的前缀，且启用了 Automatic Prefix Caching：

```text
A 已计算并保留前缀 KV
B 到来后命中同一前缀
→ B 可以跳过共享部分的 Prefill
```

此时重复值影响的不只是计算结果，而是：

- 实际 Prefill token 数；
- KV block 命中/分配；
- `context_lens` 和 `cu_seqlens`；
- graph bucket；
- 端到端延迟。

因此“重复 token 测试”必须同时做 prefix cache 关闭和开启两组，否则会把两种完全不同的路径混在一起。

### 4.4 特殊 token

至少要单独覆盖：

- BOS；
- 一个或多个 EOS；
- padding；
- tool/function-call 标记；
- thinking/reasoning 模式控制 token；
- image/video placeholder；
- 模型自定义分隔符。

这些 token 可能影响 tokenizer 后的长度、停止条件、模板分支、多模态 encoder 和后续 Decode 次数。

### 4.5 推荐的 token 对照实验

| 用例 | 只改变什么 | 应观察什么 |
|---|---|---|
| T0 | 合法的 token id 0 vs 普通 id，mask 都有效 | Embedding 行和后续数值变化；Dense op shape 通常不变 |
| T1 | 相同 input id，padding mask 有效 vs 无效 | attention 有效长度、cache 写入和 kernel 路径 |
| T2 | `[x,x,x...]` vs 等长随机 id，prefix cache 关闭 | shape 相同，但 MoE/稀疏 index 分布可能不同 |
| T3 | 两请求重复前缀 vs 不重复，prefix cache 开启/关闭 | prefix hit、实际 Prefill token 数和延迟 |
| T4 | EOS 出现在不同 Decode step | forward 次数、活跃请求数和 batch shape |
| T5 | 文本 token vs image/video placeholder | 是否进入视觉分支、额外 token 数 |

---

## 5. MoE：token 值如何变成动态 shape

### 5.1 完整因果链

对第 `l` 个 MoE 层：

```text
input_ids、position、前序层和 cache
  → hidden_l [T,H]
  → router_logits [T,E]
  → topk_indices [T,K]
  → expert_counts n_e
  → token permutation [T×K]
  → AllToAllV send/recv counts
  → expert e 输入 [n_e,H]
  → gate/up GEMM [n_e,H] × [Ie,H]^T
  → down GEMM [n_e,Ie] × [H,Ie]^T
  → unpermute + weighted combine
  → hidden_{l+1} [T,H]
```

外部 shape `[T,H]` 在进入和离开 MoE 后都一样，但内部是一组动态 ragged shape：

```text
[n_0,H], [n_1,H], ... [n_{E-1},H]
```

### 5.2 一个最小例子

假设：

```text
T=8, E=4, K=2
总 assignment 数 = 16
```

路由 A：

```text
expert_counts = [4,4,4,4]
```

路由 B：

```text
expert_counts = [13,1,1,1]
```

两者的 router/topk tensor shape 都相同：

```text
topk_indices [8,2]
```

但专家 GEMM 分别是：

```text
A: 4 个 [4,H] GEMM
B: 1 个 [13,H] + 3 个 [1,H] GEMM
```

如果专家分布在不同机器上，AllToAllV 的消息长度也完全不同。仅用均匀随机 `topk_indices` 回放，会系统性低估热点专家、长尾通信和小 `n_e` kernel 的开销。

### 5.3 目标模型的 MoE 差异

| 模型 | `E` | `K` | 每层 assignment 数 |
|---|---:|---:|---:|
| DeepSeek-V4 Pro | 384 | 6 | `6T` |
| DeepSeek-V4 Flash | 256 | 6 | `6T` |
| GLM-5.2 | 256 | 8 | `8T` |
| MiniMax-M3 | 128 | 4 | `4T` |

同样的 `T` 下，GLM-5.2 的 routed assignment 数是 MiniMax-M3 的 2 倍，但专家数、专家宽度和层数也不同，不能只用 `T×K` 一个指标比较性能。

### 5.4 重复 token 是否一定路由到同一专家？

不一定。原因包括：

- position 不同；
- attention 上下文不同；
- layer 不同；
- KV/prefix 状态不同；
- 量化和并行归约误差不同；
- router correction bias 或 EPLB 映射不同。

正确的验证方式不是假设“重复 token → 重复路由”，而是实测：

```text
same token id 的 topk expert 重合率
same prefix 在不同 batch 中的 topk 重合率
相邻 layer 的 topk 重合率
量化前后的 topk 重合率
```

### 5.5 量化误差为什么可能放大成路径差异

如果 router 的第 `K` 名与第 `K+1` 名分数很接近：

```text
margin = score[K] - score[K+1]
```

很小的数值误差就可能交换两名专家。连续数值误差仍很小，但离散 index 已变化，继而改变：

- `expert_counts`；
- 通信目的 rank；
- GEMM shape；
- expert 输出；
- 下一层路由。

建议每层额外记录：

```text
topk_boundary_margin_min
topk_boundary_margin_p01/p50
route_overlap_with_reference
```

margin 只是风险指标；路径保持回放仍以 `topk_indices` 精确匹配为准。

### 5.6 MoE 录制的最低要求

只保存 `[T,E]` router logits 太大，只保存 histogram 又不能回放。第一版建议：

```text
必存：
  topk_indices [T,K]
  topk_weights [T,K]
  expert_counts [E]
  permutation / offsets
  expert_to_rank map/version
  send_counts / recv_counts

抽样或按需存：
  router_logits [T,E]

常驻统计：
  max/mean/p50/p95 n_e
  zero-hit expert 数
  load imbalance
  route topk overlap
```

---

## 6. layers 之间的影响

### 6.1 数值逐层传播

标准残差层可以简化为：

```text
x'      = x_l + Attention(Norm(x_l), cache_l)
x_{l+1} = x'  + MLP_or_MoE(Norm(x'))
```

因此第 1 层的微小差异会成为第 2 层的输入差异。对 Dense 模型，它可能只表现为连续误差；对 MoE/稀疏注意力模型，它可能在后面某层越过 TopK 边界，变成离散路径差异。

推荐同时画两条曲线：

```text
连续误差：每层 hidden cosine / max abs / relative L2
离散误差：每层 MoE route overlap / sparse-index overlap
```

这比只比较最终 logits 更容易定位“从哪一层开始路径分叉”。

### 6.2 每层独立状态

一般每层都有独立：

- K/V cache；
- cache dtype/layout；
- MoE router 输出；
- routed expert token counts；
- 稀疏 indexer key/cache；
- kernel、workspace 和耗时。

`block_table`、request 顺序等调度 metadata 可以跨层共用，但每层实际 cache 内容不同。不能拿第 3 层的 KV 数据去回放第 4 层。

### 6.3 显式跨层 index 共享

#### GLM-5.2

官方实现将本层的 `topk_indices` 返回给上层循环：

```text
full layer 重新计算 topk
→ 作为 prev_topk_indices
→ 后续 shared layer 直接复用
```

因此一个 full 层 index 错误会同时污染后续多个 shared 层。

#### MiniMax-M3

可选 IndexCache 中：

```text
只有每 index_topk_freq 个稀疏层中的一个层重新算 TopK
其余层复用 shared topk_indices_buffer
```

回放时要记录：

```text
index_source = recompute | reused
source_layer_id
index_cache_enabled
index_topk_freq
buffer version/hash
```

#### DeepSeek-V4

各层静态压缩比不同，C4/C128/SWA 的 attention 子图不同；如果后端额外启用 IndexCache，也要把是否复用和来源层作为运行配置记录。

### 6.4 层异构

| 模型 | 层间差异 |
|---|---|
| DeepSeek-V4 Pro | C4 与 C128 交错；只有 C4 有稀疏 indexer |
| DeepSeek-V4 Flash | 前 2 层 SWA-only，后续 C4/C128 交错 |
| GLM-5.2 | MLP 前 3 层 Dense、后 75 层 MoE；attention indexer 有 full/shared 模式 |
| MiniMax-M3 | 前 3 层 Dense/非稀疏，后 57 层 MoE/block-sparse |

所以“同一个模型的所有层 shape 一样”只对 `[T,H]` 这类外层接口近似成立，对内部权重、indexer、MoE 和 cache 不成立。

### 6.5 Pipeline Parallel 边界

使用 PP 时还会新增：

- stage 首尾 activation send/recv；
- 不同 stage 持有不同层和权重；
- microbatch/调度气泡；
- 跨机边界是否变化。

64 卡与 128 卡若 PP 切分不同，即使单层算子 shape 相同，层间通信路径也不同。性能泛化必须把 `layer_id → stage_id → rank/node` 映射一起保存。

### 6.6 什么时候可以合并层进行回放

只有当以下 fingerprint 都相同，才建议把多层归为一个性能类别：

```text
layer_type
attention_type / compress_ratio
weight logical/local/physical shape
actual token shape
sparse index source and valid topk
MoE expert-count bucket
kernel/tiling id
communication collective and payload bucket
cache dtype/layout
```

即使归为一类，也应保留每层原始观测，避免平均值掩盖热点层。

---

## 7. 面向录制回放的具体数据结构

### 7.1 权重 manifest

每个 checkpoint revision 自动生成：

```text
tensor_name
layer_id
module_type
global_logical_shape
checkpoint_stored_shape
checkpoint_dtype
quant_method
group/block_size
scale_tensor_name
scale_shape
tp_shard_axis
ep_owner
rank_local_shape
runtime_layout
runtime_shape
content_hash / checkpoint revision
```

这一步要读取 safetensors header/索引和实际后端 load 后 tensor，不能只读 `config.json`。

### 7.2 每个 forward 的请求级记录

```text
input_ids
position_ids
attention_mask 或无损压缩表示
seq_lens / context_lens / cu_seqlens
prefill/decode 请求顺序
slot_mapping / block_table / token_to_req
prefix-cache hit blocks
T_actual / T_exec / graph_bucket
sampling/MTP/EOS 决策
```

### 7.3 每层最低记录

```text
layer_id
layer_type
input/output shape、dtype、layout
input/output checksum 和数值摘要

attention:
  attention type
  Q/K/V local shape
  sparse topk indices
  valid topk count
  index_source/source_layer
  KV cache shape/dtype
  cache read/write slots

MoE:
  topk_indices / topk_weights
  expert_counts
  permutation/offsets
  expert_to_rank
  send_counts/recv_counts

runtime:
  op sequence
  kernel/tiling id
  workspace
  stream/overlap mode
  collective type/group/payload
  duration
```

### 7.4 三种回放模式要分开

#### A. 路径保持回放

直接注入录制的离散决策：

```text
MoE topk
sparse-attention topk
slot/block mapping
MTP accepted count
```

用途：复现同一路径下的性能。

#### B. 端到端决策回放

从原始 token 和状态重新计算 index，然后与录制值逐元素比较。

用途：验证新量化、新后端或新卡规模是否还能得到同一路径。

#### C. 反事实/边界回放

人工构造均匀路由、热点路由、TopK 临界 margin、cache 命中/不命中等。

用途：补齐线上样本没有覆盖的性能边界，不能冒充真实业务同路径结果。

---

## 8. 第一轮建议直接执行的验证矩阵

### 8.1 Index 与 cache

| ID | 变量 | 固定项 | 观察项 | 预期 |
|---|---|---|---|---|
| I0 | `input_ids` 内容 | `T/position/mask/cache` | op 序列、shape、MoE/稀疏 index | 外层 shape 相同，离散 index 可变 |
| I1 | DSV4 position 跨 C4/C128 边界 | token 数和内容 | 压缩写次数、cache slot | 边界位置出现写入 |
| I2 | `slot_mapping` 合法/负值 | 其他输入 | cache write kernel 有效工作量 | 负值跳过 |
| I3 | 重排 `block_table`，cache 内容同步搬移 | 逻辑 KV 不变 | 物理地址、结果、性能 | 结果应等价，物理访问不同 |
| I4 | 相同 `T`，不同 `cu_seqlens` | 总 token 数 | Prefill/Decode kernel、耗时 | 路径可显著不同 |

### 8.2 Token

| ID | 用例 | 观察项 |
|---|---|---|
| T0 | 合法 token 0 vs 普通 token | 验证不能把 id 0 当 padding |
| T1 | 重复 token vs 随机 token，等长 | 每层 route/index overlap |
| T2 | 重复前缀，APC off/on | 实际 Prefill token 数、cache hit |
| T3 | EOS 位置变化 | Decode forward 次数和活跃 batch |
| T4 | padding 分布变化但真实 token 数相同 | `T_exec`、mask 和 kernel bucket |
| T5 | MiniMax-M3 文本 vs 图像 placeholder | vision 分支和投影 shape |
| T6 | Qwen3.7 Plus 相同文本、不同图像分辨率/数量 | 媒体 token 数、总上下文长度、首 token 延迟 |
| T7 | Qwen3.7 Thinking off/on、不同 reasoning effort | 实际输出 token 数、停止原因和端到端 forward 次数 |

### 8.3 MoE

| ID | 用例 | 观察项 |
|---|---|---|
| M0 | 同一长度、不同语义 prompt | 各层 expert histogram、A2A counts |
| M1 | 注入均匀 route | grouped GEMM 的 `n_e` 分布 |
| M2 | 注入单/少数热点 expert | 热点 rank、长尾通信、容量/padding |
| M3 | BF16/FP8/INT4 同请求 | router TopK overlap、边界 margin |
| M4 | EP/TP 变化 | expert ownership、send/recv counts |
| M5 | EPLB off/on | `expert_to_rank` 版本和性能 |

### 8.4 Layers

| ID | 用例 | 观察项 |
|---|---|---|
| L0 | 全层采集 | 首次发生连续/离散差异的 layer |
| L1 | 只在早期一层替换一个 MoE index | 后续 route/index 分叉范围 |
| L2 | GLM full/shared 层 | shared 层 index 是否与 source 完全相同 |
| L3 | MiniMax IndexCache off/on | indexer op 是否按频率跳过 |
| L4 | DSV4 C4/C128/SWA 分层统计 | 每类 op、cache 写入和耗时 |
| L5 | PP stage 切分变化 | stage 边界 send/recv 与气泡 |

### 8.5 Shape、量化和机器规模

| ID | 用例 | 观察项 |
|---|---|---|
| S0 | 对每个 checkpoint 导出 weight manifest | 全局/packed/scale shape |
| S1 | TP 1/2/4/8/16（合法组合） | local head、local weight、collective |
| S2 | EP 不同规模 | rank-local experts 和 MoE dispatch |
| S3 | 64 卡 vs 128 卡但只改 DP | 验证单 rank shape 不应被误缩放 |
| S4 | 64G vs 96G/128G | 部署计划、KV block 数、graph 上限 |
| S5 | BF16/W8A8/W4A8/FP4 expert | packed weight、scale、kernel/tiling |
| S6 | `T` 跨 graph/kernel bucket 边界 | `T_actual/T_exec` 和 kernel id |
| S7 | Qwen3.7 Max/Plus 的 1M context 长度分桶 | 请求级 token shape、缓存命中和端到端延迟；不冒充内部算子 shape |

---

## 9. 第一版“泛化方法”建议收敛成三个可交付件

### 9.1 Shape 生成器

输入：

```text
model config + checkpoint manifest + quant config + TP/EP/PP + backend
```

输出：

```text
全部权重全局/局部/物理 shape
主干 activation 逻辑 shape
合法性约束
需要运行时才能确定的动态维度列表
```

它不猜 `n_e`、cache hit、TopK 内容，而是把这些标为 runtime-bound。

### 9.2 离散 index 记录器

优先支持：

```text
position/slot/block/request index
MoE topk/weights/counts/permutation
sparse attention topk 和跨层来源
sampling/EOS/MTP
```

这些数据是判断“是否同路径”的主体。

### 9.3 逐层差异定位器

输入两次运行，逐层输出：

```text
shape 首次差异
kernel 首次差异
MoE route 首次差异
sparse index 首次差异
cache state 首次差异
hidden 数值首次超阈值
```

这样可以回答：

```text
是 input/position 一开始就不同？
是量化误差在某一层跨过 TopK 边界？
是 shared index 来源不一致？
还是逻辑路径相同但 kernel/tiling 因机器规模改变？
```

---

## 10. 当前可以形成的具体结论

1. **最重要的动态 index 有三组：cache 地址 index、稀疏注意力 index、MoE expert index。**
2. **DeepSeek-V4 已存在明确的 value-dependent early return：position 压缩边界和无效 slot。**
3. **token id 为 0 不等于 padding，activation 为 0 也不等于 Dense kernel 少算。**
4. **重复 token 通常不减少层内计算；重复前缀只有在 cache 命中时才会减少 Prefill。**
5. **MoE 外部 `[T,H]` 相同不能证明性能可泛化，必须带上每层 `expert_counts` 和通信 counts。**
6. **layers 之间既有普通数值传播，也有 GLM/MiniMax 这类显式 index 共享。**
7. **权重 shape 必须同时记录逻辑、分片、量化存储和执行布局，尤其不能把量化只理解为 dtype 变化。**
8. **64 卡到 128 卡是否改变单 rank shape，取决于是增加 DP，还是重新选择 TP/EP/PP；卡数本身没有统一缩放公式。**
9. **显存大小主要通过部署方案、KV 容量和 graph/scheduler 上限间接改变 shape。**
10. **Qwen3.7 可作为 1M 上下文、Thinking、Tool Calling 和多模态请求 shape 的替代基线；在没有 checkpoint/manifest 时，不能声称已覆盖其内部权重或算子 shape。**
11. **第一阶段最有价值的验证不是追求一个总公式，而是先做 index 全量录制、weight manifest 和逐层分叉定位。**

---

## 11. 边界与待补充项

### 11.1 Qwen3.7 替代范围

本版将 Qwen3.7 纳入目标列表，采用：

```text
qwen3.7-max
qwen3.7-plus
```

两类官方服务参数。Qwen3.7 当前适合覆盖：

- 1M context 的输入长度和 cache 边界；
- 64K 最大输出下的 Decode 次数；
- Thinking/非 Thinking 请求；
- Function Calling、Built-in tools 和 Structured Output；
- Plus 的图像/视频 token 与动态分辨率 workload。

由于没有可审计的官方开放 checkpoint/config，本版仍不能从 Qwen3.7 得到权重 shape、层数、MoE 路由和单 rank 算子 shape。后续若取得内部部署 manifest 或开放权重，应优先补充：

```text
config + safetensors manifest
量化格式
TP/EP/PP 计划
逐层 op trace
MoE/sparse index
KV cache layout
```

### 11.2 “所有 shape”的实际边界

本文列出的是语义上必须覆盖的 shape。编译器和 kernel 还会产生版本相关的 workspace、tile、format-transform 临时 tensor，无法靠模型 config 一次性静态穷举。

最终完整性判定应是：

```text
静态权重 manifest
+ 运行时算子全量描述符
+ P0/P1 离散 index
+ cache/通信/图状态
```

而不是一张人工维护的 Markdown 表。

---

## 12. 主要证据来源

- [DeepSeek-V4 Pro 官方 config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json)
- [DeepSeek-V4 Flash 官方 config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json)
- [vLLM DeepSeek-V4 attention 实现文档](https://docs.vllm.ai/en/latest/api/vllm/models/deepseek_v4/attention/)
- [vLLM DeepSeek-V4 压缩与量化 cache 实现](https://docs.vllm.ai/en/latest/api/vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache/)
- [GLM-5.2 官方 config](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json)
- [GLM-5.2 官方模型卡与 IndexShare 说明](https://huggingface.co/zai-org/GLM-5.2)
- [Transformers GLM-MoE-DSA 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py)
- [MiniMax-M3 官方 config](https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/config.json)
- [vLLM MiniMax-M3 模型实现](https://docs.vllm.ai/en/latest/api/vllm/models/minimax_m3/amd/model/)
- [vLLM MiniMax-M3 sparse attention metadata](https://docs.vllm.ai/en/latest/api/vllm/models/minimax_m3/common/sparse_attention/)
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
- [vLLM Expert Parallel Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)
- [vLLM Quantization](https://docs.vllm.ai/en/latest/features/quantization/)
- [vLLM Ascend Batch Invariance](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/batch_invariance.html)
- [vLLM Ascend Expert Parallelism Load Balancer](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/expert_parallelism_load_balancer.html)
- [QwenCloud Qwen3.7 Max/Plus 模型列表与最大输出](https://docs.qwencloud.com/developer-guides/getting-started/text-generation-models)
- [Alibaba Cloud Qwen3.7 文本模型参数](https://www.alibabacloud.com/help/en/model-studio/text-generation-model)
- [Alibaba Cloud Qwen3.7 Plus 视觉输入参数](https://www.alibabacloud.com/help/en/model-studio/vision-model/)
