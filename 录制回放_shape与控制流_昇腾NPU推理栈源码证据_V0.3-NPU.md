# 录制回放中的 shape、index 与控制流：昇腾 NPU 推理栈源码证据调研（V0.3-NPU）

> 日期：2026-07-29  
> 前序文档：V0.1、V0.2、通用/NVIDIA 版 V0.3  
> 本文定位：独立的昇腾 NPU 版本，不修改原 V0.3。  
> 重点模型：DeepSeek-V4 Pro/Flash、GLM-5.2、MiniMax-M3、Qwen3.7。  
> 主要证据：vLLM Ascend、SGLang NPU backend、sgl-kernel-npu、Ascend PyTorch（torch_npu）。  

---

## 0. 先给结论

如果录制、回放运行在昇腾技术栈上，原 V0.3 的 `shape signature + path signature + state signature` 仍然成立，但还必须把下面两项提升为一等公民：

1. **physical-format signature**
   - 逻辑 shape 相同，不代表物理 storage shape 相同；
   - `ND`、`FRACTAL_NZ`、`PA_ND`、`PA_BSND`、`TND`、`BSND/BNSD` 会改变对齐、padding、stride、storage size 和可选算子；
   - BF16/FP16、INT8、W4A8/W4A8C8 的 C0、scale 和 packed weight shape 不同。
2. **operator-plan signature**
   - CANN/AscendC 算子的 tiling、workspace、图捕获 bucket、输入 layout 和 SoC 代际共同决定实际执行计划；
   - Python tensor 的输入/输出 shape 相同，tiling key、workspace、通信算法或 fused op 仍可能不同。

因此本文建议将昇腾回放对象表示为：

```text
NPU Replay Unit
  = model/weight manifest
  + rank topology
  + SoC/CANN/torch_npu/framework/operator versions
  + logical tensor signature
  + physical-format/storage signature
  + path/guard signature
  + dynamic index and ragged-workload signature
  + cache/layer/graph state signature
  + operator tiling/workspace/communication signature
```

本轮源码可以直接证明的核心结论有六条：

1. **DeepSeek-V4 的 sparse index 是算子输入，不是普通日志字段。**  
   vLLM Ascend 和 SGLang NPU 都把 C4 indexer 生成的 top-k 位置传给压缩注意力；索引值不同会读取不同 KV 位置，运算结果和有效访存都不同。
2. **MoE 的 token 值会通过路由结果改变 HCCL/All-to-All 的动态长度。**  
   每 expert token 数进一步生成 `input_splits/output_splits`，所以外层 `[T,H]` 相同也不能视为相同 workload。
3. **昇腾上“逻辑 shape”与“物理 shape”必须分开录。**  
   `FRACTAL_NZ` 对最后两维做分块和向上取整；INT8 与 FP16/BF16 的内块大小还不同。
4. **卡型不仅影响速度，也会改变 dtype、cache layout 和分支。**  
   vLLM Ascend 的 DeepSeek-V4 实现对 A5 与非 A5 选择不同 cache dtype、scale dtype 和 SWA cache head size。
5. **NPUGraph 不能只记录 batch size。**  
   框架会把 `seq_lens`、block table、dummy request、speculative width 等写入预分配 buffer；图 replay 还依赖捕获方式、固定地址/内存池和可更新输入。
6. **64 卡、128 卡本身不是 shape 参数，`DP×TP×EP×PP×CP` 分解才是。**  
   同样 64 卡可以是 `DP16×TP4`、`DP4×TP16`，也可以分成 Prefill/Decode 集群；其单 rank weight、activation、KV 和通信 shape 完全不同。

一个可操作的最小原则是：

> **凡是能改变“张量 extent、物理格式、索引集合、ragged 长度、分支、tiling、通信 count、cache 地址映射”的值，都不能只保留最终 output shape。**

### 0.1 对五个关注点的直接回答

| 关注点 | 源码结论 | 第一版必须录什么 |
|---|---|---|
| 0. 推理中算出的 index | V4/MiniMax/GLM 的 sparse top-k 会直接进入 NPU attention；MoE top-k 会生成专家分布和通信 split | index 全值或可重建引用、shape/dtype、合法范围、生成层、使用层、cache 前态 |
| 1. 所有 shape（含权重） | 要区分逻辑、单 rank、执行 bucket、storage、算子 ABI、有效 workload 六层；权重还要加 shard/packing/scale | 六层 shape、stride/format、weight shard manifest、scale/offset、padding 和有效 extent |
| 2. token 非零/重复 | token 0 通常不是 padding；重复 token 不一定少算，但会改变 cache、router、indexer 和请求结束状态 | input IDs、position、request boundary、有效长度；非零/重复率只作为统计 |
| 3. MoE 与 token 值 | token/hidden→top-k expert→expert histogram→HCCL split→本地 GEMM M | top-k IDs/weights、per-expert/rank counts、permute、send/recv、capacity/padding |
| 4. layers 之间 | shared indexer、压缩 cache、KV、residual、PP、MTP 都跨层/跨 step 携带状态 | layer ID/type、输入状态版本、index 来源层、cache 版本、PP stage、branch |

---

## 1. 证据版本、仓库和边界

### 1.1 固定提交

本报告下载并检查了以下官方开源仓库，所有链接固定到 commit：

| 层次 | 仓库 | 固定提交 | 本报告用途 |
|---|---|---|---|
| vLLM 昇腾适配 | [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend/tree/e462c42a4599bb17bae49775074eb6a9b094f528) | `e462c42a4599bb17bae49775074eb6a9b094f528` | 模型、attention、KV、MoE、量化、ACL Graph |
| SGLang 主仓 | [sgl-project/sglang](https://github.com/sgl-project/sglang/tree/1b9dfa14e66b617ed53270164549d59290b1f7c8) | `1b9dfa14e66b617ed53270164549d59290b1f7c8` | NPU backend、模型层和 graph runner |
| SGLang 昇腾算子 | [sgl-project/sgl-kernel-npu](https://github.com/sgl-project/sgl-kernel-npu/tree/3479f4d99cd4e65a1cbe316f8bafc318014a4eb9) | `3479f4d99cd4e65a1cbe316f8bafc318014a4eb9` | Lightning Indexer、Block Sparse Attention、DeepEP |
| PyTorch 昇腾插件 | [Ascend/pytorch](https://github.com/Ascend/pytorch/tree/10fe1622631286665f15d84951d53df7840e6ead) | `10fe1622631286665f15d84951d53df7840e6ead` | NPUGraph 和 NPU physical format |

这不是 Transformers 层面的调研。Transformers/HF config 只提供模型参数；本文主要检查推理框架怎样把参数和运行时值转成 NPU 算子输入。

### 1.2 源码证据的阅读格式

后文关键结论统一使用“源码证据卡片”，格式如下：

1. **代码位置**：固定 commit 的 GitHub 链接，精确到起止行；
2. **关键片段**：使用嵌入式 HTML 双栏，左栏是原始行号，右栏是源代码；
3. **证明内容**：只解释该片段能够直接支持的结论，不把推断写成代码事实。

片段采用“带行号的裁剪摘录”，不是可直接编译的完整文件：

- 行号来自本地固定提交，不是 Markdown 渲染器自动生成；
- 行号栏为空表示右侧是上一条长源码的视觉换行；
- `...` 或 `…` 是本文主动加入的省略标记，表示删除了与当前结论无关的参数或上下文，不属于仓库原文；
- 行号后的 `≈` 只用于 README/部署文档的中文转写，不冒充源码原文；
- 涉及结论的条件、shape 表达式、索引表达式和返回值不做语义改写；完整上下文以“代码位置”的固定 commit 链接为准。

示意：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>example.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>100
101</code></pre></td>
<td valign="top"><pre><code class="language-python">if condition:
    tensor = tensor.view(...)</code></pre></td>
</tr>
</tbody>
</table>

这种格式接近 GitHub 的带行号 file view，又不会把手工行号混进 Python/C++ 源码。片段只保留决定 shape、index、分支或状态的最小行；点击“代码位置”可以查看完整上下文。

### 1.3 静态扫描范围

为避免只靠人工挑例子，本文对 41 个关键文件做了候选扫描：

| 证据层 | 文件数 | 类别命中 | 去重源码行 |
|---|---:|---:|---:|
| vLLM Ascend | 16 个有命中文件（17 个纳入范围） | 6,299 | 5,452 |
| SGLang NPU | 11 | 4,050 | 3,528 |
| SGL Kernel NPU | 9 | 2,225 | 2,081 |
| Ascend PyTorch | 4 | 891 | 775 |
| 合计 | 41 | 13,465 | 11,836 |

扫描范围包括：

- `view/reshape/flatten/split/cat/slice/gather/scatter/topk/repeat/pad/reduction/allocation`；
- `contiguous/format_cast/FRACTAL_NZ/TND/PA_ND`；
- `if/elif/else/assert/return/raise`；
- `torch_npu.npu_*`、`torch.ops.custom.*`、`aclnn*`；
- NPUGraph、capture/replay、tiling、workspace；
- `seq_lens/block_table/slot_mapping/page_table/topk/expert_token/split_sizes`；
- TP/EP/DP/CP、HCCL、FlashComm、MC2、All-to-All。

完整候选：

- `survey/evidence/v0.3-npu/shape_control_candidates.csv`
- `survey/evidence/v0.3-npu/shape_control_summary.csv`
- 生成脚本：`survey/tools/scan_v03_npu_shape_control.ps1`

扫描命中只表示“需要审计”，不是自动证明 shape 一定变化。正文中的条目均经过人工检查。

### 1.4 本版没有声称覆盖的内容

本文尚未在真实 Atlas 集群上运行：

- A2/A3/A5 的算子 trace；
- CANN profiler/GE graph/AscendC tiling dump；
- 64/96/128G 多种显存卡的实测；
- 64 卡、128 卡 HCCL 通信 trace；
- 不公开权重/结构的 Qwen3.7 服务端内部实现。

因此本文是“源码证据 + 可执行的录制设计”的第一版，不是最终性能等价性证明。

---

## 2. 目标模型在昇腾栈中的证据状态

| 模型 | vLLM Ascend | SGLang NPU | 本版结论 |
|---|---|---|---|
| DeepSeek-V4 Pro/Flash | 有专用模型注册、DSA/SWA/compression、MTP、量化和部署文档 | 有专用 V4 NPU attention backend 和压缩 KV metadata | 可做源码级精细建模 |
| GLM-5.2 | 有官方部署文档；通过 GLM/DeepSeek-V2 模型路径接入专用 SFA/DSA NPU backend 和 shared indexer patch | 有 GLM MoE 模型；NPU 通用 attention/MoE 能力存在 | vLLM Ascend 证据强，SGLang 需运行验证具体后端组合 |
| MiniMax-M3 | 有专用模型注册、专用 sparse attention/index cache 实现 | 主仓有 MiniMax-M3，NPU backend 能接收 sparse top-k；未看到等价的独立 NPU 专用模型目录 | vLLM Ascend 可精细建模；SGLang 暂标部分覆盖 |
| Qwen3.7 | 固定提交中没有 `Qwen3.7` 专用注册；有 Qwen3/3.5/3.6 和 Qwen3 DSpark，但不能替代 3.7 | 固定提交中没有可核对的 Qwen3.7 开放实现 | 只能做黑盒/服务级 trace，不能伪造内部 weight 和 layer shape |

vLLM Ascend 的模型注册代码明确注册了 DeepSeek-V4、MiniMax-M3、V4 MTP 和 Qwen3 DSpark：

**源码证据 NPU-MODEL-01｜NPU 专用模型注册**

- 代码位置：[vLLM Ascend `models/__init__.py` L4-L10](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/models/__init__.py#L4-L10)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>models/__init__.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>4
5
6
7
8
9
10</code></pre></td>
<td valign="top"><pre><code class="language-python">def register_model():
    ModelRegistry.register_model("DeepseekV4ForCausalLM", ...)
    ModelRegistry.register_model(
        "MiniMaxM3SparseForCausalLM",
        "vllm_ascend.models.minimax_m3:MiniMaxM3SparseForCausalLM",
    )
    ModelRegistry.register_model("DeepSeekV4MTPModel", ...)</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：DeepSeek-V4、MiniMax-M3 和 V4 MTP 在固定提交中确实绑定到 vLLM Ascend 实现；不是仅依据 Transformers 配置推测。

这个表里的“未看到”是对固定提交的仓库审计结果，不代表厂商内部或后续版本一定不支持。

---

## 3. 昇腾上必须区分的六种 shape

### 3.1 逻辑模型 shape

来自模型配置和权重语义，例如：

```text
hidden_size
num_layers
num_attention_heads / num_kv_heads
head_dim
intermediate_size
num_experts / top_k
vocab_size
index_topk / index_head_dim
compress_ratios
```

它回答“模型是什么”，但不能单独推出单卡执行。

### 3.2 单 rank 分片 shape

由 `TP/EP/PP/CP` 决定，例如：

```text
local_q_heads = num_q_heads / attn_tp
local_experts = physical_experts / ep_size
local_layer_range = PP partition
local_hidden/intermediate shard = logical dim / TP
```

vLLM Ascend 的 MiniMax-M3 会根据 TP 计算本地 Q/KV head；当 KV head 少于 TP 时还会复制 KV head，而不是继续整除：

**源码证据 NPU-SHAPE-01｜TP 决定本地 Q/KV shape**

- 代码位置：[vLLM Ascend `minimax_m3.py` L128-L140](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/models/minimax_m3/minimax_m3.py#L128-L140)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>minimax_m3.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>128
130
131
133
134
135
136
137
139
140</code></pre></td>
<td valign="top"><pre><code class="language-python">tp_size = get_tensor_model_parallel_world_size()
assert self.total_num_heads % tp_size == 0
self.num_heads = self.total_num_heads // tp_size
if self.total_num_kv_heads &gt;= tp_size:
    assert self.total_num_kv_heads % tp_size == 0
else:
    assert tp_size % self.total_num_kv_heads == 0
self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
self.q_size = self.num_heads * self.head_dim
self.kv_size = self.num_kv_heads * self.head_dim</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：TP 不只影响通信；它直接改变本 rank 的 head 数、`q_size` 和 `kv_size`。KV head 少于 TP 时走复制约束。

所以不能从“总卡数”直接得到 shape，必须记录每个 parallel group。

### 3.3 本轮执行 shape

它由请求和调度产生：

```text
T_raw
T_padded
batch bucket
prefill/decode/verify 的 query token 数
seq_lens / prefix_lens
max_num_batched_tokens
speculative width
dummy request 数
```

例如 SGLang NPU graph 预分配的 block table 是：

```python
(max_bs, total_context_len // page_size)
```

并且 speculative tokens 会增加 `total_context_len`。证据：[SGLang `ascend_backend.py` L537-L576](https://github.com/sgl-project/sglang/blob/1b9dfa14e66b617ed53270164549d59290b1f7c8/python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py#L537-L576)

### 3.4 物理 storage shape

这是昇腾版本最需要补充的一层。以二维逻辑 shape `[A,B]` 为例，`FRACTAL_NZ` 可概念化为：

```text
[ceil(B/C0), ceil(A/16), 16, C0]
```

其中 C0 依 dtype/format 而变。Ascend PyTorch 的测试明确验证：

- FP16 非对齐 shape 使用默认 `C0=16`；
- INT8 非对齐 shape 使用默认 `C0=32`；
- int32 定制 dtype 可使用 `C0=8`；
- tensor 的 `out.shape` 可仍等于逻辑 shape，但 `untyped_storage().size()` 已经包含物理 padding。

**源码证据 NPU-FORMAT-01｜逻辑 shape 不变，storage 大小随 dtype/C0 改变**

- 代码位置：[Ascend PyTorch `test_npu_format_cast.py` L300-L323](https://github.com/Ascend/pytorch/blob/10fe1622631286665f15d84951d53df7840e6ead/test/custom_ops/test_npu_format_cast.py#L300-L323)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>test_npu_format_cast.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>302
303
305
308
313
314
317
322
323</code></pre></td>
<td valign="top"><pre><code class="language-python">t = torch.randint(-128, 127, (15, 17), dtype=torch.int8).npu()
out = torch_npu.npu_format_cast(t, self.ACL_FORMAT_FRACTAL_NZ)
self.assertEqual(out.shape, t.shape)
def test_npu_format_cast_fp16_default_c0_16_storage_size(self):
    self.assertEqual(out.untyped_storage().size(),
                     self._expected_nz_storage_bytes(shape, 16, 2))
def test_npu_format_cast_int8_default_c0_32_storage_size(self):
    self.assertEqual(out.untyped_storage().size(),
                     self._expected_nz_storage_bytes(shape, 32, 1))</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：`out.shape == t.shape` 不能代表物理存储相同；FP16 使用 C0=16，INT8 使用 C0=32，并分别校验 storage bytes。

Ascend PyTorch 还提供独立 API 读取 storage sizes，而不是只看 `tensor.shape`：

**源码证据 NPU-FORMAT-02｜storage shape 有独立读取入口**

- 代码位置：[Ascend PyTorch `NPUFormat.cpp` L25-L39](https://github.com/Ascend/pytorch/blob/10fe1622631286665f15d84951d53df7840e6ead/torch_npu/csrc/core/npu/NPUFormat.cpp#L25-L39)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>NPUFormat.cpp</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>25
26
29
31
32
35
38</code></pre></td>
<td valign="top"><pre><code class="language-cpp">int64_t get_npu_format(const at::Tensor&amp; self) {
  return NPUNativeFunctions::get_npu_format(self);
std::vector&lt;int64_t&gt; get_npu_storage_sizes(const at::Tensor&amp; self) {
  auto storage_sizes =
      torch_npu::NPUBridge::GetNpuStorageImpl(self)-&gt;npu_desc_.storage_sizes_;
  return vec_storage_sizes;
at::Tensor npu_format_cast(const at::Tensor&amp; self, int64_t acl_format) {</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：昇腾 tensor descriptor 同时维护 format 和 `storage_sizes_`；recorder 可以直接读取，而不应从逻辑 shape 猜测。

因此 NPU recorder 至少要同时记录：

```text
logical_sizes
storage_sizes
stride
acl_format
dtype
storage_offset
is_contiguous
```

### 3.5 算子 ABI shape

同一逻辑 attention 会被重排成算子要求的 layout：

- 普通 MHA：`TND`、`BSH`、`BNSD`；
- Page Attention：`PA_ND`、`PA_BSND`；
- FIA NZ：五维/分块 cache view；
- sparse attention：额外增加 `topk_indices`、compressed page table、metadata buffer。

SGLang NPU 对常规 cache 的辅助 view 为：

**源码证据 NPU-FORMAT-03｜SGLang FIA NZ 五维 cache view**

- 代码位置：[SGLang `ascend_backend.py` L61-L65](https://github.com/sgl-project/sglang/blob/1b9dfa14e66b617ed53270164549d59290b1f7c8/python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py#L61-L65)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>ascend_backend.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>61
62
63
64
65</code></pre></td>
<td valign="top"><pre><code class="language-python">def _reshape_kv_for_fia_nz(
    tensor: torch.Tensor, num_heads: int, head_dim: int, page_size: int
) -&gt; torch.Tensor:
    """Reshapes a tensor for FIA NZ format."""
    return tensor.view(-1, 1, num_heads * head_dim // 16, page_size, 16)</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：输入逻辑 KV 会被解释成五维 NPU ABI layout，其中第三维由 `num_heads * head_dim // 16` 推导。

vLLM Ascend 的 C8/NZ 路径则把 KV cache 解释成五维视图：

**源码证据 NPU-FORMAT-04｜vLLM Ascend NZ cache view**

- 代码位置：[vLLM Ascend `attention_v1.py` L1772-L1775](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/attention/attention_v1.py#L1772-L1775)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>attention_v1.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>1772
1773

1774
1775

</code></pre></td>
<td valign="top"><pre><code class="language-python">def _nz_5d_view(self, cache: torch.Tensor, block_size: int) -&gt; torch.Tensor:
    """View a KV cache tensor in NZ 5D layout:
    (num_blocks, num_kv_heads, head_size//nz, block_size, nz)."""
    NZ_FMT_LAST_DIM = 32
    return cache.view(-1, self.num_kv_heads,
                      self.head_size // NZ_FMT_LAST_DIM,
                      block_size, NZ_FMT_LAST_DIM)</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：vLLM Ascend 的 NZ cache 末维固定为 32；`head_size` 是否可被 32 合法分块是 operator guard 的一部分。

### 3.6 有效 workload shape

有些 shape 不一定体现在最终 tensor rank 上，却直接决定工作量：

```text
每序列有效 KV 长度
每 expert token 数
每 rank send/recv count
top-k 稀疏块数量
实际压缩 token 数
有效 graph token 数
workspace 大小
tiling key
```

性能回放必须把它们与普通 shape 同级处理。

### 3.7 所有 shape 变化操作怎样纳入扫描

用户前面提到的 `slide()` 如果指 tensor 切片，PyTorch 代码中更常见的是 `x[a:b]`、`narrow/select/index_select`；如果指 sliding-window attention，则还要同时记录 window、mask、prefix 和 page table。本文扫描的操作族如下：

| 操作族 | 典型形式 | NPU 额外关注 |
|---|---|---|
| rank 重解释 | `view/reshape/flatten/squeeze/unsqueeze` | format 是否允许该 view、是否触发 contiguous/copy |
| 轴变换 | `transpose/permute/movedim` | ND/NZ、weight tile 和算子 layout |
| 切分/拼接 | `split/chunk/narrow/cat/stack` | TP/CP 分片、QKV/indexer 拆分、padding 回填 |
| 切片/高级索引 | `x[a:b]`、`gather/scatter/index_select/mask` | slot/page/cache 写入位置和值相关输出长度 |
| 排序和稀疏选择 | `topk/sort/argsort/nonzero/unique` | sparse attention、MoE 和 ragged workload |
| 扩展/重复 | `expand/repeat/repeat_interleave/tile` | KV head 复制、token-to-expert 展开 |
| padding/对齐 | `pad/round_up/ceil` | C0、TP/CP、图 bucket、head/page/token 对齐 |
| reduction/前缀和 | `sum/max/histc/cumsum` | seq segment、expert counts、通信 split |
| 动态分配 | `empty/zeros/full/arange` | graph 固定 buffer、workspace、storage format |
| storage/layout | `contiguous/format_cast/FRACTAL_NZ` | 逻辑 shape 不变但 storage 和 memcpy 改变 |
| cache 重排 | page/block view、slot mapping、block table | 逻辑 token 到物理地址的映射 |
| 量化 packing | W4/W8/FP8 weight 和 scale reshape | C0、group/block、scale/offset shape |
| 分布式通信 | AllGather/AllToAll/ReduceScatter/MC2 | 输出首维和 payload 由 world/split 值决定 |
| fused op 内部 | metadata、mask、tiling、workspace | Python 输出 shape 不变，内部有效工作变化 |

`contiguous()`、format cast、通信和 workspace 虽不一定改变 `tensor.shape`，仍属于性能 shape signature。

---

## 4. DeepSeek-V4：index、压缩 KV 和层配置共同决定路径

### 4.1 A5 与非 A5 已经不是同一个物理签名

vLLM Ascend 的 V4 indexer cache 在 A5 上用 FP8，在其他设备上用 INT8；scale dtype 和 scale 维度也存在设备/`head_dim` 分支：

**源码证据 NPU-DSV4-01｜A5 改变 index cache dtype 与 scale**

- 代码位置：[vLLM Ascend `deepseek_v4.py` L154-L170](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/models/deepseek_v4.py#L154-L170)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>deepseek_v4.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>154
155
156
157
161
164
165
169
170

</code></pre></td>
<td valign="top"><pre><code class="language-python">def get_kv_cache_spec(self, vllm_config):
    if get_ascend_device_type() in {AscendDeviceType.A5}:
        self.dtype = torch.float8_e4m3fn
        vllm_config.cache_config.cache_dtype = "float8_e4m3fn"
    return AscendMLAAttentionSpec(
        head_size=self.head_dim,
        dtype=self.dtype,
        scale_dim=1 if self.head_dim == 128 else 0,
        scale_dtype=torch.float
            if get_ascend_device_type() in {AscendDeviceType.A5}
            else torch.float16,</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：SoC 和 `head_dim` 会改变 cache dtype、scale dtype 和 scale 维；这些字段必须进入物理回放签名。

SWA cache 的物理 head size 在 A5 上还会变成 `head_dim + 128`：

**源码证据 NPU-DSV4-02｜A5 改变 SWA cache head size**

- 代码位置：[vLLM Ascend `deepseek_v4.py` L193-L204](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/models/deepseek_v4.py#L193-L204)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>deepseek_v4.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>193
194
195
197


198
201
202</code></pre></td>
<td valign="top"><pre><code class="language-python">def get_kv_cache_spec(self, vllm_config):
    if get_ascend_device_type() in {AscendDeviceType.A5}:
        self.dtype = torch.float8_e4m3fn
    cached_head_size = self.head_dim + 128 if get_ascend_device_type() in {
        AscendDeviceType.A5
    } else self.head_dim
    return AscendSlidingWindowMLASpec(
        head_size=cached_head_size,
        dtype=self.dtype,</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：同一模型逻辑 `head_dim` 在 A5 上对应不同的 cache `head_size`，不能复用非 A5 的 storage/graph artifact。

这意味着即使模型配置、token 和逻辑 `[num_blocks, block_size, heads, head_dim]` 相同，只要 SoC 代际不同，cache storage 与算子输入就可能不兼容。

### 4.2 C4 top-k index 直接进入 sparse attention

SGLang NPU 为图模式预分配：

**源码证据 NPU-DSV4-03｜图模式 top-k buffer shape**

- 代码位置：[SGLang `ascend_dsv4_backend.py` L1100-L1116](https://github.com/sgl-project/sglang/blob/1b9dfa14e66b617ed53270164549d59290b1f7c8/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L1100-L1116)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>ascend_dsv4_backend.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>1100
1101

1107
1111
1112
1113
1114
1116</code></pre></td>
<td valign="top"><pre><code class="language-python"># 1024 int32 per kernel-metadata buffer
for key in ("kernel_metadata_c1a", "kernel_metadata_c4a",
            "kernel_metadata_c128a", "kernel_metadata_li_quant"):
    self.graph_metadata[key] = torch.zeros(1024, dtype=torch.int32)
self.graph_metadata["c4_topk_indices"] = torch.full(
    (max_num_tokens, self._dsv4_index_topk),
    -1,
    dtype=torch.int32,
)</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：C4 top-k 是图内固定 buffer，shape 明确为 `[max_num_tokens,index_topk]`，初值 `-1` 具有无效索引语义。

执行压缩 attention 时：

```python
if compress_ratio == 4:
    cmp_sparse_indices = topk.view(-1, 1, topk.shape[-1])
else:
    cmp_sparse_indices = None
```

**源码证据 NPU-DSV4-04｜C4 index 直接传入稀疏注意力**

- 代码位置：[SGLang `ascend_dsv4_backend.py` L1745-L1757](https://github.com/sgl-project/sglang/blob/1b9dfa14e66b617ed53270164549d59290b1f7c8/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L1745-L1757)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>ascend_dsv4_backend.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>1745
1747
1748
1750
1751
1752
1753
1754
1755
1756</code></pre></td>
<td valign="top"><pre><code class="language-python">cmp_ratio=compress_ratio,
cmp_kv=cmp_kv,
cmp_block_table=cmp_block_table,
# c4 attends via indexer topk; c128 reads full compressed history
if compress_ratio == 4:
    topk = fm.c4_topk_indices
    attn_kwargs["cmp_sparse_indices"] = topk.view(-1, 1, topk.shape[-1])
else:
    attn_kwargs["cmp_sparse_indices"] = None
out, _ = torch.ops.custom.npu_sparse_attn_sharedkv(**attn_kwargs)</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：C4/C128 不只是参数不同；C4 把 top-k reshape 为 `[T,1,K]` 并作为 fused NPU attention 的输入，C128 明确传 `None`。

所以 C4 与 C128 至少有三层不同：

| 项目 | C4 | C128 |
|---|---|---|
| compressed ratio | 4 | 128 |
| indexer | 执行 | 不执行 |
| `cmp_sparse_indices` | `[T,1,index_topk]` | `None` |
| compressed history | top-k 选择 | 全 compressed history |
| metadata/tiling | `cmp_topk=index_topk` | 无 top-k |

仅记录输出 `[T,H]` 会把这两条完全不同的路径错误合并。

### 4.3 Lightning Indexer 的输出 shape 在算子 host 端确定

`sgl-kernel-npu` 的 Lightning Indexer 不是一个抽象概念。其 host 代码明确按照 layout 构造输出：

```text
BSND -> [B, S1, N2, sparse_count]
其他 -> [T, N2, sparse_count]
```

**源码证据 NPU-INDEX-01｜Lightning Indexer 输出 shape 推导**

- 代码位置：[sgl-kernel-npu `lightning_indexer.cpp` L32-L55](https://github.com/sgl-project/sgl-kernel-npu/blob/3479f4d99cd4e65a1cbe316f8bafc318014a4eb9/csrc/lightning_indexer/op_host/lightning_indexer.cpp#L32-L55)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>lightning_indexer.cpp</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>37
44
46
47

48
49
50
51
52
53</code></pre></td>
<td valign="top"><pre><code class="language-cpp">at::SmallVector&lt;int64_t, SIZE&gt; outputSize;
TORCH_CHECK(sparse_count &gt; 0, ...);
if (query_layout_str == "BSND") {
    outputSize = {query.size(DIM_0), query.size(DIM_1),
                  key.size(DIM_2), sparse_count};
} else {
    int n_dim_index = 0;
    n_dim_index = (key_layout_str == "TND") ? DIM_1 : DIM_2;
    outputSize = {query.size(DIM_0), key.size(n_dim_index), sparse_count};
}
at::Tensor output = at::empty(outputSize, query.options().dtype(at::kInt));</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：输出 rank/extent 由 query/key layout 和 `sparse_count` 联合决定，不是固定二维 top-k。

其 ABI 还包含：

- `actual_seq_lengths_query`；
- `actual_seq_lengths_key`；
- `block_table`；
- `layout_query/layout_key`；
- `sparse_count`；
- `sparse_mode`。

**源码证据 NPU-INDEX-02｜Lightning Indexer ABI**

- 代码位置：[sgl-kernel-npu `lightning_indexer_def.h` L42-L60](https://github.com/sgl-project/sgl-kernel-npu/blob/3479f4d99cd4e65a1cbe316f8bafc318014a4eb9/csrc/lightning_indexer/op_host/lightning_indexer_def.h#L42-L60)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>lightning_indexer_def.h</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>42
47
52
57
58
59
60
</code></pre></td>
<td valign="top"><pre><code class="language-cpp">Input("actual_seq_lengths_query") ... DT_INT32 ... FORMAT_ND
Input("actual_seq_lengths_key")   ... DT_INT32 ... FORMAT_ND
Input("block_table")              ... DT_INT32 ... FORMAT_ND
Output("sparse_indices")          ... DT_INT32 ... FORMAT_ND
this-&gt;Attr("layout_query").AttrType(OPTIONAL).String("TND");
this-&gt;Attr("layout_key").AttrType(OPTIONAL).String("PA_BSND");
this-&gt;Attr("sparse_count").AttrType(OPTIONAL).Int(2048);
    // 2048:默认值，筛选前2048</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：长度、block table、layout 和 sparse count 都是公开的算子 ABI；回放时不能只传 query/key。

其官方算子说明进一步约束：

- query head `N=64`、key head `N=1`；
- `D=128`；
- block size 为 16 的倍数且不超过 1024；
- `sparse_count` 为 1～2048；
- `TND` 下 query 的实际长度必须按 batch 给出累计和。

**接口证据 NPU-INDEX-04｜Lightning Indexer 的合法 shape 域**

- 代码位置：[sgl-kernel-npu `lightning_indexer/README.md` L21-L67](https://github.com/sgl-project/sgl-kernel-npu/blob/3479f4d99cd4e65a1cbe316f8bafc318014a4eb9/csrc/lightning_indexer/README.md#L21-L67)
- 原文约束（中文转写）：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>lightning_indexer/README.md</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>30 ?
37 ?

42 ?
48 ?
64 ?
65 ?
67 ?</code></pre></td>
<td valign="top"><pre><code class="language-text">≈ PA_BSND key shape: [block_count, block_size, N2, D]
≈ TND query requires actual_seq_lengths_query;
values are cumulative token counts and must be non-decreasing
≈ block_table must be 2D: [B, &gt;= maxBlockNumPerSeq]
≈ sparse_count supports 1–2048
≈ query N supports 64; key N supports 1
≈ query/key D must equal 128
≈ block_size must be a multiple of 16 and no greater than 1024</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：这些是算子公开接口约束，超出合法域时应判为 replay invalid，而不是静默迁移。

这些约束都应该进入 replay guard；不满足时不能把同一个录制样本直接套用到新参数。

### 4.4 tiling key 是额外的执行签名

Lightning Indexer 会把下面字段组成 hash：

```text
B, N2, group, S1, S2, block_size, max_blocks_per_batch, tiling_key
```

命中时复用捕获的 tiling 数据；未命中或超过 capture 数量时走不同加载分支。

**源码证据 NPU-INDEX-03｜tiling hash 与 workspace**

- 代码位置：[sgl-kernel-npu `lightning_indexer.cpp` L131-L167](https://github.com/sgl-project/sgl-kernel-npu/blob/3479f4d99cd4e65a1cbe316f8bafc318014a4eb9/csrc/lightning_indexer/op_host/lightning_indexer.cpp#L131-L167)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>lightning_indexer.cpp</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>136
137

138

139
144
148
154
156
164
166</code></pre></td>
<td valign="top"><pre><code class="language-cpp">auto tup =
    std::make_tuple(tilingData.bSize, tilingData.n2Size,
                    tilingData.gSize, tilingData.s1Size, tilingData.s2Size,
                    tilingData.blockSize, tilingData.maxBlockNumPerBatch,
                    tilingData.tilingKey);
auto hashValue = host_utils::TupleHasher::Hash(tup);
if (captureMap.find(hashValue) != captureMap.end()) {
} else if (actualCaptureNum &gt;= MAX_CAPTURE_NUM) {
} else {
    captureMap[hashValue] = actualCaptureNum;
size_t workspaceSize = context-&gt;GetWorkspaceSize();
EXEC_KERNEL_CMD(lightning_indexer, ..., workspace, tilingTensor);</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：相同 Python 算子名会因 shape/tiling 字段命中不同计划；workspace 也是 host 端动态求出的执行量。

这说明：

> NPU 性能回放除了 tensor shape，还需要记录算子的 tiling identity；否则“shape 看起来差不多”仍可能命中另一份 plan。

### 4.5 Prefill、Decode、Verify、Idle 会生成不同累计长度

SGLang NPU 的 V4 backend 按 forward mode 分别构造：

- Prefill/Extend：`cumsum(extend_seq_lens)`；
- Decode：`arange(1, B+1)`；
- Target Verify/Draft Extend：步长为 speculative draft token 数；
- Idle：跳过 metadata kernel 或使用占位长度。

**源码证据 NPU-DSV4-05｜forward mode 改变累计长度值**

- 代码位置：[SGLang `ascend_dsv4_backend.py` L1465-L1505](https://github.com/sgl-project/sglang/blob/1b9dfa14e66b617ed53270164549d59290b1f7c8/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L1465-L1505)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>ascend_dsv4_backend.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>1467
1468
1469
1470
1471
1477
1479
1480
1481
1482
1483
1485
1486
1487
1488
1489
1490
1491
1492
1493
1494
1498
1499
1500

1501</code></pre></td>
<td valign="top"><pre><code class="language-python">if (
    forward_batch.forward_mode.is_extend()
    and not forward_batch.forward_mode.is_draft_extend_v2()
    and not forward_batch.forward_mode.is_target_verify()
):
    actual_q = torch.cumsum(seq_lens_cpu, dim=0).int().to(device)
    fm.actual_seq_lengths_q_pa = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), actual_q],
        dim=0,
    )
elif forward_batch.forward_mode.is_decode():
    fm.actual_seq_lengths_q = torch.arange(
        1, B + 1, dtype=torch.int32, device=device
    )
    fm.actual_seq_lengths_q_pa = torch.arange(
        0, B + 1, dtype=torch.int32, device=device
    )
elif (
    forward_batch.forward_mode.is_target_verify()
    or forward_batch.forward_mode.is_draft_extend_v2()
):
    n_draft = get_server_args().speculative_num_draft_tokens or 1
    actual_q = torch.arange(
        n_draft, B * n_draft + 1, n_draft,
        dtype=torch.int32, device=device
    )</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：这些 tensor 的 shape 可能都只是 `[B]`/`[B+1]`，但值生成公式随 mode 改变，决定 segment 边界。

这类累计长度虽然 shape 常为 `[B]` 或 `[B+1]`，但**值**决定每段 query/KV 的边界，必须完整录制或可精确重建。

### 4.6 compress ratio 是层级状态，不只是命令行参数

SGLang NPU 只接受 `compress_ratio in (0,4,128)`：

- 0：dense/SWA；
- 4：compressed + top-k；
- 128：compressed，无 top-k。

**源码证据 NPU-DSV4-06｜compress ratio 与 Idle 分支**

- 代码位置：[SGLang `ascend_dsv4_backend.py` L1648-L1664](https://github.com/sgl-project/sglang/blob/1b9dfa14e66b617ed53270164549d59290b1f7c8/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L1648-L1664)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>ascend_dsv4_backend.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>1648
1649
1653
1654
1656
1657
1660
1661
1662</code></pre></td>
<td valign="top"><pre><code class="language-python">if compress_ratio not in (0, 4, 128):
    raise ValueError(...)
if forward_batch.forward_mode.is_idle():
    return torch.zeros_like(q)
if save_kv_cache:
    self.store_cache(...)
if compress_ratio == 0:
    return self._forward_dense(...)
return self._forward_compressed(..., compress_ratio)</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：合法域、Idle、cache write、dense/compressed 是四个独立控制点；最终输出 shape 相同也不能合并 trace。

vLLM Ascend 同样根据 layer 的 `compress_ratios`、pattern/frequency 决定是否跳过 top-k、实例化 compressor/indexer，以及 cache dtype：

**源码证据 NPU-DSV4-07｜layer pattern 决定 skip_topk**

- 代码位置：[vLLM Ascend `deepseek_v4.py` L850-L880](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/models/deepseek_v4.py#L850-L880)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>deepseek_v4.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>851
852
853
855
856


865
875
876
878
879</code></pre></td>
<td valign="top"><pre><code class="language-python">assert pattern[0] == "F", "index_topk_pattern must start with 'F'"
if 0 &lt;= indexer_seq_idx &lt; len(pattern):
    skip_topk = pattern[indexer_seq_idx] == "S"
ascend_device_type = get_ascend_device_type()
k_dtype = torch.float8_e4m3fn
    if ascend_device_type == AscendDeviceType.A5
    else torch.bfloat16
dsa_modules = DSAModules(
    indexer=self.indexer,
    compressor=self.compressor,
    topk_indices_buffer=topk_indices_buffer,
    skip_topk=skip_topk,</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：layer pattern 的字符值会决定本层是否重新计算 top-k，并把 index buffer 和 `skip_topk` 一起传入 attention 模块。

因此 layer ID、layer type、compress ratio、是否复用 index 都必须进入 `layer_state`。

---

## 5. GLM-5.2：SFA、shared indexer 与部署拓扑

### 5.1 vLLM Ascend 的官方部署参数已经证明拓扑会变

官方 GLM-5.2 文档给出：

- BF16：2 个 A3 `128G×8` 节点或 4 个 A2 `64G×8` 节点；
- W8A8/W4A8C8：1 个 A3 `128G×8` 节点或 2 个 A2 `64G×8` 节点。

**配置证据 NPU-GLM-01｜量化版本改变最低部署容量**

- 代码位置：[vLLM Ascend `GLM5.2.md` L17-L22](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/docs/source/tutorials/models/GLM5.2.md#L17-L22)
- 原文约束（中文转写）：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>GLM5.2.md</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>19 ?

20 ?
21 ?</code></pre></td>
<td valign="top"><pre><code class="language-text">≈ GLM-5.2 (BF16): 2 × Atlas 800 A3 (128G × 8)
                or 4 × Atlas 800 A2 (64G × 8)
≈ GLM-5.2-w8a8: 1 × A3 (128G × 8) or 2 × A2 (64G × 8)
≈ GLM-5.2-w4a8c8: 1 × A3 (128G × 8) or 2 × A2 (64G × 8)</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：同一架构的 BF16/W8A8/W4A8C8 对节点数和显存容量的要求不同，量化版本必须进入 static key。这里是官方部署文档，不是 Python 源码。

同一文档的单机示例使用 `DP=2, TP=8, EP on`，多机示例改成 `DP=4, TP=8, EP on`，同时修改 HCCL buffer、FlashComm/MC2、max seqs 和 max model len：

**配置证据 NPU-GLM-02｜单机与多机不是同一 topology**

- 代码位置：
  - [单机配置 L125-L145](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/docs/source/tutorials/models/GLM5.2.md#L125-L145)
  - [多机配置 L182-L211](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/docs/source/tutorials/models/GLM5.2.md#L182-L211)
- 原文命令行（去掉续行符）：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>GLM5.2.md — 单机</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>129 ?
130 ?
131 ?
137 ?
138 ?
143 ?


182 ?
184 ?
185 ?
191 ?
196 ?
197 ?
203 ?
204 ?</code></pre></td>
<td valign="top"><pre><code class="language-text">≈ --data-parallel-size 2
≈ --enable-expert-parallel
≈ --tensor-parallel-size 8
≈ --max-num-seqs 12
≈ --max-model-len 135000
≈ --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'

GLM5.2.md — 多机
≈ export HCCL_BUFFSIZE=400
≈ export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
≈ export VLLM_ASCEND_ENABLE_FUSED_MC2=1
≈ --data-parallel-size 4
≈ --tensor-parallel-size 8
≈ --enable-expert-parallel
≈ --max-num-seqs 16
≈ --max-model-len 66000</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：部署规模变化伴随 DP、通信算法、并发和上下文配置变化，不能用“卡数比例”直接缩放一次旧 trace。

这说明“模型 + 量化”仍不能唯一决定 shape；还需要 deployment manifest。

### 5.2 SFA backend 的 KV cache shape

vLLM Ascend 的 SFA backend 把 KV cache 定义为：

```text
(num_blocks, block_size, num_kv_heads, head_size)
```

并限定 block size 为 128。

**源码证据 NPU-GLM-03｜SFA cache shape 与 block size**

- 代码位置：[vLLM Ascend `sfa_v1.py` L122-L150](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/attention/sfa_v1.py#L122-L150)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>sfa_v1.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>123
124
127
128
131
138
141
142
145
146
149
150</code></pre></td>
<td valign="top"><pre><code class="language-python">def get_builder_cls():
    if enable_sfa_dcp_replicated_indexer():
        return AscendSFADCPMetadataBuilder
    return AscendSFAMetadataBuilder
def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, ...):
    return (num_blocks, block_size, num_kv_heads, head_size)
def get_impl_cls():
    if enable_sfa_dcp_replicated_indexer():
        return AscendSFADCPImpl
    return AscendSFAImpl
def get_supported_kernel_block_sizes():
    return [128]</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：DCP 开关同时改变 metadata builder 和实现类；cache rank 为四维，kernel block size 明确限定为 128。

在 DSA context parallel 路径上，backend 还会把 token 数向 TP size 对齐，并对 cos/sin、slot mapping 和长度 metadata 做 padding：

**源码证据 NPU-GLM-04｜CP 本地长度由裁剪和 cumsum 得到**

- 代码位置：[vLLM Ascend `sfa_v1.py` L405-L429](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/attention/sfa_v1.py#L405-L429)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>sfa_v1.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>405
412
413
418
419
420
422
424
426
427</code></pre></td>
<td valign="top"><pre><code class="language-python">num_segs = cum_query_lens.shape[0]
global_start = common_attn_metadata.query_start_loc[:num_segs]
global_end = cum_query_lens
req_local_start = global_start.clamp(min=local_start)
req_local_end = global_end.clamp(max=local_end_with_pad)
num_local_tokens = req_local_end - req_local_start
local_query_lens = torch.cumsum(num_local_tokens.clamp(min=0), dim=0)
local_key_lens = torch.where(num_local_tokens &gt; 0, seq_lens - offset, 0)
actual_seq_lengths_query[:num_segs] = local_query_lens
actual_seq_lengths_key[:num_segs] = local_key_lens</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：CP rank 上的有效 Q/KV 长度是由全局 segment、local slice 和 padding 共同计算的动态值，不能只记录全局 `seq_lens`。

所以 GLM-5.2 的 replay key 需要包含：

```text
enable_dsa_cp
CP/TP rank and world size
block_size
C8 flags
token alignment before/after
local actual_seq_lengths
```

### 5.3 shared indexer 是真实的层间依赖

vLLM Ascend 根据 GLM-5.2 的 shared-indexer 类型初始化哪些层跳过 top-k：

**源码证据 NPU-GLM-05｜GLM-5.2 shared indexer 判定**

- 代码位置：[vLLM Ascend `patch_deepseek_v2.py` L36-L54](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/patch/worker/patch_deepseek_v2.py#L36-L54)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>patch_deepseek_v2.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>36
41
42
44
49
52
53
54</code></pre></td>
<td valign="top"><pre><code class="language-python">def _should_skip_indexer_init(config, prefix, skip_topk):
    if not skip_topk:
        return False
    layer_id = extract_layer_index(prefix)
    # GLM-5.2 describes checkpoint-level shared indexers explicitly.
    indexer_types = getattr(config, "indexer_types", None)
    indexer_type = indexer_types[layer_id] if ... else None
    return isinstance(indexer_type, str) and indexer_type.lower() == "shared"</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：代码用 layer ID 查 `indexer_types`，只有类型为 `shared` 才跳过本层 indexer 初始化。

在 SFA forward 中：

- `skip_topk=True` 时复用已有 index buffer；
- 否则运行 NPU Lightning Indexer；
- 最终 top-k 被传给 sparse attention。

**源码证据 NPU-GLM-06｜复用 top-k 与重新计算的真实分支**

- 代码位置：[vLLM Ascend `sfa_v1.py` L1991-L2022](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/attention/sfa_v1.py#L1991-L2022)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>sfa_v1.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>1991
1992
1994
1995
1996
1997
1998
1999
2001
2011
2012
2014
2018</code></pre></td>
<td valign="top"><pre><code class="language-python">if self.enable_dsa_cp and attn_metadata.dsa_cp_context is not None:
    topk_num_tokens = local_end_with_pad - local_start
else: topk_num_tokens = num_input_tokens or hidden_states.shape[0]
if self.skip_topk:
    topk_indices = self._get_indexcache_topk_indices(topk_num_tokens)
else:
    if not self.has_indexer:
        raise RuntimeError(...)
    topk_indices = self.indexer_select_post_process(...)
    if self.use_index_cache:
        self._update_indexcache_topk_indices(topk_indices)
attn_output = self._execute_sparse_flash_attention_process(
    topk_indices,</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：本层可能从 index cache 读取，也可能运行 indexer 并更新 cache；两条路径最终都把 top-k 交给 sparse attention。

所以“第 N 层的 input shape 一样”不代表该层可独立回放。它可能依赖前一层产生的 index buffer。

---

## 6. MiniMax-M3：本地 head、index cache 和层类型

### 6.1 TP 先改变 Q/KV/indexer 权重 shape

vLLM Ascend 的 MiniMax-M3 attention 初始化会计算：

- 本地 Q head 数；
- KV head 是切分还是复制；
- indexer 的 head/dim；
- fused QKV+indexer projection 是否可用；
- sparse top-k 和 block size。

对应的 TP 代码片段见前文 **NPU-SHAPE-01**；完整初始化上下文：[vLLM Ascend `minimax_m3.py` L101-L240](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/models/minimax_m3/minimax_m3.py#L101-L240)。

这条证据说明权重 manifest 不能只保存全局 `[out,in]`。至少要保存：

```text
global logical shape
sharding dim/rule
local shape
replication factor
fused projection layout
physical format and quant scales
```

### 6.2 index cache 的 shape 和写入位置都依赖运行时 index

MiniMax-M3 的 index cache spec 固定 head size/block size，并构造：

```text
(num_blocks, block_size, head_size)
```

**源码证据 NPU-M3-01｜MiniMax index cache shape**

- 代码位置：[vLLM Ascend `msa_m3.py` L117-L167](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/models/minimax_m3/msa_m3.py#L117-L167)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>msa_m3.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>117
120
121
124
131
132
161
162
163
164
165</code></pre></td>
<td valign="top"><pre><code class="language-python">return [128]
def is_sparse(cls):
    return True
def get_kv_cache_shape(num_blocks, block_size, ..., head_size, ...):
    del num_kv_heads, cache_dtype_str
    return (num_blocks, block_size, head_size)
def get_kv_cache_spec(self, vllm_config):
    return AscendSFAIndexerCacheSpec(
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=1,
        head_size=self.head_dim,</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：MiniMax 的 index cache 是三维 `[blocks,block_size,head_size]`，与普通四/五维 KV cache 不能混用。

模型 forward 会：

1. 只切出本轮真实 token；
2. reshape index key；
3. 按 `slot_mapping` scatter 到 cache；
4. 计算 `topk_idx`；
5. 把该 index 传给 sparse attention。

**源码证据 NPU-M3-02｜slot mapping 写 cache，再生成 top-k**

- 代码位置：
  - [cache 写入 L270-L294](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/models/minimax_m3/minimax_m3.py#L270-L294)
  - [top-k 到 sparse attention L349-L361](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/models/minimax_m3/minimax_m3.py#L349-L361)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>minimax_m3.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>273
274
276
281
287
290
292
293

358
359
360
361</code></pre></td>
<td valign="top"><pre><code class="language-python">num_tokens = main_meta.num_actual_tokens
k_insert = key[:num_tokens].view(-1, self.num_kv_heads, self.head_dim)
DeviceOperator.reshape_and_cache(
    main_meta.slot_mapping[:num_tokens],
flat = idx_cache.view(-1, self.idx_head_dim)
torch.ops._C_ascend.npu_scatter_nd_update_v2(
    index_meta.slot_mapping[:num_tokens].view(-1, 1),
    index_key[:num_tokens].to(flat.dtype),

"""Insert KV, build sparse top-k indices, then run sparse attention."""
self._insert_kv(key, value, index_key)
topk_idx = self.indexer(index_query)
self.impl.forward(self, query, self.kv_cache, topk_idx, attn_output)</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：`num_actual_tokens` 控制切片长度，slot mapping 控制物理写入位置，生成的 `topk_idx` 随后直接进入 sparse attention。

因此必须同时录：

```text
slot_mapping 的值
index key 的值或可重建输入
index cache 的前态
topk_idx 的完整值
prefill/decode 分界
```

### 6.3 fused 与 native 路径由 dtype、device、position 等共同决定

MiniMax-M3 的 QKV/indexer 后处理会检查设备、dtype、position、RoPE 等条件，决定调用 NPU fused `qkv_rmsnorm_rope` 还是 native fallback：

**源码证据 NPU-M3-03｜fused QKV/RMSNorm/RoPE guard**

- 代码位置：[vLLM Ascend `minimax_m3.py` L301-L342](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/models/minimax_m3/minimax_m3.py#L301-L342)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>minimax_m3.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>301
303
304
305
311
314
319
321
322
323
325
329
330</code></pre></td>
<td valign="top"><pre><code class="language-python">qkv, _ = self.qkv_proj(hidden_states)
if self.indexer_proj is None:
    main_qkv = qkv.narrow(-1, 0, main_qkv_size)
    index_q = qkv.narrow(-1, main_qkv_size, self.index_q_size)
else:
    index_q, index_k = index_qk.split([...], dim=-1)
if (main_qkv.device.type != "npu"
    or main_qkv.dtype != torch.bfloat16
    or positions.ndim != 1
    or not self.rotary_emb.is_neox_style):
    q, k, v = main_qkv.split([...], dim=-1)
else:
    q, k, v = torch.ops.vllm.qkv_rmsnorm_rope(...)</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：相同输出 shape 下，device/dtype/position/rope style 会切换 native split 路径和 NPU fused op。

这种分支的最终 tensor shape 可能完全一样，但算子数、memory traffic 和性能不同。回放必须记录 guard 的输入值和 branch ID。

### 6.4 层号决定 sparse/dense 与 MoE/dense

MiniMax-M3 根据 layer index 选择：

- sparse attention 或 dense attention；
- MoE 或 dense MLP。

**源码证据 NPU-M3-04｜层号选择 sparse/dense 与 MoE/dense**

- 代码位置：[vLLM Ascend `minimax_m3.py` L705-L762、L783-L789](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/models/minimax_m3/minimax_m3.py#L705-L789)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>minimax_m3.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>705
711
712
732
733
738
739
748
750
751
756
757
786
787
788
789</code></pre></td>
<td valign="top"><pre><code class="language-python">layer_idx = int(prefix.split(sep=".")[-1])
if sparse_attention_config is not None:
    is_sparse_attention_layer = layer_idx in _sparse_attention_layer_ids(config)
if is_sparse_attention_layer:
    self.self_attn = MiniMaxM3SparseAttention(...)
else:
    self.self_attn = MiniMaxM3Attention(...)
self.is_layer_sparse = moe_layer_freq[layer_idx] != 0 if ... else True
if self.is_layer_sparse:
    self.block_sparse_moe = MiniMaxM3MoE(...)
else:
    self.mlp = MiniMaxM3MLP(...)
if self.is_layer_sparse:
    hidden_states = self.block_sparse_moe(hidden_states)
else:
    hidden_states = self.mlp(hidden_states)</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：layer ID 同时决定 attention 类型和 FFN 类型，因此层号本身是 path key。

所以 layer 不能只作为日志标签；它是 path signature 的组成部分。

---

## 7. token 非零值、重复值和特殊值：应该怎样理解

### 7.1 token id 为 0 不等于 padding

普通 tokenizer 的 token ID 0 可以是合法词表项。是否 padding 应由以下字段确定：

```text
attention/causal mask
seq_lens
query_lens
position
request boundary
slot_mapping
框架的 dummy/padding 标记
```

不能用 `input_ids != 0` 推断有效 token 数。

在昇腾代码里，真正有特殊语义的值包括：

- page table 中的 `-1`：无效 page sentinel；
- MoE top-k 中的 `-1`：该 token 不分发；
- 图模式补出的 seq length `0`：dummy request；
- cache location/slot mapping 的无效或占位项。

例如 SGLang V4 graph buffer 用 `-1` 初始化 page table 和 top-k：

对应的 page table/top-k buffer 源码见 **NPU-DSV4-03**；完整上下文：[SGLang `ascend_dsv4_backend.py` L1076-L1116](https://github.com/sgl-project/sglang/blob/1b9dfa14e66b617ed53270164549d59290b1f7c8/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L1076-L1116)。

### 7.2 “非零数量相同”远远不够

即使两批输入：

```text
shape 相同
非零 token 个数相同
重复率相同
```

它们仍可产生不同：

- embedding/hidden states；
- Q/K/V 和 sparse index；
- MoE expert ID；
- prefix-cache 命中；
- speculative accepted token；
- EOS/stop 分支；
- 每序列长度和 block table；
- HCCL split sizes。

所以 `nonzero_count` 和 `duplicate_ratio` 适合做统计特征，不适合代替完整动态状态。

### 7.3 DeepSeek-V4 有直接依赖 token ID 的 MoE 路径

vLLM Ascend 的专家选择代码在特定 scoring/hash 路径中会：

1. all-gather、pad 或按通信模式处理 `input_ids`；
2. 把无效 ID `-1` 替换为 0；
3. 调用 `moe_gating_top_k_hash`，让 token ID 直接参与专家选择。

**源码证据 NPU-TOKEN-01｜input IDs 直接进入 MoE hash gating**

- 代码位置：[vLLM Ascend `experts_selector.py` L247-L286](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/ops/fused_moe/experts_selector.py#L247-L286)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>experts_selector.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>247
248
249
250
253
255
257
265
269
270
273
274
286</code></pre></td>
<td valign="top"><pre><code class="language-python">if scoring_func == "sqrtsoftplus":
    if tid2eid is not None:
        forward_context = get_forward_context()
        input_ids = forward_context.input_ids.to(torch.int64)
        if forward_context.moe_comm_type == MoECommType.ALLGATHER:
            input_ids = prepare_finalize.all_gather_input_id_with_dp_group(input_ids)
        else: input_ids = moe_comm_method.pad_and_split_input_ids(input_ids)
        input_ids = torch.where(input_ids == -1, 0, input_ids)
    topk_weights, topk_ids, _ = torch.ops._C_ascend.moe_gating_top_k_hash(
        x=router_logits,
        input_ids=input_ids,
        tid2eid=tid2eid_ones,
    return topk_weights, topk_ids</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：该分支不仅依赖 hidden/router logits；`input_ids` 会经过通信/切分并直接作为 hash gating 输入。

这比“token 先影响 hidden，再间接影响 router”更强：在该路径里，token ID 本身就是算子输入。回放不能只保留 router 输出 shape。

### 7.4 重复 token 何时影响性能

重复 token 本身通常不会让 dense GEMM 少算；`[T,H]` 仍然执行 T 行。它可能通过以下机制改变性能：

| 机制 | 重复值可能产生的影响 | 回放所需字段 |
|---|---|---|
| prefix/radix cache | 更长共享前缀，减少新 Prefill token | cache hit/miss、matched prefix、new token 数 |
| MoE router/hash | expert 分布更集中或更分散 | top-k IDs/weights、expert counts |
| sparse indexer | 选择不同 KV 位置 | top-k index 全值 |
| sampling/EOS | 请求提前结束，下一轮 batch 改变 | sampled token、finished mask |
| speculative decode | 接受长度改变 | accepted count、verify width |
| 同 token 不同 position | RoPE/KV 不同，不能视为相同 | position、request/sequence ID |

因此建议同时记录：

```text
token_stats: nonzero_count, unique_count, run_length, duplicate_ratio
semantic_values: input_ids, positions, request_offsets
derived_values: prefix_hit, router_topk, sparse_topk, accepted_tokens
```

前者用于聚类，后两类用于精确或约束回放。

---

## 8. MoE：token 值如何变成动态通信 shape

### 8.1 top-k 输出的二维 shape 相同，值却决定后续 ragged shape

vLLM Ascend 的 expert selector 输出：

```text
topk_weights [T,K]
topk_ids     [T,K]
```

混合专家放置还会把 shared experts 追加到第二维，直接改变 K：

**源码证据 NPU-MOE-01｜mix placement 直接扩展 top-k 第二维**

- 代码位置：[vLLM Ascend `experts_selector.py` L114-L131](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/ops/fused_moe/experts_selector.py#L114-L131)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>experts_selector.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>114
116
117
118
119
121
122
128
129
131</code></pre></td>
<td valign="top"><pre><code class="language-python">if mix_placement:
    batch_size = topk_ids.shape[0]
    pad_shared_expert_ids = torch.arange(
        num_logical_experts, num_logical_experts + num_shared_experts, ...
    ).repeat(batch_size, 1)
    pad_shared_expert_weights = torch.full(
        (topk_weights.shape[0], num_shared_experts), ...
    topk_ids = torch.cat([topk_ids, pad_shared_expert_ids], dim=1)
    topk_weights = torch.cat([topk_weights, pad_shared_expert_weights], dim=1)
return topk_weights, topk_ids</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：启用 mix placement 后，shared expert IDs/weights 被拼到 dim=1，`[T,K]` 的 K 会增大。

SGLang NPU 也会根据 scoring function、group、bias、renormalize 和自定义函数选择不同 NPU top-k 算子或 native fallback：

**源码证据 NPU-MOE-02｜scoring 配置选择不同 top-k 实现**

- 代码位置：[SGLang `moe/topk.py` L46-L132](https://github.com/sgl-project/sglang/blob/1b9dfa14e66b617ed53270164549d59290b1f7c8/python/sglang/srt/hardware_backend/npu/moe/topk.py#L46-L132)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>moe/topk.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>48
54
70
71
85
87
88
90
116
117
118
128
132</code></pre></td>
<td valign="top"><pre><code class="language-python">if topk_config.scoring_func == "sqrtsoftplus":
    topk_weights, topk_ids, _ = torch.ops.custom.npu_moe_gating_top_k(...)
elif not use_grouped_topk and correction_bias is None:
    topk_weights, topk_ids, _ = torch.ops.npu.npu_moe_gating_top_k_softmax(...)
elif (correction_bias is not None
      or topk_config.scoring_func == "sigmoid"
      or num_token_non_padded is not None):
    topk_weights, topk_ids, _ = torch.ops.npu.npu_moe_gating_top_k(...)
else:
    topk_config.torch_native = True
    return select_experts(...)
topk_ids = topk_ids_logical_to_physical(topk_ids, ...)
return StandardTopKOutput(topk_weights, topk_ids, router_logits)</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：scoring、bias、group 和有效 token metadata 会改变算子选择；logical expert ID 还可能被映射成 physical ID。

### 8.2 expert counts 直接生成 All-to-All 的 input/output splits

SGL Kernel NPU 的 DeepEP normal strategy 对 `topk_idx` 做 histogram，随后计算：

```text
input_splits  = 本 rank 发往各 EP rank 的 token 数
output_splits = 本 rank 从各 EP rank 收到的 token 数
num_tokens_per_expert
```

**源码证据 NPU-MOE-03｜top-k 值生成 expert/rank split**

- 代码位置：[sgl-kernel-npu `normal_strategy.py` L462-L527](https://github.com/sgl-project/sgl-kernel-npu/blob/3479f4d99cd4e65a1cbe316f8bafc318014a4eb9/python/deep_ep/deep_ep/strategies/normal_strategy.py#L462-L527)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>normal_strategy.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>464
465
469
470
471
473
474
475
481
482
483
494
521
523
524</code></pre></td>
<td valign="top"><pre><code class="language-python">group_size = self.group_size
num_local_experts = num_experts // group_size
num_local_tokens_per_expert = torch.histc(
    topk_idx, bins=num_experts, min=0, max=num_experts
)
input_splits = (
    num_local_tokens_per_expert.reshape(group_size, num_local_experts)
    .sum(axis=1) ...
num_global_tokens_per_expert = self._gather_along_first_dim(
    num_local_tokens_per_expert, group
).reshape(group_size, num_experts)
output_splits = num_global_tokens_per_local_expert.sum(axis=-1)...
self._alltoall_layout = {
    "input_splits": input_splits,
    "output_splits": output_splits,</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：`topk_idx` 的值经 histogram 变成 `input_splits/output_splits`；即使 `[T,K]` shape 相同，通信 shape 仍会不同。

随后 token 被 permute，`num_out_tokens=topk_idx.numel()`，再按 `input_splits/output_splits` 做 All-to-All：

**源码证据 NPU-MOE-04｜split 值进入真实 All-to-All**

- 代码位置：[sgl-kernel-npu `normal_strategy.py` L569-L625](https://github.com/sgl-project/sgl-kernel-npu/blob/3479f4d99cd4e65a1cbe316f8bafc318014a4eb9/python/deep_ep/deep_ep/strategies/normal_strategy.py#L569-L625)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>normal_strategy.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>571
573
574
580
581
583
585
586
600
601
602
603
616
617
623
624</code></pre></td>
<td valign="top"><pre><code class="language-python">layout = self._alltoall_layout
input_splits = layout["input_splits"]
output_splits = layout["output_splits"]
hidden_shape = x.shape
x = x.view(-1, hidden_shape[-1])
permutated_tokens, reversed_local_mapping = torch_npu.npu_moe_token_permute(
    indices=topk_idx,
    num_out_tokens=topk_idx.numel(),
_, global_input_tokens, handle_a2a = self._async_all_to_all(
    permutated_tokens,
    output_splits,
    input_splits,
dispatch_out, reversed_global_mapping = torch_npu.npu_moe_token_permute(
    global_input_tokens, global_tokens_indices
num_recv_tokens_per_expert_list = (
    num_global_tokens_per_local_expert.sum(axis=0)...</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：split 不是分析统计，它被传入实际 All-to-All，并决定接收 tensor 和第二次 permute 的长度。

这条链路是本报告最重要的值依赖之一：

```text
token/hidden
  -> router scores
  -> topk_idx [T,K]
  -> histogram per expert
  -> input/output split values
  -> receive tensor first dimension
  -> local expert GEMM M
  -> HCCL payload and latency
```

所以 MoE 性能回放至少要保留：

```text
topk_idx / topk_weights
per_expert_counts
per_rank_send_counts / recv_counts
permutation / reverse mapping
local expert active M
quant scale shape
comm algorithm and HCCL buffer
```

### 8.3 EP 大小改变算法，而不只是切分数

vLLM Ascend 会根据 EP size 和配置选择：

- All-to-All；
- All-Gather；
- MC2；
- Fused MC2。

**源码证据 NPU-MOE-05｜EP 大小改变可选通信实现**

- 代码位置：[vLLM Ascend `moe_comm_method.py` L58-L66](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/ops/fused_moe/moe_comm_method.py#L58-L66)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>moe_comm_method.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>58
59
60
61
62
63
64
65</code></pre></td>
<td valign="top"><pre><code class="language-python">def setup_moe_comm_method(moe_config):
    if moe_config.ep_size &gt; 1:
        methods[ALLTOALL] = AlltoAllCommImpl(moe_config)
        methods[ALLGATHER] = AllGatherCommImpl(moe_config)
        methods[MC2] = MC2CommImpl(moe_config)
        methods[FUSED_MC2] = FusedMC2CommImpl(moe_config)
    else:
        methods[ALLGATHER] = AllGatherCommImpl(moe_config)</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：`ep_size==1` 与 `ep_size>1` 的实现集合不同；不能仅把通信字节设为 0/非 0。

其通信工具中：

- All-to-All 输出首维为 `sum(output_split_sizes)`；
- All-Gather 首维按 world size 或 split sizes 扩大；
- world size 为 1 时直接 bypass。

**源码证据 NPU-MOE-06｜通信输出首维的具体公式**

- 代码位置：[vLLM Ascend `comm_utils.py` L34-L44、L86-L100](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/ops/fused_moe/comm_utils.py#L34-L100)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>comm_utils.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>35
37
38
40
41

86
88
89
91
92
93
98</code></pre></td>
<td valign="top"><pre><code class="language-python">if output_split_sizes is None:
    a2a_out = torch.empty_like(input_)
else:
    a2a_out = input_.new_empty(
        size=[sum(output_split_sizes)] + list(input_.size()[1:]),

world_size = torch.distributed.get_world_size(group)
if world_size == 1:
    return input_
dim_size = list(input_.size())
if output_split_sizes is None:
    dim_size[0] = dim_size[0] * world_size
else: dim_size[0] = sum(output_split_sizes)</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：All-to-All-V/All-Gather 的输出首维分别依赖 split sum 或 world size；这是可直接重建的 shape 公式。

因此 64 卡到 128 卡时，不能用一个线性倍数修正通信时间；必须重新确定 EP group、算法、split 分布和拓扑。

### 8.4 低时延 DeepEP 会先 pad 到配置上限

低时延策略会把：

- hidden `[T,H]`；
- top-k `[T,K]`；
- active mask；

pad 到 `num_max_dispatch_tokens_per_rank`。

**源码证据 NPU-MOE-07｜低时延 dispatch 的固定容量 padding**

- 代码位置：[sgl-kernel-npu `low_latency_strategy.py` L227-L274](https://github.com/sgl-project/sgl-kernel-npu/blob/3479f4d99cd4e65a1cbe316f8bafc318014a4eb9/python/deep_ep/deep_ep/strategies/low_latency_strategy.py#L227-L274)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>low_latency_strategy.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>228
233
234
239
245
246
252
259
268
269
270</code></pre></td>
<td valign="top"><pre><code class="language-python">x_active_mask = torch.zeros(num_max_dispatch_tokens_per_rank, ...)
x_active_mask[: x.size(0)] = True
padding_size = num_max_dispatch_tokens_per_rank - x.size(0)
x_padding = torch.empty(padding_size, x.size(1), ...)
x_padding = torch.cat((x, x_padding), dim=0)
topk_padding = torch.empty(padding_size, topk_ids.size(1), ...)
topk_padding = torch.cat((topk_ids, topk_padding), dim=0)
weight_padding = torch.cat((topk_weights, weight_padding), dim=0)
self._npu_low_latency_dispatch(
    x=x_padding,
    topk_idx=topk_padding,</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：实际 dispatch 容量是配置上限，不是原始 T；active mask、hidden、top-k、weight 被同步 padding。

所以这里必须区分：

```text
T_raw
T_active
T_dispatch_capacity
padding_size
```

只记录最终 `[T,H]` 输出会漏掉大量固定容量通信与 kernel 工作。

### 8.5 fused Deep MoE 还有 weight layout 和模式约束

`fused_deep_moe` 接收：

- `x [bs,hidden]`；
- `topk_idx/topk_weights [bs,K]`；
- GMM1/GMM2 weight 和 scale；
- 每 rank 最大 dispatch token；
- fuse mode。

不同 fuse mode 对 GMM 权重要求 tile-N permutation 或 NZ format，并返回不同 expert count shape：

**源码证据 NPU-MOE-08｜fused MoE 的 weight layout 与返回 shape**

- 代码位置：[sgl-kernel-npu `buffer.py` L807-L861](https://github.com/sgl-project/sgl-kernel-npu/blob/3479f4d99cd4e65a1cbe316f8bafc318014a4eb9/python/deep_ep/deep_ep/buffer.py#L807-L861)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>buffer.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>808
810
812
815
816
818
828
851
853
857
859</code></pre></td>
<td valign="top"><pre><code class="language-python">x: [bs, hidden]
topk_idx: [bs, num_topk], int64; -1 means no expert selected
topk_weights: [bs, num_topk], float32
gmm1_permuted_weight:
  FUSED_DEEP_MOE requires tile-N permuted layout
  DISPATCH_FFN_COMBINE uses standard NZ format
num_max_dispatch_tokens_per_rank controls buffer allocation
FUSED_DEEP_MOE output: [bs, hidden]
ep_recv_count: [num_local_experts * num_ranks]
DISPATCH_FFN_COMBINE output: [bs, hidden]
expert_token_nums: [num_local_experts]</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：两个 fuse mode 的最终 output 都是 `[bs,hidden]`，但权重物理 layout 和附加 count tensor shape 不同。

这再次说明 weight 的“逻辑 shape 相同”不足以复用录制结果。

---

## 9. 权重与量化：必须记录逻辑、分片、物理格式和 scale

### 9.1 DeepSeek-V4 MoE 权重受 TP/EP 和 physical experts 共同影响

vLLM Ascend 会计算：

```text
moe_tp_size / moe_ep_size
logical experts / physical experts
expert start/end
shared expert 是否启用
```

并创建与 `vocab_size × num_experts_per_tok` 有关的 token-to-expert/hash 权重。

**源码证据 NPU-WEIGHT-01｜logical/physical/local expert shape**

- 代码位置：[vLLM Ascend `deepseek_v4.py` L364-L429](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/models/deepseek_v4.py#L364-L429)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>deepseek_v4.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>364
371
373
374
391
392
393
394
396
397
418
423</code></pre></td>
<td valign="top"><pre><code class="language-python">self.tp_size = get_tensor_model_parallel_world_size()
self.ep_group = get_ep_group().device_group
self.ep_size = self.ep_group.size()
self.n_routed_experts = config.n_routed_experts
self.n_redundant_experts = eplb_config.num_redundant_experts
self.n_logical_experts = self.n_routed_experts
self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
self.n_local_physical_experts = self.n_physical_experts // self.ep_size
self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
self.physical_expert_end = self.physical_expert_start + self.n_local_physical_experts
self.hash = layer_idx &lt; config.num_hash_layers and not is_draft_layer
torch.zeros(config.vocab_size, config.num_experts_per_tok, dtype=torch.int32)</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：本地 expert 数由 physical experts 和 EP size 决定；冗余专家会让 physical 与 logical expert 数不同；hash 层还创建 `[vocab_size,K]` 权重。

所以权重 manifest 要区分：

```text
logical_experts
physical_experts
local_expert_ids
shared_expert placement
replicated vs sharded tensors
```

### 9.2 NZ policy 由 dtype、设备和配置决定

vLLM Ascend 的 NZ 策略会检查：

- FP32；
- meta tensor；
- 310P；
- BF16/FP16；
- quantization；
- NZ 配置模式。

最后可能调用 `torch_npu.npu_format_cast(...FRACTAL_NZ)`。

**源码证据 NPU-WEIGHT-02｜NZ format 的 dtype/设备/config guard**

- 代码位置：[vLLM Ascend `utils.py` L253-L290](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/utils.py#L253-L290)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>utils.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>253
255
256
259
260
263
264
268
271
272
275
276
287
288
289
290</code></pre></td>
<td valign="top"><pre><code class="language-python">def _should_trans_nz(weight):
    if weight.dtype == torch.float32:
        return False
    if weight.is_meta:
        return False
    if is_310p():
        return True
    nz_mode = config.weight_nz_mode
    if not nz_mode:
        return False
    if weight.dtype in {torch.bfloat16, torch.float16}:
        return nz_mode == 2
def maybe_trans_nz(weight):
    if not _should_trans_nz(weight):
        return weight
    return torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ)</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：是否转换 NZ 不是只看 shape；FP32、meta、310P、BF16/FP16 和配置分别走不同 guard。

同一文件还给出 ND→NZ 的显式 reshape/permute 推导；INT8 的内块宽度为 32，其他常见类型为 16：

**源码证据 NPU-WEIGHT-03｜ND→NZ 物理 shape 公式**

- 代码位置：[vLLM Ascend `utils.py` L1460-L1482](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/utils.py#L1460-L1482)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>utils.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>1463
1464
1467
1468
1469
1470
1472


1480
1481</code></pre></td>
<td valign="top"><pre><code class="language-python">batch = cache_tensor.shape[:-2]
a, b = cache_tensor.shape[-2], cache_tensor.shape[-1]
if dtype == torch.int8:
    a0, b0 = 16, 32
else:
    a0, b0 = 16, 16
nz_shape = list(batch) + [
    math.ceil(b / b0), math.ceil(a / a0), a0, b0
]
cache_tensor = cache_tensor.reshape(nz_shape[:-4] + [m1, m0, n1, n0])
cache_tensor = cache_tensor.permute(*array_trans)</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：最后两维向上取整，并且 INT8/其他 dtype 的 `b0` 不同；`reshape+permute` 共同形成物理 layout。

### 9.3 scale 本身也有 shape

FP8/block quant 权重 scale 初始为：

```text
[out_features / block_n, in_features / block_k]
```

后续还会 repeat、reshape、transpose；MoE W4A8MXFP 的 scale 是带 expert 维的三维 tensor。

**源码证据 NPU-WEIGHT-04｜block scale 和 MoE scale shape**

- 代码位置：[vLLM Ascend `fp8.py` L49-L78、L92-L130](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/quantization/methods/fp8.py#L49-L130)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>fp8.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>53
54
61
63

65
66
96
97
99
103
104
</code></pre></td>
<td valign="top"><pre><code class="language-python">weight_scale = torch.empty(
    output_size // block_size, input_size // block_size, dtype=float32)
weight_scale = weight_scale.view(torch.int32) &gt;&gt; 23 &amp; 0xFF
weight_scale = weight_scale.repeat_interleave(4, dim=1)
                           .repeat_interleave(128, dim=0)
weight_scale = weight_scale.reshape(n_dim, k_dim // 2, 2)
weight = weight.transpose(0, 1)
w13_weight_scale = torch.empty(
    num_experts, 2 * intermediate_size_per_partition,
    hidden_sizes // group_size, ...)
w2_weight_scale = torch.empty(
    num_experts, hidden_sizes,
    intermediate_size_per_partition // group_size, ...)</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：scale 是有 rank/extent 的真实 tensor；block size、expert 数、local intermediate 和 group size 都参与 shape。

W8A8 MoE 的物理权重和 scale 形如：

```text
w13 [E, 2I, H]
w2  [E, H, I]
scale/offset [E, 2I, 1] / [E, H, 1]
```

**源码证据 NPU-WEIGHT-05｜W8A8 MoE 权重与 scale shape**

- 代码位置：[vLLM Ascend `w8a8_dynamic.py` L185-L209](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/quantization/methods/w8a8_dynamic.py#L185-L209)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>w8a8_dynamic.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>189
190
191
192
193
194
201
202
207</code></pre></td>
<td valign="top"><pre><code class="language-python">w13_weight = torch.empty(
    num_experts, 2 * intermediate_size_per_partition, hidden_sizes,
    dtype=torch.int8)
w2_weight = torch.empty(
    num_experts, hidden_sizes, intermediate_size_per_partition,
    dtype=torch.int8)
w13_weight_scale = torch.empty(
    num_experts, 2 * intermediate_size_per_partition, 1, ...)
w2_weight_scale = torch.empty(num_experts, hidden_sizes, 1, ...)</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：权重与 scale 的 expert/intermediate/hidden 轴不同；不能用单一 `[out,in]` 描述量化 MoE 权重。

### 9.4 算子上限会触发拆分分支

当某个量化 projection 的输出维达到 NPU quant matmul 限制，并启用 DSA CP 时，vLLM Ascend 会把权重拆成两份，并删除原权重；否则走 NZ/scale flatten 路径：

**源码证据 NPU-WEIGHT-06｜65536 阈值触发权重二分**

- 代码位置：[vLLM Ascend `w8a8_dynamic.py` L126-L151](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/quantization/methods/w8a8_dynamic.py#L126-L151)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>w8a8_dynamic.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>127
128
131
132
134
135
136
137
142
145
148
149</code></pre></td>
<td valign="top"><pre><code class="language-python">layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
if "wq_b" in layer.prefix and layer.weight.shape[1] &gt;= 65536 and enable_dsa_cp():
    chunk_size = layer.weight.shape[1] // 2
    assert chunk_size &lt; 65536
    layer.weight_1 = maybe_trans_nz(layer.weight.data[:, :chunk_size].contiguous())
    layer.weight_2 = maybe_trans_nz(layer.weight.data[:, chunk_size:].contiguous())
    layer.weight_1_scale = layer.weight_scale.data[:chunk_size].flatten()
    layer.weight_2_scale = layer.weight_scale.data[chunk_size:].flatten()
    del layer.weight
else:
    layer.weight.data = maybe_trans_nz(layer.weight.data)
    layer.weight_scale.data = layer.weight_scale.data.flatten()</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：`prefix + weight.shape[1] + DSA-CP` 联合决定是二分权重还是单 tensor NZ 路径，属于必须记录的 guard。

这类 guard 需要录制：

```text
original logical shape
threshold
guard result
split count and split shapes
post-load physical format
```

---

## 10. Attention/KV：block table、slot mapping 和 layout 是值相关状态

### 10.1 vLLM Ascend DSA backend 的 cache spec

DSA backend 定义：

```text
KV cache: (num_blocks, block_size, num_kv_heads, head_size)
scale:    与量化/cache backend 有关
```

并给出支持的 block size。

**源码证据 NPU-KV-01｜DSA cache/scale shape**

- 代码位置：[vLLM Ascend `dsa_v1.py` L201-L231](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/attention/dsa_v1.py#L201-L231)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>dsa_v1.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>202
205
208
209
212
213
216
217
223
226
227
230
231</code></pre></td>
<td valign="top"><pre><code class="language-python">def get_builder_cls():
    if enable_dsa_cp():
        return AscendDSACPMetadataBuilder
    return AscendDSAMetadataBuilder
def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
    return num_blocks, block_size, num_kv_heads, head_size
def get_scale_shape(num_blocks, block_size, scale_size):
    return num_blocks, block_size, scale_size
if enable_dsa_cp():
    return AscendDSACPImpl
return AscendDSAImpl
def get_supported_kernel_block_sizes():
    return [2, 4, 8, 16, 32, 64, 128]</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：cache 与 scale 是不同 rank 的 tensor；DSA-CP 还会替换 builder/impl，block size 有明确离散合法集合。

普通 attention backend 则可能使用带 K/V 维的：

```text
(2, num_blocks, block_size, num_kv_heads, head_size)
```

**源码证据 NPU-KV-02｜普通 attention cache 多一个 K/V 轴**

- 代码位置：[vLLM Ascend `attention_v1.py` L99-L107、L123-L139](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/attention/attention_v1.py#L99-L139)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>attention_v1.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>100
103
107
124
128
129
131
132
133
134
135
138
139</code></pre></td>
<td valign="top"><pre><code class="language-python">def get_kv_cache_shape(num_blocks, block_size,
                       num_kv_heads, head_size, ...):
    return (2, num_blocks, block_size, num_kv_heads, head_size)
def copy_blocks(kv_caches, src_to_dists):
    src_indices = src_to_dists[:, 0]
    dst_indices = src_to_dists[:, 1]
    for kv_cache in kv_caches:
        key_caches = kv_cache[0]
        value_caches = kv_cache[1]
        key_caches[dst_indices] = key_caches[src_indices]
        value_caches[dst_indices] = value_caches[src_indices]
def get_supported_kernel_block_sizes():
    return [128]</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：普通 backend 的 cache shape 是 `[2,blocks,block,kv_heads,head]`；copy 的 src/dst index 值还决定物理 block 重排。

因此不能假定所有模型、backend 的 KV cache rank 相同。

### 10.2 slot mapping 在 A5 和其他设备上 shape 不同

vLLM Ascend DSA metadata builder 中：

- A5 使用一维 slot mapping；
- 其他设备使用 `[max_tokens,2]`；
- speculative tokens 还会额外分配 buffer；
- NPU FIA TND decode 对 token 数有限制。

**源码证据 NPU-KV-03｜A5/非 A5 slot mapping shape**

- 代码位置：[vLLM Ascend `dsa_v1.py` L459-L484](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/attention/dsa_v1.py#L459-L484)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>dsa_v1.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>461
464
466
467
468
469
470
471
472
473
479
480</code></pre></td>
<td valign="top"><pre><code class="language-python">self.max_blocks = (max_model_len + block_size - 1) // block_size
self.decode_threshold = 1
if get_ascend_device_type() in {AscendDeviceType.A5}:
    self.slot_mapping_shape = (max_num_batched_tokens,)
else:
    self.slot_mapping_shape = (max_num_batched_tokens, 2)
if self.speculative_config:
    spec_token_num = self.speculative_config.num_speculative_tokens
    self.spec_slot_mapping = [
        torch.zeros(self.slot_mapping_shape, dtype=torch.int32, ...)
    self.decode_threshold += spec_token_num
    assert self.decode_threshold &lt;= 16</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：SoC 改变 slot mapping rank；speculative token 数改变 buffer 数量和 decode threshold。

### 10.3 seq_lens 的值决定 gather、block table 和 decode/prefill 切分

DSA metadata 会从 schedule 结果生成：

```text
query_lens
prefix_lens
start offsets
visible_lens
block table
slot ids
repeat_interleave index
```

**源码证据 NPU-KV-04｜seq/block table 生成真实 slot IDs**

- 代码位置：[vLLM Ascend `dsa_v1.py` L373-L426](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/attention/dsa_v1.py#L373-L426)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>dsa_v1.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>374
375
404
405
406
407
414
417
419
420
421
423</code></pre></td>
<td valign="top"><pre><code class="language-python">min_width = int(window_size) + int(block_size)
return ((min_width + alignment - 1) // alignment) * alignment
query_lens = query_start_loc[1:] - query_start_loc[:-1]
prefix_lens = seq_lens - query_lens
start_pos = (prefix_lens - int(window_size)).clamp(min=0)
visible_lens = seq_lens - start_pos
block_nums = pos // block_size
safe_nums = block_nums.clamp(max=int(block_table.shape[1]) - 1)
block_ids = torch.gather(block_table, 1, safe_nums)
slot_ids = (block_ids * block_size + block_offsets).to(torch.int32)
slot_ids = slot_ids.where(col_mask, torch.full_like(slot_ids, -1))
per_token_slots = torch.repeat_interleave(slot_ids, query_lens, dim=0, ...)</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：slot ID 是由长度、窗口、block table 值和 block size 共同计算的；`-1` 是明确的无效位置。

随后按 scheduled token 数和阈值重排 Decode/Prefill，并构造独立 metadata：

**源码证据 NPU-KV-05｜scheduled token 数改变 Decode/Prefill 排列**

- 代码位置：[vLLM Ascend `dsa_v1.py` L553-L598](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/attention/dsa_v1.py#L553-L598)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>dsa_v1.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>560
561
563
564
565
566
567
568
585
588
589
598</code></pre></td>
<td valign="top"><pre><code class="language-python">decodes = []
prefills = []
for i, req_id in enumerate(input_batch.req_ids):
    num_tokens = scheduler_output.num_scheduled_tokens[req_id]
    if num_tokens &lt;= self.decode_threshold:
        decodes.append(i)
    else:
        prefills.append(i)
for i in range(1, min(num_decodes, num_prefills) + 1):
    if decodes[num_decodes - i] &gt;= num_decodes:
        input_batch.swap_states(prefills[first_prefill], decodes[num_decodes - i])
return modified_batch</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：控制流依赖每个 request 的 scheduled token 值；分支还会修改 batch 内请求顺序，影响后续 metadata 对齐。

### 10.4 SGLang NPU 的 `topk_indices is not None` 是大分支

SGLang 常规 Ascend backend 的 decode forward 首先检查 sparse top-k：

```python
if topk_indices is not None:
    return self.forward_sparse(...)
if graph_mode and not torch_compile:
    return self.forward_decode_graph(...)
...
```

**源码证据 NPU-KV-06｜top-k 存在性先于 graph 分支**

- 代码位置：[SGLang `ascend_backend.py` L2440-L2483](https://github.com/sgl-project/sglang/blob/1b9dfa14e66b617ed53270164549d59290b1f7c8/python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py#L2440-L2483)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>ascend_backend.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>2451
2456
2458
2459
2460
2469
2472
2473</code></pre></td>
<td valign="top"><pre><code class="language-python">topk_indices: Optional[torch.Tensor] = None,
if is_mla_preprocess_enabled() and self.use_mla:
    save_kv_cache = False
if topk_indices is not None:
    return self.forward_sparse(
        topk_indices,
if self.graph_mode and (not self.enable_torch_compile):
    return self.forward_decode_graph(...)</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：有 top-k 时直接进入 sparse path；只有 top-k 为 `None` 才继续考虑 graph decode，分支优先级也需要回放。

所以 sparse index 的存在性本身就是 branch guard，其值又决定 sparse 读取位置。

### 10.5 Block Sparse Attention 的输出 shape 可不变，内部工作却改变

`sgl-kernel-npu` 的 Block Sparse Attention wrapper 创建：

```python
attention_out = empty_with_format(query.sizes(), query.format)
```

然后把 `sparse_mask/sparse_count_table/actual_seq_lengths` 传给 `aclnnBlockSparseAttention`。

**源码证据 NPU-KV-07｜输出 shape 不变但 sparse workload 改变**

- 代码位置：[sgl-kernel-npu `block_sparse_attention.cpp` L24-L47](https://github.com/sgl-project/sgl-kernel-npu/blob/3479f4d99cd4e65a1cbe316f8bafc318014a4eb9/csrc/attentions/csrc/plugin/block_sparse_attention.cpp#L24-L47)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>block_sparse_attention.cpp</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>24
25
28
31
32
33
42
43
44
45
47</code></pre></td>
<td valign="top"><pre><code class="language-cpp">block_sparse_attention(query, key, value,
                       sparse_mask, sparse_count_table, ...)
                       actual_seq_lengths, actual_seq_lengths_kv)
TORCH_CHECK(input_layout != "TND", ...)
at::Tensor attention_out =
    empty_with_format(query.sizes(), query.options(), get_npu_format(query));
EXEC_NPU_CMD&lt;aclnnBlockSparseAttention&gt;(
    query, key, value, ..., actSeqLen, actSeqLenKv, ...,
    sparse_mask, sparse_count_table, ...,
    inputLayoutPtr, ..., sparse_size, causal, attention_out);
return attention_out;</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：输出按 `query.sizes()` 创建，但 sparse mask/count 和有效长度仍进入 NPU 算子；仅比较 input/output shape 会漏掉内部工作量。

也就是说，output shape 可以始终等于 query shape；真正的有效 workload 由 sparse mask、count table 和长度值决定。这是“只看 tensor shape 会漏掉性能差异”的直接底层证据。

---

## 11. 机器规模、显存和并行分解怎样进入 shape

### 11.1 先区分“容量可部署”与“执行 shape”

vLLM Ascend 官方文档给出：

- DeepSeek-V4 Pro W4A8：至少 2 个 A3 `128G×8` 节点或 4 个 A2 `64G×8` 节点；
- DeepSeek-V4 Flash W8A8：1 个 A3 `128G×8` 节点或 1 个 A2 `64G×8` 节点。

**配置证据 NPU-DEPLOY-01｜V4 Pro/Flash 的最低容量不同**

- 代码位置：
  - [DeepSeek-V4 Pro 文档 L22-L28](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/docs/source/tutorials/models/DeepSeek-V4-Pro.md#L22-L28)
  - [DeepSeek-V4 Flash 文档 L22-L28](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/docs/source/tutorials/models/DeepSeek-V4-Flash.md#L22-L28)
- 原文约束（中文转写）：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>DeepSeek-V4-Pro.md</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>26 ?



26 ?
</code></pre></td>
<td valign="top"><pre><code class="language-text">≈ DeepSeek-V4-Pro-w4a8-mtp:
2 × Atlas 800 A3 (128G × 8) or 4 × Atlas 800 A2 (64G × 8)

DeepSeek-V4-Flash.md
≈ DeepSeek-V4-Flash-w8a8-mtp:
1 × Atlas 800 A3 (128G × 8) or 1 × Atlas 800 A2 (64G × 8)</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：Pro/Flash 的量化方式和最低节点数不同；模型家族名不能代替真实 variant/weight manifest。

这些数字只能证明“最低容量/推荐部署”，不能直接推出单 rank shape。单 rank shape 还取决于实际：

```text
DP, TP, EP, PP, attention-TP, CP
P/D 是否分离
每个 parallel group 的成员
weight 是否复制
KV cache 是否按 rank 分片
最大并发/上下文配置
```

### 11.2 同样卡数可以有不同的 topology signature

DeepSeek-V4 Pro 的官方场景表同时给出：

- 32 个 A3：`DP2×TP16`；
- 64 个 A3 的 1P1D：可以是 `DP16×TP2` 或 `DP2×TP16`。

DeepSeek-V4 Flash 的示例则有 `DP4×TP4`、`DP16×TP1`：

**配置证据 NPU-DEPLOY-02｜同模型的并行分解也不唯一**

- 代码位置：
  - [DeepSeek-V4 Pro 文档 L1232-L1240](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/docs/source/tutorials/models/DeepSeek-V4-Pro.md#L1232-L1240)
  - [DeepSeek-V4 Flash 文档 L944-L953](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/docs/source/tutorials/models/DeepSeek-V4-Flash.md#L944-L953)
- 原文表格（中文转写）：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>DeepSeek-V4-Pro.md</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>1237 ?
1238 ?
1240 ?


950 ?
951 ?
952 ?</code></pre></td>
<td valign="top"><pre><code class="language-text">≈ 32 A3, Single-Node Mixed: dp2 tp16
≈ 64 A3, 1P1D: dp16 tp2 or dp2 tp16
≈ 64 A3, Long Context 1P1D: dp2 tp16 on P and D

DeepSeek-V4-Flash.md
≈ 16 A3, Single-Node Mixed: dp4 tp4
≈ 32 A3, 1P1D: dp16 tp1 on P and D
≈ 8 A3, Long Context: dp4 tp4</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：即使卡数固定，DP/TP/P-D role 仍可能不同；topology signature 必须保存真实 group。

所以 64 卡、128 卡要保存的是一个拓扑对象，而不是一个整数：

```yaml
topology:
  nodes: 8
  devices_per_node: 8
  total_devices: 64
  dp: 4
  tp: 8
  ep: 16
  pp: 1
  cp: 2
  attention_tp: 8
  rank_mapping: [...]
  prefill_decode_role: mixed
```

其中某些维度可能重叠或属于不同 group，不能假设乘积总等于总卡数；应保存框架创建出的真实 process group。

### 11.3 显存大小主要通过容量决策间接改变 shape

显存从 64G、96G 到 128G，通常会先改变：

- 权重能否单卡/单机容纳；
- 需要的 TP/PP/EP；
- KV cache block/token 容量；
- `max_model_len/max_num_seqs/max_num_batched_tokens`；
- graph bucket 能否预分配；
- 是否启用更激进量化；
- workspace 和通信 buffer 是否可用。

随后这些配置才改变单 rank tensor 和 workload shape。

本轮固定仓库的目标模型文档直接举证了 64G、128G；没有看到可安全归属于这些目标模型的 96G 官方组合。因此 96G 应作为实机 manifest 的一个真实值记录，而不是套用未经验证的 A2/A3 推断。

### 11.4 SoC 代际必须单独记录

建议最少记录：

```text
SoC family and exact chip name
device memory bytes
CANN version/build
Ascend driver/firmware
torch_npu commit/package
framework and NPU plugin commit
custom operator package/version
HCCL version
```

理由不是“环境信息越多越好”，而是本报告已经找到 A5/非 A5 cache dtype、slot mapping shape、SWA head size和算子支持范围的真实分支。

---

## 12. NPUGraph/ACL Graph：图 bucket、地址和可更新输入

### 12.1 vLLM Ascend 的图按 BatchDescriptor 建立

vLLM Ascend 的 ACL Graph wrapper 按 `BatchDescriptor` 找 graph entry；不同 descriptor 会创建不同 capture。代码还明确说明 dynamic inputs 由外层稳定 buffer 管理，debug 模式检查输入地址。

**源码证据 NPU-GRAPH-01｜descriptor、地址和 replay**

- 代码位置：
  - [输入假设 L60-L82](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/compilation/acl_graph.py#L60-L82)
  - [descriptor capture L133-L167](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/compilation/acl_graph.py#L133-L167)
  - [地址检查/replay L243-L267](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/compilation/acl_graph.py#L243-L267)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>acl_graph.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>68
73
76
81

135
138
145
147
149
153
163
165

243
245
246
266</code></pre></td>
<td valign="top"><pre><code class="language-python">wrapper receives runtime_mode and batch_descriptor(key)
capture if key does not exist; replay if key exists
wrapper does not store persistent buffers or copy runtime inputs
input addresses are checked in DEBUG

batch_descriptor = forward_context.batch_descriptor
if runtime_mode == NONE or runtime_mode != self.runtime_mode:
    return self.runnable(*args, **kwargs)
if batch_descriptor not in self.concrete_aclgraph_entries:
    entries[batch_descriptor] = ACLGraphEntry(...)
if entry.aclgraph is None:
    input_addresses = [x.data_ptr() for x in args if isinstance(x, Tensor)]
    aclgraph = torch.npu.NPUGraph()

if self.is_debugging_mode:
    new_input_addresses = [x.data_ptr() for x in args if isinstance(x, Tensor)]
    assert new_input_addresses == entry.input_addresses
entry.aclgraph.replay()</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：graph cache key 是 descriptor；运行时 mode 可绕过 graph；replay 依赖外部稳定 buffer，debug 下要求输入地址一致。

所以 graph signature 至少包括：

```text
batch descriptor/bucket
capture mode
buffer addresses or stable-buffer IDs
input logical/storage shapes
workspace
graph memory pool
dynamic input update policy
```

### 12.2 FIA TND 需要把 dummy request 计入执行 shape

vLLM Ascend model runner 为满足 FIA TND 对累计 Q 长度的检查，会区分 uniform/mixed batch，并插入 dummy request/padding：

**源码证据 NPU-GRAPH-02｜FIA TND 为 graph 插入 dummy request**

- 代码位置：[vLLM Ascend `model_runner_v1.py` L767-L814](https://github.com/vllm-project/vllm-ascend/blob/e462c42a4599bb17bae49775074eb6a9b094f528/vllm_ascend/worker/model_runner_v1.py#L767-L814)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>model_runner_v1.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>777
782
784
786
791
798
799
802
803
807
808
809
810</code></pre></td>
<td valign="top"><pre><code class="language-python"># TND: hidden_states dim-0 must equal last actual_seq_lengths_q
if runtime_mode == FULL and configured_mode == FULL:
    num_reqs_padded = num_reqs
else: num_reqs_padded = batch_desc_num_reqs or num_reqs
if num_tokens_padded == num_reqs_padded * uniform_decode_query_len:
    last_loc = query_start_loc.np[num_reqs]
    query_start_loc.np[num_reqs + 1:num_reqs_padded + 1] = ...
else:
    # Mixed-batch case
    if query_start_loc.np[num_reqs_padded] &lt; num_tokens_padded:
        # Insert a dummy request
        query_start_loc.np[num_reqs_padded + 1] = num_tokens_padded
        num_reqs_padded += 1</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：graph 执行 request 数可能多于 raw request 数；dummy request 是满足 FIA ABI 的执行实体，必须记录。

因此要区分：

```text
raw request count
captured batch size
raw token count
graph token count
dummy request lengths
有效输出 slice
```

### 12.3 SGLang NPU 在 replay 前更新 seq_lens

SGLang 的 NPUGraphRunner：

- 用 `torch.compile(... fullgraph=True, dynamic=False)`；
- 创建 `torch.npu.NPUGraph()`；
- graph capture 设置 `auto_dispatch_capture=True`；
- replay 前更新 `seq_lens`；
- Target Verify 会给长度加 captured request width；
- batch 不足时用 0 补到 capture batch；
- DeepSeek DSA/V4 走独立 replay 路径。

**源码证据 NPU-GRAPH-03｜SGLang capture/update/replay**

- 代码位置：
  - [capture L68-L176](https://github.com/sgl-project/sglang/blob/1b9dfa14e66b617ed53270164549d59290b1f7c8/python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py#L68-L176)
  - [replay 更新 L209-L256](https://github.com/sgl-project/sglang/blob/1b9dfa14e66b617ed53270164549d59290b1f7c8/python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py#L209-L256)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>npu_graph_runner.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>75
77
139
140
148
154
170
174
175

236
238
242
243
244
249
255
256</code></pre></td>
<td valign="top"><pre><code class="language-python">if enable_compile:
    yield torch.compile(..., fullgraph=True, dynamic=False, backend="npugraph_ex")
def _create_device_graph(self):
    return torch.npu.NPUGraph()
with torch.npu.graph(
    auto_dispatch_capture=True,
def _update_inputs(self, seq_lens):
    self.graphs[self.bs].update(
        cpu_update_input=[{self.update_attr_name: seq_lens}])

graph_key = self._make_graph_key(self.bs)
if not (is_deepseek_dsa(config) or is_deepseek_v4(config)):
    if forward_mode.is_target_verify():
        seq_lens_cpu = seq_lens.cpu() + captured_req_width
        seq_lens = seq_lens_cpu.tolist() + [0] * (bs - raw_bs)
    output = backend.replay_with_input_update(graph_key, seq_lens=seq_lens, ...)
else:
    output = backend.replay(graph_key, forward_batch)</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：capture 是 static/fullgraph；普通模型 replay 前更新并补齐 `seq_lens`，DeepSeek DSA/V4 则走另一条 replay API。

这说明 graph replay 的动态输入不能用一个 `batch_size` 概括。

### 12.4 torch_npu 底层证明确有 graph 更新模式和内存池约束

Ascend PyTorch 的 `NPUGraph`：

- capture 时 `report_shape=True`；
- `update()` 只在 `auto_dispatch_capture=True` 时允许；
- graph 可共享 memory pool；
- capture 对 stream、allocation 和 pinned-memory 配置有约束。

**源码证据 NPU-GRAPH-04｜torch_npu update 与底层 capture guard**

- 代码位置：
  - [`graphs.py` L607-L663](https://github.com/Ascend/pytorch/blob/10fe1622631286665f15d84951d53df7840e6ead/torch_npu/npu/graphs.py#L607-L663)
  - [`graphs.py` L835-L876](https://github.com/Ascend/pytorch/blob/10fe1622631286665f15d84951d53df7840e6ead/torch_npu/npu/graphs.py#L835-L876)
  - [`NPUGraph.cpp` L165-L225](https://github.com/Ascend/pytorch/blob/10fe1622631286665f15d84951d53df7840e6ead/torch_npu/csrc/core/npu/NPUGraph.cpp#L165-L225)
- 关键片段：

<table class="source-evidence">
<thead>
<tr><th align="right">行号</th><th align="left"><code>graphs.py</code></th></tr>
</thead>
<tbody>
<tr>
<td align="right" valign="top"><pre><code>624
637
640
655
657
658
663
849
850
858


172
178
190
200
214
221</code></pre></td>
<td valign="top"><pre><code class="language-python">super().capture_begin(..., report_shape=True)
def replay(self):
    super().replay()
def update(self, cpu_update_input):
    if not self.auto_dispatch_capture:
        raise RuntimeError(...)
    self.graph_dispatch_mode.update_capture_record(cpu_update_input)
self.pool = () if pool is None else (pool,)
self.capture_stream = stream if stream is not None else default_capture_stream
self.npu_graph.auto_dispatch_capture = auto_dispatch_capture

NPUGraph.cpp
TORCH_CHECK(_task_queue_enable != 2, ...)
TORCH_CHECK(!pin_memory_expandable_segments(), ...)
TORCH_CHECK(!has_graph_exec_, ...)
TORCH_CHECK(stream != getDefaultNPUStream(), ...)
if (pool.first != 0 || pool.second != 0):
    mempool_id_ = pool</code></pre></td>
</tr>
</tbody>
</table>

- 证明内容：graph update 只对 auto-dispatch capture 合法；capture 还受 task queue、pinned allocator、stream 和 memory pool 约束。

因此跨环境回放时，graph capture artifact 不应被当作纯模型资产；它属于“环境 + bucket + buffer plan”的联合产物。

---

## 13. `if/else` 和隐式控制流应该怎样录

### 13.1 七类高风险分支

| 分支类型 | 本报告中的例子 | 应记录 |
|---|---|---|
| forward mode | Prefill/Decode/Verify/Idle | mode、query/KV lengths、branch ID |
| 稀疏模式 | dense/C4/C128/SFA/SWA | compress ratio、top-k 存在性和值 |
| 设备代际 | A5/非 A5、A2/A3 | SoC、dtype/layout/head padding |
| dtype/量化 | BF16/FP16/INT8/W4A8/C8 | weight/scale shape、format、kernel |
| 图模式 | eager/capture/replay/compile | graph key、buffer、update policy |
| 分布式 | world size=1 bypass、AllGather/AllToAll/MC2 | group、counts、algorithm |
| 阈值/合法性 | token 数、head dim、matmul 限制 | predicate 输入、阈值、结果 |

### 13.2 推荐的 branch event

```yaml
branch_event:
  op_scope: "layer.17.attn"
  source:
    repo: "sglang"
    commit: "1b9dfa1..."
    file: "python/sglang/.../ascend_dsv4_backend.py"
    line: 1648
  predicate: "compress_ratio in (0, 4, 128)"
  dependency_values:
    compress_ratio: 4
    forward_mode: "decode"
    graph_mode: true
  selected_branch: "compressed_c4"
  downstream_operator: "custom::npu_sparse_attn_sharedkv"
```

只记录 `selected_branch` 不够；如果不记录 predicate 的依赖值，换模型参数后无法判断该分支是否仍合法。

### 13.3 隐式控制流也要算

以下行为未必写成 Python `if/else`，但同样影响路径：

- NPU custom op 的 tiling key；
- workspace 查询结果；
- dynamic shape/format support；
- sparse mask/count table；
- HCCL count；
- graph capture 是否命中已有 descriptor；
- kernel 中对无效 page、`-1` expert、越界 token 的 mask；
- operator support check 后的 fallback。

所以控制流扫描要包括 host C++、算子 metadata/tiling 和运行时 trace。

---

## 14. 第一版泛化方法：三类胶囊 + 约束化重建

### 14.1 胶囊 A：Semantic Step Capsule

保存“这一轮推理在语义上做什么”：

```text
request/sequence IDs
input_ids / positions
prefill/decode/verify/idle
seq/query/prefix lengths
finished mask / accepted tokens
layer range
cache 前态版本
```

它用于重建值相关决策，不要求直接保存所有中间 activation。

### 14.2 胶囊 B：Operator ABI Capsule

对每个高风险大算子保存：

```text
operator name and implementation
all tensor logical/storage shapes
dtype/layout/stride
all index and metadata tensors
attrs such as layout/sparse_count/block_size
workspace bytes
tiling identity
branch/guard
input/output aliasing
```

优先覆盖：

- Lightning Indexer；
- sparse/shared-KV attention；
- FIA/SFA/DSA；
- compressor/cache write；
- MoE top-k/permute/GMM/combine；
- All-to-All/All-Gather/MC2；
- NPUGraph capture/replay。

### 14.3 胶囊 C：Physical Execution Capsule

保存环境与物理计划：

```text
SoC and memory
CANN/driver/firmware/torch_npu
framework/plugin/custom-op commits
parallel groups and rank map
weight shard manifest
NPU storage format
graph bucket and memory pool
HCCL algorithm/buffer/network topology
```

### 14.4 把 shape 记录成“值 + 来源表达式 + guard”

不要只保存：

```yaml
shape: [4096, 7168]
```

建议保存：

```yaml
extent:
  value: 4096
  symbol: T_exec
  expression: "round_up(T_raw, attn_tp_size)"
  dependencies:
    T_raw: 4013
    attn_tp_size: 8
  guards:
    - "enable_dsa_cp == true"
    - "forward_mode == prefill"
```

这样换模型尺寸、并行度、batch 时，系统可以重新求值并判断录制样本是否适用。

### 14.5 index 泛化分三个等级

| 等级 | 方法 | 适用目标 | 风险 |
|---|---|---|---|
| I：Exact | 保存完整 top-k/block/slot/router index 和 cache 前态 | 结果与路径复现 | 数据量大，跨参数不可直接迁移 |
| II：Recompute | 保存 input/state，在目标配置上重算 index | 新模型参数的正确路径 | 需要能执行 indexer/router 前置计算 |
| III：Constrained Synthetic | 生成满足长度、范围、直方图、局部性、重复率的 index | 独立性能微基准 | 不保证模型数值等价 |

不能把等级 III 的结果包装成模型级 Exact Replay。

### 14.6 MoE 泛化以“分布向量”而非平均 token 数为核心

建议把 MoE workload 表示为：

```text
T, K
expert_count_vector
rank_send_count_vector
rank_recv_count_vector
capacity/padding
local_GEMM_M_per_expert
shared_expert tokens
permutation locality
```

在不能保存 token/activation 时，至少按这些向量生成 synthetic routing；只保持 `T×K` 不足以保持通信和专家 GEMM 性能。

### 14.7 层间状态用版本化引用，不要复制成孤立 layer trace

例如：

```yaml
layer_state:
  kv_cache_version_in: 8321
  kv_cache_version_out: 8322
  sparse_index_source:
    kind: "shared_from_layer"
    layer_id: 15
    buffer_version: 410
  residual_version_in: 2001
  pipeline_stage: 1
```

这样可以区分：

- 本层新算 top-k；
- 跨层复用 top-k；
- 从 cache 读取；
- graph 中原地更新。

---

## 15. 建议的回放键和记录 schema

### 15.1 `static_key`

```text
model architecture + model revision
weight tensor manifest/hash
quantization scheme and packing
SoC + CANN + torch_npu
framework/plugin/operator commit
DP/TP/EP/PP/CP groups
weight/cache physical format
graph/eager mode
```

### 15.2 `step_key`

```text
forward mode
raw/execution batch and token bucket
seq/query/prefix length vectors
speculative width/accepted count
layer ID/type/compress ratio
index presence + digest/distribution
cache version + block/page/slot digest
MoE expert/rank count vectors
branch IDs
operator tiling/workspace IDs
```

### 15.3 最小建议 schema

```yaml
trace_version: "0.3-npu"

environment:
  soc: "Ascend..."
  device_memory_bytes: 0
  cann: "..."
  torch_npu: "10fe162..."
  framework: "vllm-ascend"
  framework_commit: "e462c42..."
  custom_ops:
    sgl_kernel_npu: null

topology:
  nodes: 0
  devices_per_node: 0
  rank: 0
  groups:
    tp: {size: 0, ranks: []}
    ep: {size: 0, ranks: []}
    dp: {size: 0, ranks: []}
    pp: {size: 0, ranks: []}
    cp: {size: 0, ranks: []}

model:
  architecture: "..."
  revision: "..."
  quantization: "..."
  weight_manifest_ref: "..."

step:
  mode: "prefill|decode|target_verify|draft_extend|idle"
  input_ids_ref: "..."
  positions_ref: "..."
  seq_lens: []
  query_lens: []
  prefix_lens: []
  raw_tokens: 0
  exec_tokens: 0
  graph_bucket: null

tensor:
  logical_shape: []
  storage_shape: []
  stride: []
  dtype: "..."
  acl_format: "ND|FRACTAL_NZ|..."
  valid_extents: []

dynamic_state:
  block_table_ref: "..."
  slot_mapping_ref: "..."
  kv_cache_version: 0
  sparse_topk_ref: "..."
  router_topk_ref: "..."
  expert_counts: []
  send_counts: []
  recv_counts: []

operator:
  name: "..."
  backend: "..."
  attrs: {}
  guard_results: {}
  workspace_bytes: 0
  tiling_id: "..."
```

### 15.4 哪些值可只存 digest

| 字段 | Exact Replay | Performance Replay |
|---|---|---|
| input IDs/positions | 全值或可重建引用 | 可脱敏生成，但需保持长度/重复/stop 约束 |
| sparse top-k | 全值 | digest + 直方图 + 局部性；微基准时生成 |
| MoE top-k | 全值 | expert/rank count 向量 + 合法 synthetic index |
| block/slot table | 全值与 cache snapshot | 合法映射 + page locality 统计 |
| weight | 固定 revision/hash | 同 shape/dtype/format 的 synthetic weight 可用于纯性能 |
| activation | 必要节点全值/seed | dtype、范围、稀疏/重复统计 |

“只存 digest”只能用于确认两个 trace 是否相同，不能靠 digest 恢复 index。

---

## 16. 针对四类模型的第一版落地建议

### 16.1 DeepSeek-V4 Pro/Flash

P0 必录：

```text
variant and weight revision
A2/A3/A5 + quantization
compress_ratios per layer
C4 top-k / C128 metadata
SWA/compressed cache specs and page tables
index/cache dtype and scale shape
MTP/speculative step
MoE router/expert/rank counts
FlashComm/MC2/DeepEP selection
graph bucket and tiling IDs
```

Pro 和 Flash 即便复用同一个模型类，也不能只用名称区分；必须以实际 config/weight manifest 为准。

### 16.2 GLM-5.2

P0 必录：

```text
BF16/W8A8/W4A8C8
SFA/DSA CP/C8 flags
shared indexer layer map
本层 compute/reuse top-k
DP/TP/EP and HCCL/FlashComm/MC2
block size 128 and padded local lengths
graph/speculative configuration
```

### 16.3 MiniMax-M3

P0 必录：

```text
local Q/KV heads and KV replication
fused QKV+indexer guard
index cache + slot mapping
prefill/decode top-k
sparse/dense attention layer map
MoE/dense MLP layer map
weight shard/quant/format
```

SGLang NPU 路径在本版应先标为“需要运行确认”，不要把 vLLM Ascend 的专用实现自动投射过去。

### 16.4 Qwen3.7

固定源码证据不足时分两层处理：

1. **服务级 recorder**
   - 请求 token/position；
   - batch/seq/prefix；
   - TTFT/TPOT；
   - 可见的并行、量化、显存、graph 配置；
   - NPU profiler 中实际算子 shape/layout。
2. **内部 shape**
   - 只有获得真实 config、权重 manifest 或服务端 trace 后再填写；
   - 不用 Qwen3.5/3.6/普通 Qwen3 参数冒充 3.7；
   - surrogate 只能标注为框架/硬件性能近似，不能标注为模型路径等价。

---

## 17. 验证计划

### P0：先实现观测，不改模型

在 vLLM Ascend 和 SGLang NPU 增加统一 hook，记录：

1. 每个高风险算子输入/输出的 logical shape、storage shape、stride、dtype、format；
2. branch event；
3. seq/block/slot/top-k/expert count；
4. graph bucket、workspace、tiling identity；
5. HCCL send/recv count；
6. layer/cache version。

先选：

- DeepSeek-V4 Flash W8A8；
- GLM-5.2 W4A8C8；
- MiniMax-M3 一个公开可部署权重。

### P1：验证五条关键因果链

#### 实验 A：logical shape 相同，NZ storage 不同

控制逻辑 `[A,B]`，切换 dtype/format，验证：

```text
logical shape 不变
storage size/C0/padding 改变
kernel 或性能改变
```

#### 实验 B：C4 top-k 值不同

固定 `[T,H]`、`index_topk` 和 cache shape，只改变合法 top-k 位置，验证：

```text
output 数值改变
访存局部性/latency 改变
```

#### 实验 C：MoE expert count 分布不同

固定 `[T,H]` 和 K，构造：

- 均匀路由；
- 单 expert 热点；
- 单 rank 热点；
- 跨节点热点。

验证 All-to-All count、local GEMM M、HCCL latency。

#### 实验 D：同总卡数，不同 DP/TP/EP

固定总 NPU 数，切换至少两种并行分解，比较：

```text
local weight shape
local heads/experts
KV capacity
communication payload
graph bucket
```

#### 实验 E：跨层 index 复用

对 GLM-5.2/DeepSeek-V4 记录本层算 index 与复用前层 index，验证单层孤立回放为何不等价。

### P2：覆盖设备与规模

最小矩阵建议：

| 维度 | 取值 |
|---|---|
| SoC | 实际拥有的 A2/A3/A5 |
| 显存 | 64G、96G、128G（按真实卡记录） |
| 规模 | 1 机、8 机/64 卡、16 机/128 卡 |
| 量化 | BF16、W8A8、W4A8/W4A8C8 |
| 模式 | Prefill、Decode、Verify、Idle |
| 图 | eager、NPUGraph/ACL Graph |
| sequence | 短/中/长、长尾混合 |
| MoE | 均匀、热点、跨节点热点 |
| cache | cold、prefix hit、碎片化 page |

不要求一次跑完笛卡尔积。先用源码 guard 和 shape graph 做 pairwise/边界组合，再补生产分布热点。

### P3：定义可比性判定

回放前输出：

```text
EXACT
  全部 static/step/state/operator signature 匹配

PERF-COMPARABLE
  关键 workload、format、tiling、communication signature 匹配，
  token/weight 可为 synthetic

INVALID
  设备/算子 ABI、guard、index 合法性、cache/layout 或 topology 不匹配
```

不要在不匹配时静默 pad、截断或换 backend；应显式报告差异。

---

## 18. 本版仍未解决的问题

1. CANN 内置的 `npu_sparse_attn_sharedkv`、FIA/SFA 等闭包内部所有 tiling 规则没有完全展开；当前可见的是框架调用、metadata 和部分开源 AscendC 算子。
2. `torch_npu/CANN` 的动态 shape、图更新与 custom op 支持矩阵会随版本变化，必须用实际部署版本重新固定。
3. A5 的 DeepSeek-V4 路径在当前源码中已出现，但本版未做 A5 实机验证。
4. SGLang 对 GLM-5.2、MiniMax-M3 的具体 NPU feature 组合，需要以启动日志和 profiler 确认最终选择的 backend。
5. Qwen3.7 没有足够开放证据，不能给出可信的内部权重和 layer shape。
6. prefix cache、scheduler、PD 分离会改变跨 step 状态；本文 schema 已预留，但尚未形成完整状态机。
7. HCCL 拓扑、网卡、链路和拥塞会使相同 count 产生不同性能；count signature 是必要条件，不是充分条件。
8. 静态扫描无法看到 fused op 内部所有 shape 和分支，后续必须结合 CANN profiler、GE/图 dump 和算子 tiling dump。

---

## 19. 最终建议

第一版可以先落地为三个模块：

1. **NPU Shape/Format Recorder**
   - 记录 logical/storage shape、stride、dtype、ACL format、有效 extent。
2. **Dynamic Decision Recorder**
   - 记录 branch、seq/block/slot/top-k、expert/rank counts、layer/cache version。
3. **Operator/Topology Recorder**
   - 记录 NPU op、attrs、tiling/workspace、graph bucket、HCCL count 和完整 parallel groups。

优先级建议：

```text
P0  index/cache/MoE counts/physical format/branch
P1  graph/tiling/workspace/HCCL topology
P2  token 统计特征和参数化 synthetic generator
P3  自动 shape constraint solver 与跨配置可比性判定
```

最后把本文核心判断压缩成一句话：

> **昇腾录制回放的泛化单位不能是“同模型、同 tensor.shape”，而应是“同语义约束、同本地分片、同物理格式、同动态索引/通信分布、同算子 plan 与兼容环境”的执行胶囊。**
