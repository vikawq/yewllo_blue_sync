# CANNON × SGLang × Qwen3-8B —— 跑通记录、trace/profiling 与 gap 分析

> 机器：`192.168.41.177`（Ubuntu 24.04.5 / aarch64 鲲鹏 taishan200 / 128 核 / 502 GB / **无昇腾卡**）
> 日期：2026-09-16
> 容器：`cannon-sglang:0.1`（3.67 GB），宿主机未安装任何东西
> CANNON：`/root/xuy/workspace/CANNON`（rustc 1.98.1 编出的 `libascendcl.so`）
> SGLang：`/root/xuy/workspace/JamesBond` dev @ `7accfae6e`（2026-09-09）
> 模型：`/root/xuy/models/qwen3_8b`（Qwen3ForCausalLM，36 层，bf16 权重，16 GB）
> 环境怎么搭的：见 [`DEPLOY.md`](DEPLOY.md)
> CANNON 侧改动：分支 `feat/ops-for-qwen3-sglang`（832 行，27 个文件 + 2 个新文件）
>
> ⚠ **本文有两轮走查，刻意都留着。** 第一轮（§2.2–§2.8）是**补算子之前**量的，四堵墙
> 靠实验台侧的垫片绕过去；第二轮（§2.9）是**四堵墙补进 CANNON 本体、垫片全部撤掉**之后量的。
> 第一轮的数字**没有被改写**——它们是一组自洽的测量，retcon 掉就等于把「当时量到什么」和
> 「现在是什么」搅成一团。要读「现在 CANNON 能做到什么」，看 §2.9 和第 3 章每条的状态行。

---

## 0. 结论先行

1. **环境搭起来了，而且是真的在跑**：SGLang 认出「Ascend910B1 / 59.6 GiB HBM」，
   真 Qwen3-8B 权重加载进模拟 HBM，36 层的 prefill/decode 逐算子落到 CANNON 引擎，
   trace 里有 `aclnnMm` / `aclnnFusedInferAttentionScoreV4` / `aclnnRmsNorm` / `aclnnSwiGlu` /
   `aclnnScatterNdUpdate` 的真实 span。
2. **第一轮「开箱即跑」不成立**：从 `sglang.Engine(...)` 到第一个 token 一共撞了 **6 堵墙**，
   其中 **4 堵是 CANNON 的缺口**，1 堵是环境/依赖，1 堵是仿真速度。
   每一堵都在第 3 章逐条记了 `file:line` 和判据。
3. **那 4 堵已经补进 CANNON 本体，走查现在【不带任何实验台垫片】就能跑到底。**
   `aten::unfold` 进了 C++ 半边、FIA 的属性拼法与 `split_qkv_rmsnorm_rope` 进了 `_impl.REGISTRY`、
   `aclnnCat` 的 strided 行进了策略表；外加一颗**全新的引擎算子 `aclnnMul`**（五处 + 门禁七类
   全部走完）。撤掉 `shims/` 之后连跑三趟，`driver 退出码 = 0`，`kind=call` 为 0。见 §2.9。
4. **最值得修的两条，都不是「缺算子」，是「缺覆盖面」**：
   一个视图算子没注册（`unfold`，而 `as_strided` 已经在），
   一个算子只注册了两种拼法里的一种（`torch.ops.npu.*` 有、`torch_npu.*` 没有）。
   这两条各自只差几行，却分别挡死了整条 RoPE 和整条 attention。
5. **补算子这一轮自己产出了一条比算子更值钱的结论：实现 ≠ 被用上。**
   `aclnnMul` 注册好、编出来、519 条一致性用例全绿、符号面从 113 涨到 115 之后，
   走查的 profiling 表回来是**逐字节相同的**，`aten::mul.out` 照旧 1280 次 fallback ——
   没有任何东西把它路由过去。CANNON 自己 `PROVENANCE.md` §1 给 `aclnnSub` 写过同一句：
   **「导出符号不等于路由到它」**。补上 `_deep._FORWARDED_ATEN` 的那一行之后才真的用上。
6. **CANNON 的错误质量非常高**。六堵墙里没有一堵是「静默算错」——
   全部是点名道姓的拒绝（`unimplemented: <符号名>`、`ForwardError: aclnnCat: ...`、
   `AtbError: atb_layout.json is missing`）。这正是它自己承诺的「能证伪，不能证真」。

---

## 1. 容器环境（要求 ①）→ 见 [`DEPLOY.md`](DEPLOY.md)

容器怎么搭的、与 vAscend 那份镜像的三处刻意不同、「不污染宿主」的边界在哪、
构建期踩到的六个坑，全部挪到了 **[`DEPLOY.md`](DEPLOY.md)**。
本文从第 2 章起只讲**测出了什么**。

一句话交代前提：镜像 `cannon-sglang:0.1`（3.67 GB），Rust 工具链与 Python venv 都在镜像
`/opt` 下，源码与模型运行期挂载 `/root/xuy`，**宿主的 apt / pip / /usr/local 一个字节没动**；
Dockerfile 改完后用全新 tag 从头重建验证过，无任何手工 pip 步骤即可 `import sglang`。

---

## 2. 推理与 trace/profiling（要求 ②）

### 2.1 跑法

```bash
docker exec cannon-lab bash -lc '
  ASCEND_USE_FIA=1 MAX_NEW_TOKENS=2 MAX_TOTAL_TOKENS=2048 MEM_FRACTION=0.5 \
  PROMPT="你好" bash /root/xuy/workspace/cannon-lab/run/run_qwen3.sh main ascend910b'
```

四个不是「调参」而是**必须**的开关，理由都在脚本注释里：

* `dtype="float16"` —— 模型权重是 **bf16**，而 CANNON v1 的算子表里 bf16 是**按名拒收**的
  （`aclnnMm`/`aclnnRmsNorm`/`aclnnSwiGlu`/`aclnnFusedInferAttentionScoreV4` 全部只收 fp32/fp16）。
  不改这一条，第一颗算子就 `200006`。
* `MAX_TOTAL_TOKENS` 封顶 —— deep 模式下 KV 池在**宿主内存**里，而 SGLang 按
  `torch.npu.mem_get_info()` 的**人设显存**（59.6 GB）去要。不封顶机器会开始换页，
  走查还没摸到算子就先 OOM，报出一份**比真相短**的缺口清单。
* `ASCEND_USE_FIA=1` —— 见 §3.5：ATB 顶替面在本机建不出来。
* `watchdog_timeout` 放大 —— 见 §3.6：SGLang 默认 300s 看门狗会把「太慢」报成一次崩溃。

⚠ **`shims/` 现在默认不上 `PYTHONPATH`**（`USE_LAB_SHIMS=1` 才放回来，用于 A/B 对照）。
第一轮那四堵墙是靠它绕过去的；补进 CANNON 之后它就只剩对照价值了。

### 2.2 跑完了：真 Qwen3-8B 端到端出 token

```
[drive] Engine 起好，用时 79.2s          （其中 load_weight 56.7s，16 GB 权重进模拟 HBM）
[drive] generate 用时 2053.2s
[run]   driver 退出码 = 0
```

`drive-result.json`：

```json
{"prompt": "你好", "max_new_tokens": 2,
 "output": [{"text": "danny", "output_ids": [67, 12888],
             "meta_info": {"prompt_tokens": 1, "completion_tokens": 2,
                           "finish_reason": {"type": "length", "length": 2},
                           "e2e_latency": 2053.196}}]}
```

⚠⚠ **token 出来了，但 token 的【内容】不可信，这一条必须说在前面。**
「你好」greedy 解出 `danny` 显然不是这个模型该说的话。三个已知嫌疑，按可疑度排序：

1. **bf16 → fp16 的强制转换**（§2.1 第一条）。Qwen3-8B 是 bf16 训的，fp16 的指数位少 3 bit，
   8B 模型的激活值域很容易溢出。这是 CANNON 自己「不可信区」里就点名的那一类
   （v1 用 fp16 承载，数值不是昇腾会给的）。
2. **我们自己的 `split_qkv_rmsnorm_rope` 垫片**（§3.2）。它不在 CANNON 的门禁之下，
   逐头 RMSNorm 与 neox RoPE 的推导写在文件注释里，但**没有对着厂商实现做过差分验证**。
3. `attenMask` 这条 —— CANNON 「比厂商严」的唯一一处，只收逐格满足 `masked ⇔ c > r` 的因果平面。
   这一趟没有被拒（否则会报错），但不构成「mask 语义与厂商一致」的证据。

⚠ 这一趟是**第一轮**，带着实验台垫片。补完算子后又连跑三趟（§2.9），`text` 一直是 `danny`，
`driver 退出码` 一直是 0 —— 也就是说**补算子没有改变数值结论**，下面这三条嫌疑一条都没被排除。

⟹ **这次走查证明的是「接口面通了、账记下来了」，不是「算对了」。**
正符合 CANNON 自己的那句话：**能证伪，不能证真**。要给数值背书，需要一趟
bf16 原生 + 无自备垫片 + 与真机逐 token 对照的走查 —— 三个前提今天一个都不成立。

### 2.3 profiling（第一轮，`scripts/trace_stats.py`，主跑 pid 5768）

⚠ **这张表是【补算子之前】的，带着实验台垫片。**保留原样不做改写 —— 它和 §2.4 /
§2.7 / §2.8 是一组自洽的读数。补完之后的表在 §2.9。

`chip Ascend910B1 / cost model roofline-v1 / dropped events 0`

| operator | calls | dur (虚拟 µs) | flops | bytes | bound |
|---|---:|---:|---:|---:|---|
| `MemcpyD2D` | 8813 | 56862.52 | 0 | 16,381,961,844 | memory-bound ×8813 |
| **`aclnnMm[fp16]`** | **435** | **37661.92** | **45,408,583,680** | **45,423,233,280** | memory-bound ×435 |
| `MemcpyH2D` | 1301 | 19350.20 | 0 | 16,441,992,077 | memory-bound ×1301 |
| `MemcpyD2H` | 1861 | 9392.85 | 0 | 112,560,586 | memory-bound ×1861 |
| `aclnnEmbedding[fp16]` | 3 | 2932.19 | 0 | 3,734,003,736 | memory-bound ×3 |
| `aclnnScatterNdUpdate[fp16]` | 216 | 2584.44 | 0 | 1,925,629,632 | memory-bound ×216 |
| **`aclnnFusedInferAttentionScoreV4[fp16]`** | **108** | **1411.38** | 3,538,944 | 1,115,357,616 | memory-bound ×108 |
| `aclnnCat[fp32,fp16,int64,int32]` | 223 | 1166.19 | 0 | 65,519,904 | memory-bound ×223 |
| `aclnnAdd[int64,fp16]` | 220 | 1102.59 | 1,105,928 | 3,317,848 | memory-bound ×220 |
| `aclnnRmsNorm[fp16]` | 219 | 1096.79 | 2,278,284 | 2,358,156 | memory-bound ×219 |
| `aclnnAddRmsNorm[fp16]` | 216 | 1086.91 | 4,424,544 | 8,848,224 | memory-bound ×216 |
| `aclnnSwiGlu[fp16]` | 108 | 546.26 | 7,962,624 | 7,962,624 | memory-bound ×108 |
| `Memset` | 12 | 311.24 | 0 | 321,585,224 | memory-bound ×12 |
| `aclnnAdds/Sub/Subs`、`aclnnIndex`、`aclnnIndexPutImpl` | 59 | 295.53 | 56 | 672,960 | memory-bound |
| `aclnnArgMax[fp32]` | 3 | 16.43 | 0 | 1,823,256 | memory-bound ×3 |
| **total** | **13797** | **135817.42** | **45,427,894,060** | **85,546,826,967** | |

**逐层对账（这是判断 trace 是否可信的第一步）**：这一趟是 **3 次前向**
（SGLang 自己的 warmup 一次 + 我们的 prefill 一次 + 1 步 decode），模型 36 层：

| 算子 | 实测 | 期望 | |
|---|---:|---|---|
| `aclnnFusedInferAttentionScoreV4` | 108 | 3 × 36 | ✅ |
| `aclnnSwiGlu` | 108 | 3 × 36 | ✅ |
| `aclnnAddRmsNorm` | 216 | 3 × 36 × 2（input / post-attention） | ✅ |
| `aclnnRmsNorm` | 219 | 3 × 36 × 2（q_norm / k_norm）+ 3（最后的 norm） | ✅ |
| `aclnnScatterNdUpdate` | 216 | 3 × 36 × 2（K 写 + V 写） | ✅ |
| `aclnnMm` | 435 | 3 × 36 × 4（qkv/o/gate_up/down）+ 3（lm_head） | ✅ |
| `aclnnArgMax` | 3 | 3 次采样 | ✅ |

**每一格都对得上 —— 这不是噪声，是 Qwen3-8B 的结构本身。**

### 2.4 资源占用：这颗 910B 上 cube 闲着 97.7%

`cannond trace summary` 的逐轨表（`BUSY` = 该资源自己被占多久，`SPAN` = 这条轨的跨度）：

| 轨 | SPANS | BUSY (µs) | SPAN (µs) | 占用率 |
|---|---:|---:|---:|---:|
| `resource: cube` | 543 | **2274.68** | 101090.59 | **2.25 %** |
| `resource: vector` | 809 | **1.13** | 103620.05 | **≈ 0 %** |
| `resource: hbm` | 13797 | **66841.94** | 133442.46 | **50.1 %** |

设备级聚合：`SUM 135817.423 / UNION 133447.455 / WALL 133447.455 / SUM·WALL 1.018 / IDLE 0.000`。

这四个数字合起来就是**这次 profiling 最硬的一条结论**：

* **彻底 memory-bound，连 GEMM 都是。** `aclnnMm` 的算术强度
  `45.41 GFLOP / 45.42 GB ≈ 1.0 FLOP/byte`，而 910B 的 ridge point 在 **~235**
  （376 TFLOPS ÷ 1.6 TB/s）—— 差两个数量级。batch=1 的 decode 本来就该长这样，
  **模型是对的**。这正是 USAGE §3 里说「能回答」的那一格。
* **cube 2.25%、vector ≈ 0%**：算力完全没用上，瓶颈全在搬权重。
  要提利用率只有两条路：**加 batch**（提高算术强度）或**量化**（减少 bytes）。
  这条结论不需要真机就能得出，而且换 `CANNON_CHIP` 就能看它怎么变 —— 这是本项目最实用的用法。
* **权重确实被搬了两遍**：`MemcpyH2D` 16.44 GB（safetensors → 模拟 HBM）之后，
  `MemcpyD2D` 又搬了 15.26 GB（加载器在设备内 `copy_` 一次）。
  这一次重复搬运在真机上同样存在，是 load_weight 阶段一个实打实的可优化点。
  ⚠ **但不要照着 stream 轨的 `56.9 ms` 读这笔账** —— 见 §2.7：
  那里面 44.1 ms 是 8813 次调用 × 5 µs 的固定 launch 开销，真正走带宽的只有 12.8 ms。
  按**访存腿**算，拷贝合计 25.7 ms，而 `aclnnMm` 一个人就 35.5 ms ——
  **计算腿是大头，不是搬运腿。**
* **`SUM/WALL = 1.018`，不要读成「并发度差」** —— 见 §4.8 最后一段，这是结构性的。

`validate_trace.py` 的判词：
`58687 events, 58652 spans, pids=0,1, streams=5, resources=cube,hbm,vector, edges=6,`
`published=all of 1810 sticky-capable request(s), contextRecords=0` —— **绿**。

### 2.5 换人设：同一份二进制，数值逐位相同，时延跟着 spec 走

用 CANNON 自带的 ACL 例子做对照（二进制一个字节没改，只换 `CANNON_CHIP`）：

| 人设 | HBM | BW | `MATMUL_ELAPSED_MS`（fp32，compute-bound） | `BANDWIDTH_ELAPSED_MS`（memory-bound） | CHECKSUM |
|---|---:|---:|---:|---:|---|
| `ascend910b` | 64 GB | 1.6 TB/s | 0.008360 | 0.083643 | `0x92cddef542b444aa` / `0x6d20cbab86454278` |
| `ascend910c` | 128 GB | 3.2 TB/s | 0.006680 | 0.044322 | **逐位相同** |
| `ascend950pr` | 144 GB | 4.0 TB/s | 0.014106 | 0.035457 | **逐位相同** |
| `ascend950dt` | 288 GB | 9.6 TB/s | 0.014106 | 0.017107 | **逐位相同** |

带宽那一列 83.6 → 44.3 → 35.5 → 17.1 µs，和 1.6 → 3.2 → 4.0 → 9.6 TB/s 逐项对得上。
matmul 那一列 **950 系比 910B 慢**，不是 bug：950 拿 fp32 吞吐换了低精度吞吐
（fp32 是 fp16 的 1/16，910B 是 1/4），USAGE §2.3 专门警告过别从 `CUBE TFLOPS` 那一列推 fp32。

### 2.6 CANNON 自身的验收

容器内 `bash scripts/e2e.sh`：**151 passed / 1 failed（152 条断言全部求值）**。
唯一那条红是「仓里存档的 19 份 gap 清单引用的 `libascendcl.so` blob 和我们刚重编出来的对不上」——
重建后的预期结果，不是功能失败。ACL 层的跨人设 roofline 断言（§12.4）全绿。

### 2.7 这张 profiling 表里，哪些数不能照面值读（2026-09-16 复核）

出完表之后我把 `merged.5768.json` 拆开逐事件核对了一遍，有四类数**会被误读**。

**(a) 51% 的「设备时间」是固定 launch 开销，不是被建模的工作。**
每个算子的 stream 跨度是 `max(计算腿, 访存腿) + kernel_overhead`，910B 的
`kernel_overhead_us = 5`。把 stream 跨度减去它自己最长的那条资源腿：

| op | calls | stream (µs) | 最长资源腿 (µs) | 固定开销 (µs) | 开销占比 |
|---|---:|---:|---:|---:|---:|
| `MemcpyD2D` | 8813 | 56862.5 | 12806.2 | **44056.3** | **77%** |
| `MemcpyD2H` | 1861 | 9392.8 | 88.4 | 9304.5 | **99%** |
| `aclnnMm` | 435 | 37661.9 | 35486.9 | 2175.0 | 6% |
| `aclnnRmsNorm` | 219 | 1096.8 | 1.8 | 1095.0 | **99.8%** |
| `aclnnAdd` / `AddRmsNorm` / `SwiGlu` / `Cat` | 767 | 3902.0 | 67.0 | 3835.0 | **98%** |
| **合计** | **13797** | **135817.4** | — | **68975.5** | **51%** |

⟹ **`135.8 ms` 这个总数有一半是「调用了多少次」决定的，而不是「搬了多少字节 / 算了多少 flop」。**
更糟的是那 8813 次 D2D 里 **8652 次小于 1 MiB**（尺寸分布：8652 个 ~0、
72 个 ~32 MiB、36 个 ~64 MiB、36 个 ~128 MiB、2 个 ~1 GiB），
也就是说 **15 GiB 的权重只用了 146 次拷贝，剩下 8652 次几乎不搬字节却吃掉 43 ms 的账** ——
那是**探针自己的张量 marshalling**，是仪器开销落进了被测量里。

**(b) gather / scatter / paged 三类算子按「操作数张量整体大小」计账，错 3–5 个数量级。**
成本模型的 `bytes_moved` 取的是操作数的 size，对 GEMM/norm 是对的，对这三类不是：

| op | 实际操作数 | 记的 bytes | 真机实际要动的 | 倍数 |
|---|---|---:|---:|---:|
| `aclnnEmbedding` | in `fp16[151936,4096]`, idx `int64[1]` → out `fp16[1,4096]` | **1,244,667,912** | 8 KiB（一行） | ~**152000×** |
| `aclnnScatterNdUpdate` | var `fp16[2176,8,128]`, upd `fp16[1,8,128]` | **8,914,952** | 2 KiB（一个 token 的 K） | ~**4000×** |
| `aclnnFusedInferAttentionScoreV4` | q `fp16[1,32,128]`, mask `bool[2048,2048]`, k/v `fp16[17,128,1024]` | **13,123,588** | ~8 KiB（kv_len≈1）+ 压缩 mask | ~**1000×** |

⚠ **后果不只是数偏大，是一整类问题会被答反**：按这个模型，
**KV 池开大一倍，decode 的 attention 和 KV 写就慢一倍** —— 而 paged attention 的全部意义
就是它**不会**。`block_table` 与 `actual_seq_lengths` 都传进来了，计费时没被用上。
同理，**attention 随序列长度的增长在这张表里看不见**（记的是池子大小，不是 kv_len）。

**(c) `resource: vector` 的 1.13 µs 是下限夹出来的，不含信息。**
我上一版把「vector ≈ 0%」当成结论，这个说法要收回。逐 span 查：
`aclnnAdds`/`Sub`/`Subs`/`IndexPutImpl` **100%** 落在 `Claim::MIN_OCCUPANCY = 1 ns`，
`aclnnAdd`/`RmsNorm`/`AddRmsNorm`/`ArgMax` 的 vector 腿也全部在下限上
（它们各发两条资源腿，vector 那条到底，hbm 那条没到，所以计数上是 50%）。
809 条 vector span × ~1.4 ns ≈ 1.1 µs —— **这个数只说明「vector 腿一律被夹到地板」，
不说明向量单元的利用率**。`cube 2.25%` 和 `hbm 50.1%` 是有信息的，vector 这格不是。

**(d) `IDLE = 0.000` / `SUM/WALL = 1.018` 不是利用率。**
独立核算：13797 条 stream span 合并后是**整整一段**，空洞 0 个、总长 0.000 µs。
原因是**虚拟时钟只在有工作提交时才前进** —— 两次 kernel 之间那 2000 多秒的真实 host 时间，
在设备时间轴上根本没有表示。所以 `IDLE=0` 是「这把尺子没有空闲这个刻度」，
不是「设备一刻没闲着」。同理 SGLang 回给调用方的 `e2e_latency: 2053.196`
是一次 CPU 仿真的墙钟，**不是任何东西的时延**。

### 2.8 墙钟去哪了：97.3% 在一颗算子里

host 轨（`pid=0`，真实墙钟）29706 个事件、累计 **2070.6 s**：

| host 事件 | calls | 合计 | 占比 | 平均 |
|---|---:|---:|---:|---:|
| `aclnnMm` | 435 | **2015.7 s** | **97.3%** | **4.63 s/次** |
| `aclnnEmbedding` | 3 | 18.1 s | 0.9% | 6.02 s/次 |
| `h2d` | 10114 | 12.3 s | 0.6% | 1.22 ms |
| `d2h` | 10674 | 9.4 s | 0.5% | 0.88 ms |
| `aclnnScatterNdUpdate` | 216 | 8.5 s | 0.4% | 39.1 ms |
| `aclnnFusedInferAttentionScoreV4` | 108 | 6.0 s | 0.3% | 55.1 ms |
| 其余 17 种 | ~4400 | < 1 s | — | — |

* **`aclnnMm` 一次 GEMV（`fp16[1,4096] × fp16[4096,6144]`，104 MFLOP）跑 4.63 s
  ⟹ ~22 MFLOP/s。** 一个朴素的单线程标量三重循环也有几百 MFLOP/s，
  所以问题不止是「没并行」，**这条路本身比朴素实现还慢几十倍**（大概率是逐元素的
  fp16 转换 / 边界检查 / 无分块）。这是整个开发回路上**唯一值得先修的性能项**。
* **`aclnnEmbedding` 取一行要 6 s** —— 和 (b) 同源：后端真的把 1.24 GB 的表整个过了一遍，
  所以它不只是**记账**错，是**真算**也白算。`ScatterNdUpdate` 39 ms/次、FIA 55 ms/次 同理。
* **host 与设备两条轨的拷贝计数刚好对得上，而且揭示了一件事**：
  `host h2d 10114 = 设备 H2D 1301 + D2D 8813`，`host d2h 10674 = 设备 D2H 1861 + D2D 8813`
  —— **逐位精确**。⟹ **CPU 后端把每一次「设备内拷贝」实现成了 host 的一次 d2h + 一次 h2d**，
  即 D2D 在真身上是绕经宿主内存的往返。设备时间轴把它建模成一条 D2D 是对的，
  但这解释了 load_weight 为什么要 56 s。

---

### 2.9 第二轮：四堵墙补进 CANNON、撤掉垫片之后（2026-09-16 下午）

分支 `feat/ops-for-qwen3-sglang`。三趟走查，`USE_LAB_SHIMS=0`：

| 趟 | 改动状态 | 结果 |
|---|---|---|
| `noshim` | 四堵墙已补进 CANNON，垫片撤掉 | `driver 退出码 = 0`，Engine 76.2s，generate 2100.6s |
| `withmul` | 上面 + 引擎里新增 `aclnnMul`（未路由） | `退出码 = 0`，profiling 表与 `noshim` **逐字节相同** |
| `mul2` | 上面 + `aten::mul.out` 路由到 `aclnnMul` | `退出码 = 0`，Engine 78.6s，generate 2042.5s |

**第一条结论就是 `withmul`**：算子进了引擎、过了 519 条一致性用例、符号面 113 → 115、
e2e 156 条断言全评，**而走查一点没变**。见 §0 第 5 条。

#### (a) profiling 对照（`noshim` → `mul2`）

| operator | noshim calls | noshim dur (µs) | **mul2** calls | **mul2** dur (µs) | Δ dur |
|---|---:|---:|---:|---:|---:|
| `aclnnMm[fp16]` | 435 | 37661.9 | 435 | 37661.9 | — |
| `MemcpyD2D` | 4493 | 35262.5 | 4493 | 35262.5 | — |
| **`MemcpyD2H`** | **6613** | **33158.9** | **5301** | **26583.6** | **−6575.3** |
| **`MemcpyH2D`** | **1949** | **22594.9** | **1515** | **20413.2** | **−2181.7** |
| `aclnnEmbedding[fp16]` | 3 | 2932.2 | 3 | 2932.2 | — |
| `aclnnScatterNdUpdate[fp16]` | 216 | 2584.4 | 216 | 2584.4 | — |
| **`aclnnMul[fp32,int64]`** | — | — | **439** | **2210.5** | **+2210.5** |
| `aclnnFusedInferAttentionScoreV4[fp16]` | 108 | 1411.4 | 108 | 1411.4 | — |
| `aclnnCat` / `Add` / `RmsNorm` / `AddRmsNorm` / `SwiGlu` / `ArgMax` / … | | | 全部不变 | 全部不变 | — |
| **total** | **14877** | **141232.5** | **13570** | **134686.1** | **−6546.5（−4.6%）** |

Δ 逐项对得上：`+2210.5 − 6575.3 − 2181.7 = −6546.5`。

#### (b) fallback 对照

| | noshim | mul2 |
|---|---:|---:|
| fallback 总调用 | **2496** | **768**（−69%） |
| 涉及 aten 算子 | 20 | 19 |
| 搬运字节 | 91.18 MiB | 72.44 MiB |
| **`aten::mul.out`** | **1280 / 18.68 MiB** | **0 —— 整条消失** |
| `__fallback__`（未归因） | 703 | 268 |
| `aten::neg.out` | 422 / 2.07 MiB | 409 / 2.00 MiB |

**两个数对得起来，而且它们数的不是同一件事**：`gaps.jsonl` 里 `kind=forward` 的
`aten::mul.out` 是 **439** 次，和 trace 里 `aclnnMul` 的 439 条 span 逐一相等；
census 的 1280 是**拷贝事件数**，它的三个大桶是 423 / 421 / 420（16384 B、512 B、4096 B），
正是每次调用把 `self` / `other` / `out` 各搬一趟 —— 439 × 3 ≈ 1280。
⚠ 这也补了 §2.7 的一个漏：**census 的计数不能当调用数读**，我第一版就是这么读的。

标量臂有独立记录：`kind=workaround` 的 `aten::mul.out scalar arm` **5 次**（RoPE 的
`1.0 / (...)` → `__rdiv__` → `tensor * float`），按登记的偏离走了张量臂，见 §3.7。

#### (c) 这一轮最值钱的一条：把算子搬上设备，反而让「设备时间」小了 4.6%

`aclnnMul` 自己加了 2210 µs，而少掉的 **1746 次 fallback 往返**省了 8757 µs。
其中 **1746 × 5 µs = 8730 µs 是固定 launch 开销，占 99.7%**。

⟹ **这是 §2.7(a) 的一次独立验证**，而且是反方向来的：那一节说「这台仪器的设备时间有一半由
调用次数决定，不是由字节决定」；这里消掉 18.68 MiB 的往返，省下来的几乎全是次数的钱，
跟字节数基本无关。一条靠拆事件得出的结论，被一次真实改动复现了。

#### (d) 没有变的东西，同样是读数

| 资源轨 | noshim | mul2 |
|---|---|---|
| `resource: cube` | 543 spans / 2274.681 µs | **完全相同** |
| `resource: hbm` | 14877 spans / 66857.5 µs | 13570 spans / 66845.6 µs（−12 µs） |
| `resource: vector` | 809 spans / **1.133 µs** | 1248 spans（+439，正是 Mul）/ **1.707 µs** |
| `SUM/WALL` | 1.021 | 1.018 |

* **cube 一动没动** —— `aclnnMul` 是 vector 算子。GEMM 仍是唯一的 cube 消费者，利用率仍 2.25%。
* **hbm busy 基本不变** —— 字节账本来就守恒，只是从 Memcpy 头上挪到了 Mul 头上。
* **vector 那一格仍然没有信息** —— 439 条新 span 全部落在 `Claim::MIN_OCCUPANCY = 1 ns` 地板上，
  busy 只从 1.133 涨到 1.707 µs。§2.7(c) 的结论不变。
* **输出 token 仍是 `danny`** —— 补算子没碰数值问题，§2.2 那三条嫌疑一条都没排除。

---

## 3. CANNON 的 gap 与不足（要求 ②的后半）

**按撞到的顺序**记。每一条都给「墙在哪、判据是什么、要多大工作量」。

### 3.1 【✅ 已修】视图算子没注册：`aten::unfold`

> **状态（2026-09-16 下午）**：已补进 CANNON —— `client/torch_npu_shim/csrc/device.cpp` 的
> `TORCH_LIBRARY_IMPL(aten, PrivateUse1)` 多了一行 `m.impl("unfold", cannon_npu::unfold)`，
> size/stride 推导逐行转录自 `at::native::unfold`（含 rank-0 臂），落到已经注册的 `as_strided`。
> `-Werror` 编过；`client/torch_npu_shim/selftest.py` 从 **145 项涨到 163 项全绿**，新增段
> 直接复现原来那堵墙：`reshape → repeat → chunk` 数值正确、视图真的 alias、写穿视图能落进
> base、100 次调用 `alloc_calls` 与 `live_bytes` 都不动。下面是**当时**的现场记录，保留原样。

```
torch_npu._gaps.Unimplemented: unimplemented: aten.unfold.default —
  its return aliases an npu operand and this fallback cannot produce an alias.
  Register a device kernel for it in client/torch_npu_shim/ (task #185).
```

* **触发点**：`sglang/srt/layers/rotary_embedding/base.py:223`
  `cos_sin.reshape(-1, 2, d//2).repeat(1, 1, 2).chunk(2, dim=-2)`。
  `Tensor.repeat` 在 ATen 里（`native/Repeat.cpp`）是用 `unfold` 取视图再 `copy_` **写进去**的。
* **为什么拒绝是对的**：deep 模式的 fallback 是「搬 host → 算 → 搬回」，**造不出 alias**；
  返回副本的话 `repeat` 会写进一块没人看的内存，**结果静默错**。
* **为什么这条最值得修**：CANNON 的 C++ 半边**已经注册了 `as_strided`**
  （`client/torch_npu_shim/csrc/device.cpp:1782`，同一张表里还有 `view`、`_reshape_alias`），
  而 `unfold` 按 ATen 自己的定义就是一次 `as_strided`。
  **缺的不是能力，是覆盖面。** 我们在实验台侧用十几行 Python 补上就过去了
  （`torch.library.Library("aten","IMPL").impl("unfold", ..., "PrivateUse1")`），
  搬进 `csrc/device.cpp` 是同样的十几行。
* **射程**：`repeat` / `unfold` 是通用写法，不是 Qwen3 特有。这堵墙会拦住一大类模型。

### 3.2 【✅ 那一颗已顶替，包仍未顶替】`sgl_kernel_npu`

> **状态（2026-09-16 下午）**：`split_qkv_rmsnorm_rope` 的顶替件已从实验台搬进 CANNON ——
> `torch_npu/_impl.py` 的 `REGISTRY` 多了一项：逐头 RMSNorm 走本仓自己的 `npu_rms_norm`
> 转发器（落 `aclnnRmsNorm`），RoPE **重定向到 sglang 自己的**
> `apply_rotary_pos_emb_native_eager` —— 和既有的 `alloc_extend_kernel` 重定向到
> `alloc_extend_naive` 是同一套纪律：调用方自己带着一份纯 torch 的等价实现，照它走比我们
> 重写一遍更不容易错，也更容易被证伪。对 f64 参考：q 3.97e-4 / k 6.72e-4 / v 严格为 0，
> 外加一条阴性对照（改 `q_norm` 权重答案必须变）。
> 
> ⚠ **这一条只倒了一颗，不是一堵墙倒了**：`sgl_kernel_npu` 仍然是 sglang 自己的外部编译件，
> 本仓现在顶替了它的**两颗**（KV 槽位记账 + 这一颗），其余照旧答 `Ghost`。换个模型
> （Qwen3-MoE / GLM4 / mamba 系）会立刻撞到别的。
> ⚠⚠ **融合被拆开的代价没有变**：真机上这一跳是一颗 kernel，这里是「1 颗 RmsNorm + 一串
> 逐元素」，per-op 单价可读、把这一跳加起来不可读（`DESIGN.md` §3.1）。

```
unimplemented: sgl_kernel_npu.norm.split_qkv_rmsnorm_rope.split_qkv_rmsnorm_rope
```

* **触发点**：`sglang/srt/models/qwen3.py:58` 在模块作用域 `if _is_npu:` 下无条件 import；
  `:191` 在每层的 `forward_prepare_npu` 里调用。
* **没有逃生口**：`qwen3.py:289` 是 `elif not _is_npu:` 才走 `forward_prepare_native`，
  而 `_is_npu` 一旦为真就再也回不去（USAGE §5.3 说的就是这件事）。
* **这个包是什么**：sglang **自己的**昇腾编译件（github.com/sgl-project/sgl-kernel-npu），
  不在 PyPI、不在 JamesBond 树里、要 CANN 工具链才编得出来。
  CANNON 只顶替了它的**一颗**（`torch_npu/_impl.py:139` 的 KV 槽位记账 `alloc_extend_kernel`，
  委托给 sglang 自己的 `alloc_extend_naive`），其余全部答 `Ghost`。
  CANNON 的 README 自己也写着这一条。
* **我们怎么过去的**：在 `cannon-lab/shims/` 放了一个自备的 `sgl_kernel_npu` 包，
  用 CANNON **已实现**的 `torch_npu.npu_rms_norm` + 逐元素算子把那颗融合算子拼出来。
  能盖住是因为 CANNON 的 ghost finder 是 `sys.meta_path.append` 上去的
  （`torch_npu/_ghost.py:218`），排在标准 `PathFinder` **后面** ——
  ⚠ 这也意味着**我们没有把仪器弄哑**：垫片没实现的子模块照旧落到 ghost 上按名报缺。
* ⚠⚠ **代价要说清楚**：真机上这一跳是**一颗融合 kernel**，我们拆成了
  「1 颗 RmsNorm + 一串逐元素」。按 `DESIGN.md` §3.1 的规矩，**per-op 单价仍可读，
  把这一跳的时延加起来不可读**。

### 3.3 【✅ 已修，但是用「记账」而不是「主张」修的】`aclnnCat` 拒收 strided 只读操作数

> **状态（2026-09-16 下午）**：`torch_npu/_forward.py` 的 `_STRIDED_OPERAND_POLICY` 补了
> `aclnnCat` 一行，**动作是 `_COMPACT` 而不是 `_PASS`**。理由写在行上：读厂商两层需要
> `.reference/`（gitignore 的 7.1 GB 厂商树），这台机器上没有，所以**不假装读过**；
> 而且 `PROVENANCE.md` §4 早就记着 `aclnn_cat.h` 既无公式也无计算图也无 dtype 子句，
> 它唯一那句散文写的还是**另一个算子**的接口。
> ⟹ `_COMPACT` 把这些字节记在 **`unsourced`** 头下 —— 那正是这个头的定义：「我们搬了这些
> 字节，而且无法出处证明厂商也搬」。这是一个**有名字、有计数的洞**，不是一个判决。
> 重开条件写在行上：`.reference/` 到位后读 `CatKernelNpuOpApi.cpp` 与 aclnn 实现，
> 读到了就换 `_PASS`（并给 domain）或 `_MIRROR`（并附厂商自己 contiguous 的行号）。

```
ForwardError: aclnnCat: a read-only operand arrived strided and no row in
  `_STRIDED_OPERAND_POLICY` says what the vendor does with it.
  (client/torch-npu-probe/torch_npu/_forward.py:1174)
```

* **触发点**：任何 `torch.cat((x[..., :n], x[..., n:]), -1)` —— 也就是**每一份
  `rotate_half` 的标准写法**。
* **拒绝的理由是好的**（注释里写着：「默认压实」曾让这个 marshaller 收下过真实栈会拒的输入），
  但 **`_STRIDED_OPERAND_POLICY` 这张表是按需长出来的，现在缺 `aclnnCat` 这一行**。
* **工作量**：读两层（op-plugin 的 `*KernelNpuOpApi.cpp` + aclnn 实现）然后加一行。
  调用方侧的绕法是 `.contiguous()`，但那**改变了字节搬运的账**，不是等价替换。

### 3.4 【✅ 已修，一行】同一颗算子的两种拼法只覆盖了一种

> **状态（2026-09-16 下午）**：`torch_npu/_impl.py` 的 `REGISTRY` 加了
> `"torch_npu.npu_fused_infer_attention_score": _forwarded(...)` —— 一行。
> 同时配了独立重算用例 `case_fia_attr`（`accept_forward.py` 的 `CASES` 是 deny-by-default
> 的，不配就红）：对 f64 golden 误差 4.04e-4（容差 1e-3），并断言**属性拼法与分发器拼法
> 逐位相同** —— 两个入口给出不同答案，就是两颗算子顶着一个名字。
> 
> ⚠ **根因没修，也修不了**：本仓的分母是对着一个**钉住的** sglang commit 机械提取的，
> 而 sglang 每周在动。这一行只是把**这一次**的漂移补上。下一次漂移仍然会以
> 「`unimplemented: <一个其实已经实现的算子>`」的形状出现。

```
unimplemented: torch_npu.npu_fused_infer_attention_score
```

* **触发点**：`sglang/srt/hardware_backend/npu/attention/ascend_backend.py:1419`，
  `attn_output, _ = torch_npu.npu_fused_infer_attention_score(query=q, ...)`。
* **CANNON 其实已经实现了这颗算子**：转发器在 `torch_npu/_forward.py:2207`，
  签名完整，落到引擎的 `aclnnFusedInferAttentionScoreV4`。
  但它只被注册在**一个名字**下 —— `torch_npu/_impl.py:211` 的
  `"torch.ops.npu.npu_fused_infer_attention_score"`。
  属性拼法 `torch_npu.npu_fused_infer_attention_score` 在 `_impl.REGISTRY` 里查不到，
  于是 `__getattr__` 返回 `Ghost`，一调用就按名报缺。
* **根因是版本漂移**：CANNON 的分母是对着**钉住的** sglang commit
  `e635577431cbdfb8ce5fafb0fcd8a4ac074062c6`（2026-08-30）机械提取的
  （`tools/sglang-callsites/`，它自己就统计了 3 个 homonym）；
  而这台机器上的 JamesBond 是 dev @ `7accfae6e`（2026-09-09），**换了拼法**。
* **工作量**：`_impl.REGISTRY` 加一行。我们在实验台侧就是一行 `setattr` 接过去的。
* **这条最能说明 CANNON 的结构性风险**：它的「支持矩阵」是对着一个钉住的 commit 量的，
  而 SGLang 每周都在动。**分母会过期，而过期的方式是静默的**——
  报出来的是「unimplemented」，看起来像缺算子，其实是缺一行别名。

### 3.5 【CANNON 缺口 · 环境】ATB 顶替面在没有 `.reference/` 的机器上建不出来

> **状态（2026-09-16 下午）：仍在。** 这台机器上 `.reference/` 还是没有，ATB 顶替面还是
> 建不出来，走查仍然靠 `ASCEND_USE_FIA=1` 绕开。⚠ 所以**默认配置到今天为止一次都没被测过** ——
> 本文所有读数都出自 FIA 那条路。

```
AtbError: .../client/atb_shim/build/atb_layout.json is missing —
  build the ATB stand-in first (bash client/atb_shim/build.sh).
```

跑 `client/atb_shim/build.sh`：

```
atb_shim/build.sh: the ATB target needs pinned include roots that are not here:
  .reference/torch_npu/third_party/op-plugin/op_plugin/third_party
  .reference/cann-acl/inc/external
  .reference/asc-devkit/include/adv_api/hccl/internal/hcomm
See DEVELOPING.md §6 ... this is exit 2 — could not be attempted — and not a success.
```

* **为什么重要**：SGLang **默认**的 KV cache 写入不走 aclnn，走 ATB
  （op-plugin 给 `_npu_reshape_and_cache` 建 `atb::infer::ReshapeAndCacheParam` 再 `RunAtbCmd`，
  整条路上一个 `aclnnXxx` 都没有 —— 那是**第二个 ABI**，`OpEntry` 拦不到）。
* **`.reference/` 是 gitignore 的 7.1 GB 厂商树**，一个新 clone 上没有，我们这台也没有。
  ⟹ **一个「零依赖、cargo build 就能跑」的项目，在 SGLang 这条路上其实有一个 7 GB 的隐性前置。**
* **绕法**：`ASCEND_USE_FIA=1`，SGLang 改走 `npu_scatter_nd_update_`（aclnn）那条路。
  这个绕法有效（trace 里 `aclnnScatterNdUpdate` 出现了），
  但**它换掉了被测配置** —— 默认配置没被测到。

### 3.6 【不是缺口，是量级】CPU 后端单线程，撞了 SGLang 的 300s 看门狗

> **状态（2026-09-16 下午）：仍在，而且第二轮又量了一次。** 三趟走查 generate 分别是
> 2100.6s / 2057.2s / 2042.5s（36 层 × 3 次前向），`%CPU` 稳定在 ~100%。补 `aclnnMul` 让
> **模拟设备时间**降了 4.6%（§2.9c），但**墙钟基本没动** —— 省下的是账面上的 launch 开销，
> 不是 CPU 上真花掉的时间。⚠ 这两件事在这台仪器上是分开的，别把前者读成后者。

```
Scheduler watchdog timeout (self.watchdog_timeout=300, self.soft=False)
```

* 实测：一次 36 层的前向在这台 128 核机器上跑 **~8 s/层**，且 `%CPU` 稳定在 **~100%**
  —— 也就是**单线程**。8B 模型每 token 约 16 GFLOP，这个速度说明后端 GEMM 没有做并行/分块。
* **这不是「模拟器慢」的托词**：CANNON 定位是功能测试，速度本不是它的承诺。
  但它有一个**实际后果**：默认配置下跑真模型会被上层框架的看门狗打死，
  而「被看门狗打死」和「撞到缺口」在日志里长得不一样但同样是红的，
  第一次遇到会当成功能问题查。
* **建议**：文档里加一句「跑 ≥1B 的真模型时把 `watchdog_timeout` 放大」，
  或者后端 GEMM 上个 BLAS/多线程 —— 128 核上白扔 127 个太可惜。

### 3.7 汇总：走查的缺口账

`gaps.jsonl` 里 `kind=call`（SGLang 点名要过而这里没有）在**最终配置下是 0**。
但按 CANNON 自己的话说，**「零 call 不等于零 fallback」**。
`copy-census` 显示这一趟掉进 CPU boxed fallback 的 aten 算子里，
量最大的几个是 RoPE 的逐元素运算：

⚠ **这句原本写的是「正是我们垫片引入的」，第二轮把它证伪了**：撤掉 `shims/` 之后
（`noshim` 那趟）`aten::mul.out` 仍然是 **1280 次**。这些乘法是 SGLang 自己 RoPE
路径上的，不是垫片带进来的 —— 垫片只是换了个地方做同一件事。

主跑（pid 5768）的 `copy-census`：**2496 次 fallback 调用，涉及 20 个 aten 算子**。

| aten 算子 | 次数 | D2H+H2D 字节 | 说明 |
|---|---:|---:|---|
| `aten::mul.out` | **1280** | 14.45 MiB | RoPE 的 `x*cos` / `rotate_half(x)*sin`。**CANNON v1 没有 `aclnnMul`** |
| `__fallback__`（未归因） | 703 | 0 | |
| `aten::neg.out` | 422 | 1.03 MiB | `rotate_half` 里的取负 |
| `aten::index_select` | 6 | 30.19 MiB | rope cache 按 position 取行 —— **单次 5 MB，最贵的一颗** |
| `aten::cos.out` / `aten::sin.out` | 各 4 | 各 20.12 MiB | rope cache 构造（`[40960, 64]` fp32） |
| `_local_scalar_dense` / `floor_divide` / `cumsum` / `max` / `eq` / `clamp` / `remainder` / `arange` / `lt` / `div` / `pow` … | 各 2–14 | ≈0 | 调度器与 attention metadata 的索引算术 |

三条读数：

1. **`aclnnMul` 的缺席是一条实打实的缺口**：21 颗算子里有 `Add`/`Adds`/`Sub`/`Subs`，
   唯独没有逐元素乘 —— 而 RoPE、attention scaling、门控、残差缩放到处都是乘法。
   它一个人贡献了一半以上的 fallback 次数。

   > **✅ 已修（2026-09-16 下午）。** `aclnnMul` 成了这个 build 的第 22 颗算子，
   > DEVELOPING.md §2 的「五处 + 会红的七类」逐条走完：`OpId::Mul = 45` / `N_OPS 38→39` /
   > 10 个穷尽 match 臂 / `cannon_backend.h` 的 `#define` / `OpEntry` + `prepare_mul` /
   > `CpuBackend::mul` 独立 kernel（没有 alpha，塞不进 `AlphaBinary`）/ shim 两个
   > `#[no_mangle]` / `cannon-oracle` 的 f64 参考 / `tests/cases/mul.toml` 12 条用例 /
   > `examples/30_mul.c`。门禁：**op_conformance 519 个用例、31 项全过**、
   > **符号面 113 → 115 导出，`check_symbols.sh` 8 passed / 0 failed**、
   > **e2e 156/156 断言全评**、两条棘轮（`PINNED_REACHLESS` 89→91、`KNOWN_LONG` 两项）
   > 刻意抬并写了理由。改后数据见 §2.9：`aten::mul.out` **1280 → 0**。
   >
   > ⚠ **标量臂是一处登记在案的偏离。** 厂商在 `IsCPUScalar(other)` 上分叉，标量侧走
   > `aclnnMuls` —— 本 build 没有这颗。走查**确实会走**这条臂（`RotaryEmbedding.__init__`
   > 的 `1.0 / (...)` → `__rdiv__` → `tensor * float`，每层一次，比第一个 token 早 36 层），
   > 所以现在把操作数搬上设备走张量臂，并记一条 `kind=workaround`（实测 5 次）。
   > **数值完全相同，偏离的是符号轨迹** —— README 的 quant-matmul 那行已给这一类命过名。
   > 重开条件：实现 `aclnnMuls`（这一族的第六颗，`aclnnAdds` / `aclnnSubs` 是它的兄弟）。
   >
   > ⚠⚠ **`aclnnSub` / `aclnnSubs` 至今仍是「引擎里有、没人路由」的状态**，和 `aclnnMul`
   > 被路由之前一模一样 —— 表里那 13 / 12 次是别的调用点来的，不是 `aten::sub.out`。
2. **`aten::index_select` 才是字节大头**（30 MiB / 6 次）。CANNON 有 `aclnnIndex` 和
   `aclnnEmbedding`，但 `index_select` 这条路没接上 —— 又是一个「能力在、覆盖面不在」。
   ⚠ **仍未修**，而且 `aten::mul.out` 走掉之后，它成了 fallback 字节数的第一名。
3. **一堆 int64 标量算术在 device↔host 之间来回搬**（`_local_scalar_dense` 等）。
   字节量可以忽略，但每一次都是一轮同步往返 —— 在 remoting 落地后这会变成真实开销。
   CANNON 的异步 fire-and-forget 设计（`DESIGN.md` §4.5b）正是为这件事准备的。

---

## 4. trace 是怎么跑出来的（要求 ③）

### 4.1 一条命令

```bash
cannon --chip ascend910b --deep --device --trace --out <目录> -- python3 qwen3_drive.py
```

`cannon` 是 CANNON 的 `torch-npu-probe` 包装出来的前端（`pip install -e` 时作为
console_script 装进 venv）。它**自己不做接管**，只做四件事：设开关、起子进程、
等它退出、把子进程留下的 trace 段**按 pid** 合并。

### 4.2 接管是怎么发生的（不是 `cannon` 做的）

`client/torch-npu-probe/pyproject.toml` 声明：

```toml
[project.entry-points."torch.backends"]
cannon_npu = "torch_npu:_autoload"
```

而 torch 在 `import torch` 的**末尾**会加载这一组 entry point 并逐个调用。
⟹ **只要这个包装进了 venv，任何 `import torch` 的脚本都已经被接管**，
跟命令行上有没有 `cannon` 无关。`_autoload` 做的事：

1. `torch.utils.rename_privateuse1_backend("npu")`，把 PrivateUse1 命名为 `npu`；
2. `ctypes.CDLL` 按绝对路径加载 `client/torch_npu_shim/build/libcannon_npu.so`（C++ 半边），
   它的 `DT_NEEDED` 又把 `libascendcl.so`（Rust 引擎）带起来 ——
   因为 cdylib 的 `SONAME` 正是 `libascendcl.so`，`dlopen(..., RTLD_NOLOAD)` 拿到**同一个句柄**，
   所以**一个进程只有一颗芯片、一本显存账**；
3. 装上 `sys.meta_path` 的 ghost finder，把 `torch_npu.*` 与 `sgl_kernel_npu` 全兜住；
4. 在 `--deep` 下装上转发层：`torch_npu.npu_rms_norm` 这类入口点不再是 ghost，
   而是把张量与算子**交给引擎真算**。

### 4.3 `--trace` 到底打开了什么

`cannon --trace` 写两个环境变量：

```
CANNON_TRACE=<out>/trace.json          # ⚠ 这是【基名】，不是最终文件名
CANNON_PROBE_FINALIZE_DIR=<out>/finalize-witness
```

引擎侧读到 `CANNON_TRACE` 非空，才会在**中间件链**上装 trace 装饰器
（USAGE §2.4：「不设则根本不装 trace 装饰器 —— 没人读的记录器是白花的开销」）。

装上之后有**两个记录器，两条时间轴**（`crates/cannon-middleware/src/trace.rs` 的文件头）：

| | 谁写 | 时钟 | Chrome Trace 的 `pid` |
|---|---|---|---|
| `TraceRecorder` | 引擎自己，把 `cannon-sched` 算出来的**模拟 span** 写进去 | **虚拟时钟** | `1 + 设备号`（`DEVICE_PID_BASE`） |
| `Trace` 装饰器 | 记它看到的 host 侧 backend 调用 | **墙钟** | `0` |

两者共享同一个 epoch（recorder 构造那一刻），都从 0 开始。
**故意分在不同 `pid`**：两者量级差可达 88000:1，画在同一条时间轴上虚拟轨会被压成一根线。

> ⚠ **设备维在 `pid` 上，不在轨名上**：两颗设备的默认流都是 raw handle `0`，
> 轨名逐字相同（`stream 0x0000000000000000`），靠 `pid` 区分 ——
> 一条轨的身份是 `(pid, tid)`。

### 4.4 时延是怎么算出来的（这是整件事的核心）

**不是测出来的，是按目标芯片 spec 记出来的。** `cannon-costmodel` 是一个 roofline 解析模型，
对每个算子算**两条腿**：

```
计算腿 = padded_flops / effective_rate        （cube 或 vector 的算力）
访存腿 = bytes_moved  / effective_bandwidth   （HBM 带宽）
算子跨度 = max(计算腿, 访存腿) + kernel_overhead
```

然后**每个资源各记各的账**（`DESIGN.md` §4.5）：

| 资源 | 占用取自哪条腿 |
|---|---|
| `Cube` / `Vector` | 计算腿 |
| `HbmBandwidth` | 访存腿 |
| `HccsLink` | 整段 |

⟹ trace 里 `stream ` 轨画的是算子的**整段跨度**，`resource: ` 轨画的是**那个资源自己被占了多久**。
一个 memory-bound 的 kernel 在 `stream` 轨上是长条、在 `resource: vector` 轨上只有一小格 ——
**这正是「这个算子把向量单元闲着」的样子，不是画错了**。
同一条 `resource:` 轨上两个 span **永远不许重叠**，`scripts/validate_trace.py` 会为此拒收 trace。

`cannon-sched` 是一个极小的离散事件模拟器：每条 stream 一个时钟，
`RecordEvent` 打虚拟时间戳、`StreamWaitEvent` 做 `now = max(now, event.ts)`、
`SynchronizeDevice` 取全局 max。`aclrtEventElapsedTime()` 返回的是**虚拟时钟差值** ——
这是模拟时序唯一透给上层引擎看的接口。

### 4.5 落盘：边跑边写分段，按 pid 合并

引擎**边跑边写分段**：每积累 `CANNON_TRACE_MAX_EVENTS / 8` 个事件写一段，文件名是

```
<基名>.<pid>.seg<NNNN>.json
```

pid 进文件名，是因为多进程引擎（SGLang 一趟留下 3 个 pid）共用一个 `CANNON_TRACE` 设置时，
最后写的那个会盖掉别人。分段是为了**被信号打死的进程也留得下已落盘的部分**：

> 判别式在段的内容里：每段带 `pid`/`ppid`/`segment`/`firstEventIndex`/`eventCount`/`finalSegment`。
> **一个进程恰好一个段标 `finalSegment: true`**；一个都没有 ⟹ 这个进程被杀了。

本次主跑的 `out/` 目录里：

```
trace.<pid>.seg0000.json … seg00NN.json   ← 原始段（引擎边跑边写）
trace.<pid>.json                           ← cannon 前端按 pid 合并好的文档
copy-census.json.<pid>                     ← 掉进 CPU fallback 的 aten 算子与字节数
gaps.jsonl                                 ← 缺口清单（kind=call / attr / import / forward / …）
finalize-witness/                          ← finalize-<pid>.pid / .done
```

⚠ **一份合并文档只能是一个进程的**（任务 #234）：引擎是 per-process 的静态量，
把两个进程折进一份就是把两条时间线原样叠印，`costModel`、teardown 普查会记到别人头上。
所以工具**拒绝**替你挑一个：

```bash
python3 scripts/trace_segments.py <基名> --list            # 这个基名下有哪几个进程
python3 scripts/trace_segments.py <基名> --pid <PID> -o merged.json
python3 scripts/validate_trace.py merged.json              # 三条普查闸，按「缺席不等于零」判红
```

⚠ **别自己手搓合并器**——CANNON 自己记过一次：#198 的 A/B 量测手搓过一个，
写在 teardown 普查落地之前，于是两条臂的普查都悄悄没了。

### 4.6 让 trace 真的收尾（`finalize` 那一跳）

`engine.shutdown()` 最后会走 SGLang 的 `kill_process_tree`，**SIGKILL 没有缓冲期** ——
调度器子进程来不及 `aclFinalize`，于是一个 `finalSegment: true` 都没有。
所以驱动在 `shutdown()` **之前**先做一件事（照抄 `probe/drive.py` 任务 #230）：

1. 读 `CANNON_PROBE_FINALIZE_DIR` 下的 `finalize-<pid>.pid` 见证文件；
2. **只给写了见证的 pid** 发 `SIGUSR1`（SIGUSR1 默认动作是终止进程，
   发给没挂 handler 的 detokenizer 会把它打死并引发 SGLang 自己的 SIGQUIT 级联）；
3. 等 `finalize-<pid>.done` 出现（30s 上限）再让 `shutdown()` 去 kill。

顺序**不是论证出来的，是构造出来的**：kill 发生在这一步返回之后，
而这一步在最后一段落盘之前不返回。

### 4.7 怎么看

* 合并好的 `trace.<pid>.json` 直接拖进 `chrome://tracing` 或 Perfetto。
* 每个设备侧事件都带着记账口径，直接可读：

```json
{"ph":"X","pid":1,"tid":17,"ts":220.792,"dur":37.768,"name":"aclnnCat","cat":"stream",
 "args":{"op":"aclnnCat","dtype":"fp32","inputs":"fp32[40960,64], fp32[40960,64]",
         "outputs":"fp32[40960,128]","flops":0,"bytes":41943040,
         "resources":"hbm","bound":"memory-bound"}}
```

* 命令行聚合：
  `python3 scripts/trace_stats.py <trace>` → §2.2 那张逐算子表（calls / dur / flops / bytes / bound）
  `target/release/cannond trace summary <trace>` → 逐轨 + 按设备的 `SUM/UNION/WALL/SUM·WALL/IDLE` 聚合 + teardown 普查

### 4.8 这些数字能读到什么程度

| 想问的 | 能不能 |
|---|---|
| 换 910B → 910C，KV cache 还装得下吗 | ✅ 显存按人设计费 |
| 这个 GEMM 是 compute-bound 还是 memory-bound | ✅ per-resource 轨直接画出来（§2.2 就是答案） |
| 我的代码有没有句柄泄漏 / 越界拷贝 / 用错 `aclrtMemcpyKind` | ✅ CANNON 最强的地方 |
| 真机上每秒多少 token | ❌ 解析模型，不是周期精确模拟器 |
| 这一跳融合 kernel 的时延 | ❌ 我们把它拆开了（§3.2），单价可读、求和不可读 |
| kernel 内流水线重叠 / double buffer 掩盖率 | ❌ **CANNON 永不输出这类数字**，`tests/output_contract.rs` 有机器在守 |

还有一条本次走查自己的射程限制：**`SUM/WALL ≈ 1.00`，不要读成「并发度差」**。
SGLang 的调度器是**单线程循环、一次一个 batch**，
`event_loop_overlap` 的 overlap 是 **CPU 处理 × 设备计算**，而设备聚合按定义**不含 host 轨**。
⟹ 单设备 SGLang 下 `sum/wall ≈ 1` 是**结构性的**，不是待修的缺口。

---

## 5. 性价比排序 —— 已做的与没做的

### 5.1 已经做掉的（分支 `feat/ops-for-qwen3-sglang`）

| # | 事项 | 实际工作量 | 证据 |
|---|---|---|---|
| 1 | `csrc/device.cpp` 注册 `aten::unfold`（照 `as_strided`） | 60 行（含论证注释），实现体 ~20 行 | selftest 145 → **163 项全绿** |
| 2 | `_impl.REGISTRY` 补 `torch_npu.npu_fused_infer_attention_score` 属性拼法 | **1 行** + 一条重算用例 | `case_fia_attr` 误差 4.04e-4，两种拼法逐位相同 |
| 3 | `_STRIDED_OPERAND_POLICY` 补 `aclnnCat` 的行 | 1 行 + 论证 | 记在 `unsourced` 头下，附重开条件（§3.3） |
| 4 | 顶替 `sgl_kernel_npu.norm.split_qkv_rmsnorm_rope` | 一个 stand-in（重定向式） | 对 f64 参考 q 3.97e-4 / k 6.72e-4 / v = 0 + 阴性对照 |
| 5 | **实现 `aclnnMul`** | 五处 + 门禁七类 + example，27 个文件 832 行 | op_conformance **519 用例 31 项全过**；符号面 113 → **115**；e2e **156/156** |
| 6 | **把 `aten::mul.out` 路由到它** | `_FORWARDED_ATEN` 一行 + 转发器 + 重算用例 | `aten::mul.out` fallback **1280 → 0**（§2.9b） |

⚠ 第 5 项和第 6 项是**两件事**，这是这一轮最贵的一课：只做第 5 项时走查的 profiling 表
**逐字节没变**。见 §0 第 5 条。

### 5.2 还没做的，按性价比

| # | 事项 | 工作量 | 收益 |
|---|---|---|---|
| 1 | 把 `aten::sub.out` / `aten::index_select` 也路由过去 | 每条一行 + 一个转发器 | `aclnnSub`/`aclnnSubs` 现在仍是「引擎里有、没人用」；`index_select` 是 fallback 字节数新的第一名（30 MiB / 6 次） |
| 2 | 实现 `aclnnMuls`（标量臂） | 一颗算子 | 消掉 §3.7 那条登记在案的符号轨迹偏离 |
| 3 | 顶替 `sgl_kernel_npu` 余下的融合算子 | 每颗一个转发器 | Qwen3-MoE / GLM4 / mamba 系的路径；否则每换一个模型就撞一次 |
| 4 | 修 gather/scatter/paged 三类的 `bytes_moved` 口径 | 中 | §2.7(b)：现在错 3–5 个数量级，而且会把「KV 池变大 decode 会不会变慢」**答反** |
| 5 | `.reference/` 的替代方案（vendored 最小头文件，或收窄 ATB 面的依赖） | 中 | 让「cargo build 就能跑」在 SGLang 这条路上也成立；顺带解锁默认配置的走查 |
| 6 | 后端 GEMM 上多线程 / BLAS | 中 | `aclnnMm` 现在 ~22 MFLOP/s（§2.8），128 核只用 1 核 |
| 7 | trace 事件带上 layer / phase 归因 | 中 | §2.7 的「信息缺失」第 2 条：现在问不了「第 17 层的 MLP 慢不慢」 |
| 8 | 文档：CANNON `USAGE.md` §1 的支持矩阵补一格「真权重大模型」的现状 | 小 | 现在那格写的是 tiny-qwen2 + dummy 权重，读者会高估射程 |

### 5.3 交卡前还欠的仪式

`cargo test --workspace` 现在只剩 **`repo_hygiene` 的三条仪式跟踪线**：覆盖率要重测
（100% 地板）、gates 仪式要重跑、探针收据要重盖。三条都不是正确性问题，是「你动了树就得
重走一遍典礼并盖章」，而盖章是仓库主人的动作。
