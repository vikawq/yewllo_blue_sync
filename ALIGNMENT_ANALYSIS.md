# 大 shape 下内存地址对不齐的成因分析

针对 `mojo_opset/backends/ttx/kernels/npu/{layernorm,rmsnorm}.py` 在 standalone 模拟器
（内存窗口 `[0, 0x80000000)`）上的对齐告警。

---

## 0. 先分清两类"对不齐"

两者的现象在日志里很像，但成因完全不同，混在一起会把排查带偏。

| | A 类：野指针 | B 类：结构性行对齐 |
|---|---|---|
| 成因 | `BLOCK_SIZE_M` 未绑定，`rows_off` / `rows_mask` 是垃圾值 | 行起始地址天然不满足 32 字节对齐 |
| 与 shape 的关系 | **无关**，小 shape 一样炸 | **强相关**，只有特定 `n_cols` 命中 |
| 地址低 2 位 | 可以非 0 | **必定为 0**（它是合法的元素指针） |
| 地址低 5 位 | 任意 | 非 0，且等于 `(行号 × n_cols × esize) mod 32` |
| `addr - base` | 无规律 | 能被 `n_cols × esize` 整除 —— 正好落在某一行起点 |
| 伴随现象 | 大范围越界读写，地址完全在分配区之外 | 只是对齐告警 / 性能塌陷，地址仍在 buffer 内 |

**已报的三个地址全部是 A 类：**

| 报告地址 | 低 2 位 | 按 fp32 解释 | 判定 |
|---|---|---|---|
| `0x3D1FE691` | `01` | 0.03904 | A 类 |
| `0x3CF7F662` | `10` | 0.03027 | A 类 |
| `0x3D67EC4D` | `01` | 0.05662 | A 类 |

任何指向 fp32 数组的合法元素指针，低 2 位必然是 `00`。这三个都不是，所以它们不是算出来的地址，
而是被当成地址用的 float 数值（量级正好是各自 `n_cols` 的 `mean(randn)`）。根因是
`BLOCK_SIZE_M` 未绑定，详见下面第 4 节。

**下面第 1–3 节讲的是 B 类**，也就是"为什么大 shape 会对不齐"这个问题本身。

---

## 1. B 类的机制：行跨距决定一切

两个 kernel 的地址都是这样算的：

```python
x_ptrs = X_ptr + rows_off[:, None] * stride_x_row + cols_off[None, :]
```

`x_2d = x.reshape(-1, dim)` 对连续输入来说 `stride_x_row == n_cols`，所以**第 r 行的起始字节偏移是**：

```
offset(r) = r × n_cols × element_size
```

`X_ptr` 本身是分配器给的（512 字节对齐，没问题），但每一行的起点只由 `n_cols × element_size` 决定。
硬件在 vector 场景下要求 UB 访问 32 字节对齐
（见 `triton-npu-kernel-opt/guides/platform/migration_guide/migrate_from_gpu.md`:319），
于是对齐条件是：

```
n_cols × element_size ≡ 0  (mod 32)
```

换成元素数：

| dtype | element_size | 要求 |
|---|---|---|
| fp32 | 4 | `n_cols % 8 == 0` |
| fp16 / bf16 | 2 | `n_cols % 16 == 0` |

**只要这个条件不满足，除第 0 行之外的每一行起点都是错位的**，而且错位相位逐行递推
（`offset(r) mod 32` 以 `n_cols × esize mod 32` 为步长循环）。这跟 buffer 多大、分配在哪都无关,
纯粹由 `n_cols` 决定。

---

## 2. 实测：命中的全是不规则的大 hidden size

把两个测试列表里所有 shape 过一遍这个条件：

### LayerNorm（`test_layernorm.py`）

| shape | n_cols | `n_cols*4 %32` | `n_cols*2 %32` | 行对齐 |
|---|---|---|---|---|
| (256, 128) | 128 | 0 | 0 | ✅ |
| (1/128/2048, 768) | 768 | 0 | 0 | ✅ |
| (64, 64, 256) | 256 | 0 | 0 | ✅ |
| (1/128/2048, 4096) | 4096 | 0 | 0 | ✅ |
| (512/128/65535, 8192) | 8192 | 0 | 0 | ✅ |

**LayerNorm 全部通过。** 所有 hidden size 都是 2 的幂，自然是 8 和 16 的倍数。
这也反证了：你在 LayerNorm 上看到的对齐告警不可能是 B 类，只能是 A 类野指针。

### RMSNorm（`test_rmsnorm_950_perf.py`）

| shape | n_cols | `n_cols*4 %32` | `n_cols*2 %32` | 行对齐 |
|---|---|---|---|---|
| (1, 32, 2048) | 2048 | 0 | 0 | ✅ |
| (1, 32, 1024) | 1024 | 0 | 0 | ✅ |
| (1, 64, 8192) | 8192 | 0 | 0 | ✅ |
| (1, 2, 256) | 256 | 0 | 0 | ✅ |
| **(1, 77, 489)** | 489 | **4** | **18** | ❌ |
| **(1, 57, 7338)** | 7338 | **8** | **20** | ❌ |
| **(1, 763, 8777)** | 8777 | **4** | **18** | ❌ |
| **(1, 7762, 18778)** | 18778 | **8** | **20** | ❌ |

**这就是答案的核心**：950 shape 列表里 4 个不规则 hidden size —— 489、7338、8777、18778 ——
**无一例外**全部违反 32 字节行对齐，三种 dtype 都违反。而规则的 2 的幂全部通过。

"大 shape 才对不齐"的真正原因不是"大"，而是**大 shape 恰好就是那些取自真实模型、
不是 2 的幂的 hidden size**。7338、8777、18778 这种数字是业务维度，不会凑成 8 的倍数；
而 1024/2048/8192 这类规则尺寸天然满足条件。两者在测试列表里正好和"小/大"重合了。

---

## 3. 为什么 shape 一大，后果就被放大

行对齐条件只和 `n_cols` 有关，但**后果的严重程度**确实随 shape 增长，有四个叠加因素：

### 3.1 错位行的数量正比于 n_rows

每一行是一个独立的 DMA descriptor。`n_rows` 从 2 涨到 7762，坏 descriptor 就从 1 个涨到 7761 个。
小 `n_rows` 时编译器还可能把整块当一次性搬运处理（peel 掉、或退化成标量访问而不报警），
`n_rows` 一大就必须走 strided DMA，相位问题全部暴露。

对照表里 `(1, 2, 256)` 对齐且只有 2 行，`(1, 7762, 18778)` 既不对齐又有 7762 行 —— 两头的极端。

### 3.2 `align()` 在 256 字节粒度上补齐，对奇数 n_cols 产出非 2 的幂

宿主机算 `BLOCK_SIZE_N` 用的是：

```python
def align(x, n_cols, aligned_bytes):      # aligned_bytes = VEC_ALIGN_BYTES = 256
    aligned_bytes = ceil_div(x.element_size() * n_cols, aligned_bytes) * aligned_bytes
    return aligned_bytes // x.element_size()
```

它按**字节**补到 256 的倍数，再除回元素数，所以结果不保证是 2 的幂：

| n_cols | `align()` fp32 | `align()` fp16 | `next_power_of_2` |
|---|---|---|---|
| 7338 | **7360** | **7424** | 8192 |
| 489 | 512 | 512 | 512 |
| 8777 | **8832** | 8832 | 16384 |
| 18778 | **18816** | 18816 | 32768 |

两个后果：

1. `tl.arange(0, 7360)` 不是 2 的幂。
2. **fp32 和 fp16 算出来的 BLOCK_SIZE_N 不一样**（7360 vs 7424）—— 所以这些奇数 shape 上
   fp32 和 fp16 走的是两套不同的 tiling，不能互相当对照。

注意 tile 宽度被补到 7360，但行跨距仍然是真实的 7338。tile 逻辑上靠 `col_mask` 兜住了，
但每行起点的错位一点没变。

### 3.3 tile 超 UB 之后编译器要切分，切点继承坏跨距

LayerNorm 的 tile 占用（fp32）：

| n_cols | BLOCK_M | BLOCK_N | tile 大小 |
|---|---|---|---|
| 128 | 24 | 128 | 12.0 KiB |
| 256 | 18 | 256 | 18.0 KiB |
| 768 / 1024 | 8 | 1024 | 32.0 KiB |
| 2048 / 4096 / 8192 | 4 | 2048 | 32.0 KiB |

单个 tile 不大，但 forward kernel 要**三趟**扫同一块数据（求和 / 求方差 / 写输出），
fp32 下每趟 32 KiB。一旦 `BLOCK_M × BLOCK_N × 4` 逼近 UB 上限，编译器必须把 tile 再切，
而切出来的每个子块起点都是 `起点 + k × stride`，坏跨距会按子块数量翻倍放大。

### 3.4 大 shape 把越界从"踩到自己"变成"踩出窗口"

同一个错位或越界偏移，在小 buffer 上大概率还落在堆里（静默读到别的数据，不报警），
buffer 一大就直接撞出 `[0, 0x80000000)`，这才变成可见的 `out-of-range`。

`(65535, 8192)` fp32 是个极端例子：

```
最后一个元素的字节偏移 = (65534 × 8192 + 8191) × 4 = 0x7FFF7FFC
2^31 = 0x80000000
余量 = 32 KiB
```

x 单独一份就吃掉整个窗口的 99.998%，再加上 y 和参考输出根本放不下。这个 case 跟对齐无关，
是容量问题，standalone 脚本默认跳过它并打印原因。但也说明一点：这个 shape 下**任何**
地址计算的毛刺都必然越界，没有任何余量可以容错。

---

## 4. A 类（你已经报的那三个）的成因

`mojo_opset/backends/ttx/kernels/npu/layernorm.py:39` 原本是：

```python
#@triton.heuristics({"BLOCK_SIZE_M": layer_norm_fwd_heuristics})   # 被注释掉
@libentry()
@triton.autotune(configs=[], key=["n_rows", "n_cols"])             # 顶替上来的空操作
@triton.jit
def _layernorm_fwd_kernel(..., BLOCK_SIZE_M: tl.constexpr):
```

`BLOCK_SIZE_M` 三条供给路径同时断掉：

1. 唯一提供它的 `@triton.heuristics` 被注释；
2. 替上来的 `autotune` 不做事 —— triton-ascend 自己的文档写明
   （`guides/platform/examples/06_autotune_example.md`）：
   *"This parameter can be left empty. If both split_params and tiling_params are empty,
   **autotune is not performed**."*
3. `layernorm_infer_impl` / `layernorm_fwd_impl` 两个 launch 点都只传 `BLOCK_SIZE_N=`。

本该报 `TypeError: missing a required argument`，但 `triton.runtime.libentry` 的
`LibEntry.run` 对取不到值的参数是 `continue` 跳过而不抛错，于是 launch 照常进行。结果：

```python
num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
rows_off  = block_start_row + tl.arange(0, BLOCK_SIZE_M)
rows_mask = rows_off < n_rows          # ← 保护一起失效
```

`rows_mask` 失效后 `tl.load(X_ptr + rows_off[:,None]*stride_x_row + cols_off)` 直接越界 ——
日志里的 `addr=0x8882bdf7a46a size=0x400` 就是它。`0x400 = 1024` 字节正好是
`(64,64,256)` 下 `BLOCK_SIZE_N=256` 一个列 tile 的 **fp32** 大小（fp16 是 `0x200`），
这也解释了为什么 fp32 炸而 fp16 过：同一个垃圾 `rows_off`，fp32 字节跨距是 fp16 的两倍，
走出去的距离也是两倍，fp16/bf16 侥幸还落在分配区内。

---

## 5. 排查判据

拿到一条对齐告警，按这个顺序判：

1. **看地址低 2 位。** 非 `00` → A 类野指针，不用再查对齐，直接查 `BLOCK_SIZE_M` 绑定。
2. **低 2 位是 `00`，算 `(addr - buffer_base) % (n_cols × esize)`。** 等于 0 → B 类，
   是某一行的起点，查 `n_cols × esize % 32`。
3. **看告警里的 `size`。** 它等于 `BLOCK_SIZE_N × esize` 时，说明越界发生在列 tile 的整体搬运上，
   偏移来自 `rows_off`；远小于它则是单元素标量访问的退化路径。

用 standalone 脚本做 A/B：

```bash
# A 类：repro 挂、fixed 过 → BLOCK_SIZE_M 绑定问题确认
python3 test_layernorm_standalone.py --variant repro --dtype float32
python3 test_layernorm_standalone.py --variant fixed --dtype float32
```

```bash
# B 类：默认用 align()（可能非 2 的幂），--block-n-pow2 向上取到 2 的幂
python3 test_rmsnorm_950_standalone.py --dtype float32
python3 test_rmsnorm_950_standalone.py --dtype float32 --block-n-pow2
```

---

## 6. 修复

### A 类（已改）

`mojo_opset/backends/ttx/kernels/npu/layernorm.py`，三处：

1. 恢复 `@triton.heuristics({"BLOCK_SIZE_M": layer_norm_fwd_heuristics})`，删掉空操作的 `@triton.autotune`。
2. `BLOCK_SIZE_N` 改用 `triton.next_power_of_2(n_cols)`（原本用 `align()`，`n_cols=768` 会原样返回 768）。
3. `mean` / `rstd` 改成 `dtype=torch.float32`（原本跟 `x.dtype`）。

同一个缺陷还在 `fused_add_layernorm.py`、`fused_add_rmsnorm.py`、`gelu.py`、`silu.py` 四个文件里，
launch 时都没传 `BLOCK_SIZE_M`。

### B 类（未改，需要决策）

行跨距是输入张量决定的，kernel 改不了，三个方向：

1. **宿主机 pad**：把 `x_2d` 的最后一维 pad 到 `32 / esize` 的倍数再进 kernel，出来再切回去。
   代价是一次额外拷贝 + 显存，但对齐问题根除。
   `guides/platform/migration_guide/migrate_from_gpu.md` 和 SKILL.md 第 5 节
   （*"host-side padding is usually safer than tail peeling"*）都推荐这个方向。
2. **按行切 program**：让每个 program 处理整行、`BLOCK_SIZE_M=1`，这样每次搬运是一维连续的
   `n_cols` 个元素，只有一个 head 需要修正，而不是每行一个。代价是 `n_cols` 小的时候并行度不足。
3. **接受它**：如果只是性能塌陷而不是功能错误，先量化代价再决定是否值得。
   建议用 `(1, 763, 8777)` 和一个 pad 到 8784 的版本对比一次 `--mode perf`。

对照实现在 `D:\workspaces\qemu\mojo_opset\mojo_opset\backends\ttx\kernels\npu\a5\`（a2/a5 拆分后的新版本），
那边的 `layernorm.py` 这三处本来就是对的，还多了一层 `STORE_STATS: tl.constexpr`：
推理路径传 `y, y` 占位、完全不写 stats，把 Mean/RSTD 那一组存储整个去掉。
