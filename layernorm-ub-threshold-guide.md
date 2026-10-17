# LayerNorm UB tile 阈值问题 — 源码排查指南

> 面向隔离环境的独立排查文档。阅读本文不需要其它上下文。
> 版本：2026-10-17

---

## 1. 任务

在 Ascend 950 的 standalone 仿真上，`_layernorm_fwd_kernel` 当**单份 X tile 的字节数超过约 6.5 KB** 时会产出错误结果（或 abort、或挂死）；在 6.5 KB 以内完全正确且 bit-exact。

```
出错 ⟺ BLOCK_SIZE_M × BLOCK_SIZE_N × itemsize  >  6656 B
```

现象层面已经定位清楚（见 §3），**要找的是这个阈值在源码里对应什么** —— 哪个常量、哪个 pass、哪条判断。

本文的目标产出：指出产生该阈值的源码位置，以及它为什么会导致生成错误的 UB 地址。

---

## 2. 一句话结论（已确立，不需要重做）

故障指令是一条向量 load（`RV_VLDI`，`size=256`），它的 **UB 地址操作数是从内存里读出来的张量数据，而不是计算出来的地址**。

证据：把崩溃日志里的"地址" `0x3EC7E035` 按小端字节序在仿真器 dump 出的输入 `X_ptr.bin` 里搜索，**唯一命中**，位置 `byte_off=0x018714`，即 `X[195, 69]`；`W`/`B`/`args` 全部零命中。按 `randn` 在该处的概率密度与 fp32 量化间隔估算，偶然命中概率约 0.036%。

进一步验证：**同一份 `kernel.o`，换输入数据就换故障地址**。把 X 全部填成常数 1.0，日志里的地址变成常数；早期脚本把输入缩放 0.1 倍时，地址的浮点解读也同步缩小一个数量级（0.039 对 0.390）。数据不影响 codegen，所以这几次是同一份二进制。

---

## 3. 关键数字与它们施加的约束

### 3.1 阈值区间

| 量 | 值 |
|---|---|
| 最大通过 tile | **6656 B** = 0x1A00 = 6.50 KB = 26×256 = 13×512 |
| 最小失败 tile | **7168 B** = 0x1C00 = 7.00 KB = 28×256 = 14×512 |

真阈值落在 `[6656, 7168)` 之间。

### 3.2 换个写法，搜索目标就变了

如果源码里的判断是 `>=` 形式，那么常量就是 **7168**，而 7168 的几种写法都非常"像手写的常量"：

```
7168 = 7 * 1024      ← 最可能
     = 0x1C00
     = 28 * 256       （256 = 向量寄存器宽度）
     = 14 * 512
```

**所以 grep 的首选目标是 7168 / 0x1C00 / 7*1024 / 28 / 14，而不是 6656。**

区间内另一个整齐值是 `6912 = 0x1B00 = 6.75 KB`，也值得一搜。

### 3.3 256 这个数

`RV_VLDI` 的 `size=256`，而 950 的向量寄存器宽度是 256 字节（64 个 fp32 lane）。把阈值换算成寄存器份数：

```
26 份通过  |  28 份失败
```

这个量级很像"向量寄存器文件被用满后开始溢出"。若 VRF 有 32 个寄存器、保留若干，可用数就在 26~28 之间。**这是优先级最高的假设（见 §5 H1）。**

### 3.4 已测到的放大系数

npuir 层的 UB 分配总量与 tile 的比值在两个配置上一致：

| 配置 | tile | npuir 层 UB 总占用 | 比值 |
|---|---|---|---|
| BM=24, BN=128, fp32 | 12 KB | 37.8 KB | 3.15× |
| BM=32, BN=128, fp32 | 16 KB | 50.0 KB | 3.125× |

比值稳定说明分配器行为是规整的，不是在某点失控。UB 共 256 KB，所以**容量不是瓶颈**。

---

## 4. 复现配置（供对照，不需要重跑）

被测 kernel 与生产代码逐字一致，只有 `constexpr` 取值不同。`BLOCK_SIZE_N = triton.next_power_of_2(n_cols)`（前向；反向才用 `align`）。

| BM | BN | dtype | n_rows | tile | 结果 |
|---:|---:|---|---:|---:|---|
| 4 | 128 | fp32 | 256 | 2.0 KB | 正确 |
| 6 | 128 | fp32 | 256 | 3.0 KB | 正确 |
| 8 | 128 | fp32 | 256 | 4.0 KB | 正确 |
| 12 | 128 | fp32 | 256 | 6.0 KB | 正确 |
| 24 | **64** | fp32 | 256 | 6.0 KB | 正确 |
| 24 | 128 | **fp16** | 256 | 6.0 KB | 正确 |
| **13** | 128 | fp32 | 256 | **6.5 KB** | **正确（最大通过）** |
| **14** | 128 | fp32 | 256 | **7.0 KB** | **结果错（最小失败）** |
| 15 | 128 | fp32 | 256 | 7.5 KB | 结果错 |
| 16 | 128 | fp32 | 256 | 8.0 KB | 结果错 / abort |
| 17 | 128 | fp32 | 256 | 8.5 KB | 结果错 |
| 18 | 128 | fp32 | 256 | 9.0 KB | abort |
| 24 | 128 | fp32 | 256 | 12.0 KB | abort（生产配置） |
| 32 | 128 | fp32 | 256 | 16.0 KB | 结果错 |
| 2 | 1024 | fp32 | **64** | 8.0 KB | abort |
| 18 | 256 | fp32 | 64 | 18.0 KB | 超时挂死（生产配置） |
| 64 | 128 | fp32 | 256 | 32.0 KB | 超时挂死 |

**6.0 KB 那一档由三条独立路径分别达到**（砍 `BLOCK_SIZE_M`、砍 `BLOCK_SIZE_N`、砍 `itemsize`），全部正确 —— 说明三者压的是同一个资源，判据只与乘积有关。

### 4.1 行数会掩盖故障（重要）

当 `ceil(n_rows / BLOCK_SIZE_M) ≤ grid` 时，每个核最多跑一轮行循环，故障可能不出现：

| 配置 | tile | n_rows | row task 数 | 结果 |
|---|---:|---:|---:|---|
| BM=2, BN=1024, fp32 | 8.0 KB | 8 | 4 | **通过** |
| BM=2, BN=1024, fp32 | 8.0 KB | 64 | 32 | **abort** |

同一 tile、同一 BM/BN，只改 `n_rows` 就从通过变 abort。**所以小规模用例通过不能作为安全依据。**

### 4.2 三种故障形态

同一个原因，表现取决于那个垃圾地址恰好落在哪里：

| 形态 | 条件 | 说明 |
|---|---|---|
| abort | 地址不满足对齐要求 | 命中仿真器 `rvec_dec.cpp:1005` 的 `check_addr_aligned` 断言。**真实硬件没有这道检查** |
| 跑完但结果错 | 地址碰巧对齐 | 只是读错位置，静默产出错数据。线上的主要风险 |
| 仿真挂死 | tile 很大（18 KB、32 KB） | 超时，原因未查 |

仿真器里还有一条**非致命**的检查：`rvec_dec.cpp:543` 的 `judge_valid_addr` → `ub_addr_overflow`，只打 ERROR 不中断。所以一个越界但对齐的地址会让程序带着错值继续跑。

---

## 5. 候选根因与排查顺序

### H0 — 设备属性被 stub 污染（**先查这个**）

**假设**：编译器从 driver / `torch_npu` 读取设备属性（UB 大小、向量寄存器数、core 数）来决定 tiling 与分配。该仿真环境里的 `torch_npu` 是**手搓的 stub**（`__version__ = "2.6.0.fake"`，`include/` 与 `lib/` 均为空目录），`npu_utils.so` 也是现场编译出来的。如果某个属性值不对，tiling 决策就会错。

**为什么优先**：若成立，则 6.5 KB 这个数**是环境产物，不能外推到真实硬件** —— 这会改变整个问题的性质和优先级。必须先排除。

**怎么查**：

```bash
# 编译器实际读到的属性
python3 -c "
import triton
u = triton.runtime.driver.active.utils
for k in dir(u):
    if not k.startswith('_'): print(k)
print(u.get_device_properties('npu'))
"
```

```bash
# npu_utils.cpp 暴露了哪些属性，默认值是什么
grep -nE "ub_size|UB_SIZE|vector|vreg|VREG|num_|core|get_device_properties" \
     <site-packages>/triton/backends/ascend/npu_utils.cpp
```

```bash
# stub torch_npu 自己返回了什么
sed -n '1,80p' <site-packages>/torch_npu/__init__.py
cat <site-packages>/torch_npu/version.py
cat <site-packages>/torch_npu/_C.py
```

**判据**：拿到 UB 大小、向量寄存器宽度/数量、core 数的实际取值，与 950 的真实规格核对。
- 若 UB 报成 256 KB、向量寄存器 256 B —— 属性正确，H0 排除，继续 H1
- 若有任何一项是占位值（0、1、硬编码的默认数） —— H0 成立，先修属性再谈阈值

---

### H1 — 向量寄存器压力 / 溢出（优先级最高的"真 bug"假设）

**假设**：tile 需要的向量寄存器数超过可用数（26~28 之间）后，编译器把 tile 溢出到 UB，而**溢出槽的寻址是错的** —— 这能直接解释"地址操作数读到 X 数据"（溢出槽与数据 tile 的地址算混了）。

**为什么合理**：
- 阈值换算成 256 B 的寄存器份数正好是 26 | 28，而 VRF 典型大小是 32
- 三条独立路径压到 24 份都通过
- 溢出代码的典型形态是成对的 store/load，可在反汇编里直接数

**怎么查**：

```bash
# 寄存器文件规格与溢出相关的常量
grep -rnE "VREG|vreg|NumVRegs|num_vregs|VECTOR_REG|RegFile|regfile" <triton-ascend 源码根> | head -40
grep -rniE "spill" <triton-ascend 源码根> | head -40
```

```bash
# 若有 MLIR pass 源码，找分配/切分 pass
grep -rlniE "tiledAlloc|tightly_coupled_buffer|hivm\.hir\.pointer_cast|address_space" <源码根>
```

**判据**：对比 BM=13（正确）与 BM=14（出错）两份编译产物的反汇编，看 BM=14 是否**多出成对的 store/load**。若有 → H1 成立。

---

### H2 — 硬编码的缓冲预算常量

**假设**：某个 pass 里有一个写死的"单缓冲上限"或"tile 上限"，超过就换一条有 bug 的 lowering 路径。

**怎么查**（按 §3.2 的优先顺序）：

```bash
R=<源码根>
# 首选：7 KB 的各种写法
grep -rnE "\b7168\b|0x1[cC]00|7\s*\*\s*1024|1024\s*\*\s*7" $R
# 次选：区间内其它整齐值
grep -rnE "\b6656\b|\b6912\b|0x1[aA]00|0x1[bB]00|13\s*\*\s*512|14\s*\*\s*512" $R
# 以份数为单位
grep -rnE "\b2[678]\b\s*\*\s*256|256\s*\*\s*\b2[678]\b" $R
# 带语义的上限名
grep -rniE "maxbuf|max_buf|buffer_limit|bufferLimit|maxtile|max_tile|tilelimit|budget|kMax.*(UB|Buf|Tile)" $R
```

**判据**：找到的常量 `B` 必须满足 `6656 ≤ B < 7168`（若判断是 `>`）或 `6656 < B ≤ 7168`（若是 `>=`）。不满足就不是它。

---

### H3 — npuir 以下的 lowering（当前证据指向这里）

**已知**：npuir 层（`kernel.npuir.mlir`）的 UB 分配表**是干净的**。已检查四份失败配置的产物（BM=24 与 BM=32），结果：

- 无缓冲重叠
- 无越过 UB 末端（最大占用 50 KB，UB 共 256 KB）
- 全部基址 32 字节对齐
- 故障地址 `0x18` **不是任何缓冲的基址**

npuir 把每个缓冲的 UB 基址写成字面常量，所以这一层是纯文本可读的：

```
%36 = hivm.hir.pointer_cast(%c256_i64) : memref<64x128xf32, #hivm.address_space<ub>>
         基址 256 字节 ↑                         ↑ 形状与 dtype
```

**结论**：bug 很可能在 npuir → 机器码这一段，而不是 npuir 的分配器。

**怎么查**：
1. 看 `kernel.ttadapter.mlir`（在 npuir 之后的一层），里面可能已经带上具体的地址计算
2. 对比 BM=13 / BM=14 两份的 ttadapter 与最终二进制
3. 定位 §7 列出的故障 PC，往上找最后一次写该地址寄存器的指令

**注意**：我们对 npuir 的解析只覆盖了 `hivm.hir.pointer_cast`。IR 里还存在 `memref.alloca`、`annotation.mark ... tiledAlloc` 等形式，**所以"50 KB"可能只是下界**。先确认这一点：

```bash
grep -oE "=\s*[a-z_][a-z0-9_.]*" <dump>/kernel.npuir.mlir | sort | uniq -c | sort -rn | head -20
```

---

### H4 — multibuffer / 双缓冲放大

**假设**：`multibuffer` 之类的流水优化把缓冲复制若干份，放大系数（实测 3.125×）与阈值相乘后越过某个真实限制。

**怎么查**：

```bash
grep -rniE "multibuffer|multi_buffer|double_buffer|pipeline_depth|num_stages" <源码根> | head -30
```

同一仓库的其它 kernel（如 `npu/rmsnorm.py` 的 `_rmsnorm_infer_kernel_single`）会把 `multibuffer`、`enable_vf_fusion` 作为启动参数传入，可以用它们做对照实验：固定 tile、只开关 multibuffer，看阈值是否移动。

**判据**：若关掉 multibuffer 后阈值从 6.5 KB 升高，则放大系数是阈值的一个乘数因子，H4 成立。

---

## 6. 源码落点清单

| 路径 | 内容 | 看什么 |
|---|---|---|
| `<site-packages>/triton/backends/ascend/driver.py` | NPU driver、`NPUUtils`、编译缓存 | 设备属性读取（H0）；`_build_or_get_cached_so` 的缓存 key |
| `<site-packages>/triton/backends/ascend/npu_utils.cpp` | 属性查询的 C++ 实现 | UB 大小、向量宽度等属性来源（H0） |
| `<site-packages>/triton/backends/ascend/utils.py` | 编译命令拼装 | 编译宏（`-DUSE_TORCH_NPU`、`-DTRITON_CANN_910`）如何影响 codegen |
| `<site-packages>/triton/backends/ascend/compiler.py` | 编译 pipeline、pass 顺序 | tiling/分配 pass 的参数与上限（H1/H2） |
| triton-ascend 的 MLIR pass 源码（若有） | `hivm` 方言、npuir 降级 | 分配器与地址生成（H1/H3） |
| `/opt/ascend/cann-*/tools/bisheng_compiler/` | 后端编译器 | 多为二进制；若有源码则是 H3 的落点 |
| mojo_opset `backends/ttx/kernels/npu/layernorm.py` | `TOKEN_BLOCK_SIZE_TABLE`、`layer_norm_fwd_heuristics` | **已知问题，见 §9** |

若只有安装好的包、没有编译器源码，那 H2/H3 的源码路线走不通，应把精力放在 H0 与 H1 的实测验证上。

---

## 7. 已观测到的故障点（用于反汇编定位）

`kernel.o` 是 ELF64，`Machine 0x1029`，LOAD 段 `VirtAddr=0`、`FileSiz=0x1e80`，`.text` 在文件偏移 `0xb0`。运行期 PC 基址是 `0x4000000`，所以：

```
.text 偏移 = PC - 0x4000000
```

`llvm-objdump --adjust-vma=0x4000000` 可让输出地址直接对上日志里的 PC。

| 配置 | 输入 | 致命 PC | 故障地址 | 按 fp32 解读 |
|---|---|---|---|---|
| BM=24, BN=128 | randn | `0x400173c` | `0x3EC7E035` | `X[195,69]` = 0.390 |
| BM=24, BN=128 | 全填 1.0 | `0x4001800` | `0x00000018` | — （= BM 24） |
| BM=24, BN=128 | 全填 0.0 | `0x4001800` | `0x00000018` | — |
| BM=18, BN=128 | 全填 0.0 | `0x4001c00` | `0x00000012` | — （= BM 18） |
| BM=16, BN=128 | randn | `0x4001e30` | `0x3EBEA9F3` | 0.372 |
| BM=2, BN=1024 | randn | `0x4001c78` | `0x00002024` | — （= tile 0x2000 + 36） |

同一组还有一条非致命记录：`pc: 0x4001c4c`，`start_addr: 0xc00c80`，`size: 256`。

**注意**：`地址 = BLOCK_SIZE_M` 这个规律只在 BM=24 和 BM=18 两个样本上成立，**不能外推** —— BM=16 的地址若等于 `0x10` 同样不对齐、本该 abort，实际只是算错。这些是各份编译产物的具体表现，不是通用机理。

---

## 8. 需要产出的证据

排查结束时希望拿到以下任意一项即可定性：

1. **一个常量**：源码里某处的上限值，满足 `6656 ≤ B < 7168`（或 `>=` 形式的 `6656 < B ≤ 7168`），且位于影响 tiling/分配/地址生成的路径上
2. **一段 diff**：BM=13 与 BM=14 两份编译产物的差异，显示 BM=14 多出溢出代码或换了寻址模式
3. **一个错误的设备属性**：编译器读到的 UB 大小 / 向量寄存器规格与真实硬件不符（这会使 6.5 KB 失去普适性）
4. **一条地址链**：从故障 PC 往上追到那个地址寄存器的赋值点，指出它为何取自数据缓冲

拿到 1 或 2 就可以向 triton-ascend 提单；拿到 3 则需要先修环境再重测。

---

## 9. 不需要重做的部分

### 9.1 已被实验推翻的六个模型

| 模型 | 推翻它的实验 |
|---|---|
| UB 总容量被打穿 | 致命地址 `0x18` 在 UB 范围内且无 `ub_addr_overflow`；BM=32（16 KB）比 BM=24（12 KB）更大却不 abort；npuir 层最大占用仅 50 KB |
| 向量寄存器份数 26 对 28 是边界 | 与字节阈值是同一个量，当时被三个 8 KB 的通过样本挡住（后查明那三个是 `n_rows` 太小掩盖了故障） |
| `BM % 32 != 0` 才出错 | BM=16（非 32 倍数）不 abort；BM=32（是 32 倍数）照样算错 |
| fp32 codegen 本身有问题 | fp32 在 tile ≤ 6.5 KB 时全部正确且 bit-exact |
| 失败只出现在"每行 512 字节"这一族 | `BN=256` fp32 BM=18（1024 B/行）超时挂死 |
| 字节阈值被 8 KB 的通过样本推翻 | 那三个样本 `n_rows ≤ 8`；把 `(8,1024)` BM=2 改成 `(64,1024)` BM=2 即 abort，阈值恢复 |

### 9.2 mojo_opset 侧的已知问题（不是本次排查目标）

`layer_norm_fwd_heuristics` 只用 `n_cols` 查 `TOKEN_BLOCK_SIZE_TABLE` 得出 `BLOCK_SIZE_M`，**既不看 `itemsize` 也不看 `BLOCK_SIZE_N`** —— 而判据是三者的乘积。所以每个 fp32 生产配置都落在阈值的数倍之外：

| n_cols | BN | 表值 BM | 表值 tile | 超线倍数 | 安全 BM 上限 |
|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 24 | 12.0 KB | 1.8× | 13 |
| 256 | 256 | 18 | 18.0 KB | 2.8× | 6 |
| 512 | 512 | 10 | 20.0 KB | 3.1× | 3 |
| 1024 | 1024 | 8 | 32.0 KB | 4.9× | 1 |
| 2048 | 2048 | 4 | 32.0 KB | 4.9× | 0（连 BM=1 都超） |

fp16/bf16 只有 `n_cols=128`（6.0 KB）在安全区内，从 `n_cols=256` 起同样超线。最初"fp16 全部通过"的观察是被测 shape 恰好都在线内造成的假象。

同一张表还在 `npu/rmsnorm.py`、`npu/fused_add_layernorm.py`、`npu/fused_add_rmsnorm.py` 里。后两个的 `@triton.heuristics` 是注释掉的，不吃这张表；`rmsnorm.py` 的训练前向 `rmsnorm_fwd_impl` 吃（表值 `128 → 48`，fp32 下 24 KB）。rmsnorm 这条线尚未验证。

这是"为何生产会踩到"的原因，opset 侧的止血方案是：

```python
BLOCK_SIZE_M = max(1, 6656 // (BLOCK_SIZE_N * itemsize))
# 当 BLOCK_SIZE_N * itemsize > 6656 时，须先降 BLOCK_SIZE_N
```

但这是绕开，不是修复。真正的修复在工具链侧，也就是本文要找的东西。

---

## 10. 环境注意事项

本排查**不需要仿真器**：读 IR 和反汇编 `kernel.o` 都只需要文件。但若要重新编译生成 IR dump，注意两点：

1. **`torch_npu` 元数据缺失**。容器里的 `torch_npu` 是 stub，没有 dist-info，而 `triton/backends/ascend/driver.py` 的 `NPUUtils.__init__` 会调 `importlib.metadata.version("torch_npu")` 给编译缓存做 key，必然抛 `PackageNotFoundError`。两种绕法：
   - 进程内给 `importlib.metadata.version` 打回退补丁（零足迹）
   - 在 site-packages 建 `torch_npu-<ver>.dist-info/METADATA`（持久，但会让 `pip list` 把 stub 报成已安装包）

2. **编译缓存缺失时会现场编译 `npu_utils.cpp`**，而它 `#include <torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h>` —— stub 里没有 C++ 头，必然失败。平时能跑是因为缓存命中。若缓存被清，需要找回预构建的 `npu_utils*.so`，或者预置 `triton.runtime.driver._active` 跳过真 driver 构造（编译本身不需要它）。

生成 IR dump：`TRITON_KERNEL_DUMP=1`，落在 `~/.triton/dump/<hash>/`，每份含 `kernel.ttir.mlir`、`kernel.ttadapter.mlir`、`kernel.npuir.mlir`。**dump 在编译阶段落盘，仿真器跑不跑完无所谓。**

仿真是 CPU-bound，多个 `pem_davinci_standalone` 并发会互相抢核、慢几十倍，必须串行；`timeout` 设 180 秒级别，因为挂死是常见形态。

---

## 11. 一页速查

```
问题        tile = BM × BN × itemsize > 6656 B 时，生成的 kernel 产出错误结果
阈值        [6656, 7168) = [0x1A00, 0x1C00) = [26, 28] × 256 B = [6.5, 7.0) KB
首选 grep   7168 / 0x1C00 / 7*1024 / 28*256 / 14*512   （>= 形式的常量）
机理        某条向量 load 的 UB 地址操作数读到了 X 的数据（已用字节比对证实）
已排除      UB 容量、BM 对齐、dtype、npuir 分配器（无重叠/越界/未对齐）
最可疑      ① 设备属性被 stub 污染  ② 向量寄存器溢出  ③ npuir 以下的 lowering
关键对照    BM=13 (6.5K, 正确)  vs  BM=14 (7.0K, 出错)，BN=128 fp32，n_rows=256
易踩的坑    n_rows 太小会掩盖故障；小用例通过不等于安全
PC 基址     .text 偏移 = PC - 0x4000000
```
