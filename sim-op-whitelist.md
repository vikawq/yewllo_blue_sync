# 仿真器"CPU 真算"算子白名单（交付给仿真团队）

> 背景：仿真器当前只做时延建模，compute kernel 不执行，输出缓冲区内容 = 分配那一刻的残留。
> 实测：`x = torch.arange(10).npu()` 读回正确（H2D/D2H 通道正常）；`y = x*2+1` 读回全 0。
> 目标：让 **控制流相关的小整型算子** 真算，从而使真机/仿真的算子序列与 shape 对齐。
> 模型：Qwen3-VL（`MindSpeed-MM/mindspeed_mm/models/transformers/qwen3vl/modeling_qwen3_vl.py`，真机侧为基准）

---

## 0. 一句话

把 §2 的算子加入 CPU 真算白名单。它们**全部是 int64 / bool、numel 通常 < 1000**，CPU 上是 μs 级，
**不影响性能建模，也不需要给它们精确建时延**。所有 FP 算子（matmul / attention / norm / 激活 / 优化器）
继续只建模，**一个都不要放进白名单**。

---

## 1. 最重要的约束：白名单必须对依赖链**闭包**

这是整件事成败的关键，比"加哪些算子"更重要：

> **链条上缺任何一个算子，它下游的一切都是垃圾。**

例：`rot_pos_emb:186`

```python
total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)
```

`grid_thw`（H2D 来的，正确）→ `prod` → `sum` → `.item()` → `total_tokens` → **决定 `pos_ids` 的 shape**。
只要 `prod` 或 `sum` 有一个没真算，`total_tokens` 就是垃圾，`torch.empty` 直接崩或整条 vision 链 shape 全错。
**加了 `sum` 却漏了 `prod`，效果等于什么都没加。**

所以 §2 按链条给，**请整段加，不要挑着加**。§3 列出五条完整链路供闭包检查。

---

## 2. 白名单

### 2.1 Tier A —— 必须（直接决定 shape 或控制流分支）

| aten op | 对应 aclnn | 出现位置 | 决定了什么 |
| --- | --- | --- | --- |
| `nonzero` / `argwhere` | `aclnnNonzeroV2` ✔ | `:700` `:702`、`_deepstack_process`、bool 索引 | **输出 shape 本身**。当前 702→704 的根因 |
| `sum` / `sum.dim_IntList` | `aclnnReduceSum` ✔ | `:186` `:704` `:705` `:851` `:852` `:934` | `total_tokens`、`image_nums`、`n_image_tokens`、`visual_pos_masks.sum()`（被当 shape 用） |
| `prod.dim_int` | `aclnnProdDim`(待确认) | `:186` `:822` | `total_tokens`、`split_sizes` |
| `floor_divide` / `div.Tensor_mode` | `aclnnFloorDivides` ✔ | `:191` `:740-741` `:822` `:284` | `llm_grid_h/w`、`split_sizes`、view 的 shape |
| `max` / `max.dim` | `aclnnMax` ✔ | `:182` | `max_hw` → `rotary_pos_emb(max_hw)` 的 shape |
| `eq.Scalar` / `eq.Tensor` | `aclnnEqScalar` ✔ | `:700` `:702` `:704` `:705` `:847` `:848` | 所有 token 掩码；`cache_position[0]==0` |
| `index.Tensor`（含 bool 索引） | `aclnnIndex` / `aclnnNonzeroV2` | `:700` `:703` `:932` `:933` | 筛选后的序列长度 |
| `index_put_` / `_index_put_impl_` | `aclnnIndexPutImpl` ✔ | `:764` `:936` `:937` | 写回位置 |
| `masked_select` | `aclnnMaskedSelect`(待确认) | bool 索引的等价路径 | 输出 shape |
| `cumsum` | `aclnnCumsum`(待确认) | `:336` `:773` | `cu_seqlens` |
| `repeat_interleave` | `aclnnRepeatInterleave*`(待确认) | `:328` `:336` `:672` `:990` | `sequence_lengths` |
| `arange` | `aclnnArange`(待确认) | `:212-215` `:986` | 位置索引的值 |

✔ = 已在你们真机 `kernel_details.csv` / 报告中确认出现过的名字。（待确认）的请按 §4 从自己的数据里取准确名字。

### 2.2 Tier B —— 链条上游（喂给 Tier A，不加则 Tier A 拿到垃圾）

| aten op | 对应 aclnn | 说明 |
| --- | --- | --- |
| `mul.Tensor` / `mul.Scalar` | `aclnnMul` / `aclnnMuls` | `:328` `grid_thw[:,1]*grid_thw[:,2]`、`:269` `h*w` |
| `add.Tensor` / `add.Scalar` | `aclnnAdd` / `aclnnAdds` | `:703` `vision_start_indices + 1` |
| `sub` | `aclnnSub` / `aclnnSubs` | 位置差 |
| `gt.Scalar` / `lt` / `ne` | `aclnnGtScalar` 等 | `:207` `num_frames > 1`、`:711` `remain_images > 0` |
| `bitwise_or` / `logical_or` | `aclnnBitwiseOrTensor` / `aclnnLogicalOr` | `:931` `image_mask \| video_mask` |
| `cat` | `aclnnCat`(待确认) | `:291-292` `:762` |
| `stack` | 通常分解为 unsqueeze+cat | `:216` `:757` |
| `constant_pad_nd` | `aclnnConstantPadNd`(待确认) | `:344` `F.pad(cu_seqlens,(1,0))` |
| `repeat` | `aclnnRepeat`(待确认) | `:209` `:274-275` |
| `copy_` / `_to_copy` / `to.dtype`（int/bool） | `aclnnInplaceCopy` ✔ / `aclnnCast` ✔ | `:216` `pos_ids[a:b] = coords`；int 之间的 dtype 转换 |
| `fill_` / `zero_` / `new_zeros`（int/bool） | `aclnnInplaceFillScalar` / `aclnnInplaceZero` | `:934` `new_zeros(...)` |

**纯 view 算子不下发 kernel，不用进白名单**：`view` / `reshape` / `select` / `slice` / `narrow` / `expand` /
`squeeze` / `unsqueeze` / `permute` / `flatten` / `as_strided`。（除非需要 contiguous 化，那会转成 `aclnnInplaceCopy`，已在表内。）

### 2.3 明确**不要**进白名单

所有 dtype 为 fp16 / bf16 / fp32 的算子：`matmul` / `bmm` / attention（FA）/ `layer_norm` / `rms_norm` /
`softmax` / `gelu` / `silu` / `conv` / 优化器（`ApplyAdamW`）/ `masked_scatter`（FP）/ `embedding`。

它们的输出 shape 不依赖数值，对控制流没有影响；放进去只会拖慢仿真且毫无收益。

**建议的过滤规则**：`dtype ∈ {bool, int8, int16, int32, int64, uint8}` **且** `numel ≤ 4M`。
（4M 这个上限是为了兜住 `:852` 那个 `expand` 出来的 `(1,704,4096)` bool 掩码 = 2.88M。）

---

## 3. 五条依赖链（用来做闭包检查）

按顺序修，每修完一条就能消掉报告里一整片差异。

### 链 1：`total_tokens` → vision 位置编码的 shape

```
grid_thw (H2D ✅) → prod(dim=1) → sum() → .item() → total_tokens
                  → torch.empty((total_tokens, 2))        # rot_pos_emb:186-187
grid_thw → [:,1:] → max() → .item() → max_hw → rotary_pos_emb(max_hw)   # :182
```
需要：`prod` `sum` `max`
消掉：`rot_pos_emb` 的 `aten::fill_` ×6（仿真独有），`aten::copy_` shape 从 `1,3` 恢复成 `16,16,2,2`

### 链 2：`split_sizes` → `torch.split` 的切分点

```
image_grid_thw (H2D ✅) → prod(-1) → floor_divide(merge_size²) → .tolist()
                        → torch.split(image_embeds, split_sizes)   # get_image_features:822-823
```
需要：`prod` `floor_divide`
消掉：`get_image_features` 的 `aten::to`/`copy_` shape 从 `1,3` 恢复成 `1`

### 链 3：有效序列长度 702 —— **当前 702 vs 704 的根因**

```
input_ids, attention_mask (H2D ✅)
  → attention_mask[i] == 1            (eq)
  → input_ids[mask]                   (index/bool → nonzero)  ← 输出长度在这里定
  → input_ids == vision_start_token_id (eq) → argwhere (nonzero) → squeeze
  → input_ids[indices + 1]            (add, index)
  → (vision_tokens == image_token_id).sum()  (eq, sum) → image_nums
  → input_ids.tolist()                (D2H ✅)
  → t.item(), h.item()//merge, w.item()//merge  (select, floor_divide)
```
需要：`eq` `nonzero` `index` `add` `sum` `floor_divide`
消掉：`aten::argwhere` 704 → 702；`aten::reshape` `0` → `3,702`；`aten::to` `0` → `702`；
`__floordiv__` 那 5 类多出来的算子（`empty`/`empty_strided`/`to`/`_to_copy`/`copy_`，共 +62）

### 链 4：deepstack —— 掩码计数被当 shape 用

```
input_ids == image_token_id (eq) → special_image_mask
  → .sum().item()  vs  image_features.numel()      # get_placeholder_mask:852 的校验
  → image_mask | video_mask            (bitwise_or)
  → image_mask[visual_pos_masks]       (bool index → nonzero)
  → img_embed.new_zeros(visual_pos_masks.sum(), ...)   # :934 ← sum() 直接当 shape
  → embed_joint[image_mask_joint,:] = img_embed        # index_put_
```
需要：`eq` `sum` `bitwise_or` `index` `nonzero` `index_put_` `new_zeros`
消掉：`get_placeholder_mask_mock` 可以摘掉；恢复 `aclnnReduceSum` ×6；
`_deepstack_process` 的 `aclnnNonzeroV2` ×12 / `select` ×24 / `reshape` ×12 回来

### 链 5：prefill 判定 —— 决定 `get_rope_index` 调 18 次还是 10 次

```
cache_position[0] == 0   (eq + item)          # forward:952
past_key_values.get_seq_length() == 0
  → if (prefill_...) or self.rope_deltas is None:      # forward:970
```
需要：`eq`（0-dim）+ `_local_scalar_dense`（本来就正常）
消掉：`get_rope_index` 调用次数 10 → 18

---

## 4. 用你们自己的数据生成准确的 aclnn 名字

上表里标"待确认"的，不要照抄，从真机 profiling 里取：

1. 从 `pytrace.md` §5 拿到这些 Python 函数直接下发的 aten 算子全集：
   `rot_pos_emb` / `fast_pos_embed_interpolate` / `get_rope_index` / `get_image_features` /
   `get_placeholder_mask` / `_deepstack_process` / vision `forward`
2. 在真机 `ASCEND_PROFILER_OUTPUT/kernel_details.csv` 里用 **Correlation Id** 把这些 aten 算子回溯到实际下发的 aclnn kernel 名
3. 按 `Input Data Type` 过滤，只留 `INT64` / `INT32` / `BOOL` / `UINT8`
4. 得到的集合就是准确白名单，可以直接喂给仿真器

这一步也顺带保证了**闭包性** —— 只要这几个函数下发的整型算子全在表里，链条不会断。

---

## 5. 给仿真团队的三个附加确认

这三条直接影响 profiling 对齐的效果，请一并确认：

| # | 问题 | 期望 | 为什么重要 |
| --- | --- | --- | --- |
| 1 | 真算会不会在 profiling 里**新增事件**？（额外的 memcpy、额外的 CANN API 调用） | 不新增。CPU 计算发生在 aclnn 实现内部，torch 侧仍然只看到一次 `Enqueue@aclnnXxx` + 一次 device kernel | 若新增，等于用一类差异换另一类差异 —— 这正是我们在 Python 层踩过的坑 |
| 2 | 真算的**CPU 耗时会不会计入时延模型**？ | 不计入，时延仍由代价模型给 | 否则性能仿真结论被污染（我们还要用同一份数据做性能对齐） |
| 3 | 白名单的**粒度**是什么？算子名？算子名+dtype？还是算子名+shape 上限？ | 最好支持 `算子名 + dtype` 两级，能把 FP 版本排除在外 | 例如 `aclnnReduceSum` 既用于 int 掩码计数，也用于 FP 归约；只想真算前者 |

关于**数值一致性**：白名单里全是整型/bool 算子，**CPU 与 NPU 位级完全一致**，不存在精度风险。
这也是为什么必须把 FP 算子排除在外 —— 一旦引入 FP，CPU 与 NPU 的舍入差异会带来新的不确定性。

---

## 6. 验收

### 6.1 单算子级

```python
import torch, torch_npu
x = torch.arange(10).npu()
y = x * 2 + 1
assert y.cpu().tolist() == [1,3,5,7,9,11,13,15,17,19]      # mul/add 真算

m = torch.tensor([1]*8 + [0]*2).npu().bool()
assert torch.nonzero(m).shape[0] == 8                       # nonzero 真算
assert x[m].shape[0] == 8                                   # bool 索引真算
assert int(x.sum().item()) == 45                            # sum 真算
```

### 6.2 模型级（撤掉绕过代码之后）

先把这些全部撤掉：

- `simulator_patch` L176 的 `Qwen3VLModel.get_placeholder_mask = get_placeholder_mask_mock`
- 视觉侧被注释掉的那一小段
- 所有为规避 `.item()` 而加的 `.cpu()` / CPU fallback（`grid_thw` 等搬回 device）

然后在这几处打印，与真机对比：

| 位置 | 打印 | 真机值 |
| --- | --- | --- |
| `rot_pos_emb:186` | `total_tokens` | 与真机一致 |
| `get_image_features:822` | `split_sizes` | 与真机一致 |
| `get_rope_index:706` | `len(input_tokens)` | **702** |
| `get_rope_index:704` | `image_nums` | 与真机一致 |
| `forward:934` | `visual_pos_masks.sum()` | 与真机一致 |

### 6.3 Profiling 级（逐级验收，哪级不过停在哪级）

| 级别 | 检查项 |
| --- | --- |
| L1 | `pytrace.md` §2「只在仿真执行到的函数」为空（mock 已摘） |
| L2 | `__floordiv__` 组：`aclnnFloorDivides` real == sim，且 `aten::empty`/`empty_strided`/`to`/`_to_copy`/`copy_` 的 +62 消失 |
| L3 | §5b 里 `get_rope_index` / `rot_pos_emb` / `get_image_features` 的 shape 差异清零 |
| L4 | `aten::argwhere` real == sim（702 == 702） |
| L5 | `get_rope_index` 调用次数 real == sim（18 == 18） |
| L6 | 剩余差异只在 Communication track 和 `_get_global_min_max_time` |
| L7 | **回归检查**：真算白名单没有给 CANN track 引入新的事件类型（对比开启前后的仿真 trace） |

---

## 7. 白名单解不掉的部分（不要期望它们消失）

| 差异 | 规模 | 建议 |
| --- | --- | --- |
| 通信原语：真机 `Notify_Wait` 8317 / `NOTIFY_WAIT_SQE` 8052 / `UBDMA` 6894 / `Ub_Inline_Write` 4578（仿真全 0）；仿真 `Communication@Memcpy` 303 / `Launch_Ffts` 303 | Communication track 对齐率 3% | 仿真器用 memcpy + event 模拟集合通信，基础设施级差异。对比工具里把 Communication track 单独归一化，不计入对齐率 |
| `_get_global_min_max_time`：仿真独有 `aclnnMin` ×24 + `aclnnMax` ×24 + `aten::item` ×48 + `_local_scalar_dense` ×48 | ~144 事件/step | 仿真器自己的通信时延诊断逻辑。加环境变量开关关掉，或打 `record_function` 标签后在对比时剔除 |
