#!/usr/bin/env python3
"""
仿真器真算能力状态板 v2。一次跑完，直接对着最后的「剩余待办」清单派活。

相对 v1 的修改：

1. **双输入防回收**（最重要）。v1 有两条因缓冲区回收而误判：
     · `all(0)` 在 base=[[T,T],[T,F]] 下与 `all(-1)` 的正确答案都是 [True,False]，
       复用上一条的缓冲区也能"通过"；
     · `all(-1)` 在两轮里分别给出 [False,False] 和 [True,False]，同一个测试结论相反。
   现在每个算子都用**两组不同的输入**跑，两组都对才判 OK ——
   回收来的缓冲区不可能同时满足两组不同的期望值。

2. 期望值全部避开「退化成全归约」会得到的那个数，也避开 0。

3. 补齐 v1 漏测的：`min()` 全归约、`amin`、`argmin`、`aminmax`、`any()` 全归约、
   `sum` 的空 dim / 非空 dim 分支。

4. 加入模型关键点：`:496`（TND flash attention 的 cu_seqlens，bs≥2 才暴露）、`:776`。

5. 加入 `index_put` 的 `None` 索引回归（§2.7）。

用法：  python probe_dim_reduce.py
"""
import torch
import torch_npu  # noqa: F401

TODO = {}


def _tl(x):
    return x.cpu().tolist() if torch.is_tensor(x) else x


def chk2(tag, fn, want_a, want_b, todo=None):
    """同一个算子用两组不同输入各跑一次，两组都对才算 OK。"""
    A = torch.tensor([[2, 7], [5, 3]], dtype=torch.int64).npu()
    B = torch.tensor([[11, 4], [6, 20]], dtype=torch.int64).npu()
    try:
        got_a, got_b = _tl(fn(A)), _tl(fn(B))
    except Exception as e:
        print(f"EXC  | {tag:<34} {type(e).__name__}: {e}")
        if todo:
            TODO[tag] = todo
        return
    ok = (got_a == want_a) and (got_b == want_b)
    print(("OK   | " if ok else "FAIL | ") + f"{tag:<34} A={got_a}  B={got_b}")
    if not ok:
        print(f"     | {'':<34} 期望 A={want_a}  B={want_b}")
        if todo:
            TODO[tag] = todo


def chk(tag, got, want, todo=None):
    got = _tl(got)
    ok = got == want
    print(("OK   | " if ok else "FAIL | ") + f"{tag:<34} got={got}")
    if not ok:
        print(f"     | {'':<34} want={want}")
        if todo:
            TODO[tag] = todo


def head(t):
    print()
    print("=" * 100)
    print(t)
    print("=" * 100)


print("A = [[2,7],[5,3]]   全和17 全积210 全max7 全min2")
print("B = [[11,4],[6,20]] 全和41 全积5280 全max20 全min4")
print("（两组输入的期望值互不相同 —— 回收缓冲区无法同时骗过两组）")

head("§1  sum —— aclnnReduceSum，全/dim 共用同一个 kernel 名")
chk2("sum()      全归约", lambda t: int(t.sum().item()), 17, 41)
chk2("sum(0)     dim 归约", lambda t: t.sum(0), [7, 10], [17, 24],
     "改 reduce_sum runner：len(dim)==0 走全归约，否则 torch.sum(self,dim=dim,keepdim=keepdim,dtype=dtype)")
chk2("sum(1)     dim 归约", lambda t: t.sum(1), [9, 8], [15, 26],
     "同上")
chk2("sum(1, keepdim=True)", lambda t: t.sum(1, keepdim=True), [[9], [8]], [[15], [26]],
     "同上，注意 keepdim 也要透传")
chk2("sum(dim=[0,1]) 多维 dim", lambda t: int(t.sum(dim=[0, 1]).item()), 17, 41,
     "reduce_sum 的 dim 是 int[]，要支持多元素")

head("§2  prod —— 对照组，已知正常（aclnnProd / aclnnProdDim 两个独立名字）")
chk2("prod()     全归约", lambda t: int(t.prod().item()), 210, 5280)
chk2("prod(0)    dim 归约", lambda t: t.prod(0), [10, 21], [66, 80])
chk2("prod(1)    dim 归约", lambda t: t.prod(1), [14, 15], [44, 120])

head("§3  max / min —— aclnnMax(全) vs aclnnMaxDim(dim)，两个不同名字")
chk2("max()      全归约", lambda t: int(t.max().item()), 7, 20)
chk2("min()      全归约", lambda t: int(t.min().item()), 2, 4,
     "若 FAIL：aclnnMin 也没进白名单（torch.min 同名，加名字即可）")
chk2("max(1).values", lambda t: t.max(1).values, [7, 5], [11, 20],
     "sp_op/max_dim_runner：values,indices = torch.max(self, dim, keepdim)，双输出分别回写")
chk2("max(1).indices", lambda t: t.max(1).indices, [1, 0], [0, 1],
     "同上，indices dtype 必须是 int64")
chk2("max(0).values", lambda t: t.max(0).values, [5, 7], [11, 20],
     "同上")
chk2("min(1).values", lambda t: t.min(1).values, [2, 3], [4, 6],
     "sp_op/min_dim_runner：values,indices = torch.min(self, dim, keepdim)")
chk2("min(1).indices", lambda t: t.min(1).indices, [0, 1], [1, 0],
     "同上")
chk2("amax(1)", lambda t: t.amax(1), [7, 5], [11, 20],
     "aclnnAmax → torch.amax 同名，加白名单即可")
chk2("amin(1)", lambda t: t.amin(1), [2, 3], [4, 6],
     "aclnnAmin → torch.amin 同名，加白名单即可")
chk2("argmax(1)", lambda t: t.argmax(1), [1, 0], [0, 1],
     "sp_op：torch.argmax（不是 torch.arg_max）")
chk2("argmin(1)", lambda t: t.argmin(1), [0, 1], [1, 0],
     "sp_op：torch.argmin")
chk2("aminmax(dim=1).min", lambda t: torch.aminmax(t, dim=1).min, [2, 3], [4, 6],
     "sp_op/aminmax_runner：双输出 (min,max)")
chk2("aminmax(dim=1).max", lambda t: torch.aminmax(t, dim=1).max, [7, 5], [11, 20],
     "同上")

head("§4  bool 归约 —— aclnnAll / aclnnAny（全与 dim 共用一个名字）")
print("     bA = [[T,T],[F,T]]   bB = [[T,F],[T,T]]   —— 两组的按行/按列答案互不相同")
bA = torch.tensor([[True, True], [False, True]]).npu()
bB = torch.tensor([[True, False], [True, True]]).npu()
chk("all()   全归约  A", bA.all().item(), False)
chk("any()   全归约  A", bA.any().item(), True,
    "aclnnAny 整体缺失（全归约也错）→ torch.any 同名，加白名单即可")
chk("any()   全归约  B", bB.any().item(), True, "同上")
chk("all(-1) 按行  A", bA.all(-1), [True, False])
chk("all(0)  按列  A", bA.all(0), [False, True], "与按行答案不同，才有区分度")
chk("all(-1) 按行  B", bB.all(-1), [False, True])
chk("all(0)  按列  B", bB.all(0), [True, False], "与 A 的按列答案相反，交叉验证")
chk("any(-1) 按行  A", bA.any(-1), [True, True], "aclnnAny")
chk("any(0)  按列  A", bA.any(0), [True, True], "aclnnAny")

head("§5  cumsum —— 已知正常（aclnnCumsum → torch.cumsum 同名）")
chk2("cumsum(1)", lambda t: t.cumsum(1), [[2, 9], [5, 8]], [[11, 15], [6, 26]])
chk2("cumsum(0)", lambda t: t.cumsum(0), [[2, 7], [7, 10]], [[11, 4], [17, 24]])

head("§6  模型关键点")

am = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]]).npu()
chk(":496  attention_mask.sum(dim=-1, int32)", am.sum(dim=-1, dtype=torch.int32), [3, 2, 4],
    "→ cu_seqlens → TND flash attention 的 actual_seq_len。全归约会得 9。bs=1 时蒙对，这里 bs=3")
chk(":496  再 cumsum 一步", torch.cumsum(am.sum(dim=-1, dtype=torch.int32), dim=0, dtype=torch.int32),
    [3, 5, 9], "cu_seqlens 的完整链路")

pid = torch.tensor([[[1, 5], [2, 3]], [[4, 0], [9, 7]], [[6, 2], [8, 1]]], dtype=torch.int64).npu()
chk(":776  max(0)[0].max(-1,keepdim)[0]",
    pid.max(0, keepdim=False)[0].max(-1, keepdim=True)[0], [[6], [9]],
    "aclnnMaxDim，需要 sp_op/max_dim_runner")

g = torch.tensor([[1, 28, 36], [1, 14, 18]], dtype=torch.int64).npu()
chk(":186  prod(grid_thw, dim=1)", torch.prod(g, dim=1), [1008, 252])
chk(":822  prod(-1) // 4", g.prod(-1) // 4, [252, 63])

head("§7  index_put 的 None 索引回归（§2.7，op_decoder 协议问题）")
t = torch.zeros(3, 8, dtype=torch.int64).npu()
t[:, torch.tensor([True] * 5 + [False] * 3).npu()] = torch.arange(1, 16).view(3, 5).npu()
chk("t[:, mask] = v   （indices=[None, mask]）", t[:, :5],
    [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
    "op_decoder 要能表达 None（0 维/None/空张量三者要能区分），不能用空张量代替 None")
t2 = torch.zeros(3, 1, 8, dtype=torch.int64).npu()
t2[..., 0, torch.tensor([True] * 5 + [False] * 3).npu()] = torch.arange(1, 16).view(3, 5).npu()
chk(":762  t[..., i, mask] = v", t2[:, 0, :5],
    [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]], "同上")
t3 = torch.zeros(5, 4, dtype=torch.int64).npu()
t3[torch.tensor([True, False, True, False, True]).npu(), :] = torch.arange(1, 13).view(3, 4).npu()
chk(":935  t[mask, :] = v （尾部 None 被截掉，应正常）", t3,
    [[1, 2, 3, 4], [0, 0, 0, 0], [5, 6, 7, 8], [0, 0, 0, 0], [9, 10, 11, 12]])

head("剩余待办（按 op server 侧的动作归类）")
if not TODO:
    print("全绿。")
else:
    print(f"{len(TODO)} 项：\n")
    for tag, action in TODO.items():
        print(f"  · {tag}")
        print(f"      → {action}")
    print("""
提醒 op server 团队三件事：

  1. 通用 runner `getattr(torch, op_name)(*args)` 只在 kernel 名恰好等于 torch 函数名时成立。
     max_dim / min_dim / arg_max / arg_min / reduce_sum / prod_dim 都不满足，必须走 sp_op/。
  2. 空 dim 列表 = 全归约。aclnnReduceSum / aclnnAll / aclnnAny 的全归约版本传的就是空 dim，
     runner 要显式分支，不能把空列表透传给 torch（PyTorch 对 dim=[] 的语义改过）。
  3. 加个启动自检，把「名字没映射」从运行时提前到启动时：
       unmapped = [n for n in WHITELIST if n not in SP_OP_REGISTRY and not hasattr(torch, n)]
""")
