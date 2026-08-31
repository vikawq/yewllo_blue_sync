#!/usr/bin/env python3
"""
第三轮：dim 归约族。

上一轮 probe_4k_boundary.py 的结论：

  Q1-Q4 全绿 —— 4 KiB 天花板已随阈值改成 4MB 而消失；
                stride / storageOffset / stride==0 expand / aclop BroadcastTo 全部正常；
                §2.5 里 `:852` 那个 252 应该已经不复现了。
  Q5 命中   —— sum 带 dim 参数时结果错，且错得很有规律：

        m = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]   全和 = 78
        m.sum()      → 78              ✅
        m.sum(0)     → [78, 1, 1, 1]   ❌ 期望 [15,18,21,24]
        m.sum(1)     → [78, 1, 1]      ❌ 期望 [10,26,42]
        m.t().sum(0) → [78, 1, 1]      ❌ 期望 [10,26,42]

    输出 shape 是对的，第 0 个元素恒等于**全归约**的结果，其余是残留。
    ⇒ `dim` 参数没传到 op server（或 runner 忽略了它），退化成了全归约，
      结果写进输出缓冲区的第 0 个槽位。

  但 Q5 里 prod / max 那两条我写错了，测不出问题，本脚本重做：
    · `prod(dim=1)` 用的是单行 grid_thw `[[1,28,36]]`，输出只有 1 个元素，
      而且行内乘积恰好等于全乘积 —— 退化与否结果一样，测了等于没测。
    · `max(dim=1)` 我写成了 `g[:,1:].max().item()`，那是**全** max，根本没测 max.dim。

本脚本的设计约束：**每个 dim 归约的输出至少 2 个元素，且全归约的答案与每个输出元素都不同**，
这样"退化成全归约"必定被抓到。

用法：  python probe_dim_reduce.py
"""
import torch
import torch_npu  # noqa: F401

FAILS = []


def show(tag, got, want, note=""):
    ok = got == want
    if not ok:
        FAILS.append(tag)
    print(("OK   | " if ok else "FAIL | ") + f"{tag:<40} got={got}")
    if not ok:
        print(f"     | {'':<40} want={want}" + (f"   [{note}]" if note else ""))


def head(t):
    print()
    print("=" * 96)
    print(t)
    print("=" * 96)


# 基张量：全归约的答案与任何一个行/列归约结果都不撞车
b = torch.tensor([[2, 7], [5, 3]], dtype=torch.int64).npu()
print("base = [[2,7],[5,3]]   全和=17  全积=210  全max=7  全min=2")

head("§1  sum —— 已知坏，确认退化形态")
show("sum()            全归约", int(b.sum().item()), 17)
show("sum(0)", b.sum(0).cpu().tolist(), [7, 10], "退化会得 [17, 残留]")
show("sum(1)", b.sum(1).cpu().tolist(), [9, 8], "退化会得 [17, 残留]")
show("sum(0, keepdim=True)", b.sum(0, keepdim=True).cpu().tolist(), [[7, 10]])
show("sum(dim=[0,1])  多维 dim", int(b.sum(dim=[0, 1]).item()), 17)

head("§2  prod —— 上一轮测法无效，重测（链 1 `total_tokens` / 链 2 `split_sizes` 依赖它）")
show("prod()           全归约", int(b.prod().item()), 210)
show("prod(0)", b.prod(0).cpu().tolist(), [10, 21], "退化会得 [210, 残留]")
show("prod(1)", b.prod(1).cpu().tolist(), [14, 15], "退化会得 [210, 残留]")

head("§3  max / min / argmax —— 上一轮根本没测到 dim 版本")
show("max()            全归约", int(b.max().item()), 7)
show("max(1).values", b.max(1).values.cpu().tolist(), [7, 5])
show("max(1).indices", b.max(1).indices.cpu().tolist(), [1, 0])
show("max(0).values", b.max(0).values.cpu().tolist(), [5, 7])
show("amax(1)", b.amax(1).cpu().tolist(), [7, 5])
show("min(1).values", b.min(1).values.cpu().tolist(), [2, 3])
show("argmax(1)", b.argmax(1).cpu().tolist(), [1, 0])

head("§4  其余同族（bool / cumsum）")
bb = torch.tensor([[True, True], [True, False]]).npu()
show("all(-1)", bb.all(-1).cpu().tolist(), [True, False])
show("any(-1)", bb.any(-1).cpu().tolist(), [True, True])
show("all(0)", bb.all(0).cpu().tolist(), [True, False])
show("cumsum(1)", b.cumsum(1).cpu().tolist(), [[2, 9], [5, 8]])
show("cumsum(0)", b.cumsum(0).cpu().tolist(), [[2, 7], [7, 10]])

head("§5  模型原型：多图场景（单图会侥幸蒙对，多图才暴露）")
g = torch.tensor([[1, 28, 36], [1, 14, 18]], dtype=torch.int64).npu()
print("     grid_thw = [[1,28,36],[1,14,18]]   行积 = [1008, 252]   全积 = 254016")
show(":186  prod(grid_thw, dim=1)", torch.prod(g, dim=1).cpu().tolist(), [1008, 252],
     "退化会得 [254016, 残留] → total_tokens 直接崩")
show(":186  prod(dim=1).sum()", int(torch.prod(g, dim=1).sum().item()), 1260)
show(":822  prod(-1) // 4", (g.prod(-1) // 4).cpu().tolist(), [252, 63],
     "split_sizes，喂给 torch.split")
show(":182  grid_thw[:,1:].max()", int(g[:, 1:].max().item()), 36, "全 max，本来就正常")
p = torch.arange(6).view(3, 1, 2).npu()
show(":776  position_ids.max(0)[0].max(-1,keepdim=True)[0]",
     p.max(0, keepdim=False)[0].max(-1, keepdim=True)[0].cpu().tolist(), [[5]])

head("小结")
if FAILS:
    print(f"共 {len(FAILS)} 项 FAIL：")
    for f in FAILS:
        print(f"  · {f}")
    print("""
若 FAIL 全部集中在「带 dim 参数」的那些行，而全归约版本全绿
  ⇒ 根因是 dim 参数在 gRPC 传输/解码时丢了，runner 退化成全归约。
     这跟 index_put 的 None 丢失是同一类问题：**非张量参数没被正确传递**。
     建议 op server 一次性排查所有 runner 的标量/列表/可选参数
     （dim、keepdim、accumulate、indices 里的 None ……），不要逐个算子撞。""")
else:
    print("全绿。")
