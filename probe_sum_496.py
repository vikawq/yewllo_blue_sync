#!/usr/bin/env python3
"""
把 `:496` 那一行拆成四个自变量逐个扫，定位 sum 到底卡在哪个参数上。

    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)

之前的探针只测过 `t.sum(0)`（正 dim、无 dtype、int64 输入、输出 2 个元素），
而 :496 这一行同时踩了四个没测过的点：

    1. dim = -1        负数维度
    2. dtype = int32   输出 dtype 与输入不同（白名单若按「算子名 + dtype」两级过滤，
                       int32 输出可能被单独挡掉）
    3. attention_mask 的实际 dtype 未知（int64 / int32 / bool 都有可能）
    4. bs = 1          输出只有 1 个元素，与全归约同值 —— 极易蒙对

四个自变量交叉扫一遍，第一个 FAIL 的组合就是断点。

防回收：每个组合用两组不同的掩码各跑一次（和为 427 / 446），两组都对才判 OK。
两个数都远离 0，也远离 151644 那种 token id 残留。

用法：  python probe_sum_496.py
"""
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

FAILS = []


def build(n_total, n_ones, dt):
    h = torch.zeros(1, n_total, dtype=dt)
    h[0, :n_ones] = 1
    return h.npu()


def chk(tag, fn, want_a, want_b):
    """A: [1,432] 里 427 个 1   B: [1,448] 里 446 个 1"""
    try:
        ga = fn(A)
        gb = fn(B)
        ga = ga.cpu().tolist() if torch.is_tensor(ga) else ga
        gb = gb.cpu().tolist() if torch.is_tensor(gb) else gb
    except Exception as e:
        print(f"EXC  | {tag:<46} {type(e).__name__}: {e}")
        FAILS.append(tag)
        return
    ok = (ga == want_a) and (gb == want_b)
    print(("OK   | " if ok else "FAIL | ") + f"{tag:<46} A={ga}  B={gb}")
    if not ok:
        print(f"     | {'':<46} 期望 A={want_a}  B={want_b}")
        FAILS.append(tag)


def head(t):
    print()
    print("=" * 104)
    print(t)
    print("=" * 104)


print("A = [1,432] 前 427 个是 1  → 各种 sum 都应得 427")
print("B = [1,448] 前 446 个是 1  → 应得 446")
print("（427/446 既不为 0，也不可能是回收缓冲区里的 token id）")

for dt in (torch.int64, torch.int32, torch.bool):
    name = str(dt).split(".")[-1]
    A = build(432, 427, dt)
    B = build(448, 446, dt)
    head(f"输入 dtype = {name}")

    chk("1  sum()                   全归约基线", lambda t: int(t.sum().item()), 427, 446)
    chk("2  sum(dtype=int32)        全归约+dtype",
        lambda t: int(t.sum(dtype=torch.int32).item()), 427, 446)
    chk("3  sum(dim=1)              正 dim，无 dtype", lambda t: t.sum(dim=1), [427], [446])
    chk("4  sum(dim=-1)             负 dim，无 dtype", lambda t: t.sum(dim=-1), [427], [446])
    chk("5  sum(dim=1, dtype=int32) 正 dim+dtype",
        lambda t: t.sum(dim=1, dtype=torch.int32), [427], [446])
    chk("6  sum(dim=-1,dtype=int32) ← :496 原样",
        lambda t: t.sum(dim=-1, dtype=torch.int32), [427], [446])

head("bs=3：输出 3 个元素，排除「1 个元素与全归约同值而蒙对」")
A = torch.tensor([[1] * 3 + [0], [1] * 2 + [0] * 2, [1] * 4], dtype=torch.int64).npu()
B = torch.tensor([[1] * 4, [1] * 1 + [0] * 3, [1] * 2 + [0] * 2], dtype=torch.int64).npu()
print("     A 行和 = [3,2,4]（全和 9）   B 行和 = [4,1,2]（全和 7）")
chk("7  sum(dim=-1, dtype=int32)  bs=3",
    lambda t: t.sum(dim=-1, dtype=torch.int32), [3, 2, 4], [4, 1, 2])

head(":496 完整链路（sum → cumsum → F.pad → 切片）")
A = build(432, 427, torch.int64)
B = build(448, 446, torch.int64)


def chain(t):
    s = t.sum(dim=-1, dtype=torch.int32)
    c = F.pad(torch.cumsum(s, dim=0, dtype=torch.int32), (1, 0))
    return c[1:] if len(c) > 1 else c


chk("8  完整 cu_seqlens", lambda t: chain(t), [427], [446])
print("     ↑ cu_seqlens 的最后一个值必须 ≤ total_seq_len，否则 FlashAttention 的 tiling 直接拒绝")

head("读法")
if not FAILS:
    print("全绿 —— :496 已经好了，可以撤掉临时旁路。")
else:
    print(f"{len(FAILS)} 项 FAIL：")
    for f in FAILS:
        print(f"  · {f}")
    print("""
  1/2 挂        → 全归约本身就坏了（和之前结论矛盾，先确认环境）
  1/2 过，3 起挂 → 就是「带 dim 的 overload 没被调度」，与 dtype、正负 dim 无关
  3 过 4 挂      → 负数 dim 没做归一化（应先转成 dim + ndim）
  3/4 过 5/6 挂  → dtype 参数惹的祸：白名单按「算子名 + dtype」过滤时把 int32 输出挡掉了
  只有 bool 那轮挂 → 白名单的 dtype 过滤漏了 bool 输入

请把结果连同下面这条一起发给 op server 团队 —— 它决定改客户端白名单还是服务端 runner：

    grep "recv op:" server.log | sed 's/.*recv op: //' | sort | uniq -c | sort -rn

  列表里有 sum 相关名字 → 在客户端白名单里，服务端缺 runner
  列表里没有            → 请求根本没发出去，要先加客户端白名单
""")

print()
print("另外请在 :496 打一行，确认 attention_mask 的真实 dtype 与本脚本哪一轮对应：")
print('    print(attention_mask.dtype, attention_mask.shape)')
