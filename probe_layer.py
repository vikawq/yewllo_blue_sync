#!/usr/bin/env python3
"""
用「耗时」判断每个失败的算子缺在哪一层，一次性分类所有待办。

原理（已用 sum 验证）：
  · 进了客户端白名单 → 走一次完整 gRPC 往返 → 每次约 4.2~4.6 ms
  · 没进客户端白名单 → 请求根本不出网 → 每次约 0.6 ms

  实测：sum()=4616us  prod(dim=1)=4163us  sum(dtype)=4331us  ←  走服务端
        sum(dim=1)=626us                                     ←  没出网

  ⇒ 缺在客户端白名单的，服务端加 runner 是白费；反之亦然。先分类再派活。

每行同时给出「结果对不对」和「在哪一层」，两者组合才是完整结论：

  ✅正确 + 走服务端  → 已经好了
  ❌错误 + 走服务端  → 在白名单里，**服务端 runner** 没处理好参数
  ❌错误 + 没出网    → **客户端白名单**缺这个 overload
  ✅正确 + 没出网    → 可疑！很可能是回收缓冲区蒙对的假阳性

用法：  python probe_layer.py
"""
import time
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

N = 30
ROWS = []


def bench(fn):
    fn()
    torch.npu.synchronize()
    s = time.perf_counter()
    for _ in range(N):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - s) / N * 1e6


def case(tag, fn, want, baseline=False):
    try:
        got = fn()
        got = got.cpu().tolist() if torch.is_tensor(got) else got
        us = bench(fn)
    except Exception as e:
        ROWS.append((tag, f"EXC {type(e).__name__}", None, None, baseline))
        return
    ROWS.append((tag, got, want, us, baseline))


A = torch.zeros(1, 432, dtype=torch.int64)
A[0, :427] = 1
A = A.npu()
M = torch.tensor([[2, 7], [5, 3]], dtype=torch.int64).npu()
Bm = torch.tensor([[True, True], [False, True]]).npu()

# ---- 基准：已知正常、确定走服务端 ----
case("sum()                 【基准】", lambda: int(A.sum().item()), 427, baseline=True)
case("prod(1)               【基准】", lambda: M.prod(1), [14, 15], baseline=True)
case("all(-1)               【基准】", lambda: Bm.all(-1), [True, False], baseline=True)
case("cumsum(0,dtype=int32) 【基准】",
     lambda: torch.cumsum(torch.tensor([3, 2, 4], dtype=torch.int32).npu(), dim=0, dtype=torch.int32),
     [3, 5, 9], baseline=True)

# ---- 待分类 ----
case("sum(dim=1)", lambda: A.sum(dim=1), [427])
case("sum(dtype=int32)", lambda: int(A.sum(dtype=torch.int32).item()), 427)
case("sum(dim=1,dtype=int32)", lambda: A.sum(dim=1, dtype=torch.int32), [427])
case("max(1).values", lambda: M.max(1).values, [7, 5])
case("max(1).indices", lambda: M.max(1).indices, [1, 0])
case("min(1).values", lambda: M.min(1).values, [2, 3])
case("amax(1)", lambda: M.amax(1), [7, 5])
case("argmax(1)", lambda: M.argmax(1), [1, 0])
case("argmin(1)", lambda: M.argmin(1), [0, 1])
case("aminmax(dim=1).max", lambda: torch.aminmax(M, dim=1).max, [7, 5])
case("any()", lambda: Bm.any().item(), True)
case("any(-1)", lambda: Bm.any(-1), [True, True])


def ip():
    t = torch.zeros(3, 8, dtype=torch.int64).npu()
    t[:, torch.tensor([True] * 5 + [False] * 3).npu()] = torch.arange(1, 16).view(3, 5).npu()
    return t[:, :5]


case("index_put  t[:, m] = v  (None)", ip,
     [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])

# ---- 输出 ----
base = sorted(us for _, _, _, us, b in ROWS if b and us)
med = base[len(base) // 2] if base else 0
print(f"基准往返耗时中位数：{med:.0f} us（{len(base)} 个基准算子）")
print(f"判据：≥ {med*0.5:.0f} us 视为走了服务端；< {med*0.3:.0f} us 视为没出网\n")
print(f"{'算子':<34}{'对?':<5}{'耗时':>10}  {'层次':<12} 结论")
print("-" * 104)

todo = {"client": [], "server": [], "suspect": []}
for tag, got, want, us, b in ROWS:
    if us is None:
        print(f"{tag:<34}{'EXC':<5}{'-':>10}  {got}")
        continue
    ok = got == want
    if us >= med * 0.5:
        layer, lk = "走服务端", "server"
    elif us < med * 0.3:
        layer, lk = "没出网", "client"
    else:
        layer, lk = "不确定", "suspect"
    if ok and lk == "server":
        concl = "已通过"
    elif ok and lk == "client":
        concl = "⚠️ 可疑：没出网却正确 = 回收缓冲区蒙对"
        todo["suspect"].append(tag)
    elif not ok and lk == "server":
        concl = "→ 服务端 runner 没处理好参数"
        todo["server"].append(tag)
    elif not ok and lk == "client":
        concl = "→ 客户端白名单缺这个 overload"
        todo["client"].append(tag)
    else:
        concl = "耗时落在灰区，加大 N 重测"
        todo["suspect"].append(tag)
    print(f"{tag:<34}{'✅' if ok else '❌':<4}{us:>9.0f}us  {layer:<12} {concl}")
    if not ok:
        print(f"{'':<34}     got={got}  want={want}")

print()
print("=" * 104)
print("派活清单")
print("=" * 104)
for k, title in (("client", "① 客户端白名单（服务端加 runner 无效）"),
                 ("server", "② 服务端 runner（已在白名单里，参数没处理好）"),
                 ("suspect", "③ 需要复核")):
    if todo[k]:
        print(f"\n{title}")
        for t in todo[k]:
            print(f"   · {t}")
print(f"""
另：每个真算算子约 {med:.0f} us 的 gRPC 往返 —— 这就是当初把阈值调到 64MB 后跑不完的原因。
一个 step 里若有 1000 个真算算子，光往返就是 {med*1000/1e6:.1f} 秒。白名单要按这个量级来定，
宁可全局阈值小、给个别算子开例外，也不要统一放宽。""")
