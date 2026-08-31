#!/usr/bin/env python3
"""
第二轮定位。上一轮 probe_stride_and_numel.py 的结论：

  · 最大通过：bool  n=4096  → 4096 字节
  ·           int64 n=360   → 2880 字节
  · 最小失败：int64 n=4096  → 32768 字节
  ·           bool  n=65536 → 65536 字节
  · 超限时 bool 的 device sum 恒为 4096 → 只有前 4096 字节到了 device

  ⇒ 天花板是 **4096 字节（4 KiB / 一个页）**，按字节算，不是按 numel 算。

  · stride / storageOffset / stride==0 / aclop BroadcastTo —— 小张量上全部正确，
    上一轮 §2.5 猜的 P1/P2 已作废。

本脚本要回答四个还没定的问题：

  Q1  天花板的精确位置在哪？H2D 到底送了多少字节上去？
  Q2  D2H 是独立坏的吗？（device sum 说有 4096 个 1，d.cpu() 却全 0）
  Q3  int64 超限时 device sum = 1，不是预期的 512。为什么？
  Q4  852 那个 252 是不是同一个 4 KiB 天花板？（逻辑 1.47MB > 4096）

  Q5  sum 的 dim 归约变体是真坏的（96 字节，远在天花板之下）—— 是 permute 引起的还是 dim 引起的？

用法：  python probe_4k_boundary.py

规矩同前：输入一律 H2D；单元素读取用 .item()（每次 8 字节，走已验证正常的小通道）。
"""
import torch
import torch_npu  # noqa: F401


def line(tag, got, want, tell=""):
    flag = "OK  " if got == want else "FAIL"
    msg = f"{flag} | {tag:<46} got={got!s:<14} want={want}"
    if flag == "FAIL" and tell:
        msg += f"   [{tell}]"
    print(msg)


def head(t):
    print()
    print("=" * 100)
    print(t)
    print("=" * 100)


head("Q1  H2D 到底送了多少字节？用单元素 .item() 逐点探边界（每次只 D2H 8 字节）")
print("     bool 源全 True。找到第一个变 False 的下标 = H2D 实际送达的字节数")
n = 65536
d = torch.ones(n, dtype=torch.bool).npu()
probe = [0, 1, 1023, 1024, 2047, 2048, 4094, 4095, 4096, 4097, 8191, 8192, 65535]
got = {i: bool(d[i].item()) for i in probe}
print("     ", {i: int(v) for i, v in got.items()})
edge = next((i for i in range(len(probe)) if not got[probe[i]]), None)
if edge is None:
    print("      → 所有探点都是 True，H2D 其实是完整的，问题在别处")
else:
    print(f"      → 最后一个 True 在 {probe[edge-1]}，第一个 False 在 {probe[edge]}")
    print(f"        边界落在 ({probe[edge-1]}, {probe[edge]}] 字节")

print()
print("     int64 版本（源全 1）。若边界同样是 4096 字节，第 511 个元素应为 1、第 512 个为 0")
d64 = torch.ones(4096, dtype=torch.int64).npu()
probe64 = [0, 1, 2, 100, 510, 511, 512, 513, 1023, 4095]
got64 = {i: int(d64[i].item()) for i in probe64}
print("     ", got64)
print("      → 若只有 0 号是 1     ⇒ int64 只送了 8 字节，与 bool 的 4096 字节不同规则")
print("      → 若 0..511 都是 1    ⇒ 同样是 4096 字节，那 device sum=1 是 sum 自己的问题")

head("Q2  D2H 是不是独立坏的？")
print("     device 上确认有 4096 个 True（上一轮 d.sum()=4096），但 d.cpu().sum() = 0")
line("d.sum()  (device 归约，8 字节读回)", int(d.sum().item()), 4096,
     "跟上一轮不一致，先确认环境没变")
line("d.cpu().sum()  (整块 D2H 65536 字节)", int(d.cpu().sum()), 4096,
     "device 上明明有 4096 个 1，D2H 回来全 0 ⇒ D2H 超限时什么都没写")
line("d[:4096].cpu().sum()  (D2H 4096 字节)", int(d[:4096].cpu().sum()), 4096,
     "连 4096 字节的 D2H 都不对，天花板比 4096 更小")
line("d[:2048].cpu().sum()  (D2H 2048 字节)", int(d[:2048].cpu().sum()), 2048)

head("Q3  int64 超限时 sum=1 的来源")
for n64 in (256, 512, 513, 1024, 4096):
    dd = torch.ones(n64, dtype=torch.int64).npu()
    line(f"int64 n={n64:<5} ({n64*8:>6} 字节) device sum", int(dd.sum().item()), n64,
         "得 512 = 前 4096 字节；得 1 = 另有原因")

head("Q4  852 的 252 是不是同一个 4 KiB 天花板？（expand 的逻辑字节数跨过 4096）")
b = torch.zeros(8, dtype=torch.bool)
b[:5] = True
d8 = b.npu()                                     # storage 只有 8 字节
line("storage 8B  → expand 逻辑 800B  (<4K)", int(d8.unsqueeze(-1).expand(8, 100).sum().item()), 500,
     "小于天花板都不对，那 852 另有原因")
line("storage 8B  → expand 逻辑 4096B (=4K)", int(d8.unsqueeze(-1).expand(8, 512).sum().item()), 2560)
line("storage 8B  → expand 逻辑 32768B(>4K)", int(d8.unsqueeze(-1).expand(8, 4096).sum().item()), 20480,
     "得 5 = 只算了底层 storage ⇒ 与 852 的 252 完全同型，同一个 4KiB 天花板")

head("Q5  sum 的 dim 归约：是 permute 引起的还是 dim 引起的？（96 字节，远在天花板之下）")
m = (torch.arange(12) + 1).npu().view(3, 4)      # [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
line("连续 + 全归约   m.sum()", int(m.sum().item()), 78)
line("连续 + dim 归约 m.sum(0)", m.sum(0).cpu().tolist(), [15, 18, 21, 24],
     "连连续张量的 dim 归约都错 ⇒ 坏的是 sum.dim_IntList 变体本身")
line("连续 + dim 归约 m.sum(1)", m.sum(1).cpu().tolist(), [10, 26, 42])
line("permute + 全归约 m.t().sum()", int(m.t().sum().item()), 78,
     "permute 全归约就错 ⇒ 坏的是 permute 的 stride 处理")
line("permute + dim 归约 m.t().sum(0)", m.t().sum(0).cpu().tolist(), [10, 26, 42],
     "上一轮得 [63,1,1]，63 正是上一条测试的结果 ⇒ 输出缓冲区没被写，复用了回收块")
print()
print("     顺带测同族的 dim 归约（链 1 依赖 prod(dim=1)，链 4 依赖 all(-1)）")
g = torch.tensor([[1, 28, 36]], dtype=torch.int64).npu()
line("prod(dim=1)  grid_thw 同型", g.prod(dim=1).cpu().tolist(), [1008],
     "链 1 的 total_tokens 就断在这里")
line("max(dim=1)   grid_thw[:,1:]", int(g[:, 1:].max().item()), 36)
bb = torch.tensor([[True, True], [True, False]]).npu()
line("all(-1)      get_placeholder_mask:841", bb.all(-1).cpu().tolist(), [True, False])
