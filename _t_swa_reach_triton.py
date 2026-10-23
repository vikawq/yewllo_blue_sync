"""把 mask 和输出分配都留在 CPU, 试着让 SWA 的 Triton 正向 kernel 真正跑起来。

为什么这样就有机会:
    仿真机上原生 torch 算子一旦用到 5 个以上 block 必挂 (实测分界线 = 4 block),
    但 Triton kernel 在 grid=1/2/4/5/8 下全部正常, H2D/D2H 也不起 kernel。
    所以只要让 swa_fwd_impl 里那几个大的 torch 算子不在设备上执行, 就能摸到 Triton kernel。

swa_fwd_impl 在启动 kernel 之前有四处拦路者, 本脚本分两招处理, **都不改 swa.py**:

  1) mask (swa.py:1774 -> swa.py:80-92) —— 不设 default device 即可。
     那几句 `torch.ones(M, N, dtype=torch.bool).tril()` 全都没写 device, 本来就是靠
     默认设备; swa.py:93 才 `.to(device="npu")`。mojo_opset 的 conftest 调了
     `torch.set_default_device('npu')`, 才把它们顶到设备上去。本脚本是独立进程、
     不碰 default device, 于是 mask 按作者原意在 CPU 上拼好、一次 H2D 搬上去。
     (实测: CPU 建 1792x2048 bool 再 H2D = 0.13s 干净通过)

  2) 输出分配 (swa.py:1785 zeros_like / 1786 zeros+(-inf) / 1792 o_f32 zeros_like)
     —— 用 TorchFunctionMode 拦下来, 在 CPU 上建好再 H2D。这是进程内的拦截,
     swa.py 源码一个字节都不动。

代价说清楚: 被拦下来的那几个算子**不再经过仿真器**, 所以这次运行不能用来validate
那几步本身。它只回答一个问题: Triton 的 SWA 正向 kernel 在这台仿真机上能不能跑完。

用法 (cwd 留在 NPU-Sim):
    python _t_swa_reach_triton.py                  # 默认小 shape, 先验证通路
    SWA_SHAPE=full python _t_swa_reach_triton.py   # 原测试的 M_F32 shape
    SWA_OUTPUT_F32=0 python _t_swa_reach_triton.py # 少一处 o_f32 分配
    SWA_NO_MODE=1 python _t_swa_reach_triton.py    # 不装拦截层, 对照用 (应挂在 1785)

判读 —— 看日志最后停在哪:
  * 打完 3 条 [mode] 拦截记录后卡住      -> 已经进 Triton kernel (或 autotune), 拦截成功,
                                            问题在 kernel 本身或 autotune 计时
  * [done] 带出 o 的统计                -> **Triton 正向 kernel 在仿真上跑通了**
  * 第 1 条 [mode] 之前就卡住            -> 还有没拦到的 torch 算子, 看 Ctrl-C 的栈
"""
import os
import sys
import time


def _add_repo_root_to_path():
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isfile(os.path.join(d, "mojo_opset", "__init__.py")):
            if d not in sys.path:
                sys.path.insert(0, d)
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


REPO = _add_repo_root_to_path()

import torch
import torch_npu  # noqa: F401
from torch.overrides import TorchFunctionMode

# 原测试 M_F32: B=2 Q_LEN=1024 -> 2048 tokens, 16 q heads, 4 kv heads, D=128, fp32
SHAPES = {
    "small": dict(tot_q=256, q_h=4, kv_h=1, d=128, dtype=torch.float32, splits=[0, 128, 256]),
    "full": dict(tot_q=2048, q_h=16, kv_h=4, d=128, dtype=torch.float32, splits=[0, 1024, 2048]),
}
LOCAL, GLOBAL_W = 255, 4
GQA_INTERLEAVE = True


def _ts():
    return time.strftime("%H:%M:%S")


class CpuAllocMode(TorchFunctionMode):
    """把在 npu 上做的「整块新建」和「加标量」挪到 CPU, 结果再 H2D 搬回去。

    只认白名单里的那几个函数, 其余一律原样放过, 所以 Triton 的 launch 路径
    (data_ptr / is_contiguous / stride ...) 完全不受影响。
    """

    LIKE = {torch.zeros_like: torch.zeros, torch.ones_like: torch.ones,
            torch.empty_like: torch.empty}
    FACTORY = {torch.zeros, torch.ones, torch.empty, torch.full}
    ADDS = {torch.add, torch.Tensor.add, torch.Tensor.__add__}

    def __init__(self):
        super().__init__()
        self.hits = []

    def _log(self, text):
        self.hits.append(text)
        print(f"[{_ts()}] [mode] #{len(self.hits)} {text}", flush=True)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(func, "__name__", str(func))

        # zeros_like / ones_like / empty_like, 参照张量在 npu 上
        if func in self.LIKE and args and isinstance(args[0], torch.Tensor) \
                and args[0].device.type == "npu":
            ref = args[0]
            dtype = kwargs.get("dtype") or ref.dtype
            out = self.LIKE[func](tuple(ref.shape), dtype=dtype, device="cpu").to(ref.device)
            self._log(f"{name}{tuple(ref.shape)} {dtype} -> CPU 新建 + H2D")
            return out

        # zeros / ones / empty / full, 显式 device=npu
        if func in self.FACTORY:
            dev = kwargs.get("device")
            if dev is not None and torch.device(dev).type == "npu":
                kw = dict(kwargs)
                kw["device"] = "cpu"
                out = func(*args, **kw).to(dev)
                self._log(f"{name}(device={dev}) -> CPU 新建 + H2D")
                return out

        # tensor + 标量
        if func in self.ADDS and len(args) >= 2 and isinstance(args[0], torch.Tensor) \
                and args[0].device.type == "npu" \
                and isinstance(args[1], (int, float)) and not isinstance(args[1], bool):
            a, b = args[0], args[1]
            out = (a.cpu() + b).to(a.device)
            self._log(f"{name}(npu tensor{tuple(a.shape)} + {b!r}) -> D2H 加完再 H2D")
            return out

        return func(*args, **kwargs)


class _Null:
    """SWA_NO_MODE=1 时顶替拦截层, 什么都不做。"""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main():
    import math

    if REPO is None:
        sys.exit("没找到仓库根 (需要能看到 mojo_opset/__init__.py)")
    shape_name = os.environ.get("SWA_SHAPE", "small")
    if shape_name not in SHAPES:
        sys.exit(f"SWA_SHAPE 只能是 {list(SHAPES)}")
    cfg = SHAPES[shape_name]
    output_f32 = os.environ.get("SWA_OUTPUT_F32", "1") not in ("0", "false", "False")
    use_mode = os.environ.get("SWA_NO_MODE", "") in ("", "0")

    from mojo_opset.backends.ttx.kernels.npu import swa as swa_mod
    from mojo_opset.backends.ttx.kernels.npu.utils import get_num_cores

    print(f"[info] repo={REPO}", flush=True)
    print(f"[info] shape={shape_name} {cfg['tot_q']}x{cfg['q_h']}x{cfg['d']} kv_h={cfg['kv_h']} "
          f"{cfg['dtype']}  output_f32={output_f32}  拦截层={'on' if use_mode else 'off'}", flush=True)
    print(f"[info] default device 没有设置 -> mask 会在 CPU 上构造 "
          f"(torch.empty(0).device = {torch.empty(0).device})", flush=True)
    try:
        print(f"[info] Triton 正向 kernel 的 grid 将是 ({get_num_cores('cube')},)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[info] 取 cube 核数失败: {e!r}", flush=True)

    # 输入全部 CPU 建好再 .npu() —— 本机 torch.tensor(..., device='npu') 不可用
    torch.manual_seed(43)
    tot_q, q_h, kv_h, d, dtype = cfg["tot_q"], cfg["q_h"], cfg["kv_h"], cfg["d"], cfg["dtype"]
    q = torch.randn(tot_q, q_h, d, dtype=dtype).npu()
    k = torch.randn(tot_q, kv_h, d, dtype=dtype).npu()
    v = torch.randn(tot_q, kv_h, d, dtype=dtype).npu()
    cu = torch.tensor(cfg["splits"], dtype=torch.int32).npu()
    torch.npu.synchronize()
    print(f"[{_ts()}] [info] 输入已就位 (H2D 完成)", flush=True)

    mode = CpuAllocMode()
    ctx = mode if use_mode else _Null()
    t0 = time.perf_counter()
    print(f"[{_ts()}] [info] 调 swa_fwd_impl ...", flush=True)
    with ctx:
        out = swa_mod.swa_fwd_impl(
            q=q, k=k, v=v, cu_q_lens=cu, cu_total_seq_lens=cu,
            is_causal=True, local_window_size=LOCAL, global_window_size=GLOBAL_W,
            softmax_scale=1.0 / math.sqrt(d), gqa_interleave=GQA_INTERLEAVE,
            output_f32=output_f32,
        )
    torch.npu.synchronize()
    dt = time.perf_counter() - t0

    o, lse = out[0], out[1]
    mask = swa_mod._COMPRESSED_MASK
    o_c, lse_c = o.cpu(), lse.cpu()
    print(f"[{_ts()}] [done] swa_fwd_impl 返回, 耗时 {dt:.1f}s", flush=True)
    print(f"  mask: {tuple(mask.shape) if mask is not None else None} "
          f"device={mask.device if mask is not None else '-'}", flush=True)
    print(f"  o:   shape={tuple(o_c.shape)} finite={bool(torch.isfinite(o_c).all())} "
          f"nonzero={int((o_c != 0).sum())}/{o_c.numel()}", flush=True)
    print(f"  lse: shape={tuple(lse_c.shape)} 仍为 -inf 的个数="
          f"{int(torch.isneginf(lse_c).sum())}/{lse_c.numel()}", flush=True)
    print(f"  拦截层共拦下 {len(mode.hits)} 次", flush=True)
    for i, h in enumerate(mode.hits, 1):
        print(f"    {i}. {h}", flush=True)
    if int((o_c != 0).sum()) == 0:
        print("  [warn] o 全零 —— kernel 可能没真正写回, 别当成跑通", flush=True)


if __name__ == "__main__":
    main()
