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
    SWA_TILING=64,64 python _t_swa_reach_triton.py # 换 tiling (默认 128,128 = 原 Config 的值)
    SWA_MULTIBUFFER=1 python _t_swa_reach_triton.py
    SWA_NO_TILING=1 python _t_swa_reach_triton.py  # 不注入 tiling (autotune 还在时用)
    SWA_DTYPE=bf16 python _t_swa_reach_triton.py   # f32 / bf16 / f16; 原测试另两档是 bf16
    SWA_DUMP_META=1 python _t_swa_reach_triton.py  # 只读: 翻出上次编译写下的 grid_offset, 不碰 NPU
    SWA_GRID=8 python _t_swa_reach_triton.py       # 强制 grid=8 (默认取 cube 核数 32)

已知卡点 (2026-10-23): BLOCK_M=BLOCK_N=128 + fp32 + HEAD_DIM=128 编译不过, bishengir 报
    ub overflow, requires 3715072 bits while 2031616 bits available
即需要 453.5 KiB 而 UB 只有 248 KiB。这是**正式编译器**在 --target=Ascend950PR_9589 下报的,
不是仿真器的执行引擎。绕法两条: SWA_TILING=64,64 把分块砍小, 或 SWA_DTYPE=bf16 (粗算约 227 KiB,
刚好装得下)。注意两者都偏离了原 Config 的生产配置。

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
    "small": dict(tot_q=256, q_h=4, kv_h=1, d=128, splits=[0, 128, 256]),
    "full": dict(tot_q=2048, q_h=16, kv_h=4, d=128, splits=[0, 1024, 2048]),
    # 原测试的另两档是 bf16; BF16_PADDIM 的 head_dim 是 96
    "paddim": dict(tot_q=2048, q_h=16, kv_h=4, d=96, splits=[0, 1024, 2048]),
}
DTYPES = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}
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


class _KernelWithTiling:
    """给已经去掉 @triton.autotune 的 kernel 补回 Config 里那几个值, 不改 swa.py。

    swa.py 原本是 `@triton.autotune(configs=[Config({"BLOCK_M":128,"BLOCK_N":128,
    "multibuffer":False})], key=["HEAD_DIM"])`, 三层推导式展开只有 1 个 config。
    BLOCK_M / BLOCK_N 是 kernel 签名末尾的 tl.constexpr 形参, swa_fwd_impl 的
    launch 里没传, 全靠 Config 注入; 一旦把装饰器注释掉就会报
        TypeError: dynamic_func() missing 2 required positional arguments
    这个代理在 `kernel[grid](...)` 时把它们补上。
    """

    def __init__(self, kernel, **fixed):
        self._kernel = kernel
        self._fixed = fixed

    def __getitem__(self, grid):
        inner = self._kernel[grid]
        fixed = self._fixed

        def launch(*args, **kwargs):
            for key, value in fixed.items():
                kwargs.setdefault(key, value)
            try:
                return inner(*args, **kwargs)
            except TypeError as error:
                # multibuffer 不是 kernel 形参, 而是编译选项; 万一这个 triton
                # 版本不认, 去掉它再来一次, 免得白跑一趟。
                if "multibuffer" in str(error) and "multibuffer" in kwargs:
                    kwargs.pop("multibuffer")
                    print(f"[info] launch 不认 multibuffer ({error}), 去掉后重试", flush=True)
                    return inner(*args, **kwargs)
                raise

        return launch

    def __getattr__(self, name):
        return getattr(self._kernel, name)


class _Null:
    """SWA_NO_MODE=1 时顶替拦截层, 什么都不做。"""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def dump_meta():
    """只读模式: 把 hal_policy 为 Triton kernel 写下的 metadata.json 翻出来看。

    hal_policy.py 在编译成功时会把 grid_offset 注册给 HAL, 并落一份
        <NPUSIM_SESSION_DIR>/native-triton/<sha256>/metadata.json
    里面有 grid_offset / resources / kernel.metadata。grid_offset 是
    「grid 在参数块里的字节偏移」, HAL 靠它去读 grid; 偏移不对, kernel 里
    `block_idx % num_programs(0)` 就会拿到 0 -> scalar_rem div by 0。
    本模式不碰 NPU, 不起仿真器。
    """
    import glob
    import json

    roots = []
    if os.environ.get("NPUSIM_SESSION_DIR"):
        roots.append(os.environ["NPUSIM_SESSION_DIR"])
    roots += glob.glob(os.path.join(os.getcwd(), "build", "sessions", "*"))
    pats = [os.path.join(r, "**", "native-triton", "*", "metadata.json") for r in roots]
    files = sorted({p for pat in pats for p in glob.glob(pat, recursive=True)},
                   key=lambda p: os.path.getmtime(p), reverse=True)
    if not files:
        print("[meta] 没找到 native-triton/*/metadata.json", flush=True)
        print(f"[meta] 找过: NPUSIM_SESSION_DIR={os.environ.get('NPUSIM_SESSION_DIR')!r} "
              f"和 {os.path.join(os.getcwd(), 'build/sessions/*')}", flush=True)
        return
    for path in files[:int(os.environ.get("SWA_META_N", "4"))]:
        try:
            data = json.loads(open(path, encoding="utf-8").read())
        except (OSError, ValueError) as error:
            print(f"[meta] 读不了 {path}: {error}", flush=True)
            continue
        age = time.time() - os.path.getmtime(path)
        print(f"\n[meta] {path}  ({age/60:.1f} 分钟前)", flush=True)
        print(f"   kernel      = {data.get('kernel')}", flush=True)
        print(f"   grid_offset = {data.get('grid_offset')}   <-- HAL 按这个偏移去参数块里读 grid",
              flush=True)
        print(f"   resources   = {data.get('resources')}", flush=True)
        md = data.get("metadata") or {}
        for key in ("name", "num_warps", "shared", "mix_mode", "kernel_arg_size",
                    "BLOCK_M", "BLOCK_N", "multibuffer"):
            if key in md:
                print(f"   metadata.{key} = {md[key]}", flush=True)
        extra = {k: v for k, v in md.items()
                 if k not in ("name", "num_warps", "shared", "mix_mode", "kernel_arg_size")}
        if extra:
            print(f"   metadata 其余键 = {sorted(extra)}", flush=True)


def main():
    import math

    if os.environ.get("SWA_DUMP_META", "") not in ("", "0"):
        dump_meta()
        return

    if REPO is None:
        sys.exit("没找到仓库根 (需要能看到 mojo_opset/__init__.py)")
    shape_name = os.environ.get("SWA_SHAPE", "small")
    if shape_name not in SHAPES:
        sys.exit(f"SWA_SHAPE 只能是 {list(SHAPES)}")
    cfg = dict(SHAPES[shape_name])
    dtype_name = os.environ.get("SWA_DTYPE", "f32")
    if dtype_name not in DTYPES:
        sys.exit(f"SWA_DTYPE 只能是 {list(DTYPES)}")
    cfg["dtype"] = DTYPES[dtype_name]
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
        print(f"[info] get_num_cores('cube') = {get_num_cores('cube')}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[info] 取 cube 核数失败: {e!r}", flush=True)

    # SWA_GRID: 强行换掉 swa.py:1800 那句 get_num_cores("cube"), 用来把 grid 这个变量
    # 单独拎出来试。已验: Triton 的 aiv / mix 最小 kernel 在 grid=1/2/4/5/8 下全过,
    # 而 SWA 用的是 grid=32 —— 32 从没在 Triton 侧验过。
    if os.environ.get("SWA_GRID"):
        forced = int(os.environ["SWA_GRID"])
        swa_mod.get_num_cores = lambda op_type="vector", _n=forced: _n
        print(f"[info] SWA_GRID={forced} -> 已把 swa.get_num_cores 换成常量, grid 将是 ({forced},)",
              flush=True)
    else:
        print(f"[info] grid 将是 ({get_num_cores('cube')},)  (SWA_GRID=N 可强制改成 N)", flush=True)

    # @triton.autotune 被注释掉之后, Config 里的 BLOCK_M/BLOCK_N 没人提供 -> 代理补上
    kern = swa_mod._swa_fwd_kernel
    n_cfg = len(getattr(kern, "configs", []) or [])
    if os.environ.get("SWA_NO_TILING", "") not in ("", "0"):
        print("[info] SWA_NO_TILING 已设, 不注入 tiling", flush=True)
    elif n_cfg:
        print(f"[info] _swa_fwd_kernel 仍带 autotune (configs={n_cfg}), 不注入 tiling", flush=True)
    else:
        bm, bn = (int(x) for x in os.environ.get("SWA_TILING", "128,128").split(","))
        mb = os.environ.get("SWA_MULTIBUFFER", "0") not in ("0", "false", "False")
        swa_mod._swa_fwd_kernel = _KernelWithTiling(
            kern, BLOCK_M=bm, BLOCK_N=bn, multibuffer=mb)
        print(f"[info] _swa_fwd_kernel 没有 autotune -> 注入 BLOCK_M={bm} BLOCK_N={bn} "
              f"multibuffer={mb}", flush=True)

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
