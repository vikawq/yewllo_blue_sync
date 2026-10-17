"""探查 @triton.autotune 在 standalone 仿真下到底执行不执行。

背景: mojo_opset 的 npu kernels 里有 16 个文件用了 @triton.autotune，
      其中 rmsnorm 的 _rmsnorm_infer_kernel_single 有 9x2x2 = 36 个 config。
      autotune 的语义是「每个 config 真跑一遍计时，选最快的」，
      在周期级仿真器上这意味着 36 次完整仿真 —— 但实测 rmsnorm 是秒级完成的，
      所以它多半没有正常执行，而是由 standalone_runtime 自己决定了跑哪个变体
      (traceback 里 variant 这个参数一路传到 bridge._run_standalone)。

本脚本做三件事，互相独立，任一失败不影响其余:
  A  静态扫 standalone_runtime 的源码，找 variant / autotune 相关的处理
  B  自建一个带 4 个 config 的探针 kernel，跑一次，数产生了几个 launch 目录
     —— 这是决定性判据，不需要读源码，也不需要 mojo_opset
  C  把探针 kernel 的装饰器链扒开，看 Autotuner 选了什么
  D  若 mojo_opset 可导入，顺便自省真实的 rmsnorm 两个 kernel

用法 (先 source /root/env.sh):
    python3 _t_autotune.py
"""
import glob
import os
import re
import sys
import traceback

# triton ascend driver 初始化要查 torch_npu 的包元数据，而本机的 torch_npu 是
# 手搓 stub、没有 dist-info。这里在进程内打个回退补丁，不动 site-packages。
import importlib.metadata as _md

_orig_version = _md.version


def _version(name):
    try:
        return _orig_version(name)
    except _md.PackageNotFoundError:
        if name == "torch_npu":
            import torch_npu

            return getattr(torch_npu, "__version__", "0.0.0")
        raise


_md.version = _version

os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")

SR_DIR = "/opt/stars/standalone_runtime"
ART_DIR = "/opt/stars/artifacts/triton-runtime"
N_CONFIGS = 4


def hr(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------- A 静态扫描
def part_a():
    hr("A  standalone_runtime 里怎么处理 variant / autotune")
    files = sorted(glob.glob(os.path.join(SR_DIR, "*.py")))
    if not files:
        print(f"  {SR_DIR} 下没有 .py —— 跳过")
        return
    pats = {
        "variant": re.compile(r"\bvariant\b"),
        "autotune/Autotuner": re.compile(r"[Aa]utotun"),
        "configs/best_config": re.compile(r"\bconfigs\b|\bbest_config\b"),
        "do_bench/benchmark": re.compile(r"do_bench|benchmark"),
    }
    for label, pat in pats.items():
        hits = []
        for f in files:
            for i, ln in enumerate(open(f, encoding="utf-8", errors="replace"), 1):
                if pat.search(ln):
                    hits.append((os.path.basename(f), i, ln.rstrip()[:118]))
        print(f"\n  --- {label}：{len(hits)} 处 ---")
        for name, i, ln in hits[:14]:
            print(f"    {name}:{i}  {ln}")
        if len(hits) > 14:
            print(f"    ... 还有 {len(hits) - 14} 处")


# ------------------------------------------------- B 探针 kernel + 数 launch
def count_launches():
    return len(glob.glob(os.path.join(ART_DIR, "launch-*")))


def part_b():
    hr(f"B  自建 {N_CONFIGS}-config 的探针 kernel，数它产生几个 launch 目录")

    import torch
    import triton
    import triton.language as tl
    import standalone_runtime

    @triton.autotune(
        configs=[triton.Config({"BLOCK": b}) for b in (64, 128, 256, 512)],
        key=["n"],
    )
    @triton.jit
    def _probe_kernel(X, Y, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(X + offs, mask=mask, other=0.0)
        tl.store(Y + offs, x * 2.0, mask=mask)

    n = 1024
    x = torch.arange(n, dtype=torch.float32) * 0.001
    y = torch.empty_like(x)

    before = count_launches()
    print(f"  跑之前的 launch 目录数: {before}")
    ok = False
    try:
        with standalone_runtime.enable(timeout=180):
            _probe_kernel[(8,)](x, y, n, compile_on_910_95=True)
            standalone_runtime.synchronize()
        ok = True
    except Exception as e:  # noqa: BLE001
        print(f"  探针执行抛异常: {type(e).__name__}: {str(e).splitlines()[0][:110]}")
        print("  (即使失败，launch 目录通常也已经落盘，下面的计数仍然有意义)")
    after = count_launches()
    delta = after - before
    print(f"  跑之后的 launch 目录数: {after}   新增 = {delta}")

    print("\n  判据:")
    if delta <= 1:
        print(f"    新增 {delta} 个 -> autotune **没有** benchmark，只跑了一个变体。")
        print("    说明仿真下选哪个 config 不是 triton 的 autotuner 决定的。")
    elif delta >= N_CONFIGS:
        print(f"    新增 {delta} 个 (>= config 数 {N_CONFIGS}) -> autotune 真的在逐个计时。")
        print("    那么 36-config 的 rmsnorm 每次调用要跑 36 次仿真。")
    else:
        print(f"    新增 {delta} 个，介于 1 和 {N_CONFIGS} 之间 -> 部分 config 被 prune 掉了。")
    if ok:
        err = (y - x * 2.0).abs().max().item()
        print(f"\n  探针数值校验 max_abs_error = {err:.3e}")
    return _probe_kernel


# ------------------------------------------------------------ C 装饰器链自省
def walk(name, k, limit=8):
    print(f"\n  --- {name} ---")
    depth = 0
    while k is not None and depth < limit:
        cfgs = getattr(k, "configs", None)
        best = getattr(k, "best_config", None)
        cache = getattr(k, "cache", None)
        bits = [f"{depth}  {type(k).__module__}.{type(k).__name__}"]
        if cfgs is not None:
            bits.append(f"configs={len(cfgs)}")
        if best is not None:
            bits.append(f"best={best}")
        if isinstance(cache, dict):
            bits.append(f"cache={len(cache)} 项")
        print("    " + "  ".join(bits))
        nxt = getattr(k, "fn", None)
        if nxt is k:
            break
        k = nxt
        depth += 1


def part_c(probe):
    hr("C  探针 kernel 的装饰器链（跑过一次之后）")
    if probe is None:
        print("  探针没建起来，跳过")
        return
    walk("_probe_kernel", probe)


# ------------------------------------------------- D 真实 rmsnorm kernel 自省
def part_d():
    hr("D  真实 mojo_opset rmsnorm kernel 的装饰器链")
    try:
        from mojo_opset.backends.ttx.kernels.npu import rmsnorm as R
    except Exception as e:  # noqa: BLE001
        print(f"  mojo_opset 导入不了 ({type(e).__name__}: {str(e)[:90]}) —— 跳过")
        print("  若这台机器上有 mojo_opset 源码，把它加进 PYTHONPATH 再跑一次。")
        return
    for nm in ("_rmsnorm_infer_kernel", "_rmsnorm_infer_kernel_single"):
        k = getattr(R, nm, None)
        if k is None:
            print(f"\n  --- {nm} --- 不存在")
            continue
        walk(nm, k)


def main():
    print(f"[env] python      = {sys.executable}")
    print(f"[env] ASCEND_HOME = {os.environ.get('ASCEND_HOME_PATH') or '(未设置)'}")
    if not os.environ.get("ASCEND_HOME_PATH"):
        print("\n!! ASCEND_HOME_PATH 未设置。A 段仍可跑，B/C/D 需要:")
        print("!!   source /root/env.sh && python3 _t_autotune.py")

    probe = None
    for fn, arg in ((part_a, None), (part_b, None)):
        try:
            r = fn() if arg is None else fn(arg)
            if fn is part_b:
                probe = r
        except Exception:  # noqa: BLE001
            print(f"\n  [{fn.__name__} 异常]")
            traceback.print_exc(limit=6)
    for fn, arg in ((part_c, probe), (part_d, None)):
        try:
            fn(arg) if fn is part_c else fn()
        except Exception:  # noqa: BLE001
            print(f"\n  [{fn.__name__} 异常]")
            traceback.print_exc(limit=6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
