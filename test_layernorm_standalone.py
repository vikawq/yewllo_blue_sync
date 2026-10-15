#!/usr/bin/env python3
"""Self-contained LayerNorm NPU kernel repro. Needs only torch / torch_npu / triton.

Does NOT import mojo_opset -- the kernel is inlined below, copied from
mojo_opset/backends/ttx/kernels/npu/layernorm.py, so you can scp this one file
anywhere and run it.

Two kernel variants, same body, different argument binding:

  --variant repro   BLOCK_SIZE_M is left unbound, exactly as the shipped
                    layernorm.py does it: @triton.heuristics commented out,
                    @triton.autotune(configs=[]) in its place (which per
                    triton-ascend docs performs no autotune at all), and no
                    BLOCK_SIZE_M= at the launch. This is the crash.

  --variant fixed   BLOCK_SIZE_M computed on the host and passed explicitly,
                    BLOCK_SIZE_N forced to a power of two, mean/rstd in fp32.

Typical session:

    python3 test_layernorm_standalone.py --list
    python3 test_layernorm_standalone.py --variant repro --dtype float32
    python3 test_layernorm_standalone.py --variant fixed --dtype float32
    python3 test_layernorm_standalone.py --case 0 --inproc --variant repro
    python3 test_layernorm_standalone.py --variant fixed --block-m 7  # force a value

--inproc runs in this process (no fork), so simulator / msdebug output is
attributable to a single kernel launch. Without it each case gets its own
subprocess and a device abort only kills that one case.
"""

import argparse
import os
import subprocess
import sys
import time
import traceback

import torch

try:
    import torch_npu  # noqa: F401  (registers the npu device)
except ImportError:
    torch_npu = None

import triton
import triton.language as tl

try:
    from triton.language.math import rsqrt
except ImportError:
    rsqrt = tl.rsqrt

try:
    from triton.runtime.libentry import libentry
except ImportError:
    def libentry():
        def _decorator(fn):
            return fn
        return _decorator


# ---------------------------------------------------------------- shapes ----

LAYERNORM_SHAPES = [
    (256, 128),
    (1, 768),
    (128, 768),
    (2048, 768),
    (64, 64, 256),
    (1, 4096),
    (128, 4096),
    (2048, 4096),
    (512, 8192),
    (128, 8192),
    (65535, 8192),
]

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# The standalone memory window is [0, 0x80000000). x, y and the reference
# output all have to live in it at once.
MEM_WINDOW_BYTES = 0x80000000


# ------------------------------------------------- host-side block sizes ----

COL_BLOCKING_THRESHOLD = 2048

TOKEN_BLOCK_SIZE_TABLE = {
    2048: 4,
    1024: 8,
    512: 10,
    256: 18,
    128: 24,
}


def pick_block_m(n_cols):
    """The value @triton.heuristics was supposed to supply."""
    if n_cols <= COL_BLOCKING_THRESHOLD:
        if n_cols in TOKEN_BLOCK_SIZE_TABLE:
            return TOKEN_BLOCK_SIZE_TABLE[n_cols]
        for dim_thresh, block_size in sorted(TOKEN_BLOCK_SIZE_TABLE.items()):
            if n_cols <= dim_thresh:
                return block_size
        return 1
    return 4


def pick_block_n(n_cols):
    if n_cols > COL_BLOCKING_THRESHOLD:
        return 2048
    # NOT align(x, n_cols, 256): that returns 768 for n_cols=768, and
    # tl.arange needs a power of two.
    return triton.next_power_of_2(n_cols)


# ---------------------------------------------------------------- kernels ----

@triton.jit
def _fwd_body(
    X_ptr, Y_ptr, W_ptr, B_ptr, Mean_ptr, RSTD_ptr,
    stride_x_row, stride_y_row, n_rows, n_cols, eps,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for row_task_id in range(pid, num_row_tasks, num_programs):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows

        sum_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            x_chunk = tl.load(
                X_ptr + rows_off[:, None] * stride_x_row + cols_off[None, :], mask=block_mask, other=0.0
            ).to(tl.float32)

            sum_acc += tl.sum(x_chunk, axis=1)

        mean = sum_acc / n_cols

        tl.store(Mean_ptr + rows_off, mean, mask=rows_mask)

        var_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            x_chunk = tl.load(
                X_ptr + rows_off[:, None] * stride_x_row + cols_off[None, :], mask=block_mask, other=0.0
            ).to(tl.float32)

            x_centered = x_chunk - mean[:, None]

            var_acc += tl.sum(x_centered * x_centered, axis=1)

        var = var_acc / n_cols
        rstd = rsqrt(var + eps)

        tl.store(RSTD_ptr + rows_off, rstd, mask=rows_mask)

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            x_chunk = tl.load(
                X_ptr + rows_off[:, None] * stride_x_row + cols_off[None, :], mask=block_mask, other=0.0
            ).to(tl.float32)

            w_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0).to(tl.float32)
            b_chunk = tl.load(B_ptr + cols_off, mask=cols_mask, other=0.0).to(tl.float32)

            x_centered = x_chunk - mean[:, None]
            y_chunk = x_centered * rstd[:, None] * w_chunk[None, :] + b_chunk[None, :]

            tl.store(
                Y_ptr + rows_off[:, None] * stride_y_row + cols_off[None, :],
                y_chunk.to(Y_ptr.dtype.element_ty),
                mask=block_mask,
            )


# The decorator stack as shipped: heuristics commented out, a no-op autotune
# in its place. BLOCK_SIZE_M has no source of a value.
@libentry()
@triton.autotune(
    configs=[],
    key=["n_rows", "n_cols"],
)
@triton.jit
def _layernorm_fwd_kernel_repro(
    X_ptr, Y_ptr, W_ptr, B_ptr, Mean_ptr, RSTD_ptr,
    stride_x_row, stride_y_row, n_rows, n_cols, eps,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
):
    _fwd_body(
        X_ptr, Y_ptr, W_ptr, B_ptr, Mean_ptr, RSTD_ptr,
        stride_x_row, stride_y_row, n_rows, n_cols, eps,
        BLOCK_SIZE_N, BLOCK_SIZE_M,
    )


@libentry()
@triton.jit
def _layernorm_fwd_kernel_fixed(
    X_ptr, Y_ptr, W_ptr, B_ptr, Mean_ptr, RSTD_ptr,
    stride_x_row, stride_y_row, n_rows, n_cols, eps,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
):
    _fwd_body(
        X_ptr, Y_ptr, W_ptr, B_ptr, Mean_ptr, RSTD_ptr,
        stride_x_row, stride_y_row, n_rows, n_cols, eps,
        BLOCK_SIZE_N, BLOCK_SIZE_M,
    )



# ------------------------------------------------------------------ host ----

def pick_device():
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_device(0)
        return "npu"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        return "cuda"
    return "cpu"


def sync(device):
    if device == "npu":
        torch.npu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def num_programs():
    try:
        return triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    except Exception:
        return 48


def layernorm_npu(x, w, b, eps, variant, block_m=None, block_n=None, stats_fp32=True):
    """Inlined layernorm_infer_impl."""
    shape = x.shape
    dim = shape[-1]
    x_2d = x.reshape(-1, dim)
    n_rows, n_cols = x_2d.shape

    if block_n is None:
        block_n = pick_block_n(n_cols)

    y = torch.empty_like(x_2d)
    stats_dtype = torch.float32 if stats_fp32 else x.dtype
    mean = torch.empty(n_rows, dtype=stats_dtype, device=x.device)
    rstd = torch.empty(n_rows, dtype=stats_dtype, device=x.device)

    grid = (num_programs(),)

    args = (
        x_2d, y, w, b, mean, rstd,
        x_2d.stride(0), y.stride(0),
        n_rows, n_cols, eps,
    )

    if variant == "repro":
        # No BLOCK_SIZE_M, exactly like the shipped call site.
        _layernorm_fwd_kernel_repro[grid](*args, BLOCK_SIZE_N=block_n)
    else:
        if block_m is None:
            block_m = pick_block_m(n_cols)
        _layernorm_fwd_kernel_fixed[grid](
            *args, BLOCK_SIZE_N=block_n, BLOCK_SIZE_M=block_m
        )

    return y.reshape(*shape)


def layernorm_ref(x, w, b, eps):
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = xf.var(dim=-1, unbiased=False, keepdim=True)
    out = (xf - mean) * torch.rsqrt(var + eps) * w.float() + b.float()
    return out.to(x.dtype)


# ------------------------------------------------------------------ cases ----

def build_cases():
    cases = []
    for shape in LAYERNORM_SHAPES:
        for dtype_name in DTYPES:
            cases.append({"shape": shape, "dtype": dtype_name})
    return cases


def case_label(case):
    return "x={} dtype={}".format(tuple(case["shape"]), case["dtype"])


def case_footprint(case):
    elems = 1
    for d in case["shape"]:
        elems *= d
    return 3 * elems * torch.empty(0, dtype=DTYPES[case["dtype"]]).element_size()


def make_inputs(case, device, weight_dtype):
    shape = tuple(case["shape"])
    dtype = DTYPES[case["dtype"]]
    norm_size = shape[-1]

    x = torch.randn(shape, dtype=dtype, device=device)
    # test_layernorm.py draws weight/bias in bf16 then copies into an fp32
    # parameter, so the kernel always sees fp32 here.
    w = torch.randn(norm_size, dtype=weight_dtype, device=device).to(torch.float32)
    b = torch.randn(norm_size, dtype=weight_dtype, device=device).to(torch.float32)

    return x, w, b


def run_one(case, args, device):
    n_cols = case["shape"][-1]
    block_n = args.block_n if args.block_n else pick_block_n(n_cols)
    block_m = args.block_m if args.block_m else pick_block_m(n_cols)

    print("--- {} [{}] variant={} BLOCK_SIZE_N={} BLOCK_SIZE_M={}".format(
        case_label(case), device, args.variant, block_n,
        "<unbound>" if args.variant == "repro" else block_m))

    x, w, b = make_inputs(case, device, DTYPES[args.weight_dtype])

    torch.manual_seed(42)
    y = layernorm_npu(x, w, b, args.eps, args.variant,
                      block_m=args.block_m, block_n=args.block_n,
                      stats_fp32=not args.stats_input_dtype)
    sync(device)

    if args.mode in ("perf", "both"):
        for _ in range(args.warmup):
            layernorm_npu(x, w, b, args.eps, args.variant,
                          block_m=args.block_m, block_n=args.block_n,
                          stats_fp32=not args.stats_input_dtype)
        sync(device)

        start = time.perf_counter()
        for _ in range(args.iters):
            layernorm_npu(x, w, b, args.eps, args.variant,
                          block_m=args.block_m, block_n=args.block_n,
                          stats_fp32=not args.stats_input_dtype)
        sync(device)
        elapsed = time.perf_counter() - start
        print("    perf: {:.3f} ms/iter over {} iters".format(1e3 * elapsed / args.iters, args.iters))

    if args.mode in ("acc", "both"):
        ref = layernorm_ref(x, w, b, args.eps)
        sync(device)

        if x.dtype == torch.float32:
            atol, rtol = 1e-4, 1e-5
        else:
            atol, rtol = 5e-2, 1e-2

        diff = (y.float() - ref.float()).abs()
        tol = atol + rtol * ref.float().abs()
        bad = int((diff > tol).sum().item())
        max_abs = float(diff.max().item())

        if bad:
            raise AssertionError(
                "acc: {} / {} elements over tol (atol={} rtol={}), max|diff|={:.3e}".format(
                    bad, y.numel(), atol, rtol, max_abs))
        print("    acc: ok, max|diff|={:.3e} (atol={} rtol={})".format(max_abs, atol, rtol))


def run_isolated(index, args):
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--case", str(index),
        "--inproc",
        "--variant", args.variant,
        "--mode", args.mode,
        "--eps", repr(args.eps),
        "--iters", str(args.iters),
        "--warmup", str(args.warmup),
        "--weight-dtype", args.weight_dtype,
    ]
    if args.block_m:
        cmd += ["--block-m", str(args.block_m)]
    if args.block_n:
        cmd += ["--block-n", str(args.block_n)]
    if args.stats_input_dtype:
        cmd.append("--stats-input-dtype")
    if args.include_huge:
        cmd.append("--include-huge")

    return subprocess.run(cmd).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="print the case table and exit")
    parser.add_argument("--case", type=int, default=None, help="run a single case by index")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default=None, help="only run this dtype")
    parser.add_argument("--variant", choices=["repro", "fixed"], default="fixed",
                        help="repro = BLOCK_SIZE_M unbound (the bug); fixed = passed explicitly")
    parser.add_argument("--mode", choices=["perf", "acc", "both"], default="both")
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--block-m", type=int, default=None, help="force BLOCK_SIZE_M (fixed variant)")
    parser.add_argument("--block-n", type=int, default=None, help="force BLOCK_SIZE_N")
    parser.add_argument("--weight-dtype", choices=sorted(DTYPES), default="bfloat16",
                        help="dtype weight/bias are drawn in before the fp32 cast")
    parser.add_argument("--stats-input-dtype", action="store_true",
                        help="allocate mean/rstd as x.dtype instead of fp32 (the shipped behaviour)")
    parser.add_argument("--inproc", action="store_true", help="run in this process instead of forking per case")
    parser.add_argument("--include-huge", action="store_true",
                        help="also run cases whose x+y+ref footprint exceeds the 2 GiB device window")
    args = parser.parse_args()

    cases = build_cases()

    if args.list:
        for i, case in enumerate(cases):
            n_cols = case["shape"][-1]
            mib = case_footprint(case) / (1024.0 * 1024.0)
            flag = "  (over window)" if case_footprint(case) >= MEM_WINDOW_BYTES else ""
            print("{:3d}  {:<34} BLOCK_N={:5d} BLOCK_M={:3d}  ~{:8.1f} MiB{}".format(
                i, case_label(case), pick_block_n(n_cols), pick_block_m(n_cols), mib, flag))
        return 0

    selected = list(enumerate(cases))
    if args.case is not None:
        selected = [(args.case, cases[args.case])]
    if args.dtype is not None:
        selected = [(i, c) for i, c in selected if c["dtype"] == args.dtype]

    if args.inproc:
        device = pick_device()
        failures = 0
        for _, case in selected:
            try:
                run_one(case, args, device)
            except Exception:
                traceback.print_exc()
                failures += 1
        return 1 if failures else 0

    passed, failed, skipped = [], [], []
    for index, case in selected:
        if not args.include_huge and case_footprint(case) >= MEM_WINDOW_BYTES:
            print("--- {} SKIP (needs ~{:.1f} MiB, over the 2 GiB window; --include-huge to force)".format(
                case_label(case), case_footprint(case) / (1024.0 * 1024.0)))
            skipped.append(case_label(case))
            continue

        rc = run_isolated(index, args)
        if rc == 0:
            print("    PASS")
            passed.append(case_label(case))
        else:
            print("    FAIL (exit {})".format(rc))
            failed.append((case_label(case), rc))

    print("\n==== summary: {} passed, {} failed, {} skipped ====".format(
        len(passed), len(failed), len(skipped)))
    for label, rc in failed:
        print("  FAIL  {}  (exit {})".format(label, rc))

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
