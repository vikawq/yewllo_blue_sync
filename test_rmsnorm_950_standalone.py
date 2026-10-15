#!/usr/bin/env python3
"""Self-contained RMSNorm NPU kernel repro. Needs only torch / torch_npu / triton.

Does NOT import mojo_opset -- both kernels are inlined below, copied from
mojo_opset/backends/ttx/kernels/npu/rmsnorm.py, so you can scp this one file
anywhere and run it.

rmsnorm_infer_impl picks between two kernels on BLOCK_SIZE_N vs n_cols:

  _rmsnorm_infer_kernel_single   BLOCK_SIZE_N >= n_cols, whole row in one tile
  _rmsnorm_infer_kernel          BLOCK_SIZE_N <  n_cols, loops over columns

Unlike layernorm, BLOCK_SIZE_M here does get a value, so this is mostly about
the non-power-of-2 hidden sizes in the 950 shape list -- 7338, 489, 8777 and
18778 -- which reach tl.arange(0, BLOCK_SIZE_N) with BLOCK_SIZE_N = n_cols.
--block-n-pow2 rounds BLOCK_SIZE_N up to a power of two to tell a tiling
complaint apart from a real kernel bug.

Typical session:

    python3 test_rmsnorm_950_standalone.py --list
    python3 test_rmsnorm_950_standalone.py --dtype float32
    python3 test_rmsnorm_950_standalone.py --dtype float32 --block-n-pow2
    python3 test_rmsnorm_950_standalone.py --case 3 --inproc

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
    from triton.runtime.libentry import libentry
except ImportError:
    def libentry():
        def _decorator(fn):
            return fn
        return _decorator


# ---------------------------------------------------------------- shapes ----

RMSNORM_SHAPES = [
    (1, 32, 2048),
    (1, 32, 1024),
    (1, 64, 8192),
    (1, 57, 7338),
    (1, 77, 489),
    (1, 2, 256),
    (1, 763, 8777),
    (1, 7762, 18778),
]

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

MEM_WINDOW_BYTES = 0x80000000


# ------------------------------------------------- host-side block sizes ----

COL_BLOCKING_THRESHOLD = 10240

TOKEN_BLOCK_SIZE_TABLE = {
    10240: 2,
    8192: 2,
    4096: 6,
    2048: 8,
    1024: 10,
    512: 18,
    256: 24,
    128: 48,
}

VEC_ALIGN_BYTES = 256


def ceil_div(x, y):
    return (x + y - 1) // y


def align(element_size, n_cols, aligned_bytes):
    """The shipped align(): byte-aligned, so it can return a non-power-of-2."""
    total = ceil_div(element_size * n_cols, aligned_bytes) * aligned_bytes
    return total // element_size


def pick_block_m(n_cols):
    if n_cols <= COL_BLOCKING_THRESHOLD:
        if n_cols in TOKEN_BLOCK_SIZE_TABLE:
            return TOKEN_BLOCK_SIZE_TABLE[n_cols]
        for dim_thresh, block_size in sorted(TOKEN_BLOCK_SIZE_TABLE.items()):
            if n_cols <= dim_thresh:
                return block_size
        return 1
    return 1


def pick_block_n(element_size, n_cols, pow2=False):
    if n_cols > COL_BLOCKING_THRESHOLD:
        block_n = COL_BLOCKING_THRESHOLD
    else:
        block_n = align(element_size, n_cols, VEC_ALIGN_BYTES)
    if pow2:
        block_n = triton.next_power_of_2(block_n)
    return block_n


# ---------------------------------------------------------------- kernels ----

@triton.heuristics({"BLOCK_SIZE_M": lambda args: pick_block_m(args["n_cols"])})
@libentry()
@triton.jit
def _rmsnorm_infer_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    stride_x_row,
    stride_y_row,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M

        current_row_offsets = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        row_mask = current_row_offsets < n_rows

        ss_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            col_offsets = col_offset + tl.arange(0, BLOCK_SIZE_N)
            col_mask = col_offsets < n_cols

            x_ptrs = X_ptr + (current_row_offsets[:, None] * stride_x_row + col_offsets[None, :])

            x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)

            ss_acc += tl.sum(x * x, axis=1)

        ss_acc = tl.where(row_mask, ss_acc, 0)

        mean_square = ss_acc / n_cols
        rrms = tl.rsqrt(mean_square + eps)

        rrms = tl.where(row_mask, rrms, 0.0)

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            col_offsets = col_offset + tl.arange(0, BLOCK_SIZE_N)
            col_mask = col_offsets < n_cols

            x_ptrs = X_ptr + (current_row_offsets[:, None] * stride_x_row + col_offsets[None, :])
            w_ptrs = W_ptr + col_offsets
            y_ptrs = Y_ptr + (current_row_offsets[:, None] * stride_y_row + col_offsets[None, :])

            x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=col_mask, other=0.0)

            x_f32 = x.to(tl.float32)
            w_f32 = w.to(tl.float32)

            x_normalized = x_f32 * rrms[:, None]

            y = x_normalized * w_f32[None, :]

            tl.store(
                y_ptrs,
                y.to(Y_ptr.dtype.element_ty),
                mask=row_mask[:, None] & col_mask[None, :],
            )


@triton.heuristics({"BLOCK_SIZE_M": lambda args: pick_block_m(args["n_cols"])})
@libentry()
@triton.jit
def _rmsnorm_infer_kernel_single(
    X_ptr,
    Y_ptr,
    W_ptr,
    stride_x_row,
    stride_y_row,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    col_mask = col_offsets < n_cols

    w_ptrs = W_ptr + col_offsets
    w = tl.load(w_ptrs, mask=col_mask, other=0.0)
    w_f32 = w.to(tl.float32)

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M

        current_row_offsets = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        row_mask = current_row_offsets < n_rows

        x_ptrs = X_ptr + (current_row_offsets[:, None] * stride_x_row + col_offsets[None, :])

        x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)

        ss_acc = tl.sum(x * x, axis=1)

        ss_acc = tl.where(row_mask, ss_acc, 0)

        mean_square = ss_acc / n_cols
        rrms = tl.rsqrt(mean_square + eps)

        rrms = tl.where(row_mask, rrms, 0.0)

        y_ptrs = Y_ptr + (current_row_offsets[:, None] * stride_y_row + col_offsets[None, :])

        x_normalized = x * rrms[:, None]

        y = x_normalized * w_f32[None, :]

        tl.store(
            y_ptrs,
            y.to(Y_ptr.dtype.element_ty),
            mask=row_mask[:, None] & col_mask[None, :],
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


def rmsnorm_npu(x, w, eps, block_n=None, block_n_pow2=False, force_blocked=False):
    """Inlined rmsnorm_infer_impl."""
    assert x.size(-1) == w.size(-1)
    shape = x.shape
    dim = shape[-1]
    x_2d = x.reshape(-1, dim)
    n_rows, n_cols = x_2d.shape

    y = torch.empty_like(x_2d)

    if block_n is None:
        block_n = pick_block_n(x.element_size(), n_cols, pow2=block_n_pow2)

    grid = (num_programs(),)

    use_blocked = force_blocked or block_n < n_cols

    if use_blocked:
        _rmsnorm_infer_kernel[grid](
            x_2d, y, w,
            x_2d.stride(0), y.stride(0),
            n_rows, n_cols, eps,
            BLOCK_SIZE_N=min(block_n, n_cols) if force_blocked else block_n,
        )
    else:
        _rmsnorm_infer_kernel_single[grid](
            x_2d, y, w,
            x_2d.stride(0), y.stride(0),
            n_rows, n_cols, eps,
            BLOCK_SIZE_N=block_n,
        )

    return y.reshape(*shape), ("blocked" if use_blocked else "single"), block_n


def rmsnorm_ref(x, w, eps):
    xf = x.float()
    ms = xf.pow(2).mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(ms + eps) * w.float()).to(x.dtype)


# ------------------------------------------------------------------ cases ----

def build_cases():
    cases = []
    for shape in RMSNORM_SHAPES:
        for dtype_name in DTYPES:
            cases.append({"shape": shape, "dtype": dtype_name})
    return cases


def case_label(case):
    n_cols = case["shape"][-1]
    pow2 = (n_cols & (n_cols - 1)) == 0
    return "x={} dtype={}{}".format(
        tuple(case["shape"]), case["dtype"], "" if pow2 else " [n_cols not pow2]")


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
    # test_rmsnorm_950_perf.py draws the weight then copies it into an fp32
    # parameter, so the kernel always sees fp32 here.
    w = torch.randn(norm_size, dtype=weight_dtype, device=device).to(torch.float32)

    return x, w


def run_one(case, args, device):
    x, w = make_inputs(case, device, DTYPES[args.weight_dtype])

    torch.manual_seed(42)
    y, path, block_n = rmsnorm_npu(x, w, args.eps, block_n=args.block_n,
                                   block_n_pow2=args.block_n_pow2,
                                   force_blocked=args.force_blocked)
    sync(device)

    print("--- {} [{}] path={} BLOCK_SIZE_N={} BLOCK_SIZE_M={}".format(
        case_label(case), device, path, block_n, pick_block_m(case["shape"][-1])))

    if args.mode in ("perf", "both"):
        for _ in range(args.warmup):
            rmsnorm_npu(x, w, args.eps, block_n=args.block_n,
                        block_n_pow2=args.block_n_pow2, force_blocked=args.force_blocked)
        sync(device)

        start = time.perf_counter()
        for _ in range(args.iters):
            rmsnorm_npu(x, w, args.eps, block_n=args.block_n,
                        block_n_pow2=args.block_n_pow2, force_blocked=args.force_blocked)
        sync(device)
        elapsed = time.perf_counter() - start
        print("    perf: {:.3f} ms/iter over {} iters".format(1e3 * elapsed / args.iters, args.iters))

    if args.mode in ("acc", "both"):
        ref = rmsnorm_ref(x, w, args.eps)
        sync(device)

        if x.dtype == torch.float32:
            atol, rtol = 1e-5, 1e-6
        else:
            atol, rtol = 3e-2, 6e-3

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
        "--mode", args.mode,
        "--eps", repr(args.eps),
        "--iters", str(args.iters),
        "--warmup", str(args.warmup),
        "--weight-dtype", args.weight_dtype,
    ]
    if args.block_n:
        cmd += ["--block-n", str(args.block_n)]
    if args.block_n_pow2:
        cmd.append("--block-n-pow2")
    if args.force_blocked:
        cmd.append("--force-blocked")
    if args.include_huge:
        cmd.append("--include-huge")

    return subprocess.run(cmd).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="print the case table and exit")
    parser.add_argument("--case", type=int, default=None, help="run a single case by index")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default=None, help="only run this dtype")
    parser.add_argument("--mode", choices=["perf", "acc", "both"], default="both")
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--block-n", type=int, default=None, help="force BLOCK_SIZE_N")
    parser.add_argument("--block-n-pow2", action="store_true",
                        help="round BLOCK_SIZE_N up to a power of two instead of using align()")
    parser.add_argument("--force-blocked", action="store_true",
                        help="use the column-looping kernel even when a row fits in one tile")
    parser.add_argument("--weight-dtype", choices=sorted(DTYPES), default="float32",
                        help="dtype the weight is drawn in before the fp32 cast")
    parser.add_argument("--inproc", action="store_true", help="run in this process instead of forking per case")
    parser.add_argument("--include-huge", action="store_true",
                        help="also run cases whose x+y+ref footprint exceeds the 2 GiB device window")
    args = parser.parse_args()

    cases = build_cases()

    if args.list:
        for i, case in enumerate(cases):
            n_cols = case["shape"][-1]
            esize = torch.empty(0, dtype=DTYPES[case["dtype"]]).element_size()
            block_n = pick_block_n(esize, n_cols)
            mib = case_footprint(case) / (1024.0 * 1024.0)
            flag = "  (over window)" if case_footprint(case) >= MEM_WINDOW_BYTES else ""
            path = "blocked" if block_n < n_cols else "single"
            print("{:3d}  {:<48} {:7s} BLOCK_N={:6d} BLOCK_M={:3d}  ~{:7.1f} MiB{}".format(
                i, case_label(case), path, block_n, pick_block_m(n_cols), mib, flag))
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
