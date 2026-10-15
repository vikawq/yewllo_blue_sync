#!/usr/bin/env python3
"""Standalone driver for the RMSNorm NPU kernel (no pytest).

Mirrors mojo_opset/tests/specific_shape_perf_acc/test_rmsnorm_950_perf.py, but
each case runs in its own subprocess by default, so a device-level abort (bus
error, out-of-range access, simulator assert) only kills that one case and the
remaining shapes still get exercised.

    python test_rmsnorm_950_standalone.py --list
    python test_rmsnorm_950_standalone.py                        # every case, isolated
    python test_rmsnorm_950_standalone.py --dtype float32        # only fp32
    python test_rmsnorm_950_standalone.py --case 0 --inproc      # one case, no fork
    python test_rmsnorm_950_standalone.py --mode acc             # accuracy only

Note which shapes have a non-power-of-2 hidden size: 7338, 489, 8777 and 18778
all land on the `_rmsnorm_infer_kernel_single` path with BLOCK_SIZE_N = n_cols,
so they are the ones to watch for tl.arange / tiling complaints.
"""

import argparse
import os
import subprocess
import sys
import time
import traceback

import torch

# Same list as test_rmsnorm_950_perf.py.
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

# The standalone memory window is [0, 0x80000000). x, y and the torch reference
# all have to live in it at once, so anything near that is reported, not run.
MEM_WINDOW_BYTES = 0x80000000


def build_cases():
    cases = []
    for shape in RMSNORM_SHAPES:
        for dtype_name in DTYPES:
            cases.append({"shape": shape, "dtype": dtype_name})
    return cases


def case_label(case):
    n_cols = case["shape"][-1]
    pow2 = (n_cols & (n_cols - 1)) == 0
    return "x={} dtype={}{}".format(tuple(case["shape"]), case["dtype"], "" if pow2 else " [n_cols not pow2]")


def case_footprint(case):
    """Bytes of device memory the case needs for x + y + reference output."""
    elems = 1
    for d in case["shape"]:
        elems *= d
    return 3 * elems * torch.empty(0, dtype=DTYPES[case["dtype"]]).element_size()


def setup_device():
    from mojo_opset.utils.platform import get_platform, get_torch_device

    platform = get_platform()
    torch.set_default_device(get_torch_device())

    if platform == "npu":
        torch.npu.set_device(0)
    elif platform == "mlu":
        torch.mlu.set_device(0)
    elif platform == "ilu":
        torch.cuda.set_device(0)

    return platform


def make_inputs(case, weight_dtype):
    shape = tuple(case["shape"])
    dtype = DTYPES[case["dtype"]]
    norm_size = shape[-1]

    x = torch.randn(size=shape, dtype=dtype)
    weight = torch.nn.Parameter(torch.randn(size=(norm_size,), dtype=weight_dtype))

    return x, weight


def make_op(norm_size, eps, device, backend=None):
    from mojo_opset import MojoRMSNorm

    if backend is None:
        op = MojoRMSNorm(norm_size, eps)
    else:
        op = MojoRMSNorm._registry.get(backend)(norm_size, eps)

    return op.to(device)


def load_params(op, weight):
    with torch.no_grad():
        op.weight.copy_(weight.to(torch.float32))


def sync(platform):
    if platform == "npu":
        torch.npu.synchronize()
    elif platform == "mlu":
        torch.mlu.synchronize()
    elif platform == "ilu":
        torch.cuda.synchronize()


def run_perf(case, eps, weight_dtype, platform, iters, warmup):
    x, weight = make_inputs(case, weight_dtype)
    op = make_op(weight.size(0), eps, x.device)
    load_params(op, weight)

    for _ in range(warmup):
        op(x)
    sync(platform)

    start = time.perf_counter()
    for _ in range(iters):
        op(x)
    sync(platform)
    elapsed = time.perf_counter() - start

    print("    perf: {:.3f} ms/iter over {} iters".format(1e3 * elapsed / iters, iters))


def run_acc(case, eps, weight_dtype, platform):
    x, weight = make_inputs(case, weight_dtype)

    op = make_op(weight.size(0), eps, x.device)
    load_params(op, weight)

    ref = make_op(weight.size(0), eps, x.device, backend="torch")
    load_params(ref, weight)

    if x.dtype == torch.float32:
        atol, rtol = 1e-5, 1e-6
    else:
        atol, rtol = 3e-2, 6e-3

    op.forward_diff_with(ref, x, atol=atol, rtol=rtol)
    sync(platform)
    print("    acc: match within atol={} rtol={}".format(atol, rtol))


def run_one(case, args):
    platform = setup_device()
    print("--- {} [{}]".format(case_label(case), platform))

    if args.mode in ("perf", "both"):
        run_perf(case, args.eps, DTYPES[args.weight_dtype], platform, args.iters, args.warmup)
    if args.mode in ("acc", "both"):
        run_acc(case, args.eps, DTYPES[args.weight_dtype], platform)


def run_isolated(index, args):
    """Re-exec this script for a single case so a device abort stays contained."""
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--case", str(index),
        "--inproc",
        "--mode", args.mode,
        "--eps", repr(args.eps),
        "--iters", str(args.iters),
        "--warmup", str(args.warmup),
        "--weight-dtype", args.weight_dtype,
    ]
    if args.include_huge:
        cmd.append("--include-huge")

    completed = subprocess.run(cmd)
    return completed.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="print the case table and exit")
    parser.add_argument("--case", type=int, default=None, help="run a single case by index")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default=None, help="only run this dtype")
    parser.add_argument("--mode", choices=["perf", "acc", "both"], default="both")
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--weight-dtype", choices=sorted(DTYPES), default="float32",
                        help="dtype the weight is drawn in before the fp32 copy_ (matches test_rmsnorm_950_perf.py)")
    parser.add_argument("--inproc", action="store_true", help="run in this process instead of forking per case")
    parser.add_argument("--include-huge", action="store_true",
                        help="also run cases whose x+y+ref footprint exceeds the 2 GiB device window")
    args = parser.parse_args()

    cases = build_cases()

    if args.list:
        for i, case in enumerate(cases):
            mib = case_footprint(case) / (1024.0 * 1024.0)
            flag = "  (over window)" if case_footprint(case) >= MEM_WINDOW_BYTES else ""
            print("{:3d}  {:<48} ~{:8.1f} MiB{}".format(i, case_label(case), mib, flag))
        return 0

    selected = list(enumerate(cases))
    if args.case is not None:
        selected = [(args.case, cases[args.case])]
    if args.dtype is not None:
        selected = [(i, c) for i, c in selected if c["dtype"] == args.dtype]

    if args.inproc:
        failures = 0
        for _, case in selected:
            try:
                run_one(case, args)
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
