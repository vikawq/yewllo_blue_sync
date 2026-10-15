#!/usr/bin/env python3
"""Standalone driver for the LayerNorm NPU kernel (no pytest).

Mirrors mojo_opset/tests/specific_shape_perf_acc/test_layernorm.py, but each
case runs in its own subprocess by default, so a device-level abort (bus
error, out-of-range access, simulator assert) only kills that one case and the
remaining shapes still get exercised.

    python test_layernorm_standalone.py --list
    python test_layernorm_standalone.py                        # every case, isolated
    python test_layernorm_standalone.py --dtype float32        # only fp32
    python test_layernorm_standalone.py --case 0 --inproc      # one case, no fork
    python test_layernorm_standalone.py --mode acc             # accuracy only

Use --inproc on the case you want to inspect: the kernel then runs in this
process, so simulator / msdebug / gdb output is attributable to one launch.
"""

import argparse
import os
import subprocess
import sys
import time
import traceback

import torch

# Same list as test_layernorm.py, plus the 3-D shape that showed the fp32 fault.
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

# The standalone memory window is [0, 0x80000000). x, y and the torch reference
# all have to live in it at once, so anything near that is reported, not run.
MEM_WINDOW_BYTES = 0x80000000


def build_cases():
    cases = []
    for shape in LAYERNORM_SHAPES:
        for dtype_name in DTYPES:
            cases.append({"shape": shape, "dtype": dtype_name})
    return cases


def case_label(case):
    return "x={} dtype={}".format(tuple(case["shape"]), case["dtype"])


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
    weight = torch.randn(size=(norm_size,), dtype=weight_dtype)
    bias = torch.randn(size=(norm_size,), dtype=weight_dtype)

    return x, weight, bias


def make_op(norm_size, eps, device, backend=None):
    from mojo_opset import MojoLayerNorm

    if backend is None:
        op = MojoLayerNorm(norm_size=norm_size, eps=eps)
    else:
        op = MojoLayerNorm._registry.get(backend)(norm_size=norm_size, eps=eps)

    return op.to(device)


def load_params(op, weight, bias):
    with torch.no_grad():
        op.weight.copy_(weight.to(torch.float32))
        op.bias.copy_(bias.to(torch.float32))


def sync(platform):
    if platform == "npu":
        torch.npu.synchronize()
    elif platform == "mlu":
        torch.mlu.synchronize()
    elif platform == "ilu":
        torch.cuda.synchronize()


def run_perf(case, eps, weight_dtype, platform, iters, warmup):
    x, weight, bias = make_inputs(case, weight_dtype)
    op = make_op(weight.size(0), eps, x.device)
    load_params(op, weight, bias)

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
    x, weight, bias = make_inputs(case, weight_dtype)

    op = make_op(weight.size(0), eps, x.device)
    load_params(op, weight, bias)

    ref = make_op(weight.size(0), eps, x.device, backend="torch")
    load_params(ref, weight, bias)

    if x.dtype == torch.float32:
        atol, rtol = 1e-4, 1e-5
    else:
        atol, rtol = 5e-2, 1e-2

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
    parser.add_argument("--weight-dtype", choices=sorted(DTYPES), default="bfloat16",
                        help="dtype the weight/bias are drawn in before the fp32 copy_ (matches test_layernorm.py)")
    parser.add_argument("--inproc", action="store_true", help="run in this process instead of forking per case")
    parser.add_argument("--include-huge", action="store_true",
                        help="also run cases whose x+y+ref footprint exceeds the 2 GiB device window")
    args = parser.parse_args()

    cases = build_cases()

    if args.list:
        for i, case in enumerate(cases):
            mib = case_footprint(case) / (1024.0 * 1024.0)
            flag = "  (over window)" if case_footprint(case) >= MEM_WINDOW_BYTES else ""
            print("{:3d}  {:<34} ~{:8.1f} MiB{}".format(i, case_label(case), mib, flag))
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
