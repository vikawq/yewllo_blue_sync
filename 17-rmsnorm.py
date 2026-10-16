"""Algo #17: rmsnorm standalone e2e 验证脚本。

严格对齐生产路径:
    mojo_opset/tests/specific_shape_perf_acc/test_rmsnorm_950_perf.py
      -> MojoRMSNorm -> TTXRMSNorm.forward
      -> mojo_opset/backends/ttx/kernels/npu/rmsnorm.py :: rmsnorm_infer_impl
      -> _rmsnorm_infer_kernel_single   (n_cols <= 10240, 绝大多数 shape)
         或 _rmsnorm_infer_kernel       (n_cols >  10240)

对齐要点(与上一版脚本的差异):
  1. **分发走错了 kernel**。rmsnorm_infer_impl 里
         BLOCK_SIZE_N = align(x, n_cols, VEC_ALIGN_BYTES)   # 向上取整
         if BLOCK_SIZE_N < n_cols: _rmsnorm_infer_kernel
         else:                     _rmsnorm_infer_kernel_single
     align() 是向上取整, 所以 BLOCK_SIZE_N >= n_cols 恒成立, 只有
     n_cols > COL_BLOCKING_THRESHOLD(10240) 把 BLOCK_SIZE_N 钳到 10240 时
     才会 < n_cols。也就是说: n_cols <= 10240 一律走 _rmsnorm_infer_kernel_single
     (单次整行 load, W 在 row 循环外只 load 一次, BLOCK_SIZE_M 由 autotune 决定);
     上一版脚本却一直在跑 _rmsnorm_infer_kernel + 表格 BLOCK_SIZE_M, 这是生产里
     根本不会出现的组合。
  2. weight 恒为 float32 —— MojoRMSNorm 的 nn.Parameter 用默认 dtype 创建,
     与 x.dtype 无关。
  3. 两个 kernel body 与生产代码逐字一致。
  4. 参考值与容差照抄 test_rmsnorm_950_perf.py::test_rmsnorm_accuracy:
     参考实现 = MojoRMSNorm._registry.get("torch") = F.rms_norm;
     fp32: atol=1e-5, rtol=1e-6; fp16/bf16: atol=3e-2, rtol=6e-3; ptol=1.0。

与生产不同、且无法在 standalone 下照搬的两处(都已显式打印出来):
  * grid: 生产用 get_device_properties("npu")["num_vectorcore"], 这里用 aicore_num。
    kernel 是 grid-stride 循环, 核数只影响每核的 row task 数, 不影响结果。
  * autotune: 生产对 _rmsnorm_infer_kernel_single 在
        BLOCK_SIZE_M in [1,2,4,8,16,32,64,128,256] x multibuffer x enable_vf_fusion
    (36 组) 上 autotune, 对 _rmsnorm_infer_kernel 在
        BLOCK_SIZE_N in [2048,4096,6144,8192,10240]
    上 autotune。仿真器下跑完整 autotune 不现实, 这里取单组配置, 可用环境变量
        MOJO_RMSNORM_BM / MOJO_RMSNORM_BN / MOJO_RMSNORM_MULTIBUFFER /
        MOJO_RMSNORM_VF_FUSION
    逐组扫, 用来定位"是哪一组配置编译出问题"。
"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")

import standalone_runtime


def ensure_blas_shim_dtype(dtype):
    if dtype == torch.bfloat16 and "BLAS_SHIM_DTYPE" not in os.environ:
        os.environ["BLAS_SHIM_DTYPE"] = "bfloat16"
        print("[env] bf16 detected -> BLAS_SHIM_DTYPE=bfloat16")


# ---- 以下常量/启发式/kernel 照抄 kernels/npu/rmsnorm.py + kernels/utils.py ----
VEC_ALIGN_BYTES = 256
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

# _rmsnorm_infer_kernel 的 autotune 候选 (triton.Config({"BLOCK_SIZE_N": ...}))
LOOP_KERNEL_BLOCK_N_CONFIGS = [2048, 4096, 8192, 6144, 10240]
# _rmsnorm_infer_kernel_single 的 autotune 候选 (BLOCK_SIZE_M x multibuffer x enable_vf_fusion)
SINGLE_KERNEL_BLOCK_M_CONFIGS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def ceil_div(x, y):
    return (x + y - 1) // y


def align(x, n_cols, aligned_bytes):
    aligned_bytes = ceil_div(x.element_size() * n_cols, aligned_bytes) * aligned_bytes
    return aligned_bytes // x.element_size()


def rms_norm_fwd_heuristics(n_cols):
    hidden_dim = n_cols
    if hidden_dim <= COL_BLOCKING_THRESHOLD:
        if hidden_dim in TOKEN_BLOCK_SIZE_TABLE:
            return TOKEN_BLOCK_SIZE_TABLE[hidden_dim]

        for dim_thresh, block_size in sorted(TOKEN_BLOCK_SIZE_TABLE.items()):
            if hidden_dim <= dim_thresh:
                return block_size
        return 1
    else:
        return 1


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


def _env_int(name, default):
    v = os.environ.get(name)
    return default if v in (None, "") else int(v)


def _env_bool(name, default):
    v = os.environ.get(name)
    if v in (None, ""):
        return default
    return v.lower() in ("1", "true", "yes", "on")


def _pick_single_block_m(n_cols, itemsize):
    """autotune 的确定性替身: 取 UB 单份占用 <= 32KB 的最大合法 BLOCK_SIZE_M。"""
    override = _env_int("MOJO_RMSNORM_BM", 0)
    if override:
        return override
    budget = 32 * 1024
    best = 1
    for bm in SINGLE_KERNEL_BLOCK_M_CONFIGS:
        if bm * n_cols * itemsize <= budget:
            best = bm
    return best


def rmsnorm_infer_impl(x, w, eps, aicore_num):
    """照抄 kernels/npu/rmsnorm.py::rmsnorm_infer_impl, 只把 grid / autotune 换成显式配置。"""
    assert x.size(-1) == w.size(-1)
    shape = x.shape
    dim = shape[-1]
    X_2d = x.reshape(-1, dim)
    n_rows, n_cols = X_2d.shape

    y = torch.empty_like(X_2d)

    if n_cols > COL_BLOCKING_THRESHOLD:
        BLOCK_SIZE_N = COL_BLOCKING_THRESHOLD
    else:
        BLOCK_SIZE_N = align(x, n_cols, VEC_ALIGN_BYTES)

    grid = (aicore_num,)

    if BLOCK_SIZE_N < n_cols:
        # 只有 n_cols > 10240 才会走到这里; BLOCK_SIZE_M 来自 heuristics,
        # BLOCK_SIZE_N 来自 autotune 候选表。
        block_m = rms_norm_fwd_heuristics(n_cols)
        block_n = _env_int("MOJO_RMSNORM_BN", LOOP_KERNEL_BLOCK_N_CONFIGS[0])
        print(
            f"  [cfg] kernel=_rmsnorm_infer_kernel n_rows={n_rows} n_cols={n_cols} "
            f"BLOCK_SIZE_M={block_m} BLOCK_SIZE_N={block_n} grid={grid} "
            f"x={X_2d.dtype} w={w.dtype}"
        )
        _rmsnorm_infer_kernel[grid](
            x,
            y,
            w,
            X_2d.stride(0),
            y.stride(0),
            n_rows=n_rows,
            n_cols=n_cols,
            eps=eps,
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            compile_on_910_95=True,
        )
    else:
        block_m = _pick_single_block_m(n_cols, X_2d.element_size())
        multibuffer = _env_bool("MOJO_RMSNORM_MULTIBUFFER", True)
        vf_fusion = _env_bool("MOJO_RMSNORM_VF_FUSION", True)
        print(
            f"  [cfg] kernel=_rmsnorm_infer_kernel_single n_rows={n_rows} n_cols={n_cols} "
            f"BLOCK_SIZE_M={block_m} BLOCK_SIZE_N={n_cols} (align->{BLOCK_SIZE_N}) grid={grid} "
            f"multibuffer={multibuffer} enable_vf_fusion={vf_fusion} "
            f"x={X_2d.dtype} w={w.dtype}"
        )
        _rmsnorm_infer_kernel_single[grid](
            x,
            y,
            w,
            X_2d.stride(0),
            y.stride(0),
            n_rows=n_rows,
            n_cols=n_cols,
            eps=eps,
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=n_cols,
            multibuffer=multibuffer,
            enable_vf_fusion=vf_fusion,
            compile_on_910_95=True,
        )

    return y.reshape(*shape)


def rmsnorm_reference(x, w, eps):
    """= MojoRMSNorm._registry.get("torch")().forward, 即 F.rms_norm (w 为 fp32)。"""
    return F.rms_norm(x, [x.shape[-1]], weight=w, eps=eps)


def tolerances(dtype):
    # 照抄 test_rmsnorm_950_perf.py::test_rmsnorm_accuracy
    if dtype == torch.float32:
        return 1e-5, 1e-6
    return 3e-2, 6e-3


def run_one(shape, dtype, eps, aicore_num, timeout):
    print(f"\n{'=' * 70}")
    print(f"[shape] {shape} dtype={dtype} eps={eps} aicore={aicore_num}")
    ensure_blas_shim_dtype(dtype)

    torch.manual_seed(42)
    # 与原测试一致: x 为被测 dtype, weight 恒为 fp32 参数。
    x = torch.randn(size=shape, dtype=dtype)
    w = torch.randn(size=(shape[-1],), dtype=torch.float32)

    atol, rtol = tolerances(dtype)

    t_start = time.perf_counter()
    with standalone_runtime.enable(timeout=timeout) as runtime:
        y = rmsnorm_infer_impl(x, w, eps, aicore_num)
        standalone_runtime.synchronize()
        wall_s = time.perf_counter() - t_start

        y_ref = rmsnorm_reference(x, w, eps)
        err = (y.double() - y_ref.double()).abs().max().item()
        print(f"  y.sum()={y.double().sum().item():.4f}  ref.sum()={y_ref.double().sum().item():.4f}")
        print(f"  max_abs_error={err:.6e} (atol={atol} rtol={rtol})")
        torch.testing.assert_close(y, y_ref, rtol=rtol, atol=atol)
        print("  forward diff: Pass! ✅")
        print(f"  [time] kernel={wall_s:.2f}s")
        return {
            "passed": True,
            "shape": str(shape),
            "dtype": str(dtype),
            "atol": atol,
            "rtol": rtol,
            "max_abs_error": err,
            "kernel_wall_seconds": round(wall_s, 3),
            "sim_ticks": runtime.last_report.get("ticks"),
            "runtime_report": runtime.last_report,
        }


# 取 test_rmsnorm_950_perf.py shape 列表里 standalone 仿真可承受的几项。
# 全部走 _rmsnorm_infer_kernel_single —— 这与生产分发一致。
# (1, 77, 489) 特意保留: n_cols=489 是奇数、非 2 的幂, BLOCK_SIZE_N 直接等于 489,
# 是最容易暴露向量对齐问题的生产配置。
DEFAULT_SHAPES = [(1, 2, 256), (1, 77, 489), (1, 32, 1024), (1, 32, 2048)]
# 想额外覆盖 _rmsnorm_infer_kernel(循环版, 生产里只有 n_cols > 10240 才走) 时加上
# 环境变量 MOJO_RMSNORM_LOOP=1, 会追加一个 n_cols=12288 的 shape。
LOOP_KERNEL_SHAPES = [(1, 4, 12288)]
DEFAULT_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

_DTYPE_BY_NAME = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    dtypes = [_DTYPE_BY_NAME[a] for a in argv if a in _DTYPE_BY_NAME] or DEFAULT_DTYPES
    shapes = list(DEFAULT_SHAPES)
    if _env_bool("MOJO_RMSNORM_LOOP", False):
        shapes += LOOP_KERNEL_SHAPES
    eps = 1e-5

    t_total = time.perf_counter()
    results = []
    for dtype in dtypes:
        for shape in shapes:
            try:
                results.append(run_one(shape, dtype, eps, aicore_num=8, timeout=3600))
            except Exception as e:  # noqa: BLE001 - 单个 shape 失败不中断整轮
                results.append(
                    {"passed": False, "shape": str(shape), "dtype": str(dtype),
                     "error": f"{type(e).__name__}: {e}"}
                )
                print(f"  ERROR: {type(e).__name__}: {e}")

    total_s = time.perf_counter() - t_total
    print(f"\n{'=' * 70}\nSummary:")
    passed = sum(1 for r in results if r.get("passed"))
    for r in results:
        status = "PASS" if r.get("passed") else "FAIL"
        print(f"  [{status}] shape={r['shape']} dtype={r['dtype']}")
    print(f"  [{passed}/{len(results)} passed, total={total_s:.2f}s]")
    print(json.dumps(results, indent=2, default=str))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
