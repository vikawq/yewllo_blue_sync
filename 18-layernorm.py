"""Algo #18: layernorm (_layernorm_fwd_kernel) standalone e2e 验证脚本。

严格对齐生产路径:
    mojo_opset/tests/specific_shape_perf_acc/test_layernorm.py
      -> MojoLayerNorm -> TTXLayerNorm.forward
      -> mojo_opset/backends/ttx/kernels/npu/layernorm.py :: layernorm_infer_impl
      -> _layernorm_fwd_kernel

对齐要点(与上一版脚本的差异, 都会影响 codegen/精度, 不要再改回去):
  1. BLOCK_SIZE_N = triton.next_power_of_2(n_cols) —— 前向用的是 next_power_of_2,
     只有 *反向* (layernorm_bwd_impl) 才用 align(x, n_cols, VEC_ALIGN_BYTES)。
     n_cols=768 时两者分别是 1024 / 768, 编译出来完全是两个 kernel。
  2. weight / bias 恒为 float32 —— 它们是 MojoLayerNorm 的 nn.Parameter, 用默认
     dtype 创建, 与 x.dtype 无关 (原测试只是拿 bf16 随机数 copy 进 fp32 参数)。
  3. kernel body 与生产代码逐字一致, 包含 variance 循环里那两条"死"的
     w_chunk/b_chunk load —— 删掉它们会改变编译器看到的 IR。
  4. Mean / RSTD 用 torch.empty (生产代码如此), 不是 torch.zeros。
  5. 参考值与容差照抄 test_layernorm_950_perf.py::test_layernorm_accuracy:
     参考实现 = MojoLayerNorm._registry.get("torch") = F.layer_norm;
     fp32: atol=1e-4, rtol=1e-5; fp16/bf16: atol=5e-2, rtol=1e-2; ptol=1.0 (全元素)。

唯一与生产不同的地方: grid。生产用
    triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"],
standalone 下核数由 aicore_num 给出(默认 8)。kernel 是 grid-stride 循环, 核数只影响
每核承担的 row task 数, 不影响单核 UB 占用, 也不影响计算结果。
"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from triton.language.math import rsqrt

os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")

import standalone_runtime


def ensure_blas_shim_dtype(dtype):
    if dtype == torch.bfloat16 and "BLAS_SHIM_DTYPE" not in os.environ:
        os.environ["BLAS_SHIM_DTYPE"] = "bfloat16"
        print("[env] bf16 detected -> BLAS_SHIM_DTYPE=bfloat16")


# ---- 以下常量/启发式照抄 kernels/npu/layernorm.py ----
COL_BLOCKING_THRESHOLD = 2048

TOKEN_BLOCK_SIZE_TABLE = {
    2048: 4,
    1024: 8,
    512: 10,
    256: 18,
    128: 24,
}


def layer_norm_fwd_heuristics(n_cols):
    hidden_dim = n_cols
    if hidden_dim <= COL_BLOCKING_THRESHOLD:
        if hidden_dim in TOKEN_BLOCK_SIZE_TABLE:
            return TOKEN_BLOCK_SIZE_TABLE[hidden_dim]

        for dim_thresh, block_size in sorted(TOKEN_BLOCK_SIZE_TABLE.items()):
            if hidden_dim <= dim_thresh:
                return block_size
        return 1
    else:
        return 4


@triton.jit
def _layernorm_fwd_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    B_ptr,
    Mean_ptr,
    RSTD_ptr,
    stride_x_row,
    stride_y_row,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
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

            w_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0).to(tl.float32)
            b_chunk = tl.load(B_ptr + cols_off, mask=cols_mask, other=0.0).to(tl.float32)

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


def layernorm_infer_impl(hidden_states, weight, bias, eps, aicore_num):
    """照抄 kernels/npu/layernorm.py::layernorm_infer_impl, 只把 grid 换成 aicore_num。"""
    shape = hidden_states.shape
    dim = shape[-1]
    x_2d = hidden_states.reshape(-1, dim)
    n_rows, n_cols = x_2d.shape

    if n_cols > COL_BLOCKING_THRESHOLD:
        BLOCK_SIZE_N = 2048
    else:
        BLOCK_SIZE_N = triton.next_power_of_2(n_cols)

    BLOCK_SIZE_M = layer_norm_fwd_heuristics(n_cols)

    # 仅用于定位 UB 容量问题: 不改默认行为, 想验证"fp32 下 BLOCK_SIZE_M 过大导致
    # UB 溢出"时设 MOJO_LAYERNORM_BM=12 之类再跑一遍。
    bm_override = os.environ.get("MOJO_LAYERNORM_BM")
    if bm_override:
        BLOCK_SIZE_M = int(bm_override)
        print(f"  [probe] BLOCK_SIZE_M overridden by MOJO_LAYERNORM_BM -> {BLOCK_SIZE_M} (非生产配置)")

    grid = (aicore_num,)

    y = torch.empty_like(x_2d)
    mean = torch.empty(n_rows, dtype=torch.float32, device=hidden_states.device)
    rstd = torch.empty(n_rows, dtype=torch.float32, device=hidden_states.device)

    print(
        f"  [cfg] n_rows={n_rows} n_cols={n_cols} BLOCK_SIZE_N={BLOCK_SIZE_N} "
        f"BLOCK_SIZE_M={BLOCK_SIZE_M} grid={grid} "
        f"x={x_2d.dtype} w={weight.dtype} b={bias.dtype}"
    )

    _layernorm_fwd_kernel[grid](
        x_2d,
        y,
        weight,
        bias,
        mean,
        rstd,
        x_2d.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        eps,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        compile_on_910_95=True,
    )

    return y.reshape(*shape)


def layernorm_reference(x, w, b, eps):
    """= MojoLayerNorm._registry.get("torch")().forward, 即 F.layer_norm (w/b 为 fp32)。"""
    return F.layer_norm(x, [x.shape[-1]], weight=w, bias=b, eps=eps)


def tolerances(dtype):
    # 照抄 test_layernorm_950_perf.py::test_layernorm_accuracy
    if dtype == torch.float32:
        return 1e-4, 1e-5
    return 5e-2, 1e-2


def run_one(shape, dtype, eps, aicore_num, timeout):
    print(f"\n{'=' * 70}")
    print(f"[shape] {shape} dtype={dtype} eps={eps} aicore={aicore_num}")
    ensure_blas_shim_dtype(dtype)

    torch.manual_seed(42)
    # 与原测试一致: x 为被测 dtype, weight/bias 恒为 fp32 参数。
    x = torch.randn(size=shape, dtype=dtype)
    w = torch.randn(size=(shape[-1],), dtype=torch.float32)
    b = torch.randn(size=(shape[-1],), dtype=torch.float32)

    atol, rtol = tolerances(dtype)

    t_start = time.perf_counter()
    with standalone_runtime.enable(timeout=timeout) as runtime:
        y = layernorm_infer_impl(x, w, b, eps, aicore_num)
        standalone_runtime.synchronize()
        wall_s = time.perf_counter() - t_start

        y_ref = layernorm_reference(x, w, b, eps)
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


# 取 test_layernorm.py shape 列表里 standalone 仿真可承受的前几项
# (后面 4096/8192 列、65535 行的 case 在仿真器上跑不完)。
DEFAULT_SHAPES = [(256, 128), (1, 768), (128, 768)]
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
    shapes = DEFAULT_SHAPES
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
