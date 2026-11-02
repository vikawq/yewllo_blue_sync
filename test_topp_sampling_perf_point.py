"""Compare Triton/NPU top-p sampling with a PyTorch CPU reference.

The largest input is (1024, 151936), with topk=1000, topp=0.75 and
min_tokens_to_keep=1. Select it with ``-k '1024x151936 and fp16'``.
Performance and accuracy are separate tests; select the performance function
explicitly to avoid running the 400-sample KS check during timing experiments.

The first Triton call prints stage timings by default. Set TOPP_PERF_TRACE=0
to disable this diagnostic. TOPP_PERF_STACK_TIMEOUT (default: 60 seconds,
0 to disable) periodically dumps Python stacks while the first call is pending.
Stage timings include synchronization overhead; warmup/repeat calls are unmodified.
"""

import ast
from contextlib import contextmanager
import faulthandler
import inspect
import os
import textwrap
import time

import pytest
import torch

from mojo_opset import MojoTopPSampling


DTYPES = [
    pytest.param(torch.float16, id="fp16"),
    pytest.param(torch.bfloat16, id="bf16"),
    pytest.param(torch.float32, id="fp32"),
]
# Same configurations as test_topp_sampling.py, excluding its duplicate entry.
CASES = [
    pytest.param((1, 32000), 1000, 0.9, 1, id="1x32000"),
    pytest.param((2, 32000), 50, 0.5, 1, id="2x32000"),
    pytest.param((8, 65024), 100, 0.6, 1, id="8x65024"),
    pytest.param((16, 32000), 2000, 0.9, 1, id="16x32000"),
    pytest.param((32, 32000), 100, 0.6, 2, id="32x32000"),
    pytest.param((128, 32000), 2000, 0.9, 1, id="128x32000"),
    pytest.param((512, 32000), 1000, 0.5, 2, id="512x32000"),
    pytest.param((1024, 151936), 1000, 0.75, 1, id="1024x151936"),
    pytest.param((16, 92544), 1000, 0.75, 1, id="16x92544"),
]
NUM_SAMPLES = 400
KS_D_CRIT = 3.04 * (2 / NUM_SAMPLES) ** 0.5


def _sampling(backend, topp, min_tokens_to_keep, topk):
    op = MojoTopPSampling._registry.get(backend)(
        top_p=topp, min_tokens_to_keep=min_tokens_to_keep, rand_top_k=topk
    )
    # Registry.get() can silently fall back to another backend.
    assert op._backend == backend, (
        f"Requested {backend}, got {type(op).__name__} ({op._backend}); "
        "check that the requested backend is installed and registered."
    )
    return op


@contextmanager
def _trace_stage(label):
    print(f"[stage] {label} BEGIN", flush=True)
    start = time.perf_counter()
    phase = "host_call"
    try:
        yield
        host_ms = (time.perf_counter() - start) * 1000
        print(f"[stage] {label} HOST_RETURN host={host_ms:.3f} ms; SYNC_BEGIN", flush=True)
        phase = "synchronize"
        sync_start = time.perf_counter()
        torch.npu.synchronize()
        sync_ms = (time.perf_counter() - sync_start) * 1000
        print(
            f"[stage] {label} END host={host_ms:.3f} ms, "
            f"sync_wait={sync_ms:.3f} ms, sum={host_ms + sync_ms:.3f} ms",
            flush=True,
        )
    except BaseException as error:
        print(f"[stage] {label} FAILED during {phase}: {error!r}", flush=True)
        raise


def _traced_executor(op, logits, monkeypatch):
    """Instrument the installed implementation in memory; keep its real kernel.

    Using its source avoids copying an algorithm that may differ on the target
    machine. The entry point is restored before normal warmup and repeat calls.
    """
    from mojo_opset.backends.ttx import kernels

    implementation = kernels.top_p_sampling_impl
    forward_globals = op.forward.__func__.__globals__
    if forward_globals.get("top_p_sampling") is not implementation:
        raise RuntimeError("Stage tracing requires MOJO_RUN_MODE=EAGER (or TOPP_PERF_TRACE=0).")

    source_lines, first_line = inspect.getsourcelines(implementation)
    filename = inspect.getsourcefile(implementation)
    tree = ast.parse(textwrap.dedent("".join(source_lines)))
    ast.increment_lineno(tree, first_line - 1)
    stages = {
        "logits": "cast_fp32",
        "sorted_logits": "torch.topk [NPU]",
        "rand_data": "random [NPU]",
        "output_data": "allocate_output",
        "output_scores": "allocate_scores",
        "output_final_probs": "allocate_probs",
        "sampled_index_in_topk": "torch.argmax [NPU]",
        "output_tokens": "postprocess_tokens [NPU]",
        "output_probs": "postprocess_probs [NPU]",
    }
    instrumented = []

    class AddStages(ast.NodeTransformer):
        def wrap(self, node, stage):
            label = f"{stage} ({filename}:{node.lineno})"
            instrumented.append(stage)
            wrapped = ast.With(
                items=[ast.withitem(context_expr=ast.Call(
                    func=ast.Name(id="_topp_perf_trace_stage", ctx=ast.Load()),
                    args=[ast.Constant(value=label)], keywords=[],
                ))],
                body=[node],
            )
            return ast.copy_location(wrapped, node)

        def visit_Assign(self, node):
            for target in node.targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name) and name.id in stages:
                        return self.wrap(node, stages[name.id])
            return node

        def visit_Expr(self, node):
            call = node.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Subscript)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "_top_p_sample_kernel"):
                return self.wrap(node, "triton_kernel [host includes JIT/launch]")
            return node

    tree = ast.fix_missing_locations(AddStages().visit(tree))
    if "torch.topk [NPU]" not in instrumented or not any(
        stage.startswith("triton_kernel") for stage in instrumented
    ):
        raise RuntimeError("Installed sampling implementation differs; cannot trace topk/kernel safely.")
    namespace = dict(implementation.__globals__)
    namespace["_topp_perf_trace_stage"] = _trace_stage
    exec(compile(tree, filename, "exec"), namespace)
    traced = namespace[implementation.__name__]
    print(f"[trace] implementation={filename}:{first_line}; first call only", flush=True)

    def execute():
        with monkeypatch.context() as patch:
            patch.setitem(forward_globals, "top_p_sampling", traced)
            return op(logits)

    return execute


def _measure(executor, label, synchronize=None, first_executor=None):
    warmup = int(os.environ.get("TOPP_PERF_WARMUP", "3"))
    repeat = int(os.environ.get("TOPP_PERF_REPEAT", "10"))
    assert warmup >= 1 and repeat >= 1

    print(f"\n[{label}] starting first call (includes any JIT compilation)", flush=True)
    stack_timeout = int(os.environ.get("TOPP_PERF_STACK_TIMEOUT", "60"))
    dump_stacks = first_executor is not None and stack_timeout > 0
    if dump_stacks:
        faulthandler.dump_traceback_later(stack_timeout, repeat=True)
    try:
        if synchronize is not None:
            print(f"[{label}] initial_sync BEGIN", flush=True)
            synchronize()
            print(f"[{label}] initial_sync END", flush=True)
        start = time.perf_counter()
        (first_executor if first_executor is not None else executor)()
        if synchronize is not None:
            print(f"[{label}] first call HOST_RETURN; final_sync BEGIN", flush=True)
            synchronize()
            print(f"[{label}] final_sync END", flush=True)
        first_ms = (time.perf_counter() - start) * 1000
    finally:
        if dump_stacks:
            faulthandler.cancel_dump_traceback_later()
    diagnostic = " (includes stage logging/synchronization)" if first_executor is not None else ""
    print(f"[{label}] first_call={first_ms:.3f} ms{diagnostic}", flush=True)

    print(f"[{label}] remaining warmup calls={warmup - 1} BEGIN", flush=True)
    for _ in range(warmup - 1):
        executor()
    if synchronize is not None:
        synchronize()

    print(f"[{label}] warmup END; measuring {repeat} uninstrumented calls BEGIN", flush=True)
    start = time.perf_counter()
    for _ in range(repeat):
        executor()
        if synchronize is not None:
            synchronize()
    mean_ms = (time.perf_counter() - start) * 1000 / repeat
    print(
        f"[{label}] mean={mean_ms:.3f} ms/call, warmup={warmup}, repeat={repeat}",
        flush=True,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape, topk, topp, min_tokens_to_keep", CASES)
@pytest.mark.parametrize("backend", ["cpu", "triton"])
@torch.inference_mode()
def test_topp_sampling_perf(shape, topk, topp, min_tokens_to_keep, dtype, backend, monkeypatch):
    torch.manual_seed(20260806)
    # The session fixture sets the default device to NPU. Be explicit here.
    logits = torch.randn(shape, dtype=dtype, device="cpu").npu()
    label = f"{backend}, shape={shape}, dtype={dtype}"
    if backend == "cpu":
        logits_cpu = logits.cpu()
        top_p_sampling_ref = _sampling("torch", topp, min_tokens_to_keep, topk)
        # Move data before timing; the reference executes on CPU, not Ascend.
        with torch.device("cpu"):
            _measure(lambda: top_p_sampling_ref(logits_cpu), label)
    else:
        top_p_sampling = _sampling("ttx", topp, min_tokens_to_keep, topk)
        traced = None
        if os.environ.get("TOPP_PERF_TRACE", "1") == "1":
            traced = _traced_executor(top_p_sampling, logits, monkeypatch)
        _measure(lambda: top_p_sampling(logits), label, torch.npu.synchronize, traced)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape, topk, topp, min_tokens_to_keep", CASES)
@torch.inference_mode()
def test_topp_sampling_accuracy(shape, topk, topp, min_tokens_to_keep, dtype):
    torch.manual_seed(20260806)
    logits = torch.randn(shape, dtype=dtype, device="cpu").npu()
    logits_cpu = logits.cpu()
    top_p_sampling_ref = _sampling("torch", topp, min_tokens_to_keep, topk)
    top_p_sampling = _sampling("ttx", topp, min_tokens_to_keep, topk)

    print("\n[accuracy] collecting 400 CPU reference samples", flush=True)
    ref_probs_list = []
    with torch.device("cpu"):
        for _ in range(NUM_SAMPLES):
            ref_p, _ = top_p_sampling_ref(logits_cpu.clone())
            ref_probs_list.append(ref_p.float())

    print("[accuracy] collecting 400 Triton/NPU samples", flush=True)
    npu_probs_list = []
    for _ in range(NUM_SAMPLES):
        npu_p, _ = top_p_sampling(logits.clone())
        npu_probs_list.append(npu_p.float())

    # Keep the original two-sample KS criterion and run the check on CPU.
    with torch.device("cpu"):
        ref_all = torch.cat(ref_probs_list, dim=1)
        npu_all = torch.cat(npu_probs_list, dim=1).cpu()
        a_sorted = npu_all.sort(dim=1).values
        b_sorted = ref_all.sort(dim=1).values
        values = torch.cat([npu_all, ref_all], dim=1).sort(dim=1).values
        fa = torch.searchsorted(a_sorted, values, right=True) / NUM_SAMPLES
        fb = torch.searchsorted(b_sorted, values, right=True) / NUM_SAMPLES
        d = (fa - fb).abs().max(dim=1).values
        row = int(d.argmax().item())
        max_d = float(d[row].item())

    assert max_d <= KS_D_CRIT, (
        f"Triton/CPU KS test failed: row={row} D={max_d:.4f} > D_crit={KS_D_CRIT:.4f}"
    )
