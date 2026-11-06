"""Ascend 950 test copy with OP_TIME JSON rows for a timing heatmap.

torch_api rows time explicit setup calls; dispatcher_op rows time operators reached
during compute. NPU values use synchronized wall time, including the final sync.
stage_with_probes includes measurement overhead and must not be summed with op rows.
"""

import json
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from uuid import uuid4
from pathlib import Path
from time import perf_counter_ns

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
import torch_npu
import math

from mojo_opset import MojoSWAFunction

from mojo_opset.tests.utils import assert_close
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.tests.utils import write_profile_caseid


# Phase accounting uses non-overlapping wall intervals, separate from OP_TIME.
_case_clock = ContextVar("case_clock", default=None)
_collection_id = uuid4().hex


def _wall_log(marker, row):
    print(marker + " " + json.dumps({"script": Path(__file__).name, "run": _collection_id, **row}), flush=True)


def _sync_wall():
    npu = getattr(torch, "npu", None)
    if getattr(npu, "is_initialized", lambda: False)():
        npu.synchronize()


class _CaseClock:
    def __init__(self, request):
        self.test = request.node.originalname
        self.case = getattr(getattr(request.node, "callspec", None), "id", self.test)
        self.setup_case = self.case
        self.phases = {}
        self.accounted_us = 0.0
        self.suppress_nested = False


def _timed_case(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        clock = _CaseClock(kwargs["request"])
        token = _case_clock.set(clock)
        try:
            _sync_wall()
            start = perf_counter_ns()
            result = func(*args, **kwargs)
            _sync_wall()
            elapsed_us = (perf_counter_ns() - start) / 1_000
        finally:
            _case_clock.reset(token)
        _wall_log("CASE_TIME", {
            "test": clock.test,
            "case": clock.case,
            "setup_case": clock.setup_case,
            "elapsed_us": elapsed_us,
            "phases": clock.phases,
        })
        return result
    return wrapped


@contextmanager
def _phase_wall(phase, *, split_nested=False):
    clock = _case_clock.get()
    if clock is None or clock.suppress_nested:
        yield
        return
    _sync_wall()
    start = perf_counter_ns()
    accounted_before = clock.accounted_us
    clock.suppress_nested = not split_nested
    try:
        yield
        _sync_wall()
        elapsed_us = (perf_counter_ns() - start) / 1_000
        own_us = max(0.0, elapsed_us - (clock.accounted_us - accounted_before))
        clock.phases[phase] = clock.phases.get(phase, 0) + own_us
        clock.accounted_us += own_us
    finally:
        clock.suppress_nested = False


# Keep the timing helper local so this copy can run beside its original test.
def _npu_device(value):
    if isinstance(value, torch.Tensor):
        value = value.device
    if isinstance(value, torch.device):
        return value if value.type == "npu" else None
    if isinstance(value, (tuple, list)):
        for item in value:
            device = _npu_device(item)
            if device is not None:
                return device
    if isinstance(value, dict):
        return _npu_device(tuple(value.values()))
    return None


class _Times:
    def __init__(self, test, case, phase):
        self.tags = {"script": Path(__file__).name, "test": test, "case": case, "phase": phase}
        self.rows = []

    def __enter__(self):
        self.wall = _phase_wall(self.tags["phase"])
        self.wall.__enter__()
        self.collection_start = (
            perf_counter_ns() if self.tags["phase"] == "setup" and _case_clock.get() is None else None
        )
        return self

    def __exit__(self, *exc):
        try:
            if self.rows:
                print("\n".join("OP_TIME " + json.dumps(row) for row in self.rows), flush=True)
            if self.collection_start is not None and exc[0] is None:
                _wall_log("CASE_SETUP", {
                    "test": self.tags["test"],
                    "case": self.tags["case"],
                    "elapsed_us": (perf_counter_ns() - self.collection_start) / 1_000,
                })
        finally:
            self.wall.__exit__(*exc)

    def measure(self, kind, op, func, args, kwargs, device_hint=None):
        requested = kwargs.get("device")
        if isinstance(requested, str):
            requested = torch.device(requested)
        device = _npu_device((device_hint, requested, getattr(func, "__self__", None), args, kwargs))
        if op == "Tensor.npu" and device is None:
            device = torch.device("npu", torch.npu.current_device())
        if device is not None:
            torch.npu.synchronize(device)
        start = perf_counter_ns()
        result = func(*args, **kwargs)
        host_end = perf_counter_ns()
        if device is None:
            device = _npu_device(result)
        if device is not None:
            torch.npu.synchronize(device)
        elapsed_us = (perf_counter_ns() - start) / 1_000
        self.rows.append({
            **self.tags,
            "seq": len(self.rows) + 1,
            "kind": kind,
            "op": op,
            "device": str(device) if device is not None else "cpu",
            "host_us": (host_end - start) / 1_000,
            "elapsed_us": elapsed_us,
        })
        return result

    def api(self, op, func, /, *args, **kwargs):
        return self.measure("torch_api", op, func, args, kwargs)


class _DispatchTimes(TorchDispatchMode):
    def __init__(self, times):
        super().__init__()
        self.times = times

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return self.times.measure("dispatcher_op", str(func), func, args, kwargs or {})


def _trace_call(test, case, phase, func, /, *args, **kwargs):
    with _Times(test, case, phase) as times:
        def traced(*args, **kwargs):
            with _DispatchTimes(times):
                return func(*args, **kwargs)

        # Includes the probes themselves; keep this row out of per-op totals.
        return times.measure(
            "stage_with_probes", "__stage_total__", traced, args, kwargs,
            device_hint=getattr(func, "__self__", None),
        )


def generate_sdpa_data(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_q_len: int,
    max_kv_computed_len: int,
    dtype: torch.dtype,
    *,
    test: str,
    case: str,
):
    with _Times(test, case, "setup") as times:
        times.api("torch.manual_seed", torch.manual_seed, 43)
        #q_lens = torch.randint(max_q_len // 2, max_q_len, (batch_size,), dtype=torch.int32)
        q_lens = times.api("torch.tensor", torch.tensor, [max_q_len for _ in range(batch_size)], dtype=torch.int32)
        q_lens = times.api("torch.clamp", torch.clamp, q_lens, min=1)
        zero = times.api("torch.tensor", torch.tensor, [0], dtype=torch.int32)
        cumulative = times.api("torch.cumsum", torch.cumsum, q_lens, 0)
        cumulative = times.api("Tensor.to", cumulative.to, torch.int32)
        cu_q_lens = times.api("torch.cat", torch.cat, [zero, cumulative])

        if max_kv_computed_len <= 0:
            kv_cache_lens = None
            kv_lens = q_lens
        else:
            kv_cache_lens = times.api(
                "torch.randint", torch.randint, max_kv_computed_len // 2, max_kv_computed_len,
                (batch_size,), dtype=torch.int32,
            )
            kv_lens = q_lens + kv_cache_lens
        zero = times.api("torch.tensor", torch.tensor, [0], dtype=torch.int32)
        cumulative = times.api("torch.cumsum", torch.cumsum, kv_lens, 0)
        cumulative = times.api("Tensor.to", cumulative.to, torch.int32)
        cu_total_seq_lens = times.api("torch.cat", torch.cat, [zero, cumulative])

        total_q_tokens = times.api("Tensor.item", cu_q_lens[-1].item)
        total_kv_tokens = times.api("Tensor.item", cu_total_seq_lens[-1].item)

        query = times.api("torch.randn", torch.randn, total_q_tokens, num_q_heads, head_dim, dtype=dtype)
        key = times.api("torch.randn", torch.randn, total_kv_tokens, num_kv_heads, head_dim, dtype=dtype)
        value = times.api("torch.randn", torch.randn, total_kv_tokens, num_kv_heads, head_dim, dtype=dtype)
        grad_out = times.api("torch.randn", torch.randn, total_q_tokens, num_q_heads, head_dim, dtype=dtype)

        return query, key, value, grad_out, cu_q_lens, cu_total_seq_lens

def _prepare_grads(test, case, phase, *tensors):
    with _Times(test, case, phase) as times:
        prepared = []
        for tensor in tensors:
            tensor = times.api("Tensor.clone", tensor.clone)
            tensor = times.api("Tensor.detach", tensor.detach)
            prepared.append(times.api("Tensor.requires_grad_", tensor.requires_grad_, True))
        return prepared


test_configs_swa = [
    (2, 16, 4, 128, 1024, 0, torch.float32, "M_F32"),
    (2, 16, 4, 96, 1024, 0, torch.bfloat16, "M_BF16_PADDIM"),
    (2, 16, 4, 128, 4096, 0, torch.bfloat16, "M_BF16_LONG"),
]

@pytest.mark.parametrize(
    "query, key, value, grad_out, cu_q_lens, cu_total_seq_lens",
    [
        pytest.param(
            *generate_sdpa_data(
                batch_size=B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_q_len=Q_LEN,
                max_kv_computed_len=KV_COMPUTED_LEN,
                dtype=dtype,
                test="test_swa_function",
                case=ID,
            ),
            id=ID,
        )
        for B, Q_H, KV_H, D, Q_LEN, KV_COMPUTED_LEN, dtype, ID in test_configs_swa
    ],
)
@pytest.mark.parametrize("gqa_interleave, global_window, local_window", [
    (True, 4, 255),
    (False, 4, 1023),
])
@bypass_not_implemented
@auto_switch_platform()
@_timed_case
def test_swa_function(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_out: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_total_seq_lens: torch.Tensor,
    gqa_interleave: bool,
    global_window: int,
    local_window: int,
    request,
):
    test, case = request.node.originalname, request.node.callspec.id
    _case_clock.get().setup_case = next(config[-1] for config in test_configs_swa if case.endswith("-" + config[-1]))
    swa_func = MojoSWAFunction.apply

    swa_func_ref = MojoSWAFunction._registry.get("torch").apply

    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q, k, v = _prepare_grads(test, case, "accuracy_prepare_grads", query, key, value)
    o = _trace_call(
        test, case, "accuracy_forward", swa_func,
        q,
        k,
        v,
        cu_q_lens,
        cu_total_seq_lens,
        True,
        local_window,
        global_window,
        softmax_scale,
        gqa_interleave,
        True,
    )
    _trace_call(test, case, "accuracy_backward", o.backward, grad_out)

    q_ref, k_ref, v_ref = _prepare_grads(test, case, "torch_reference_prepare_grads", query, key, value)
    o_ref = _trace_call(
        test, case, "torch_reference_forward", swa_func_ref,
        q_ref,
        k_ref,
        v_ref,
        cu_q_lens,
        cu_total_seq_lens,
        True,
        local_window,
        global_window,
        softmax_scale,
        gqa_interleave,
        True,
    )
    _trace_call(test, case, "torch_reference_backward", o_ref.backward, grad_out)

    with _phase_wall("accuracy_check"):
        assert_close(o, o_ref)
        assert_close(q.grad, q_ref.grad)
        assert_close(k.grad, k_ref.grad)
        assert_close(v.grad, v_ref.grad)


def _run_swa_perf(input_shape, gqa_interleave, global_window, local_window, trace_dir, test):
    case = Path(trace_dir).name
    _case_clock.get().case = case
    B, Q_H, KV_H, D, Q_LEN, KV_COMPUTED_LEN, dtype = input_shape
    query, key, value, grad_out, cu_q_lens, cu_total_seq_lens \
        = generate_sdpa_data(
            batch_size=B,
            num_q_heads=Q_H,
            num_kv_heads=KV_H,
            head_dim=D,
            max_q_len=Q_LEN,
            max_kv_computed_len=KV_COMPUTED_LEN,
            dtype=dtype,
            test=test,
            case=case,
        )
    swa_func = MojoSWAFunction.apply

    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    def forward(q, k, v):
        return swa_func(
            q,
            k,
            v,
            cu_q_lens,
            cu_total_seq_lens,
            True,
            local_window,
            global_window,
            softmax_scale,
            gqa_interleave,
            True,
        )

    with _phase_wall("perf_benchmark"):
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
            l2_cache=False,
            data_simplification=False,
        )

        with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=0, warmup=5, active=5, repeat=1, skip_first=0),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(trace_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=False,
                with_modules=False,
                experimental_config=experimental_config,
        ) as prof:
            for _ in range(10):
                q = query.clone().detach().requires_grad_(True)
                k = key.clone().detach().requires_grad_(True)
                v = value.clone().detach().requires_grad_(True)
                o = forward(q, k, v)
                o.backward(grad_out)
                prof.step()
                torch.npu.synchronize()

        write_profile_caseid(prof)

    # Release the last profiler iteration before the extra diagnostic iteration.
    del q, k, v, o
    q, k, v = _prepare_grads(test, case, "perf_prepare_grads", query, key, value)
    o = _trace_call(test, case, "perf_forward", forward, q, k, v)
    _trace_call(test, case, "perf_backward", o.backward, grad_out)


@auto_switch_platform()
@bypass_not_implemented
@_timed_case
def test_swa_function_perf_m_f32(request):
    input_shape = (2, 16, 4, 128, 1024, 0, torch.float32)
    _run_swa_perf(input_shape, True, 4, 255, "./npu_profiling/M_F32", request.node.originalname)


@auto_switch_platform()
@bypass_not_implemented
@_timed_case
def test_swa_function_perf_m_bf16_paddim(request):
    input_shape = (2, 16, 4, 96, 1024, 0, torch.bfloat16)
    _run_swa_perf(input_shape, True, 4, 255, "./npu_profiling/M_BF16_PADDIM", request.node.originalname)


@auto_switch_platform()
@bypass_not_implemented
@_timed_case
def test_swa_function_perf_m_bf16_long(request):
    input_shape = (2, 16, 4, 128, 4096, 0, torch.bfloat16)
    _run_swa_perf(input_shape, True, 4, 255, "./npu_profiling/M_BF16_LONG", request.node.originalname)


@auto_switch_platform()
@bypass_not_implemented
@_timed_case
def test_swa_function_perf_m_f32_no_interleave(request):
    input_shape = (2, 16, 4, 128, 1024, 0, torch.float32)
    _run_swa_perf(input_shape, False, 4, 1023, "./npu_profiling/M_F32_NO_INTERLEAVE", request.node.originalname)

@auto_switch_platform()
@bypass_not_implemented
@_timed_case
def test_swa_function_perf_m_bf16_no_interleave(request):
    input_shape = (1, 12, 4, 128, 131072, 0, torch.bfloat16)
    _run_swa_perf(input_shape, False, 4, 1023, "./npu_profiling/M_BF16_NO_INTERLEAVE", request.node.originalname)