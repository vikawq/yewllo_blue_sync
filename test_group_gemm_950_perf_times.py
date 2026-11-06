"""Ascend 950 test copy with OP_TIME JSON rows for a timing heatmap.

torch_api rows time explicit setup calls; dispatcher_op rows time operators reached
during compute. NPU values use synchronized wall time, including the final sync.
stage_with_probes includes measurement overhead and must not be summed with op rows.
"""

import json
from contextvars import ContextVar
from functools import wraps
from uuid import uuid4
from pathlib import Path
from time import perf_counter_ns
from contextlib import contextmanager
from functools import partial
from unittest.mock import patch

import random

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from mojo_opset import MojoGroupGemm
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented


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


@contextmanager
def _accuracy_forwards(test, case, candidate, reference):
    # forward_diff_with calls .forward directly; module hooks would miss both calls.
    with (
        patch.object(candidate, "forward", partial(_trace_call, test, case, "accuracy_forward", candidate.forward)),
        patch.object(reference, "forward", partial(_trace_call, test, case, "torch_reference_forward", reference.forward)),
    ):
        yield


def generate_random_list(length, total_sum, times):
    avg = total_sum // length
    lst = [0] * length
    for i in range(length):
        lst[i] = random.randint(0, 2 * int(avg))
    ratio = total_sum / sum(lst)
    lst = [int(x * ratio) for x in lst]

    diff = total_sum - sum(lst)
    lst[-1] += diff
    groups = times.api("torch.Tensor", torch.Tensor, lst)
    return times.api("Tensor.to", groups.to, torch.int32)


def _group_gemm_cases(test):
    specs = [
        ((8 * 2560, 4096), (8, 4096, 4096), False, None, "g8"),
        ((4 * 1024, 2048), (4, 2048, 1024), False, None, "g4"),
        ((6 * 512, 1024), (6, 2048, 1024), True, None, "g6_trans"),
        ((256, 128), (1, 128, 64), False, [256], "single_group"),
        ((192, 64), (4, 64, 96), False, [16, 64, 32, 80], "uneven_groups"),
        ((256, 128), (4, 96, 128), True, [48, 80, 64, 64], "trans_weight_uneven"),
    ]
    cases = []
    for input_shape, weight_shape, trans_weight, lengths, label in specs:
        for dtype in (torch.float16, torch.bfloat16):
            case = f"{label}_fp={'bf16' if dtype is torch.bfloat16 else 'fp16'}"
            with _Times(test, case, "setup") as times:
                input = times.api("torch.randn", torch.randn, size=input_shape, dtype=dtype)
                input = times.api("Tensor.npu", input.npu)
                weight = times.api("torch.randn", torch.randn, size=weight_shape, dtype=dtype)
                weight = times.api("Tensor.npu", weight.npu)
                if lengths is None:
                    groups = generate_random_list(weight_shape[0], input_shape[0], times)
                else:
                    groups = times.api("torch.tensor", torch.tensor, lengths, dtype=torch.int32)
                group_list = times.api("Tensor.npu", groups.npu)
            cases.append(pytest.param(input, weight, group_list, trans_weight, id=case))
    return cases


@pytest.mark.parametrize(
    "input, weight, group_list, trans_weight",
    _group_gemm_cases("test_group_gemm"),
)
@bypass_not_implemented
@auto_switch_platform(set_perf=True)
@_timed_case
def test_group_gemm(input, weight, group_list, trans_weight, request):
    group_gemm = MojoGroupGemm(
        trans_weight=trans_weight,
        weight=weight,
    )


    with _phase_wall("perf_benchmark"):
        perf(lambda: group_gemm(input, group_list))  # noqa: F821
    _trace_call(
        request.node.originalname, request.node.callspec.id, "perf_forward", group_gemm, input, group_list,
    )


@pytest.mark.parametrize(
    "input, weight, group_list, trans_weight",
    _group_gemm_cases("test_group_gemm_accuracy"),
)
@bypass_not_implemented
@_timed_case
def test_group_gemm_accuracy(input, weight, group_list, trans_weight, request):
    group_gemm = MojoGroupGemm(
        trans_weight=trans_weight,
        weight=weight,
    )

    group_gemm_ref = MojoGroupGemm._registry.get("torch")(
        trans_weight=trans_weight,
        weight=weight,
    )

    if input.dtype == torch.float32:
        atol, rtol = 1e-5, 1e-6
    else:
        atol, rtol = 3e-2, 6e-3

    with _phase_wall("accuracy_compare", split_nested=True):
        with _accuracy_forwards(request.node.originalname, request.node.callspec.id, group_gemm, group_gemm_ref):
            group_gemm.forward_diff_with(group_gemm_ref, input, group_list, atol=atol, rtol=rtol)