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

import math

from typing import Optional

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from mojo_opset import MojoPagedDecodeGQA
from mojo_opset import MojoPagedPrefillGQA
from mojo_opset import MojoSdpa
from mojo_opset import MojoPagedPrefillSWA
from mojo_opset import MojoPagedDecodeSWA
from mojo_opset import MojoSWA
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


def generate_paged_decode_data(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    block_size: int,
    dtype: torch.dtype,
    *,
    test: str,
    case: str,
):
    with _Times(test, case, "setup") as times:
        times.api("torch.manual_seed", torch.manual_seed, 43)
        query = times.api("torch.randn", torch.randn, batch_size, num_q_heads, head_dim, dtype=dtype)
        query = times.api("Tensor.npu", query.npu)
        if max_seq_len > 0:
            total_seq_lens = times.api(
                "torch.randint", torch.randint, 0, max_seq_len, (batch_size,), dtype=torch.int32,
            )
            total_seq_lens = times.api("Tensor.npu", total_seq_lens.npu)
            total_seq_lens = times.api("Tensor.npu", times.api("torch.clamp", torch.clamp, total_seq_lens, min=1).npu)
        else:
            total_seq_lens = times.api("torch.randperm", torch.randperm, batch_size, dtype=torch.int32)
            total_seq_lens = times.api("Tensor.npu", total_seq_lens.npu)

        max_seq = times.api("Tensor.max", total_seq_lens.max)
        max_num_blocks_per_seq = (times.api("Tensor.item", max_seq.item) + block_size - 1) // block_size
        blocks_per_seq = times.api(
            "torch.div", torch.div, total_seq_lens + block_size - 1, block_size, rounding_mode="floor",
        )
        blocks_sum = times.api("Tensor.sum", blocks_per_seq.sum)
        total_blocks_needed = int(times.api("Tensor.item", blocks_sum.item))

        if total_blocks_needed == 0:
            total_blocks_needed = batch_size * max_num_blocks_per_seq

        num_total_blocks = total_blocks_needed + 10

        k_cache = times.api(
            "torch.randn", torch.randn, num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype,
        )
        k_cache = times.api("Tensor.npu", k_cache.npu)
        v_cache = times.api(
            "torch.randn", torch.randn, num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype,
        )
        v_cache = times.api("Tensor.npu", v_cache.npu)

        block_tables = times.api(
            "torch.full", torch.full, (batch_size, max_num_blocks_per_seq), -1, dtype=torch.int32,
        )
        block_tables = times.api("Tensor.npu", block_tables.npu)
        free_blocks = times.api("Tensor.npu", times.api("torch.randperm", torch.randperm, num_total_blocks, dtype=torch.int32).npu)

        current_block_offset = 0
        for i in range(batch_size):
            seq_len = times.api("Tensor.item", total_seq_lens[i].item)
            num_blocks_for_seq = (seq_len + block_size - 1) // block_size

            if current_block_offset + num_blocks_for_seq > num_total_blocks:
                raise ValueError("Not enough blocks to generate test data.")

            assigned_blocks = free_blocks[current_block_offset : current_block_offset + num_blocks_for_seq]
            block_tables[i, :num_blocks_for_seq] = assigned_blocks
            current_block_offset += num_blocks_for_seq

        return query, k_cache, v_cache, total_seq_lens, block_tables


test_configs_decode = [
    (8, 16, 4, 128, 1024, 32, torch.bfloat16, "M_BF16"),
    (8, 16, 4, 128, 1024, 128, torch.bfloat16, "M_BF16_PADDIM"),
    (8, 8, 1, 128, 8192, 1024, torch.bfloat16, "M_BF16_LONG"),
    (8, 8, 1, 128, 2048, 1024, torch.bfloat16, "M_BF16_BIGPAGE"),
    (8, 8, 1, 128, 0, 1024, torch.bfloat16, "M_BF16_PADSEQ"),
    (8, 8, 1, 128, 16384, 128, torch.bfloat16, "M_BF16_LONG_16384"),
    (8, 8, 1, 128, 32768, 128, torch.bfloat16, "M_BF16_LONG_32768"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, total_seq_lens, block_tables",
    [
        pytest.param(
            *generate_paged_decode_data(
                batch_size=B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_seq_len=S_LEN,
                block_size=BLK_S,
                dtype=dtype,
                test="test_paged_decode_gqa",
                case=f"AABB-{ID}",
            ),
            id=ID,
        )
        for B, Q_H, KV_H, D, S_LEN, BLK_S, dtype, ID in test_configs_decode
    ],
)
@pytest.mark.parametrize("gqa_layout", ["AABB"])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
@_timed_case
def test_paged_decode_gqa(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    total_seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    gqa_layout: str,
    request,
):
    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    paged_attn_decode = MojoPagedDecodeGQA(
        is_causal=True,
        gqa_layout=gqa_layout,
    )

    with _phase_wall("perf_benchmark"):
        perf(  # noqa: F821
            lambda: paged_attn_decode(
                query,
                k_cache,
                v_cache,
                total_seq_lens,
                block_tables,
                softmax_scale=softmax_scale,
            )
        )
    _trace_call(
        request.node.originalname,
        request.node.callspec.id,
        "perf_forward",
        paged_attn_decode,
        query,
        k_cache,
        v_cache,
        total_seq_lens,
        block_tables,
        softmax_scale=softmax_scale,
    )


@pytest.mark.parametrize(
    "query, k_cache, v_cache, total_seq_lens, block_tables",
    [
        pytest.param(
            *generate_paged_decode_data(
                batch_size=B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_seq_len=S_LEN,
                block_size=BLK_S,
                dtype=dtype,
                test="test_paged_decode_gqa_accuracy",
                case=f"AABB-{ID}_acc",
            ),
            id=f"{ID}_acc",
        )
        for B, Q_H, KV_H, D, S_LEN, BLK_S, dtype, ID in test_configs_decode
    ],
)
@pytest.mark.parametrize("gqa_layout", ["AABB"])
@bypass_not_implemented
@_timed_case
def test_paged_decode_gqa_accuracy(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    total_seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    gqa_layout: str,
    request,
):
    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    paged_attn_decode = MojoPagedDecodeGQA(
        is_causal=True,
        gqa_layout=gqa_layout,
    )

    paged_attn_decode_ref = MojoPagedDecodeGQA._registry.get("torch")(
        is_causal=True,
        gqa_layout=gqa_layout,
    )

    atol, rtol = (3e-2, 6e-3) if query.dtype != torch.float32 else (1e-5, 1e-6)

    with _phase_wall("accuracy_compare", split_nested=True):
        with _accuracy_forwards(
            request.node.originalname, request.node.callspec.id, paged_attn_decode, paged_attn_decode_ref,
        ):
            paged_attn_decode.forward_diff_with(
                paged_attn_decode_ref,
                query,
                k_cache,
                v_cache,
                total_seq_lens,
                block_tables,
                softmax_scale=softmax_scale,
                atol=atol,
                rtol=rtol,
            )