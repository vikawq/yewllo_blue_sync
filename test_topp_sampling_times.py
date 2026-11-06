"""TopP timing copy. OP_TIME rows use the same schema as the other timing tests.

The performance test records one diagnostic forward after the benchmark.
Accuracy probes observe the first sample pair. The many-seed KS test only
instruments the first seed and its first KS row.
"""

import json
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from uuid import uuid4
from pathlib import Path
from time import perf_counter_ns

import math
import pytest
import random
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from mojo_opset import MojoTopPSampling
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


SEED_SPACE = 2 ** 32
META_SEED = 20260806
KS_LAMBDA = 3.04
NUM_SAMPLES = 400
KS_D_CRIT = KS_LAMBDA * ((NUM_SAMPLES + NUM_SAMPLES) / (NUM_SAMPLES * NUM_SAMPLES)) ** 0.5

def _random_seeds(count: int) -> list:
    return random.Random(META_SEED).sample(range(SEED_SPACE), count)


def two_sample_ks(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_sorted, _ = a.sort(dim=1)
    b_sorted, _ = b.sort(dim=1)
    v, _ = torch.cat([a, b], dim=1).sort(dim=1)
    ia = torch.searchsorted(a_sorted, v, right=True)
    ib = torch.searchsorted(b_sorted, v, right=True)
    fa = ia / a.shape[1]
    fb = ib / b.shape[1]
    return (fa - fb).abs().max(dim=1).values


def _ks_two_sample_pvalue(d: float, n: int, m: int) -> float:
    lam = d * math.sqrt((n * m) / (n + m))
    total = 0.0
    k = 1
    while True:
        term = 2.0 * ((-1.0) ** (k - 1)) * math.exp(-2.0 * k * k * lam * lam)
        total += term
        if abs(term) < 1e-12 * abs(total) or k > 10000:
            break
        k += 1
    return max(0.0, min(1.0, total))


def _setup_logits(test, case, shape, dtype, seed=None):
    with _Times(test, case, "setup") as times:
        if seed is not None:
            times.api("torch.manual_seed", torch.manual_seed, seed)
        logits = times.api("torch.randn", torch.randn, shape, dtype=dtype)
        return times.api("Tensor.npu", logits.npu)


def _sample_cloned(sampler, logits):
    return sampler(logits.clone())


def _collect_probs(reference, candidate, logits, count, tags=None):
    ref_probs_list = []
    npu_probs_list = []
    for sample in range(count):
        if tags is not None and sample == 0:
            ref_p, _ = _trace_call(*tags, "torch_reference_forward", _sample_cloned, reference, logits)
            npu_p, _ = _trace_call(*tags, "accuracy_forward", _sample_cloned, candidate, logits)
            with _Times(*tags, "accuracy_prepare") as times:
                ref_probs_list.append(times.api("Tensor.float", ref_p.float))
                npu_probs_list.append(times.api("Tensor.float", npu_p.float))
        else:
            ref_p, _ = reference(logits.clone())
            npu_p, _ = candidate(logits.clone())
            ref_probs_list.append(ref_p.float())
            npu_probs_list.append(npu_p.float())
    return ref_probs_list, npu_probs_list


def _join_probs(ref_probs_list, npu_probs_list):
    ref_all = torch.cat(ref_probs_list, dim=1).cpu()
    npu_all = torch.cat(npu_probs_list, dim=1).cpu()
    return ref_all, npu_all


def _max_ks(npu_all, ref_all):
    d = two_sample_ks(npu_all, ref_all)
    row = int(d.argmax().item())
    return row, float(d[row].item())


def _row_ks(npu_all, ref_all, i):
    return float(two_sample_ks(npu_all[i : i + 1], ref_all[i : i + 1])[0].item())


dtypes = [torch.float16, torch.bfloat16, torch.float32]
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("shape, topk, topp, min_tokens_to_keep", [
    ((1, 32000), 1000, 0.9, 1),
    ((2, 32000), 50, 0.5, 1),
    ((8, 65024), 100, 0.6, 1),
    ((8, 65024), 100, 0.6, 1),
    ((16, 32000), 2000, 0.9, 1),
    ((32, 32000), 100, 0.6, 2),
    ((128, 32000), 2000, 0.9, 1),
    ((512, 32000), 1000, 0.5, 2),
    ((1024, 151936), 1000, 0.75, 1),
    ((16, 92544), 1000, 0.75, 1),
])
@auto_switch_platform(set_perf=True)
@_timed_case
def test_topp_sampling(shape, topk, topp, min_tokens_to_keep, dtype, request):
    tags = (request.node.originalname, request.node.callspec.id)
    logits = _setup_logits(*tags, shape, dtype)
    top_p_sampling = MojoTopPSampling(top_p=topp, min_tokens_to_keep=min_tokens_to_keep, rand_top_k=topk)
    with _phase_wall("perf_benchmark"):
        perf(lambda: top_p_sampling(logits))  # noqa: F821
    _trace_call(*tags, "perf_forward", top_p_sampling, logits)



@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("shape, topk, topp, min_tokens_to_keep", [
    ((1, 32000), 1000, 0.9, 1),
    ((2, 32000), 50, 0.5, 1),
    ((8, 65024), 100, 0.6, 1),
    ((16, 32000), 2000, 0.9, 1),
    ((32, 32000), 100, 0.6, 2),
    ((128, 32000), 2000, 0.9, 1),
    ((512, 32000), 1000, 0.5, 2),
    ((1024, 151936), 1000, 0.75, 1),
    ((16, 92544), 1000, 0.75, 1),
])
@auto_switch_platform()
@bypass_not_implemented
@_timed_case
def test_topp_sampling_accuracy(shape, topk, topp, min_tokens_to_keep, dtype, request):
    tags = (request.node.originalname, request.node.callspec.id)
    logits = _setup_logits(*tags, shape, dtype)
    top_p_sampling_ref = MojoTopPSampling._registry.get("torch")(
        top_p=topp, min_tokens_to_keep=min_tokens_to_keep, rand_top_k=topk)
    top_p_sampling = MojoTopPSampling(
        top_p=topp, min_tokens_to_keep=min_tokens_to_keep, rand_top_k=topk)
    with _phase_wall("accuracy_sampling"):
        ref_probs_list, npu_probs_list = _collect_probs(
            top_p_sampling_ref, top_p_sampling, logits, NUM_SAMPLES, tags,
        )
    ref_all, npu_all = _trace_call(*tags, "accuracy_join_probs", _join_probs, ref_probs_list, npu_probs_list)
    row, max_d = _trace_call(*tags, "accuracy_ks", _max_ks, npu_all, ref_all)
    lam = max_d * math.sqrt((NUM_SAMPLES * NUM_SAMPLES) / (2 * NUM_SAMPLES))
    p_ub = 2.0 * math.exp(-2.0 * lam * lam)
    assert max_d <= KS_D_CRIT, (
        "npu/ref KS test failed: row=%d D=%.4f > D_crit=%.4f, p≤%.3e"
        % (row, max_d, KS_D_CRIT, p_ub))

new_dtypes = [torch.float16]
@pytest.mark.parametrize("dtype", new_dtypes)
@pytest.mark.parametrize("shape, topk, topp, min_tokens_to_keep", [
    ((32, 32000), 100, 0.6, 2),
])
@auto_switch_platform()
@bypass_not_implemented
@_timed_case
def test_topp_sampling_accuracy_kstest(shape, topk, topp, min_tokens_to_keep, dtype, request):
    tags = (request.node.originalname, request.node.callspec.id)
    batch_size = shape[0]
    num_samples = max(100, min(1000, 16000 // batch_size))
    num_seeds = 10000
    alpha = 0.05
    pass_threshold = 0.9

    num_tests = 0
    num_pass = 0
    for seed_index, seed in enumerate(_random_seeds(num_seeds)):
        sample_tags = tags if seed_index == 0 else None
        with _phase_wall("setup"):
            if sample_tags is not None:
                logits = _setup_logits(*sample_tags, shape, dtype, seed=seed)
            else:
                torch.manual_seed(seed)
                logits = torch.randn(shape, dtype=dtype).npu()
            top_p_sampling_ref = MojoTopPSampling._registry.get("torch")(
                top_p=topp, min_tokens_to_keep=min_tokens_to_keep, rand_top_k=topk)
            top_p_sampling = MojoTopPSampling(
                top_p=topp, min_tokens_to_keep=min_tokens_to_keep, rand_top_k=topk)
        with _phase_wall("accuracy_sampling"):
            ref_probs_list, npu_probs_list = _collect_probs(
                top_p_sampling_ref, top_p_sampling, logits, num_samples, sample_tags,
            )
        with _phase_wall("accuracy_join_probs"):
            if sample_tags is not None:
                ref_all, npu_all = _trace_call(
                    *sample_tags, "accuracy_join_probs", _join_probs, ref_probs_list, npu_probs_list,
                )
            else:
                ref_all, npu_all = _join_probs(ref_probs_list, npu_probs_list)
        with _phase_wall("accuracy_ks"):
            for i in range(batch_size):
                if sample_tags is not None and i == 0:
                    d = _trace_call(*sample_tags, "accuracy_ks", _row_ks, npu_all, ref_all, i)
                else:
                    d = _row_ks(npu_all, ref_all, i)
                pval = _ks_two_sample_pvalue(d, num_samples, num_samples)
                num_tests += 1
                if pval >= alpha:
                    num_pass += 1

    pass_rate = num_pass / num_tests
    print(f"pass_rate={pass_rate:.4f}")
    assert pass_rate >= pass_threshold, (
        f"KS test pass rate {pass_rate:.4f} < {pass_threshold:.4f}: "
        f"{num_pass}/{num_tests} passed at alpha={alpha:.4f}")