import pytest
import torch
import torch_npu
import math

from mojo_opset import MojoSWAFunction

from mojo_opset.tests.utils import assert_close
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.tests.utils import write_profile_caseid

def generate_sdpa_data(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_q_len: int,
    max_kv_computed_len: int,
    dtype: torch.dtype,
):
    torch.manual_seed(43)
    #q_lens = torch.randint(max_q_len // 2, max_q_len, (batch_size,), dtype=torch.int32)
    q_lens = torch.tensor([max_q_len for _ in range(batch_size)], dtype=torch.int32)
    q_lens = torch.clamp(q_lens, min=1)
    cu_q_lens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(q_lens, 0).to(torch.int32)])

    if max_kv_computed_len <= 0:
        kv_cache_lens = None
        kv_lens = q_lens
    else:
        kv_cache_lens = torch.randint(max_kv_computed_len // 2, max_kv_computed_len, (batch_size,), dtype=torch.int32)
        kv_lens = q_lens + kv_cache_lens
    cu_total_seq_lens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(kv_lens, 0).to(torch.int32)])

    total_q_tokens = cu_q_lens[-1].item()
    total_kv_tokens = cu_total_seq_lens[-1].item()

    query = torch.randn(total_q_tokens, num_q_heads, head_dim, dtype=dtype)
    key = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype)
    value = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype)
    grad_out = torch.randn(total_q_tokens, num_q_heads, head_dim, dtype=dtype)

    return query, key, value, grad_out, cu_q_lens, cu_total_seq_lens

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
):
    swa_func = MojoSWAFunction.apply

    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    o = swa_func(
        query,
        key,
        value,
        cu_q_lens,
        cu_total_seq_lens,
        True,
        local_window,
        global_window,
        softmax_scale,
        gqa_interleave,
        True,
    )


def _run_swa_perf(input_shape, gqa_interleave, global_window, local_window, trace_dir):
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
        )
    swa_func = MojoSWAFunction.apply

    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

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
            o = swa_func(
                query,
                key,
                value,
                cu_q_lens,
                cu_total_seq_lens,
                True,
                local_window,
                global_window,
                softmax_scale,
                gqa_interleave,
                True,
            )
            prof.step()
            torch.npu.synchronize()

    write_profile_caseid(prof)


@auto_switch_platform()
@bypass_not_implemented
def test_swa_function_perf_m_f32():
    input_shape = (2, 16, 4, 128, 1024, 0, torch.float32)
    _run_swa_perf(input_shape, True, 4, 255, "./npu_profiling/M_F32")


@auto_switch_platform()
@bypass_not_implemented
def test_swa_function_perf_m_bf16_paddim():
    input_shape = (2, 16, 4, 96, 1024, 0, torch.bfloat16)
    _run_swa_perf(input_shape, True, 4, 255, "./npu_profiling/M_BF16_PADDIM")


@auto_switch_platform()
@bypass_not_implemented
def test_swa_function_perf_m_bf16_long():
    input_shape = (2, 16, 4, 128, 4096, 0, torch.bfloat16)
    _run_swa_perf(input_shape, True, 4, 255, "./npu_profiling/M_BF16_LONG")


@auto_switch_platform()
@bypass_not_implemented
def test_swa_function_perf_m_f32_no_interleave():
    input_shape = (2, 16, 4, 128, 1024, 0, torch.float32)
    _run_swa_perf(input_shape, False, 4, 1023, "./npu_profiling/M_F32_NO_INTERLEAVE")

@auto_switch_platform()
@bypass_not_implemented
def test_swa_function_perf_m_bf16_no_interleave():
    input_shape = (1, 12, 4, 128, 131072, 0, torch.bfloat16)
    _run_swa_perf(input_shape, False, 4, 1023, "./npu_profiling/M_BF16_NO_INTERLEAVE")