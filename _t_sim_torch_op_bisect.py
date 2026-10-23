"""逐个 torch 算子在仿真上跑一遍, 找出是哪个原生算子挂住 / 报 Scalar invalid addr。

背景: _t_swa_fill_probe.py 四次运行都卡在
    swa.py:80  causal = torch.ones(M, N, dtype=torch.bool).tril()
—— 还没走到 Triton 的 _swa_fwd_kernel。get_mask_causal_with_window 里的 torch.ones / zeros
没写 device, 本意是在 CPU 上拼好 mask 再 .to(npu); 但 pytest conftest (以及 probe) 调了
torch.set_default_device("npu"), 于是这一串 bool 小算子全部经 NPU-Sim 的 PTA 前端
(code/frontend/common/frontends/pta/audit.py) 下到仿真器上跑。

本脚本不依赖 mojo_opset, 也不设 default device: 每一步的输入都在 CPU 上算好再 H2D 搬上去,
只让"被测的那一个算子"在 npu 上执行, 然后 synchronize + 搬回 CPU 和 CPU 结果逐元素比对。
所以每一步互相独立, 哪一步只打出 [start] 没有 [ ok ] / [FAIL], 挂的就是它。

用法 (cwd 留在 NPU-Sim, 与跑 pytest 时相同):
    python _t_sim_torch_op_bisect.py                     # 按顺序全跑
    python _t_sim_torch_op_bisect.py tril_bool           # 只跑某几步
    python _t_sim_torch_op_bisect.py --list              # 列出所有步骤

步骤顺序 = swa 前向路径上实际会发生的 npu 算子顺序, 前面先放两个最小冒烟:
    smoke_*      : 极小张量, 判断仿真上的 torch 原生算子是不是整体就不通
    mask 那一串  : 与 get_mask_causal_with_window 逐行对应 (M=1536, N=1792, local=255, global=4)
    lse/o 初始化 : swa_fwd_impl 里 kernel 之前的 zeros_like / zeros + (-inf)
    clone_q_f32  : pytest 里 test_swa_function 第一个碰 fp32 randn 数据的算子 (q.clone()),
                   用来验证 "programs=45 + 地址=fp32 数据" 那次是不是它
"""
import sys
import time

import torch
import torch_npu  # noqa: F401

DEV = "npu"

# get_mask_causal_with_window(256, 256, local=255, global=4) 的实际尺寸
LOCAL, GLOBAL = 255, 4
M = (GLOBAL + LOCAL + 4 * 256 + 256 - 1) // 256 * 256  # 1536
N = (GLOBAL + LOCAL + 5 * 256 + 256 - 1) // 256 * 256  # 1792

# M_F32 用例: B=2, Q_LEN=1024 -> 2048 tokens, 16 q heads, D=128
TOT_Q, Q_H, D = 2048, 16, 128


def _ts():
    return time.strftime("%H:%M:%S")


def step_smoke_ones_f32():
    return lambda: torch.ones(16, device=DEV), lambda: torch.ones(16)


def step_smoke_add_f32():
    a = torch.arange(16, dtype=torch.float32)
    a_d = a.to(DEV)
    return lambda: a_d + 1.0, lambda: a + 1.0


def step_ones_bool():
    return (lambda: torch.ones(M, N, dtype=torch.bool, device=DEV),
            lambda: torch.ones(M, N, dtype=torch.bool))


def step_tril_bool():
    x = torch.ones(M, N, dtype=torch.bool)
    x_d = x.to(DEV)
    return lambda: x_d.tril(), lambda: x.tril()


def step_zeros_bool():
    return (lambda: torch.zeros(M, N, dtype=torch.bool, device=DEV),
            lambda: torch.zeros(M, N, dtype=torch.bool))


def step_slice_assign_bool():
    x = torch.zeros(M, N, dtype=torch.bool)

    def dev():
        y = x.to(DEV)
        y[:, :GLOBAL] = True
        return y

    def ref():
        y = x.clone()
        y[:, :GLOBAL] = True
        return y

    return dev, ref


def step_triu_bool():
    x = torch.ones(M, N, dtype=torch.bool)
    x_d = x.to(DEV)
    return lambda: x_d.triu(diagonal=-LOCAL), lambda: x.triu(diagonal=-LOCAL)


def _bands():
    causal = torch.ones(M, N, dtype=torch.bool).tril()
    sink = torch.zeros(M, N, dtype=torch.bool)
    sink[:, :GLOBAL] = True
    local = torch.ones(M, N, dtype=torch.bool).triu(diagonal=-LOCAL)
    return causal, sink, local


def step_or_bool():
    _, sink, local = _bands()
    s_d, l_d = sink.to(DEV), local.to(DEV)
    return lambda: s_d | l_d, lambda: sink | local


def step_and_bool():
    causal, sink, local = _bands()
    sl = sink | local
    c_d, sl_d = causal.to(DEV), sl.to(DEV)
    return lambda: c_d & sl_d, lambda: causal & sl


def step_zeros_like_q_f32():
    q = torch.empty(TOT_Q, Q_H, D, dtype=torch.float32)
    q_d = q.to(DEV)
    return lambda: torch.zeros_like(q_d), lambda: torch.zeros_like(q)


def step_lse_init_f32():
    return (lambda: torch.zeros((Q_H, TOT_Q), dtype=torch.float32, device=DEV) + float("-inf"),
            lambda: torch.zeros((Q_H, TOT_Q), dtype=torch.float32) + float("-inf"))


def step_clone_q_f32():
    torch.manual_seed(43)
    q = torch.randn(TOT_Q, Q_H, D, dtype=torch.float32)
    q_d = q.to(DEV)
    return lambda: q_d.clone(), lambda: q.clone()


STEPS = [
    ("smoke_ones_f32", step_smoke_ones_f32),
    ("smoke_add_f32", step_smoke_add_f32),
    ("ones_bool", step_ones_bool),
    ("tril_bool", step_tril_bool),
    ("zeros_bool", step_zeros_bool),
    ("slice_assign_bool", step_slice_assign_bool),
    ("triu_bool", step_triu_bool),
    ("or_bool", step_or_bool),
    ("and_bool", step_and_bool),
    ("zeros_like_q_f32", step_zeros_like_q_f32),
    ("lse_init_f32", step_lse_init_f32),
    ("clone_q_f32", step_clone_q_f32),
]


def run(name, builder):
    dev_fn, ref_fn = builder()
    print(f"[{_ts()}] [start] {name}", flush=True)
    t0 = time.perf_counter()
    out = dev_fn()
    torch.npu.synchronize()
    dt = time.perf_counter() - t0
    got = out.cpu()
    ref = ref_fn()
    if got.shape == ref.shape and torch.equal(got, ref):
        print(f"[{_ts()}] [ ok ] {name}  {dt:.1f}s", flush=True)
        return True
    bad = (got != ref).sum().item() if got.shape == ref.shape else "shape"
    print(f"[{_ts()}] [FAIL] {name}  {dt:.1f}s  shape={tuple(got.shape)} vs {tuple(ref.shape)}  "
          f"mismatch={bad}", flush=True)
    return False


def main():
    args = sys.argv[1:]
    table = dict(STEPS)
    if args == ["--list"]:
        for name, _ in STEPS:
            print(name)
        return
    unknown = [a for a in args if a not in table]
    if unknown:
        sys.exit(f"未知步骤 {unknown}, 用 --list 查看")
    chosen = [(n, table[n]) for n in (args or [n for n, _ in STEPS])]

    print(f"[info] M={M} N={N} (mask), q=({TOT_Q},{Q_H},{D}) fp32; 不设 default device", flush=True)
    for name, builder in chosen:
        run(name, builder)
    print(f"[{_ts()}] [done] 全部步骤返回", flush=True)


if __name__ == "__main__":
    main()
