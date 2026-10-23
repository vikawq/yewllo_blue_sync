"""逐个测 swa_fwd_impl 在启动 Triton kernel 之前做的那些 torch 原生算子, 最后试跑真正的 SWA kernel。

已确立的事实 (2026-10-23, 本机 hal-direct 路径):
    Triton kernel  grid=1/2/4/5/8 全过, 0 错     -> 下发路径没问题
    torch 原生算子 block 0~3 过, 第 5 个 block 起必挂, invalid addr = 5*(k-4)
    torch 原生 clone <=32768 元素不起 kernel (走 D2D), 131072 元素起 64 block 即坏
    设备属性正确: cube_core_num=32 vector_core_num=64 (不是 4)
所以要让 SWA 的 Triton kernel 在这台仿真上跑起来, 关键是**别让大的 torch 原生算子下到设备**。
NPU-Sim 适配器把 H2D/D2H/D2D/memset 直接接走、不起 kernel, 所以:
    torch.zeros / zeros_like 可能是 memset  -> 不起 kernel, 安全
    torch.ones / x + c / tril 等是真 kernel -> 多 block 就坏
本脚本先逐个量出来 (programs 列 = 是否起了 kernel 及 block 数), 再用 mask 缓存预热跑一次真 kernel。

mask 预热的原理: get_mask_causal_with_window 开头有一段 global 缓存命中就直接 return
(swa.py:63-70, 比较 _GLOBAL_WINDOW_SIZE/_LOCAL_WINDOW_SIZE/_BLOCK_M/_BLOCK_N)。
本脚本在 CPU 上把 mask 算好、搬上设备、写进那几个模块全局, 于是 swa_fwd_impl 里那次调用直接命中,
一个 bool 算子都不下发。**只改进程内的模块全局, 不动 swa.py 源码。**

用法 (cwd 留在 NPU-Sim):
    python _t_swa_setup_probe.py                 # 全部步骤
    python _t_swa_setup_probe.py --list
    python _t_swa_setup_probe.py swa_fwd_primed  # 只跑最后那个
    SWEEP_TIMEOUT=60 python _t_swa_setup_probe.py      # 单步超时秒数, 默认 300
    SWEEP_ECHO=1 python _t_swa_setup_probe.py          # 把子进程(含仿真器)所有输出实时转发出来

子进程的 [CHILD]/[RESULT] 行实时转发, 另外每 20 秒打一次心跳 (已跑多少秒、收到多少行、
programs、invalid_addr 累计), 所以不会再出现"卡着不动看不出在干什么"。超时后自动杀掉整个
子进程组并继续下一步, 不用手动 Ctrl-C。

判读:
  * zeros_like_q / zeros_lse  programs=- 且 ok  -> 它们是 memset, 不用管
  * lse_add_neginf 坏          -> swa_fwd_impl 里 `zeros + (-inf)` 这一句就是下一个拦路虎
  * swa_fwd_primed 跑出 ok     -> SWA 的 Triton fwd 在这台仿真上可测, 只需把 setup 挪到 CPU
  * swa_fwd_primed 仍坏且错在 torch 算子 -> 按栈继续往下挪
"""
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

# M_F32 用例: B=2 Q_LEN=1024 -> 2048 tokens, 16 q heads, 4 kv heads, D=128, fp32
TOT_Q, Q_H, KV_H, D = 2048, 16, 4, 128
LOCAL, GLOBAL_W = 255, 4
RE_PROGRAMS = re.compile(r"program pool: programs=(\d+) resident=(\d+)")

STEP_NAMES = ["zeros_like_q", "zeros_lse", "lse_add_neginf", "mask_cpu_then_h2d", "swa_fwd_primed"]


def _repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isfile(os.path.join(d, "mojo_opset", "__init__.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


# ---------------- 子进程 ----------------

def child(step):
    root = _repo_root()
    if root and root not in sys.path:
        sys.path.insert(0, root)
    import torch
    import torch_npu  # noqa: F401

    dev = "npu"
    print(f"[CHILD] {step}", flush=True)

    if step == "zeros_like_q":
        q = torch.empty(TOT_Q, Q_H, D, dtype=torch.float32).to(dev)
        torch.npu.synchronize()
        print("[CHILD] H2D 16MB 完成, 开始 zeros_like", flush=True)
        t0 = time.perf_counter()
        out = torch.zeros_like(q, memory_format=torch.contiguous_format)
        torch.npu.synchronize()
        dt = time.perf_counter() - t0
        ok = bool((out.cpu() == 0).all())

    elif step == "zeros_lse":
        t0 = time.perf_counter()
        out = torch.zeros((Q_H, TOT_Q), dtype=torch.float32, device=dev)
        torch.npu.synchronize()
        dt = time.perf_counter() - t0
        ok = bool((out.cpu() == 0).all())

    elif step == "lse_add_neginf":
        base = torch.zeros((Q_H, TOT_Q), dtype=torch.float32, device=dev)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        out = base + float("-inf")
        torch.npu.synchronize()
        dt = time.perf_counter() - t0
        ok = bool(torch.isneginf(out.cpu()).all())

    elif step == "mask_cpu_then_h2d":
        mask_cpu = _build_mask_cpu(torch)
        t0 = time.perf_counter()
        out = mask_cpu.to(dev)
        torch.npu.synchronize()
        dt = time.perf_counter() - t0
        ok = bool(torch.equal(out.cpu(), mask_cpu))

    elif step == "swa_fwd_primed":
        import math

        from mojo_opset.backends.ttx.kernels.npu import swa as swa_mod

        mask_dev = _build_mask_cpu(torch).to(dev)
        torch.npu.synchronize()
        # 命中 swa.py:63-70 的缓存分支, 这样 get_mask_causal_with_window 一个算子都不下发
        swa_mod._GLOBAL_WINDOW_SIZE = GLOBAL_W
        swa_mod._LOCAL_WINDOW_SIZE = LOCAL
        swa_mod._BLOCK_M = 256
        swa_mod._BLOCK_N = 256
        swa_mod._COMPRESSED_MASK = mask_dev
        print(f"[CHILD] mask primed on device: {tuple(mask_dev.shape)} {mask_dev.dtype}", flush=True)

        torch.manual_seed(43)
        q = torch.randn(TOT_Q, Q_H, D, dtype=torch.float32).to(dev)
        k = torch.randn(TOT_Q, KV_H, D, dtype=torch.float32).to(dev)
        v = torch.randn(TOT_Q, KV_H, D, dtype=torch.float32).to(dev)
        cu = torch.tensor([0, TOT_Q // 2, TOT_Q], dtype=torch.int32).to(dev)
        torch.npu.synchronize()

        t0 = time.perf_counter()
        o, lse = swa_mod.swa_fwd_impl(
            q=q, k=k, v=v, cu_q_lens=cu, cu_total_seq_lens=cu,
            is_causal=True, local_window_size=LOCAL, global_window_size=GLOBAL_W,
            softmax_scale=1.0 / math.sqrt(D), gqa_interleave=True, output_f32=False,
        )
        torch.npu.synchronize()
        dt = time.perf_counter() - t0
        o_c = o.cpu()
        finite = bool(torch.isfinite(o_c).all())
        nonzero = int((o_c != 0).sum())
        print(f"[CHILD] o: shape={tuple(o_c.shape)} finite={finite} nonzero={nonzero}/{o_c.numel()}", flush=True)
        ok = finite and nonzero > 0

    else:
        sys.exit(f"unknown step {step}")

    print(f"[RESULT] {'ok' if ok else 'FAIL'} {dt:.2f}", flush=True)


def _build_mask_cpu(torch):
    """照搬 swa.py::get_mask_causal_with_window 的算式, 全程在 CPU 上做。"""
    BLOCK_M = BLOCK_N = 256
    AUX = 256  # swa.AUX_MASK_SIZE
    M = (GLOBAL_W + LOCAL + 4 * max(BLOCK_M, BLOCK_N) + BLOCK_M - 1) // BLOCK_M * BLOCK_M
    N = (GLOBAL_W + LOCAL + 5 * max(BLOCK_M, BLOCK_N) + BLOCK_N - 1) // BLOCK_N * BLOCK_N
    causal = torch.ones(M, N, dtype=torch.bool).tril()
    sink = torch.zeros(M, N, dtype=torch.bool)
    sink[:, :GLOBAL_W] = True
    local = torch.ones(M, N, dtype=torch.bool).triu(diagonal=-LOCAL)
    mask = causal & (sink | local)
    out = torch.zeros(M + AUX, N + AUX, dtype=torch.bool)
    out[:M, :N] = mask
    return out


# ---------------- 父进程 (只用标准库) ----------------

def spawn(step, timeout, echo=False, beat=20.0):
    """边跑边把子进程输出收下来: [CHILD] 行实时转发, 每 beat 秒打一次心跳, 不再盲等超时。"""
    cmd = [sys.executable, os.path.abspath(__file__), "--one", step]
    if shutil.which("stdbuf"):
        cmd = ["stdbuf", "-oL", "-eL"] + cmd
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace", bufsize=1, start_new_session=True)
    lines = []

    def reader():
        for line in proc.stdout:
            line = line.rstrip("\n")
            lines.append(line)
            if echo or line.startswith("[CHILD]") or line.startswith("[RESULT]"):
                print(f"   | {line}", flush=True)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    deadline = t0 + timeout
    last = t0
    timed_out = False
    while proc.poll() is None:
        now = time.perf_counter()
        if now >= deadline:
            print(f"   ... 到 {timeout:.0f}s 超时, 杀掉子进程组", flush=True)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, AttributeError):
                proc.kill()
            timed_out = True
            if shutil.which("stdbuf"):
                # 杀的是 stdbuf, 真正的 python 子孙靠进程组一起带走 (POSIX)
                pass
            break
        if now - last >= beat:
            last = now
            invalid = sum(1 for x in lines if "Scalar: invalid addr" in x)
            m = RE_PROGRAMS.search("\n".join(lines))
            print(f"   ... 已跑 {now - t0:.0f}s (还剩 {deadline - now:.0f}s), 输出 {len(lines)} 行, "
                  f"programs={m.group(1) + '/' + m.group(2) if m else '-'}, invalid_addr={invalid}", flush=True)
        time.sleep(1.0)
    # 被杀之后别无限期等: 没有进程组的平台上可能还有孙进程捏着管道
    thread.join(timeout=3)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print("   ... 子进程未在 10s 内回收, 继续下一步", flush=True)
    try:
        proc.stdout.close()
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines), timed_out, proc.returncode, time.perf_counter() - t0


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--one":
        child(sys.argv[2])
        return
    args = sys.argv[1:]
    if args == ["--list"]:
        print("\n".join(STEP_NAMES))
        return
    bad = [a for a in args if a not in STEP_NAMES]
    if bad:
        sys.exit(f"未知步骤 {bad}, 用 --list 查看")
    steps = args or STEP_NAMES
    timeout = float(os.environ.get("SWEEP_TIMEOUT", "300"))
    echo = os.environ.get("SWEEP_ECHO", "") not in ("", "0")
    print(f"[probe] repo_root={_repo_root()} steps={steps} timeout={timeout:.0f}s echo={echo}", flush=True)

    rows = []
    for step in steps:
        print(f"[probe] {step} ...", flush=True)
        out, timed_out, rc, wall = spawn(step, timeout, echo=echo)
        m = RE_PROGRAMS.search(out)
        programs = f"{m.group(1)}/{m.group(2)}" if m else "-"
        r = re.search(r"\[RESULT\] (ok|FAIL) ([\d.]+)", out)
        status = "TIMEOUT" if timed_out else (r.group(1) if r else f"CRASH(rc={rc})")
        row = {"step": step, "programs": programs, "status": status,
               "sec": r.group(2) if r else "-",
               "invalid": out.count("Scalar: invalid addr"), "unmapped": out.count("unmapped")}
        rows.append(row)
        print(f"[probe] {step} -> {status}  kernel_launched={'no' if programs == '-' else programs}  "
              f"invalid_addr={row['invalid']}  unmapped={row['unmapped']}  wall={wall:.0f}s", flush=True)
        if status.startswith("CRASH"):
            print("   ---- 子进程尾部 ----\n   " + "\n   ".join(out.splitlines()[-20:]), flush=True)

    print("\n==== 汇总 (programs='-' 表示没起 kernel, 走的是 memset/D2D) ====")
    print(f"{'step':<20} {'programs':>9} {'status':<12} {'sec':>7} {'invalid':>7} {'unmapped':>8}")
    for r in rows:
        print(f"{r['step']:<20} {r['programs']:>9} {r['status']:<12} {r['sec']:>7} {r['invalid']:>7} {r['unmapped']:>8}")


if __name__ == "__main__":
    main()
