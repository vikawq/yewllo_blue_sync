"""按元素个数扫 torch 原生算子, 找出仿真上"从能跑到挂死"的那条线, 以及那一刻的 programs 数。

已知 (2026-10-23, _t_sim_torch_op_bisect.py):
    smoke_ones_f32  16 元素          programs=1   ok  1.4s
    smoke_add_f32   16 元素          programs=1   ok  0.6s
    ones_bool       1536x1792=2.75M  programs=64  Scalar invalid addr 刷屏 + 挂死 (没有输入数据也坏)
    clone_q_f32     2048x16x128=4M   programs=45  同上 + unmapped 地址 = q 的 fp32 数据;
                                                  与 pytest 那次的 44 个地址完全一致 -> pytest 坏在 q.clone()
能跑的全是单 block, 坏的全是多 block。本脚本用来确认分界线到底卡在哪:
  * 一出现 programs>1 就坏         -> 多 block 下发 (参数/tiling 按 blk_id 寻址) 有问题
  * programs<=32 都好, >32 才坏    -> program pool 超过常驻 core_group 数时的调度/参数布局有问题
  * 跟 programs 无关, 跟字节数有关  -> 另找方向 (大块搬运 / 内存映射)

每个 (算子, 元素数) 在独立子进程里跑, 超时直接杀掉整个进程组, 扫描自动继续, 不用手动 Ctrl-C。
某个算子一旦挂死/出错, 更大的尺寸就跳过 (省时间)。最后打印汇总表。

用法 (cwd 留在 NPU-Sim):
    python _t_sim_size_sweep.py                          # 默认三个算子 x 默认尺寸
    python _t_sim_size_sweep.py clone_f32                # 只扫一个算子
    python _t_sim_size_sweep.py clone_f32 8192 32768     # 指定尺寸
    SWEEP_TIMEOUT=600 python _t_sim_size_sweep.py        # 单个尺寸的超时秒数, 默认 300
    SWEEP_NO_STOP=1 python _t_sim_size_sweep.py ...      # 出问题后不跳过更大尺寸 (看 invalid 数随 block 数怎么涨)

第一轮结果 (2026-10-23): 分界线不在 32 —— ones_f32 programs=2 过、programs=8 挂;
ones_bool 同样 2 过 8 挂; clone_f32 <=32768 元素根本不起 kernel (走 D2D 拷贝), 131072 起 64 block 即坏。
fill 类算子每 block 正好 4KB (f32 1024 元素 / bool 4096 元素), 所以 ones_f32 n=1024*k 就是 k 个 block。

子进程用 stdbuf -oL 跑 (若存在), 这样被杀之前仿真器打的 programs=N 也能抓到。
"""
import os
import re
import shutil
import signal
import subprocess
import sys
import time

OPS = ["ones_f32", "ones_bool", "clone_f32"]
DEFAULT_SIZES = [16, 256, 2048, 8192, 32768, 131072, 524288, 2097152, 4194304]
RE_PROGRAMS = re.compile(r"program pool: programs=(\d+) resident=(\d+)")


# ---------------- 子进程: 只跑一个 (算子, 元素数) ----------------

def child(op, n):
    import torch
    import torch_npu  # noqa: F401

    dev = "npu"
    if op == "ones_f32":
        dev_fn = lambda: torch.ones(n, dtype=torch.float32, device=dev)  # noqa: E731
        ref = torch.ones(n, dtype=torch.float32)
    elif op == "ones_bool":
        dev_fn = lambda: torch.ones(n, dtype=torch.bool, device=dev)  # noqa: E731
        ref = torch.ones(n, dtype=torch.bool)
    elif op == "clone_f32":
        torch.manual_seed(43)
        x = torch.randn(n, dtype=torch.float32)
        x_d = x.to(dev)
        torch.npu.synchronize()
        dev_fn = lambda: x_d.clone()  # noqa: E731
        ref = x
    else:
        sys.exit(f"unknown op {op}")

    print(f"[CHILD] start {op} n={n}", flush=True)
    t0 = time.perf_counter()
    out = dev_fn()
    torch.npu.synchronize()
    dt = time.perf_counter() - t0
    ok = torch.equal(out.cpu(), ref)
    print(f"[RESULT] {'ok' if ok else 'FAIL'} {dt:.2f}", flush=True)


# ---------------- 父进程: 调度 + 超时 + 汇总 (只用标准库, 不碰 torch) ----------------

def run_one(op, n, timeout):
    cmd = [sys.executable, os.path.abspath(__file__), "--one", op, str(n)]
    if shutil.which("stdbuf"):
        cmd = ["stdbuf", "-oL", "-eL"] + cmd
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace", start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
        status = None
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        status = "TIMEOUT"
    wall = time.perf_counter() - t0

    m = RE_PROGRAMS.search(out or "")
    programs = f"{m.group(1)}/{m.group(2)}" if m else "-"
    n_invalid = (out or "").count("Scalar: invalid addr")
    n_unmapped = (out or "").count("unmapped")
    r = re.search(r"\[RESULT\] (ok|FAIL) ([\d.]+)", out or "")
    if status is None:
        status = r.group(1) if r else f"CRASH(rc={proc.returncode})"
    kernel_s = r.group(2) if r else "-"
    return {"op": op, "n": n, "programs": programs, "status": status,
            "kernel_s": kernel_s, "wall_s": f"{wall:.0f}",
            "invalid": n_invalid, "unmapped": n_unmapped}


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--one":
        child(sys.argv[2], int(sys.argv[3]))
        return

    args = sys.argv[1:]
    ops = [a for a in args if a in OPS] or OPS
    sizes = [int(a) for a in args if a.isdigit()] or DEFAULT_SIZES
    bad = [a for a in args if a not in OPS and not a.isdigit()]
    if bad:
        sys.exit(f"不认识的参数 {bad}; 算子只能是 {OPS}")
    timeout = float(os.environ.get("SWEEP_TIMEOUT", "300"))
    no_stop = os.environ.get("SWEEP_NO_STOP", "") not in ("", "0")

    print(f"[sweep] ops={ops} sizes={sizes} timeout={timeout:.0f}s stdbuf={'yes' if shutil.which('stdbuf') else 'no'}",
          flush=True)
    rows = []
    for op in ops:
        for n in sizes:
            print(f"[sweep] {op} n={n} ...", end=" ", flush=True)
            row = run_one(op, n, timeout)
            rows.append(row)
            print(f"{row['status']}  programs={row['programs']}  invalid_addr={row['invalid']}  "
                  f"unmapped={row['unmapped']}  wall={row['wall_s']}s", flush=True)
            if row["status"] != "ok" and not no_stop:
                print(f"[sweep] {op} 在 n={n} 出问题, 跳过更大的尺寸", flush=True)
                break

    print("\n==== 汇总 (programs 列 = 实际 block 数 / 常驻 core_group 数) ====")
    print(f"{'op':<10} {'numel':>9} {'programs':>9} {'status':<12} {'kernel_s':>8} {'invalid':>7} {'unmapped':>8}")
    for r in rows:
        print(f"{r['op']:<10} {r['n']:>9} {r['programs']:>9} {r['status']:<12} {r['kernel_s']:>8} "
              f"{r['invalid']:>7} {r['unmapped']:>8}")


if __name__ == "__main__":
    main()
