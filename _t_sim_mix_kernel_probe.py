"""最小 A/B: 同一个 kernel 骨架, 一个纯 AIV、一个用 tl.dot 逼出 mix(AIC+AIV), 看是不是 mix 就坏。

为什么要做这个:
    之前那组「Triton 路径干净」的结论 (grid=1/2/4/5/8 全过) 全部来自 _mark_blocks ——
    metadata 显示它 mix_mode=aiv、kick 时 1 subcore。而 SWA 的 _swa_fwd_kernel 是
    mix_mode=mix、3 subcore, 执行期刷:
        [ERROR] scalar_func.h:413  scalar_rem [SCALAR_FUNC]: Error: div by 0!
        [ERROR] scalar_func.h:1880 memory_map_calc_op0 Scalar: invalid addr.
    div by 0 对得上 IR 里的 `block_idx % arg24`, 而 arg24 就是 grid[0] —— 也就是
    kernel 拿到的 grid 是 0。但 grid_offset 已经验证是对的 (144, 和 MLIR 签名逐字节
    吻合; 对照组 _mark_blocks 是 24 也对), 而且仿真器确实按 grid kick 了 32 个 program。
    所以剩下唯一的不对称就是 mix。

两个 kernel 除了 tl.dot 之外结构完全一样, 每个 block 往自己那一格写
    BLOCK**3 + pid + 1        (mix 版; BLOCK**3 来自全 1 矩阵的 dot 再求和)
    pid + 1                   (aiv 版)
跑完逐 block 核对, 所以 pid 不对 / grid 不对会直接暴露成某几格的值错。

用法 (cwd 留在 NPU-Sim):
    python _t_sim_mix_kernel_probe.py                 # aiv 和 mix 各扫 grid = 1 2 4 8
    python _t_sim_mix_kernel_probe.py mix             # 只跑 mix
    python _t_sim_mix_kernel_probe.py mix 1           # 只跑 mix, grid=1
    MIX_BLOCK=64 python _t_sim_mix_kernel_probe.py    # 换 dot 的分块 (默认 32)
    SWEEP_TIMEOUT=120 python _t_sim_mix_kernel_probe.py

判读:
  * aiv 全过 + mix 全挂 (连 grid=1 都挂) -> 和 block 数无关, **mix kernel 这条路本身不通**,
    这就是给维护方的最小复现; 同时把「Triton 干净」修正为「纯 AIV 干净」
  * aiv 全过 + mix 只在 grid>=N 挂       -> mix 叠加了 block 数问题
  * mix 也全过                            -> 问题在 SWA kernel 自身别的特性上, 这条排除
每个用例跑完会把 hal_policy 落下的 mix_mode / grid_offset 一并读出来, 确认 mix 真的被
编成了 mix (别白跑一轮)。
"""
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

MODES = ("aiv", "mix", "gm", "gmmix")
DEFAULT_GRIDS = [1, 2, 4, 8]
RE_PROGRAMS = re.compile(r"program pool: programs=(\d+) resident=(\d+)")
RE_SUBCORE = re.compile(r"kicked program \d+ \(blk_id=\d+, (\d+) subcore")
KERNEL_NAMES = {"aiv": "_probe_aiv", "mix": "_probe_mix", "gm": "_probe_gm",
                "gmmix": "_probe_gmmix"}
GM_BOUNDS = [0, 128, 256]       # 模仿 cu_q_lens: bsz=2, 每段 128
GMMIX_BOUNDS = [0, 8, 16]       # gmmix 每轮内层都要做一次 dot, 段长收小免得仿真跑太久


def _ts():
    return time.strftime("%H:%M:%S")


# ---------------- 子进程 ----------------

def child(mode, grid, block):
    global triton, tl
    import torch
    import torch_npu  # noqa: F401
    import triton
    import triton.language as tl

    @triton.jit
    def _probe_aiv(out_ptr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        tl.store(out_ptr + pid, (pid + 1).to(tl.float32))

    @triton.jit
    def _probe_gm(bounds_ptr, out_ptr, n_iter, BLOCK: tl.constexpr):
        """把 SWA 的三个结构特征浓缩进来, 其余什么都不做:

        1. 外层 for 的上界 n_iter 是运行期 i32 参数 (SWA 那边是 bsz)
        2. 循环边界从 GM 标量读出来 (SWA 那边是 cu_q_lens / cu_total_seq_lens)
        3. 内层 for 的步长 = tl.num_programs(0) (SWA 那边 `scf.for step %arg24`)

        第 3 条正好会生成 `remsi block_idx, %arg24` 和以 %arg24 为步长的循环 ——
        也就是 SWA 报 `scalar_rem div by 0` 的那两处。
        """
        pid = tl.program_id(0)
        n_progs = tl.num_programs(0)
        total = 0
        for i in range(n_iter):
            lo = tl.load(bounds_ptr + i)
            hi = tl.load(bounds_ptr + i + 1)
            for t in range(pid, hi - lo, n_progs):
                total += t
        tl.store(out_ptr + pid, total)

    @triton.jit
    def _probe_gmmix(bounds_ptr, a_ptr, b_ptr, out_ptr, n_iter, BLOCK: tl.constexpr):
        """gm + mix 同时出现 —— 这是特征矩阵里唯一没测过的那一格。

        关键是 tl.dot **放在以 num_programs(0) 为步长的循环里面**: 这样 AIC 那一侧
        也必须依赖 %arg24 (grid[0]) 才知道自己该做几轮。_probe_mix 的 dot 在循环外,
        AIC 侧压根不需要 arg24; _probe_gm 需要 arg24 但是纯 AIV。SWA 两样都有。
        """
        pid = tl.program_id(0)
        n_progs = tl.num_programs(0)
        offs = tl.arange(0, BLOCK)
        idx = offs[:, None] * BLOCK + offs[None, :]
        a = tl.load(a_ptr + idx)
        b = tl.load(b_ptr + idx)
        acc = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
        total = 0
        for i in range(n_iter):
            lo = tl.load(bounds_ptr + i)
            hi = tl.load(bounds_ptr + i + 1)
            for t in range(pid, hi - lo, n_progs):
                acc += tl.dot(a, b)
                total += t
        tl.store(out_ptr + pid, tl.sum(acc) + total.to(tl.float32))

    @triton.jit
    def _probe_mix(a_ptr, b_ptr, out_ptr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        idx = offs[:, None] * BLOCK + offs[None, :]
        a = tl.load(a_ptr + idx)
        b = tl.load(b_ptr + idx)
        c = tl.dot(a, b)                      # 这一句把 kernel 逼成 mix (AIC+AIV)
        tl.store(out_ptr + pid, tl.sum(c) + (pid + 1).to(tl.float32))

    out_dtype = torch.int32 if mode == "gm" else torch.float32
    out = torch.zeros(grid, dtype=out_dtype).npu()
    torch.npu.synchronize()
    print(f"[CHILD] mode={mode} grid={grid} BLOCK={block}", flush=True)

    t0 = time.perf_counter()
    if mode == "gmmix":
        bounds = torch.tensor(GMMIX_BOUNDS, dtype=torch.int32).npu()
        a = torch.ones(block, block, dtype=torch.float32).npu()
        b = torch.ones(block, block, dtype=torch.float32).npu()
        torch.npu.synchronize()
        n_iter = len(GMMIX_BOUNDS) - 1
        _probe_gmmix[(grid,)](bounds, a, b, out, n_iter, BLOCK=block)
        expect = []
        for pid in range(grid):
            n_dots, total = 0, 0
            for i in range(n_iter):
                steps = list(range(pid, GMMIX_BOUNDS[i + 1] - GMMIX_BOUNDS[i], grid))
                n_dots += len(steps)
                total += sum(steps)
            # a,b 全 1 -> dot 每个元素 = block; acc 求和 = n_dots * block**3
            expect.append(float(n_dots * block ** 3 + total))
    elif mode == "gm":
        bounds = torch.tensor(GM_BOUNDS, dtype=torch.int32).npu()
        torch.npu.synchronize()
        n_iter = len(GM_BOUNDS) - 1
        _probe_gm[(grid,)](bounds, out, n_iter, BLOCK=block)
        expect = []
        for pid in range(grid):
            total = 0
            for i in range(n_iter):
                total += sum(range(pid, GM_BOUNDS[i + 1] - GM_BOUNDS[i], grid))
            expect.append(total)
    elif mode == "aiv":
        _probe_aiv[(grid,)](out, BLOCK=block)
        expect = [float(i + 1) for i in range(grid)]
    else:
        a = torch.ones(block, block, dtype=torch.float32).npu()
        b = torch.ones(block, block, dtype=torch.float32).npu()
        torch.npu.synchronize()
        _probe_mix[(grid,)](a, b, out, BLOCK=block)
        expect = [float(block ** 3 + i + 1) for i in range(grid)]
    torch.npu.synchronize()
    dt = time.perf_counter() - t0

    got = out.cpu().tolist()
    bad = [i for i, (g, e) in enumerate(zip(got, expect)) if g != e]
    print(f"[RESULT] {'ok' if not bad else 'FAIL'} {dt:.2f} bad={bad}", flush=True)
    if bad:
        show = bad[:4]
        print(f"[DETAIL] 期望 {[expect[i] for i in show]} 实得 {[got[i] for i in show]}", flush=True)


# ---------------- 父进程 (只用标准库) ----------------

def newest_meta(kernel_name):
    """把 hal_policy 落下的 metadata.json 里这个 kernel 最新那份找出来。"""
    import glob
    import json

    roots = []
    if os.environ.get("NPUSIM_SESSION_DIR"):
        roots.append(os.environ["NPUSIM_SESSION_DIR"])
    roots += glob.glob(os.path.join(os.getcwd(), "build", "sessions", "*"))
    paths = {p for r in roots
             for p in glob.glob(os.path.join(r, "**", "native-triton", "*", "metadata.json"),
                                recursive=True)}
    best = None
    for path in sorted(paths, key=os.path.getmtime, reverse=True):
        try:
            data = json.loads(open(path, encoding="utf-8").read())
        except (OSError, ValueError):
            continue
        if data.get("kernel") == kernel_name:
            best = data
            break
    if best is None:
        return "-", "-"
    return (best.get("metadata", {}).get("mix_mode", "?"), best.get("grid_offset", "?"))


def spawn(mode, grid, block, timeout, beat=20.0, echo=False):
    cmd = [sys.executable, os.path.abspath(__file__), "--one", mode, str(grid), str(block)]
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
            if echo or line.startswith(("[CHILD]", "[RESULT]", "[DETAIL]")):
                print(f"   | {line}", flush=True)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    deadline, last, timed_out = t0 + timeout, t0, False
    while proc.poll() is None:
        now = time.perf_counter()
        if now >= deadline:
            print(f"   ... {timeout:.0f}s 超时, 杀掉子进程组", flush=True)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, AttributeError):
                proc.kill()
            timed_out = True
            break
        if now - last >= beat:
            last = now
            blob = "\n".join(lines)
            print(f"   ... 已跑 {now - t0:.0f}s (还剩 {deadline - now:.0f}s), "
                  f"div_by_0={blob.count('div by 0')}, "
                  f"invalid_addr={blob.count('Scalar: invalid addr')}", flush=True)
        time.sleep(1.0)
    thread.join(timeout=3)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print("   ... 子进程未在 10s 内回收, 继续", flush=True)
    try:
        proc.stdout.close()
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines), timed_out, proc.returncode, time.perf_counter() - t0


def main():
    args = sys.argv[1:]
    if args and args[0] == "--one":
        child(args[1], int(args[2]), int(args[3]))
        return

    modes = [a for a in args if a in MODES] or list(MODES)
    grids = [int(a) for a in args if a.isdigit()] or DEFAULT_GRIDS
    bad = [a for a in args if a not in MODES and not a.isdigit()]
    if bad:
        sys.exit(f"不认识的参数 {bad}; mode 只能是 {MODES}")
    block = int(os.environ.get("MIX_BLOCK", "32"))
    timeout = float(os.environ.get("SWEEP_TIMEOUT", "180"))
    echo = os.environ.get("SWEEP_ECHO", "") not in ("", "0")
    print(f"[probe] modes={modes} grids={grids} BLOCK={block} timeout={timeout:.0f}s", flush=True)

    rows = []
    for mode in modes:
        for grid in grids:
            print(f"\n[probe] {mode} grid={grid} ...", flush=True)
            out, timed_out, rc, wall = spawn(mode, grid, block, timeout, echo=echo)
            m = RE_PROGRAMS.search(out)
            sub = RE_SUBCORE.search(out)
            r = re.search(r"\[RESULT\] (ok|FAIL) ([\d.]+) bad=(\[.*?\])", out)
            status = "TIMEOUT" if timed_out else (r.group(1) if r else f"CRASH(rc={rc})")
            mix_mode, grid_off = newest_meta(KERNEL_NAMES[mode])
            row = {"mode": mode, "grid": grid, "status": status,
                   "programs": f"{m.group(1)}/{m.group(2)}" if m else "-",
                   "subcore": sub.group(1) if sub else "-",
                   "mix_mode": mix_mode, "grid_offset": grid_off,
                   "div0": out.count("div by 0"),
                   "invalid": out.count("Scalar: invalid addr"),
                   "bad": r.group(3) if r else "-", "wall": f"{wall:.0f}"}
            rows.append(row)
            print(f"[probe] {mode} grid={grid} -> {status}  programs={row['programs']}  "
                  f"subcore={row['subcore']}  mix_mode={mix_mode}  grid_offset={grid_off}  "
                  f"div_by_0={row['div0']}  invalid_addr={row['invalid']}  wall={row['wall']}s",
                  flush=True)
            if status.startswith("CRASH"):
                print("   ---- 子进程尾部 ----\n   " + "\n   ".join(out.splitlines()[-20:]),
                      flush=True)

    print("\n==== 汇总 ====")
    print(f"{'mode':<5} {'grid':>4} {'status':<9} {'programs':>9} {'sub':>3} {'mix_mode':>8} "
          f"{'g_off':>5} {'div0':>5} {'invalid':>7} {'wall':>5}  bad_blocks")
    for r in rows:
        print(f"{r['mode']:<5} {r['grid']:>4} {r['status']:<9} {r['programs']:>9} "
              f"{r['subcore']:>3} {str(r['mix_mode']):>8} {str(r['grid_offset']):>5} "
              f"{r['div0']:>5} {r['invalid']:>7} {r['wall']:>5}  {r['bad']}")
    print("\n提醒: mix 那几行的 mix_mode 必须真的是 'mix'、subcore 是 3, 否则 tl.dot 没把它逼成混合核,")
    print("      这一轮就不算验到东西 —— 可以调 MIX_BLOCK 再试。")


if __name__ == "__main__":
    main()
