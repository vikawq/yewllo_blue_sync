"""用最简单的 Triton kernel 测多 block 下发: 第 4 号 block 之后坏的是"下发路径", 还是只有 torch 原生算子坏。

已知 (2026-10-23, _t_sim_size_sweep.py, 本机 hal-direct 路径):
    ones_f32 programs=1..4 全过; programs=5/6/7/8 全挂, Scalar invalid addr 正好 5/10/15/20 条
    -> invalid = 5 * (k - 4): block 0~3 干净, 第 4 号起每多一个 block 多 5 条错
本脚本让每个 block 往自己那 8 个 int32 里写 (pid + 1), 跑完逐 block 检查:
  * Triton grid=5 也只坏 block 4+     -> 毛病在下发路径 (按 blk_id / core_group 4+ 的参数区或配置), 与算子无关
  * Triton grid=8 全对               -> 只有 torch 原生算子坏, 更可能是它的 tiling/workspace 按"4 核"分配
另外先打印 host 侧看到的设备属性 (torch_npu 与 triton 各自的核数), 看有没有哪里报的是 4。

用法 (cwd 留在 NPU-Sim):
    python _t_sim_triton_grid_probe.py                 # 设备属性 + grid = 1 2 4 5 8
    python _t_sim_triton_grid_probe.py 5 6             # 只测指定 grid
    SWEEP_TIMEOUT=600 python _t_sim_triton_grid_probe.py   # 单个用例超时秒数, 默认 300 (首次要编译)

每个用例在独立子进程里跑, 超时整组杀掉后自动继续。输入输出张量都在 CPU 上建好再 H2D/D2H,
不经过任何多 block 的 torch 原生算子。
"""
import os
import re
import shutil
import signal
import subprocess
import sys
import time

DEFAULT_GRIDS = [1, 2, 4, 5, 8]
BLOCK = 8  # 8 个 int32 = 32B, 对齐
RE_PROGRAMS = re.compile(r"program pool: programs=(\d+) resident=(\d+)")


# ---------------- 子进程 ----------------

def child_props():
    import torch
    import torch_npu  # noqa: F401

    try:
        print(f"[PROPS] torch.npu.get_device_properties(0) = {torch.npu.get_device_properties(0)}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[PROPS] torch.npu.get_device_properties 失败: {e!r}", flush=True)
    try:
        import triton

        props = triton.runtime.driver.active.utils.get_device_properties("npu")
        print(f"[PROPS] triton get_device_properties('npu') = {props}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[PROPS] triton get_device_properties 失败: {e!r}", flush=True)


def child_grid(k):
    # kernel 里的 tl 要能从 __globals__ 里找到, 所以绑成模块全局
    global triton, tl
    import torch
    import torch_npu  # noqa: F401
    import triton
    import triton.language as tl

    @triton.jit
    def _mark_blocks(out_ptr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        vals = tl.zeros([BLOCK], dtype=tl.int32) + (pid + 1)
        tl.store(out_ptr + offs, vals)

    out = torch.zeros(k * BLOCK, dtype=torch.int32).to("npu")  # CPU 建好再 H2D
    torch.npu.synchronize()
    print(f"[CHILD] launch grid=({k},)", flush=True)
    t0 = time.perf_counter()
    _mark_blocks[(k,)](out, BLOCK=BLOCK)
    torch.npu.synchronize()
    dt = time.perf_counter() - t0
    res = out.cpu().view(k, BLOCK)
    good = [i for i in range(k) if bool((res[i] == i + 1).all())]
    bad = [i for i in range(k) if i not in good]
    print(f"[RESULT] {'ok' if not bad else 'FAIL'} {dt:.2f} good={good} bad={bad}", flush=True)
    if bad:
        print(f"[DETAIL] first bad block {bad[0]} -> {res[bad[0]].tolist()}", flush=True)


# ---------------- 父进程 (只用标准库) ----------------

def spawn(args, timeout):
    cmd = [sys.executable, os.path.abspath(__file__)] + args
    if shutil.which("stdbuf"):
        cmd = ["stdbuf", "-oL", "-eL"] + cmd
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace", start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        timed_out = True
    return out or "", timed_out, proc.returncode, time.perf_counter() - t0


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--props":
        child_props()
        return
    if len(sys.argv) >= 3 and sys.argv[1] == "--grid":
        child_grid(int(sys.argv[2]))
        return

    args = sys.argv[1:]
    if any(not a.isdigit() for a in args):
        sys.exit("参数只能是 grid 大小 (正整数)")
    grids = [int(a) for a in args] or DEFAULT_GRIDS
    timeout = float(os.environ.get("SWEEP_TIMEOUT", "300"))

    out, timed_out, rc, wall = spawn(["--props"], 120)
    for line in out.splitlines():
        if line.startswith("[PROPS]"):
            print(line, flush=True)
    if timed_out or not any(l.startswith("[PROPS]") for l in out.splitlines()):
        print(f"[props] 没拿到 (timeout={timed_out} rc={rc}); 尾部输出:\n" + "\n".join(out.splitlines()[-15:]), flush=True)

    rows = []
    for k in grids:
        print(f"[grid] grid={k} ...", end=" ", flush=True)
        out, timed_out, rc, wall = spawn(["--grid", str(k)], timeout)
        m = RE_PROGRAMS.search(out)
        programs = f"{m.group(1)}/{m.group(2)}" if m else "-"
        r = re.search(r"\[RESULT\] (ok|FAIL) ([\d.]+) good=(\[.*?\]) bad=(\[.*?\])", out)
        status = "TIMEOUT" if timed_out else (r.group(1) if r else f"CRASH(rc={rc})")
        row = {
            "grid": k, "programs": programs, "status": status,
            "kernel_s": r.group(2) if r else "-",
            "good": r.group(3) if r else "-", "bad": r.group(4) if r else "-",
            "invalid": out.count("Scalar: invalid addr"), "unmapped": out.count("unmapped"),
        }
        rows.append(row)
        print(f"{status}  programs={programs}  invalid_addr={row['invalid']}  bad_blocks={row['bad']}  "
              f"wall={wall:.0f}s", flush=True)
        if status.startswith("CRASH") or (status == "TIMEOUT" and programs == "-"):
            # 没起到 kernel 就死了: 多半是编译/环境问题, 把尾巴打出来
            print("  ---- 子进程尾部输出 ----\n  " + "\n  ".join(out.splitlines()[-20:]), flush=True)

    print("\n==== 汇总 ====")
    print(f"{'grid':>4} {'programs':>9} {'status':<12} {'kernel_s':>8} {'invalid':>7} {'unmapped':>8}  bad_blocks")
    for r in rows:
        print(f"{r['grid']:>4} {r['programs']:>9} {r['status']:<12} {r['kernel_s']:>8} {r['invalid']:>7} "
              f"{r['unmapped']:>8}  {r['bad']}")


if __name__ == "__main__":
    main()
