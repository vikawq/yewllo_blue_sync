"""从 test_swa_fwd_bwd_950_perf.py 生成一组"逐步注释"的变体, 定位到底哪一条语句 rejected + 卡死。

原文件一个字节都不改: 变体是另存的新文件, 且**用注释而不是删除**, 所以每个变体的行号与原文件
完全对齐 —— traceback 里的 swa.py / 测试文件行号可以直接和原文件对照。

切分点取自 test_swa_function 的函数体语句 (ast 解析, 不是手数的):
    [0] swa_func = MojoSWAFunction.apply
    [1] swa_func_ref = ..._registry.get("torch").apply
    [2] head_dim = query.shape[-1]
    [3] softmax_scale = ...
    [4] q = query.clone()...            <- 第一个真正下发到设备的算子
    [5] k = key.clone()...
    [6] v = value.clone()...
    [7] o = swa_func(...)               <- 进 Triton 路径; 内部先构造 mask
    [8] o.backward(grad_out)
    [9..11] q_ref/k_ref/v_ref = ...clone()
    [12] o_ref = swa_func_ref(...)      <- torch 参考实现
    [13] o_ref.backward(grad_out)
    [14..17] assert_close(...)
变体 k 保留 [0..k-1], 把 [k..] 全部注释掉; 5 个 perf 用例和 _run_swa_perf 在所有变体里都注释掉,
只留 test_swa_function, 免得 pytest 顺手跑了 perf。

用法 (在仿真机上, cwd 留在 NPU-Sim):
    # 1) 只生成, 看一眼每个变体切在哪
    python _t_swa_bisect_gen.py --gen

    # 2) 生成并逐个跑 (推荐): 每个变体单独 pytest, 带超时和心跳, 挂死自动杀掉继续下一个
    SWEEP_TIMEOUT=120 python _t_swa_bisect_gen.py --run

    # 3) 只跑指定变体 / 只跑一个 pytest 参数组
    python _t_swa_bisect_gen.py --run 4 7
    BISECT_K="True-4-255-M_F32" SWEEP_TIMEOUT=120 python _t_swa_bisect_gen.py --run 4

变体文件写到原测试文件同目录的 _bisect/ 子目录, pytest 会自动继承上一级的 conftest.py
(torch.set_default_device('npu') 等), 所以行为和跑原文件一致。

判读:
  * 变体 4 就 TIMEOUT / 刷 rejected  -> 卡在 query.clone(), 即 torch 原生多 block 算子
  * 变体 4 过、变体 7 挂              -> clone 没问题, 问题在 swa_func 内部 (mask 构造或 Triton kernel)
  * 一路到 14 都过, 只有 assert_close 失败 -> 纯精度问题, 不是仿真器
默认只跑第一个 pytest 参数组 (M_F32 + gqa_interleave=True), 用 BISECT_K 改。
"""
import ast
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

TEST_REL = os.path.join("mojo_opset", "tests", "specific_shape_perf_acc", "test_swa_fwd_bwd_950_perf.py")
TARGET_FUNC = "test_swa_function"
DEFAULT_K = "True-4-255-M_F32"
RE_PROGRAMS = re.compile(r"program pool: programs=(\d+) resident=(\d+)")


def repo_root(explicit=None):
    if explicit:
        if not os.path.isfile(os.path.join(explicit, TEST_REL)):
            raise SystemExit(f"--repo {explicit} 下没有 {TEST_REL}")
        return os.path.abspath(explicit)
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    while True:  # 脚本放在仓库内时往上找
        if os.path.isfile(os.path.join(d, TEST_REL)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    try:  # 脚本放在仓库外面时, 看一眼同级目录
        for name in sorted(os.listdir(here)):
            cand = os.path.join(here, name)
            if os.path.isfile(os.path.join(cand, TEST_REL)):
                return cand
    except OSError:
        pass
    return None


def analyse(src):
    """返回 (切分点语句列表, 要整体注释掉的 (起, 止) 行区间列表)。行号都是 1-based 闭区间。"""
    tree = ast.parse(src)
    target, drop = None, []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name == TARGET_FUNC:
            target = node
            continue
        # perf 用例和它的 helper: 连装饰器一起注释掉
        if node.name.startswith("test_") or node.name == "_run_swa_perf":
            start = min([d.lineno for d in node.decorator_list] + [node.lineno])
            drop.append((start, node.end_lineno))
    if target is None:
        raise SystemExit(f"没在文件里找到 {TARGET_FUNC}")
    return target, drop


def make_variant(lines, target, drop, cut):
    """保留 target.body[:cut], 其余注释掉。返回新的行列表。"""
    out = list(lines)
    comment = set()
    for start, end in drop:
        comment.update(range(start, end + 1))
    if cut < len(target.body):
        tail_start = target.body[cut].lineno
        comment.update(range(tail_start, target.end_lineno + 1))
    for n in sorted(comment):
        text = out[n - 1]
        if text.strip():
            indent = len(text) - len(text.lstrip())
            out[n - 1] = text[:indent] + "# [bisect] " + text[indent:]
        else:
            out[n - 1] = "# [bisect]"
    if cut == 0:
        # 函数体全空了, 得补个 pass。就地替换第一条语句那一行, 不插新行, 这样行号仍与原文件对齐。
        n = target.body[0].lineno
        first = lines[n - 1]
        pad = " " * (len(first) - len(first.lstrip()))
        out[n - 1] = pad + "pass  # [bisect] 空函数体, 原为: " + first.strip()
    return out


def label(lines, stmt):
    text = lines[stmt.lineno - 1].strip()
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text).strip("_")
    return text[:40] or "stmt"


def generate(root):
    path = os.path.join(root, TEST_REL)
    src = open(path, encoding="utf-8").read()
    lines = src.splitlines()
    target, drop = analyse(src)
    outdir = os.path.join(os.path.dirname(path), "_bisect")
    os.makedirs(outdir, exist_ok=True)
    open(os.path.join(outdir, "__init__.py"), "w").close()

    made = []
    for cut in range(len(target.body) + 1):
        if cut < len(target.body):
            name = f"test_bisect_{cut:02d}_upto_{label(lines, target.body[cut])}.py"
            desc = f"保留 [0..{cut - 1}], 从 L{target.body[cut].lineno} 起注释"
        else:
            name = f"test_bisect_{cut:02d}_full.py"
            desc = "完整函数体 (等于原文件, 只去掉 perf 用例)"
        body = make_variant(lines, target, drop, cut)
        header = [
            '"""自动生成的二分变体, 勿手改; 源文件未被修改。',
            f"切分: {desc}",
            f'生成自: {TEST_REL.replace(os.sep, "/")}"""',
        ]
        dest = os.path.join(outdir, name)
        with open(dest, "w", encoding="utf-8", newline="\r\n") as f:
            f.write("\n".join(header + body) + "\n")
        made.append((cut, dest, desc))
    return outdir, made


# ---------------- 跑 ----------------

def spawn(cmd, timeout, beat=20.0, echo=False):
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
            if echo or re.match(r"^(PASSED|FAILED|ERROR|=+ |_+ |E   |\S+::)", line) or "[bisect]" in line:
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
            m = RE_PROGRAMS.search(blob)
            print(f"   ... 已跑 {now - t0:.0f}s (还剩 {deadline - now:.0f}s), 输出 {len(lines)} 行, "
                  f"programs={m.group(1) + '/' + m.group(2) if m else '-'}, "
                  f"invalid_addr={blob.count('Scalar: invalid addr')}, "
                  f"rejected={blob.count('rejected')}", flush=True)
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
    mode = "--run" if "--run" in args else "--gen"
    picks = {int(a) for a in args if a.isdigit()}
    explicit = None
    if "--repo" in args:
        explicit = args[args.index("--repo") + 1]
    root = repo_root(explicit)
    if root is None:
        raise SystemExit(f"没找到仓库根 (需要能看到 {TEST_REL}); 可以用 --repo <仓库路径> 指定")
    outdir, made = generate(root)
    print(f"[bisect] repo_root={root}\n[bisect] 变体目录={outdir}", flush=True)
    for cut, dest, desc in made:
        print(f"  {cut:>2}  {os.path.basename(dest):<52} {desc}", flush=True)
    if mode == "--gen":
        print("\n逐个跑 (或直接 --run):", flush=True)
        print(f"  pytest -sv {os.path.join(outdir, os.path.basename(made[4][1]))} -k '{DEFAULT_K}'", flush=True)
        return

    timeout = float(os.environ.get("SWEEP_TIMEOUT", "180"))
    kexpr = os.environ.get("BISECT_K", DEFAULT_K)
    echo = os.environ.get("SWEEP_ECHO", "") not in ("", "0")
    chosen = [m for m in made if not picks or m[0] in picks]
    print(f"\n[bisect] 开始跑: {len(chosen)} 个变体, timeout={timeout:.0f}s, -k {kexpr!r}", flush=True)

    rows = []
    for cut, dest, desc in chosen:
        print(f"\n[bisect] 变体 {cut}: {desc}", flush=True)
        cmd = [sys.executable, "-m", "pytest", "-sv", "-p", "no:cacheprovider", dest, "-k", kexpr]
        out, timed_out, rc, wall = spawn(cmd, timeout, echo=echo)
        m = RE_PROGRAMS.search(out)
        verdict = ("TIMEOUT" if timed_out else
                   "pass" if re.search(r"\b1 passed", out) else
                   "FAIL" if re.search(r"\b1 failed", out) else
                   "skip" if re.search(r"\b1 skipped", out) else
                   f"rc={rc}")
        row = {"cut": cut, "verdict": verdict, "programs": f"{m.group(1)}/{m.group(2)}" if m else "-",
               "invalid": out.count("Scalar: invalid addr"), "rejected": out.count("rejected"),
               "wall": f"{wall:.0f}", "desc": desc}
        rows.append(row)
        print(f"[bisect] 变体 {cut} -> {verdict}  programs={row['programs']}  "
              f"invalid_addr={row['invalid']}  rejected={row['rejected']}  wall={row['wall']}s", flush=True)
        if verdict == "TIMEOUT":
            print("[bisect] 这个变体挂死了; 它比上一个过了的变体多出来的那条语句就是元凶", flush=True)
            break

    print("\n==== 汇总 ====")
    print(f"{'cut':>3} {'verdict':<9} {'programs':>9} {'invalid':>7} {'rejected':>8} {'wall':>5}  切分")
    for r in rows:
        print(f"{r['cut']:>3} {r['verdict']:<9} {r['programs']:>9} {r['invalid']:>7} {r['rejected']:>8} "
              f"{r['wall']:>5}  {r['desc']}")


if __name__ == "__main__":
    main()
