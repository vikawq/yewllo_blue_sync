"""反汇编 triton-ascend 的 kernel.o，并对比两份产物的差异。

不需要仿真器、不需要 source env —— 只读 ELF，和当前的 torch_npu 环境故障无关。

用法:
    python3 _t_disasm.py probe
        试一遍机器上的各个反汇编器，报告哪个能用

    python3 _t_disasm.py dump <kernel.o 或 launch 目录> [out.asm]
        反汇编到 .asm (默认 ./<目录名>.asm)

    python3 _t_disasm.py at <out.asm> 0x4001800 [窗口行数]
        打印某个 PC 前后的指令。PC 可以写日志里的值(0x4001800)，
        也可以写 .text 偏移(0x1800)，两种都会尝试

    python3 _t_disasm.py diff <a.asm> <b.asm>
        指令助记符直方图差异 —— 看越过阈值后编译器多做/少做了什么

背景: kernel.o 是 ELF64, Machine 0x1029, LOAD 段 VirtAddr=0、FileSiz=0x1e80,
      而运行期 PC 基址是 0x4000000, 所以 PC - 0x4000000 = .text 偏移。
      llvm-objdump 用 --adjust-vma=0x4000000 可以让输出地址直接对上日志里的 PC。
"""
import collections
import os
import re
import subprocess
import sys

PC_BASE = 0x4000000

CANDIDATES = [
    ["/opt/ascend/cann-9.1.0/tools/msobjdump/msobjdump", "-d"],
    ["/opt/ascend/cann-9.1.0/bin/llvm-objdump", "-d", f"--adjust-vma={hex(PC_BASE)}"],
    ["/opt/ascend/cann-9.1.0/tools/bisheng_compiler/bin/llvm-objdump", "-d",
     f"--adjust-vma={hex(PC_BASE)}"],
    ["/opt/ascend/cann-9.1.0/bin/llvm-objdump", "-d"],
    ["llvm-objdump", "-d", f"--adjust-vma={hex(PC_BASE)}"],
    ["objdump", "-d", f"--adjust-vma={hex(PC_BASE)}"],
]

# "   4001800: 18 00 18 00   vldi  v3, ..."  /  "4001800 <foo>:" 等都要能认
RE_INSN = re.compile(r"^\s*([0-9a-fA-F]{4,16})\s*:\s+(.*)$")


def resolve(path):
    """接受 kernel.o 路径或 launch 目录。"""
    if os.path.isdir(path):
        cand = os.path.join(path, "kernel.o")
        if not os.path.exists(cand):
            sys.exit(f"{path} 下没有 kernel.o")
        return cand
    if not os.path.exists(path):
        sys.exit(f"找不到 {path}")
    return path


def run(cmd, timeout=20):
    """超时压到 20 秒 —— 认不出架构的反汇编器会空转, 6 个候选各等 2 分钟就像卡死了。"""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", "not found"
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"


def score(out):
    """能数出多少条像指令的行 —— 用来判断这个反汇编器是否真认得这个架构。"""
    return sum(1 for ln in out.splitlines() if RE_INSN.match(ln))


def cmd_probe(args):
    target = resolve(args[0]) if args else None
    print(f"{'反汇编器':58} {'exit':>5} {'指令行':>7}  备注")
    print("-" * 92)
    best = None
    for c in CANDIDATES:
        if not os.path.isabs(c[0]) or os.path.exists(c[0]):
            label = " ".join(c)
            if target:
                rc, out, err = run(c + [target])
                n = score(out)
                note = "" if n else (err.strip().splitlines() or [""])[0][:34]
                print(f"{label:58} {rc:>5} {n:>7}  {note}")
                if n and (best is None or n > best[1]):
                    best = (c, n)
            else:
                print(f"{label:58} {'-':>5} {'-':>7}  (给个 kernel.o 才能真试)")
    if target:
        print()
        if best:
            print(f"  可用: {' '.join(best[0])}   ({best[1]} 条指令)")
        else:
            print("  没有一个认得 Machine 0x1029。下一步看 _tools/peek_msobjdump.sh 怎么调，")
            print("  或给 llvm-objdump 显式指定 --triple/-m。")
    return 0


def disassemble(path):
    for c in CANDIDATES:
        if os.path.isabs(c[0]) and not os.path.exists(c[0]):
            continue
        rc, out, err = run(c + [path])
        if score(out) > 20:
            return " ".join(c), out
    return None, None


def cmd_dump(args):
    if not args:
        sys.exit("用法: dump <kernel.o 或 launch 目录> [out.asm]")
    src = resolve(args[0])
    out_path = args[1] if len(args) > 1 else \
        os.path.basename(os.path.dirname(src) if src.endswith("kernel.o") else src) + ".asm"
    used, out = disassemble(src)
    if out is None:
        sys.exit("所有反汇编器都认不出来。先跑 probe 看详情。")
    with open(out_path, "w") as f:
        f.write(f"# source: {src}\n# via: {used}\n{out}")
    print(f"  via  {used}")
    print(f"  写入 {out_path}  ({score(out)} 条指令)")
    return 0


def load_insns(asm):
    rows = []
    for ln in open(asm, encoding="utf-8", errors="replace"):
        m = RE_INSN.match(ln)
        if m:
            rows.append((int(m.group(1), 16), m.group(2).rstrip()))
    return rows


def cmd_at(args):
    if len(args) < 2:
        sys.exit("用法: at <out.asm> <PC 或 .text 偏移> [窗口行数]")
    rows = load_insns(args[0])
    want = int(args[1], 16)
    win = int(args[2]) if len(args) > 2 else 14
    if not rows:
        sys.exit("这个 .asm 里没解析出指令行")
    lo, hi = rows[0][0], rows[-1][0]
    tries = [want, want - PC_BASE, want + PC_BASE]
    addr = next((t for t in tries if lo <= t <= hi), None)
    if addr is None:
        sys.exit(f"{hex(want)} 不在 [{hex(lo)}, {hex(hi)}] 内；试过 {[hex(t) for t in tries]}")
    idx = min(range(len(rows)), key=lambda i: abs(rows[i][0] - addr))
    print(f"  地址空间 [{hex(lo)}, {hex(hi)}]，命中 {hex(rows[idx][0])}\n")
    for i in range(max(0, idx - win), min(len(rows), idx + win + 1)):
        mark = "  <<< 出事指令" if i == idx else ""
        print(f"  {hex(rows[i][0]):>10}  {rows[i][1]}{mark}")
    return 0


def mnemonic(text):
    t = text.split("#")[0].strip()
    # 去掉前导的字节码列，取第一个像助记符的 token
    for tok in t.replace("\t", " ").split():
        if re.fullmatch(r"[0-9a-fA-F]{2}", tok):
            continue
        return tok.lower().rstrip(",")
    return "?"


def cmd_diff(args):
    if len(args) < 2:
        sys.exit("用法: diff <a.asm> <b.asm>")
    a, b = (collections.Counter(mnemonic(t) for _, t in load_insns(p)) for p in args[:2])
    na, nb = sum(a.values()), sum(b.values())
    print(f"  A = {args[0]}  共 {na} 条")
    print(f"  B = {args[1]}  共 {nb} 条")
    print(f"  总数差 {nb - na:+d}\n")
    print(f"  {'助记符':<18}{'A':>7}{'B':>7}{'B-A':>7}")
    print("  " + "-" * 39)
    for k in sorted(set(a) | set(b), key=lambda k: (-abs(b[k] - a[k]), k)):
        d = b[k] - a[k]
        if d or a[k] != b[k]:
            print(f"  {k:<18}{a[k]:>7}{b[k]:>7}{d:>+7}")
    print("\n  重点看: B 里新出现的 store/load 对(溢出的痕迹)、寻址模式变化、")
    print("          以及总指令数的跳变 —— 这是越过阈值后 codegen 换了路子的证据。")
    return 0


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    sub, args = sys.argv[1], sys.argv[2:]
    fn = {"probe": cmd_probe, "dump": cmd_dump, "at": cmd_at, "diff": cmd_diff}.get(sub)
    if fn is None:
        print(__doc__)
        return 2
    return fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
