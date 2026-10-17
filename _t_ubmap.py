"""从 triton-ascend 的 kernel.npuir.mlir 里还原 UB 分配表，并对比两份产物。

npuir 这一层把每个缓冲的 UB 基址写成了字面常量:
    %36 = hivm.hir.pointer_cast(%c256_i64) : memref<64x128xf32, #hivm.address_space>
             ^ 基址 256 字节          ^ 形状与 dtype

所以分配表是纯文本可读的 —— 不需要反汇编, 也不需要仿真器。

用法:
    python3 _t_ubmap.py scan [kernel 名片段]
        扫 /root/.triton/dump, 列出所有含该 kernel 的 dump 及其 tile 形状、UB 总占用
        默认片段是 layernorm

    python3 _t_ubmap.py map <dump 目录或 .mlir 文件>
        打印完整分配表, 并检查重叠 / 越界 / 未对齐

    python3 _t_ubmap.py cmp <A> <B>
        并排对比两份的分配表与总占用

要点: UB 共 0x40000 (256 KB)。向量指令要求 32 字节对齐。
"""
import glob
import os
import re
import sys

UB_SIZE = 0x40000
DUMP_ROOT = "/root/.triton/dump"

ITEMSIZE = {"f32": 4, "f16": 2, "bf16": 2, "f64": 8,
            "i64": 8, "i32": 4, "i16": 2, "i8": 1, "i1": 1}

RE_CAST = re.compile(r"pointer_cast\(\s*%c(-?\d+)_i\d+\s*\)\s*:\s*memref<([^,>]+)")
RE_AS = re.compile(r"#hivm\.address_space(?:<([^>]*)>)?")
RE_FUNC = re.compile(r"func\s+(?:public\s+)?@([A-Za-z_]\w*)")


def parse_shape(s):
    """'64x128xf32' -> ([64,128], 'f32')；含 ? 的返回 (None, dtype)。"""
    parts = s.split("x")
    dtype = parts[-1]
    dims = []
    for p in parts[:-1]:
        if not p.isdigit():
            return None, dtype
        dims.append(int(p))
    return dims, dtype


def buffers(mlir_path):
    rows = []
    for ln in open(mlir_path, encoding="utf-8", errors="replace"):
        m = RE_CAST.search(ln)
        if not m:
            continue
        base = int(m.group(1))
        dims, dtype = parse_shape(m.group(2))
        a = RE_AS.search(ln)
        space = (a.group(1) or "default") if a else "?"
        es = ITEMSIZE.get(dtype)
        size = None
        if dims is not None and es is not None:
            size = es
            for d in dims:
                size *= d
        rows.append({"base": base, "size": size, "shape": m.group(2), "space": space})
    # 同一个基址可能被 cast 成多种视图, 去重时保留最大的 size
    merged = {}
    for r in rows:
        k = (r["base"], r["space"])
        if k not in merged or (r["size"] or 0) > (merged[k]["size"] or 0):
            merged[k] = r
    return sorted(merged.values(), key=lambda r: (r["space"], r["base"]))


def kernel_names(mlir_path):
    names = set()
    for ln in open(mlir_path, encoding="utf-8", errors="replace"):
        for m in RE_FUNC.finditer(ln):
            names.add(m.group(1))
    return names


def tile_hint(rows):
    """挑最大的那个缓冲当 tile 指纹 —— 它通常就是 BM x BN 的 X tile。"""
    sized = [r for r in rows if r["size"]]
    return max(sized, key=lambda r: r["size"])["shape"] if sized else "?"


def resolve(p):
    if os.path.isdir(p):
        c = os.path.join(p, "kernel.npuir.mlir")
        if os.path.exists(c):
            return c
        sys.exit(f"{p} 下没有 kernel.npuir.mlir")
    if os.path.exists(p):
        return p
    sys.exit(f"找不到 {p}")


def check(rows):
    """返回问题列表。"""
    bad = []
    for r in rows:
        if r["base"] % 32:
            bad.append(f"基址未 32B 对齐: {hex(r['base'])} ({r['base']})  {r['shape']}")
        if r["size"] and r["base"] + r["size"] > UB_SIZE and r["space"] in ("default", "?", "ub"):
            end = r["base"] + r["size"]
            bad.append(f"越过 UB 末端: [{hex(r['base'])}, {hex(end)}) > {hex(UB_SIZE)}  {r['shape']}")
    by_space = {}
    for r in rows:
        by_space.setdefault(r["space"], []).append(r)
    for space, rs in by_space.items():
        rs = [r for r in rs if r["size"]]
        for a, b in zip(rs, rs[1:]):
            if a["base"] + a["size"] > b["base"]:
                bad.append(f"重叠({space}): [{hex(a['base'])},{hex(a['base'] + a['size'])}) "
                           f"{a['shape']}  与  [{hex(b['base'])},...) {b['shape']}")
    return bad


def footprint(rows):
    ends = [r["base"] + r["size"] for r in rows if r["size"]]
    return max(ends) if ends else 0


def cmd_scan(args):
    frag = (args[0] if args else "layernorm").lower()
    hits = []
    for d in sorted(glob.glob(os.path.join(DUMP_ROOT, "*"))):
        f = os.path.join(d, "kernel.npuir.mlir")
        if not os.path.exists(f):
            continue
        names = kernel_names(f) | kernel_names(os.path.join(d, "kernel.ttir.mlir")) \
            if os.path.exists(os.path.join(d, "kernel.ttir.mlir")) else kernel_names(f)
        if not any(frag in n.lower() for n in names):
            continue
        rows = buffers(f)
        hits.append((d, sorted(names)[:1], tile_hint(rows), footprint(rows), len(rows), check(rows)))
    if not hits:
        print(f"  {DUMP_ROOT} 下没有包含 '{frag}' 的 dump。")
        print("  需要重新编译一次并开 dump: TRITON_KERNEL_DUMP=1 <你的 run 命令>")
        print("  (只要能走到编译就行, 仿真器跑不跑完无所谓)")
        return 1
    print(f"  {'dump':46}{'tile(最大缓冲)':>18}{'UB 占用':>12}{'缓冲数':>7}  问题")
    print("  " + "-" * 100)
    for d, nm, tile, fp, n, bad in sorted(hits, key=lambda h: h[3]):
        flag = f"{len(bad)} 处" if bad else "-"
        print(f"  {os.path.basename(d):46}{tile:>18}{fp / 1024:11.1f}K{n:>7}  {flag}")
    print(f"\n  共 {len(hits)} 份。UB = {UB_SIZE / 1024:.0f} KB。")
    print("  下一步: map <目录> 看完整分配表; cmp <A> <B> 对比跨阈值的两份。")
    return 0


def cmd_map(args):
    if not args:
        sys.exit("用法: map <dump 目录或 .mlir>")
    f = resolve(args[0])
    rows = buffers(f)
    if not rows:
        sys.exit(f"{f} 里没解析到 pointer_cast —— 确认这是 kernel.npuir.mlir")
    print(f"  {f}")
    print(f"  kernel: {', '.join(sorted(kernel_names(f))[:3])}\n")
    print(f"  {'base':>10}{'end':>10}{'size':>10}  {'shape':<20}{'space':<10}对齐")
    print("  " + "-" * 72)
    for r in rows:
        end = (r["base"] + r["size"]) if r["size"] else None
        print(f"  {hex(r['base']):>10}{(hex(end) if end else '?'):>10}"
              f"{(r['size'] if r['size'] else '?'):>10}  {r['shape']:<20}{r['space']:<10}"
              f"{'32B' if r['base'] % 32 == 0 else '**否**'}")
    fp = footprint(rows)
    print(f"\n  UB 占用上界 = {fp} B = {fp / 1024:.1f} KB  (UB = {UB_SIZE / 1024:.0f} KB,"
          f" 用掉 {100 * fp / UB_SIZE:.1f}%)")
    bad = check(rows)
    if bad:
        print("\n  !! 发现问题:")
        for b in bad:
            print(f"     {b}")
    else:
        print("\n  未发现重叠 / 越界 / 未对齐。")
    return 0


def cmd_cmp(args):
    if len(args) < 2:
        sys.exit("用法: cmp <A> <B>")
    fa, fb = resolve(args[0]), resolve(args[1])
    ra, rb = buffers(fa), buffers(fb)
    print(f"  A = {fa}\n      tile {tile_hint(ra)}  占用 {footprint(ra) / 1024:.1f}K  缓冲 {len(ra)}")
    print(f"  B = {fb}\n      tile {tile_hint(rb)}  占用 {footprint(rb) / 1024:.1f}K  缓冲 {len(rb)}")
    print(f"\n  占用差 {(footprint(rb) - footprint(ra)) / 1024:+.1f}K   缓冲数差 {len(rb) - len(ra):+d}\n")
    print(f"  {'A base':>10}{'A shape':>20}  |{'B base':>10}{'B shape':>20}")
    print("  " + "-" * 66)
    for i in range(max(len(ra), len(rb))):
        a = ra[i] if i < len(ra) else None
        b = rb[i] if i < len(rb) else None
        la = f"{hex(a['base']):>10}{a['shape']:>20}" if a else " " * 30
        lb = f"{hex(b['base']):>10}{b['shape']:>20}" if b else ""
        print(f"  {la}  |{lb}")
    for tag, rows in (("A", ra), ("B", rb)):
        bad = check(rows)
        if bad:
            print(f"\n  !! {tag} 的问题:")
            for x in bad:
                print(f"     {x}")
    return 0


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    sub, args = sys.argv[1], sys.argv[2:]
    fn = {"scan": cmd_scan, "map": cmd_map, "cmp": cmd_cmp}.get(sub)
    if fn is None:
        print(__doc__)
        return 2
    return fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
