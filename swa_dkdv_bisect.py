#!/usr/bin/env python3
"""
_swa_bwd_dkdv_kernel 编译崩溃 (HIVM setBaseMemRefTypeScope assert) 的分档实验脚本。

崩溃不是 ub_overflow（BLOCK=32 仍复现），而是同一个 base memref 被要求放在两个
不同的 memory scope 里。下面每个 variant 只改动一处，用来二分到底是哪条路径。

用法（在远端有 CANN 的机器上）:
    python swa_dkdv_bisect.py --file <path>/mojo_opset/backends/ttx/kernels/npu/swa.py --variant 1
    # 跑测试，记录结果，然后
    python swa_dkdv_bisect.py --file <...>/swa.py --restore
    python swa_dkdv_bisect.py --file <...>/swa.py --variant 2
    ...

每次 --variant 都会先从 .orig 备份恢复，所以 variant 之间互不叠加。
想叠加用 --variant 1,2,5 这种写法。
注意：改完必须让 triton 重新编译 —— 加环境变量 TRITON_ALWAYS_COMPILE=1，
否则可能直接命中旧 cache。
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 每个 variant: (名字, 假设, [(old, new), ...], 作用域提示)
# old 串必须在 dkdv 这一段里唯一命中，脚本会校验。
# ---------------------------------------------------------------------------

VARIANTS = {
    1: dict(
        name="unit_flag=False (仅 dkdv)",
        why="unit_flag 就是命令行里 --enable-hivm-unit-flag-sync=True 的来源，"
            "而崩溃发生在 HIVM 的 scope 工具里、紧挨 hivm-graph-sync-solver。"
            "sdpa.py 里注释说它是『cube 搬出的一个优化项』—— 正好管 L0C→UB/GM，"
            "也正好是 dk/dv 累加器所在的那条路。fwd kernel 不开它，能编过。",
        edits=[("DKDV_LAUNCH::unit_flag=True,", "unit_flag=False,")],
    ),
    2: dict(
        name="去掉 matmul_at_least_once 两个 compile_hint",
        why="整个 repo 只有这两行用了这个 hint。它把 dk/dv 钉在 cube/L0C 上，"
            "可同一个 buffer 又被 tl.zeros 初始化、又要被 cast+store（见 variant 3）。"
            "一个 base memref 两个 scope = 这条 assert。",
        edits=[
            ('tl.extra.cann.extension.compile_hint(dv, "matmul_at_least_once")',
             '# [bisect] tl.extra.cann.extension.compile_hint(dv, "matmul_at_least_once")'),
            ('tl.extra.cann.extension.compile_hint(dk, "matmul_at_least_once")',
             '# [bisect] tl.extra.cann.extension.compile_hint(dk, "matmul_at_least_once")'),
        ],
    ),
    3: dict(
        name="打断 fp32 下 dk/dv 存回时的 no-op cast",
        why="本例是 fp32，所以 dk.to(dk_ptr.type.element_ty) 是空操作，"
            "L0C 里的累加值被直接喂给 GM store，中间没有任何在 UB 里落地的算子。"
            "bf16/fp16 时这个 cast 是真算子，会自然插入 L0C→UB 拷贝 —— "
            "这解释了为什么只有 F32 档炸。乘 1.0 强制它在 UB 里落一次地。",
        edits=[
            ("tl.store(cur_dv_block_ptr, dv.to(dv_ptr.type.element_ty), boundary_check=(0, 1))",
             "tl.store(cur_dv_block_ptr, (dv * 1.0).to(dv_ptr.type.element_ty), boundary_check=(0, 1))"),
            ("tl.store(cur_dk_block_ptr, dk.to(dk_ptr.type.element_ty), boundary_check=(0, 1))",
             "tl.store(cur_dk_block_ptr, (dk * 1.0).to(dk_ptr.type.element_ty), boundary_check=(0, 1))"),
        ],
    ),
    4: dict(
        name="hfusion_enable_multiple_consumer_fusion=False (仅 dkdv)",
        why="p 有两个消费者、在两个不同单元上：tl.trans(p_cast)→cube，"
            "ds = p * (dp - d) →vector。fp32 下 p_cast = p.to(q.dtype) 是空操作，"
            "两个消费者共用同一块 buffer。multiple-consumer-fusion 把它们融到一起，"
            "就要同时给这块 buffer 配 L1/L0A 和 UB。",
        edits=[("DKDV_LAUNCH::hfusion_enable_multiple_consumer_fusion=True,",
                "hfusion_enable_multiple_consumer_fusion=False,")],
    ),
    5: dict(
        name="intra_cache_num=1 → 3 + inter_cache_num=2 (对齐 dq kernel)",
        why="dkdv 是 repo 里唯一一个 intra_cache_num=1、且不给 inter_cache_num 的 launch，"
            "紧挨着的 dq kernel 用的是 3/2。cache buffer 给太少会迫使分配器"
            "把不同 scope 的值复用到同一块 buffer 上。",
        edits=[("DKDV_LAUNCH::intra_cache_num=1,", "intra_cache_num=3,\n        inter_cache_num=2,")],
    ),
    6: dict(
        name="打断 p / ds 在 fp32 下与 cube 操作数的别名",
        why="同 variant 4 的假设，但从 kernel 侧而不是 flag 侧下手："
            "强制 p_cast / ds_cast 成为独立 buffer，fp32 下也不会被折叠掉。",
        edits=[
            ("    p_cast = p.to(q.dtype)", "    p_cast = (p * 1.0).to(q.dtype)"),
            ("    ds_cast = ds.to(q.dtype)", "    ds_cast = (ds * 1.0).to(q.dtype)"),
        ],
    ),
}

# variant 6 改的是 helper 函数，不在 dkdv kernel 体内，单独标记
HELPER_SCOPED = {6}

DKDV_KERNEL_BEGIN = "def _swa_bwd_dkdv_kernel("
DKDV_LAUNCH_BEGIN = "_swa_bwd_dkdv_kernel[grid]("
HELPER_BEGIN = "def _sdpa_single_block_bwd_dkdv("


def find_region(text, begin_marker, end_marker_re=r"\n(?:@triton\.|def |_swa_bwd_dq_kernel\[grid\]\()"):
    i = text.index(begin_marker)
    m = re.search(end_marker_re, text[i + len(begin_marker):])
    j = i + len(begin_marker) + (m.start() if m else len(text))
    return i, j


def apply_edit(text, old, new, label):
    """在限定区域内做唯一替换。old 以 'DKDV_LAUNCH::' 开头表示只在 launch 调用里找。"""
    if old.startswith("DKDV_LAUNCH::"):
        old = old[len("DKDV_LAUNCH::"):]
        lo, hi = find_region(text, DKDV_LAUNCH_BEGIN,
                             r"_swa_bwd_dq_kernel\[grid\]\(")
    elif label in HELPER_SCOPED:
        lo, hi = find_region(text, HELPER_BEGIN)
    else:
        lo, hi = find_region(text, DKDV_KERNEL_BEGIN)

    region = text[lo:hi]
    n = region.count(old)
    if n != 1:
        raise SystemExit(
            f"[FAIL] variant {label}: 在目标区域里找到 {n} 处 {old!r}（需要恰好 1 处）。"
            "\n       源码跟我这边的版本不一致，请手工改，或把这段贴给我重新对齐。"
        )
    return text[:lo] + region.replace(old, new, 1) + text[hi:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="npu/swa.py 的路径")
    ap.add_argument("--variant", help="要应用的 variant，如 1 或 1,3")
    ap.add_argument("--restore", action="store_true", help="从 .orig 备份恢复")
    ap.add_argument("--list", action="store_true", help="列出所有 variant")
    args = ap.parse_args()

    if args.list:
        for k, v in VARIANTS.items():
            print(f"\n[{k}] {v['name']}\n    {v['why']}")
        return

    path = Path(args.file)
    backup = path.with_suffix(path.suffix + ".orig")

    if not path.exists():
        raise SystemExit(f"[FAIL] 找不到 {path}")

    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"[ok] 备份 -> {backup}")

    if args.restore:
        shutil.copy2(backup, path)
        print(f"[ok] 已从备份恢复 {path}")
        return

    if not args.variant:
        raise SystemExit("需要 --variant 或 --restore 或 --list")

    # 每次都从干净的备份开始，避免 variant 互相叠加
    shutil.copy2(backup, path)
    text = path.read_text(encoding="utf-8")

    ids = [int(x) for x in args.variant.split(",")]
    for vid in ids:
        if vid not in VARIANTS:
            raise SystemExit(f"[FAIL] 没有 variant {vid}")
        v = VARIANTS[vid]
        for old, new in v["edits"]:
            text = apply_edit(text, old, new, vid)
        print(f"[ok] 已应用 variant {vid}: {v['name']}")

    path.write_text(text, encoding="utf-8")
    print(f"\n[ok] 写回 {path}")
    print("\n下一步：")
    print("  TRITON_ALWAYS_COMPILE=1 pytest -s <你的测试> -k 'False-4-1023-M_F32'")
    print("  （必须带 TRITON_ALWAYS_COMPILE=1，否则可能命中旧 cache 看不出变化）")


if __name__ == "__main__":
    main()
