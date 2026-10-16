"""按 tile 字节数阈值验证若干用例, 并自动对比预期与实测。

环境: 先 source /root/env.sh

用法:
    python3 _t_cases.py            # 跑内置的收尾验证三格
    python3 _t_cases.py 64 256 6 256 float32    # 临时加一格: n_rows n_cols BM BN dtype

阈值假设(v5): 出错 <=> BLOCK_SIZE_M * BLOCK_SIZE_N * itemsize > 6656 B (6.5 KB)
    安全上限取自实测最大通过样本 BM=13/BN=128/fp32;
    最小失败样本是 7.0 KB, 真阈值落在两者之间。

异常分类: SIM(abort) / TIMEOUT(挂死) / ACC(数值错) / ENV(没跑起来, 无效)
    前三类都算「出错」, ENV 不计入结论。
"""
import importlib.util
import os
import re
import sys
import traceback

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
THRESHOLD = 6656  # 6.5 KB

# 不用 torch.empty().element_size() —— 在 run_one 之前碰 torch 建张量似乎会干扰
# standalone_runtime 的初始化(表现为 run_one 里抛 torch_npu 元数据缺失)。
ITEMSIZE = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}

# (shape, BM, BN, dtype)
CASES = [
    ((64, 2048), 1, 2048, torch.float32),   # n_cols>=2048 连 BM=1 都超线?
    ((64, 256), 18, 256, torch.float16),    # fp16 是否同样服从阈值
    ((64, 256), 6, 256, torch.float32),     # 反推出的安全值在别的几何上是否真安全
]

RE_ALIGN = re.compile(r"Address\s+(0x[0-9A-Fa-f]+)\s+is not aligned to\s+(\d+)\s+bytes")
RE_PC = re.compile(r"PC\s*:?\s*(0x[0-9A-Fa-f]+)")


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def classify(exc):
    msg = str(exc)
    if type(exc).__name__ == "SimulationError":
        return "TIMEOUT" if "timeout" in msg.lower() else "SIM"
    if isinstance(exc, AssertionError):
        return "ACC"
    return "ENV"


def preflight():
    print(f"[env] python      = {sys.executable}")
    print(f"[env] torch       = {torch.__version__}")
    print(f"[env] ASCEND_HOME = {os.environ.get('ASCEND_HOME_PATH') or '(未设置)'}")
    if not os.environ.get("ASCEND_HOME_PATH"):
        print("!" * 72)
        print("!! ASCEND_HOME_PATH 未设置 —— 请在同一条命令里 source:")
        print("!!   source /root/env.sh && python3 _t_cases.py")
        print("!" * 72)
        return False
    return True


def main():
    if not preflight():
        return 2

    cases = list(CASES)
    a = sys.argv[1:]
    if len(a) >= 5:
        cases = [((int(a[0]), int(a[1])), int(a[2]), int(a[3]), getattr(torch, a[4]))]

    m = load("ln18", "18-layernorm.py")
    rows = []
    for shape, bm, bn, dt in cases:
        tile = bm * bn * ITEMSIZE[dt]
        want = "正确" if tile <= THRESHOLD else "出错"
        tag = f"{shape} BM={bm} BN={bn} {str(dt).replace('torch.', '')}"
        print(f"\n{'#' * 72}\n#### {tag}   tile={tile} B = {tile / 1024:.1f} KB   预期 {want}\n{'#' * 72}")

        os.environ["MOJO_LAYERNORM_BM"] = str(bm)
        os.environ["MOJO_LAYERNORM_BN"] = str(bn)
        os.environ.pop("MOJO_LAYERNORM_X_FILL", None)
        try:
            m.run_one(shape, dt, 1e-5, 8, 180)
            kind, got, detail = "PASS", "正确", ""
        except Exception as e:  # noqa: BLE001
            kind = classify(e)
            got = "无效" if kind == "ENV" else "出错"
            msg = str(e)
            al = RE_ALIGN.search(msg)
            pcs = RE_PC.findall(msg)
            detail = (f"addr={al.group(1)} need={al.group(2)}B " if al else "") + \
                     (f"PC={pcs[-1]}" if pcs else msg.splitlines()[0][:70])
            print(f"  RESULT {kind}: {detail}")
            if kind == "ENV":
                # ENV 说明根本没跑起来, 原因必须看全栈, 否则只能瞎猜
                print("  ---- 完整 traceback ----")
                print("  " + traceback.format_exc().replace("\n", "\n  "))
        rows.append((tag, tile, want, got, kind, detail))

    print(f"\n{'=' * 96}\nSUMMARY   阈值假设: 出错 <=> tile > {THRESHOLD} B ({THRESHOLD / 1024:.1f} KB)")
    hdr = f"{'case':44}{'tile':>10}  {'预期':>5} {'实测':>5} {'kind':>8}  符合"
    print(hdr)
    print("-" * 96)
    hit = tot = 0
    for tag, tile, want, got, kind, _ in rows:
        if kind == "ENV":
            mark = "—(无效)"
        else:
            tot += 1
            ok = want == got
            hit += ok
            mark = "是" if ok else "**否**"
        print(f"{tag:44}{tile / 1024:9.1f}K  {want:>5} {got:>5} {kind:>8}  {mark}")
    if tot:
        print(f"\n  阈值假设符合 {hit}/{tot}" + ("  -> 结论可用于改代码" if hit == tot else "  -> 有反例, 阈值需修正"))
    else:
        print("\n  没有有效数据点。")
    return 0 if tot and hit == tot else 1


if __name__ == "__main__":
    raise SystemExit(main())
