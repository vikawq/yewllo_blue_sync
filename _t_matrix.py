"""把剩余待测矩阵一次跑完, 并自动从 SimulationError 里提取 PC / 故障地址 / 对齐要求。

环境: 先 source /root/env.sh (不是 /set_env.sh)

用法:
    python3 _t_matrix.py pending   # 待测矩阵: (64,256) 和 (8,768), fp16/fp32 配对
    python3 _t_matrix.py addr      # 追查 0x00000018: fill=0.0 扫 BM, 看地址是否 == BM
    python3 _t_matrix.py all       # 两组都跑

输出的 summary 表里:
    kind   PASS / SIM(仿真器崩) / ACC(数值不对) / ENV(没跑起来, 无效)
    PC     致命指令地址
    addr   被判非对齐的那个地址
    need   该指令要求的对齐字节数
    ovf    是否同时出现了 ub_addr_overflow(非致命)
    ==BM   addr 的十进制值是否正好等于 BLOCK_SIZE_M
"""
import importlib.util
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))

RE_PC = re.compile(r"PC\s*:?\s*(0x[0-9A-Fa-f]+)")
RE_ALIGN = re.compile(r"Address\s+(0x[0-9A-Fa-f]+)\s+is not aligned to\s+(\d+)\s+bytes")
RE_START = re.compile(r"start_addr:\s*(0x[0-9A-Fa-f]+)")
RE_ART = re.compile(r"artifacts:\s*(\S+)")

# (tag, shape, dtype, BM, BN(None=脚本自算), fill(None=randn))
PENDING = [
    ("A1 (64,256) fp32 生产BM", (64, 256), torch.float32, 18, None, None),
    ("A2 (64,256) fp32 BM砍半", (64, 256), torch.float32, 9, None, None),
    ("A3 (64,256) fp16 生产BM", (64, 256), torch.float16, 18, None, None),
    ("A4 (8,768)  fp32 生产BM", (8, 768), torch.float32, 8, None, None),
    ("A5 (8,768)  fp16 生产BM", (8, 768), torch.float16, 8, None, None),
]

# 追查 0x00000018: 固定 fill=0.0 让 0x400173c 那条不致命, 单看后面那个故障点
ADDR = [
    (f"B{i} (256,128) fp32 fill=0 BM={bm}", (256, 128), torch.float32, bm, None, 0.0)
    for i, bm in enumerate((24, 18, 12, 8), start=1)
]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def classify(exc):
    if type(exc).__name__ == "SimulationError":
        return "SIM"
    if isinstance(exc, AssertionError):
        return "ACC"
    return "ENV"


def dig(msg):
    pcs = RE_PC.findall(msg)
    al = RE_ALIGN.search(msg)
    starts = RE_START.findall(msg)
    art = RE_ART.search(msg)
    return {
        "pc": pcs[-1] if pcs else "-",
        "addr": al.group(1) if al else "-",
        "need": al.group(2) if al else "-",
        "ovf": "yes" if "ub_addr_overflow" in msg else "no",
        "starts": starts[:3],
        "artifacts": art.group(1) if art else "-",
    }


def preflight():
    if not os.environ.get("ASCEND_HOME_PATH"):
        print("!" * 72)
        print("!! ASCEND_HOME_PATH 未设置 —— 仿真器不会启动, 本轮不会产出有效数据。")
        print("!! 先执行:  source /root/env.sh")
        print("!" * 72)
        return False
    return True


def run_case(m, tag, shape, dtype, bm, bn, fill):
    os.environ["MOJO_LAYERNORM_BM"] = str(bm)
    if bn is None:
        os.environ.pop("MOJO_LAYERNORM_BN", None)
    else:
        os.environ["MOJO_LAYERNORM_BN"] = str(bn)
    if fill is None:
        os.environ.pop("MOJO_LAYERNORM_X_FILL", None)
    else:
        os.environ["MOJO_LAYERNORM_X_FILL"] = repr(fill)

    print(f"\n{'#' * 72}\n#### {tag}\n{'#' * 72}")
    rec = {"tag": tag, "bm": bm, "dtype": str(dtype).replace("torch.", "")}
    try:
        m.run_one(shape, dtype, 1e-5, 8, 3600)
        rec["kind"] = "PASS"
        rec.update(pc="-", addr="-", need="-", ovf="-", starts=[], artifacts="-")
    except Exception as e:  # noqa: BLE001
        rec["kind"] = classify(e)
        rec.update(dig(str(e)))
        print(f"  RESULT {rec['kind']}: {type(e).__name__}")
        print(f"    PC={rec['pc']}  addr={rec['addr']}  need={rec['need']}B  "
              f"ub_addr_overflow={rec['ovf']}")
        if rec["starts"]:
            print(f"    start_addr 前几个: {rec['starts']}")
        print(f"    artifacts: {rec['artifacts']}")
    return rec


def main():
    if not preflight():
        return 2
    mode = (sys.argv[1] if len(sys.argv) > 1 else "pending").lower()
    cases = {"pending": PENDING, "addr": ADDR, "all": PENDING + ADDR}.get(mode)
    if cases is None:
        print(__doc__)
        return 2

    m = load("ln18", "18-layernorm.py")
    recs = [run_case(m, *c) for c in cases]

    print(f"\n{'=' * 100}\nSUMMARY  (mode={mode})")
    hdr = f"{'case':30}{'dtype':8}{'BM':>4}  {'kind':5} {'PC':>11} {'addr':>12} {'need':>5} {'ovf':>4} {'==BM':>5}"
    print(hdr)
    print("-" * len(hdr))
    for r in recs:
        eqbm = "-"
        if r["addr"] not in ("-", None):
            try:
                eqbm = "YES" if int(r["addr"], 16) == r["bm"] else "no"
            except ValueError:
                eqbm = "?"
        print(f"{r['tag']:30}{r['dtype']:8}{r['bm']:>4}  {r['kind']:5} {r['pc']:>11} "
              f"{r['addr']:>12} {r['need']:>5} {r['ovf']:>4} {eqbm:>5}")

    env = [r["tag"] for r in recs if r["kind"] == "ENV"]
    if env:
        print(f"\n  !! ENV(无效, 不计入结论): {env}")
    print("\n  完整 stderr 在各自 artifacts 目录下的 stderr.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
