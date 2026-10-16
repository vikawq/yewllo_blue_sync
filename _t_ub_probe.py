"""UB 超配验证: 在 layernorm fp32 上扫 BLOCK_SIZE_M, 找出最大能活下来的那档。

用法 (在仿真机 /root/fa_test 下):
    python _t_ub_probe.py                      # 默认扫 (256,128): 24(生产) 12 8 6 4
    python _t_ub_probe.py 64 64 256            # 扫 (64,64,256): 18(生产) 9 6 4 2
    MOJO_LAYERNORM_BM_LIST="24,12" python _t_ub_probe.py

判读:
  * BM=24 崩 + BM=12 过  -> UB 超配确认: TOKEN_BLOCK_SIZE_TABLE 是 dtype 无关的,
    fp32 下 tile 字节数翻倍, 256KB UB 装不下。修法在 heuristic 里按 itemsize 缩放。
  * BM=24 和 BM=12 都崩   -> 不是容量问题, 是这条 kernel 在 fp32 上的 codegen 问题,
    下一步该看 BLOCK_SIZE_N(把 128 换成 64)和 .to(tl.float32) 那几条。
  * BM=24 就过了         -> 之前的崩溃来自旧脚本的其它差异, 以新脚本结果为准。
"""
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))

# 每个 shape 的生产 BLOCK_SIZE_M (TOKEN_BLOCK_SIZE_TABLE) 及往下的对折序列
DEFAULT_BM_LIST = {
    128: [24, 12, 8, 6, 4],
    256: [18, 9, 6, 4, 2],
    768: [8, 4, 2, 1],
}


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    dims = [int(a) for a in sys.argv[1:] if a.isdigit()]
    shape = tuple(dims) if len(dims) >= 2 else (256, 128)
    n_cols = shape[-1]

    env_list = os.environ.get("MOJO_LAYERNORM_BM_LIST")
    if env_list:
        bm_list = [int(v) for v in env_list.replace(" ", "").split(",") if v]
    else:
        bm_list = DEFAULT_BM_LIST.get(n_cols, [8, 4, 2, 1])

    m = load("ln18", "18-layernorm.py")
    prod_bm = m.layer_norm_fwd_heuristics(n_cols)

    print(f"\n#### UB probe: shape={shape} dtype=fp32 n_cols={n_cols} "
          f"prod_BLOCK_SIZE_M={prod_bm} sweep={bm_list}")

    rows = []
    for bm in bm_list:
        tile_bytes = bm * (2048 if n_cols > 2048 else 1 << (n_cols - 1).bit_length()) * 4
        tag = " (生产配置)" if bm == prod_bm else ""
        print(f"\n---- BLOCK_SIZE_M={bm}{tag}  单份 X tile = {tile_bytes} B "
              f"= {tile_bytes / 1024:.1f} KB  (UB = 256 KB) ----")
        os.environ["MOJO_LAYERNORM_BM"] = str(bm)
        try:
            m.run_one(shape, torch.float32, 1e-5, 8, 3600)
            rows.append((bm, tile_bytes, "PASS", ""))
        except Exception as e:  # noqa: BLE001
            head = str(e).splitlines()[0][:160]
            rows.append((bm, tile_bytes, type(e).__name__, head))
            print(f"  RESULT exception: {type(e).__name__}: {head}")

    print(f"\n{'=' * 70}\nUB probe summary  shape={shape} fp32")
    for bm, tile_bytes, status, head in rows:
        tag = " <- 生产" if bm == prod_bm else ""
        print(f"  BM={bm:4}  tile={tile_bytes / 1024:6.1f} KB  {status:18}{tag}  {head}")

    passed = [bm for bm, _, s, _ in rows if s == "PASS"]
    if passed:
        print(f"\n  最大存活 BLOCK_SIZE_M = {max(passed)} (生产用 {prod_bm})")
        if max(passed) < prod_bm:
            print("  => UB 超配确认: 生产的 BLOCK_SIZE_M 在 fp32 下过大。")
    else:
        print("\n  全部 BM 都崩 => 不是容量问题, 往 BLOCK_SIZE_N / codegen 方向查。")


if __name__ == "__main__":
    main()
