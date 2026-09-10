# -*- coding: utf-8 -*-
"""第 4 步：把这一轮的全部数据压成一个可回传的小文件 + 一段可直接粘贴的摘要。

为什么需要：一轮 run_all.sh 产出的 CSV 有 3 MB 左右，从机器上拷出来不方便。
但 04 真正用到的量只有一小部分，而且相邻 M 之间高度相关 —— 差分 + 定点 + gzip
之后能压到 ~50 KB（实测 3057 KB -> 53 KB，还原误差 0.005%，比测量噪声底低 37 倍）。

产出两样东西：
  1) data/digest.json.gz     ~50 KB，上传这个文件，我用 `04_analyze.py --digest` 全量复算
  2) 终端打印的 SUMMARY 块   ~150 行纯文本，拷贝不方便时直接贴这一段也能看主结论

压缩策略（都不影响任何一段分析）：
  * 多趟只保留**奇偶两组的均值** —— R 闸门本来就只用这两组，其余各段用它们的平均
  * 数值转成相对该算子中位数的定点整数（1e-4），再沿 M 做差分
  * kernel 二进制名映射成整数 id

用法:
    python 05_digest.py                    # 扫 data/ 和 out_env_probe.json
    python 05_digest.py --no-file          # 只打印 SUMMARY，不写文件
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd

CAND_T = [8, 16, 32, 64, 128, 256]


def dz(a):
    """差分编码：平滑曲线的差分是小整数，gzip 能吃掉大部分。"""
    a = np.asarray(a, dtype=np.int64)
    return [int(a[0])] + np.diff(a).astype(int).tolist()


def undz(a):
    return np.cumsum(np.asarray(a, dtype=np.int64))


def enc_sweep(d):
    out = {}
    for op in sorted(d.op.unique()):
        g = d[d.op == op]
        if "pass_id" not in g.columns or g.pass_id.nunique() < 2:
            h = [g.groupby("num_tokens").agg(med=("median", "mean"), mn=("min", "mean"),
                                             an=("anchor_ms", "mean"),
                                             cv=("cv_pct", "mean"))] * 2
        else:
            h = [g[g.pass_id % 2 == k].groupby("num_tokens").agg(
                med=("median", "mean"), mn=("min", "mean"),
                an=("anchor_ms", "mean"), cv=("cv_pct", "mean")) for k in (0, 1)]
        M = np.intersect1d(h[0].index, h[1].index)
        if len(M) < 3:
            continue
        sc = float(np.median(h[0].med.loc[M]))
        if not np.isfinite(sc) or sc <= 0:
            sc = 1.0
        e = dict(M=dz(M), scale=sc,
                 tm=str(g.timing_mode.iloc[0]) if "timing_mode" in g else "?",
                 npass=int(g.pass_id.nunique()) if "pass_id" in g else 1,
                 iters=int(g.n.median()) if "n" in g else 0)
        for k in (0, 1):
            e["med%d" % k] = dz(np.round(h[k].med.loc[M] / sc * 1e4))
            e["mn%d" % k] = dz(np.round(h[k].mn.loc[M].fillna(0) / sc * 1e4))
            a = h[k].an.loc[M]
            e["an%d" % k] = dz(np.round(a.fillna(0) / sc * 1e4)) if a.notna().any() else []
        e["cv"] = dz(np.round(((h[0].cv.loc[M] + h[1].cv.loc[M]) / 2).fillna(0) * 20))
        # 各段真正用的是"全部趟的中位数"，它不等于两组均值的均值。单独存一份，
        # 这样除 R 段（本来就只看奇偶两组）以外的重建与原始 CSV 逐点一致。
        agg = g.groupby("num_tokens")["median"].median().loc[M]
        e["agg"] = dz(np.round(agg / sc * 1e4))
        # 锚点归一后的聚合值要**先算比值再取中位数**（各段就是这么算的），
        # 不能存 median(median)/median(anchor) —— 两者不等，实测差到 0.1 个百分点。
        if "anchor_ms" in g.columns and (g.anchor_ms > 0).any():
            rr = (g["median"] / g.anchor_ms).groupby(g.num_tokens).median().loc[M]
            e["aggn"] = dz(np.round(rr * 1e5))
        out[op] = e
    return out


def enc_tiling(t):
    kb = sorted(t.kernel_bin.astype(str).unique()) if "kernel_bin" in t else [""]
    kmap = {k: i for i, k in enumerate(kb)}
    T = {"kernels": kb, "ops": {}}
    num = dict(cu="cube_util", cy="cycles", mac="mac_ratio", mte2="mte2_ratio",
               mte1="mte1_ratio", fix="fixpipe_ratio", spr="duration_spread_pct")
    scale = dict(cu=100, cy=1e-3, mac=1000, mte2=1000, mte1=1000, fix=1000, spr=100)
    for op, g in t.groupby("op"):
        g = g.sort_values("num_tokens")
        sc = float(g.duration_us.median())
        e = dict(M=dz(g.num_tokens), scale=sc,
                 dur=dz(np.round(g.duration_us / sc * 1e4)),
                 bd=dz(g.block_dim) if "block_dim" in g else [],
                 kb=dz([kmap[str(x)] for x in g.kernel_bin]) if "kernel_bin" in g else [])
        for k, col in num.items():
            if col in g.columns and g[col].notna().any():
                e[k] = dz(np.round(g[col].fillna(0) * scale[k]))
        T["ops"][op] = e
    return T


def dec_sweep(e):
    """还原成 04 能直接吃的 DataFrame（两趟：pass_id 0/1）。"""
    rows = []
    for op, c in e.items():
        M = undz(c["M"])
        sc = c["scale"]
        for k in (0, 1):
            med = undz(c["med%d" % k]) / 1e4 * sc
            mn = undz(c["mn%d" % k]) / 1e4 * sc
            an = (undz(c["an%d" % k]) / 1e4 * sc) if c.get("an%d" % k) else np.full(len(M), np.nan)
            cv = undz(c["cv"]) / 20.0
            agg = (undz(c["agg"]) / 1e4 * sc) if c.get("agg") else med
            aggn = (undz(c["aggn"]) / 1e5) if c.get("aggn") else None
            for i, m in enumerate(M):
                rows.append(dict(op=op, num_tokens=int(m), pass_id=k,
                                 agg=float(agg[i]),
                                 agg_norm=(float(aggn[i]) if aggn is not None else np.nan),
                                 **{"median": float(med[i]), "min": float(mn[i])},
                                 cv_pct=float(cv[i]),
                                 anchor_ms=(float(an[i]) if np.isfinite(an[i]) and an[i] > 0
                                            else np.nan),
                                 n=c.get("iters", 50) or 50,
                                 timing_mode=c.get("tm", "?")))
    return pd.DataFrame(rows)


def dec_tiling(T):
    kb = T.get("kernels", [""])
    rows = []
    inv = dict(cu=("cube_util", 100.0), cy=("cycles", 1e-3), mac=("mac_ratio", 1000.0),
               mte2=("mte2_ratio", 1000.0), mte1=("mte1_ratio", 1000.0),
               fix=("fixpipe_ratio", 1000.0), spr=("duration_spread_pct", 100.0))
    for op, c in T["ops"].items():
        M = undz(c["M"])
        dur = undz(c["dur"]) / 1e4 * c["scale"]
        bd = undz(c["bd"]) if c.get("bd") else np.full(len(M), -1)
        ki = undz(c["kb"]) if c.get("kb") else np.zeros(len(M), int)
        extra = {}
        for k, (col, s) in inv.items():
            if c.get(k):
                extra[col] = undz(c[k]) / s
        for i, m in enumerate(M):
            r = dict(op=op, num_tokens=int(m), duration_us=float(dur[i]),
                     block_dim=int(bd[i]), kernel_bin=kb[int(ki[i])])
            for col, v in extra.items():
                r[col] = float(v[i])
            rows.append(r)
    return pd.DataFrame(rows)


def load_digest(path):
    """给 04 用：返回 {文件名: DataFrame} 和 tiling DataFrame。"""
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        blob = json.load(f)
    sweeps = {k: dec_sweep(v) for k, v in blob.items()
              if k not in ("tiling", "env", "meta", "bounds")}
    tiling = dec_tiling(blob["tiling"]) if "tiling" in blob else None
    meta = dict(blob.get("meta", {}))
    if "bounds" in blob:
        meta["bounds"] = blob["bounds"]
    return sweeps, tiling, blob.get("env", {}), meta


# --------------------------------------------------------------- SUMMARY
def _curve(d, op):
    g = d[d.op == op]
    use_anchor = "anchor_ms" in g and g.anchor_ms.notna().all() and (g.anchor_ms > 0).all()
    y = g["median"] / g.anchor_ms if use_anchor else g["median"]
    s = y.groupby(g.num_tokens).median().sort_index()
    return s.index.values.astype(float), s.values.astype(float)


def _noise(d, op):
    """趟间噪声（与 shape 结构无关的唯一估计）。"""
    g = d[d.op == op]
    if "pass_id" not in g or g.pass_id.nunique() < 2:
        return float("nan")
    use_anchor = "anchor_ms" in g and g.anchor_ms.notna().all() and (g.anchor_ms > 0).all()
    g = g.assign(_y=g["median"] / g.anchor_ms if use_anchor else g["median"])
    h = [g[g.pass_id % 2 == k].groupby("num_tokens")._y.mean() for k in (0, 1)]
    m = np.intersect1d(h[0].index, h[1].index)
    if len(m) < 10:
        return float("nan")
    a, b = h[0].loc[m].values, h[1].loc[m].values
    return float(np.std((a - b) / np.maximum((a + b) / 2, 1e-12)) * 100 / np.sqrt(2))


def ident_T(M, y, floor):
    best, row = None, {}
    for T in CAND_T:
        g = np.ceil(M / T).astype(int)
        cvs = [100 * y[g == k].std() / y[g == k].mean()
               for k in np.unique(g) if (g == k).sum() >= 3]
        cv = float(np.median(cvs)) if cvs else float("nan")
        row[T] = cv
        if np.isfinite(cv) and cv <= max(2 * floor, 0.20):
            best = T
    return best, row


def short(name):
    """sweep_llama2-7b_tp1_dense_jitOFF.csv -> dense_jitOFF（模型/tp 在 [FILES] 里已有）"""
    b = os.path.basename(name)
    b = re.sub(r"^sweep_.*?_tp\d+_", "", b)
    return re.sub(r"\.csv$", "", b) or b


def summary_bounds(bounds, out=sys.stdout):
    """06_scan_boundaries.py 找到的台阶。位置本身就是结论，值得单独列出来。"""
    if not bounds:
        return
    print("", file=out)
    print("[BOUNDS]", file=out)
    for name, b in sorted(bounds.items()):
        bs = b.get("boundaries", {})
        lo, hi = b.get("lo", 0), b.get("hi", 0)
        per = int((b.get("args") or {}).get("period") or 256)
        known = sum(1 for k in bs if per > 0 and (int(k) - 1) % per == 0)
        nm, nd = b.get("n_measured", 0), max(b.get("n_dense", 1), 1)
        print("  %s  M in [%d,%d]  实测 %d/%d 点 (%.1f%%)  边界 %d 道"
              " (mod%d %d 道 / 不对齐模数 %d 道)"
              % (name, lo, hi, nm, nd, 100.0 * nm / nd,
                 len(bs), per, known, len(bs) - known), file=out)
        for k in sorted(bs, key=lambda x: int(x)):
            tag = "  <-mod%d" % per if per > 0 and (int(k) - 1) % per == 0 else ""
            body = " ".join("%s%+.1f" % (o.replace("_proj", ""), x)
                            for o, x in sorted(bs[k].items()))
            print("    M=%-6s off+%-4d %s%s"
                  % (k, (int(k) - 1) % per + 1 if per > 0 else 0, body, tag),
                  file=out)
        if b.get("spikes"):
            print("    丢弃的尖峰(跳了没保持，多半是采集侧的): %s"
                  % ",".join(str(x) for x in b["spikes"][:24]), file=out)


def summary(sweeps, tiling, env, bounds=None, out=sys.stdout):
    p = lambda *a: print(*a, file=out)
    p("### ASCEND-DIGEST v1")
    p("# env " + " ".join("%s=%s" % (k, env.get(k, "")) for k in
                          ("device_name", "ASCEND_HOME_PATH", "torch_npu", "device_index")))
    p("#")
    p("# [FILES]  文件 行数 算子 M范围 点数 趟数")
    for f, d in sorted(sweeps.items()):
        p("F %-16s %6d %2d [%d,%d] %4d %d"
          % (short(f), len(d), d.op.nunique(), d.num_tokens.min(), d.num_tokens.max(),
             d.num_tokens.nunique(), d.pass_id.nunique()))
    p("#")
    p("# [NOISE]  文件 算子 趟间噪声% 结构std% med/min 批内CV% 计时 —— med/min>1.1 且 min 稳 = 抢卡")
    for f, d in sorted(sweeps.items()):
        for op in sorted(d.op.unique()):
            M, y = _curve(d, op)
            if len(M) < 5:
                continue
            n = _noise(d, op)
            r = 100 * np.std(y - np.polyval(np.polyfit(M, y, 1), M)) / y.mean()
            g = d[d.op == op]
            mm = float(g["median"].median() / max(g["min"].median(), 1e-12))
            p("N %-16s %-14s %6.2f %6.2f %6.3f %6.2f %s"
              % (short(f), op, n, r, mm, g.cv_pct.median(), g.timing_mode.iloc[0]))
    p("#")
    p("# [STEPS]  文件 算子 M 跳变% —— 相邻点相对线性趋势的偏离超过 4x 趟间噪声")
    for f, d in sorted(sweeps.items()):
        for op in sorted(d.op.unique()):
            M, y = _curve(d, op)
            if len(M) < 20 or int(np.median(np.diff(M))) != 1:
                continue
            n = _noise(d, op)
            if not np.isfinite(n):
                continue
            slope = np.polyfit(M, y, 1)[0]
            dd = 100 * (np.diff(y) - slope * np.diff(M)) / y[:-1]
            # 只看真正相邻的一对：win: 网格里跨空隙的差分会被当成假跳变
            dM = np.diff(M)
            ok = np.abs(dM - np.median(dM)) < 1e-9
            thr = max(4 * n, 1.0)
            for i in np.where((np.abs(dd) > thr) & ok)[0][:14]:
                p("S %-16s %-14s %5d %+7.2f" % (short(f), op, int(M[i + 1]), dd[i]))
    p("#")
    p("# [TILE]   来源 算子 判定T 噪声底% 各T的组内CV%(8,16,32,64,128,256)")
    srcs = [(f, d) for f, d in sorted(sweeps.items())]
    for f, d in srcs:
        for op in sorted(d.op.unique()):
            M, y = _curve(d, op)
            if len(M) < 40 or int(np.median(np.diff(M))) != 1:
                continue
            n = _noise(d, op)
            if not np.isfinite(n):
                continue
            T, row = ident_T(M, y, n)
            p("T %-16s %-14s %5s %6.2f %s"
              % (short(f), op, T, n, " ".join("%.2f" % row[t] for t in CAND_T)))
    if tiling is not None and len(tiling):
        for op, g in tiling.groupby("op"):
            g = g.sort_values("num_tokens")
            M = g.num_tokens.values.astype(float)
            y = g.duration_us.values.astype(float)
            fl = float(g.duration_spread_pct.median()) / 2 if "duration_spread_pct" in g else 0.1
            T, row = ident_T(M, y, fl)
            p("T %-16s %-14s %5s %6.2f %s"
              % ("profiler", op, T, fl, " ".join("%.2f" % row[t] for t in CAND_T)))
    p("#")
    if tiling is not None and len(tiling):
        p("# [TILING] 算子 block_dim种数 kernel种数 切换处 极差中位% cube中位% cube最低%@M")
        for op, g in tiling.groupby("op"):
            g = g.sort_values("num_tokens")
            key = list(zip(g.block_dim, g.kernel_bin))
            sw = [int(g.num_tokens.values[i]) for i in range(1, len(g))
                  if key[i] != key[i - 1]]
            cu = g.cube_util.values if "cube_util" in g else np.array([np.nan])
            p("G %-14s %d %d %s %5.2f %6.2f %6.2f@%d"
              % (op, g.block_dim.nunique(), g.kernel_bin.nunique(), sw[:8],
                 g.duration_spread_pct.median() if "duration_spread_pct" in g else -1,
                 np.nanmedian(cu), np.nanmin(cu),
                 int(g.num_tokens.values[np.nanargmin(cu)]) if np.isfinite(cu).any() else -1))
        p("#")
        p("# [XCHECK] 算子 墙钟残差std% kernel残差std% 相关r 末尾15%的M占总残差比"
          " —— r<0.5 或占比>0.4 = profiler 那份被顺序扫描混叠了")
        base = None
        for f, d in sorted(sweeps.items()):
            if "dense" in f and "jitOFF" in f:
                base = d
        if base is None and sweeps:
            base = list(sweeps.values())[0]
        for op, g in tiling.groupby("op"):
            g = g.sort_values("num_tokens")
            xt = g.num_tokens.values.astype(float)
            yt = g.duration_us.values.astype(float)
            rt = 100 * (yt - np.polyval(np.polyfit(xt, yt, 1), xt)) / yt.mean()
            xw, yw = _curve(base, op)
            if len(xw) < 10:
                continue
            rw = 100 * (yw - np.polyval(np.polyfit(xw, yw, 1), xw)) / yw.mean()
            com = np.intersect1d(xw, xt)
            if len(com) < 20:
                continue
            r = float(np.corrcoef(rw[np.searchsorted(xw, com)],
                                  rt[np.searchsorted(xt, com)])[0, 1])
            q = int(len(xt) * 0.85)
            share = float(np.sum(np.abs(rt[q:])) / max(np.sum(np.abs(rt)), 1e-12))
            p("X %-14s %6.2f %6.2f %6.3f %5.2f" % (op, rw.std(), rt.std(), r, share))
    summary_bounds(bounds, out)
    p("### END")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--env", default="out_env_probe.json")
    ap.add_argument("--out", default="data/digest.json.gz")
    ap.add_argument("--no-file", action="store_true", help="只打印 SUMMARY，不写文件")
    args = ap.parse_args()

    blob, sweeps = {}, {}
    for f in sorted(glob.glob(os.path.join(args.data, "sweep_*.csv"))):
        d = pd.read_csv(f)
        blob[os.path.basename(f)] = enc_sweep(d)
        sweeps[os.path.basename(f)] = d
    bounds = {}
    for f in sorted(glob.glob(os.path.join(args.data, "boundaries_*.json"))):
        try:
            bounds[os.path.basename(f)] = json.load(open(f, encoding="utf-8"))
        except Exception:                                       # noqa: BLE001
            print("  [warn] 读不了 %s，跳过" % f)
    if bounds:
        blob["bounds"] = bounds
    tp = os.path.join(args.data, "tiling_blockdim.csv")
    tiling = None
    if os.path.exists(tp):
        tiling = pd.read_csv(tp)
        blob["tiling"] = enc_tiling(tiling)
    env = {}
    if os.path.exists(args.env):
        try:
            env = json.load(open(args.env, encoding="utf-8")).get("env", {})
        except Exception:                                       # noqa: BLE001
            pass
    blob["env"] = env
    if not sweeps and not bounds:
        sys.exit("[FAIL] %s 下既没有 sweep_*.csv 也没有 boundaries_*.json"
                 % args.data)

    if sweeps:
        summary(sweeps, tiling, env, bounds)
    else:
        # 只跑了 06、还没跑 02 的情况：照样出一份可回传的摘要
        print("### ASCEND-DIGEST v1")
        print("")
        print("[FILES]")
        print("  (本轮没有 sweep_*.csv，只有边界扫描结果)")
        summary_bounds(bounds)
        print("### END")

    if not args.no_file:
        raw = json.dumps(blob, separators=(",", ":")).encode()
        gz = gzip.compress(raw, 9)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "wb") as f:
            f.write(gz)
        src = sum(os.path.getsize(p) for p in glob.glob(os.path.join(args.data, "*.csv")))
        print()
        print("=" * 72)
        print("写出 %s   %.0f KB  (原始 CSV 合计 %.0f KB，压到 1/%.0f)"
              % (args.out, len(gz) / 1024, src / 1024, src / max(len(gz), 1)))
        print("sha256 %s" % hashlib.sha256(gz).hexdigest()[:16])
        print()
        print("回传方式，任选其一：")
        print("  A) 上传 %s（约 %.0f KB）—— 我用 04_analyze.py --digest 全量复算" %
              (args.out, len(gz) / 1024))
        print("  B) 拷不动文件就把上面 ### ASCEND-DIGEST 到 ### END 之间整段贴给我")
        print("=" * 72)


if __name__ == "__main__":
    main()
