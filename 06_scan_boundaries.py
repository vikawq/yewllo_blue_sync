#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全域台阶搜索采样器 —— 粗扫 + 二分 + 确认，直接产出分段常数成本表。

为什么是一步到位而不是"先 mod 256 分段、再补第二层"：
    M=1844 那道台阶四个 GEMM 全中、幅度 +7.0%，比同周期起点 1793 的 mod-256
    边界（+4.6~6.0%）还大，而 1843 不是任何 2 的幂的倍数。所以"全算子共享 +
    幅度大"不再等于"第一层"，两层没法先验分开。实测普遍程度约 0.75 个/周期
    （>4%、>=3 个算子），放宽到 >2%、>=1 个算子是 2.5 个/周期。

三个阶段：
    1. 粗扫       每 --coarse 个 M 一个点，外加 mod-256 边界（已知，白送）
    2. 二分       对"两端电平对不上"的区间折半查找，log2(coarse) 次定位到 ±0
    3. 确认       每个候选点采 {b-2,b-1,b,b+1,b+2}，要求是**持续位移**而不是尖峰

第 3 步不能省。2026-09-08 实测：某些 M 上四个算子同时多出一个固定 ~70 us 的
每次调用开销（不随算子大小缩放，attn_o 走打包计时时是 70/inner），下一个 M
立刻回落，而同一批 M 在前一天完全没有 —— 是采集侧的东西（多半是显存分配器
不命中）。不做持续性检验的话，边界数会多报 10 倍。

输出:
    data/boundaries_<tag>.json      边界、分段、所有实测点
    data/boundaries_<tag>_grid.txt  可直接喂给 02_sweep_ops.py 的 --grid list:...

离线自检（不碰设备，用已有的 dense 数据当预言机，验证采样器本身）:
    python 06_scan_boundaries.py --replay ../result_0908/digest.json \\
        --replay-sweep sweep_llama2-7b_tp1_wide_1024_2048.csv --lo 1024 --hi 2048
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np


# --------------------------------------------------------------------------
# 预言机：给一个 M，返回 {op: 毫秒}
# --------------------------------------------------------------------------
class Oracle:
    ops: list

    def measure(self, M):
        raise NotImplementedError

    def close(self):
        pass


class ReplayOracle:
    """用已有的稠密数据当预言机。只为验证采样器逻辑，不碰设备。"""

    def __init__(self, path, sweep=None, stat="agg", only=""):
        import pandas as pd
        if path.endswith(".json") or path.endswith(".json.gz"):
            import importlib.util
            here = os.path.dirname(os.path.abspath(__file__))
            sp = importlib.util.spec_from_file_location(
                "_dg", os.path.join(here, "05_digest.py"))
            dg = importlib.util.module_from_spec(sp)
            sp.loader.exec_module(dg)
            sweeps, _, _, _ = dg.load_digest(path)
            if sweep is None:
                sweep = sorted(sweeps)[0]
            if sweep not in sweeps:
                sys.exit(f"[ABORT] 摘要里没有 {sweep}，可选: {sorted(sweeps)}")
            df = sweeps[sweep]
            col = "agg" if "agg" in df.columns else "median"
        else:
            df = pd.read_csv(path)
            col = "median"
        if stat == "min":
            col = "min"
        self.ops = sorted(df.op.unique())
        if only:
            want = [o.strip() for o in only.split(",")]
            self.ops = [o for o in self.ops if o in want]
            if not self.ops:
                sys.exit(f"[ABORT] --ops 没匹配到算子，可选: {sorted(df.op.unique())}")
        self.table = {op: df[df.op == op].groupby("num_tokens")[col].median()
                      for op in self.ops}
        self.calls = 0
        self.misses = 0

    def measure(self, M):
        self.calls += 1
        out = {}
        for op, s in self.table.items():
            if M in s.index:
                out[op] = float(s.loc[M])
            else:
                self.misses += 1
                return None
        return out


class DeviceOracle:
    """真机。每个 M 把所有算子都测一遍 —— 同一个 shape 的开销是共享的，
    多测几个算子几乎不额外花钱，而边界要的是各算子的并集。"""

    def __init__(self, args):
        from common import (DTYPES, MODEL_SPECS, build_ops, env_report,
                            free_memory_gb, pick_timing_mode, setup_device,
                            time_mode)
        self._time_mode = time_mode
        self.info = setup_device(args.device, jit_compile=(
            None if args.jit_compile < 0 else bool(args.jit_compile)))
        self.env = {**env_report(), **self.info}
        spec = MODEL_SPECS[args.model]
        ops = build_ops(spec, tp=args.tp, dtype=DTYPES[args.dtype])
        if args.ops:
            want = [o.strip() for o in args.ops.split(",")]
            ops = {k: v for k, v in ops.items() if k in want}
            if not ops:
                sys.exit(f"[ABORT] --ops 没匹配到算子，可选: {list(ops)}")
        self.ops_mk = ops
        self.ops = list(ops)
        self.args = args
        free = free_memory_gb(args.device)
        print(f"  设备 {args.device}  剩余显存 {free:.1f} GB  "
              f"dtype={args.dtype} tp={args.tp}")
        if free < args.min_free_gb:
            sys.exit(f"[ABORT] 剩余显存 {free:.1f} GB < {args.min_free_gb}")

        probe_M = min(max(args.hi // 2, 64), args.hi)
        self.modes = {}
        for name, mk in ops.items():
            self.modes[name] = pick_timing_mode(mk(probe_M),
                                                min_event_ms=args.min_event_ms)
        print("  计时模式: " + "  ".join(
            f"{k}={v[0]}" + (f"x{v[1]}" if v[0] == "batched" else "")
            for k, v in self.modes.items()))
        self.calls = 0

    def measure(self, M):
        self.calls += 1
        out = {}
        for name, mk in self.ops_mk.items():
            mode, inner = self.modes[name]
            reps = []
            for _ in range(self.args.reps):
                a = self._time_mode(mk(M), mode, inner,
                                    warmup=self.args.warmup,
                                    iters=self.args.iters)
                reps.append(float(np.median(a)))
            out[name] = float(np.median(reps))
        if self.args.cooldown_ms > 0:
            time.sleep(self.args.cooldown_ms / 1000.0)
        return out


# --------------------------------------------------------------------------
# 采样器
# --------------------------------------------------------------------------
class Scanner:
    def __init__(self, oracle, lo, hi, coarse, thr_pct, keep, verbose=True):
        self.o = oracle
        self.lo, self.hi = lo, hi
        self.coarse = coarse
        self.thr = thr_pct
        self.keep = keep
        self.verbose = verbose
        self.y = {}                      # M -> {op: ms}
        self.slope = {}                  # op -> ms / M
        self.spikes = set()

    # ---- 采样（带缓存） ----
    def at(self, M):
        M = int(M)
        if M < self.lo or M > self.hi:
            return None
        if M not in self.y:
            v = self.o.measure(M)
            if v is None:
                return None
            self.y[M] = v
        return self.y[M]

    # ---- 电平：扣掉段内线性上升，剩下的就是"这一段在哪个台阶上" ----
    def level(self, M, op):
        v = self.at(M)
        return None if v is None else v[op] - self.slope[op] * M

    def fit_slope(self, xs):
        """用粗扫增量的**中位数**估斜率。含边界的区间是少数，中位数自动躲开它们。"""
        xs = sorted(x for x in xs if x in self.y)
        for op in self.o.ops:
            d = [(self.y[b][op] - self.y[a][op]) / (b - a)
                 for a, b in zip(xs, xs[1:]) if b > a]
            self.slope[op] = float(np.median(d)) if d else 0.0

    def step_pct(self, a, b):
        """区间两端的电平差，按各算子取最大（返回 最大值%, 该算子）。"""
        best, who = 0.0, None
        for op in self.o.ops:
            la, lb = self.level(a, op), self.level(b, op)
            if la is None or lb is None:
                continue
            base = max(self.y[b][op], 1e-12)
            d = abs(lb - la) / base * 100
            if d > best:
                best, who = d, op
        return best, who

    # ---- 二分：把边界夹到 ±0 ----
    def bisect(self, a, b, op):
        la, lb = self.level(a, op), self.level(b, op)
        while b - a > 1:
            m = (a + b) // 2
            if self.at(m) is None:
                break
            lm = self.level(m, op)
            if abs(lm - la) <= abs(lm - lb):
                a, la = m, lm
            else:
                b, lb = m, lm
        return b

    # ---- 尖峰：先认出来，再谈台阶 ----
    def is_spike(self, M, op):
        """y(M) 同时远离左右邻居，而左右邻居彼此吻合 —— 那 M 就是个孤立尖峰。

        必须先把尖峰单独摘掉，不能指望"中位数会压掉它"：
        1842 处那个 +14% 的尖峰，数值正好落在旧电平和新电平**中间**，
        于是它就是 {1842,1843,1844} 的中位数，任何基于中位的判据都看不见它。
        """
        a, b, c = self.y.get(M - 1), self.y.get(M), self.y.get(M + 1)
        if not (a and b and c):
            return False
        ya, yb, yc = a[op], b[op], c[op]
        base = max(abs(yb), 1e-12)
        d1 = abs(yb - ya) / base * 100
        d2 = abs(yb - yc) / base * 100
        d3 = abs(ya - yc) / base * 100
        return d1 > self.thr and d2 > self.thr and d3 < 0.5 * max(d1, d2)

    def side(self, b, op, direction, want=3, reach=6):
        """从 b 往一侧取 want 个**非尖峰**的电平。"""
        out = []
        rng = range(b - 1, b - 1 - reach, -1) if direction < 0 \
            else range(b, b + reach)
        for m in rng:
            if m < self.lo or m > self.hi:
                break
            if self.at(m) is None:
                break
            if self.is_spike(m, op):
                continue
            v = self.level(m, op)
            if v is not None:
                out.append(v)
            if len(out) >= want:
                break
        return out

    # ---- 确认：持续位移才算台阶 ----
    def confirm(self, b):
        """b 是台阶起点当且仅当：两侧各自内部平坦，而两侧之间差得够大。

        1842 会被拒（右侧摘掉尖峰后是 {1843 基线, 1844 台阶, 1845 台阶}，内部不平），
        1280 也会被拒（右侧 {1280 基线, 1281 台阶, 1282 台阶}）——
        真边界是 1844 和 1281。
        """
        if self.at(b) is None or self.at(b - 1) is None:
            return {}
        ok = {}
        for op in self.o.ops:
            L = self.side(b, op, -1)
            R = self.side(b, op, +1)
            if len(L) < 2 or len(R) < 2:
                continue
            base = max(abs(self.y[b][op]), 1e-12)
            shift = (np.median(R) - np.median(L)) / base * 100
            if abs(shift) < self.thr:
                continue
            # 摘掉尖峰之后，两侧还得各自平坦，否则 b 落在了错的位置
            scatter = max(np.ptp(L), np.ptp(R)) / base * 100
            if scatter > self.keep * abs(shift):
                continue
            # b 本身必须已经在新电平上。少了这一条，1842/1843（一个上尖峰、
            # 一个下尖峰）会因为"跳过尖峰后两侧都很平"而各自被判成边界，
            # 于是同一道台阶报三次。
            if self.is_spike(b, op):
                continue
            here = abs(self.level(b, op) - np.median(R)) / base * 100
            if here > self.keep * abs(shift):
                continue
            ok[op] = float(shift)
        return ok

    # ---- 主搜索 ----
    def run(self, seeds):
        # 阶段 1：粗扫。mod-256 边界是白送的先验，直接切进网格，
        # 省掉发现它们的 log2(coarse) 次探测。
        grid = set(range(self.lo, self.hi + 1, self.coarse))
        grid |= {self.lo, self.hi}
        for s in seeds:
            if self.lo < s <= self.hi:
                grid |= {s - 1, s}
        grid = sorted(x for x in grid if self.lo <= x <= self.hi)
        if self.verbose:
            print(f"\n阶段 1 粗扫: {len(grid)} 个点 "
                  f"(步长 {self.coarse}, 含 {len(seeds)} 个已知 mod-256 边界)")
        for k, M in enumerate(grid):
            if self.at(M) is None:
                continue
            if self.verbose and (k + 1) % 40 == 0:
                print(f"    {k+1}/{len(grid)}")
        self.fit_slope(grid)

        # 阶段 2+3：对每个"两端电平对不上"的粗区间递归定位
        found = {}
        for s in seeds:
            if self.lo < s <= self.hi:
                got = self.confirm(s)
                if got:
                    found[s] = got
        if self.verbose:
            print(f"\n阶段 2 二分 + 阶段 3 确认")
        todo = [(a, b) for a, b in zip(grid, grid[1:]) if b - a > 1]
        while todo:
            a, b = todo.pop(0)
            if b - a < 1 or self.at(a) is None or self.at(b) is None:
                continue
            d, op = self.step_pct(a, b)
            if op is None or d < self.thr:
                continue
            x = self.bisect(a, b, op)
            if x in found or x in self.spikes:
                continue
            got = self.confirm(x)
            if got:
                found[x] = got
                if self.verbose:
                    amp = max(abs(v) for v in got.values())
                    print(f"    边界 M={x:5d}   {len(got)} 个算子   "
                          f"最大位移 {amp:5.2f}%   (已测 {len(self.y)} 点)")
                # 一个粗区间里可能不止一道台阶，两边继续找
                if x - 1 > a:
                    todo.append((a, x - 1))
                if b > x:
                    todo.append((x, b))
            else:
                self.spikes.add(x)
                if self.verbose:
                    print(f"    [尖峰] M={x:5d} 跳了但没保持，丢弃并继续找这个区间")
                if x - 1 > a:
                    todo.append((a, x - 1))
                if b > x + 1:
                    todo.append((x + 1, b))
        return found


# --------------------------------------------------------------------------
def predict(op, bounds, lo, hi, measured, targets):
    """分段内线性插值。段边界处不跨段 —— 这正是"边界对齐采样"的全部意义。"""
    segs = segments_for(op, bounds, lo, hi)
    xs = np.array(sorted(measured), dtype=float)
    ys = np.array([measured[int(x)][op] for x in xs], dtype=float)
    out = {}
    for a, b in segs:
        inside = (xs >= a) & (xs <= b)
        sx, sy = xs[inside], ys[inside]
        tg = [t for t in targets if a <= t <= b]
        if len(sx) == 0 or not tg:
            continue
        if len(sx) == 1:
            for t in tg:
                out[t] = float(sy[0])
        else:
            for t in tg:
                out[t] = float(np.interp(t, sx, sy))
    return out


def evaluate(oracle, bounds, lo, hi, measured, budget):
    """拿采样器给出的表去预测**全部** M，和真值比。
    对照组是等预算的均匀网格（Vidur 的做法）。"""
    print()
    print("=" * 92)
    print("离线评估：用采样到的点预测全部 M（真值来自 replay 的稠密数据）")
    print("=" * 92)
    print("  注：真值里那些「跳上去下一格就掉回来」的单点（实测是 ~70us 的采集侧假象）")
    print("      不计入误差 —— 它们不是 shape 的函数，任何模型都不该去拟合它们。")
    print()
    print(f"{'算子':16s}{'段数':>6s}{'剔除尖峰':>9s}{'边界对齐 MAPE':>15s}{'max':>9s}"
          f"{'等预算均匀 MAPE':>17s}{'max':>9s}{'Vidur 32 步 max':>17s}")
    print("-" * 108)
    uni = sorted({int(round(x)) for x in np.linspace(lo, hi, budget)})
    vid = sorted({int(round(x)) for x in np.linspace(lo, hi, 32)})
    rows = []
    for op in oracle.ops:
        s_ = oracle.table[op]
        allM = [int(m) for m in s_.index if lo <= m <= hi]
        truth = np.array([float(s_.loc[m]) for m in allM])
        # 真值里的孤立尖峰：两侧邻居彼此吻合，而它同时远离两侧
        bad = np.zeros(len(allM), dtype=bool)
        for i in range(1, len(allM) - 1):
            if allM[i] - allM[i - 1] != 1 or allM[i + 1] - allM[i] != 1:
                continue
            base = max(abs(truth[i]), 1e-12)
            d1 = abs(truth[i] - truth[i - 1]) / base * 100
            d2 = abs(truth[i] - truth[i + 1]) / base * 100
            d3 = abs(truth[i - 1] - truth[i + 1]) / base * 100
            bad[i] = d1 > 2.0 and d2 > 2.0 and d3 < 0.5 * max(d1, d2)
        pr = predict(op, bounds, lo, hi, measured, allM)
        keep = [i for i, m in enumerate(allM) if m in pr and not bad[i]]
        p1 = np.array([pr[allM[i]] for i in keep])
        t1 = truth[keep]
        e1 = 100 * np.abs(p1 - t1) / t1

        def grid_err(g):
            g = [m for m in g if m in s_.index]
            gy = np.array([float(s_.loc[m]) for m in g])
            pp = np.interp(allM, g, gy)
            e = 100 * np.abs(pp - truth) / truth
            return e[[i for i in range(len(allM)) if not bad[i]]]

        e2, e3 = grid_err(uni), grid_err(vid)
        n_seg = len(segments_for(op, bounds, lo, hi))
        print(f"{op:16s}{n_seg:6d}{int(bad.sum()):9d}{np.mean(e1):14.2f}%"
              f"{np.max(e1):8.2f}%{np.mean(e2):16.2f}%{np.max(e2):8.2f}%"
              f"{np.max(e3):16.2f}%")
        rows.append((np.mean(e1), np.max(e1), np.mean(e2), np.max(e2), np.max(e3)))
    r = np.array(rows)
    print("-" * 108)
    print(f"{'均值':16s}{'':6s}{'':9s}{r[:,0].mean():14.2f}%{r[:,1].mean():8.2f}%"
          f"{r[:,2].mean():16.2f}%{r[:,3].mean():8.2f}%{r[:,4].mean():16.2f}%")
    print(f"  预算: 边界对齐 {budget} 点 vs 均匀 {len(uni)} 点（同预算）"
          f" vs Vidur {len(vid)} 点")


def segments_for(op, bounds, lo, hi):
    edges = sorted({lo} | {b for b, ops in bounds.items() if op in ops} | {hi + 1})
    return [(edges[i], edges[i + 1] - 1) for i in range(len(edges) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--model", default="llama2-7b")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--ops", default="", help="逗号分隔, 留空=全部")
    ap.add_argument("--lo", type=int, default=1)
    ap.add_argument("--hi", type=int, default=4096)
    ap.add_argument("--coarse", type=int, default=32,
                    help="粗扫步长。实测边界平均间隔约 73，32 能保证多数粗区间里最多一道台阶")
    ap.add_argument("--period", type=int, default=256,
                    help="已知的第一层周期，边界在 k*period+1。设 0 关闭这个先验")
    ap.add_argument("--thr", type=float, default=2.0,
                    help="判为台阶的最小持续位移(%%)。低于它的段不值得单独建表")
    ap.add_argument("--keep", type=float, default=0.4,
                    help="持续性：位移至少保留单步跳幅的这个比例，否则判为尖峰")
    ap.add_argument("--reps", type=int, default=3, help="每个 M 原地重复几次取中位")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--cooldown-ms", type=float, default=100.0)
    ap.add_argument("--min-event-ms", type=float, default=0.30)
    ap.add_argument("--jit-compile", type=int, default=0)
    ap.add_argument("--min-free-gb", type=float, default=4.0)
    ap.add_argument("--tag", default="scan")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--replay", default="",
                    help="离线自检：拿已有的 dense CSV / digest.json 当预言机")
    ap.add_argument("--replay-sweep", default="")
    ap.add_argument("--replay-stat", default="agg", choices=["agg", "min", "median"])
    args = ap.parse_args()

    t0 = time.time()
    if args.replay:
        o = ReplayOracle(args.replay, args.replay_sweep or None, args.replay_stat,
                         only=args.ops)
        print(f"  [replay] {args.replay}  算子 {o.ops}")
    else:
        o = DeviceOracle(args)

    seeds = ([] if args.period <= 0 else
             [k * args.period + 1 for k in range(0, args.hi // args.period + 1)
              if args.lo < k * args.period + 1 <= args.hi])
    sc = Scanner(o, args.lo, args.hi, args.coarse, args.thr, args.keep)
    bounds = sc.run(seeds)

    n_dense = args.hi - args.lo + 1
    print()
    print("=" * 92)
    print(f"结果: {len(bounds)} 道边界   实测 {len(sc.y)} 个 M / 全测 {n_dense} "
          f"({100*len(sc.y)/n_dense:.1f}%)   丢弃尖峰 {len(sc.spikes)} 个")
    print(f"      用时 {(time.time()-t0)/60:.1f} 分钟")
    print("=" * 92)
    per = args.period if args.period > 0 else 256
    known = {b for b in bounds if args.period > 0 and (b - 1) % per == 0}
    print(f"  其中 mod {per} 的第一层边界 {len(known)} 道，"
          f"其余 {len(bounds)-len(known)} 道不对齐模数"
          f"（{(len(bounds)-len(known))/max(n_dense/per,1):.2f} 道/周期）")
    print()
    print(f"{'M':>7s}{'周期内偏移':>11s}{'算子数':>7s}   各算子持续位移%")
    print("-" * 92)
    for b in sorted(bounds):
        tag = "  <- mod%d" % per if b in known else ""
        print(f"{b:7d}{(b-1) % per + 1:>11d}{len(bounds[b]):>7d}   "
              + " ".join(f"{k.replace('_proj','')}{v:+.1f}" for k, v in
                         sorted(bounds[b].items())) + tag)

    if args.replay:
        evaluate(o, bounds, args.lo, args.hi, sc.y, len(sc.y))

    os.makedirs(args.out_dir, exist_ok=True)
    grid_pts = sorted({args.lo, args.hi} | set(bounds)
                      | {b - 1 for b in bounds if b - 1 >= args.lo})
    for op in o.ops:
        for a, b in segments_for(op, bounds, args.lo, args.hi):
            grid_pts.append(a + (b - a) // 2)          # 每段中间取一个代表点
    grid_pts = sorted({p for p in grid_pts if args.lo <= p <= args.hi})

    out = dict(
        args=vars(args), lo=args.lo, hi=args.hi,
        n_measured=len(sc.y), n_dense=n_dense,
        slope={k: v for k, v in sc.slope.items()},
        spikes=sorted(sc.spikes),
        boundaries={str(b): v for b, v in sorted(bounds.items())},
        segments={op: segments_for(op, bounds, args.lo, args.hi) for op in o.ops},
        measured={str(k): v for k, v in sorted(sc.y.items())},
        grid=grid_pts,
    )
    jp = os.path.join(args.out_dir, f"boundaries_{args.tag}.json")
    with open(jp, "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=1)
    gp = os.path.join(args.out_dir, f"boundaries_{args.tag}_grid.txt")
    spec = "list:" + ",".join(str(x) for x in grid_pts)
    with open(gp, "w", encoding="utf-8") as fp:
        fp.write(spec + "\n")

    print()
    print(f"  已写出 {jp}")
    print(f"  已写出 {gp}   （{len(grid_pts)} 个点）")
    print()
    print("  下一步：这些点是**边界对齐**的，拿去正式多趟测一遍就是成本表 ——")
    print("    python 02_sweep_ops.py --device %d --model %s --tp %d --dtype %s \\"
          % (args.device, args.model, args.tp, args.dtype))
    print("        --grid \"$(cat %s)\" --tag costtable \\" % gp)
    print("        --jit-compile 0 --order random --repeats 4 "
          "--cooldown-ms 200 --timing auto --anchor 4096")


if __name__ == "__main__":
    main()
