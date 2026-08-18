#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pa_pytrace.py -- 从两侧 trace 的 Python 调用树里把控制流差异全量枚举出来。

    python pa_pytrace.py --real REAL/trace_view.json --sim SIM/trace_view.json \\
        --repo D:/code/MindSpeed-MM --out out/pytrace.md

为什么这一步比逐个算子对齐更接近答案
------------------------------------
`with_stack=True` 时 torch profiler 会给每次 Python 调用发一个事件，名字就是
`/path/file.py(123): func`，并且天然按调用关系嵌套。把两侧的调用树各建一遍再做差，
得到的东西正是我们要找的：

- 只在真机出现的函数 = 仿真**没走到**的代码路径
- 只在仿真出现的函数 = 仿真器的桩/替换实现（`*_mock`、`simulator_patch/*` 之类）
- 同名函数行号不同 = 两侧装的**不是同一个版本**的库或 patch
- caller→callee 调用次数不同 = caller 里有一个 `if`/循环走了不同的分支
- caller→算子 数量不同 = 这个分支**具体多发/少发了哪些算子**

最后一条把"分支"和"算子"直接连了起来，不需要再靠猜。第 4 节的每一行都是一个候选分支，
把它喂给 pa_srcbranch.py 就能拿到判据表达式和涉及的变量/tensor 名。
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pa_common import (force_utf8, load_trace, parse_pyfunc, norm_pypath,  # noqa: E402
                       normalize_name, find_steps, write_json, shape_of)

STUB_HINT = re.compile(r"mock|stub|fake|dummy|simulat|sim_|_sim\b|noop|bypass|skip", re.I)


def build_calltree(trace, roots, t0=None, t1=None, max_children=40):
    """返回 (sites, edges, op_edges, examples)

    sites:    (file, func) -> Counter{line: n}
    edges:    ((file, func), (file, func)) -> n          父函数调用子函数
    op_edges: ((file, func), op_name)      -> n          父函数直接下发的算子
    """
    sites = defaultdict(Counter)
    edges = Counter()
    op_edges = Counter()
    examples = {}
    shapes = defaultdict(Counter)
    n_py = 0

    for tk in trace.tracks_by_side("host"):
        stack = []                       # [(end_ts, key, ev)] 只放 pyfunc
        for ev in tk.events:
            ts, end = ev["ts"], ev["ts"] + ev["dur"]
            while stack and stack[-1][0] <= ts:
                stack.pop()
            pf = parse_pyfunc(ev["name"])
            parent = stack[-1][1] if stack else None
            if pf:
                n_py += 1
                f, line, fn = pf
                key = (norm_pypath(f, roots), fn)
                sites[key][line] += 1
                if parent is not None:
                    edges[(parent, key)] += 1
                examples.setdefault(key, ev["uid"])
                stack.append((end, key, ev))
            elif parent is not None:
                op = normalize_name(ev["name"])
                op_edges[(parent, op)] += 1
                sh = shape_of(ev)
                if sh:
                    shapes[(parent, op)][sh] += 1
    return {"sites": sites, "edges": edges, "op_edges": op_edges,
            "examples": examples, "shapes": shapes, "n_py": n_py}


def _fmt_site(key):
    return "%s :: %s" % key


THIRD_PARTY = re.compile(r"^(torch|torch_npu|numpy|apex|transformers|datasets|accelerate|"
                         r"tokenizers|safetensors|einops|_distutils|importlib|"
                         r"logging|typing|collections|json|os|re)[/\\]", re.I)


def _in_repo(key, repo_markers):
    """模型仓库里的代码优先：分歧要能落到自己能改的代码上才有用。

    显式给的 repo 名优先；没匹配上时，只要不是第三方库路径就当成模型代码——
    路径经过归一化后，模型代码通常长这样：`mindspeed_mm/training.py`。
    """
    p = key[0].lower()
    if any(m and m in p for m in repo_markers):
        return True
    return not THIRD_PARTY.match(p.lstrip("./"))


def diff(real, sim, repo_markers, top=60):
    rs, ss = real["sites"], sim["sites"]
    only_real, only_sim, moved = [], [], []
    for k, lines in rs.items():
        if k not in ss:
            only_real.append((k, sum(lines.values()), sorted(lines)))
    for k, lines in ss.items():
        if k not in rs:
            only_sim.append((k, sum(lines.values()), sorted(lines)))
    for k in set(rs) & set(ss):
        lr, ls = set(rs[k]), set(ss[k])
        if lr != ls:
            moved.append((k, sorted(lr), sorted(ls)))
    only_real.sort(key=lambda x: (not _in_repo(x[0], repo_markers), -x[1]))
    only_sim.sort(key=lambda x: (not _in_repo(x[0], repo_markers), -x[1]))
    moved.sort(key=lambda x: not _in_repo(x[0], repo_markers))

    def edge_delta(re_, se_):
        out = []
        for k in set(re_) | set(se_):
            a, b = re_.get(k, 0), se_.get(k, 0)
            if a != b:
                out.append({"caller": k[0], "callee": k[1], "real": a, "sim": b, "delta": b - a})
        out.sort(key=lambda x: (not _in_repo(x["caller"], repo_markers), -abs(x["delta"])))
        return out

    return {
        "only_real": only_real[:top], "only_sim": only_sim[:top], "moved": moved[:top],
        "call_edges": edge_delta(real["edges"], sim["edges"])[:top],
        "op_edges": edge_delta(real["op_edges"], sim["op_edges"])[:top],
        "counts": {"real_py_events": real["n_py"], "sim_py_events": sim["n_py"],
                   "real_funcs": len(rs), "sim_funcs": len(ss)},
    }


def to_md(d, real_shapes, sim_shapes, repo, args):
    c = d["counts"]
    L = ["# Python 执行路径差异（控制流分支全量枚举）", "",
         "- real py 事件 %d / 函数 %d；sim py 事件 %d / 函数 %d" % (
             c["real_py_events"], c["real_funcs"], c["sim_py_events"], c["sim_funcs"]),
         "- repo: `%s`" % (repo or "(未指定)"), ""]
    if c["real_py_events"] == 0 or c["sim_py_events"] == 0:
        L += ["> **一侧没有 python_function 事件**，说明这份 profiling 采集时没开 `with_stack=True`。",
              "> 本分析的绝大部分能力依赖它。请重采：",
              "> `torch_npu.profiler.profile(..., with_stack=True, with_modules=True, record_shapes=True)`",
              "> 在拿到之前，只能退回到算子序列层面（pa_align.py）做粗定位。", ""]
        return "\n".join(L)

    L += ["## 1. 只在真机执行到的函数（仿真跳过的代码路径）", "",
          "| 函数 | 真机调用次数 | 行号 |", "| --- | --- | --- |"]
    for k, n, lines in d["only_real"]:
        L.append("| `%s` | %d | %s |" % (_fmt_site(k), n, ",".join(map(str, lines[:4]))))
    if not d["only_real"]:
        L.append("| (无) | | |")

    L += ["", "## 2. 只在仿真执行到的函数（仿真器的桩 / 替换实现）", "",
          "带 ⚑ 的名字里含 mock/stub/simulat 等字样，几乎可以确定是仿真器替换掉的实现——"
          "**这类差异不是模型数值问题，是仿真器没实现真实逻辑**，要单独归类。", "",
          "| 函数 | 仿真调用次数 | 行号 |", "| --- | --- | --- |"]
    for k, n, lines in d["only_sim"]:
        flag = " ⚑" if STUB_HINT.search(k[0]) or STUB_HINT.search(k[1]) else ""
        L.append("| `%s`%s | %d | %s |" % (_fmt_site(k), flag, n, ",".join(map(str, lines[:4]))))
    if not d["only_sim"]:
        L.append("| (无) | | |")

    L += ["", "## 3. 同一函数、两侧行号不同（版本 / patch 不一致）", "",
          "同名同文件但行号不同，说明两边装的不是同一份代码。这会让后面所有"
          "『行为不同』的结论失去意义——**先把版本对齐再谈仿真精度**。", "",
          "| 函数 | 真机行号 | 仿真行号 |", "| --- | --- | --- |"]
    for k, lr, ls in d["moved"]:
        L.append("| `%s` | %s | %s |" % (_fmt_site(k), lr[:4], ls[:4]))
    if not d["moved"]:
        L.append("| (无) | | |")

    L += ["", "## 4. 调用次数不同的调用边 —— **候选分支清单**", "",
          "每一行的含义：`caller` 里存在一个分支或循环，它决定了 `callee` 被调用几次。"
          "两侧次数不同 ⇒ 这个分支两侧走向不同。按是否在模型仓库、差异大小排序。", "",
          "| caller | callee | real | sim | Δ |", "| --- | --- | --- | --- | --- |"]
    for e in d["call_edges"]:
        L.append("| `%s` | `%s` | %d | %d | %+d |" % (
            _fmt_site(e["caller"]), _fmt_site(e["callee"]), e["real"], e["sim"], e["delta"]))
    if not d["call_edges"]:
        L.append("| (无) | | | | |")

    L += ["", "## 5. 各 Python 函数直接下发的算子数差异 —— **分支 ⇄ 算子的对应关系**", "",
          "这一节把『哪个分支』和『多了/少了哪些算子』直接连起来：同一个 caller 下，"
          "某个算子的数量两侧不同，就是该分支造成的算子差异。写报告时的"
          "『受影响算子清单』直接取这里。", "",
          "| 发起的 Python 函数 | 算子 | real | sim | Δ | 典型 shape(real→sim) |",
          "| --- | --- | --- | --- | --- | --- |"]
    for e in d["op_edges"]:
        key = (e["caller"], e["callee"])
        sr = real_shapes.get(key)
        ss_ = sim_shapes.get(key)
        shp = "%s → %s" % (sr.most_common(1)[0][0] if sr else "-",
                           ss_.most_common(1)[0][0] if ss_ else "-")
        L.append("| `%s` | `%s` | %d | %d | %+d | %s |" % (
            _fmt_site(e["caller"]), e["callee"], e["real"], e["sim"], e["delta"], shp[:70]))
    if not d["op_edges"]:
        L.append("| (无) | | | | | |")

    same_cnt_diff_shape = []
    for key, sr in real_shapes.items():
        ss_ = sim_shapes.get(key)
        if not ss_:
            continue
        a, b = sr.most_common(1)[0], ss_.most_common(1)[0]
        if a[0] != b[0]:
            same_cnt_diff_shape.append((key, a[0], b[0], a[1], b[1]))
    same_cnt_diff_shape.sort(key=lambda x: -(x[3] + x[4]))
    L += ["", "## 5b. 算子数量相同但 shape 不同（tensor 级差异，控制流还没岔开）", "",
          "这些位置控制流一致、数据已经不同。它们是**下一个最可能翻转的分支的上游**，"
          "也是 msprobe 该去比对的 tensor 清单。", "",
          "| 发起的 Python 函数 | 算子 | real shape | sim shape |", "| --- | --- | --- | --- |"]
    for key, a, b, _na, _nb in same_cnt_diff_shape[:25]:
        L.append("| `%s` | `%s` | %s | %s |" % (_fmt_site(key[0]), key[1], a[:60], b[:60]))
    if not same_cnt_diff_shape:
        L.append("| (无) | | | |")

    L += ["", "## 6. 下一步：把候选分支变成代码证据", "",
          "对上面表里排在前面、且落在模型仓库内的 caller，逐个跑：", "", "```bash"]
    shown, seen = 0, set()
    for e in d["call_edges"] + d["op_edges"]:
        if shown >= 6:
            break
        caller = e["caller"]
        if not _in_repo(caller, []):
            continue
        callee = e["callee"][1] if isinstance(e["callee"], tuple) else ""
        key = (caller, callee)
        if key in seen:
            continue
        seen.add(key)
        if callee:
            L.append('python pa_srcbranch.py --repo "%s" --file "%s" --func "%s" --callee "%s"'
                     % (repo or "<repo>", caller[0], caller[1], callee))
        else:
            L.append('python pa_srcbranch.py --repo "%s" --scan-file "%s" --scan-func "%s"'
                     % (repo or "<repo>", caller[0], caller[1]))
        shown += 1
    L.append('python pa_srcbranch.py --repo "%s" --from-pytrace <本文件同名的 .json>   # 批量'
             % (repo or "<repo>"))
    L += ["```", "",
          "它会用 AST 找出这些调用被哪些 `if/while/for` 包着、判据表达式是什么、"
          "依赖哪些变量与 tensor。", ""]
    return "\n".join(L)


def main():
    force_utf8()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", required=True)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--repo", default="", help="模型仓库根目录，用于路径归一化与排序")
    ap.add_argument("--extra-root", action="append", default=[],
                    help="额外的路径前缀（torch / torch_npu / mindspeed 源码根），可多次给出")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    roots = [r for r in ([args.repo] + args.extra_root) if r]
    print("loading real ...", file=sys.stderr)
    tr = load_trace(args.real)
    print("loading sim ...", file=sys.stderr)
    ts_ = load_trace(args.sim)

    def win(trace):
        steps = find_steps(trace)
        if args.step is None or not steps:
            return None, None
        i = args.step if args.step >= 0 else len(steps) + args.step
        return steps[i][1], steps[i][2]

    rw, sw = win(tr), win(ts_)
    real = build_calltree(tr, roots, rw[0], rw[1])
    sim = build_calltree(ts_, roots, sw[0], sw[1])
    markers = [os.path.basename(str(r).replace("\\", "/").rstrip("/")).lower() for r in roots]
    markers = [m for m in markers if m] or ["mindspeed", "megatron"]
    d = diff(real, sim, markers, top=args.top)
    md = to_md(d, real["shapes"], sim["shapes"], args.repo, args)
    print(md)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(md)
        write_json(os.path.splitext(args.out)[0] + ".json", {
            "counts": d["counts"],
            "only_real": [{"file": k[0], "func": k[1], "count": n, "lines": ln}
                          for k, n, ln in d["only_real"]],
            "only_sim": [{"file": k[0], "func": k[1], "count": n, "lines": ln}
                         for k, n, ln in d["only_sim"]],
            "moved": [{"file": k[0], "func": k[1], "real_lines": a, "sim_lines": b}
                      for k, a, b in d["moved"]],
            "call_edges": [{"caller_file": e["caller"][0], "caller_func": e["caller"][1],
                            "callee_file": e["callee"][0], "callee_func": e["callee"][1],
                            "real": e["real"], "sim": e["sim"], "delta": e["delta"]}
                           for e in d["call_edges"]],
            "op_edges": [{"caller_file": e["caller"][0], "caller_func": e["caller"][1],
                          "op": e["callee"], "real": e["real"], "sim": e["sim"],
                          "delta": e["delta"]} for e in d["op_edges"]],
        })
        print("\nwrote %s" % args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
