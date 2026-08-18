#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pa_srcbranch.py -- 用 AST 把源码里的控制流分支全量列出来，并标出判据变量/tensor。

三种用法：

    # A. 某个调用被哪些 if 包着（pa_pytrace.py 第 4/5 节的候选边直接喂进来）
    python pa_srcbranch.py --file mindspeed_mm/patchs/adaptive_clip_grad_patch.py \\
        --func get_grad_norm_fp32_async --callee multi_tensor_applier --repo <repo>

    # B. 把一个文件/函数里所有分支枚举出来，按"数值敏感度"排序
    python pa_srcbranch.py --scan-file megatron/core/optimizer/optimizer.py --repo <repo>

    # C. 批量：吃 pa_pytrace.py 的 json，对所有候选 caller 自动跑 A
    python pa_srcbranch.py --from-pytrace out/pytrace.json --repo <repo> --out out/branches.md

为什么要做这一步
----------------
trace 只能告诉你"这个函数被调用了 3 次 vs 2 次"，告诉不了你**是哪一行 if 决定的、它判的是什么**。
源码里这件事是确定的：把 AST 拿出来，找到调用点，往上收集所有把它包住的 `if/while/for`，
就得到了完整的守卫条件链；再把条件里出现的名字回溯到它们的赋值语句，就得到了判据变量
以及产生它的那次计算。这一步不需要 `with_stack`，也不依赖 profiling 采到什么，
所以它是"把可能的分支列全"的唯一可靠办法。
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pa_common import force_utf8  # noqa: E402

# 判据里出现这些东西，说明它依赖设备算出来的值 —— 仿真数值一偏就可能翻转
DEVICE_VALUE_PAT = re.compile(
    r"\.item\(\)|\.cpu\(\)|\.tolist\(\)|\.numpy\(\)|\.any\(\)|\.all\(\)|\.max\(\)|\.min\(\)|"
    r"\.sum\(\)|\.numel\(\)|\.nonzero\(|\.count_nonzero\(|is_nonzero|torch\.(any|all|isnan|isinf|"
    r"isfinite|equal|allclose|nonzero|argmax|argmin|topk|unique)", re.I)
# `float(x)` / `int(x)` 本身不说明什么（`norm_type = float(norm_type)` 只是个类型转换），
# 只有当被转换的东西看起来是张量时才算把设备值拉到了 host。
_CAST_ON_TENSOR = re.compile(
    r"(bool|int|float)\s*\(\s*[^)]*(inf|nan|flag|found|mask|norm|loss|scale|tensor|"
    r"\.item\(\)|\.sum\(|\.max\(|\.numel\()", re.I)

TENSORISH_NAME = re.compile(
    r"norm|inf|nan|found|scale|coeff|clip|loss|mask|idx|index|len|count|num_|size|shape|"
    r"success|flag|overflow|skip|topk|expert|token|capacity|seq|batch|finish|eos|done|"
    r"threshold|ratio|prob|logit|grad", re.I)

CONFIG_NAME = re.compile(r"^(args|self\.config|config|cfg|opt|options|self\.args)\b", re.I)


def _src_seg(src, node, lines):
    try:
        seg = ast.get_source_segment(src, node)
        if seg:
            return seg.strip()
    except Exception:
        pass
    a = getattr(node, "lineno", 1) - 1
    b = getattr(node, "end_lineno", a + 1)
    return "\n".join(lines[a:b]).strip()


def _names(node):
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute):
            parts, cur = [], n
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
                out.append(".".join(reversed(parts)))
        elif isinstance(n, ast.Name):
            out.append(n.id)
    seen, res = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res


def _parents(tree):
    pm = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            pm[child] = parent
    return pm


def _guards(node, pm, src, lines, stop_at=None):
    """收集把 node 包住的所有条件，从外到内。"""
    chain, cur, child = [], pm.get(node), node
    while cur is not None and cur is not stop_at:
        if isinstance(cur, ast.If):
            branch = "then"
            if any(child is s or child in ast.walk(s) for s in cur.orelse):
                branch = "else"
            chain.append({"kind": "if", "line": cur.lineno, "branch": branch,
                          "test": _src_seg(src, cur.test, lines), "names": _names(cur.test)})
        elif isinstance(cur, (ast.While,)):
            chain.append({"kind": "while", "line": cur.lineno, "branch": "body",
                          "test": _src_seg(src, cur.test, lines), "names": _names(cur.test)})
        elif isinstance(cur, ast.For):
            chain.append({"kind": "for", "line": cur.lineno, "branch": "body",
                          "test": _src_seg(src, cur.iter, lines), "names": _names(cur.iter)})
        elif isinstance(cur, ast.Try):
            chain.append({"kind": "try", "line": cur.lineno, "branch": "body",
                          "test": "(exception path)", "names": []})
        child = cur
        cur = pm.get(cur)
    return list(reversed(chain))


def _defs_of(funcnode, name, src, lines):
    """在函数体里找 name 的赋值处 —— 判据 tensor 是谁算出来的。"""
    base = name.split(".")[0]
    out = []
    for n in ast.walk(funcnode):
        targets = []
        if isinstance(n, ast.Assign):
            targets = n.targets
        elif isinstance(n, (ast.AnnAssign, ast.AugAssign)):
            targets = [n.target]
        elif isinstance(n, (ast.For,)):
            targets = [n.target]
        else:
            continue
        for t in targets:
            for nm in _names(t):
                if nm == name or nm.split(".")[0] == base:
                    out.append({"line": n.lineno, "src": _src_seg(src, n, lines)[:200]})
                    break
    return out[:6]


# 一个名字如果由这些东西算出来，就认为它带着"设备数值"的味道
TAINT_SOURCE = re.compile(
    r"all_reduce|allreduce|all_gather|reduce_scatter|dist\.|torch\.distributed|"
    r"l2_norm|multi_tensor|\.norm\(|\.grad\b|\.sum\(|\.mean\(|\.max\(|\.min\(|\.abs\(|"
    r"float_status|found_inf|get_grad_norm|clip_coeff|\.item\(\)|\.cpu\(\)|\.tolist\(\)|"
    r"torch\.tensor|\.numel\(\)|\.size\(|\.shape\b|topk|softmax|argmax|nonzero", re.I)

BUILTIN_NAMES = {"isinstance", "len", "bool", "int", "float", "str", "max", "min", "sum",
                 "any", "all", "type", "getattr", "hasattr", "torch", "np", "abs", "print",
                 "range", "enumerate", "list", "dict", "set", "tuple", "None", "True", "False"}

_RE_PURE_ISINSTANCE = re.compile(r"^\s*(not\s+)?isinstance\s*\(", re.I)


class FuncCtx(object):
    """函数内的轻量数据流：判据里的名字最终是不是从设备值算出来的。

    只看单个函数体，不做跨函数分析——够用了，因为分支和它的判据几乎总是写在一起，
    而跨函数的部分由 pa_pytrace.py 的调用边负责串起来。
    """

    def __init__(self, funcnode, src, lines):
        self.defs = {}
        self._memo = {}
        if funcnode is None:
            return
        for n in ast.walk(funcnode):
            targets = []
            if isinstance(n, ast.Assign):
                targets = n.targets
            elif isinstance(n, (ast.AnnAssign, ast.AugAssign)):
                targets = [n.target]
            elif isinstance(n, ast.For):
                targets = [n.target]
            else:
                continue
            seg = _src_seg(src, n, lines)
            for t in targets:
                for nm in _names(t):
                    self.defs.setdefault(nm.split(".")[0], []).append(
                        {"line": n.lineno, "src": seg[:200]})

    def tainted(self, name, depth=0):
        base = name.split(".")[0]
        if base in BUILTIN_NAMES or depth > 3:
            return False
        if base in self._memo:
            return self._memo[base]
        self._memo[base] = False            # 防环
        res = False
        for d in self.defs.get(base, []):
            if TAINT_SOURCE.search(d["src"]) or DEVICE_VALUE_PAT.search(d["src"]) \
                    or _CAST_ON_TENSOR.search(d["src"]):
                res = True
                break
            rhs = d["src"].split("=", 1)[-1]
            for nm in _names(ast.parse(rhs.strip()).body[0]) if _safe(rhs) else []:
                if self.tainted(nm, depth + 1):
                    res = True
                    break
            if res:
                break
        self._memo[base] = res
        return res

    def defs_of(self, name):
        return self.defs.get(name.split(".")[0], [])[:4]


def _safe(expr):
    try:
        ast.parse(expr.strip())
        return True
    except Exception:
        return False


def _sensitivity(test_src, names, ctx=None):
    """0 = 静态配置/类型分支，1 = 可能依赖运行时值，2 = 明确依赖设备数值。"""
    t = test_src or ""
    if _RE_PURE_ISINSTANCE.match(t):
        # 类型分派：确实是个分支，但翻转的原因是"框架返回了 tensor 还是 float"，
        # 不是数值误差，所以压到中等而不是最高。
        return 1
    if DEVICE_VALUE_PAT.search(t) or _CAST_ON_TENSOR.search(t):
        return 2
    if ctx is not None and any(ctx.tainted(n) for n in names):
        return 2
    real_names = [n for n in names if n.split(".")[0] not in BUILTIN_NAMES]
    if real_names and all(CONFIG_NAME.match(n) for n in real_names):
        return 0
    if any(TENSORISH_NAME.search(n) for n in real_names):
        return 1
    return 0


SENS_LABEL = {2: "🔴 依赖设备数值（仿真最可能翻转）", 1: "🟡 可能依赖运行时数值",
              0: "⚪ 看起来是静态配置"}


def load_file(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        src = fh.read()
    return src, src.splitlines(), ast.parse(src)


def find_callsites(path, func_name, callee, context=0):
    src, lines, tree = load_file(path)
    pm = _parents(tree)
    funcs = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and (not func_name or n.name == func_name)]
    cal = (callee or "").split(".")[-1]
    hits = []
    for fn in funcs:
        for n in ast.walk(fn):
            if not isinstance(n, ast.Call):
                continue
            target = n.func
            nm = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
            if cal and nm != cal:
                continue
            guards = _guards(n, pm, src, lines, stop_at=fn)
            ctx = FuncCtx(fn, src, lines)
            for g in guards:
                g["names"] = [x for x in g["names"] if x.split(".")[0] not in BUILTIN_NAMES]
                g["sensitivity"] = _sensitivity(g["test"], g["names"], ctx)
            hits.append({
                "file": path, "func": fn.name, "func_line": fn.lineno,
                "call_line": n.lineno, "call_src": _src_seg(src, n, lines)[:200],
                "guards": guards,
                "guard_defs": {g["line"]: {nm2: ctx.defs_of(nm2) for nm2 in g["names"][:6]}
                               for g in guards},
                "context": _context(lines, n.lineno, context) if context else "",
            })
    return hits, src, lines


def _context(lines, line, n):
    a, b = max(0, line - 1 - n), min(len(lines), line + n)
    return "\n".join("%5d | %s" % (i + 1, lines[i]) for i in range(a, b))


def scan_branches(path, func_name=None, context=0):
    src, lines, tree = load_file(path)
    pm = _parents(tree)
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, (ast.If, ast.While, ast.IfExp)):
            continue
        fn = None
        cur = pm.get(n)
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn = cur
                break
            cur = pm.get(cur)
        if func_name and (fn is None or fn.name != func_name):
            continue
        test_src = _src_seg(src, n.test, lines)
        names = [x for x in _names(n.test) if x.split(".")[0] not in BUILTIN_NAMES]
        ctx = FuncCtx(fn, src, lines)
        sens = _sensitivity(test_src, names, ctx)
        out.append({
            "file": path, "func": fn.name if fn else "(module)",
            "line": n.lineno, "kind": type(n).__name__,
            "test": test_src[:220], "names": names[:10], "sensitivity": sens,
            "tainted": [nm for nm in names[:8] if ctx.tainted(nm)],
            "defs": {nm: ctx.defs_of(nm) for nm in names[:5]},
            "context": _context(lines, n.lineno, context) if context else "",
        })
    out.sort(key=lambda x: (-x["sensitivity"], x["line"]))
    return out


# --------------------------------------------------------------------------


def md_callsites(hits, title):
    L = ["## %s" % title, ""]
    if not hits:
        L += ["(没有找到调用点：函数名/被调用名可能对不上，或者调用发生在别的文件里)", ""]
        return L
    for h in hits:
        L.append("### `%s` 第 %d 行 → `%s`" % (h["func"], h["call_line"], h["call_src"][:80]))
        L.append("")
        if not h["guards"]:
            L += ["- **无条件调用**：这个调用不在任何 if 里，次数差异来自更上层的 caller。", ""]
            continue
        L.append("守卫条件链（从外到内）：")
        L.append("")
        for g in h["guards"]:
            sens = g.get("sensitivity", 0)
            L.append("- L%d `%s (%s)` 走 **%s** 分支 — %s" % (
                g["line"], g["kind"], g["test"][:150], g["branch"], SENS_LABEL[sens]))
            for nm in g["names"][:6]:
                defs = h["guard_defs"].get(g["line"], {}).get(nm) or []
                if defs:
                    L.append("  - 判据变量 `%s` 定义于: %s" % (
                        nm, "; ".join("L%d `%s`" % (d["line"], d["src"][:90]) for d in defs[:2])))
                else:
                    L.append("  - 判据变量 `%s`（来自参数或外层作用域）" % nm)
        L.append("")
        if h.get("context"):
            L += ["```python", h["context"], "```", ""]
    return L


def md_scan(rows, path, limit=80):
    L = ["## 文件分支清单: `%s`" % path, "",
         "按数值敏感度排序。🔴 的判据直接来自设备张量，是仿真最容易翻转的地方；"
         "⚪ 多半由启动参数决定，两侧配置一致时不会分叉。", "",
         "| 敏感度 | 位置 | 函数 | 判据 | 判据变量 |", "| --- | --- | --- | --- | --- |"]
    for r in rows[:limit]:
        L.append("| %s | L%d | `%s` | `%s` | %s |" % (
            {2: "🔴", 1: "🟡", 0: "⚪"}[r["sensitivity"]], r["line"], r["func"],
            r["test"].replace("\n", " ")[:110], ", ".join("`%s`" % n for n in r["names"][:5])))
    L.append("")
    hot = [r for r in rows if r["sensitivity"] == 2][:12]
    if hot:
        L += ["### 🔴 判据变量的来源", ""]
        for r in hot:
            L.append("- L%d `%s`" % (r["line"], r["test"][:120]))
            for nm, defs in (r.get("defs") or {}).items():
                if defs:
                    L.append("  - `%s` ← %s" % (nm, "; ".join(
                        "L%d `%s`" % (d["line"], d["src"][:90]) for d in defs[:2])))
        L.append("")
    return L


def main():
    force_utf8()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="", help="仓库根目录，用于把相对路径补全")
    ap.add_argument("--file", default="")
    ap.add_argument("--func", default="")
    ap.add_argument("--callee", default="")
    ap.add_argument("--scan-file", default="")
    ap.add_argument("--scan-func", default="")
    ap.add_argument("--from-pytrace", default="", help="pa_pytrace.py 输出的 json")
    ap.add_argument("--max-callers", type=int, default=12)
    ap.add_argument("--context", type=int, default=6, help="打印调用点上下文行数")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    def resolve(p):
        if os.path.isfile(p):
            return p
        if args.repo:
            q = os.path.join(args.repo, p)
            if os.path.isfile(q):
                return q
            # 路径可能已被 norm_pypath 截断过，去仓库里按后缀找
            tail = p.replace("\\", "/").split("/")[-1]
            for dp, _dn, fns in os.walk(args.repo):
                if tail in fns:
                    cand = os.path.join(dp, tail)
                    if cand.replace("\\", "/").endswith(p.replace("\\", "/")) or True:
                        return cand
        return ""

    L = ["# 源码控制流分支分析", ""]
    if args.scan_file:
        path = resolve(args.scan_file)
        if not path:
            raise SystemExit("找不到文件: %s（试试 --repo 指向仓库根）" % args.scan_file)
        L += md_scan(scan_branches(path, args.scan_func or None, args.context), path)
    elif args.file:
        path = resolve(args.file)
        if not path:
            raise SystemExit("找不到文件: %s（试试 --repo 指向仓库根）" % args.file)
        hits, _, _ = find_callsites(path, args.func or None, args.callee, args.context)
        L += md_callsites(hits, "`%s` 中对 `%s` 的调用" % (os.path.basename(path), args.callee))
    elif args.from_pytrace:
        with open(args.from_pytrace, "r", encoding="utf-8") as fh:
            pt = json.load(fh)
        seen, done = set(), 0
        for e in (pt.get("call_edges") or []) + (pt.get("op_edges") or []):
            if done >= args.max_callers:
                break
            key = (e["caller_file"], e["caller_func"], e.get("callee_func") or e.get("op"))
            if key in seen:
                continue
            seen.add(key)
            path = resolve(e["caller_file"])
            if not path:
                L += ["## `%s`" % e["caller_file"], "", "(仓库里找不到这个文件，跳过)", ""]
                continue
            callee = e.get("callee_func") or ""
            if callee:
                hits, _, _ = find_callsites(path, e["caller_func"], callee, args.context)
                L += md_callsites(hits, "`%s::%s` → `%s`（real %s / sim %s）" % (
                    os.path.basename(path), e["caller_func"], callee, e["real"], e["sim"]))
            else:
                rows = [r for r in scan_branches(path, e["caller_func"], args.context)]
                L += md_scan(rows, "%s::%s（算子 %s: real %s / sim %s）" % (
                    path, e["caller_func"], e.get("op"), e["real"], e["sim"]), limit=25)
            done += 1
    else:
        ap.error("需要 --file / --scan-file / --from-pytrace 之一")

    md = "\n".join(L)
    print(md)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(md)
        print("\nwrote %s" % args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
