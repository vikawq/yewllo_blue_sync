#!/usr/bin/env python3
"""
扫描 vAscend 里所有 aclnn 拦截函数，找出「CPU server 分支比 else 分支少传参数」
以及「dims 非空就提前 return」这两类漏洞。

背景 —— aclnnReduceSum 的实现暴露了一个会重复出现的写法问题：

    aclnnStatus ret = ACLNN_ORIGIN_CALL(...);
    if (ret != OK || (IsUseCpuServer() && dims->Size() != 0)) {
        return ret;                                    // ← (A) 短路：dims 非空就不注册 runner
    }
    auto runner = ...;
    if (IsUseCpuServer()) {
        runner->SetOpName("sum").AddTensor(self).SetOutput(out);
        //                       ↑ 只传了 self，dims/keepDims/dtype 全丢   ← (B) 漏参
    } else {
        runner->SetOpName(NNOP_NAME).AddTensor(self).AddIntArray(dims)
            .AddBool(keepDims).AddDataType(dtype).AddTensor(out).SetOutput(out);
        //  ↑ 这条分支是完整的，说明框架本来就支持传这些参数
    }

(A) 让请求根本不出网（客户端耗时只有几百 us）；
(B) 让服务端拿不到参数，退化成最朴素的调用形式。

判据：**两个分支的 .AddXxx() 调用序列应该一一对应**（除 SetOpName 的实参不同）。
凡是 CPU server 分支比 else 分支少，就是 (B)。

用法：
    python scan_cpu_server_branches.py <源码根目录>
    例：python scan_cpu_server_branches.py vAscend/src
"""
import os
import re
import sys

ADD_RE = re.compile(r"\.(Add[A-Za-z0-9_]+|SetOutput|SetOpName)\s*\(")
GUARD_RE = re.compile(r"IsUseCpuServer\s*\(\s*\)\s*&&")


def block_after(text, start):
    """从 start 处第一个 '{' 开始做花括号配对，返回块内文本和块结束位置。"""
    i = text.find("{", start)
    if i < 0:
        return "", start
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j], j
        j += 1
    return text[i + 1:], len(text)


def scan(path):
    try:
        src = open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return []
    out = []
    for m in re.finditer(r"if\s*\(\s*IsUseCpuServer\s*\(\s*\)\s*\)", src):
        cpu_blk, end = block_after(src, m.end())
        rest = src[end:end + 400].lstrip()
        else_blk = ""
        if rest.startswith("else"):
            else_blk, _ = block_after(src[end:], 0)
        cpu_args = ADD_RE.findall(cpu_blk)
        else_args = ADD_RE.findall(else_blk)
        fn = "?"
        head = src[:m.start()]
        fm = list(re.finditer(r"aclnn(\w+)GetWorkspaceSize", head))
        if fm:
            fn = "aclnn" + fm[-1].group(1)
        line = src[:m.start()].count("\n") + 1
        if else_args and len(cpu_args) < len(else_args):
            missing = [a for a in else_args if a not in cpu_args]
            out.append(("MISSING_ARGS", path, line, fn, cpu_args, else_args, missing))
    for m in GUARD_RE.finditer(src):
        line = src[:m.start()].count("\n") + 1
        head = src[:m.start()]
        fm = list(re.finditer(r"aclnn(\w+)GetWorkspaceSize", head))
        fn = "aclnn" + fm[-1].group(1) if fm else "?"
        ctx = src[max(0, m.start() - 120):m.start() + 120].replace("\n", " ")
        out.append(("SHORT_CIRCUIT", path, line, fn, ctx, None, None))
    return out


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    miss, short = [], []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith((".cpp", ".cc", ".h", ".hpp")):
                for r in scan(os.path.join(dp, fn)):
                    (miss if r[0] == "MISSING_ARGS" else short).append(r)

    print("=" * 100)
    print(f"(A) 短路：`IsUseCpuServer() && ...` 出现在提前 return 的条件里  —— {len(short)} 处")
    print("    这类算子在条件成立时请求根本不出网，服务端加 runner 无效")
    print("=" * 100)
    for _, path, line, fn, ctx, _, _ in short:
        print(f"  {path}:{line}  [{fn}]")
        print(f"      ...{ctx.strip()}...")

    print()
    print("=" * 100)
    print(f"(B) CPU server 分支比 else 分支少传参数  —— {len(miss)} 处")
    print("    这类算子请求发得出去，但服务端拿不到 dim/keepdim/dtype 等参数")
    print("=" * 100)
    for _, path, line, fn, cpu, els, missing in miss:
        print(f"  {path}:{line}  [{fn}]")
        print(f"      cpu :  {' '.join(cpu)}")
        print(f"      else:  {' '.join(els)}")
        print(f"      少了:  {' '.join(missing)}")

    print()
    print(f"合计 {len(short)} 处短路 + {len(miss)} 处漏参。")
    print("已确认的样例：aclnnReduceSum 两类都中（sum(dim=...) 不出网、sum(dtype=...) 丢参数）。")
    print("对照探针 probe_layer.py 的派活清单交叉验证：")
    print("  · 清单①「没出网」的算子 → 应该出现在 (A) 里")
    print("  · 清单②「走服务端但错」的算子 → 应该出现在 (B) 里")


if __name__ == "__main__":
    main()
