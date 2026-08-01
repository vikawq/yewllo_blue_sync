#!/usr/bin/env python3
"""Stream large Ascend profiler exports into a small, shareable summary.

Only Python's standard library is required. The script never modifies the
profiling directory. It discovers every ASCEND_PROFILER_OUTPUT directory under
the input root and writes compact per-rank CSV/JSON summaries plus an overall
Markdown report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


COMM_PATTERN = re.compile(
    r"hccl|hcom|all.?reduce|all.?gather|reduce.?scatter|all.?to.?all|"
    r"broadcast|send|recv|receive|mc2",
    re.IGNORECASE,
)
RANK_PATTERN = re.compile(r"(?:^|_)rank(\d+)(?:_|$)", re.IGNORECASE)


def parse_float(value: Any) -> float:
    if value is None:
        return 0.0
    text = str(value).strip().replace(",", "")
    if not text or text.upper() in {"N/A", "NA", "NONE", "NULL", "-"}:
        return 0.0
    try:
        result = float(text)
        return result if math.isfinite(result) else 0.0
    except ValueError:
        return 0.0


def parse_int(value: Any) -> int:
    return int(parse_float(value))


def clean_text(value: Any, limit: int = 1000) -> str:
    text = "" if value is None else str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


@dataclass
class Aggregate:
    count: int = 0
    total_us: float = 0.0
    min_us: float = math.inf
    max_us: float = 0.0
    wait_us: float = 0.0

    def add(self, duration_us: float, count: int = 1, wait_us: float = 0.0) -> None:
        self.count += count
        self.total_us += duration_us
        self.wait_us += wait_us
        if count > 0:
            avg = duration_us / count
            self.min_us = min(self.min_us, avg)
            self.max_us = max(self.max_us, avg)

    def row(self, key: tuple[str, ...]) -> dict[str, Any]:
        return {
            "key": " | ".join(key),
            "count": self.count,
            "total_us": round(self.total_us, 6),
            "avg_us": round(self.total_us / self.count, 6) if self.count else 0.0,
            "min_us": round(self.min_us, 6) if self.min_us != math.inf else 0.0,
            "max_us": round(self.max_us, 6),
            "wait_us": round(self.wait_us, 6),
        }


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else ["message"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def rank_name(output_dir: Path) -> str:
    for parent in [output_dir.parent, *output_dir.parents]:
        match = RANK_PATTERN.search(parent.name)
        if match:
            return f"rank{match.group(1)}"
    return output_dir.parent.name


def discover_outputs(root: Path) -> list[Path]:
    if root.name == "ASCEND_PROFILER_OUTPUT" and root.is_dir():
        return [root]
    return sorted(path for path in root.rglob("ASCEND_PROFILER_OUTPUT") if path.is_dir())


def inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            rows.append(
                {
                    "relative_path": str(path.relative_to(root)),
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / 1024 / 1024, 3),
                }
            )
    return rows


def sorted_aggregate_rows(
    values: dict[tuple[str, ...], Aggregate], top_n: int
) -> list[dict[str, Any]]:
    items = sorted(values.items(), key=lambda item: item[1].total_us, reverse=True)
    return [aggregate.row(key) for key, aggregate in items[:top_n]]


def summarize_kernel_details(path: Path, top_n: int) -> dict[str, Any]:
    by_kernel: dict[tuple[str, ...], Aggregate] = defaultdict(Aggregate)
    by_signature: dict[tuple[str, ...], Aggregate] = defaultdict(Aggregate)
    by_type: dict[tuple[str, ...], Aggregate] = defaultdict(Aggregate)
    comm: dict[tuple[str, ...], Aggregate] = defaultdict(Aggregate)
    row_count = 0
    first_start = math.inf
    last_end = 0.0

    with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_count += 1
            name = clean_text(row.get("Name"), 500)
            op_type = clean_text(row.get("Type"), 200)
            core = clean_text(row.get("Accelerator Core"), 100)
            duration = parse_float(row.get("Duration(us)"))
            wait = parse_float(row.get("Wait Time(us)"))
            start = parse_float(row.get("Start Time(us)"))
            if start:
                first_start = min(first_start, start)
                last_end = max(last_end, start + duration)

            by_kernel[(name, op_type, core)].add(duration, wait_us=wait)
            by_type[(op_type, core)].add(duration, wait_us=wait)

            signature = (
                name,
                clean_text(row.get("Input Shapes"), 1000),
                clean_text(row.get("Input Data Types"), 300),
                clean_text(row.get("Input Formats"), 300),
                clean_text(row.get("Output Shapes"), 1000),
                clean_text(row.get("Output Data Types"), 300),
                clean_text(row.get("Output Formats"), 300),
            )
            by_signature[signature].add(duration, wait_us=wait)
            if COMM_PATTERN.search(f"{name} {op_type}"):
                comm[(name, op_type, core)].add(duration, wait_us=wait)

    return {
        "row_count": row_count,
        "first_start_us": 0.0 if first_start == math.inf else first_start,
        "last_end_us": last_end,
        "window_us": 0.0 if first_start == math.inf else max(0.0, last_end - first_start),
        "kernel_rows": sorted_aggregate_rows(by_kernel, top_n),
        "type_rows": sorted_aggregate_rows(by_type, top_n),
        "signature_rows": sorted_aggregate_rows(by_signature, top_n),
        "communication_rows": sorted_aggregate_rows(comm, top_n),
        "communication_total_us": sum(value.total_us for value in comm.values()),
        "all_kernel_total_us": sum(value.total_us for value in by_kernel.values()),
    }


def summarize_op_statistic(path: Path, top_n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "device_id": clean_text(row.get("Device_id"), 50),
                    "op_type": clean_text(row.get("OP Type"), 300),
                    "core_type": clean_text(row.get("Core Type"), 100),
                    "count": parse_int(row.get("Count")),
                    "total_us": parse_float(row.get("Total Time(us)")),
                    "min_us": parse_float(row.get("Min Time(us)")),
                    "avg_us": parse_float(row.get("Avg Time(us)")),
                    "max_us": parse_float(row.get("Max Time(us)")),
                    "ratio_percent": parse_float(row.get("Ratio(%)")),
                }
            )
    return sorted(rows, key=lambda row: row["total_us"], reverse=True)[:top_n]


def summarize_operator_details(path: Path, top_n: int) -> list[dict[str, Any]]:
    values: dict[tuple[str, ...], dict[str, float | int]] = defaultdict(
        lambda: {
            "count": 0,
            "host_self_us": 0.0,
            "host_total_us": 0.0,
            "device_self_us": 0.0,
            "device_total_us": 0.0,
            "device_aicore_total_us": 0.0,
        }
    )
    with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        for row in csv.DictReader(handle):
            key = (clean_text(row.get("Name"), 500), clean_text(row.get("Input Shapes"), 1500))
            item = values[key]
            item["count"] += 1
            item["host_self_us"] += parse_float(row.get("Host Self Duration(us)"))
            item["host_total_us"] += parse_float(row.get("Host Total Duration(us)"))
            item["device_self_us"] += parse_float(row.get("Device Self Duration(us)"))
            item["device_total_us"] += parse_float(row.get("Device Total Duration(us)"))
            item["device_aicore_total_us"] += parse_float(row.get("Device Total Duration With AICore(us)"))

    result: list[dict[str, Any]] = []
    for (name, shapes), item in values.items():
        result.append({"name": name, "input_shapes": shapes, **item})
    return sorted(
        result,
        key=lambda row: max(float(row["device_total_us"]), float(row["host_total_us"])),
        reverse=True,
    )[:top_n]


def read_small_csv(path: Path, max_rows: int = 10000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if index >= max_rows:
                break
            rows.append({key: clean_text(value, 1000) for key, value in row.items()})
    return rows


def truncate_json(value: Any, depth: int = 0) -> Any:
    if depth >= 7:
        return "<max-depth>"
    if isinstance(value, dict):
        items = list(value.items())
        result = {str(key): truncate_json(item, depth + 1) for key, item in items[:100]}
        if len(items) > 100:
            result["<truncated-keys>"] = len(items) - 100
        return result
    if isinstance(value, list):
        result = [truncate_json(item, depth + 1) for item in value[:100]]
        if len(value) > 100:
            result.append(f"<truncated-items:{len(value) - 100}>")
        return result
    if isinstance(value, str):
        return clean_text(value, 1000)
    return value


def flatten_communication_metrics(value: Any, limit: int = 20000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def visit(node: Any, path: list[str]) -> None:
        if len(rows) >= limit:
            return
        if isinstance(node, dict):
            for key, child in node.items():
                visit(child, path + [str(key)])
        elif isinstance(node, list):
            for index, child in enumerate(node[:10000]):
                visit(child, path + [str(index)])
                if len(rows) >= limit:
                    break
        elif isinstance(node, (int, float)):
            joined = "/".join(path)
            if re.search(r"time|duration|bandwidth|size|count|ratio|wait|transit|sync", joined, re.I):
                rows.append({"path": joined, "value": node})

    visit(value, [])
    return rows


def summarize_json_file(path: Path, output_dir: Path, copy_max_mb: float) -> dict[str, Any]:
    result = {"name": path.name, "size_bytes": path.stat().st_size, "parsed": False}
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            value = json.load(handle)
        result["parsed"] = True
        result["top_level_type"] = type(value).__name__
        if isinstance(value, dict):
            result["top_level_keys"] = list(value.keys())[:100]
        write_json(output_dir / f"{path.stem}_summary.json", truncate_json(value))
        if path.name in {"communication.json", "communication_matrix.json"}:
            metrics = flatten_communication_metrics(value)
            write_csv(output_dir / f"{path.stem}_metrics.csv", metrics, ["path", "value"])
            result["metric_count"] = len(metrics)
        if path.stat().st_size <= copy_max_mb * 1024 * 1024:
            shutil.copy2(path, output_dir / path.name)
            result["copied_full"] = True
    except Exception as exc:  # profiling exports can contain non-standard JSON values
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def markdown_table(rows: list[dict[str, Any]], columns: list[str], top_n: int = 15) -> list[str]:
    if not rows:
        return ["_无数据_", ""]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows[:top_n]:
        values = [clean_text(row.get(column, ""), 120).replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return lines


def summarize_rank(output_dir: Path, destination: Path, top_n: int, copy_max_mb: float) -> dict[str, Any]:
    rank = rank_name(output_dir)
    rank_out = destination / rank
    rank_out.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"rank": rank, "source": str(output_dir), "files": {}}

    kernel_path = output_dir / "kernel_details.csv"
    if kernel_path.exists():
        kernel = summarize_kernel_details(kernel_path, top_n)
        result["kernel"] = kernel
        write_csv(rank_out / "kernel_top.csv", kernel["kernel_rows"])
        write_csv(rank_out / "kernel_type_top.csv", kernel["type_rows"])
        write_csv(rank_out / "kernel_shape_top.csv", kernel["signature_rows"])
        write_csv(rank_out / "communication_kernel_top.csv", kernel["communication_rows"])

    op_stat_path = output_dir / "op_statistic.csv"
    if op_stat_path.exists():
        rows = summarize_op_statistic(op_stat_path, top_n)
        result["op_statistic"] = rows
        write_csv(rank_out / "op_statistic_top.csv", rows)

    operator_path = output_dir / "operator_details.csv"
    if operator_path.exists():
        rows = summarize_operator_details(operator_path, top_n)
        result["operator"] = rows
        write_csv(rank_out / "operator_top.csv", rows)

    for name in ["step_trace_time.csv", "api_statistic.csv"]:
        path = output_dir / name
        if path.exists():
            rows = read_small_csv(path)
            result[name] = rows
            write_csv(rank_out / name, rows)

    json_summaries = []
    candidates = [
        output_dir / "communication.json",
        output_dir / "communication_matrix.json",
        output_dir.parent / "profiler_metadata.json",
    ]
    candidates.extend(sorted(output_dir.parent.glob("profiler_info_*.json")))
    for path in candidates:
        if path.exists():
            json_summaries.append(summarize_json_file(path, rank_out, copy_max_mb))
    result["json_files"] = json_summaries
    return result


def comparison_rows(results: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, float | int]] = defaultdict(dict)
    ranks = [str(result["rank"]) for result in results]
    for result in results:
        rank = str(result["rank"])
        for row in result.get("kernel", {}).get("kernel_rows", []):
            key = str(row["key"])
            by_key[key][f"{rank}_total_us"] = float(row["total_us"])
            by_key[key][f"{rank}_count"] = int(row["count"])

    rows: list[dict[str, Any]] = []
    for key, values in by_key.items():
        totals = [float(values.get(f"{rank}_total_us", 0.0)) for rank in ranks]
        positive = [value for value in totals if value > 0]
        ratio = max(positive) / min(positive) if len(positive) >= 2 else math.inf
        row: dict[str, Any] = {
            "kernel": key,
            "max_min_ratio": round(ratio, 6) if math.isfinite(ratio) else "only_one_rank",
            "max_total_us": round(max(totals, default=0.0), 6),
        }
        row.update(values)
        rows.append(row)
    return sorted(rows, key=lambda row: float(row["max_total_us"]), reverse=True)[:top_n]


def build_report(root: Path, results: list[dict[str, Any]], comparison: list[dict[str, Any]]) -> str:
    lines = [
        "# Ascend 双卡 Profiling 摘要",
        "",
        f"> 源目录：`{root}`  ",
        f"> 发现 rank/profile 数：{len(results)}  ",
        "> 本报告由流式聚合生成；原始 profiling 文件未被修改。",
        "",
        "## 1. Rank 概览",
        "",
        "| Rank | kernel 行数 | 时间窗口(us) | kernel 累计(us) | 通信 kernel 累计(us) | 通信占累计 kernel 时间 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        kernel = result.get("kernel", {})
        all_total = float(kernel.get("all_kernel_total_us", 0.0))
        comm_total = float(kernel.get("communication_total_us", 0.0))
        ratio = 100.0 * comm_total / all_total if all_total else 0.0
        lines.append(
            f"| {result['rank']} | {kernel.get('row_count', 0)} | "
            f"{kernel.get('window_us', 0):.3f} | {all_total:.3f} | {comm_total:.3f} | {ratio:.2f}% |"
        )
    lines.extend(["", "## 2. 每 rank OP Statistic Top", ""])
    for result in results:
        lines.extend([f"### {result['rank']}", ""])
        lines.extend(
            markdown_table(
                result.get("op_statistic", []),
                ["op_type", "core_type", "count", "total_us", "avg_us", "max_us", "ratio_percent"],
            )
        )

    lines.extend(["## 3. 每 rank Kernel Top", ""])
    for result in results:
        lines.extend([f"### {result['rank']}", ""])
        lines.extend(markdown_table(result.get("kernel", {}).get("kernel_rows", []), ["key", "count", "total_us", "avg_us", "max_us", "wait_us"]))

    lines.extend(["## 4. 两 rank Kernel 对比", ""])
    if len(results) < 2:
        lines.extend(["_只发现一个 rank。请确认是否将 rank1 的 profiler 目录放在输入根目录下。_", ""])
    else:
        columns = ["kernel", "max_min_ratio", "max_total_us"]
        for result in results:
            columns.extend([f"{result['rank']}_total_us", f"{result['rank']}_count"])
        lines.extend(markdown_table(comparison, columns, 30))

    lines.extend(
        [
            "## 5. 解读注意事项",
            "",
            "- `kernel 累计时间` 是所有 kernel duration 的求和，不等于端到端墙钟时间；多 stream 重叠时会更大。",
            "- HCCL AICPU kernel 的 duration 可能包含等待或同步，必须结合两个 rank 和 communication.json 判断。",
            "- profiling 窗口如果包含 warmup、graph capture 或空闲等待，不应直接作为稳态 decode 性能。",
            "- `kernel_shape_top.csv` 可用于识别同一算子的动态 shape/bucket；完整原 CSV 无需分享。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="profiling run 根目录")
    parser.add_argument("--output", required=True, type=Path, help="小型摘要输出目录")
    parser.add_argument("--top", type=int, default=100, help="每类保留的 Top N，默认 100")
    parser.add_argument(
        "--copy-json-max-mb",
        type=float,
        default=10.0,
        help="小于该大小的通信/元数据 JSON 同时复制，默认 10 MB",
    )
    args = parser.parse_args()

    root = args.input.resolve()
    output = args.output.resolve()
    if not root.exists() or not root.is_dir():
        parser.error(f"输入目录不存在：{root}")
    if output == root or root in output.parents:
        parser.error("输出目录不能位于 profiling 输入目录内，避免污染原始数据")

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "file_inventory.csv", inventory(root))
    outputs = discover_outputs(root)
    if not outputs:
        parser.error(f"未找到 ASCEND_PROFILER_OUTPUT：{root}")

    results = [summarize_rank(path, output, args.top, args.copy_json_max_mb) for path in outputs]
    comparison = comparison_rows(results, args.top)
    write_csv(output / "rank_kernel_comparison.csv", comparison)
    write_json(output / "summary.json", results)
    (output / "summary.md").write_text(build_report(root, results, comparison), encoding="utf-8")

    print(f"Completed: discovered {len(results)} rank/profile export(s)")
    print(f"Summary directory: {output}")
    print(f"Start with: {output / 'summary.md'}")
    if len(results) < 2:
        print("Warning: only one rank was found; a dual-card comparison also needs rank1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
