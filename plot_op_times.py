#!/usr/bin/env python3
"""Build a self-contained HTML heatmap from the OP_TIME records in a pytest log."""

import argparse
import codecs
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

MARKER = re.compile(r"(?<!\w)OP_TIME[ \t]+(?=\{)")
STAGE_KIND = "stage_with_probes"


@dataclass
class Timing:
    count: int = 0
    elapsed: float = 0.0
    host_count: int = 0
    host: float = 0.0

    def add(self, elapsed, host):
        self.count += 1
        self.elapsed += elapsed
        if host is not None:
            self.host_count += 1
            self.host += host


def duration(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("duration is not a number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("duration must be finite and nonnegative")
    return value


def label(row, key, default=None):
    value = row.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing or invalid {key}")
    return value


def log_encoding(path):
    with path.open("rb") as file:
        head = file.read(4)
    if head.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"
    return "utf-8-sig"


def read_timings(path):
    totals = defaultdict(Timing)
    decoder = json.JSONDecoder()
    valid = invalid = 0
    bad_lines = []
    with path.open(encoding=log_encoding(path), errors="replace") as log:
        for line_number, line in enumerate(log, 1):
            cursor = 0
            while match := MARKER.search(line, cursor):
                cursor = match.end()
                try:
                    row, cursor = decoder.raw_decode(line, cursor)
                    op = label(row, "op")
                    phase = label(row, "phase")
                    kind = label(row, "kind", "unknown")
                    if op == "__stage_total__":
                        kind = STAGE_KIND
                    key = (
                        phase, kind, op, label(row, "device", "unknown"),
                        label(row, "script", "unknown"), label(row, "test", "unknown"),
                        label(row, "case"),
                    )
                    elapsed = duration(row["elapsed_us"])
                    host = None if row.get("host_us") is None else duration(row["host_us"])
                except (KeyError, TypeError, ValueError, OverflowError):
                    invalid += 1
                    if len(bad_lines) < 8:
                        bad_lines.append(line_number)
                    continue
                totals[key].add(elapsed, host)
                valid += 1
    cells = [
        {
            "phase": phase, "kind": kind, "op": op, "device": device,
            "script": script, "test": test, "case": case,
            "count": timing.count, "elapsed": timing.elapsed,
            "hostCount": timing.host_count, "host": timing.host,
        }
        for (phase, kind, op, device, script, test, case), timing in totals.items()
    ]
    return {"source": path.name, "valid": valid, "invalid": invalid, "badLines": bad_lines, "cells": cells}


HTML = r"""<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>算子耗时热力图</title>
<style>
:root { color-scheme: dark; --page: #101620; --panel: #172130; --ink: #e7eef7; --muted: #aebed1; --border: #344458; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, sans-serif; }
main { padding: clamp(16px, 4vw, 44px); }
h1 { font-size: clamp(26px, 5vw, 38px); margin: 0 0 4px; }
p { margin: 7px 0; }
.meta, .hint { color: var(--muted); }
.controls { display: flex; flex-wrap: wrap; gap: 12px; margin: 20px 0 12px; }
label { display: grid; gap: 4px; color: var(--muted); font-size: 13px; }
select, input { background: var(--panel); color: var(--ink); padding: 8px; border: 1px solid var(--border); border-radius: 6px; font: inherit; }
input { width: 13em; }
.scroll { overflow: auto; max-height: 68vh; margin: 10px 0; border: 1px solid var(--border); border-radius: 8px; }
table { border-spacing: 0; width: max-content; min-width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 7px 10px; border-right: 1px solid var(--border); border-bottom: 1px solid var(--border); }
thead th { position: sticky; top: 0; z-index: 2; background: var(--panel); text-align: left; width: 155px; }
thead th:first-child { z-index: 3; }
th:first-child { position: sticky; left: 0; background: var(--panel); text-align: left; }
tbody th { z-index: 1; white-space: nowrap; }
td { text-align: right; min-width: 105px; }
td.missing { color: var(--muted); background: var(--page); }
small { display: block; font-size: 11px; color: var(--muted); overflow-wrap: anywhere; max-width: 180px; }
.legend { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin: 10px 0; color: var(--muted); }
.legend[hidden] { display: none; }
.gradient { width: 140px; height: 12px; border-radius: 3px; background: linear-gradient(90deg, #212d3d, #873917, #ce7b23, #ffc775); }
.note { color: #ffd18f; }
</style>
<main>
  <h1>算子耗时热力图</h1>
  <p class="meta" id="source"></p>
  <p class="note" id="warning" hidden></p>
  <div class="controls">
    <label>阶段<select id="phase"></select></label>
    <label>记录类型<select id="kind"></select></label>
    <label>指标<select id="metric">
      <option value="elapsed">累计耗时 (µs)</option>
      <option value="mean">平均单次耗时 (µs)</option>
      <option value="count">调用次数</option>
      <option value="host">Host 调用累计 (µs)</option>
    </select></label>
    <label>色阶<select id="scale"><option value="log">对数</option><option value="linear">线性</option></select></label>
    <label>显示行数<select id="limit"><option>30</option><option>100</option><option value="all">全部</option></select></label>
    <label>搜索算子 / 设备<input type="search" id="operator"></label>
    <label>搜索脚本 / 用例<input type="search" id="case"></label>
  </div>
  <p id="selection" aria-live="polite"></p>
  <div class="legend" id="legend"><span>0</span><span class="gradient"></span><span id="max"></span><span>· — 表示该用例没有这项数据</span></div>
  <p class="hint" id="hint"></p>
  <div class="scroll" id="matrix" tabindex="0" aria-label="算子耗时矩阵"></div>
  <noscript>请启用 JavaScript 查看热力图。</noscript>
</main>
<script type="application/json" id="data">__DATA__</script>
<script>
"use strict";
const get = id => document.getElementById(id);
const data = JSON.parse(get("data").textContent);
const format = new Intl.NumberFormat("zh-CN", {maximumFractionDigits: 3});
const integer = new Intl.NumberFormat("zh-CN", {maximumFractionDigits: 0});
const stageKind = "stage_with_probes";
const kindNames = new Map([
  ["dispatcher_op", "计算算子"],
  ["torch_api", "Torch / Tensor API"],
  [stageKind, "整段调用（含打点开销）"]
]);

function node(tag, text) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  return element;
}

function options(select, values, preferred, names = new Map()) {
  const previous = select.value;
  select.replaceChildren(...values.map(value => {
    const option = node("option", names.get(value) ?? value);
    option.value = value;
    return option;
  }));
  select.value = [previous, preferred, values[0]].find(value => values.includes(value));
}

function unique(values) {
  return [...new Set(values)].sort((a, b) => a.localeCompare(b));
}

function updateKinds() {
  const kinds = unique(data.cells.filter(cell => cell.phase === get("phase").value).map(cell => cell.kind));
  options(get("kind"), kinds, kinds.includes("dispatcher_op") ? "dispatcher_op" : kinds.find(kind => kind !== stageKind), kindNames);
}

function valueOf(timing, metric) {
  if (!timing) return null;
  if (metric === "count") return timing.count;
  if (metric === "host") return timing.hostCount ? timing.host : null;
  if (metric === "mean") return timing.elapsed / timing.count;
  return timing.elapsed;
}

function paint(cell, value, max, scale) {
  const weight = max === 0 ? 0 : scale === "log"
    ? Math.log1p(value) / Math.log1p(max) : value / max;
  // Use a solid color from the same palette at the chosen point on the scale.
  const stops = [[33,45,61], [135,57,23], [206,123,35], [255,199,117]];
  const index = Math.min(stops.length - 2, Math.floor(weight * (stops.length - 1)));
  const fraction = weight * (stops.length - 1) - index;
  const rgb = stops[index].map((start, c) => Math.round(start + (stops[index + 1][c] - start) * fraction));
  cell.style.background = "rgb(" + rgb.join(",") + ")";
  cell.style.color = weight > 0.61 ? "#17110c" : "#fff";
}

function render() {
  const phase = get("phase").value;
  const kind = get("kind").value;
  const metric = get("metric").value;
  const scale = get("scale").value;
  const limit = get("limit").value;
  const opFilter = get("operator").value.trim().toLowerCase();
  const caseFilter = get("case").value.trim().toLowerCase();
  const groups = new Map();
  const columns = new Map();

  for (const cell of data.cells) {
    if (cell.phase !== phase || cell.kind !== kind) continue;
    if (!(cell.script + " " + cell.test + " " + cell.case).toLowerCase().includes(caseFilter)) continue;
    const columnKey = JSON.stringify([cell.script, cell.test, cell.case]);
    columns.set(columnKey, cell);
    if (!(cell.op + " " + cell.device).toLowerCase().includes(opFilter)) continue;
    const rowKey = JSON.stringify([cell.op, cell.device]);
    if (!groups.has(rowKey)) {
      groups.set(rowKey, {op: cell.op, device: cell.device, cells: new Map(),
        count: 0, elapsed: 0, host: 0, hostCount: 0});
    }
    const group = groups.get(rowKey);
    group.cells.set(columnKey, cell);
    for (const field of ["count", "elapsed", "host", "hostCount"]) group[field] += cell[field];
  }
  const cases = [...columns.entries()].sort(([a], [b]) => a.localeCompare(b, "en", {numeric: true}));
  const ranked = [...groups.values()].sort((a, b) =>
    (valueOf(b, metric) ?? -1) - (valueOf(a, metric) ?? -1)
    || a.op.localeCompare(b.op) || a.device.localeCompare(b.device)
  );
  const shown = limit === "all" ? ranked : ranked.slice(0, Number(limit));
  const box = get("matrix");
  box.replaceChildren();
  box.hidden = !shown.length;
  get("legend").hidden = !shown.length;
  get("hint").textContent = kind === stageKind
    ? "整段耗时包含打点开销，单独查看。"
    : "颜色按当前视图中最大的单元格值缩放。累计值包含日志中该组合的所有调用，平均值按调用次数计算。";
  get("selection").textContent = shown.length
    ? "显示 " + shown.length + " / " + ranked.length + " 行 · " + cases.length + " 个用例 · 按当前指标排序"
    : "没有匹配的记录。";
  if (!shown.length) return;

  let max = 0;
  for (const group of shown) {
    for (const cell of group.cells.values()) max = Math.max(max, valueOf(cell, metric) ?? 0);
  }
  get("max").textContent = format.format(max) + (metric === "count" ? " 次" : " µs");
  const table = node("table");
  const header = table.createTHead().insertRow();
  header.append(node("th", "算子 / 设备"));
  for (const [, value] of cases) {
    const th = node("th", value.case);
    th.scope = "col";
    th.append(node("small", value.test));
    th.title = value.script + "\n" + value.test + "\n" + value.case;
    header.append(th);
  }
  const tbody = table.createTBody();
  for (const group of shown) {
    const row = tbody.insertRow();
    const name = node("th", group.op);
    name.scope = "row";
    name.append(node("small", group.device));
    row.append(name);
    for (const [caseKey] of cases) {
      const cell = row.insertCell();
      const timing = group.cells.get(caseKey);
      const value = valueOf(timing, metric);
      if (value === null) {
        cell.className = "missing";
        cell.textContent = "—";
        cell.title = timing ? "此记录没有 Host 耗时" : "此用例没有该算子 / 设备的记录";
        continue;
      }
      cell.textContent = format.format(value);
      const details = [
        "调用 " + integer.format(timing.count) + " 次",
        "累计 " + format.format(timing.elapsed) + " µs",
        "平均 " + format.format(timing.elapsed / timing.count) + " µs"
      ];
      if (timing.hostCount) details.push("Host 累计 " + format.format(timing.host) + " µs（" + timing.hostCount + " 次）");
      cell.title = details.join("\n");
      paint(cell, value, max, scale);
    }
  }
  box.append(table);
}

get("source").textContent = data.source + " · " + integer.format(data.valid) + " 条有效计时记录";
if (data.invalid) {
  get("warning").hidden = false;
  get("warning").textContent = "跳过 " + data.invalid + " 条格式异常的记录，首批异常行号 " + data.badLines.join("、") + "。";
}
options(get("phase"), unique(data.cells.map(cell => cell.phase)), "perf_forward");
updateKinds();
get("phase").addEventListener("change", () => { updateKinds(); render(); });
for (const id of ["kind", "metric", "scale", "limit"]) get(id).addEventListener("change", render);
for (const id of ["operator", "case"]) get(id).addEventListener("input", render);
render();
</script>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="input pytest log containing OP_TIME JSON lines")
    parser.add_argument("-o", "--output", type=Path, help="HTML output (default: input filename with .html suffix)")
    args = parser.parse_args()
    target = args.output or args.log.with_suffix(".html")
    if args.log.resolve() == target.resolve():
        parser.error("output must be a different file from the input log")
    try:
        data = read_timings(args.log)
        if not data["valid"]:
            parser.error(f"no valid OP_TIME records in {args.log} ({data['invalid']} invalid)")
        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        payload = payload.replace("<", r"\u003c")
        target.write_text(HTML.replace("__DATA__", payload), encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if data["invalid"]:
        print(f"Skipped {data['invalid']} malformed OP_TIME records.", file=sys.stderr)
    print(f"Wrote {target} ({data['valid']} OP_TIME records)")


if __name__ == "__main__":
    main()
