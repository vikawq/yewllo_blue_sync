#!/usr/bin/env python3
"""Build a self-contained HTML timing chart from the OP_TIME records in a pytest log."""

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
<title>算子耗时分析</title>
<style>
:root { color-scheme: dark; --page: #101620; --panel: #172130; --ink: #e7eef7; --muted: #aebed1; --border: #344458; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, sans-serif; }
main { padding: clamp(16px, 4vw, 44px); }
h1 { font-size: clamp(26px, 5vw, 38px); margin: 0 0 4px; }
h2 { font-size: 18px; margin: 0 0 4px; }
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
[hidden] { display: none !important; }
.gradient { width: 140px; height: 12px; border-radius: 3px; background: linear-gradient(90deg, #212d3d, #873917, #ce7b23, #ffc775); }
.note { color: #ffd18f; }
.bar-group { margin: 14px 0; padding: 14px; border: 1px solid var(--border); border-radius: 8px; overflow-wrap: anywhere; }
.bar-row { margin-top: 12px; }
.bar-label { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 2px 12px; }
.bar-label > * { min-width: 0; max-width: 100%; }
.bar-value { color: var(--muted); font-variant-numeric: tabular-nums; }
.bar-track { width: 100%; height: 12px; background: var(--panel); border-radius: 4px; overflow: hidden; margin-top: 4px; }
.bar-fill { height: 100%; max-width: 100%; background: #eaa34e; }
</style>
<main>
  <h1 id="heading">算子耗时分析</h1>
  <p class="meta" id="source"></p>
  <p class="note" id="warning" hidden></p>
  <div class="controls">
    <label>图形<select id="view"><option value="heatmap">热力图</option><option value="bars">横向条形图</option></select></label>
    <label>阶段<select id="phase"></select></label>
    <label>记录类型<select id="kind"></select></label>
    <label>指标<select id="metric">
      <option value="elapsed">累计耗时 (µs)</option>
      <option value="share" id="share-option">累计耗时占比 (%)</option>
      <option value="mean">平均单次耗时 (µs)</option>
      <option value="count">调用次数</option>
      <option value="host">Host 调用累计 (µs)</option>
    </select></label>
    <label id="scale-control">色阶<select id="scale"><option value="log">对数</option><option value="linear">线性</option></select></label>
    <label>显示行数<select id="limit"><option>30</option><option>100</option><option value="all">全部</option></select></label>
    <label>搜索算子 / 设备<input type="search" id="operator"></label>
    <label>搜索脚本 / 用例<input type="search" id="case"></label>
  </div>
  <p id="selection" aria-live="polite"></p>
  <div class="legend" id="legend"><span>0</span><span class="gradient"></span><span id="max"></span><span>· — 表示该用例没有这项数据</span></div>
  <p class="hint" id="hint"></p>
  <div class="scroll" id="matrix" tabindex="0" aria-label="算子耗时矩阵"></div>
  <div id="bars" aria-label="按用例分组的算子耗时条形图" hidden></div>
  <noscript>请启用 JavaScript 查看算子耗时图。</noscript>
</main>
<script type="application/json" id="data">__DATA__</script>
<script>
"use strict";
const get = id => document.getElementById(id);
const data = JSON.parse(get("data").textContent);
const format = new Intl.NumberFormat("zh-CN", {maximumFractionDigits: 3});
const integer = new Intl.NumberFormat("zh-CN", {maximumFractionDigits: 0});
const percent = new Intl.NumberFormat("zh-CN", {style: "percent", maximumFractionDigits: 2});
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

function shareOf(timing, total) {
  return total > 0 ? timing.elapsed / total : null;
}

function valueOf(timing, metric, total) {
  if (!timing) return null;
  if (metric === "share") return shareOf(timing, total);
  if (metric === "count") return timing.count;
  if (metric === "host") return timing.hostCount ? timing.host : null;
  if (metric === "mean") return timing.elapsed / timing.count;
  return timing.elapsed;
}

function formatValue(value, metric, units = false) {
  if (value === null) return "—";
  if (metric === "share") return percent.format(value);
  const unit = metric === "count" ? " 次" : " µs";
  return format.format(value) + (units ? unit : "");
}

function timingDetails(timing, total) {
  const details = [
    "调用 " + integer.format(timing.count) + " 次",
    "累计 " + format.format(timing.elapsed) + " µs",
    "平均 " + format.format(timing.elapsed / timing.count) + " µs"
  ];
  if (total !== null) details.push("耗时占比 " + formatValue(shareOf(timing, total), "share", true));
  if (timing.hostCount) details.push("Host 累计 " + format.format(timing.host) + " µs（" + timing.hostCount + " 次）");
  return details.join("\n");
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

function collectCases(phase, kind, caseFilter) {
  const cases = new Map();
  for (const cell of data.cells) {
    if (cell.phase !== phase || cell.kind !== kind) continue;
    if (!(cell.script + " " + cell.test + " " + cell.case).toLowerCase().includes(caseFilter)) continue;
    const key = JSON.stringify([cell.script, cell.test, cell.case]);
    if (!cases.has(key)) cases.set(key, {key, info: cell, cells: [], total: 0});
    const group = cases.get(key);
    group.cells.push(cell);
    group.total += cell.elapsed;
  }
  return [...cases.values()].sort((a, b) => a.key.localeCompare(b.key, "en", {numeric: true}));
}

function matchesOperator(cell, filter) {
  return (cell.op + " " + cell.device).toLowerCase().includes(filter);
}

function compareOperators(a, b) {
  return a.op.localeCompare(b.op) || a.device.localeCompare(b.device);
}

function takeRows(rows, limit) {
  return limit === "all" ? rows : rows.slice(0, Number(limit));
}

function renderHeatmap(cases, kind, metric, limit, opFilter) {
  const groups = new Map();
  for (const {key, cells, total} of cases) {
    for (const cell of cells) {
      if (!matchesOperator(cell, opFilter)) continue;
      const rowKey = JSON.stringify([cell.op, cell.device]);
      if (!groups.has(rowKey)) {
        groups.set(rowKey, {op: cell.op, device: cell.device, cells: new Map(),
          count: 0, elapsed: 0, host: 0, hostCount: 0, peakShare: -1});
      }
      const group = groups.get(rowKey);
      group.cells.set(key, cell);
      group.peakShare = Math.max(group.peakShare, shareOf(cell, total) ?? -1);
      for (const field of ["count", "elapsed", "host", "hostCount"]) group[field] += cell[field];
    }
  }
  const rankOf = group => metric === "share" ? group.peakShare : valueOf(group, metric) ?? -1;
  const ranked = [...groups.values()].sort((a, b) => rankOf(b) - rankOf(a) || compareOperators(a, b));
  const shown = takeRows(ranked, limit);
  const box = get("matrix");
  box.hidden = !shown.length;
  get("legend").hidden = !shown.length;
  const ranking = metric === "share" ? "按各算子的最高用例占比排序" : "按当前指标排序";
  get("selection").textContent = shown.length
    ? "显示 " + shown.length + " / " + ranked.length + " 行 · " + cases.length + " 个用例 · " + ranking
    : "没有匹配的记录。";
  if (!shown.length) return;

  let max = 0;
  for (const group of shown) {
    for (const {key, total} of cases) {
      max = Math.max(max, valueOf(group.cells.get(key), metric, total) ?? 0);
    }
  }
  get("max").textContent = formatValue(max, metric, true);
  const scale = get("scale").value;
  const table = node("table");
  const header = table.createTHead().insertRow();
  header.append(node("th", "算子 / 设备"));
  for (const {info: value} of cases) {
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
    for (const {key, total} of cases) {
      const cell = row.insertCell();
      const timing = group.cells.get(key);
      const value = valueOf(timing, metric, total);
      if (value === null) {
        cell.className = "missing";
        cell.textContent = "—";
        cell.title = timing ? timingDetails(timing, kind === stageKind ? null : total) : "此用例没有该算子 / 设备的记录";
        continue;
      }
      cell.textContent = formatValue(value, metric);
      cell.title = timingDetails(timing, kind === stageKind ? null : total);
      paint(cell, value, max, scale);
    }
  }
  box.append(table);
}

function renderBars(cases, kind, metric, limit, opFilter) {
  const container = get("bars");
  let visibleCases = 0;
  for (const {info, cells, total} of cases) {
    const value = timing => valueOf(timing, metric, total);
    const ranked = cells.filter(cell => matchesOperator(cell, opFilter))
      .sort((a, b) => (value(b) ?? -1) - (value(a) ?? -1) || compareOperators(a, b));
    if (!ranked.length) continue;
    visibleCases++;
    const max = cells.reduce((largest, cell) => Math.max(largest, value(cell) ?? 0), 0);
    const group = node("section");
    group.className = "bar-group";
    group.append(node("h2", info.case), node("div", info.script + " · " + info.test));
    const shown = takeRows(ranked, limit);
    const summary = node("p", "显示 " + shown.length + " / " + ranked.length + " 行 · 组内最大值 " + formatValue(max, metric, true));
    summary.className = "meta";
    group.append(summary);
    for (const timing of shown) {
      const measured = value(timing);
      const row = node("div");
      row.className = "bar-row";
      row.title = timingDetails(timing, kind === stageKind ? null : total);
      const label = node("div");
      label.className = "bar-label";
      let description = formatValue(measured, metric, true);
      if (kind !== stageKind) {
        description += metric === "share"
          ? " · 累计 " + formatValue(timing.elapsed, "elapsed", true)
          : " · 耗时占比 " + formatValue(shareOf(timing, total), "share", true);
      }
      const detail = node("span", description);
      detail.className = "bar-value";
      label.append(node("span", timing.op + " · " + timing.device), detail);
      const track = node("div");
      track.className = "bar-track";
      const fill = node("div");
      fill.className = "bar-fill";
      const fraction = max > 0 && measured !== null ? measured / max : 0;
      fill.style.width = 100 * Math.min(1, fraction) + "%";
      track.append(fill);
      row.append(label, track);
      group.append(row);
    }
    container.append(group);
  }
  container.hidden = !visibleCases;
  get("selection").textContent = visibleCases ? "显示 " + visibleCases + " 个用例 · 每个用例独立缩放" : "没有匹配的记录。";
}

function render() {
  const bars = get("view").value === "bars";
  const kind = get("kind").value;
  const metricControl = get("metric");
  get("share-option").disabled = kind === stageKind;
  if (kind === stageKind && metricControl.value === "share") metricControl.value = "elapsed";
  const cases = collectCases(get("phase").value, kind, get("case").value.trim().toLowerCase());
  const opFilter = get("operator").value.trim().toLowerCase();
  const limit = get("limit").value;
  get("heading").textContent = bars ? "算子耗时条形图" : "算子耗时热力图";
  get("scale-control").hidden = bars;
  for (const id of ["matrix", "bars"]) {
    get(id).replaceChildren();
    get(id).hidden = true;
  }
  get("legend").hidden = true;
  const scaleHint = bars
    ? "条长按该用例当前指标的最大值线性缩放。组内最大值对应满宽，跨用例请比较标注数值。"
    : "颜色按当前视图中最大的单元格值缩放。";
  const shareHint = kind === stageKind
    ? "整段耗时包含打点开销，单独查看。"
    : "占比 = 算子累计耗时 ÷ 同一用例、阶段及记录类型下所有算子的累计耗时。搜索和行数限制不改变分母。";
  get("hint").textContent = scaleHint + shareHint;
  if (bars) renderBars(cases, kind, metricControl.value, limit, opFilter);
  else renderHeatmap(cases, kind, metricControl.value, limit, opFilter);
}

get("source").textContent = data.source + " · " + integer.format(data.valid) + " 条有效计时记录";
if (data.invalid) {
  get("warning").hidden = false;
  get("warning").textContent = "跳过 " + data.invalid + " 条格式异常的记录，首批异常行号 " + data.badLines.join("、") + "。";
}
options(get("phase"), unique(data.cells.map(cell => cell.phase)), "perf_forward");
get("view").value = data.initialView ?? "heatmap";
updateKinds();
get("phase").addEventListener("change", () => { updateKinds(); render(); });
for (const id of ["view", "kind", "metric", "scale", "limit"]) get(id).addEventListener("change", render);
for (const id of ["operator", "case"]) get(id).addEventListener("input", render);
render();
</script>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="input pytest log containing OP_TIME JSON lines")
    parser.add_argument("--view", choices=("heatmap", "bars"), default="heatmap", help="initial chart view")
    parser.add_argument("-o", "--output", type=Path, help="HTML output path")
    args = parser.parse_args()
    default_name = args.log.stem + ("_bars.html" if args.view == "bars" else ".html")
    target = args.output or args.log.with_name(default_name)
    if args.log.resolve() == target.resolve():
        parser.error("output must be a different file from the input log")
    try:
        data = read_timings(args.log)
        if not data["valid"]:
            parser.error(f"no valid OP_TIME records in {args.log} ({data['invalid']} invalid)")
        data["initialView"] = args.view
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
