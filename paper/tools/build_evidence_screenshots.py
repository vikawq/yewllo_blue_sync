#!/usr/bin/env python3
"""Render cited PDF pages and add evidence screenshot cards to paper notes.

The script is intentionally deterministic and idempotent:

* source PDFs are never modified;
* one page image is reused by every citation of that physical PDF page;
* generated Markdown is isolated between marker comments;
* rerunning removes old inline links/cards before rebuilding them.

Usage:
    python build_evidence_screenshots.py --dry-run
    python build_evidence_screenshots.py --render
    python build_evidence_screenshots.py --inject
    python build_evidence_screenshots.py --all
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader


PAPER_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PAPER_ROOT.parents[1]
EVIDENCE_ROOT = PAPER_ROOT / "evidence_pages"

POPPLER = (
    Path.home()
    / ".cache"
    / "codex-runtimes"
    / "codex-primary-runtime"
    / "dependencies"
    / "native"
    / "poppler"
    / "Library"
    / "bin"
    / "pdftoppm.exe"
)

APPENDIX_BEGIN = "<!-- EVIDENCE_SCREENSHOTS:BEGIN -->"
APPENDIX_END = "<!-- EVIDENCE_SCREENSHOTS:END -->"
INLINE_RE = re.compile(
    r"\s*〔\[原文截图 E\d{3}\]\(#evidence-e\d{3}\)〕"
)

# Directly following PDF page locators used across the notes:
#   PDF p.5, PDF pp.5-7, PDF 5-7, PDF 第 5-7 页
PAGE_RE = re.compile(
    r"PDF\s*(?:第\s*)?(?:pp?\.\s*)?(\d{1,3})"
    r"(?:\s*[\-–—]\s*(\d{1,3}))?\s*(?:页)?",
    re.IGNORECASE,
)

SKIP_LOCATOR_TERMS = (
    "页码口径",
    "页码约定",
    "本文页码",
    "下文“PDF",
    "下文以",
    "PDF 共",
    "本地 PDF 共",
)


@dataclass(frozen=True)
class NoteSpec:
    note: str
    slug: str
    pdf: str | None
    source_kind: str = "pdf"  # pdf | tpds_extract | web_abstract


SPECS = [
    NoteSpec(
        "00_TPDS2024_Distributed_DNN_Performance_Modeling_Survey.md",
        "tpds2024-survey",
        "sources/TPDS2024_Distributed_DNN_Survey.pdf",
        "tpds_extract",
    ),
    NoteSpec(
        "route1_trace_replay/01_daydream.md",
        "daydream",
        "route1_trace_replay/sources/daydream.pdf",
    ),
    NoteSpec(
        "route1_trace_replay/02_dpro.md",
        "dpro",
        "route1_trace_replay/sources/dpro.pdf",
    ),
    NoteSpec(
        "route1_trace_replay/03_echo.md",
        "echo",
        "route1_trace_replay/sources/echo.pdf",
    ),
    NoteSpec(
        "route1_trace_replay/04_lumos.md",
        "lumos",
        "route1_trace_replay/sources/lumos.pdf",
    ),
    NoteSpec(
        "route2_profile_prediction/01_habitat.md",
        "habitat",
        "route2_profile_prediction/sources/habitat.pdf",
    ),
    NoteSpec(
        "route2_profile_prediction/02_vidur.md",
        "vidur",
        "route4_serving_simulation/sources/vidur.pdf",
    ),
    NoteSpec(
        "route2_profile_prediction/03_neusight_gpu_forecasting.md",
        "neusight",
        "route2_profile_prediction/sources/neusight.pdf",
    ),
    NoteSpec(
        "route2_profile_prediction/04_precision_aware_training_predictor.md",
        "precision-aware",
        "route2_profile_prediction/sources/precision-aware.pdf",
    ),
    NoteSpec(
        "route3_fullstack_training/01_simai.md",
        "simai",
        "route3_fullstack_training/sources/simai.pdf",
    ),
    NoteSpec(
        "route3_fullstack_training/02_astra_sim_2.md",
        "astra-sim-2",
        "route3_fullstack_training/sources/astra-sim2.pdf",
    ),
    NoteSpec(
        "route3_fullstack_training/03_proteus.md",
        "proteus",
        "route3_fullstack_training/sources/proteus.pdf",
    ),
    NoteSpec(
        "route3_fullstack_training/04_flexflow.md",
        "flexflow",
        "route3_fullstack_training/sources/flexflow.pdf",
    ),
    NoteSpec(
        "route3_fullstack_training/05_parallelsim.md",
        "parallelsim",
        "route3_fullstack_training/sources/parallelsim_official_page.pdf",
        "web_abstract",
    ),
    NoteSpec(
        "route3_fullstack_training/06_multiverse.md",
        "multiverse",
        "route3_fullstack_training/sources/multiverse.pdf",
    ),
    NoteSpec(
        "route4_serving_simulation/01_vidur_serving.md",
        "vidur",
        "route4_serving_simulation/sources/vidur.pdf",
    ),
    NoteSpec(
        "route4_serving_simulation/02_llmservingsim.md",
        "llmservingsim",
        "route4_serving_simulation/sources/llmservingsim.pdf",
    ),
    NoteSpec(
        "route4_serving_simulation/03_apex.md",
        "apex",
        "route4_serving_simulation/sources/apex.pdf",
    ),
    NoteSpec(
        "route4_serving_simulation/04_frontier.md",
        "frontier",
        "route4_serving_simulation/sources/frontier.pdf",
    ),
    NoteSpec(
        "route4_serving_simulation/05_charon.md",
        "charon",
        "route4_serving_simulation/sources/charon.pdf",
    ),
]


@dataclass
class Evidence:
    evidence_id: str
    line_number: int
    locator_line: str
    pages: list[int]
    page_label: str


def strip_generated(text: str) -> str:
    if APPENDIX_BEGIN in text:
        text = text.split(APPENDIX_BEGIN, 1)[0].rstrip() + "\n"
    return INLINE_RE.sub("", text)


def add_intro(lines: list[str], source_kind: str) -> list[str]:
    marker = "证据截图说明：正文中的"
    if any(marker in line for line in lines[:16]):
        return lines
    if source_kind == "tpds_extract":
        detail = (
            "TPDS 终版下载端点受站点限制；其卡片使用公开终版 PDF 的逐页文本抽取快照，"
            "保留期刊页码与抽取文本行号，但不冒充版式截图。"
        )
    elif source_kind == "web_abstract":
        detail = (
            "该论文全文受订阅限制；卡片只使用 Springer 官方摘要页快照，"
            "不能替代正文 PDF 证据。"
        )
    else:
        detail = "截图按 PDF 物理页码生成；原有章节、图表、算法和段落定位保持不变。"
    intro = (
        f"> 证据截图说明：正文中的 `原文截图 E###` 可跳转到文末证据卡片。{detail}"
    )
    insert_at = 1 if lines and lines[0].startswith("# ") else 0
    return lines[:insert_at] + ["", intro, ""] + lines[insert_at:]


def parse_pages(line: str) -> list[int]:
    pages: list[int] = []
    for match in PAGE_RE.finditer(line):
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if end < start:
            start, end = end, start
        # A citation spanning an implausibly large range is usually page-count
        # metadata rather than evidence. Keep the document compact and skip it.
        if end - start > 8:
            continue
        pages.extend(range(start, end + 1))
    return sorted(set(pages))


def collect_evidence(lines: list[str], source_kind: str) -> list[Evidence]:
    evidence: list[Evidence] = []
    in_fence = False
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or any(term in line for term in SKIP_LOCATOR_TERMS):
            continue
        pages = parse_pages(line)
        if source_kind == "web_abstract":
            is_fact = (
                "官方摘要" in line
                and any(term in line for term in ("原文事实", "可确认", "平均模拟误差", "IR"))
            )
            if not is_fact:
                continue
            pages = [1, 2]
            label = "Springer 官方摘要页（网页快照第 1-2 页，非论文 PDF 页）"
        elif not pages:
            continue
        else:
            label = "PDF p." + ", ".join(str(page) for page in pages)
        evidence.append(
            Evidence(
                evidence_id=f"E{len(evidence) + 1:03d}",
                line_number=idx,
                locator_line=stripped,
                pages=pages,
                page_label=label,
            )
        )
    return evidence


def insert_inline_link(line: str, evidence_id: str) -> str:
    link = f"〔[原文截图 {evidence_id}](#evidence-{evidence_id.lower()})〕"
    stripped = line.rstrip()
    if stripped.startswith("|") and stripped.endswith("|"):
        pos = line.rfind("|")
        return line[:pos].rstrip() + " " + link + " |"
    return line.rstrip() + " " + link


def font(size: int, mono: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/consola.ttf") if mono else Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def render_text_panels(slug: str, pages: set[int]) -> dict[int, list[Path]]:
    """Render TPDS PDF-extracted text as clearly labeled, line-numbered panels."""
    source = PAPER_ROOT / "sources" / "TPDS2024_Distributed_DNN_Survey_extracted.md"
    raw = source.read_text(encoding="utf-8")
    indices = {journal_page: raw.find(str(journal_page)) for journal_page in range(2463, 2479)}
    if any(index < 0 for index in indices.values()):
        raise RuntimeError("TPDS journal page markers 2463-2478 were not all found")

    out_dir = EVIDENCE_ROOT / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[int, list[Path]] = {}
    body_font = font(22, mono=True)
    header_font = font(30)
    small_font = font(18)
    width, height = 1600, 1900
    margin_x, top_y = 70, 145
    line_height = 31
    max_lines = (height - top_y - 70) // line_height

    for physical_page in sorted(pages):
        journal_page = 2462 + physical_page
        start = indices[journal_page]
        end = indices.get(journal_page + 1, len(raw))
        chunk = raw[start:end]
        chunk = re.sub(r"\s+", " ", chunk).strip()
        wrapped = textwrap.wrap(chunk, width=104, break_long_words=False, break_on_hyphens=False)
        panels: list[Path] = []
        for panel_index in range(0, len(wrapped), max_lines):
            panel_lines = wrapped[panel_index : panel_index + max_lines]
            suffix = chr(ord("a") + len(panels))
            target = out_dir / f"p{physical_page:03d}-{suffix}.png"
            image = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, width, 105), fill=(25, 55, 82))
            draw.text(
                (55, 28),
                f"TPDS 2024 original-PDF text extraction | physical p.{physical_page} | journal p.{journal_page}",
                fill="white",
                font=header_font,
            )
            draw.text(
                (55, 112),
                "Layout is not preserved. Line numbers below are snapshot extraction lines.",
                fill=(150, 35, 35),
                font=small_font,
            )
            y = top_y
            for offset, body in enumerate(panel_lines, start=panel_index + 1):
                draw.text((55, y), f"{offset:03d}", fill=(170, 45, 45), font=body_font)
                draw.text((125, y), body, fill=(25, 25, 25), font=body_font)
                y += line_height
            image.save(target, optimize=True)
            panels.append(target)
        result[physical_page] = panels
    return result


def render_pdf_pages(spec: NoteSpec, pages: set[int]) -> dict[int, list[Path]]:
    if spec.source_kind == "tpds_extract":
        return render_text_panels(spec.slug, pages)
    if not spec.pdf:
        raise RuntimeError(f"No source configured for {spec.note}")
    source = PAPER_ROOT / spec.pdf
    if not source.exists():
        raise FileNotFoundError(source)
    page_count = len(PdfReader(str(source)).pages)
    invalid = sorted(page for page in pages if page < 1 or page > page_count)
    if invalid:
        raise RuntimeError(f"{spec.note}: invalid physical PDF pages {invalid}; page_count={page_count}")
    out_dir = EVIDENCE_ROOT / spec.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[int, list[Path]] = {}
    for page in sorted(pages):
        target = out_dir / f"p{page:03d}.png"
        if not target.exists():
            prefix = target.with_suffix("")
            subprocess.run(
                [
                    str(POPPLER),
                    "-f",
                    str(page),
                    "-l",
                    str(page),
                    "-r",
                    "135",
                    "-png",
                    "-singlefile",
                    str(source),
                    str(prefix),
                ],
                check=True,
            )
        result[page] = [target]
    return result


def build_appendix(
    spec: NoteSpec,
    note_path: Path,
    evidence: list[Evidence],
    rendered: dict[int, list[Path]],
) -> str:
    sections = [
        APPENDIX_BEGIN,
        "",
        "## 原文证据截图附录",
        "",
        "正文中的 `原文截图 E###` 与本节一一对应。卡片保留原笔记行号和原有页码/章节定位；图片按 PDF 物理页生成。截图用于快速核读，正式引用仍以原论文为准。",
        "",
    ]
    if spec.source_kind == "tpds_extract":
        sections.extend(
            [
                "> **来源限制：** IEEE/ResearchGate 的终版 PDF 下载端点在当前环境被 418/403 拒绝。以下图片由公开终版 PDF 的逐页文本抽取生成，保留期刊页码和抽取行号，但不保持双栏版式；不应称为版式截图。",
                "",
            ]
        )
    elif spec.source_kind == "web_abstract":
        sections.extend(
            [
                "> **来源限制：** ParallelSim 合法全文受订阅限制。以下只展示 Springer 官方摘要页网页快照；正文方法、图表与算法仍不可核验。",
                "",
            ]
        )

    for item in evidence:
        sections.extend(
            [
                f'<a id="evidence-{item.evidence_id.lower()}"></a>',
                "",
                "<details>",
                f"<summary><strong>{item.evidence_id}</strong> - 原笔记第 {item.line_number} 行 - {html.escape(item.page_label)}</summary>",
                "",
                f"<p><strong>原定位：</strong> <code>{html.escape(item.locator_line)}</code></p>",
                "",
            ]
        )
        for page in item.pages:
            for image_path in rendered[page]:
                rel = image_path.relative_to(note_path.parent) if image_path.is_relative_to(note_path.parent) else None
                if rel is None:
                    rel_path = Path(
                        Path(image_path).relative_to(PAPER_ROOT)
                    )
                    # Notes are at most one directory below PAPER_ROOT.
                    prefix = "" if note_path.parent == PAPER_ROOT else "../"
                    markdown_path = prefix + rel_path.as_posix()
                else:
                    markdown_path = rel.as_posix()
                sections.append(
                    f"![{item.evidence_id} - {item.page_label}]({markdown_path})"
                )
                sections.append("")
        sections.extend(["</details>", ""])
    sections.extend([APPENDIX_END, ""])
    return "\n".join(sections)


def process_spec(spec: NoteSpec, render: bool, inject: bool) -> dict[str, object]:
    note_path = PAPER_ROOT / spec.note
    base_text = strip_generated(note_path.read_text(encoding="utf-8"))
    lines = add_intro(base_text.rstrip().splitlines(), spec.source_kind)
    evidence = collect_evidence(lines, spec.source_kind)
    pages = {page for item in evidence for page in item.pages}

    rendered: dict[int, list[Path]] = {}
    if render or inject:
        rendered = render_pdf_pages(spec, pages)

    if inject:
        links_by_line = {item.line_number: item.evidence_id for item in evidence}
        output_lines = [
            insert_inline_link(line, links_by_line[index]) if index in links_by_line else line
            for index, line in enumerate(lines, start=1)
        ]
        appendix = build_appendix(spec, note_path, evidence, rendered)
        note_path.write_text("\n".join(output_lines).rstrip() + "\n\n" + appendix, encoding="utf-8")

    return {
        "note": spec.note,
        "source_kind": spec.source_kind,
        "evidence_cards": len(evidence),
        "pages": sorted(pages),
        "page_count": len(pages),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--render", action="store_true")
    mode.add_argument("--inject", action="store_true")
    mode.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not POPPLER.exists() and not args.dry_run:
        raise FileNotFoundError(f"pdftoppm not found: {POPPLER}")

    results = []
    for spec in SPECS:
        results.append(
            process_spec(
                spec,
                render=args.render or args.all,
                inject=args.inject or args.all,
            )
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
