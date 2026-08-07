#!/usr/bin/env python3
"""Audit generated evidence links, cards, and screenshot paths."""

from __future__ import annotations

import json
import html
import re
import sys
from pathlib import Path


PAPER_ROOT = Path(__file__).resolve().parents[1]
BEGIN = "<!-- EVIDENCE_SCREENSHOTS:BEGIN -->"
END = "<!-- EVIDENCE_SCREENSHOTS:END -->"
ANCHOR_RE = re.compile(r'<a id="evidence-(e\d{3})"></a>')
LINK_RE = re.compile(r'\[原文截图 (E\d{3})\]\(#evidence-(e\d{3})\)')
IMAGE_RE = re.compile(r'!\[[^]]*\]\(([^)]+\.png)\)')
PAGE_LINK_RE = re.compile(r'href="#source-page-(p\d{3})"')
PAGE_ANCHOR_RE = re.compile(r'<a id="source-page-(p\d{3})"></a>')
DETAIL_RE = re.compile(
    r'<summary><strong>(E\d{3})</strong> - 原笔记第 (\d+) 行 - .*?</summary>\s*'
    r'<p><strong>原定位：</strong> <code>(.*?)</code></p>',
    re.DOTALL,
)
INLINE_RE = re.compile(r"\s*〔\[原文截图 E\d{3}\]\(#evidence-e\d{3}\)〕")


def main() -> None:
    notes = sorted(
        path
        for path in PAPER_ROOT.rglob("*.md")
        if BEGIN in path.read_text(encoding="utf-8")
    )
    report: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    referenced_files: set[Path] = set()

    for path in notes:
        text = path.read_text(encoding="utf-8")
        body, appendix = text.split(BEGIN, 1)
        anchors = ANCHOR_RE.findall(appendix)
        link_pairs = LINK_RE.findall(body)
        links = [label.lower() for label, target in link_pairs]
        images = IMAGE_RE.findall(appendix)
        page_links = PAGE_LINK_RE.findall(appendix)
        page_anchors = PAGE_ANCHOR_RE.findall(appendix)
        duplicate_image_embeds = sorted(
            image for image in set(images) if images.count(image) > 1
        )
        line_mismatches: list[dict[str, object]] = []
        body_lines = body.rstrip().splitlines()
        for evidence_id, line_number_text, locator_html in DETAIL_RE.findall(appendix):
            line_number = int(line_number_text)
            if line_number < 1 or line_number > len(body_lines):
                line_mismatches.append(
                    {"evidence_id": evidence_id, "line_number": line_number, "reason": "out_of_range"}
                )
                continue
            actual_line = body_lines[line_number - 1].strip()
            actual_locator = INLINE_RE.sub("", actual_line).strip()
            expected_locator = html.unescape(locator_html).strip()
            if (
                f"原文截图 {evidence_id}" not in actual_line
                or actual_locator != expected_locator
            ):
                line_mismatches.append(
                    {
                        "evidence_id": evidence_id,
                        "line_number": line_number,
                        "expected": expected_locator,
                        "actual": actual_locator,
                    }
                )
        missing = [
            image
            for image in images
            if not (path.parent / image).resolve().exists()
        ]
        mismatched_links = [
            (label, target)
            for label, target in link_pairs
            if label.lower() != target
        ]
        for image in images:
            referenced_files.add((path.parent / image).resolve())

        if (
            anchors != links
            or missing
            or mismatched_links
            or line_mismatches
            or set(page_links) != set(page_anchors)
            or len(page_anchors) != len(set(page_anchors))
            or duplicate_image_embeds
            or text.count(BEGIN) != 1
            or text.count(END) != 1
        ):
            errors.append(
                {
                    "note": str(path.relative_to(PAPER_ROOT)),
                    "anchors": len(anchors),
                    "links": len(links),
                    "missing_images": missing,
                    "mismatched_links": mismatched_links,
                    "line_mismatches": line_mismatches,
                    "page_links_without_matching_anchor": sorted(
                        set(page_links) - set(page_anchors)
                    ),
                    "page_anchors_without_link": sorted(
                        set(page_anchors) - set(page_links)
                    ),
                    "duplicate_image_embeds": duplicate_image_embeds,
                }
            )

        report.append(
            {
                "note": str(path.relative_to(PAPER_ROOT)),
                "cards": len(anchors),
                "image_embeds": len(images),
                "unique_images": len(set(images)),
                "duplicate_image_embeds": len(images) - len(set(images)),
            }
        )

    png_files = set((PAPER_ROOT / "evidence_pages").rglob("*.png"))
    unreferenced = sorted(str(path.relative_to(PAPER_ROOT)) for path in png_files - referenced_files)
    summary = {
        "notes": len(notes),
        "cards": sum(int(item["cards"]) for item in report),
        "image_embeds": sum(int(item["image_embeds"]) for item in report),
        "unique_referenced_files": len(referenced_files),
        "unreferenced_images": unreferenced,
        "errors": errors,
        "per_note": report,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors or unreferenced:
        sys.exit(1)


if __name__ == "__main__":
    main()
