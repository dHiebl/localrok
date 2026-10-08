"""Stage 3: derived Markdown view (humans + E1 grep baseline). Not indexed."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from localrok.models import SectionDraft

_ANNOTATION_TAG_RE = re.compile(r'<!--<annotation kind="[^"]*">-->|<!--<annotation/>-->')


def strip_repeated_lines(md: str, min_occurrences: int = 2) -> str:
    """Remove lines that appear multiple times verbatim (page headers/footers)."""
    lines = md.split("\n")
    counts: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if stripped:
            counts[stripped] = counts.get(stripped, 0) + 1
    repeated = {text for text, n in counts.items() if n >= min_occurrences and len(text) < 300}
    if repeated:
        lines = [line for line in lines if line.strip() not in repeated]
    return "\n".join(lines)


def collapse_blank_lines(md: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", md)


def clean_markdown(md: str) -> str:
    md = _ANNOTATION_TAG_RE.sub("", md)
    md = strip_repeated_lines(md)
    md = collapse_blank_lines(md)
    return md


def _page_transitions(sections: list[SectionDraft]) -> list[int]:
    """Ordered unique pages encountered in unit order."""
    pages: list[int] = []
    last: int | None = None
    for section in sections:
        for unit in section.units:
            if unit.page is None:
                continue
            if unit.page != last:
                pages.append(unit.page)
                last = unit.page
    return pages


def insert_page_anchors(md: str, pages: list[int]) -> str:
    """Insert HTML page comments at heading boundaries when we have page data.

    Docling markdown has no structural page markers. We prefix the file with
    the page sequence we actually indexed so grep-baseline users can still
    recover provenance, and insert a comment before the first heading.
    """
    if not pages:
        return md
    first = pages[0]
    comment = f"<!-- page: {first} -->\n"
    if md.lstrip().startswith("#"):
        return comment + md
    return comment + md


def emit_markdown(
    doc: Any,
    dest: Path,
    sections: list[SectionDraft] | None = None,
) -> str:
    raw = doc.export_to_markdown()
    md = clean_markdown(raw)
    if sections:
        md = insert_page_anchors(md, _page_transitions(sections))
        # Also stamp page comments at each markdown heading we can align by text
        md = _stamp_heading_pages(md, sections)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(md, encoding="utf-8")
    return md


def _stamp_heading_pages(md: str, sections: list[SectionDraft]) -> str:
    lines = md.split("\n")
    used: set[int] = set()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        stamped = False
        if stripped.startswith("#"):
            heading_text = stripped.lstrip("#").strip()
            for i, section in enumerate(sections):
                if i in used:
                    continue
                if section.heading and section.heading.strip() == heading_text:
                    page = section.page_start
                    if page is not None:
                        out.append(f"<!-- page: {page} -->")
                    used.add(i)
                    stamped = True
                    break
        out.append(line)
        if stamped:
            pass
    return "\n".join(out)
