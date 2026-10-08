"""Parse LocalRök tool output and match it against a question's gold locations.

Shared by eval/retrieval.py (search only) and eval/run.py (full agent runs),
so both score "did the agent get to the right section" the same way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HIT_RE = re.compile(r"^\[\d+\] (.+)$")
_PAGE_RE = re.compile(r"^p(\d+)$")
_PAGE_RANGE_RE = re.compile(r"^p(\d+)(?:-(\d+))?$")
_COMPLETENESS = {"fragment", "complete_section"}


@dataclass(frozen=True)
class Hit:
    filename: str
    section: str
    page: int | None = None
    # Set only for page-range reads ("file.pdf > p12-14").
    page_end: int | None = None


def parse_hits(text: str) -> list[Hit]:
    """Result headers of a docs_search payload, in rank order.

    Header shape: `[i] filename | section path | pN | completeness`. Section
    paths may themselves contain " | ", so the fixed fields are peeled off
    both ends.
    """
    hits: list[Hit] = []
    for line in text.splitlines():
        m = _HIT_RE.match(line)
        if not m:
            continue
        parts = m.group(1).split(" | ")
        if len(parts) < 3 or parts[-1] not in _COMPLETENESS:
            continue
        filename, rest = parts[0], parts[1:-1]
        page = None
        if rest and _PAGE_RE.match(rest[-1]):
            page = int(rest[-1][1:])
            rest = rest[:-1]
        hits.append(Hit(filename, " | ".join(rest), page))
    return hits


def parse_read_labels(text: str) -> list[Hit]:
    """Targets a docs_read payload actually returned text for.

    Each returned block starts with its label line followed by a `## ` heading;
    error and budget blocks do not, so they are skipped.
    """
    hits: list[Hit] = []
    for block in text.split("\n---\n"):
        lines = block.strip("\n").splitlines()
        if len(lines) < 2:
            continue
        if not (lines[1].startswith("## ") or lines[1] == "(no text in range)"):
            continue
        filename, sep, rest = lines[0].partition(" > ")
        if not sep:
            hits.append(Hit(filename, ""))
            continue
        m = _PAGE_RANGE_RE.match(rest.strip())
        if m:
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else start
            hits.append(Hit(filename, "", start, end))
        else:
            hits.append(Hit(filename, rest.strip()))
    return hits


def matches_gold(hit: Hit, gold: dict) -> bool:
    if gold["doc"].lower() not in hit.filename.lower():
        return False
    page = gold.get("page")
    if hit.page_end is not None:
        return page is not None and hit.page <= page <= hit.page_end
    section = gold.get("section")
    if section and section.lower() not in hit.section.lower():
        return False
    if page is not None and hit.page is not None and hit.page != page:
        return False
    return True


def first_gold_rank(hits: list[Hit], golds: list[dict]) -> int | None:
    """1-based rank of the first hit matching any gold location."""
    for i, hit in enumerate(hits, start=1):
        if any(matches_gold(hit, g) for g in golds):
            return i
    return None
