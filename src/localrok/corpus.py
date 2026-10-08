"""Stage 6: CORPUS.md generation. Fail loudly if over the token budget."""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from localrok import __version__
from localrok.config import PromptConfig

_SECTION_NUM_RE = re.compile(r"^[\dIVX]+(?:[.\-][\dA-Za-z]+)*[.)]?\s+", re.IGNORECASE)
_TRAILING_RE = re.compile(r"[\s·|ǀ:.,\-—–]+$")


class CorpusBudgetError(RuntimeError):
    pass


def _estimate_tokens(text: str, chars_per_token: int) -> int:
    return max(1, (len(text) + chars_per_token - 1) // chars_per_token)


def _normalize_heading(heading: str) -> str:
    return _SECTION_NUM_RE.sub("", _TRAILING_RE.sub("", heading.strip())).strip().lower()


def _boilerplate_headings(per_doc: list[list[str]]) -> set[str]:
    """Headings shared by more than half the corpus.

    Approval, Change Log, Reference Documents, Scope and friends are identical
    in every document, so they consume the map's preview budget while telling
    the agent nothing about which document to open. Derived from the corpus
    instead of a keyword list, so it holds for any language or house style.
    """
    if len(per_doc) < 4:
        return set()
    counts = Counter(h for headings in per_doc for h in {_normalize_heading(x) for x in headings})
    threshold = max(3, len(per_doc) // 2)
    return {h for h, n in counts.items() if h and n >= threshold}


def _informative_headings(
    headings: list[str], boilerplate: set[str], doc_ident: str | None
) -> list[str]:
    """Drop boilerplate, repeats, and the document restating its own identifier."""
    ident = (doc_ident or "").lower()
    seen: set[str] = set()
    kept: list[str] = []
    for heading in headings:
        key = _normalize_heading(heading)
        if not key or key in boilerplate or key in seen:
            continue
        if ident and ident in key:
            continue
        seen.add(key)
        kept.append(heading.strip())
    return kept


def _heading_list(headings: list[str], limit: int = 400) -> str:
    joined = " · ".join(headings)
    if len(joined) <= limit:
        return joined
    return joined[: limit - 1] + "…"


def _doc_block(
    filename: str,
    title: str | None,
    doc_type: str | None,
    version: str | None,
    page_count: int | None,
    score: float | None,
    headings: list[str] | None,
) -> str:
    meta_bits = [title or filename]
    if doc_type and doc_type != "unknown":
        meta_bits.append(doc_type)
    if version:
        ver = str(version)
        meta_bits.append(ver if ver.lower().startswith("v") else f"v{ver}")
    if page_count:
        meta_bits.append(f"{page_count} pages")
    if score is not None:
        meta_bits.append(f"structure {score:.2f}")
    warn = ""
    if score is not None and score < 0.5:
        warn = "  ⚠ low structure"
    lines = [
        f"## {filename}",
        " · ".join(str(b) for b in meta_bits if b) + warn,
    ]
    if headings:
        lines.append(f"  {_heading_list(headings)}")
    lines.append("")
    return "\n".join(lines)


def generate_corpus_markdown(
    conn: sqlite3.Connection,
    prompt: PromptConfig,
    docling_ver: str | None = None,
) -> str:
    docs = conn.execute("SELECT * FROM documents ORDER BY filename").fetchall()
    unit_count = conn.execute("SELECT COUNT(*) AS n FROM units").fetchone()["n"]
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    header = (
        f"# Corpus — {len(docs)} documents, {unit_count:,} units\n"
        f"Generated {generated} by LocalRök {__version__}"
        + (f" / docling {docling_ver}" if docling_ver else "")
        + "\n\n"
    )

    entries: list[dict] = []
    for doc in docs:
        headings = [
            r["heading"]
            for r in conn.execute(
                """
                SELECT heading FROM sections
                WHERE doc_id = ? AND depth = 1 AND heading IS NOT NULL
                ORDER BY ordinal
                """,
                (doc["doc_id"],),
            )
        ]
        entries.append(
            {
                "filename": doc["filename"],
                "title": doc["title"],
                "doc_type": doc["doc_type"],
                "version": doc["version"],
                "page_count": doc["page_count"],
                "score": doc["structure_score"],
                "headings": headings,
                "doc_ident": doc["doc_ident"],
            }
        )

    boilerplate = _boilerplate_headings([e["headings"] for e in entries])
    for e in entries:
        e["headings"] = _informative_headings(e["headings"], boilerplate, e["doc_ident"])

    def render(heading_map: dict[str, list[str] | None]) -> str:
        parts = [header]
        for e in entries:
            parts.append(
                _doc_block(
                    e["filename"],
                    e["title"],
                    e["doc_type"],
                    e["version"],
                    e["page_count"],
                    e["score"],
                    heading_map[e["filename"]],
                )
            )
        return "".join(parts)

    heading_map: dict[str, list[str] | None] = {e["filename"]: e["headings"] for e in entries}
    text = render(heading_map)
    budget = prompt.corpus_map_max_tokens
    cpt = prompt.corpus_chars_per_token

    if _estimate_tokens(text, cpt) <= budget:
        return text

    # Shorten heading lists, longest first. Never drop a document.
    order = sorted(entries, key=lambda e: len(_heading_list(e["headings"] or [])), reverse=True)
    for e in order:
        heading_map[e["filename"]] = None
        text = render(heading_map)
        if _estimate_tokens(text, cpt) <= budget:
            return text

    raise CorpusBudgetError(
        f"CORPUS.md is {_estimate_tokens(text, cpt)} tokens "
        f"(budget {budget}) even with all heading lists removed. "
        "Split the corpus or raise prompt.corpus_map_max_tokens."
    )


def write_corpus(
    conn: sqlite3.Connection,
    dest: Path,
    prompt: PromptConfig,
    docling_ver: str | None = None,
) -> str:
    text = generate_corpus_markdown(conn, prompt, docling_ver)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return text
