"""BM25 search with evidence windows (build_spec §8)."""

from __future__ import annotations

import re
import sqlite3
from collections import defaultdict

from localrok.config import SearchConfig
from localrok.db import resolve_document, unknown_doc_message
from localrok.figures import figure_url

_FTS_OPS = {"AND", "OR", "NOT", "NEAR"}
_IDENT_RE = re.compile(r"^[A-Z]{1,4}[0-9]{1,4}$")
_STRIP_RE = re.compile(r'["\'()*]')


def is_identifier(token: str) -> bool:
    if _IDENT_RE.match(token):
        return True
    return any(c.isdigit() for c in token) and any(c.isalpha() for c in token)


def fts_quote(s: str) -> str:
    return '"' + s.replace('"', '""') + '"'


def parse_query(query: str, *, exact: bool) -> tuple[str, list[str]]:
    """Return (MATCH expression, display terms)."""
    query = query.strip()
    if not query:
        return "", []
    if exact:
        return fts_quote(query), [query]

    raw_tokens = query.split()
    terms: list[str] = []
    match_parts: list[str] = []
    for tok in raw_tokens:
        cleaned = _STRIP_RE.sub("", tok).strip()
        if not cleaned or cleaned.upper() in _FTS_OPS:
            continue
        terms.append(cleaned)
        if is_identifier(cleaned) or any(c in cleaned for c in "-./:"):
            match_parts.append(fts_quote(cleaned))
        elif len(cleaned) >= 4:
            match_parts.append(f"{cleaned}*")
        else:
            match_parts.append(cleaned)

    if not match_parts:
        return fts_quote(query), [query]
    return " OR ".join(match_parts), terms


def _window(text: str, max_chars: int, terms: list[str]) -> tuple[str, bool]:
    """Return (window, truncated). `truncated` drives the completeness flag."""
    if len(text) <= max_chars:
        return text, False
    lower = text.lower()
    mid = 0
    for t in terms:
        needle = t.rstrip("*").lower()
        if not needle:
            continue
        i = lower.find(needle)
        if i >= 0:
            mid = i
            break
    start = max(0, mid - max_chars // 2)
    end = min(len(text), start + max_chars)
    start = max(0, end - max_chars)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return prefix + text[start:end] + suffix, bool(prefix or suffix)


def _indent_block(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line if line else prefix.rstrip() for line in text.splitlines())


def search(
    conn: sqlite3.Connection,
    query: str,
    cfg: SearchConfig,
    *,
    doc: str | None = None,
    section: str | None = None,
    exact: bool = False,
    limit: int | None = None,
) -> str:
    limit = cfg.default_limit if limit is None else limit
    limit = max(1, min(cfg.max_limit, int(limit)))

    doc_id = None
    if doc:
        row = resolve_document(conn, doc)
        if row is None:
            return unknown_doc_message(conn, doc)
        doc_id = row["doc_id"]

    match_expr, terms = parse_query(query, exact=exact)
    if not match_expr:
        return "Empty query. Pass search terms, e.g. an identifier like J5."

    fetch = max(limit * 5, 40)
    # Tables of contents and similar navigation aids match any heading-shaped
    # query and cite back to the front matter. Readable, but never ranked.
    filters = ["u.kind <> 'navigation'"]
    args: list = [match_expr]
    if doc_id is not None:
        filters.append("u.doc_id = ?")
        args.append(doc_id)
    if section:
        filters.append("s.section_path LIKE ?")
        args.append(f"{section}%")
    where = " AND ".join(["units_fts MATCH ?"] + filters)

    try:
        unit_rows = conn.execute(
            f"""
            SELECT u.unit_id, u.section_id, u.doc_id, u.text, u.page, u.kind,
                   s.section_path, s.unit_count, d.filename
            FROM units_fts
            JOIN units u ON u.unit_id = units_fts.rowid
            JOIN sections s ON s.section_id = u.section_id
            JOIN documents d ON d.doc_id = u.doc_id
            WHERE {where}
            ORDER BY bm25(units_fts, 1.0, ?)
            LIMIT ?
            """,
            [*args, cfg.heading_weight, fetch],
        ).fetchall()
    except sqlite3.OperationalError as exc:
        return (
            f"Search failed for query {query!r} (parsed: {match_expr}): {exc}. "
            "Try simpler terms, or exact=true for a phrase."
        )

    total_matches = conn.execute(
        f"""
        SELECT COUNT(*) AS n
        FROM units_fts
        JOIN units u ON u.unit_id = units_fts.rowid
        JOIN sections s ON s.section_id = u.section_id
        WHERE {where}
        """,
        args,
    ).fetchone()["n"]

    per_section: dict[int, int] = defaultdict(int)
    picked: list[sqlite3.Row] = []
    for row in unit_rows:
        n = per_section[row["section_id"]]
        if n >= cfg.units_per_section_cap:
            continue
        per_section[row["section_id"]] += 1
        picked.append(row)
        if len(picked) >= limit:
            break

    if not picked:
        return (
            f"No matches for query: {query!r}\n"
            f"Parsed terms: {match_expr}\n"
            "Try domain synonyms, the other language (DE/EN), or docs_outline "
            "on a candidate document from the corpus map."
        )

    figure_map: dict[int, list[sqlite3.Row]] = defaultdict(list)
    section_ids = {r["section_id"] for r in picked}
    placeholders = ",".join("?" * len(section_ids))
    if section_ids:
        for fig in conn.execute(
            f"""
            SELECT section_id, caption, path FROM figures
            WHERE kept = 1 AND section_id IN ({placeholders})
            ORDER BY page, figure_id
            """,
            list(section_ids),
        ):
            figure_map[fig["section_id"]].append(fig)

    blocks: list[str] = []
    for i, row in enumerate(picked, start=1):
        window, truncated = _window(row["text"], cfg.window_max_chars, terms)
        # A one-unit section is only complete if the window also showed all of
        # it. Claiming `complete_section` over elided text tells the agent it
        # may cite a value without calling docs_read (AGENTS.md rule 1).
        complete = row["unit_count"] == 1 and not truncated
        completeness = "complete_section" if complete else "fragment"
        page_bit = f" | p{row['page']}" if row["page"] is not None else ""
        header = (
            f"[{i}] {row['filename']} | {row['section_path'] or '(preamble)'}"
            f"{page_bit} | {completeness}"
        )
        lines = [header, _indent_block(window)]
        for fig in figure_map.get(row["section_id"], []):
            cap = fig["caption"] or "figure"
            lines.append(f"    figure: {cap}")
            url = figure_url(fig["path"])
            if url:
                lines.append(f"            {url}")
        blocks.append("\n".join(lines))

    footer = f"{len(picked)} results (of {total_matches} matches). Query: {query!r}"
    return "\n\n".join(blocks) + "\n\n" + footer
