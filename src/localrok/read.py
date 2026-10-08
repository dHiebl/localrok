"""Section / page / document assembly for docs_read and docs_outline."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from localrok.config import ReadConfig
from localrok.db import resolve_document, unknown_doc_message
from localrok.figures import figure_alt, figure_url

_PAGE_RANGE_RE = re.compile(
    r"^(?P<doc>.+?)\s*>\s*p(?P<start>\d+)(?:\s*-\s*(?P<end>\d+))?\s*$",
    re.IGNORECASE,
)


@dataclass
class ResolvedTarget:
    kind: str  # section | pages | document
    doc_id: int
    filename: str
    title: str | None
    label: str
    section_id: int | None = None
    page_start: int | None = None
    page_end: int | None = None


def _doc_row(conn: sqlite3.Connection, spec: str) -> sqlite3.Row | str:
    row = resolve_document(conn, spec)
    if row is None:
        return unknown_doc_message(conn, spec)
    return row


def _ambiguous(filename: str, path: str, rows: list[sqlite3.Row]) -> str:
    """Name the candidates by page, so a follow-up call can actually pick one.

    Duplicate section paths make the paths themselves useless as a
    disambiguator, so offer the page ranges the agent can read instead.
    """
    lines = [f"Ambiguous section {path!r} — {len(rows)} matches:"]
    for row in rows[:8]:
        start, end = row["page_start"], row["page_end"]
        if start is None:
            where = "page unknown"
        elif end is None or end == start:
            where = f"p{start}"
        else:
            where = f"p{start}-{end}"
        lines.append(f'  {row["section_path"]} | {where} | {row["char_count"]} chars')
    if len(rows) > 8:
        lines.append(f"  … and {len(rows) - 8} more")
    lines.append(
        "Pick one by page range, e.g. "
        f'docs_read(targets=["{filename} > p{rows[0]["page_start"] or 1}-'
        f'{rows[0]["page_end"] or rows[0]["page_start"] or 1}"]).'
    )
    return "\n".join(lines)


def _match_section(conn: sqlite3.Connection, doc_id: int, path: str) -> sqlite3.Row | str:
    path = path.strip()
    rows = conn.execute(
        "SELECT * FROM sections WHERE doc_id = ? ORDER BY ordinal", (doc_id,)
    ).fetchall()
    filename = conn.execute(
        "SELECT filename FROM documents WHERE doc_id = ?", (doc_id,)
    ).fetchone()["filename"]

    # Exact matches can still be plural: a mislabelled running header, or a
    # heading legitimately reused under different parents. Returning an
    # arbitrary one is a silent wrong answer, which is worse than refusing.
    exact = [r for r in rows if r["section_path"] == path]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return _ambiguous(filename, path, exact)

    lowered = path.lower()
    for candidates in (
        [r for r in rows if r["section_path"].lower().endswith(lowered)],
        [r for r in rows if r["section_path"].lower().startswith(lowered)],
        [r for r in rows if lowered in r["section_path"].lower()],
    ):
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            return _ambiguous(filename, path, candidates)
    return f"No section matching {path!r}. Call docs_outline({filename!r}) for the heading tree."


def resolve_target(
    conn: sqlite3.Connection, target: str, default_doc: str | None
) -> ResolvedTarget | str:
    target = target.strip()
    m = _PAGE_RANGE_RE.match(target)
    if m:
        doc = _doc_row(conn, m.group("doc").strip())
        if isinstance(doc, str):
            return doc
        start = int(m.group("start"))
        end = int(m.group("end")) if m.group("end") else start
        if end < start:
            start, end = end, start
        return ResolvedTarget(
            kind="pages",
            doc_id=doc["doc_id"],
            filename=doc["filename"],
            title=doc["title"],
            label=f"{doc['filename']} > p{start}" + (f"-{end}" if end != start else ""),
            page_start=start,
            page_end=end,
        )

    # "<doc> > <section path>": document names never contain '>', section
    # paths may, so only the first '>' separates them. <doc> is a filename,
    # a document ID, or any unambiguous substring of a filename.
    head, sep, rest = target.partition(">")
    if sep:
        doc = resolve_document(conn, head.strip())
        if doc is not None:
            sec = _match_section(conn, doc["doc_id"], rest.strip())
            if isinstance(sec, str):
                return sec
            return ResolvedTarget(
                kind="section",
                doc_id=doc["doc_id"],
                filename=doc["filename"],
                title=doc["title"],
                label=f"{doc['filename']} > {sec['section_path']}",
                section_id=sec["section_id"],
            )
    else:
        doc = resolve_document(conn, target)
        if doc is not None:
            return ResolvedTarget(
                kind="document",
                doc_id=doc["doc_id"],
                filename=doc["filename"],
                title=doc["title"],
                label=doc["filename"],
            )

    # Bare section path
    if default_doc is None:
        return (
            f"{unknown_doc_message(conn, head.strip())}\n"
            f"If {target!r} is a section path, it needs a document: pass doc=... "
            "or prefix it with a filename or document ID, as in '<doc> > <section>'."
        )
    doc = _doc_row(conn, default_doc)
    if isinstance(doc, str):
        return doc
    sec = _match_section(conn, doc["doc_id"], target)
    if isinstance(sec, str):
        # Neither a section of doc=... nor '<doc> > <section>': say both.
        return f"{unknown_doc_message(conn, head.strip())}\n{sec}" if sep else sec
    return ResolvedTarget(
        kind="section",
        doc_id=doc["doc_id"],
        filename=doc["filename"],
        title=doc["title"],
        label=f"{doc['filename']} > {sec['section_path']}",
        section_id=sec["section_id"],
    )


def _units_for_target(conn: sqlite3.Connection, target: ResolvedTarget) -> list[sqlite3.Row]:
    if target.kind == "section":
        return conn.execute(
            """
            SELECT * FROM units WHERE section_id = ? ORDER BY ordinal
            """,
            (target.section_id,),
        ).fetchall()
    if target.kind == "pages":
        return conn.execute(
            """
            SELECT * FROM units
            WHERE doc_id = ? AND page IS NOT NULL AND page >= ? AND page <= ?
            ORDER BY section_id, ordinal
            """,
            (target.doc_id, target.page_start, target.page_end),
        ).fetchall()
    return conn.execute(
        """
        SELECT u.* FROM units u
        JOIN sections s ON s.section_id = u.section_id
        WHERE u.doc_id = ?
        ORDER BY s.ordinal, u.ordinal
        """,
        (target.doc_id,),
    ).fetchall()


def _heading_for_units(conn: sqlite3.Connection, units: list[sqlite3.Row]) -> str:
    if not units:
        return ""
    sec = conn.execute(
        "SELECT heading, section_path FROM sections WHERE section_id = ?",
        (units[0]["section_id"],),
    ).fetchone()
    if sec is None:
        return ""
    return sec["heading"] or sec["section_path"] or ""


def assemble_units(
    conn: sqlite3.Connection,
    target: ResolvedTarget,
    units: list[sqlite3.Row],
) -> str:
    if not units:
        return f"{target.label}\n(no text in range)"

    heading = _heading_for_units(conn, units)
    title = target.title or target.filename
    first_page = units[0]["page"]
    page_bit = f", p{first_page}" if first_page is not None else ""
    head = f"## {heading or target.label}                     [{title}{page_bit}]"
    parts = [target.label, head, ""]
    last_page = first_page
    for unit in units:
        if unit["page"] is not None and last_page is not None and unit["page"] != last_page:
            parts.append(f"[p{unit['page']}]")
            parts.append("")
            last_page = unit["page"]
        elif unit["page"] is not None and last_page is None:
            last_page = unit["page"]
        parts.append(unit["text"])
        parts.append("")

    figures = _figures_for_target(conn, target, units)
    if figures:
        parts.append("Figures:")
        for fig in figures:
            cap = fig["caption"] or "figure"
            page = f" p{fig['page']}" if fig["page"] is not None else ""
            parts.append(f"- {cap}{page}")
            url = figure_url(fig["path"])
            if url:
                parts.append(f"  ![{figure_alt(fig['caption'])}]({url})")
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _figures_for_target(
    conn: sqlite3.Connection, target: ResolvedTarget, units: list[sqlite3.Row]
) -> list[sqlite3.Row]:
    if target.kind == "section" and target.section_id is not None:
        return conn.execute(
            """
            SELECT * FROM figures
            WHERE kept = 1 AND section_id = ?
            ORDER BY page, figure_id
            """,
            (target.section_id,),
        ).fetchall()
    section_ids = {u["section_id"] for u in units}
    if not section_ids:
        return []
    placeholders = ",".join("?" * len(section_ids))
    return conn.execute(
        f"""
        SELECT * FROM figures
        WHERE kept = 1 AND section_id IN ({placeholders})
        ORDER BY page, figure_id
        """,
        list(section_ids),
    ).fetchall()


def _tokens(text: str, chars_per_token: float) -> int:
    return max(1, int(round(len(text) / chars_per_token)))


def _slice_units(
    units: list[sqlite3.Row], budget_chars: int
) -> tuple[list[sqlite3.Row], list[sqlite3.Row]]:
    taken: list[sqlite3.Row] = []
    used = 0
    for u in units:
        extra = len(u["text"]) + 2
        if taken and used + extra > budget_chars:
            rest = units[len(taken) :]
            return taken, rest
        taken.append(u)
        used += extra
    return taken, []


def read(
    conn: sqlite3.Connection,
    targets: list[str],
    cfg: ReadConfig,
    *,
    doc: str | None = None,
) -> str:
    if not targets:
        return "docs_read requires at least one target."
    if len(targets) > 10:
        return "docs_read accepts at most 10 targets."

    budget_tokens = cfg.max_tokens
    budget_chars = int(budget_tokens * cfg.chars_per_token)
    used_chars = 0
    blocks: list[str] = []
    omitted: list[str] = []

    for raw in targets:
        resolved = resolve_target(conn, raw, doc)
        if isinstance(resolved, str):
            blocks.append(resolved)
            continue
        units = _units_for_target(conn, resolved)
        remaining_chars = budget_chars - used_chars
        if remaining_chars <= 0:
            omitted.append(_omit_line(resolved, units, cfg.chars_per_token))
            continue

        full = assemble_units(conn, resolved, units)
        if len(full) <= remaining_chars:
            blocks.append(full.rstrip())
            used_chars += len(full)
            continue

        # First target (or remaining budget) cannot fit the whole assembly.
        taken, rest = _slice_units(units, remaining_chars)
        if not taken:
            omitted.append(_omit_line(resolved, units, cfg.chars_per_token))
            continue
        partial = assemble_units(conn, resolved, taken)
        rest_tokens = _tokens("\n\n".join(u["text"] for u in rest), cfg.chars_per_token)
        marker = (
            f"--- OMITTED (budget) ---\n"
            f"remainder of {resolved.label!r} (about {rest_tokens} tokens)\n"
            f"Retrieve with: {_resume_hint(resolved, rest)}"
        )
        blocks.append(partial.rstrip() + "\n\n" + marker)
        used_chars = budget_chars

    if omitted:
        blocks.append("--- OMITTED (budget) ---\n" + "\n".join(omitted))

    return "\n---\n".join(blocks) if blocks else "Nothing to read."


def _resume_hint(resolved: ResolvedTarget, rest: list[sqlite3.Row]) -> str:
    """A call that makes progress.

    Repeating the target that was just truncated would return the same prefix
    and the same marker, so the agent would loop. Only a page range narrower
    than what was already read moves forward; when the remainder has no page
    provenance to narrow by, say so instead of suggesting a no-op.
    """
    pages = [u["page"] for u in rest if u["page"] is not None]
    if pages:
        start, end = min(pages), max(pages)
        span = f"p{start}" + (f"-{end}" if end != start else "")
        return f'docs_read(targets=["{resolved.filename} > {span}"])'
    return (
        f"nothing narrower to request — the remainder has no page provenance. "
        f"Call docs_outline({resolved.filename!r}) and read a subsection instead."
    )


def _omit_line(resolved: ResolvedTarget, units: list[sqlite3.Row], chars_per_token: float) -> str:
    tokens = _tokens("\n\n".join(u["text"] for u in units), chars_per_token)
    return (
        f"{resolved.label!r} (about {tokens} tokens)\n"
        f'Retrieve with: docs_read(targets=["{resolved.label}"])'
    )


def outline(conn: sqlite3.Connection, doc: str, max_depth: int = 3) -> str:
    row = resolve_document(conn, doc)
    if row is None:
        return unknown_doc_message(conn, doc)

    score = row["structure_score"]
    score_bit = f", structure_score {score:.2f}" if score is not None else ""
    pages = row["page_count"]
    page_bit = f"{pages} pages" if pages else "unknown pages"
    lines = [f"{row['filename']} — {page_bit}{score_bit}"]

    sections = conn.execute(
        """
        SELECT * FROM sections WHERE doc_id = ? AND depth >= 1 AND depth <= ?
        ORDER BY ordinal
        """,
        (row["doc_id"], max_depth),
    ).fetchall()

    fig_counts = {
        r["section_id"]: r["n"]
        for r in conn.execute(
            """
            SELECT section_id, COUNT(*) AS n FROM figures
            WHERE kept = 1 AND doc_id = ? AND section_id IS NOT NULL
            GROUP BY section_id
            """,
            (row["doc_id"],),
        )
    }

    for sec in sections:
        indent = "  " * sec["depth"]
        heading = sec["heading"] or sec["section_path"]
        page = f"p{sec['page_start']}" if sec["page_start"] is not None else "p?"
        chars = sec["char_count"]
        if chars >= 1000:
            size = f"{chars / 1000:.1f}k chars"
        else:
            size = f"{chars} chars"
        fig_n = fig_counts.get(sec["section_id"], 0)
        fig_bit = f"  [{fig_n} figure{'s' if fig_n != 1 else ''}]" if fig_n else ""
        # pad heading to a column
        pad = 42 - len(heading)
        dots = "." * max(2, pad) if pad > 0 else ".."
        lines.append(f"{indent}{heading} {dots} {page}  ({size}){fig_bit}")

    if len(lines) == 1:
        lines.append("  (no headings)")
    return "\n".join(lines) + "\n"
