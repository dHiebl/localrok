"""Stage 4: index sections, units, and figures into SQLite FTS5."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from localrok.db import delete_document
from localrok.models import DocumentMeta, FigureDraft, SectionDraft


def index_document(
    conn: sqlite3.Connection,
    meta: DocumentMeta,
    sections: list[SectionDraft],
    figures: list[FigureDraft],
    figure_section_ordinals: dict[int, int],
) -> int:
    """Replace any existing row for this filename and insert the new tree.

    Returns doc_id.
    """
    delete_document(conn, meta.filename)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cur = conn.execute(
        """
        INSERT INTO documents (
            filename, stem, title, doc_ident, doc_type, version, page_count,
            has_provenance, source_sha256, docling_version, ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            meta.filename,
            meta.stem,
            meta.title,
            meta.doc_ident,
            meta.doc_type,
            meta.version,
            meta.page_count,
            meta.has_provenance,
            meta.source_sha256,
            meta.docling_version,
            now,
        ),
    )
    doc_id = int(cur.lastrowid)

    section_ids: list[int] = []
    for section in sections:
        cur = conn.execute(
            """
            INSERT INTO sections (
                doc_id, section_path, heading, depth, ordinal,
                page_start, page_end, unit_count, char_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                section.section_path,
                section.heading,
                section.depth,
                section.ordinal,
                section.page_start,
                section.page_end,
                len(section.units),
                section.char_count,
            ),
        )
        section_ids.append(int(cur.lastrowid))
        for ord_u, unit in enumerate(section.units):
            bbox = json.dumps(unit.bbox) if unit.bbox is not None else None
            conn.execute(
                """
                INSERT INTO units (
                    section_id, doc_id, ordinal, text, section_path, page, page_end, bbox, kind
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    section_ids[-1],
                    doc_id,
                    ord_u,
                    unit.text,
                    section.section_path,
                    unit.page,
                    unit.page_end,
                    bbox,
                    unit.kind,
                ),
            )

    for fig_i, fig in enumerate(figures):
        sec_ord = figure_section_ordinals.get(fig_i)
        section_id = section_ids[sec_ord] if sec_ord is not None else None
        conn.execute(
            """
            INSERT INTO figures (
                doc_id, section_id, page, path, caption, class_name,
                confidence, label_text, chart_csv, kept
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                section_id,
                fig.page,
                fig.rel_path,
                fig.caption,
                fig.class_name,
                fig.confidence,
                fig.label_text,
                fig.chart_csv,
                fig.kept,
            ),
        )

    return doc_id
