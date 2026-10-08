"""SQLite schema, connections, and document-resolution helpers."""

from __future__ import annotations

import sqlite3
import sys
from difflib import get_close_matches
from pathlib import Path

# Bump when SCHEMA_SQL changes. An index built by another version is not
# migrated: `localrok ingest` rebuilds it from the Docling JSON, which is cheap.
SCHEMA_VERSION = 2

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    doc_id          INTEGER PRIMARY KEY,
    filename        TEXT NOT NULL UNIQUE,
    stem            TEXT NOT NULL,
    title           TEXT,
    doc_ident       TEXT,
    doc_type        TEXT,
    version         TEXT,
    page_count      INTEGER,
    has_provenance  INTEGER NOT NULL,
    structure_score REAL,
    source_sha256   TEXT NOT NULL,
    docling_version TEXT NOT NULL,
    ingested_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sections (
    section_id   INTEGER PRIMARY KEY,
    doc_id       INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    section_path TEXT NOT NULL,
    heading      TEXT,
    depth        INTEGER NOT NULL,
    ordinal      INTEGER NOT NULL,
    page_start   INTEGER,
    page_end     INTEGER,
    unit_count   INTEGER NOT NULL,
    char_count   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sections_doc  ON sections(doc_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_sections_path ON sections(doc_id, section_path);

CREATE TABLE IF NOT EXISTS units (
    unit_id    INTEGER PRIMARY KEY,
    section_id INTEGER NOT NULL REFERENCES sections(section_id) ON DELETE CASCADE,
    doc_id     INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ordinal    INTEGER NOT NULL,
    text       TEXT NOT NULL,
    -- Copy of sections.section_path, so the heading is searchable with the
    -- text it governs: many identifiers (test case IDs) appear only there.
    section_path TEXT NOT NULL,
    page       INTEGER,
    page_end   INTEGER,
    bbox       TEXT,
    kind       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_units_section ON units(section_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_units_doc     ON units(doc_id);

CREATE TABLE IF NOT EXISTS figures (
    figure_id    INTEGER PRIMARY KEY,
    doc_id       INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    section_id   INTEGER REFERENCES sections(section_id) ON DELETE SET NULL,
    page         INTEGER,
    path         TEXT NOT NULL,
    caption      TEXT,
    class_name   TEXT,
    confidence   REAL,
    label_text   TEXT,
    chart_csv    TEXT,
    kept         INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_figures_doc ON figures(doc_id, page);

-- FTS5 over units: body text plus the heading it sits under. Column weights
-- are applied at query time, bm25(units_fts, 1.0, search.heading_weight).
CREATE VIRTUAL TABLE IF NOT EXISTS units_fts USING fts5(
    text,
    section_path,
    content = 'units',
    content_rowid = 'unit_id',
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS units_ai AFTER INSERT ON units BEGIN
    INSERT INTO units_fts(rowid, text, section_path)
    VALUES (new.unit_id, new.text, new.section_path);
END;
CREATE TRIGGER IF NOT EXISTS units_ad AFTER DELETE ON units BEGIN
    INSERT INTO units_fts(units_fts, rowid, text, section_path)
    VALUES('delete', old.unit_id, old.text, old.section_path);
END;
CREATE TRIGGER IF NOT EXISTS units_au AFTER UPDATE ON units BEGIN
    INSERT INTO units_fts(units_fts, rowid, text, section_path)
    VALUES('delete', old.unit_id, old.text, old.section_path);
    INSERT INTO units_fts(rowid, text, section_path)
    VALUES (new.unit_id, new.text, new.section_path);
END;
"""


class IndexUnavailable(RuntimeError):
    """No index on disk, or one built with a different schema version."""


def connect(db_path: Path, *, create: bool = True) -> sqlite3.Connection:
    """Open the index. `create=True` is for ingest only.

    Ingest replaces an index from another schema version with an empty one;
    every other caller gets IndexUnavailable instead of silently creating an
    empty file or querying columns that do not exist.
    """
    db_path = Path(db_path)
    if create:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        version = _file_schema_version(db_path)
        if version is not None and version != SCHEMA_VERSION:
            print(
                f"{db_path.name}: schema v{version}, code needs v{SCHEMA_VERSION} — "
                "rebuilding the index from scratch",
                file=sys.stderr,
            )
            for suffix in ("", "-wal", "-shm"):
                Path(f"{db_path}{suffix}").unlink(missing_ok=True)
    elif not db_path.exists():
        raise IndexUnavailable(f"No index at {db_path}. Run `localrok ingest`.")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    if create:
        init_schema(conn)
    else:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            conn.close()
            raise IndexUnavailable(
                f"Index at {db_path} has schema v{version}, this code needs "
                f"v{SCHEMA_VERSION}. Run `localrok ingest` to rebuild it."
            )
    return conn


def _file_schema_version(db_path: Path) -> int | None:
    """Schema version of an existing index; None when there is nothing to keep."""
    if not db_path.exists():
        return None
    conn = sqlite3.connect(str(db_path))
    try:
        if conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0] == 0:
            return None
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()


def optimize_fts(conn: sqlite3.Connection) -> None:
    conn.execute("INSERT INTO units_fts(units_fts) VALUES('optimize')")
    conn.commit()


def delete_document(conn: sqlite3.Connection, filename: str) -> None:
    conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))


def lookup_document(conn: sqlite3.Connection, filename: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM documents WHERE filename = ?", (filename,)).fetchone()


def resolve_document(conn: sqlite3.Connection, spec: str) -> sqlite3.Row | None:
    """Resolve a filename, document ID, or unambiguous substring to a documents row."""
    spec = spec.strip()
    row = lookup_document(conn, spec)
    if row is not None:
        return row
    idents = conn.execute(
        "SELECT * FROM documents WHERE doc_ident = ? COLLATE NOCASE", (spec,)
    ).fetchall()
    if len(idents) == 1:
        return idents[0]
    rows = conn.execute("SELECT * FROM documents WHERE filename LIKE ?", (f"%{spec}%",)).fetchall()
    if len(rows) == 1:
        return rows[0]
    # Unique stem match
    stems = conn.execute(
        "SELECT * FROM documents WHERE stem = ? OR stem LIKE ?",
        (spec, f"%{spec}%"),
    ).fetchall()
    if len(stems) == 1:
        return stems[0]
    if len(rows) > 1:
        return None
    return None


def closest_filenames(conn: sqlite3.Connection, spec: str, n: int = 3) -> list[str]:
    names = [r["filename"] for r in conn.execute("SELECT filename FROM documents")]
    if not names:
        return []
    close = get_close_matches(spec, names, n=n, cutoff=0.2)
    if close:
        return close
    lowered = spec.lower()
    contains = [name for name in names if lowered in name.lower()]
    return contains[:n] or names[:n]


def unknown_doc_message(conn: sqlite3.Connection, spec: str) -> str:
    """Explain why a document spec did not resolve.

    `resolve_document` returns None for both "no match" and "several matches";
    calling the second case unknown sends the agent looking for a different
    name when it only needed to be more specific.
    """
    matches = [
        r["filename"]
        for r in conn.execute(
            "SELECT filename FROM documents WHERE filename LIKE ? ORDER BY filename",
            (f"%{spec.strip()}%",),
        )
    ]
    if len(matches) > 1:
        listed = ", ".join(matches[:5])
        more = f" (+{len(matches) - 5} more)" if len(matches) > 5 else ""
        return f"Ambiguous document {spec!r} — matches: {listed}{more}. Use a longer substring."
    close = closest_filenames(conn, spec)
    if not close:
        return f"Unknown document {spec!r}. Index is empty — run `localrok ingest`."
    listed = ", ".join(close)
    return f"Unknown document {spec!r}. Closest: {listed}"
