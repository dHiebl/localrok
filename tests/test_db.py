import sqlite3

import pytest

from localrok.db import SCHEMA_VERSION, IndexUnavailable, connect


def test_readers_never_create_an_index(tmp_path):
    db = tmp_path / "localrok.db"
    with pytest.raises(IndexUnavailable, match="localrok ingest"):
        connect(db, create=False)
    assert not db.exists()


def test_old_schema_is_refused_then_rebuilt_by_ingest(tmp_path):
    db = tmp_path / "localrok.db"
    old = sqlite3.connect(db)
    old.execute("CREATE TABLE units (unit_id INTEGER PRIMARY KEY, text TEXT)")
    old.commit()
    old.close()

    with pytest.raises(IndexUnavailable, match=f"needs v{SCHEMA_VERSION}"):
        connect(db, create=False)

    conn = connect(db, create=True)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(units)")}
    assert "section_path" in columns
    conn.close()
    connect(db, create=False).close()
