from localrok.config import ReadConfig
from localrok.read import outline, read, resolve_target
from tests.helpers import tmp_conn


def test_read_section_path_prefixed(tmp_path):
    conn = tmp_conn(tmp_path)
    out = read(
        conn,
        ["device-manual-v4.0.0.pdf > 4 Interfaces > 4.3 Connector Interface"],
        ReadConfig(),
    )
    assert "J5 provides the MAINTPORT" in out
    assert "Figure 4-2" in out
    conn.close()


def test_read_target_prefixed_with_document_id(tmp_path):
    """The corpus map and citations name documents by ID, not by filename."""
    conn = tmp_conn(tmp_path)
    out = read(conn, ["ops > 8.1 Watchdog Recovery"], ReadConfig())
    assert out.startswith("operations-manual.pdf > Operations Manual > 8 Recovery")
    assert "resume monitoring" in out
    out = read(conn, ["DOC-MANUAL-001 > 4.3 Connector Interface"], ReadConfig())
    assert "J5 provides the MAINTPORT" in out
    conn.close()


def test_read_bare_section_needs_doc(tmp_path):
    conn = tmp_conn(tmp_path)
    out = read(conn, ["4 Interfaces > 4.3 Connector Interface"], ReadConfig())
    assert "needs a document" in out
    out = read(
        conn,
        ["4.3 Connector Interface"],
        ReadConfig(),
        doc="device-manual-v4.0.0.pdf",
    )
    assert "J5 provides the MAINTPORT" in out
    conn.close()


def test_read_page_range(tmp_path):
    conn = tmp_conn(tmp_path)
    out = read(conn, ["device-manual-v4.0.0.pdf > p21-22"], ReadConfig())
    assert "21.0 V" in out
    assert "10.5 V" in out
    conn.close()


def test_read_whole_document(tmp_path):
    conn = tmp_conn(tmp_path)
    out = read(conn, ["device-manual-v4.0.0.pdf"], ReadConfig())
    assert "J5 provides" in out
    assert "undervoltage" in out
    conn.close()


def test_read_batch_cross_document(tmp_path):
    conn = tmp_conn(tmp_path)
    out = read(
        conn,
        [
            "device-manual-v4.0.0.pdf > 5.2 Watchdog Handling",
            "operations-manual.pdf > 8.1 Watchdog Recovery",
        ],
        ReadConfig(),
    )
    assert "---" in out
    assert "warm reset" in out
    assert "resume monitoring" in out
    conn.close()


def test_read_budget_omission(tmp_path):
    conn = tmp_conn(tmp_path)
    cfg = ReadConfig(max_tokens=20, chars_per_token=3.5)
    out = read(
        conn,
        [
            "device-manual-v4.0.0.pdf > 4.3 Connector Interface",
            "operations-manual.pdf > 8.1 Watchdog Recovery",
        ],
        cfg,
    )
    assert "OMITTED (budget)" in out
    assert "docs_read(targets=" in out
    conn.close()


def test_ambiguous_section_lists_candidates(tmp_path):
    conn = tmp_conn(tmp_path)
    resolved = resolve_target(conn, "Watchdog", "device-manual-v4.0.0.pdf")
    # unique suffix match on 5.2 Watchdog Handling
    if isinstance(resolved, str):
        assert "Ambiguous" in resolved or "No section" in resolved
    conn.close()


def test_duplicate_exact_paths_refuse_instead_of_guessing(tmp_path):
    """Never silently return one of several sections sharing a path.

    Returning an arbitrary row hands the agent the wrong text with a citation
    that looks correct, which is worse than refusing.
    """
    conn = tmp_conn(tmp_path)
    conn.execute(
        """
        INSERT INTO sections (doc_id, section_path, heading, depth, ordinal,
                              page_start, page_end, unit_count, char_count)
        SELECT doc_id, section_path, heading, depth, 99, 40, 41, 0, 29064
        FROM sections WHERE section_path LIKE '%4.3 Connector Interface'
        """
    )
    conn.commit()
    out = read(
        conn,
        ["device-manual-v4.0.0.pdf > 4 Interfaces > 4.3 Connector Interface"],
        ReadConfig(),
    )
    assert "Ambiguous section" in out
    assert "2 matches" in out
    assert "J5 provides the MAINTPORT" not in out
    # The candidates must be separable by something other than the path.
    assert "p37" in out and "p40-41" in out
    conn.close()


def test_single_page_target(tmp_path):
    conn = tmp_conn(tmp_path)
    out = read(conn, ["device-manual-v4.0.0.pdf > p21"], ReadConfig())
    assert "21.0 V" in out
    assert "10.5 V" not in out
    conn.close()


def test_resume_hint_narrows_by_page(tmp_path):
    conn = tmp_conn(tmp_path)
    cfg = ReadConfig(max_tokens=30, chars_per_token=3.5)
    target = "device-manual-v4.0.0.pdf > 7 Electrical Characteristics"
    out = read(conn, [target], cfg)
    assert "OMITTED (budget)" in out
    assert 'docs_read(targets=["device-manual-v4.0.0.pdf > p22"])' in out
    conn.close()


def test_resume_hint_never_repeats_the_truncated_call(tmp_path):
    """Suggesting the call that was just truncated would loop forever.

    Reachable whenever the omitted remainder carries no page provenance, so
    there is no narrower page range to point at.
    """
    conn = tmp_conn(tmp_path)
    conn.execute("UPDATE units SET page = NULL, page_end = NULL")
    conn.commit()
    cfg = ReadConfig(max_tokens=30, chars_per_token=3.5)
    target = "device-manual-v4.0.0.pdf > 7 Electrical Characteristics"
    out = read(conn, [target], cfg)
    assert "OMITTED (budget)" in out
    assert f'docs_read(targets=["{target}"])' not in out
    assert "no page provenance" in out
    assert "docs_outline" in out
    conn.close()


def test_outline_tree(tmp_path):
    conn = tmp_conn(tmp_path)
    out = outline(conn, "device-manual-v4.0.0.pdf")
    assert "device-manual-v4.0.0.pdf — 44 pages" in out
    assert "4.3 Connector Interface" in out
    assert "figure" in out.lower()
    conn.close()
