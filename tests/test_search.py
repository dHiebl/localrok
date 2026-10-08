from localrok.config import SearchConfig
from localrok.index import index_document
from localrok.models import DocumentMeta, SectionDraft
from localrok.search import is_identifier, parse_query, search
from tests.helpers import make_unit, tmp_conn


def test_identifier_detection():
    assert is_identifier("J5")
    assert is_identifier("DOC7")
    assert is_identifier("A3200")
    assert not is_identifier("watchdog")


def test_parse_query_or_and_prefix():
    expr, terms = parse_query("watchdog reset", exact=False)
    assert " OR " in expr
    assert "watchdog*" in expr
    assert "reset*" in expr
    assert terms == ["watchdog", "reset"]


def test_parse_query_exact_phrase():
    expr, terms = parse_query("warm reset", exact=True)
    assert expr == '"warm reset"'
    assert terms == ["warm reset"]


def test_parse_query_quotes_identifiers():
    expr, _ = parse_query("J5 pin", exact=False)
    assert '"J5"' in expr


def test_search_format_and_complete_section(tmp_path):
    conn = tmp_conn(tmp_path)
    j5 = search(conn, "J5", SearchConfig(), limit=8)
    assert "[1] device-manual-v4.0.0.pdf" in j5
    assert "J5 provides the MAINTPORT" in j5
    assert "figure: Figure 4-2" in j5
    assert "results (of" in j5
    # Figure caption/label units share the section, so J5 is a fragment.
    # A one-unit section with no figures is complete_section.
    wd = search(conn, "warm reset", SearchConfig())
    assert "complete_section" in wd
    conn.close()


def test_search_fragment_on_multi_unit_section(tmp_path):
    conn = tmp_conn(tmp_path)
    out = search(conn, "undervoltage", SearchConfig())
    assert "fragment" in out
    assert "21.0 V" in out or "10.5 V" in out
    conn.close()


def test_search_empty_has_hint(tmp_path):
    conn = tmp_conn(tmp_path)
    out = search(conn, "xyzzy-no-such-term", SearchConfig())
    assert "No matches" in out
    assert "docs_outline" in out
    conn.close()


def test_search_doc_filter(tmp_path):
    conn = tmp_conn(tmp_path)
    out = search(conn, "watchdog", SearchConfig(), doc="operations-manual.pdf")
    assert "operations-manual.pdf" in out
    assert "device-manual" not in out.split("Query:")[0]
    conn.close()


def test_search_unknown_doc(tmp_path):
    conn = tmp_conn(tmp_path)
    out = search(conn, "J5", SearchConfig(), doc="no-such.pdf")
    assert "Unknown document" in out
    assert "Closest:" in out
    conn.close()


def test_truncated_one_unit_section_is_a_fragment(tmp_path):
    """complete_section must mean the agent saw all of it.

    AGENTS.md rule 1 lets the agent cite a value straight from a
    complete_section window, so the label may not outrun the text shown.
    """
    conn = tmp_conn(tmp_path)
    out = search(conn, "watchdog expires", SearchConfig(window_max_chars=40))
    assert "…" in out
    assert "complete_section" not in out
    assert "fragment" in out
    conn.close()


def test_untruncated_one_unit_section_stays_complete(tmp_path):
    conn = tmp_conn(tmp_path)
    out = search(conn, "watchdog expires", SearchConfig(window_max_chars=5000))
    assert "complete_section" in out
    conn.close()


def test_identifier_only_in_heading_is_found(tmp_path):
    """Test reports name the case in the heading and never repeat it below."""
    conn = tmp_conn(tmp_path)
    meta = DocumentMeta(
        filename="csvr.pdf",
        stem="csvr",
        title="Verification Report",
        doc_ident="CSVR-001",
        doc_type="verification_report",
        version="2.0",
        page_count=30,
        has_provenance=1,
        source_sha256="ghi",
        docling_version="2.126.0",
    )
    section = SectionDraft(
        section_path="2.1.1 TC-001",
        heading="2.1.1 TC-001",
        depth=1,
        ordinal=0,
        units=[make_unit("Spin up to 3 deg/s and rotate freely for 140 hours.", page=6)],
    )
    index_document(conn, meta, [section], [], {})
    conn.commit()
    out = search(conn, "TC-001", SearchConfig())
    assert "[1] csvr.pdf | 2.1.1 TC-001 | p6" in out
    assert "140 hours" in out
    conn.close()


def test_navigation_units_are_not_ranked(tmp_path):
    """A table of contents matches heading-shaped queries and cites front matter."""
    conn = tmp_conn(tmp_path)
    conn.execute(
        "UPDATE units SET kind = 'navigation' WHERE text LIKE '%MAINTPORT configuration%'"
    )
    conn.commit()
    out = search(conn, "MAINTPORT", SearchConfig())
    assert "No matches" in out
    conn.close()
