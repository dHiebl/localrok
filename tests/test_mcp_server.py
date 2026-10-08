import pytest

from eval.gold import parse_read_labels
from localrok.config import ReadConfig
from localrok.read import read
from server.mcp_server import MCP_OUTPUT_MAX_BYTES, _limit_output
from tests.helpers import tmp_conn


@pytest.mark.parametrize("name", ["docs_search", "docs_read", "docs_outline"])
def test_large_multibyte_evidence_has_visible_omission_inside_pi_limit(name):
    # A character-based budget would exceed Pi's byte limit for this corpus.
    source = "handbook.pdf > Conditions | p3\n" + "条件と例外🙂\n" * 4000
    result = _limit_output(name, source)
    assert len(result.encode("utf-8")) <= MCP_OUTPUT_MAX_BYTES < 20 * 1024
    assert result.startswith("--- PARTIAL RESULT")
    assert "handbook.pdf > Conditions | p3" in result
    assert "--- OMITTED (MCP output budget) ---" in result
    assert "\ufffd" not in result


def test_short_mcp_evidence_is_preserved():
    source = "handbook.pdf > Conditions | p3\nThe policy applies only during working hours."
    assert _limit_output("docs_read", source) == source


def test_partial_read_still_records_the_returned_source_location(tmp_path):
    conn = tmp_conn(tmp_path)
    conn.execute("UPDATE units SET text = ? WHERE page = 21", ("Conditions: " + "条件🙂" * 6000,))
    conn.commit()
    source = read(conn, ["device-manual-v4.0.0.pdf > 7 Electrical Characteristics"], ReadConfig())
    result = _limit_output("docs_read", source)
    conn.close()
    assert "OMITTED (MCP output budget)" in result
    hits = parse_read_labels(result)
    assert len(hits) == 1
    assert hits[0].filename == "device-manual-v4.0.0.pdf"
    assert "7 Electrical Characteristics" in hits[0].section
