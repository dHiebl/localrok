"""The corpus map is the agent's only prior for picking a document, so its
preview budget must not go to front matter that every document shares."""

from __future__ import annotations

from localrok.corpus import _boilerplate_headings, _informative_headings, _normalize_heading

FRONT_MATTER = ["Approval", "Change Log", "1 Introduction", "1.1 Reference Documents"]


def _corpus(*per_doc_tails: list[str]) -> list[list[str]]:
    return [FRONT_MATTER + list(tail) for tail in per_doc_tails]


def test_normalize_strips_numbering_and_trailing_separators():
    assert _normalize_heading("2.1.1   Verification by Test") == "verification by test"
    assert _normalize_heading("1. Introduction") == "introduction"
    assert _normalize_heading("EXAMPLE PUBLISHER ǀ") == "example publisher"


def test_shared_front_matter_is_dropped_but_shared_content_survives():
    per_doc = _corpus(
        ["2 Verification Results", "Cooling Fans"],
        ["2 Plant Configuration", "Cooling Fans"],
        ["2.1 Grounding"],
        ["2. Report Structure"],
        ["3 Verification Control Matrix"],
        ["2.2 Model Philosophy"],
    )
    boilerplate = _boilerplate_headings(per_doc)

    assert boilerplate == {"approval", "change log", "introduction", "reference documents"}
    # Present in 2 of 6, so it is content rather than front matter.
    assert "cooling fans" not in boilerplate


def test_a_small_corpus_keeps_everything():
    """With two documents, "shared by half" would delete the whole map."""
    assert _boilerplate_headings(_corpus(["2 Results"], ["2 Design"])) == set()


def test_informative_headings_drops_repeats_and_the_documents_own_identifier():
    headings = [
        "Approval",
        "DOC-REPORT-001 Example Test Report",
        "2 Verification Results",
        "2 Verification Results",
        "2.1.1 TC-001",
    ]
    kept = _informative_headings(headings, {"approval"}, "DOC-REPORT-001")
    assert kept == ["2 Verification Results", "2.1.1 TC-001"]


def test_no_identifier_is_not_an_error():
    assert _informative_headings(["2 Results"], set(), None) == ["2 Results"]
