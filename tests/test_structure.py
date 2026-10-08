from localrok.config import PromptConfig
from localrok.corpus import CorpusBudgetError, generate_corpus_markdown
from localrok.structure import clamp01, structure_score
from tests.helpers import tmp_conn


def test_structure_score_well_formed_manual():
    # ~22 headings on 44 pages, no orphans, depth 3
    score = structure_score(heading_count=22, page_count=44, orphan_char_ratio=0.0, max_depth=3)
    assert score > 0.7


def test_structure_score_flat_disclaimer_manual():
    score = structure_score(heading_count=1, page_count=101, orphan_char_ratio=0.9, max_depth=0)
    assert score < 0.5


def test_duplicate_paths_lower_the_score():
    """The detector must not read its own target defect as good structure."""
    clean = structure_score(
        heading_count=22, page_count=44, orphan_char_ratio=0.0, max_depth=3
    )
    polluted = structure_score(
        heading_count=22,
        page_count=44,
        orphan_char_ratio=0.0,
        max_depth=3,
        duplicate_char_ratio=0.63,
    )
    assert polluted < clean
    assert polluted < 0.5, "a document with 63% unaddressable text must be flagged"


def test_clamp01():
    assert clamp01(-1) == 0
    assert clamp01(2) == 1
    assert clamp01(0.3) == 0.3


def test_corpus_map_contains_every_document(tmp_path):
    conn = tmp_conn(tmp_path)
    text = generate_corpus_markdown(conn, PromptConfig())
    assert "## device-manual-v4.0.0.pdf" in text
    assert "## operations-manual.pdf" in text
    assert "⚠" not in text or "low structure" in text  # high scores in fixture
    conn.close()


def test_corpus_budget_never_drops_documents(tmp_path):
    conn = tmp_conn(tmp_path)
    prompt = PromptConfig(corpus_map_max_tokens=1, corpus_chars_per_token=4)
    try:
        generate_corpus_markdown(conn, prompt)
        assert False, "expected budget error"
    except CorpusBudgetError as exc:
        assert "heading lists removed" in str(exc)
    conn.close()
