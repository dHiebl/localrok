import pytest

from localrok.convert import parse_document_metadata
from localrok.figures import bbox_contains, figure_units
from localrok.models import FigureDraft


@pytest.mark.parametrize(
    "stem,title,kind,identifier,doc_version",
    [
        ("DOC-POLICY-001 v2.1 Leave Policy", None, "policy", "DOC-POLICY-001", "2.1"),
        ("Annual financial report v3.0", None, "report", None, "3.0"),
        ("device_manual_v4.0.0", None, "manual", None, "4.0.0"),
        (
            "DOC-GUIDE-003 V2.0.1-rc6 Installation Guide",
            None,
            "guide",
            "DOC-GUIDE-003",
            "2.0.1-rc6",
        ),
        ("Meeting notes 1.2", None, "meeting_notes", None, "1.2"),
        ("supplier_document", "Service Agreement", "agreement", None, None),
        ("Notes", None, "unknown", None, None),
        ("DOC-SPEC-012 Interface Specification", None, "specification", "DOC-SPEC-012", None),
    ],
)
def test_general_document_metadata(stem, title, kind, identifier, doc_version):
    meta = parse_document_metadata(f"{stem}.pdf", stem, title, 12, "hash", "2.126.0", 1)
    assert (meta.doc_type, meta.doc_ident, meta.version) == (kind, identifier, doc_version)
    assert meta.title == (title or stem)
    assert meta.filename == f"{stem}.pdf"


def test_missing_provenance_remains_missing():
    meta = parse_document_metadata("notes.md", "notes", None, None, "hash", "2.126.0", 0)
    assert meta.page_count is None
    assert meta.has_provenance == 0


class _Box:
    def __init__(self, left, top, right, bottom):
        self.l, self.t, self.r, self.b = left, top, right, bottom


def test_bbox_contains_origin_agnostic():
    outer = _Box(0, 100, 50, 0)  # bottom-left origin
    inner = _Box(5, 80, 20, 20)
    assert bbox_contains(outer, inner)
    outside = _Box(0, 200, 10, 150)
    assert not bbox_contains(outer, outside)


def test_figure_units_kinds():
    fig = FigureDraft(
        page=12,
        rel_path="figures/x/fig-12-0.png",
        caption="Figure 5-3 — voltage curve",
        class_name="line_chart",
        confidence=0.8,
        label_text="V_bat t",
        chart_csv="t,V\n0,21.0\n1,20.8",
        kept=1,
    )
    units = figure_units(fig)
    kinds = {u.kind for u in units}
    assert kinds == {"caption", "figure_labels", "chart_data"}
    assert any("21.0" in u.text for u in units)
