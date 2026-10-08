from types import SimpleNamespace

from localrok.chunk import (
    furniture_headings,
    group_into_sections,
    inherit_missing_pages,
    mark_running_furniture,
    normalize_heading,
    unit_from_chunk,
)
from localrok.models import SectionDraft, UnitDraft


def _chunk(text: str, headings: list[str], page: int | None):
    prov = SimpleNamespace(page_no=page, bbox=SimpleNamespace(l=0, t=0, r=10, b=10))
    item = SimpleNamespace(prov=[prov] if page is not None else [])
    meta = SimpleNamespace(headings=headings, doc_items=[item])
    return SimpleNamespace(text=text, meta=meta)


def _text_item(text: str, label: str, pages: list[int]):
    return SimpleNamespace(
        text=text,
        label=label,
        prov=[SimpleNamespace(page_no=p, bbox=None) for p in pages],
    )


def _doc(texts: list, page_count: int):
    return SimpleNamespace(texts=texts, pages={i: None for i in range(1, page_count + 1)})


def test_group_same_headings_one_section():
    chunks = [
        _chunk("para one", ["Manual", "5 Power"], 3),
        _chunk("para two", ["Manual", "5 Power"], 3),
    ]
    sections = group_into_sections(chunks)
    assert len(sections) == 1
    assert sections[0].section_path == "Manual > 5 Power"
    assert len(sections[0].units) == 2
    assert sections[0].depth == 2
    assert sections[0].heading == "5 Power"


def test_heading_change_starts_new_section():
    chunks = [
        _chunk("a", ["Intro"], 1),
        _chunk("b", ["Setup"], 2),
    ]
    sections = group_into_sections(chunks)
    assert [s.section_path for s in sections] == ["Intro", "Setup"]
    assert [s.ordinal for s in sections] == [0, 1]


def test_empty_headings_are_preamble_depth_zero():
    chunks = [_chunk("front matter", [], 1)]
    sections = group_into_sections(chunks)
    assert sections[0].section_path == ""
    assert sections[0].depth == 0
    assert sections[0].heading is None


def test_inherit_missing_pages():
    sections = [
        SectionDraft(
            section_path="A",
            heading="A",
            depth=1,
            ordinal=0,
            units=[
                UnitDraft("has page", 10, 10, None, "text"),
                UnitDraft("no page", None, None, None, "text"),
            ],
        )
    ]
    inherit_missing_pages(sections)
    assert sections[0].units[1].page == 10


def test_furniture_from_docling_page_header_label():
    """Docling labels the running header inconsistently; one honest label is enough."""
    texts = [
        _text_item("EXAMPLE PUBLISHER | User Manual", "page_header", [2]),
        _text_item("EXAMPLE PUBLISHER | User Manual", "section_header", [11]),
        _text_item("EXAMPLE PUBLISHER | User Manual", "section_header", [12]),
        _text_item("4.3 Connector Interface", "section_header", [37]),
    ]
    furniture = furniture_headings(_doc(texts, 82))
    assert normalize_heading("EXAMPLE PUBLISHER | User Manual") in furniture
    assert normalize_heading("4.3 Connector Interface") not in furniture


def test_furniture_from_repetition_when_never_labelled():
    texts = [_text_item("Acme Handbook", "section_header", [p]) for p in range(1, 40)]
    texts.append(_text_item("7 Electrical", "section_header", [21]))
    furniture = furniture_headings(_doc(texts, 80))
    assert normalize_heading("Acme Handbook") in furniture
    assert normalize_heading("7 Electrical") not in furniture


def test_heading_reused_across_chapters_is_not_furniture():
    """A real heading repeated under a few parents must survive."""
    texts = [_text_item("Overview", "section_header", [p]) for p in (3, 12, 25, 40)]
    assert normalize_heading("Overview") not in furniture_headings(_doc(texts, 80))


def _page_doc(items: list[tuple[str, int, float]]):
    """Real DoclingDocument, 800 pt pages; items are (text, page, top as a page fraction)."""
    from docling_core.types.doc import (
        BoundingBox,
        CoordOrigin,
        DocItemLabel,
        DoclingDocument,
        ProvenanceItem,
        Size,
    )

    doc = DoclingDocument(name="t")
    for page in sorted({p for _, p, _ in items}):
        doc.add_page(page_no=page, size=Size(width=600, height=800))
    for text, page, top in items:
        box = BoundingBox(
            l=50, t=800 * top, r=400, b=800 * top + 12, coord_origin=CoordOrigin.TOPLEFT
        )
        prov = ProvenanceItem(page_no=page, bbox=box, charspan=(0, len(text)))
        doc.add_text(label=DocItemLabel.TEXT, text=text, prov=prov)
    return doc


def _layers(doc) -> dict[str, set]:
    out: dict[str, set] = {}
    for item in doc.texts:
        out.setdefault(item.text, set()).add(item.content_layer)
    return out


def test_running_header_left_in_body_becomes_furniture():
    from docling_core.types.doc import ContentLayer

    items = []
    for p in range(1, 7):
        items += [("ACME-DOC-001 Issue 1.0", p, 0.04), ("_" * 40, p, 0.07), (f"Body {p}", p, 0.5)]
    # A real heading that starts three pages and also appears mid-page on three more.
    items += [("Results", p, 0.09) for p in (1, 2, 3)] + [("Results", p, 0.4) for p in (4, 5, 6)]
    doc = _page_doc(items)
    assert mark_running_furniture(doc) == 12
    layers = _layers(doc)
    assert layers["ACME-DOC-001 Issue 1.0"] == {ContentLayer.FURNITURE}
    assert layers["_" * 40] == {ContentLayer.FURNITURE}
    assert layers["Results"] == {ContentLayer.BODY}
    assert all(layers[f"Body {p}"] == {ContentLayer.BODY} for p in range(1, 7))


def test_margin_text_on_too_few_pages_stays_body():
    doc = _page_doc([("Draft", 1, 0.04), ("Draft", 2, 0.04), ("Body", 1, 0.5)])
    assert mark_running_furniture(doc) == 0


def test_furniture_only_context_inherits_the_interrupted_section():
    """A header interrupting a section must not orphan the text that follows.

    Filtering alone would leave the middle chunk with no headings at all, which
    is just as unaddressable as the duplicated path it replaced.
    """
    furniture = {normalize_heading("Acme Handbook")}
    chunks = [
        _chunk("first half", ["Acme Handbook", "5 Power"], 10),
        _chunk("header interrupts", ["Acme Handbook"], 11),
        _chunk("second half", ["Acme Handbook", "5 Power"], 11),
    ]
    sections = group_into_sections(chunks, furniture)
    assert [s.section_path for s in sections] == ["5 Power"]
    assert len(sections[0].units) == 3
    assert sections[0].depth == 1


def test_genuine_front_matter_stays_preamble():
    """Only furniture-induced emptiness inherits; real preamble keeps depth 0."""
    chunks = [
        _chunk("cover page text", [], 1),
        _chunk("chapter text", ["1 Introduction"], 2),
    ]
    sections = group_into_sections(chunks)
    assert sections[0].section_path == ""
    assert sections[0].depth == 0
    assert sections[1].section_path == "1 Introduction"


def test_table_of_contents_units_marked_navigation():
    chunks = [
        _chunk("1 Intro .... 3", ["Table of Contents"], 3),
        _chunk("real prose", ["1 Introduction"], 5),
    ]
    sections = group_into_sections(chunks)
    assert sections[0].units[0].kind == "navigation"
    assert sections[1].units[0].kind == "text"


def test_unit_from_chunk_table_kind():
    class TableItem:
        prov = [SimpleNamespace(page_no=4, bbox=None)]

    chunk = SimpleNamespace(
        text="| a | b |",
        meta=SimpleNamespace(headings=["T"], doc_items=[TableItem()]),
    )
    unit = unit_from_chunk(chunk)
    assert unit.kind == "table"
    assert unit.page == 4
