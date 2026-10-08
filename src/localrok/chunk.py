"""Stage 2: DoclingDocument → sections + units via HybridChunker."""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable

from localrok.models import SectionDraft, UnitDraft

_TABLE_TYPE_NAMES = {"TableItem", "Table"}
_FURNITURE_LABELS = ("page_header", "page_footer")
_HEADING_LABELS = ("section_header", "title")

# Running headers Docling leaves in the body: the band at the top and bottom of
# the page they live in, how far one string may drift between pages, and how
# many pages it needs before repetition counts as evidence.
_MARGIN_BAND = 0.12
_Y_TOLERANCE = 0.02
_MIN_PAGES = 3
_RULE_RE = re.compile(r"^[_\-–—=\s]{10,}$")

# Navigation aids: real content, but useless as retrieval targets and prone to
# matching any heading-shaped query. Kept readable, excluded from ranking.
_NAV_HEADING_RE = re.compile(
    r"^(table\s+of\s+contents?|contents?|inhalt|inhaltsverzeichnis"
    r"|list\s+of\s+(figures?|tables?|abbreviations?)"
    r"|abbildungsverzeichnis|tabellenverzeichnis)\s*$",
    re.IGNORECASE,
)


def normalize_heading(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().casefold()


def furniture_headings(doc: Any) -> set[str]:
    """Heading strings that are really running page furniture.

    Docling is inconsistent on repeated page headers: it may label the same
    string `page_header` on some pages and `section_header` on others. Left
    alone, the `section_header` occurrences become sections, so one running
    header can shatter a document into many identically-named sections that
    `docs_read` cannot address and citations cannot identify.

    Two signals, both derived from the document itself:

    1. The string is labelled `page_header`/`page_footer` anywhere. Within a
       single document that is ground truth about what the string is.
    2. It appears as a heading on an implausible share of the pages. Catches
       documents where Docling never labels it as furniture. Deliberately
       conservative, so a heading legitimately reused across chapters
       ("Overview") is not mistaken for furniture.
    """
    labelled: set[str] = set()
    heading_pages: dict[str, set[int]] = defaultdict(set)

    for item in getattr(doc, "texts", None) or []:
        text = normalize_heading(getattr(item, "text", "") or "")
        if not text:
            continue
        label = str(getattr(item, "label", "") or "").lower()
        if any(f in label for f in _FURNITURE_LABELS):
            labelled.add(text)
        elif any(h in label for h in _HEADING_LABELS):
            for prov in getattr(item, "prov", None) or []:
                page = getattr(prov, "page_no", None)
                if page is not None:
                    heading_pages[text].add(int(page))

    page_count = len(getattr(doc, "pages", None) or ()) or 1
    threshold = max(5, int(0.25 * page_count))
    repeated = {t for t, pages in heading_pages.items() if len(pages) > threshold}
    return labelled | repeated


def mark_running_furniture(doc: Any) -> int:
    """Move running headers that Docling left in the body to the furniture layer.

    Docling labels most page headers itself, but on these templates the logo
    words, the document-ID/issue line, the rule under it and the
    "publisher ǀ <title>" banner come out as body text or even section
    headings. The chunker then prepends them to every unit cut from that page.

    A string is furniture when it sits inside the top or bottom margin band at
    the same height on at least three pages, and (almost) never appears in the
    body area. Pure rule lines are furniture wherever they are. The chunker
    and the Markdown export only walk the body layer, so moving an item there
    removes it from both. Returns the number of items moved.
    """
    from docling_core.types.doc import ContentLayer

    placed: dict[tuple[str, str], list[tuple[int, float, Any]]] = defaultdict(list)
    seen: Counter[str] = Counter()
    in_margin: Counter[str] = Counter()
    rules: list[Any] = []
    for item in getattr(doc, "texts", None) or []:
        if item.content_layer != ContentLayer.BODY:
            continue
        text = normalize_heading(item.text)
        if not text:
            continue
        if _RULE_RE.match(text):
            rules.append(item)
            continue
        if len(item.prov) != 1:
            continue
        prov = item.prov[0]
        page = doc.pages.get(prov.page_no)
        if page is None or page.size is None or not page.size.height:
            continue
        height = page.size.height
        box = prov.bbox.to_top_left_origin(page_height=height)
        top, bottom = box.t / height, box.b / height
        seen[text] += 1
        if bottom <= _MARGIN_BAND:
            band = "top"
        elif top >= 1 - _MARGIN_BAND:
            band = "bottom"
        else:
            continue
        in_margin[text] += 1
        placed[(text, band)].append((prov.page_no, top, item))

    moved = rules
    for (text, _band), hits in placed.items():
        if in_margin[text] < 0.8 * seen[text]:
            continue
        median = statistics.median(y for _, y, _ in hits)
        aligned = [(p, item) for p, y, item in hits if abs(y - median) <= _Y_TOLERANCE]
        if len({p for p, _ in aligned}) >= _MIN_PAGES:
            moved.extend(item for _, item in aligned)
    for item in moved:
        item.content_layer = ContentLayer.FURNITURE
    return len(moved)


def _raw_headings(chunk: Any) -> tuple[str, ...]:
    meta = getattr(chunk, "meta", None)
    headings = getattr(meta, "headings", None) if meta is not None else None
    if not headings:
        return ()
    return tuple(h for h in headings if h)


def _headings_tuple(chunk: Any, furniture: frozenset[str] = frozenset()) -> tuple[str, ...]:
    return tuple(h for h in _raw_headings(chunk) if normalize_heading(h) not in furniture)


def is_navigation_heading(headings: tuple[str, ...]) -> bool:
    return bool(headings) and _NAV_HEADING_RE.match(headings[-1].strip()) is not None


def _section_path(headings: tuple[str, ...]) -> str:
    return " > ".join(headings)


def _iter_prov(chunk: Any) -> Iterable[Any]:
    meta = getattr(chunk, "meta", None)
    items = getattr(meta, "doc_items", None) if meta is not None else None
    if not items:
        return
    for item in items:
        for prov in getattr(item, "prov", None) or []:
            yield prov


def pages_from_chunk(chunk: Any) -> tuple[int | None, int | None, list[float] | None]:
    pages: list[int] = []
    bbox: list[float] | None = None
    for prov in _iter_prov(chunk):
        page_no = getattr(prov, "page_no", None)
        if page_no is not None:
            pages.append(int(page_no))
        if bbox is None:
            box = getattr(prov, "bbox", None)
            if box is not None:
                bbox = [
                    float(getattr(box, "l", 0)),
                    float(getattr(box, "t", 0)),
                    float(getattr(box, "r", 0)),
                    float(getattr(box, "b", 0)),
                ]
    if not pages:
        return None, None, bbox
    return min(pages), max(pages), bbox


def kind_from_chunk(chunk: Any) -> str:
    meta = getattr(chunk, "meta", None)
    items = getattr(meta, "doc_items", None) if meta is not None else None
    if not items:
        return "text"
    for item in items:
        type_name = type(item).__name__
        if type_name in _TABLE_TYPE_NAMES or "Table" in type_name:
            return "table"
        label = getattr(item, "label", None)
        if label is not None and "table" in str(label).lower():
            return "table"
    return "text"


def unit_from_chunk(chunk: Any) -> UnitDraft:
    page, page_end, bbox = pages_from_chunk(chunk)
    text = (getattr(chunk, "text", None) or "").strip()
    return UnitDraft(
        text=text,
        page=page,
        page_end=page_end,
        bbox=bbox,
        kind=kind_from_chunk(chunk),
    )


def inherit_missing_pages(sections: list[SectionDraft]) -> None:
    """When a unit has no prov, inherit page from the nearest preceding unit."""
    last_page: int | None = None
    last_end: int | None = None
    for section in sections:
        for unit in section.units:
            if unit.page is None:
                unit.page = last_page
                unit.page_end = last_end
            else:
                last_page = unit.page
                last_end = unit.page_end if unit.page_end is not None else unit.page


def group_into_sections(
    chunks: list[Any], furniture: set[str] | frozenset[str] = frozenset()
) -> list[SectionDraft]:
    """Maximal runs of consecutive chunks sharing an identical headings tuple."""
    furniture = frozenset(furniture)
    sections: list[SectionDraft] = []
    current: SectionDraft | None = None
    current_headings: tuple[str, ...] | None = None
    last_real: tuple[str, ...] = ()

    for chunk in chunks:
        raw = _raw_headings(chunk)
        headings = tuple(h for h in raw if normalize_heading(h) not in furniture)
        if raw and not headings:
            # The chunker's only heading context here was page furniture, which
            # means a running header interrupted a section rather than starting
            # one. Dropping it without restoring the real context would leave
            # this text in an unnamed section: still unaddressable by docs_read
            # and still unidentifiable in a citation.
            headings = last_real
        elif headings:
            last_real = headings
        unit = unit_from_chunk(chunk)
        if not unit.text:
            continue
        if is_navigation_heading(headings):
            unit.kind = "navigation"
        if current is None or headings != current_headings:
            current_headings = headings
            current = SectionDraft(
                section_path=_section_path(headings),
                heading=headings[-1] if headings else None,
                depth=len(headings),
                ordinal=len(sections),
                units=[],
            )
            sections.append(current)
        current.units.append(unit)

    inherit_missing_pages(sections)
    return sections


def _serializer_provider() -> Any:
    """Docling's chunking serializer, minus text that is not in the document.

    By default it writes a picture's predicted class ("Logo", "Engineering
    drawing") and its extracted chart table into the surrounding unit. Neither
    was printed on the page; chart values are indexed separately as
    `chart_data` units, where citation rule 4 can attribute them to the plot.
    """
    from docling_core.transforms.chunker.hierarchical_chunker import (
        ChunkingDocSerializer,
        ChunkingSerializerProvider,
    )

    class _Provider(ChunkingSerializerProvider):
        def get_serializer(self, doc: Any) -> ChunkingDocSerializer:
            ser = ChunkingDocSerializer(doc=doc)
            ser.params = ser.params.model_copy(
                update={
                    "include_picture_classification": False,
                    "enable_chart_tables": False,
                    "blocked_meta_names": {"classification", "tabular_chart"},
                }
            )
            return ser

    return _Provider()


def chunk_document(doc: Any, max_tokens: int = 384) -> list[SectionDraft]:
    """Run HybridChunker and group into sections."""
    from docling.chunking import HybridChunker

    chunker = HybridChunker(
        max_tokens=max_tokens,
        merge_peers=True,
        repeat_table_header=True,
        serializer_provider=_serializer_provider(),
    )
    chunks = list(chunker.chunk(dl_doc=doc))
    return group_into_sections(chunks, furniture_headings(doc))
