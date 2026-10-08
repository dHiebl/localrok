"""In-memory drafts produced by chunking / figure extraction before indexing."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UnitDraft:
    text: str
    page: int | None
    page_end: int | None
    bbox: list[float] | None
    kind: str  # text|table|caption|chart_data|figure_labels|navigation


@dataclass
class SectionDraft:
    section_path: str
    heading: str | None
    depth: int
    ordinal: int
    units: list[UnitDraft] = field(default_factory=list)

    @property
    def page_start(self) -> int | None:
        pages = [u.page for u in self.units if u.page is not None]
        return min(pages) if pages else None

    @property
    def page_end(self) -> int | None:
        ends = [
            u.page_end if u.page_end is not None else u.page
            for u in self.units
            if u.page is not None or u.page_end is not None
        ]
        return max(ends) if ends else None

    @property
    def char_count(self) -> int:
        return sum(len(u.text) for u in self.units)


@dataclass
class FigureDraft:
    page: int | None
    rel_path: str
    caption: str | None
    class_name: str | None
    confidence: float | None
    label_text: str | None
    chart_csv: str | None
    kept: int  # 0 = triaged out
    bbox: list[float] | None = None


@dataclass
class DocumentMeta:
    filename: str
    stem: str
    title: str | None
    doc_ident: str | None
    doc_type: str
    version: str | None
    page_count: int | None
    has_provenance: int
    source_sha256: str
    docling_version: str
