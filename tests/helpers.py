"""Shared helpers for building a tiny on-disk LocalRök index in tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from localrok.db import connect
from localrok.figures import attach_figures_to_sections
from localrok.index import index_document
from localrok.models import DocumentMeta, FigureDraft, SectionDraft, UnitDraft


def make_unit(text: str, page: int | None = 1, kind: str = "text") -> UnitDraft:
    return UnitDraft(text=text, page=page, page_end=page, bbox=None, kind=kind)


def sample_sections() -> list[SectionDraft]:
    return [
        SectionDraft(
            section_path="Device User Manual > 4 Interfaces > 4.3 Connector Interface",
            heading="4.3 Connector Interface",
            depth=3,
            ordinal=0,
            units=[
                make_unit(
                    "J5 provides the MAINTPORT configuration interface. Pin assignment "
                    "is given in Figure 4-2. Note that pin 1 is nearest the board edge.",
                    page=37,
                )
            ],
        ),
        SectionDraft(
            section_path="Device User Manual > 7 Electrical Characteristics",
            heading="7 Electrical Characteristics",
            depth=2,
            ordinal=1,
            units=[
                make_unit(
                    "For the 8-cell configuration while idle the undervoltage "
                    "threshold is 21.0 V. Discharge must stop at this limit.",
                    page=21,
                ),
                make_unit(
                    "For the 4-cell variant the undervoltage threshold is 10.5 V. "
                    "Do not apply the 8-cell figure to 4-cell packs.",
                    page=22,
                ),
            ],
        ),
        SectionDraft(
            section_path="Device User Manual > 5 Power Control > 5.2 Watchdog Handling",
            heading="5.2 Watchdog Handling",
            depth=3,
            ordinal=2,
            units=[
                make_unit(
                    "If the watchdog expires, the controller initiates a warm reset. "
                    "The watchdog is serviced by the supervisor task every 500 ms.",
                    page=37,
                    kind="text",
                )
            ],
        ),
    ]


def sample_figures() -> list[FigureDraft]:
    return [
        FigureDraft(
            page=37,
            rel_path="figures/device-manual-v4.0.0/fig-37-1.png",
            caption="Figure 4-2 — J5 power connector pin assignment",
            class_name="flow_chart",
            confidence=0.9,
            label_text="J5 pin 1 GND pin 2 VBAT",
            chart_csv=None,
            kept=1,
        )
    ]


def populate_sample(conn: sqlite3.Connection) -> None:
    meta = DocumentMeta(
        filename="device-manual-v4.0.0.pdf",
        stem="device-manual-v4.0.0",
        title="Device User Manual",
        doc_ident="DOC-MANUAL-001",
        doc_type="manual",
        version="4.0.0",
        page_count=44,
        has_provenance=1,
        source_sha256="abc",
        docling_version="2.120.3",
    )
    figs = sample_figures()
    sections = sample_sections()
    assignment = attach_figures_to_sections(figs, sections)
    index_document(conn, meta, sections, figs, assignment)

    ops = DocumentMeta(
        filename="operations-manual.pdf",
        stem="operations-manual",
        title="Operations Manual",
        doc_ident="ops",
        doc_type="procedure",
        version="1.0",
        page_count=20,
        has_provenance=1,
        source_sha256="def",
        docling_version="2.120.3",
    )
    ops_sections = [
        SectionDraft(
            section_path="Operations Manual > 8 Recovery > 8.1 Watchdog Recovery",
            heading="8.1 Watchdog Recovery",
            depth=3,
            ordinal=0,
            units=[
                make_unit(
                    "If the watchdog trips, follow the hardware manual: expect a warm "
                    "reset and resume monitoring after 500 ms.",
                    page=12,
                )
            ],
        )
    ]
    index_document(conn, ops, ops_sections, [], {})
    conn.commit()


def tmp_conn(tmp_path: Path) -> sqlite3.Connection:
    conn = connect(tmp_path / "localrok.db")
    populate_sample(conn)
    return conn
