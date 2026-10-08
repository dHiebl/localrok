"""Chart CSV lives in `PictureItem.meta`, not in the deprecated `annotations`
list, which migrates one-way into `meta` and so is always empty for charts."""

from __future__ import annotations

import warnings

from localrok.figures import _meta_chart_csv, _tabledata_to_csv


def _table_data(rows: list[list[str]]):
    from docling_core.types.doc import TableCell, TableData

    cells = [
        TableCell(
            text=text,
            start_row_offset_idx=r,
            end_row_offset_idx=r + 1,
            start_col_offset_idx=c,
            end_col_offset_idx=c + 1,
            column_header=(r == 0),
        )
        for r, row in enumerate(rows)
        for c, text in enumerate(row)
    ]
    return TableData(table_cells=cells, num_rows=len(rows), num_cols=len(rows[0]))


def test_tabledata_to_csv_preserves_axes():
    csv_text = _tabledata_to_csv(_table_data([["Time [h]", "Knowledge_Error"], ["0", "0.031"]]))
    assert csv_text == "Time [h],Knowledge_Error\r\n0,0.031"


def test_reads_chart_csv_from_picture_meta():
    from docling_core.types.doc import PictureItem, PictureMeta, TabularChartMetaField

    item = PictureItem(self_ref="#/pictures/0")
    item.meta = PictureMeta(
        tabular_chart=TabularChartMetaField(
            chart_data=_table_data([["Time [h]", "Bus Voltage [V]"], ["0", "28.1"]])
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert item.get_annotations() == [], "charts never land in the legacy annotations list"
    assert _meta_chart_csv(item) == "Time [h],Bus Voltage [V]\r\n0,28.1"


def test_no_chart_meta_is_not_an_error():
    from docling_core.types.doc import PictureItem

    assert _meta_chart_csv(PictureItem(self_ref="#/pictures/0")) is None
