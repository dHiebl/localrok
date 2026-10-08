"""Stage 2b: pictures → PNG + caption + chart CSV + figure units."""

from __future__ import annotations

import csv
import io
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

from localrok.config import IngestConfig
from localrok.models import FigureDraft, SectionDraft, UnitDraft

TRIAGE_DROP = {"logo", "icon", "signature", "stamp"}
CAPTION_RE = re.compile(r"^(Figure|Fig\.|Abbildung|Bild)\b", re.IGNORECASE)


def figure_url(rel_path: str | None) -> str | None:
    """Root-relative, percent-encoded URL for a stored figure path.

    Figures live under the source filename, so nearly every path contains
    spaces. Markdown ends a bare link destination at the first space, so the
    raw path turns `![cap](figures/A B/fig.png)` into literal text in the
    client rather than an image. Emit the finished URL so the model copies a
    string it cannot get wrong. `web/api.py` serves this prefix.
    """
    if not rel_path:
        return None
    return "/" + quote(str(rel_path).strip("/"))


def figure_alt(caption: str | None) -> str:
    """Caption safe to use as markdown alt text."""
    return (caption or "figure").replace("[", "(").replace("]", ")")


def _bbox_tuple(box: Any) -> tuple[float, float, float, float] | None:
    if box is None:
        return None
    left = float(getattr(box, "l", 0))
    top = float(getattr(box, "t", 0))
    right = float(getattr(box, "r", 0))
    bottom = float(getattr(box, "b", 0))
    return left, top, right, bottom


def bbox_contains(outer: Any, inner: Any, tol: float = 2.0) -> bool:
    o = _bbox_tuple(outer)
    i = _bbox_tuple(inner)
    if o is None or i is None:
        return False
    o_l, o_t, o_r, o_b = o
    i_l, i_t, i_r, i_b = i
    o_top, o_bot = min(o_t, o_b), max(o_t, o_b)
    i_top, i_bot = min(i_t, i_b), max(i_t, i_b)
    return i_l >= o_l - tol and i_r <= o_r + tol and i_top >= o_top - tol and i_bot <= o_bot + tol


def _vertical_gap(picture_box: Any, text_box: Any) -> float | None:
    """Origin-agnostic gap between picture and a candidate caption below it."""
    p = _bbox_tuple(picture_box)
    t = _bbox_tuple(text_box)
    if p is None or t is None:
        return None
    p_top, p_bot = min(p[1], p[3]), max(p[1], p[3])
    t_top, t_bot = min(t[1], t[3]), max(t[1], t[3])
    # "below" = text sits just outside the picture on the vertical axis
    if t_top >= p_bot:
        return t_top - p_bot
    if p_top >= t_bot:
        return p_top - t_bot
    return 0.0


def _top_classification(item: Any) -> tuple[str | None, float | None]:
    getter = getattr(item, "get_annotations", None)
    anns = getter() if callable(getter) else getattr(item, "annotations", None) or []
    best_name: str | None = None
    best_conf: float | None = None
    for ann in anns:
        classes = getattr(ann, "predicted_classes", None) or getattr(ann, "classes", None) or []
        for cls in classes:
            name = getattr(cls, "class_name", None) or getattr(cls, "name", None)
            conf = getattr(cls, "confidence", None)
            if name is None:
                continue
            if best_conf is None or (conf is not None and conf > best_conf):
                best_name, best_conf = str(name), float(conf) if conf is not None else None
    return best_name, best_conf


def _is_chart_ann(ann: Any) -> bool:
    name = type(ann).__name__
    return "Chart" in name or getattr(ann, "kind", "") == "chart_extraction"


def _chart_to_csv(ann: Any) -> str | None:
    for attr in ("csv", "chart_csv"):
        val = getattr(ann, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    to_csv = getattr(ann, "to_csv", None)
    if callable(to_csv):
        try:
            return str(to_csv()).strip()
        except Exception:
            pass
    # Common granite-vision chart payloads: titled series of x/y values
    title = getattr(ann, "title", None) or ""
    charts = getattr(ann, "charts", None) or getattr(ann, "data", None)
    if charts is None:
        x_vals = getattr(ann, "x_values", None)
        y_vals = getattr(ann, "y_values", None)
        if x_vals is not None and y_vals is not None:
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(["x", "y"])
            for x, y in zip(x_vals, y_vals):
                writer.writerow([x, y])
            return buf.getvalue().strip()
        return None
    try:
        buf = io.StringIO()
        writer = csv.writer(buf)
        if title:
            writer.writerow([title])
        if isinstance(charts, list) and charts:
            first = charts[0]
            if isinstance(first, dict):
                writer.writerow(list(first.keys()))
                for row in charts:
                    writer.writerow(list(row.values()) if isinstance(row, dict) else [row])
            else:
                for row in charts:
                    writer.writerow([row])
        return buf.getvalue().strip() or None
    except Exception:
        return None


def _tabledata_to_csv(table_data: Any) -> str | None:
    cells = getattr(table_data, "table_cells", None) or []
    n_rows = int(getattr(table_data, "num_rows", 0) or 0)
    n_cols = int(getattr(table_data, "num_cols", 0) or 0)
    if not cells or n_rows <= 0 or n_cols <= 0:
        return None
    grid = [[""] * n_cols for _ in range(n_rows)]
    for cell in cells:
        row = getattr(cell, "start_row_offset_idx", None)
        col = getattr(cell, "start_col_offset_idx", None)
        if row is None or col is None or not (0 <= row < n_rows and 0 <= col < n_cols):
            continue
        grid[row][col] = str(getattr(cell, "text", "") or "")
    buf = io.StringIO()
    csv.writer(buf).writerows(grid)
    return buf.getvalue().strip() or None


def _meta_chart_csv(item: Any) -> str | None:
    """Read chart CSV from `meta`, where Docling's chart-extraction model writes it.

    `PictureItem.annotations` is deprecated and migrates one-way into `meta`, so
    nothing ever appears there for charts — only the legacy classifier output does.
    """
    meta = getattr(item, "meta", None)
    tabular = getattr(meta, "tabular_chart", None) if meta is not None else None
    chart_data = getattr(tabular, "chart_data", None) if tabular is not None else None
    if chart_data is None:
        return None
    try:
        return _tabledata_to_csv(chart_data)
    except Exception:
        return None


def _picture_page(item: Any) -> int | None:
    prov = getattr(item, "prov", None) or []
    pages = [int(p.page_no) for p in prov if getattr(p, "page_no", None) is not None]
    return min(pages) if pages else None


def _picture_bbox(item: Any) -> Any:
    prov = getattr(item, "prov", None) or []
    return getattr(prov[0], "bbox", None) if prov else None


def _iter_text_items(doc: Any) -> list[Any]:
    """All text items, including those nested inside pictures.

    `iterate_items` defaults to `traverse_pictures=False`, which skips exactly
    the text we are after: the labels drawn inside a diagram. Without this the
    bbox-containment pass has no candidates and every figure ends up with an
    empty `label_text`.
    """
    items = []
    try:
        try:
            walk = doc.iterate_items(traverse_pictures=True)
        except TypeError:  # older docling without the parameter
            walk = doc.iterate_items()
        for item, _level in walk:
            text = getattr(item, "text", None)
            if text and str(text).strip():
                items.append(item)
    except Exception as exc:
        print(f"iterate_items failed while collecting labels: {exc}", file=sys.stderr)
    return items


def _pages_with_text(doc: Any) -> set[int]:
    """Pages carrying any extracted text, in any content layer.

    Body text alone is the wrong test: a page of nothing but tables under a
    running header (furniture) has a text layer and is not a scan.
    """
    pages: set[int] = set()
    for item in getattr(doc, "texts", None) or []:
        if not (getattr(item, "text", None) or "").strip():
            continue
        for prov in getattr(item, "prov", None) or []:
            page = getattr(prov, "page_no", None)
            if page is not None:
                pages.add(int(page))
    return pages


def _label_text(item: Any, text_items: list[Any], page: int | None) -> str | None:
    pic_box = _picture_bbox(item)
    if pic_box is None:
        return None
    labels: list[tuple[float, float, str]] = []
    for t_item in text_items:
        text = (getattr(t_item, "text", None) or "").strip()
        if not text:
            continue
        for prov in getattr(t_item, "prov", None) or []:
            if page is not None and getattr(prov, "page_no", None) != page:
                continue
            box = getattr(prov, "bbox", None)
            if box is not None and bbox_contains(pic_box, box):
                tb = _bbox_tuple(box)
                reading = (tb[1] if tb else 0.0, tb[0] if tb else 0.0)
                labels.append((reading[0], reading[1], text))
    labels.sort()
    if not labels:
        return None
    # Preserve reading order; join uniquely to avoid header spam
    seen: set[str] = set()
    ordered: list[str] = []
    for _, _, text in labels:
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    return " ".join(ordered) if ordered else None


def _nearby_caption(item: Any, text_items: list[Any], page: int | None) -> str | None:
    pic_box = _picture_bbox(item)
    if pic_box is None:
        return None
    best: tuple[float, str] | None = None
    for t_item in text_items:
        text = (getattr(t_item, "text", None) or "").strip()
        if not text or not CAPTION_RE.match(text):
            continue
        for prov in getattr(t_item, "prov", None) or []:
            if page is not None and getattr(prov, "page_no", None) != page:
                continue
            gap = _vertical_gap(pic_box, getattr(prov, "bbox", None))
            if gap is None or gap > 80:
                continue
            if best is None or gap < best[0]:
                best = (gap, text)
    return best[1] if best else None


def _downscale(image: Any, max_edge: int) -> Any:
    w, h = image.size
    longest = max(w, h)
    if longest <= max_edge:
        return image
    scale = max_edge / longest
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    resample = getattr(getattr(image, "Resampling", None), "LANCZOS", None)
    if resample is None:
        try:
            from PIL import Image as PILImage

            resample = PILImage.Resampling.LANCZOS
        except Exception:
            resample = 1
    return image.resize(new_size, resample)


def _save_png(item: Any, doc: Any, dest: Path, max_edge: int) -> bool:
    getter = getattr(item, "get_image", None)
    if not callable(getter):
        return False
    try:
        image = getter(doc)
    except Exception as exc:
        print(f"get_image failed: {exc}", file=sys.stderr)
        return False
    if image is None:
        return False
    try:
        image = _downscale(image, max_edge)
        dest.parent.mkdir(parents=True, exist_ok=True)
        image.save(dest, format="PNG")
        return True
    except Exception as exc:
        print(f"PNG save failed for {dest}: {exc}", file=sys.stderr)
        return False


def attach_figures_to_sections(
    figures: list[FigureDraft], sections: list[SectionDraft]
) -> dict[int, int]:
    """Return map figure_index → section ordinal. Mutates sections with extra units."""
    assignment: dict[int, int] = {}
    for fig_i, fig in enumerate(figures):
        if not fig.kept:
            continue
        page = fig.page
        best_ord: int | None = None
        for section in sections:
            start = section.page_start
            end = section.page_end
            if page is None:
                continue
            if start is not None and start > page:
                continue
            if end is not None and end < page:
                continue
            if start is not None and start <= page:
                if best_ord is None or section.ordinal >= best_ord:
                    best_ord = section.ordinal
        if best_ord is None and sections:
            # Fall back to last section whose start <= page, else first
            candidates = [
                s
                for s in sections
                if s.page_start is not None and page is not None and s.page_start <= page
            ]
            best_ord = candidates[-1].ordinal if candidates else 0
        if best_ord is None:
            continue
        assignment[fig_i] = best_ord
        section = sections[best_ord]
        section.units.extend(figure_units(fig))
    return assignment


def figure_units(fig: FigureDraft) -> list[UnitDraft]:
    units: list[UnitDraft] = []
    if fig.caption:
        units.append(
            UnitDraft(
                text=fig.caption,
                page=fig.page,
                page_end=fig.page,
                bbox=fig.bbox,
                kind="caption",
            )
        )
    if fig.label_text:
        units.append(
            UnitDraft(
                text=fig.label_text,
                page=fig.page,
                page_end=fig.page,
                bbox=fig.bbox,
                kind="figure_labels",
            )
        )
    if fig.chart_csv:
        prefix = (fig.caption or "Figure data").split("\n")[0]
        units.append(
            UnitDraft(
                text=f"{prefix} data:\n{fig.chart_csv}",
                page=fig.page,
                page_end=fig.page,
                bbox=fig.bbox,
                kind="chart_data",
            )
        )
    return units


def extract_figures(
    doc: Any,
    stem: str,
    figures_dir: Path,
    derived_root: Path,
    cfg: IngestConfig,
    *,
    scanned_pages: list[int] | None = None,
) -> list[FigureDraft]:
    """Extract kept/triaged figures. scanned_pages collects pages with no text items."""
    pictures = list(getattr(doc, "pictures", None) or [])
    text_items = _iter_text_items(doc)
    text_pages = _pages_with_text(doc)
    drafts: list[FigureDraft] = []
    scanned: set[int] = set()

    for idx, item in enumerate(pictures):
        page = _picture_page(item)
        class_name, confidence = _top_classification(item)
        caption = None
        cap_fn = getattr(item, "caption_text", None)
        if callable(cap_fn):
            try:
                caption = cap_fn(doc) or None
            except Exception:
                caption = None
        if caption:
            caption = str(caption).strip() or None
        if not caption:
            caption = _nearby_caption(item, text_items, page)

        drop = class_name in TRIAGE_DROP or (class_name == "natural_image" and not caption)
        kept = 0 if drop else 1

        chart_csv = _meta_chart_csv(item)
        if not chart_csv:
            getter = getattr(item, "get_annotations", None)
            anns = getter() if callable(getter) else []
            for ann in anns:
                if _is_chart_ann(ann):
                    chart_csv = _chart_to_csv(ann)
                    if chart_csv:
                        break

        label_text = None
        if page is not None and page not in text_pages:
            scanned.add(page)
        else:
            label_text = _label_text(item, text_items, page)

        rel = ""
        box = _picture_bbox(item)
        bbox_list = list(_bbox_tuple(box)) if box is not None and _bbox_tuple(box) else None

        if kept:
            filename = f"fig-{page if page is not None else 'x'}-{idx}.png"
            dest = figures_dir / stem / filename
            saved = _save_png(item, doc, dest, cfg.figure_max_edge_px)
            if saved:
                try:
                    rel = str(dest.relative_to(derived_root))
                except ValueError:
                    rel = str(dest)
            else:
                kept = 0
                rel = str(Path("figures") / stem / filename)

        drafts.append(
            FigureDraft(
                page=page,
                rel_path=rel,
                caption=caption,
                class_name=class_name,
                confidence=confidence,
                label_text=label_text,
                chart_csv=chart_csv,
                kept=kept,
                bbox=bbox_list,
            )
        )

    if scanned_pages is not None:
        scanned_pages.extend(sorted(scanned))
    _warn_missing_chart_data(drafts)
    return drafts


CHART_CLASSES = {"bar_chart", "line_chart", "pie_chart", "scatter_chart", "stacked_bar_chart"}


def _warn_missing_chart_data(drafts: list[FigureDraft]) -> None:
    """Charts with no extracted CSV are numbers the agent will never see.

    Docling reports nothing when the chart-extraction model is unavailable, so
    without this the gap is silent and the figure looks merely uninformative.
    """
    charts = [d for d in drafts if (d.class_name or "") in CHART_CLASSES and d.kept]
    missing = [d for d in charts if not d.chart_csv]
    if not missing:
        return
    pages = ", ".join(f"p{d.page}" if d.page is not None else "p?" for d in missing)
    print(
        f"WARNING: {len(missing)}/{len(charts)} kept charts yielded no CSV ({pages}). "
        "Their values are not searchable. If do_charts is enabled, check that the "
        "chart-extraction model is downloaded (`docling-tools models download`).",
        file=sys.stderr,
    )
