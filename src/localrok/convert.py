"""Stage 1: convert source documents to canonical DoclingDocument JSON."""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

from localrok.config import Config, IngestConfig
from localrok.models import DocumentMeta

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html", ".htm", ".pptx", ".md"}

# Generic identifiers such as DOC-POLICY-001 and version labels such as v2.1.
_IDENTIFIER_RE = re.compile(r"^([A-Za-z][A-Za-z0-9]*(?:[._-][A-Za-z0-9]+)*[._-][0-9]{2,})(?=\s|$)")
_VERSION_RE = re.compile(
    r"(?:^|[ _-])v[.]?\s*([0-9]+(?:\.[0-9]+)*(?:-[A-Za-z0-9]+)*)", re.IGNORECASE
)
_BARE_VERSION_RE = re.compile(r"(?:^|[ _-])([0-9]+\.[0-9]+(?:\.[0-9]+)*)(?=$|[ _-])")
_TITLE_TYPE_KEYWORDS = (
    ("meeting minutes", "meeting_notes"),
    ("meeting notes", "meeting_notes"),
    ("specification", "specification"),
    ("datasheet", "datasheet"),
    ("procedure", "procedure"),
    ("handbook", "handbook"),
    ("manual", "manual"),
    ("guide", "guide"),
    ("policy", "policy"),
    ("agreement", "agreement"),
    ("report", "report"),
    ("presentation", "presentation"),
    ("paper", "article"),
    ("concept", "concept"),
)


class GpuBusyError(RuntimeError):
    """No configured NVIDIA device is available for document conversion."""


_GPU_FATAL_MARKERS = (
    "cuda out of memory",
    "cublas_status_alloc_failed",
    "cublas_status_not_initialized",
    "cudnn_status_alloc_failed",
    "cuda error",
    "no kernel image is available",
)


def is_gpu_fatal(exc: BaseException) -> bool:
    """True if `exc` poisoned the CUDA context, so the converter cannot be reused.

    Docling wraps model failures in its own exceptions, so walk the chain.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if type(cur).__name__ in {"OutOfMemoryError", "CudaError"}:
            return True
        text = f"{type(cur).__name__}: {cur}".casefold()
        if any(marker in text for marker in _GPU_FATAL_MARKERS):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _nvidia_smi(*, gpu: str | None = None, compute_apps: str | None = None) -> str:
    if gpu is not None:
        args = ["nvidia-smi", f"--query-gpu={gpu}", "--format=csv,noheader,nounits"]
    elif compute_apps is not None:
        args = [
            "nvidia-smi",
            f"--query-compute-apps={compute_apps}",
            "--format=csv,noheader",
        ]
    else:
        raise ValueError("pass gpu= or compute_apps=")
    return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL)


def _parse_smi_rows(raw: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("no running"):
            continue
        rows.append([p.strip() for p in line.split(",")])
    return rows


def _gpu_snapshot() -> tuple[list[dict], dict[str, list[str]]]:
    """Return (gpus, uuid -> pids). gpus have index, uuid, free_mib."""
    gpu_raw = _nvidia_smi(gpu="index,uuid,memory.free")
    gpus: list[dict] = []
    for parts in _parse_smi_rows(gpu_raw):
        if len(parts) < 3:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "uuid": parts[1].casefold(),
                    "free_mib": int(float(parts[2])),
                }
            )
        except ValueError:
            continue

    busy: dict[str, list[str]] = {}
    try:
        app_raw = _nvidia_smi(compute_apps="gpu_uuid,pid")
    except subprocess.CalledProcessError:
        print(
            "nvidia-smi compute-apps query failed; occupancy unknown — using free memory only",
            file=sys.stderr,
        )
        return gpus, busy
    for parts in _parse_smi_rows(app_raw):
        if len(parts) < 2:
            continue
        uuid, pid = parts[0].casefold(), parts[1]
        if pid:
            busy.setdefault(uuid, []).append(pid)
    return gpus, busy


def pick_free_gpu(
    gpu_ids: list[int],
    *,
    min_free_mib: int = 8192,
) -> str:
    """Pick an idle GPU from the configured IDs, or all visible GPUs."""
    try:
        gpus, busy = _gpu_snapshot()
    except FileNotFoundError:
        print("nvidia-smi not found; Docling will run on CPU", file=sys.stderr)
        return "cpu"
    except subprocess.CalledProcessError as exc:
        raise GpuBusyError(f"nvidia-smi failed: {exc}") from exc

    by_idx = {g["index"]: g for g in gpus}
    allowed = gpu_ids or sorted(by_idx)
    reasons: list[str] = []
    candidates: list[tuple[int, int]] = []
    for idx in allowed:
        gpu = by_idx.get(idx)
        if gpu is None:
            reasons.append(f"  GPU {idx}: not visible to nvidia-smi")
            continue
        pids = busy.get(gpu["uuid"], [])
        if pids:
            reasons.append(f"  GPU {idx}: busy (pid {', '.join(pids)}), {gpu['free_mib']} MiB free")
            continue
        if gpu["free_mib"] < min_free_mib:
            reasons.append(f"  GPU {idx}: {gpu['free_mib']} MiB free, need ≥ {min_free_mib} MiB")
            continue
        candidates.append((gpu["free_mib"], idx))

    if not candidates:
        raise GpuBusyError(
            "No ingest GPU available "
            f"(need ≥ {min_free_mib} MiB free and no other compute jobs).\n" + "\n".join(reasons)
        )

    best_free, best_idx = max(candidates)
    print(
        f"Selected GPU {best_idx} ({best_free} MiB free) from allowed {allowed}",
        file=sys.stderr,
    )
    return f"cuda:{best_idx}"


def resolve_device(cfg: IngestConfig, *, do_charts: bool = False) -> str:
    """Select Apple MPS, NVIDIA CUDA, or CPU without machine-specific rules."""
    if cfg.device == "cpu":
        return "cpu"
    import torch

    cuda_available = torch.cuda.is_available()
    mps = getattr(torch.backends, "mps", None)
    mps_available = mps is not None and mps.is_available()
    if cfg.device == "mps":
        if not mps_available:
            raise RuntimeError("Apple MPS is unavailable; set ingest.device to cpu or auto")
        return "mps"
    if cfg.device.startswith("cuda") or (cfg.device == "auto" and cuda_available):
        if not cuda_available:
            raise RuntimeError("CUDA is unavailable; set ingest.device to cpu or auto")
        ids = [int(cfg.device.split(":")[1])] if ":" in cfg.device else cfg.gpu_ids
        minimum = cfg.gpu_min_free_mib_charts if do_charts else cfg.gpu_min_free_mib
        device = pick_free_gpu(ids, min_free_mib=minimum)
        if device == "cpu" and cfg.device != "auto":
            raise RuntimeError("NVIDIA device inspection failed; set ingest.device to cpu or auto")
        return device
    return "mps" if mps_available else "cpu"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def list_source_files(source_dir: Path, pattern: str | None = None) -> list[Path]:
    files = sorted(
        p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if pattern:
        import fnmatch

        files = [p for p in files if fnmatch.fnmatch(p.name, pattern)]
    return files


def docling_version() -> str:
    try:
        import docling

        return getattr(docling, "__version__", "unknown")
    except ImportError:
        return "unknown"


def parse_document_metadata(
    filename: str,
    stem: str,
    title: str | None,
    page_count: int | None,
    source_sha256: str,
    version: str,
    has_provenance: int,
) -> DocumentMeta:
    hay = re.sub(r"[_-]+", " ", f"{title or ''} {stem}".casefold())
    doc_type = next(
        (kind for keyword, kind in _TITLE_TYPE_KEYWORDS if re.search(r"\b" + keyword + r"\b", hay)),
        "unknown",
    )
    ident = _IDENTIFIER_RE.match(stem)
    version_match = _VERSION_RE.search(stem) or _BARE_VERSION_RE.search(stem)
    doc_ident = ident.group(1) if ident else None
    doc_version = version_match.group(1) if version_match else None

    return DocumentMeta(
        filename=filename,
        stem=stem,
        title=title or stem,
        doc_ident=doc_ident,
        doc_type=doc_type,
        version=doc_version,
        page_count=page_count,
        has_provenance=has_provenance,
        source_sha256=source_sha256,
        docling_version=version,
    )


_chart_limits_applied: tuple[int, int] | None = None


def apply_chart_limits(max_new_tokens: int, batch_size: int) -> None:
    """Bound chart generation length and batch size to limit memory use.

    Docling derives its generation limit from the processor's model maximum,
    which can be effectively unbounded. Batch size is read from the model class.
    """
    global _chart_limits_applied
    if _chart_limits_applied == (max_new_tokens, batch_size):
        return
    try:
        from docling.models.stages.chart_extraction.granite_vision import (
            _BaseChartExtractionModelGraniteVision as ChartModel,
        )
    except ImportError as exc:
        print(f"Cannot bound chart extraction ({exc}); refusing to run charts", file=sys.stderr)
        raise

    base_init = getattr(ChartModel.__init__, "__localrok_wrapped__", None) or ChartModel.__init__

    def __init__(self, *args, **kwargs):
        base_init(self, *args, **kwargs)
        if getattr(self, "_model_max_length", 0) > max_new_tokens:
            self._model_max_length = max_new_tokens

    __init__.__localrok_wrapped__ = base_init
    ChartModel.__init__ = __init__
    ChartModel.elements_batch_size = batch_size
    _chart_limits_applied = (max_new_tokens, batch_size)


def log_chart_batches(cls) -> None:
    """Announce each chart batch. Docling is silent during enrichment, so a slow
    document is indistinguishable from a hung one without this."""
    import time

    base_call = getattr(cls.__call__, "__localrok_wrapped__", None) or cls.__call__

    def __call__(self, doc, element_batch):
        batch = list(element_batch)
        print(f"    charts: {len(batch)} in this batch", file=sys.stderr, flush=True)
        started = time.monotonic()
        yield from base_call(self, doc, batch)
        print(f"    charts: batch done in {time.monotonic() - started:.1f}s", file=sys.stderr)

    __call__.__localrok_wrapped__ = base_call
    cls.__call__ = __call__


def make_converter(cfg: IngestConfig, *, do_charts: bool, do_figures: bool):
    """Build one DocumentConverter. Reuse it across files — models stay loaded."""
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption

    charts_on = do_charts and do_figures
    device = resolve_device(cfg, do_charts=charts_on)
    opts = PdfPipelineOptions(
        do_table_structure=True,
        do_code_enrichment=cfg.do_code,
        do_formula_enrichment=cfg.do_formulas,
        do_picture_classification=do_figures,
        do_chart_extraction=charts_on,
        generate_picture_images=do_figures,
        images_scale=cfg.images_scale,
        accelerator_options=AcceleratorOptions(device=device),
    )
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    opts.table_structure_options.do_cell_matching = True

    if charts_on:
        try:
            from docling.datamodel.chart_extraction_options import (
                ChartExtractionModelKind,
                ChartExtractionModelOptions,
            )
            from docling.models.stages.chart_extraction.granite_vision import (
                ChartExtractionModelGraniteVisionV4,
            )

            # Never enable chart extraction that we have not bounded first.
            apply_chart_limits(cfg.chart_max_new_tokens, cfg.chart_batch_size)
            log_chart_batches(ChartExtractionModelGraniteVisionV4)
            opts.chart_extraction_options = ChartExtractionModelOptions(
                model=ChartExtractionModelKind.GRANITE_VISION_V4,
                chart2csv=True,
                chart2code=False,
                chart2summary=False,
            )
            print(
                f"Chart extraction on: max_new_tokens={cfg.chart_max_new_tokens}, "
                f"batch={cfg.chart_batch_size}",
                file=sys.stderr,
            )
        except (ImportError, AttributeError) as exc:
            print(f"Chart extraction unavailable ({exc}); disabling charts", file=sys.stderr)
            opts.do_chart_extraction = False

    # Explicitly never set do_picture_description / enable_remote_services.
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def convert_file(
    source: Path,
    json_path: Path,
    cfg: Config,
    *,
    do_charts: bool = False,
    do_figures: bool = True,
    converter=None,
):
    """Convert one source file to DoclingDocument JSON. Returns the document.

    Pass a shared `converter` from `make_converter` during batch ingest so
    layout/OCR/chart weights are not reloaded for every PDF.
    """
    from docling_core.types.doc import DoclingDocument

    json_path.parent.mkdir(parents=True, exist_ok=True)
    if converter is None:
        converter = make_converter(cfg.ingest, do_charts=do_charts, do_figures=do_figures)
    result = converter.convert(str(source))
    doc: DoclingDocument = result.document
    doc.save_as_json(str(json_path))
    return doc


def load_docling_json(json_path: Path):
    from docling_core.types.doc import DoclingDocument

    return DoclingDocument.load_from_json(str(json_path))


def document_page_count(doc) -> int | None:
    pages = getattr(doc, "pages", None)
    if pages is None:
        return None
    try:
        return len(pages)
    except TypeError:
        return None


def first_heading_title(doc) -> str | None:
    """Document title: first depth-1 heading if present."""
    try:
        for item, level in doc.iterate_items():
            label = getattr(item, "label", None)
            text = getattr(item, "text", None)
            if not text:
                continue
            name = getattr(label, "name", str(label) if label is not None else "")
            if "title" in name.lower() or (level == 1 and "header" in name.lower()):
                return text.strip()
            if name.lower() in {"section_header", "title"} and (level or 1) == 1:
                return text.strip()
    except Exception:
        pass
    # Fallback: first heading-like text
    try:
        texts = [
            item.text.strip() for item, _ in doc.iterate_items() if getattr(item, "text", None)
        ]
        return texts[0] if texts else None
    except Exception:
        return None
