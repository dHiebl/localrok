"""A CUDA OOM poisons the converter, so the batch must stop rather than feed
the next PDF to a dead GPU. See `convert.apply_chart_limits` for the cause."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from localrok import cli
from localrok.convert import is_gpu_fatal

REPO_ROOT = Path(__file__).resolve().parent.parent


def _chained_oom() -> Exception:
    """A torch OOM as Docling re-raises it, wrapped in a pipeline error."""
    try:
        try:
            raise MemoryError("CUDA out of memory. Tried to allocate 1.50 GiB")
        except MemoryError as inner:
            raise RuntimeError("StandardPdfPipeline failed") from inner
    except RuntimeError as outer:
        return outer


@pytest.mark.parametrize(
    "exc, fatal",
    [
        (_chained_oom(), True),
        (RuntimeError("cublas runtime error : CUBLAS_STATUS_ALLOC_FAILED"), True),
        (RuntimeError("CUDA error: device-side assert triggered"), True),
        (ValueError("could not parse page 3"), False),
        (FileNotFoundError("nvidia-smi"), False),
    ],
)
def test_is_gpu_fatal(exc, fatal):
    assert is_gpu_fatal(exc) is fatal


def _ingest_root(tmp_path: Path, names: list[str]) -> Path:
    shutil.copy(REPO_ROOT / "config.toml", tmp_path / "config.toml")
    source = tmp_path / "corpus" / "source"
    source.mkdir(parents=True)
    for name in names:
        (source / name).write_bytes(b"%PDF-1.7\n")
    return tmp_path


def _run_ingest(root: Path, failure: Exception, fails_on: str) -> list[str]:
    """Run cmd_ingest with conversion stubbed out. Returns files attempted."""
    attempted: list[str] = []

    def fake_convert(source, json_path, cfg, **_kwargs):
        attempted.append(source.name)
        if source.name == fails_on:
            raise failure
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text("{}", encoding="utf-8")
        return object()

    args = cli.build_parser().parse_args(["--root", str(root), "ingest"])
    with (
        patch.object(cli, "make_converter", return_value=object()),
        patch.object(cli, "convert_file", side_effect=fake_convert),
        patch.object(cli, "_index_from_doc"),
        patch.object(cli, "write_corpus"),
        patch.object(cli, "write_structure_report"),
    ):
        code = cli.cmd_ingest(args)
    return attempted, code


def test_gpu_oom_stops_the_batch(tmp_path):
    root = _ingest_root(tmp_path, ["a.pdf", "b.pdf", "c.pdf"])
    attempted, code = _run_ingest(root, _chained_oom(), fails_on="b.pdf")
    assert attempted == ["a.pdf", "b.pdf"], "c.pdf would be collateral on a dead converter"
    assert code == 1


def test_ordinary_failure_skips_only_that_file(tmp_path):
    root = _ingest_root(tmp_path, ["a.pdf", "b.pdf", "c.pdf"])
    attempted, code = _run_ingest(root, ValueError("bad xref"), fails_on="b.pdf")
    assert attempted == ["a.pdf", "b.pdf", "c.pdf"]
    assert code == 0


def test_charts_are_opt_in(tmp_path):
    root = _ingest_root(tmp_path, ["a.pdf"])
    parser = cli.build_parser()
    seen: list[bool] = []

    def record(_cfg, *, do_charts, do_figures):
        seen.append(do_charts)
        return object()

    for argv, expected in (
        (["ingest"], False),
        (["ingest", "--charts"], True),
        (["ingest", "--no-charts"], False),
    ):
        with (
            patch.object(cli, "make_converter", side_effect=record),
            patch.object(cli, "convert_file", return_value=object()),
            patch.object(cli, "_index_from_doc"),
            patch.object(cli, "write_corpus"),
            patch.object(cli, "write_structure_report"),
        ):
            cli.cmd_ingest(parser.parse_args(["--root", str(root), *argv, "--rebuild"]))
        assert seen[-1] is expected


def test_chart_limits_bound_the_real_docling_model():
    """Guards the upstream shape: Docling feeds `_model_max_length` straight into
    `generate()`, and `elements_batch_size` is read off the class at import."""
    from docling.models.stages.chart_extraction.granite_vision import (
        ChartExtractionModelGraniteVisionV4 as Model,
    )

    from localrok.convert import apply_chart_limits

    assert Model.elements_batch_size == 16, "docling default changed; recheck the VRAM budget"
    apply_chart_limits(512, 4)
    apply_chart_limits(512, 4)  # idempotent: must not stack wrappers
    assert Model.elements_batch_size == 4

    class Stub(Model):
        def _load_model(self, _artifacts_path):
            # HF's "unset" sentinel, exactly what granite-vision-4.1-4b ships
            self._model_max_length = 1000000000000000019884624838656

    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.chart_extraction_options import ChartExtractionModelOptions

    stub = Stub(
        enabled=True,
        artifacts_path=Path("/nonexistent"),
        options=ChartExtractionModelOptions(),
        accelerator_options=AcceleratorOptions(device="cpu"),
    )
    assert stub._model_max_length == 512
