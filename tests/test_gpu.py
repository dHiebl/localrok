from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from localrok.config import IngestConfig
from localrok.convert import GpuBusyError, pick_free_gpu, resolve_device

GPU_TABLE = "0, GPU-aaa, 40000\n1, GPU-bbb, 35000\n2, GPU-ccc, 5000\n3, GPU-ddd, 5000\n"


def _smi(gpu_table: str, apps: str):
    def fake(args, **_kwargs):
        joined = " ".join(args)
        if "--query-gpu=" in joined:
            return gpu_table
        if "--query-compute-apps=" in joined:
            return apps
        raise AssertionError(joined)

    return fake


def test_picks_idle_card_with_most_free():
    with patch("localrok.convert.subprocess.check_output", side_effect=_smi(GPU_TABLE, "")):
        assert pick_free_gpu([0, 1], min_free_mib=8192) == "cuda:0"


def test_skips_busy_card_for_idle_sibling():
    apps = "GPU-aaa, 111\n"
    with patch("localrok.convert.subprocess.check_output", side_effect=_smi(GPU_TABLE, apps)):
        assert pick_free_gpu([0, 1], min_free_mib=8192) == "cuda:1"


def test_never_selects_unconfigured_devices_even_if_emptiest():
    table = "0, GPU-aaa, 1000\n1, GPU-bbb, 1000\n2, GPU-ccc, 40000\n3, GPU-ddd, 40000\n"
    with patch("localrok.convert.subprocess.check_output", side_effect=_smi(table, "")):
        with pytest.raises(GpuBusyError, match="No ingest GPU"):
            pick_free_gpu([0, 1], min_free_mib=8192)


def test_refuses_when_both_allowed_cards_are_busy():
    apps = "GPU-aaa, 111\nGPU-bbb, 222\n"
    with patch("localrok.convert.subprocess.check_output", side_effect=_smi(GPU_TABLE, apps)):
        with pytest.raises(GpuBusyError, match="busy"):
            pick_free_gpu([0, 1], min_free_mib=8192)


def test_refuses_below_min_free():
    with patch("localrok.convert.subprocess.check_output", side_effect=_smi(GPU_TABLE, "")):
        with pytest.raises(GpuBusyError, match="need"):
            pick_free_gpu([0, 1], min_free_mib=50_000)


def test_cpu_when_nvidia_smi_missing():
    with patch("localrok.convert.subprocess.check_output", side_effect=FileNotFoundError):
        assert pick_free_gpu([0, 1]) == "cpu"


def _torch(monkeypatch, *, cuda=False, mps=False):
    import sys

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: cuda),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        ),
    )


def test_auto_selects_apple_acceleration(monkeypatch):
    _torch(monkeypatch, mps=True)
    assert resolve_device(IngestConfig()) == "mps"


def test_auto_falls_back_to_cpu(monkeypatch):
    _torch(monkeypatch)
    assert resolve_device(IngestConfig()) == "cpu"


def test_explicit_cpu_is_respected(monkeypatch):
    _torch(monkeypatch, mps=True)
    assert resolve_device(IngestConfig(device="cpu")) == "cpu"


def test_explicit_unavailable_device_reports_how_to_recover(monkeypatch):
    _torch(monkeypatch)
    with pytest.raises(RuntimeError, match="ingest.device"):
        resolve_device(IngestConfig(device="mps"))
    with pytest.raises(RuntimeError, match="ingest.device"):
        resolve_device(IngestConfig(device="cuda"))


def test_empty_gpu_ids_considers_all_visible_devices():
    with patch("localrok.convert.subprocess.check_output", side_effect=_smi(GPU_TABLE, "")):
        assert pick_free_gpu([]) == "cuda:0"
