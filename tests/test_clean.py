"""`localrok clean` removes derived artefacts. It must never touch corpus/source/."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

from localrok import cli

REPO_ROOT = Path(__file__).resolve().parent.parent


def _populated_root(tmp_path: Path) -> Path:
    shutil.copy(REPO_ROOT / "config.toml", tmp_path / "config.toml")
    source = tmp_path / "corpus" / "source"
    source.mkdir(parents=True)
    (source / "a.pdf").write_bytes(b"%PDF-1.7\n")

    derived = tmp_path / "corpus" / "derived"
    (derived / "doclingdoc").mkdir(parents=True)
    (derived / "figures" / "a").mkdir(parents=True)
    (derived / "markdown").mkdir()
    (derived / ".keep").touch()
    (derived / "doclingdoc" / "a.json").write_text("{}", encoding="utf-8")
    (derived / "figures" / "a" / "fig-1-0.png").write_bytes(b"\x89PNG")
    (derived / "markdown" / "a.md").write_text("# a", encoding="utf-8")
    (derived / "localrok.db").write_bytes(b"SQLite format 3\x00")
    (derived / "CORPUS.md").write_text("map", encoding="utf-8")
    return tmp_path


def _clean(root: Path, *argv: str) -> int:
    args = cli.build_parser().parse_args(["--root", str(root), "clean", "--yes", *argv])
    return cli.cmd_clean(args)


def test_clean_wipes_derived_but_keeps_source_and_placeholder(tmp_path):
    root = _populated_root(tmp_path)
    derived = root / "corpus" / "derived"

    assert _clean(root) == 0

    assert sorted(p.name for p in derived.iterdir()) == [".keep"]
    assert (root / "corpus" / "source" / "a.pdf").exists()


def test_keep_docling_preserves_the_expensive_conversions(tmp_path):
    root = _populated_root(tmp_path)
    derived = root / "corpus" / "derived"

    assert _clean(root, "--keep-docling") == 0

    assert (derived / "doclingdoc" / "a.json").exists()
    assert not (derived / "localrok.db").exists()
    assert not (derived / "figures").exists()


def test_clean_is_idempotent(tmp_path):
    root = _populated_root(tmp_path)
    assert _clean(root) == 0
    assert _clean(root) == 0


def test_refuses_without_yes_when_not_interactive(tmp_path):
    root = _populated_root(tmp_path)
    args = cli.build_parser().parse_args(["--root", str(root), "clean"])
    with patch.object(cli.sys.stdin, "isatty", return_value=False):
        assert cli.cmd_clean(args) == 1
    assert (root / "corpus" / "derived" / "localrok.db").exists()


def test_declining_the_prompt_deletes_nothing(tmp_path):
    root = _populated_root(tmp_path)
    args = cli.build_parser().parse_args(["--root", str(root), "clean"])
    with (
        patch.object(cli.sys.stdin, "isatty", return_value=True),
        patch("builtins.input", return_value="n"),
    ):
        assert cli.cmd_clean(args) == 1
    assert (root / "corpus" / "derived" / "localrok.db").exists()
