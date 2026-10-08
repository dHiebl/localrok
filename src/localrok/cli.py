"""`localrok` CLI. Search/read/outline share code with the MCP tools."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path

from localrok.chunk import chunk_document, mark_running_furniture
from localrok.config import Config, load_config
from localrok.convert import (
    GpuBusyError,
    convert_file,
    docling_version,
    document_page_count,
    first_heading_title,
    is_gpu_fatal,
    list_source_files,
    load_docling_json,
    make_converter,
    parse_document_metadata,
    sha256_file,
)
from localrok.corpus import CorpusBudgetError, write_corpus
from localrok.db import IndexUnavailable, connect, lookup_document, optimize_fts
from localrok.figures import attach_figures_to_sections, extract_figures
from localrok.index import index_document
from localrok.markdown import emit_markdown
from localrok.read import outline as do_outline
from localrok.read import read as do_read
from localrok.search import search as do_search
from localrok.structure import write_structure_report
from localrok.tool_specs import tools_json


def _cfg(args: argparse.Namespace) -> Config:
    root = Path(args.root).resolve() if getattr(args, "root", None) else None
    return load_config(root=root)


def _open_db(cfg: Config, *, create: bool = False) -> sqlite3.Connection:
    return connect(cfg.paths.db, create=create)


def cmd_ingest(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    source = cfg.paths.source
    if not source.exists():
        print(f"Source directory missing: {source}", file=sys.stderr)
        return 1
    files = list_source_files(source, pattern=args.only)
    if not files:
        print(f"No documents in {source}", file=sys.stderr)
        return 1

    cfg.paths.derived.mkdir(parents=True, exist_ok=True)
    conn = _open_db(cfg, create=True)
    version = docling_version()
    do_figures = not args.no_figures
    do_charts = (cfg.ingest.do_charts or args.charts) and not args.no_charts and do_figures
    converter = None
    gpu_dead = False

    for path in files:
        digest = sha256_file(path)
        existing = lookup_document(conn, path.name)
        json_path = cfg.paths.doclingdoc / f"{path.stem}.json"
        unchanged = (
            not args.rebuild
            and existing is not None
            and existing["source_sha256"] == digest
            and existing["docling_version"] == version
            and json_path.exists()
        )
        if unchanged:
            print(f"skip (unchanged): {path.name}", file=sys.stderr)
            continue

        stale = existing is not None and (
            existing["source_sha256"] != digest or existing["docling_version"] != version
        )
        need_convert = args.rebuild or not json_path.exists() or stale
        try:
            if need_convert:
                if converter is None:
                    converter = make_converter(
                        cfg.ingest, do_charts=do_charts, do_figures=do_figures
                    )
                print(f"convert: {path.name}", file=sys.stderr)
                doc = convert_file(
                    path,
                    json_path,
                    cfg,
                    do_charts=do_charts,
                    do_figures=do_figures,
                    converter=converter,
                )
            else:
                print(f"chunk from json: {path.name}", file=sys.stderr)
                doc = load_docling_json(json_path)
        except GpuBusyError as exc:
            print(str(exc), file=sys.stderr)
            conn.close()
            return 1
        except Exception as exc:
            print(f"  FAILED {path.name}: {exc}", file=sys.stderr)
            if is_gpu_fatal(exc):
                # The CUDA context is poisoned; every later file on this
                # converter would fail as collateral. Keep what is indexed.
                print(
                    "  GPU context lost — stopping the batch. Retry the remaining files "
                    "in a fresh run (pass --no-charts, or lower ingest.chart_batch_size).",
                    file=sys.stderr,
                )
                gpu_dead = True
                break
            continue

        _index_from_doc(
            conn,
            cfg,
            path,
            doc,
            digest,
            version,
            do_figures=do_figures,
        )
        conn.commit()
        print(f"  indexed {path.name}", file=sys.stderr)
        try:
            write_corpus(conn, cfg.paths.corpus_md, cfg.prompt, version)
        except CorpusBudgetError as exc:
            print(str(exc), file=sys.stderr)
            conn.close()
            return 1

    optimize_fts(conn)
    write_structure_report(conn, cfg.paths.structure_report)
    try:
        write_corpus(conn, cfg.paths.corpus_md, cfg.prompt, version)
    except CorpusBudgetError as exc:
        print(str(exc), file=sys.stderr)
        conn.close()
        return 1
    conn.close()
    if gpu_dead:
        print("ingest stopped early — index holds the files that completed", file=sys.stderr)
        return 1
    print("ingest complete", file=sys.stderr)
    return 0


def _index_from_doc(
    conn: sqlite3.Connection,
    cfg: Config,
    path: Path,
    doc,
    digest: str,
    version: str,
    *,
    do_figures: bool,
) -> None:
    mark_running_furniture(doc)
    sections = chunk_document(doc, max_tokens=cfg.ingest.unit_max_tokens)
    figures = []
    assignment: dict[int, int] = {}
    if do_figures:
        scanned: list[int] = []
        figures = extract_figures(
            doc,
            path.stem,
            cfg.paths.figures,
            cfg.paths.derived,
            cfg.ingest,
            scanned_pages=scanned,
        )
        assignment = attach_figures_to_sections(figures, sections)
        if scanned:
            print(
                f"  scanned pages (no text items, OCR not run): {scanned}",
                file=sys.stderr,
            )

    emit_markdown(doc, cfg.paths.markdown / f"{path.stem}.md", sections)

    title = first_heading_title(doc)
    page_count = document_page_count(doc)
    has_prov = 1 if path.suffix.lower() == ".pdf" else 0
    meta = parse_document_metadata(
        filename=path.name,
        stem=path.stem,
        title=title,
        page_count=page_count,
        source_sha256=digest,
        version=version,
        has_provenance=has_prov,
    )
    if not meta.title:
        meta.title = path.stem
    index_document(conn, meta, sections, figures, assignment)


def _tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def cmd_clean(args: argparse.Namespace) -> int:
    """Delete derived artefacts so the next ingest starts from nothing.

    `ingest --rebuild` re-indexes what is still in source/, but it cannot drop
    rows, JSON, markdown or figures belonging to files that were removed, nor
    figure PNGs orphaned by a changed figure count.
    """
    cfg = _cfg(args)
    derived = cfg.paths.derived
    keep = {derived / ".keep"}
    if args.keep_docling:
        keep.add(cfg.paths.doclingdoc)

    targets = sorted(p for p in derived.iterdir() if p not in keep) if derived.exists() else []
    if not targets:
        print(f"Nothing to clean in {derived}")
        return 0

    total = 0
    for target in targets:
        size = _tree_size(target)
        total += size
        print(f"  {target.relative_to(cfg.root)}  {size / 1e6:.1f} MB")
    print(f"Delete {len(targets)} entries ({total / 1e6:.1f} MB) from {derived}?")
    if args.keep_docling:
        print(f"Keeping {cfg.paths.doclingdoc.name}/ — ingest will re-chunk without the GPU.")

    if not args.yes:
        sys.stdout.flush()  # keep the listing above the prompt when stdout is piped
        if not sys.stdin.isatty():
            print("Refusing to delete without a terminal; pass --yes.", file=sys.stderr)
            return 1
        if input("Type 'yes' to confirm: ").strip().lower() != "yes":
            print("Aborted.")
            return 1

    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    print(f"Cleaned {derived}")
    return 0


def cmd_corpus(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    conn = _open_db(cfg)
    try:
        write_corpus(conn, cfg.paths.corpus_md, cfg.prompt, docling_version())
    except CorpusBudgetError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        conn.close()
    print(cfg.paths.corpus_md)
    return 0


def cmd_structure(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    conn = _open_db(cfg)
    write_structure_report(conn, cfg.paths.structure_report)
    conn.close()
    print(cfg.paths.structure_report)
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    conn = _open_db(cfg)
    print(
        do_search(
            conn,
            args.query,
            cfg.search,
            doc=args.doc,
            section=args.section,
            exact=args.exact,
            limit=args.limit,
        )
    )
    conn.close()
    return 0


def cmd_read(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    conn = _open_db(cfg)
    print(do_read(conn, args.targets, cfg.read, doc=args.doc))
    conn.close()
    return 0


def cmd_outline(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    conn = _open_db(cfg)
    print(do_outline(conn, args.doc, max_depth=args.max_depth), end="")
    conn.close()
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    conn = _open_db(cfg)
    docs = conn.execute(
        """
        SELECT filename, page_count, structure_score, doc_type,
               (SELECT COUNT(*) FROM sections s WHERE s.doc_id = d.doc_id) AS sections,
               (SELECT COUNT(*) FROM units u WHERE u.doc_id = d.doc_id) AS units,
               (SELECT COUNT(*) FROM figures f
                 WHERE f.doc_id = d.doc_id AND f.kept = 1) AS figures_kept,
               (SELECT COUNT(*) FROM figures f
                 WHERE f.doc_id = d.doc_id AND f.kept = 0) AS figures_dropped
        FROM documents d
        ORDER BY filename
        """
    ).fetchall()
    n_docs = len(docs)
    n_units = conn.execute("SELECT COUNT(*) AS n FROM units").fetchone()["n"]
    n_secs = conn.execute("SELECT COUNT(*) AS n FROM sections").fetchone()["n"]
    print(f"{n_docs} documents, {n_secs} sections, {n_units} units")
    for d in docs:
        score = f"{d['structure_score']:.2f}" if d["structure_score"] is not None else "?"
        print(
            f"  {d['filename']}: {d['sections']} sec, {d['units']} units, "
            f"{d['figures_kept']} figures (dropped {d['figures_dropped']}), "
            f"structure {score}, type {d['doc_type']}"
        )
    conn.close()
    return 0


def cmd_serve(_args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(root))
    from server.mcp_server import run

    run()
    return 0


def cmd_tool_specs(_args: argparse.Namespace) -> int:
    print(json.dumps(tools_json(), indent=2))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        mark = "OK" if passed else "FAIL"
        if not passed:
            ok = False
        extra = f" — {detail}" if detail else ""
        print(f"[{mark}] {label}{extra}")

    check("config.toml", (cfg.root / "config.toml").exists(), str(cfg.root / "config.toml"))
    check("source dir", cfg.paths.source.exists(), str(cfg.paths.source))
    try:
        conn = _open_db(cfg)
    except IndexUnavailable as exc:
        check("index", False, str(exc))
    else:
        n = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        conn.close()
        check("index", n > 0, f"{n} documents in {cfg.paths.db}")

    if cfg.paths.corpus_md.exists():
        tokens = max(
            1,
            (len(cfg.paths.corpus_md.read_text(encoding="utf-8")) + 3) // 4,
        )
        check(
            "CORPUS.md budget",
            tokens <= cfg.prompt.corpus_map_max_tokens,
            f"{tokens} tokens / {cfg.prompt.corpus_map_max_tokens}",
        )
    else:
        print("[WARN] CORPUS.md missing — run `localrok corpus`")

    from localrok.ollama import check_ollama

    status = check_ollama(cfg.llm)
    check("Ollama model", status["status"] == "up", status["detail"] or cfg.llm.model)
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="localrok")
    p.add_argument("--root", help="Project root (default: auto)")
    sub = p.add_subparsers(dest="cmd", required=True)

    ingest = sub.add_parser("ingest", help="Convert, chunk, and index the corpus")
    ingest.add_argument("--rebuild", action="store_true")
    ingest.add_argument("--only", help="fnmatch on filename")
    ingest.add_argument("--no-figures", action="store_true")
    charts = ingest.add_mutually_exclusive_group()
    charts.add_argument(
        "--charts",
        action="store_true",
        help="Extract chart values to CSV (optional, uses a local vision model)",
    )
    charts.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart extraction when Granite Vision cannot run",
    )
    ingest.set_defaults(func=cmd_ingest)

    cl = sub.add_parser("clean", help="Delete corpus/derived so ingest starts from scratch")
    cl.add_argument("--yes", "-y", action="store_true", help="Skip the confirmation prompt")
    cl.add_argument(
        "--keep-docling",
        action="store_true",
        help="Keep doclingdoc/ so the next ingest re-chunks without re-converting",
    )
    cl.set_defaults(func=cmd_clean)

    c = sub.add_parser("corpus", help="Regenerate CORPUS.md")
    c.set_defaults(func=cmd_corpus)

    s = sub.add_parser("structure", help="Regenerate structure_report.md")
    s.set_defaults(func=cmd_structure)

    se = sub.add_parser("search", help="Same output as docs_search")
    se.add_argument("query")
    se.add_argument("--doc", "-d")
    se.add_argument("--section")
    se.add_argument("--exact", action="store_true")
    se.add_argument("--limit", "-n", type=int, default=None)
    se.set_defaults(func=cmd_search)

    rd = sub.add_parser("read", help="Same output as docs_read")
    rd.add_argument("targets", nargs="+")
    rd.add_argument("--doc", "-d")
    rd.set_defaults(func=cmd_read)

    ol = sub.add_parser("outline", help="Same output as docs_outline")
    ol.add_argument("doc")
    ol.add_argument("--max-depth", type=int, default=3)
    ol.set_defaults(func=cmd_outline)

    st = sub.add_parser("stats")
    st.set_defaults(func=cmd_stats)

    sv = sub.add_parser("serve", help="MCP server on stdio")
    sv.set_defaults(func=cmd_serve)

    ts = sub.add_parser("tool-specs")
    ts.add_argument("--json", action="store_true")
    ts.set_defaults(func=cmd_tool_specs)

    dr = sub.add_parser("doctor")
    dr.set_defaults(func=cmd_doctor)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code = args.func(args)
    except IndexUnavailable as exc:
        print(exc, file=sys.stderr)
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
