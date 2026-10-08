#!/usr/bin/env python3
"""Drive Pi as a subprocess against the evaluation set.

Does not implement its own tool-calling loop. LocalRök uses Pi's built-in MCP;
the file-search baseline disables MCP and uses the same local model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from eval.gold import first_gold_rank, matches_gold, parse_hits, parse_read_labels  # noqa: E402
from localrok.config import load_config  # noqa: E402
from localrok.tools import TOOL_NAMES  # noqa: E402


def _local_tool_name(name: str | None) -> str | None:
    """Map only our native MCP names to server names for evidence scoring."""
    short = (name or "").removeprefix("mcp__localrok__")
    return short if short in TOOL_NAMES else name


def _load_questions(path: Path) -> list[dict]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return list(data.get("question", []))


def _preflight(cfg, *, baseline: bool) -> None:
    from localrok.ollama import check_ollama

    status = check_ollama(cfg.llm)
    if status["status"] != "up":
        raise SystemExit(f"preflight: {status['detail']}")
    if not Path(cfg.agent.pi_bin).is_file():
        import shutil

        if shutil.which(cfg.agent.pi_bin) is None:
            raise SystemExit("preflight: optional Pi CLI is not installed or not on PATH")

    if not baseline:
        from localrok.db import IndexUnavailable, connect

        try:
            conn = connect(cfg.paths.db, create=False)
        except IndexUnavailable as exc:
            raise SystemExit(f"preflight: {exc}") from exc
        n = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        conn.close()
        if n == 0:
            raise SystemExit("preflight: index is empty")
        if not cfg.paths.corpus_md.exists():
            raise SystemExit("preflight: CORPUS.md missing — run `localrok corpus`")
        tokens = (len(cfg.paths.corpus_md.read_text(encoding="utf-8")) + 3) // 4
        if tokens > cfg.prompt.corpus_map_max_tokens:
            raise SystemExit(
                f"preflight: CORPUS.md is {tokens} tokens, "
                f"budget {cfg.prompt.corpus_map_max_tokens}"
            )


def _text_parts(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
    )


def _parse_pi_json(stdout: str) -> dict:
    """Read `pi --mode json` output (one event per line, pi-agent-core types.d.ts).

    Tool calls come from tool_execution_start/_end paired by toolCallId; the
    answer is the text of the last assistant message_end; usage is summed over
    every assistant message, since each one is a separate prefill.
    """
    calls: dict[str, dict] = {}
    order: list[str] = []
    answer = ""
    tokens_in = tokens_cached = tokens_out = 0
    # JSONL frames end on LF; Unicode separators may occur inside document text.
    for line in stdout.split("\n"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        typ = ev.get("type")
        if typ == "tool_execution_start":
            cid = ev.get("toolCallId") or str(len(order))
            calls[cid] = {"name": ev.get("toolName"), "args": ev.get("args"), "result": ""}
            order.append(cid)
        elif typ == "tool_execution_end":
            call = calls.get(ev.get("toolCallId"))
            if call is not None:
                result = ev.get("result")
                call["result"] = _text_parts(
                    result.get("content") if isinstance(result, dict) else result
                )
                call["is_error"] = bool(ev.get("isError"))
        elif typ == "message_end":
            msg = ev.get("message") or {}
            if msg.get("role") != "assistant":
                continue
            usage = msg.get("usage") or {}
            tokens_in += int(usage.get("input") or 0)
            tokens_cached += int(usage.get("cacheRead") or 0)
            tokens_out += int(usage.get("output") or 0)
            text = _text_parts(msg.get("content"))
            if text.strip():
                answer = text

    tool_calls = [calls[c] for c in order]
    return {
        "answer": answer.strip(),
        "tool_calls": tool_calls,
        "tokens_in": tokens_in,
        "tokens_cached": tokens_cached,
        "tokens_out": tokens_out,
    }


def _gold_scores(tool_calls: list[dict], gold: list[dict]) -> dict:
    """Whether docs_search/docs_read put a gold section in front of the agent."""
    if not gold:
        return {"first_search_gold_rank": None, "gold_seen": None}
    first_rank = None
    searched = False
    seen = False
    for call in tool_calls:
        name = _local_tool_name(call["name"])
        if name == "docs_search":
            hits = parse_hits(call["result"])
            rank = first_gold_rank(hits, gold)
            if not searched:
                first_rank, searched = rank, True
            seen = seen or rank is not None
        elif name == "docs_read":
            labels = parse_read_labels(call["result"])
            seen = seen or any(matches_gold(h, g) for h in labels for g in gold)
    return {"first_search_gold_rank": first_rank, "gold_seen": seen}


def _tool_log_since(path: Path, offset: int) -> list[dict]:
    """Server-side records appended to the tool log after `offset` bytes."""
    if not path.exists():
        return []
    with path.open("rb") as fh:
        fh.seek(offset)
        lines = fh.read().decode("utf-8", errors="replace").split("\n")
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _cited_sections_returned(answer: str, payloads: list[str], calls: list[str]) -> bool:
    """True if every Sources line filename appears in a tool payload or the answer tools."""
    sources = []
    in_block = False
    for line in answer.splitlines():
        if line.strip().lower().startswith("sources"):
            in_block = True
            continue
        if in_block:
            if not line.strip():
                break
            sources.append(line.strip())
    if not sources:
        return True  # nothing cited — citation validity N/A
    blob = "\n".join(payloads) + "\n" + " ".join(calls)
    ok = True
    for src in sources:
        filename = src.split("|")[0].strip().lstrip("-").lstrip("[]0123456789. ").strip()
        if filename and filename not in blob and filename not in answer:
            ok = False
    return ok


def _contains_all(text: str, needles: list[str] | None) -> bool:
    if not needles:
        return True
    lower = text.lower()
    return all(n.lower() in lower for n in needles)


def _refusal_ok(answer: str) -> bool:
    markers = (
        "cannot",
        "can't",
        "could not",
        "not found",
        "no evidence",
        "not in the",
        "do not have",
        "unable",
        "unanswerable",
        "not specified",
        "nicht",
    )
    low = answer.lower()
    return any(m in low for m in markers)


def run_question(cfg, question: dict, tools: list[str], *, baseline: bool) -> dict:
    pi = cfg.agent.pi_bin
    thinking = cfg.agent.thinking_level
    cmd = [
        pi,
        "-p",
        "--mode",
        "json",
        "--no-session",
        "--offline",
        "--approve",
        "--no-extensions",
        "--no-context-files",
        "--no-skills",
        "--no-prompt-templates",
        "--provider",
        cfg.agent.provider,
        "--model",
        cfg.llm.model,
        "--thinking",
        thinking,
        "--tools",
        ",".join(tools),
    ]
    if not baseline:
        corpus = cfg.paths.corpus_md
        cmd.extend(
            [
                "--extension",
                "builtin:mcp",
                "--system-prompt",
                str(ROOT / "AGENTS.md"),
                "--append-system-prompt",
                str(corpus),
            ]
        )
    if baseline:
        hint = (
            f"Investigate the Markdown corpus at {cfg.paths.markdown} "
            "using only grep, find, read and ls. Do not invent page numbers."
        )
        cmd.extend(["--no-mcp", "--system-prompt", hint])
    cmd.append(question["text"])

    log_offset = cfg.paths.tool_log.stat().st_size if cfg.paths.tool_log.exists() else 0
    t0 = datetime.now(timezone.utc)
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=900,
        env=os.environ.copy(),
    )
    wall = (datetime.now(timezone.utc) - t0).total_seconds()
    parsed = _parse_pi_json(proc.stdout)
    calls = parsed["tool_calls"]
    names = [c["name"] for c in calls]
    unexpected_tools = sorted({name for name in names if name not in tools}, key=str)
    answer = parsed["answer"]
    server = [] if baseline else _tool_log_since(cfg.paths.tool_log, log_offset)
    localrok_calls = sum(1 for n in names if n in cfg.agent.tools)
    if not baseline and len(server) != localrok_calls:
        # Another client (web UI) writing to the same log, or calls Pi never saw.
        print(
            f"warn: {question['id']}: Pi made {localrok_calls} docs_* calls, "
            f"tool log recorded {len(server)}",
            file=sys.stderr,
        )
    gold = question.get("gold") or []
    # grep/read payloads have no section headers to match gold against.
    scores = (
        {"first_search_gold_rank": None, "gold_seen": None}
        if baseline
        else _gold_scores(calls, gold)
    )
    return {
        "id": question["id"],
        "text": question["text"],
        "category": question.get("category"),
        "answerable": question.get("answerable", True),
        "answer": answer,
        "tool_calls": calls,
        "hops": len(calls),
        "wall_clock_s": wall,
        "tokens_in": parsed["tokens_in"],
        "tokens_cached": parsed["tokens_cached"],
        "tokens_out": parsed["tokens_out"],
        "server_calls": len(server),
        "server_ms": round(sum(r.get("ms", 0) for r in server), 1),
        "returncode": proc.returncode,
        "stderr": proc.stderr[-4000:],
        "unexpected_tools": unexpected_tools,
        "expect_contains_ok": _contains_all(answer, question.get("expect_contains")),
        "citation_tool_returned": _cited_sections_returned(
            answer, [c["result"] for c in calls], names
        ),
        "refusal_ok": (_refusal_ok(answer) if question.get("answerable") is False else None),
        "first_tool": names[0] if names else None,
        **scores,
        "gold": gold,
        "expect_condition": question.get("expect_condition"),
    }


def _source_fingerprint() -> str:
    """Fingerprint source files without relying on external version tooling."""
    digest = hashlib.sha256()
    paths = [ROOT / name for name in ("config.toml", "AGENTS.md", "pyproject.toml", ".pi/mcp.json")]
    for directory in ("src", "server", "web/src", "eval"):
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.suffix in {".py", ".ts"})
    paths.append(ROOT / "web/api.py")
    for path in sorted(paths):
        if path.is_file():
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def write_report(dest: Path, results: list[dict], meta: dict) -> None:
    lines = [
        f"# Eval report {meta['timestamp']}",
        "",
        f"- mode: {meta['mode']}",
        f"- n: {len(results)}",
        f"- model: {meta['model']}",
        f"- thinking requested: {meta['thinking']}",
        f"- tools: {', '.join(meta['tools'])}",
        "",
        "`gold@1` is the rank of the first gold hit in the first docs_search; "
        "`seen` is whether any search or read returned a gold section.",
        "",
        "| id | category | hops | wall s | tok in/cached/out | gold@1 | seen "
        "| contains | refuse | first tool |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        tokens = f"{r['tokens_in']}/{r['tokens_cached']}/{r['tokens_out']}"
        lines.append(
            f"| {r['id']} | {r['category']} | {r['hops']} | "
            f"{r['wall_clock_s']:.1f} | {tokens} | {r['first_search_gold_rank']} | "
            f"{r['gold_seen']} | {r['expect_contains_ok']} | "
            f"{r['refusal_ok']} | {r['first_tool']} |"
        )
    lines += [
        "",
        "## Grading worksheet",
        "Fill `grades.toml` by hand. Do not use the model under test to grade itself.",
        "",
        "```toml",
    ]
    for r in results:
        lines.append("[[grade]]")
        lines.append(f'id = "{r["id"]}"')
        lines.append("answer_correct = ")  # human
        lines.append("condition_correct = ")
        lines.append('notes = ""')
        lines.append("")
    lines.append("```")
    dest.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="file-search baseline")
    parser.add_argument("--only", help="Run a single question id")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--questions",
        type=Path,
        required=True,
        help="TOML questions and expected source locations for your collection",
    )
    args = parser.parse_args(argv)

    cfg = load_config(root=ROOT)
    if not args.skip_preflight:
        _preflight(cfg, baseline=args.baseline)

    questions = _load_questions(args.questions)
    if args.only:
        questions = [q for q in questions if q["id"] == args.only]
        if not questions:
            print(f"no question {args.only}", file=sys.stderr)
            return 1

    tools = cfg.agent.baseline_tools if args.baseline else cfg.agent.tools
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "eval" / "results" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for q in questions:
        print(f"running {q['id']} …", file=sys.stderr)
        r = run_question(cfg, q, tools, baseline=args.baseline)
        if r["returncode"] != 0:
            print(
                f"FATAL: {q['id']}: Pi exited with status {r['returncode']}: "
                f"{r['stderr']}. Aborting.",
                file=sys.stderr,
            )
            (out_dir / "results.json").write_text(
                json.dumps({"aborted": r}, indent=2), encoding="utf-8"
            )
            return 2
        if r["unexpected_tools"]:
            print(
                f"FATAL: {q['id']} used tools outside the evaluation allowlist: "
                f"{', '.join(map(str, r['unexpected_tools']))}. Aborting.",
                file=sys.stderr,
            )
            (out_dir / "results.json").write_text(
                json.dumps({"aborted": r}, indent=2), encoding="utf-8"
            )
            return 2
        results.append(r)
        (out_dir / "results.json").write_text(
            json.dumps({"results": results}, indent=2), encoding="utf-8"
        )

    source_fingerprint = _source_fingerprint()
    try:
        import docling

        docling_ver = getattr(docling, "__version__", "unknown")
    except ImportError:
        docling_ver = "unavailable"
    pi_ver = subprocess.run(
        [cfg.agent.pi_bin, "--version"], capture_output=True, text=True
    ).stdout.strip()

    thinking = cfg.agent.thinking_level
    meta = {
        "timestamp": ts,
        "mode": "baseline" if args.baseline else "localrok",
        "pi_version": pi_ver,
        "mcp_support": "builtin" if not args.baseline else "disabled",
        "docling_version": docling_ver,
        "model": cfg.llm.model,
        "thinking": thinking,
        "source_fingerprint": source_fingerprint,
        "tools": tools,
        "tool_exposure": "direct" if not args.baseline else "n/a",
        "corpus_map_mode": "append",
        "config_toml": (ROOT / "config.toml").read_text(encoding="utf-8"),
        "mcp_json": (ROOT / ".pi/mcp.json").read_text(encoding="utf-8")
        if (ROOT / ".pi/mcp.json").exists()
        else "",
    }
    payload = {"meta": meta, "results": results}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(out_dir / "report.md", results, meta)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
