"""The three agent tools behind one entry point, with a call log.

The MCP server and the web API both go through call_tool, so every call an
agent makes lands in one JSONL file whatever the client. eval/run.py reads it
back to score what the agent actually saw.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

from localrok.config import Config
from localrok.db import connect
from localrok.read import outline, read
from localrok.search import search

TOOL_NAMES = ("docs_search", "docs_read", "docs_outline")


def _run(cfg: Config, name: str, args: dict[str, Any]) -> str:
    if name not in TOOL_NAMES:
        raise ValueError(f"unknown tool {name!r}")
    conn = connect(cfg.paths.db, create=False)
    try:
        if name == "docs_search":
            return search(
                conn,
                args["query"],
                cfg.search,
                doc=args.get("doc"),
                section=args.get("section"),
                exact=bool(args.get("exact", False)),
                limit=args.get("limit"),
            )
        if name == "docs_read":
            return read(conn, list(args["targets"]), cfg.read, doc=args.get("doc"))
        return outline(conn, args["doc"], max_depth=int(args.get("max_depth", 3)))
    finally:
        conn.close()


def call_tool(cfg: Config, name: str, **args: Any) -> str:
    """Run one tool and return its text. Never raises: a failure is the result."""
    t0 = time.perf_counter()
    try:
        result = _run(cfg, name, args)
    except Exception as exc:
        result = f"{name} failed: {exc}"
    log_call(cfg, name, args, result, time.perf_counter() - t0)
    return result


def log_call(cfg: Config, name: str, args: dict[str, Any], result: str, seconds: float) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "pid": os.getpid(),
        "tool": name,
        "args": args,
        "ms": round(seconds * 1000, 1),
        "chars": len(result),
        "result": result,
    }
    try:
        cfg.paths.tool_log.parent.mkdir(parents=True, exist_ok=True)
        with cfg.paths.tool_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        # stdout belongs to the MCP stdio protocol.
        print(f"localrok: tool log not written: {exc}", file=sys.stderr)
