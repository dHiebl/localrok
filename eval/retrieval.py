#!/usr/bin/env python3
"""Search-only evaluation: does docs_search put a gold section in front of the agent?

No Pi, no LLM. Each question's `queries` go through the same search() the MCP
server calls, and the rank of the first gold hit is reported. Runs in seconds,
so it is the check to repeat after every ingest or ranking change.

    uv run python eval/retrieval.py --questions questions.toml
    uv run python eval/retrieval.py --questions questions.toml -v
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from eval.gold import first_gold_rank, parse_hits  # noqa: E402
from localrok.config import load_config  # noqa: E402
from localrok.db import connect  # noqa: E402
from localrok.search import search  # noqa: E402


def _missing_gold(conn, golds: list[dict]) -> list[dict]:
    """Gold locations with no section in the index — stale after a re-ingest."""
    missing = []
    for g in golds:
        rows = conn.execute(
            """
            SELECT s.page_start, s.page_end FROM sections s
            JOIN documents d ON d.doc_id = s.doc_id
            WHERE instr(lower(d.filename), lower(?)) > 0
              AND instr(lower(s.section_path), lower(?)) > 0
            """,
            (g["doc"], g.get("section", "")),
        ).fetchall()
        page = g.get("page")
        if page is not None:
            rows = [
                r
                for r in rows
                if r["page_start"] is not None
                and r["page_start"] <= page <= (r["page_end"] or r["page_start"])
            ]
        if not rows:
            missing.append(g)
    return missing


def _fmt_rank(rank: int | None) -> str:
    return "-" if rank is None else str(rank)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="results per search (default: config)")
    parser.add_argument("--only", help="run a single question id")
    parser.add_argument("-v", "--verbose", action="store_true", help="show top hits for misses")
    args = parser.parse_args(argv)

    cfg = load_config(root=ROOT)
    conn = connect(cfg.paths.db, create=False)
    limit = args.limit or cfg.search.default_limit

    questions = tomllib.loads(args.questions.read_text(encoding="utf-8")).get("question", [])
    questions = [q for q in questions if q.get("queries") and q.get("gold")]
    if args.only:
        questions = [q for q in questions if q["id"] == args.only]

    rows = []
    for q in questions:
        stale = _missing_gold(conn, q["gold"])
        for g in stale:
            print(f"warn: {q['id']} gold not in index: {g}", file=sys.stderr)
        ranks = []
        for query in q["queries"]:
            hits = parse_hits(search(conn, query, cfg.search, limit=limit))
            ranks.append((query, first_gold_rank(hits, q["gold"]), hits))
        found = [r for _, r, _ in ranks if r is not None]
        rows.append(
            {
                "id": q["id"],
                "category": q["category"],
                "first": ranks[0][1],
                "best": min(found) if found else None,
                "ranks": ranks,
            }
        )

    print(f"{'id':<5} {'category':<20} {'first':>5} {'best':>5}  queries (rank)")
    for r in rows:
        per_query = ", ".join(f"{q!r}={_fmt_rank(k)}" for q, k, _ in r["ranks"])
        print(
            f"{r['id']:<5} {r['category']:<20} {_fmt_rank(r['first']):>5} "
            f"{_fmt_rank(r['best']):>5}  {per_query}"
        )
        if args.verbose and r["best"] is None:
            for query, _, hits in r["ranks"]:
                print(f"        {query!r}:")
                for h in hits[:5]:
                    print(f"          {h.filename[:34]} | {h.section[:60]} | p{h.page}")

    n = len(rows)
    if not n:
        print("no questions with queries and gold")
        return 1

    def rate(pred) -> str:
        k = sum(1 for r in rows if pred(r))
        return f"{k}/{n} ({100 * k / n:.0f}%)"

    print()
    print(f"first query, gold at rank 1      : {rate(lambda r: r['first'] == 1)}")
    print(f"first query, gold in top 3       : {rate(lambda r: (r['first'] or 99) <= 3)}")
    print(f"first query, gold in top {limit:<2}      : {rate(lambda r: r['first'] is not None)}")
    print(f"any query,   gold in top {limit:<2}      : {rate(lambda r: r['best'] is not None)}")

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    print()
    print("first-query top-%d by category:" % limit)
    for cat, items in sorted(by_cat.items()):
        k = sum(1 for r in items if r["first"] is not None)
        print(f"  {cat:<20} {k}/{len(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
