"""MCP server: docs_search / docs_read / docs_outline over stdio."""

from __future__ import annotations

import sys
from pathlib import Path

# FastMCP in mcp 1.x moved to MCPServer in mcp 2.x; same stdio tool surface.
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
for _p in (_ROOT, _SRC):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from mcp.server.mcpserver import MCPServer  # noqa: E402

from localrok.config import load_config  # noqa: E402
from localrok.tool_specs import DOCS_OUTLINE, DOCS_READ, DOCS_SEARCH  # noqa: E402
from localrok.tools import call_tool  # noqa: E402

mcp = MCPServer(name="localrok-docs")

# Pi cuts MCP text above 20 KiB. Keep our omission notice visible even for
# multibyte text, large figure labels, or a single unusually large indexed unit.
MCP_OUTPUT_MAX_BYTES = 19_000


def _limit_output(name: str, result: str) -> str:
    encoded = result.encode("utf-8")
    if len(encoded) <= MCP_OUTPUT_MAX_BYTES:
        return result
    hint = {
        "docs_search": "Lower limit or narrow doc/section, then read the matching section.",
        "docs_read": "Read a smaller section or page range; use docs_outline to locate it.",
        "docs_outline": "Lower max_depth, or search for a specific section.",
    }[name]
    prefix = "--- PARTIAL RESULT (MCP output budget) ---\n---\n"
    suffix = f"\n\n--- OMITTED (MCP output budget) ---\n{hint}\n"
    available = MCP_OUTPUT_MAX_BYTES - len((prefix + suffix).encode("utf-8"))
    excerpt = encoded[:available].decode("utf-8", errors="ignore")
    return prefix + excerpt + suffix


def _call(name: str, **args) -> str:
    # Config is re-read per call so a config.toml edit applies without
    # restarting the agent that owns this process.
    return _limit_output(name, call_tool(load_config(root=_ROOT), name, **args))


@mcp.tool(name=DOCS_SEARCH["name"], description=DOCS_SEARCH["description"], structured_output=False)
def docs_search(
    query: str,
    doc: str | None = None,
    section: str | None = None,
    exact: bool = False,
    limit: int = 8,
) -> str:
    return _call("docs_search", query=query, doc=doc, section=section, exact=exact, limit=limit)


@mcp.tool(name=DOCS_READ["name"], description=DOCS_READ["description"], structured_output=False)
def docs_read(targets: list[str], doc: str | None = None) -> str:
    return _call("docs_read", targets=targets, doc=doc)


@mcp.tool(
    name=DOCS_OUTLINE["name"],
    description=DOCS_OUTLINE["description"],
    structured_output=False,
)
def docs_outline(doc: str, max_depth: int = 3) -> str:
    return _call("docs_outline", doc=doc, max_depth=max_depth)


def run() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run()
