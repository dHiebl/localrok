"""Tool names, descriptions, and JSON schemas — one source of truth."""

from __future__ import annotations

DOCS_SEARCH = {
    "name": "docs_search",
    "description": (
        "Keyword search over the document collection (BM25). "
        "Returns ranked evidence windows with filename, section_path, page, "
        "completeness (complete_section or fragment), and any figures in the "
        "section. Prefer identifiers and exact terms first. A complete_section "
        "window is citable; a fragment plus a specific value/limit/threshold/"
        "procedure step requires docs_read."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search terms. Identifiers like J5 are matched as phrases.",
            },
            "doc": {
                "type": "string",
                "description": "Optional filename or unambiguous substring to restrict search.",
            },
            "section": {
                "type": "string",
                "description": "Optional section_path prefix to restrict search.",
            },
            "exact": {
                "type": "boolean",
                "description": "If true, match the whole query as a phrase.",
                "default": False,
            },
            "limit": {
                "type": "integer",
                "description": "Max hits to return (1–20, default 8).",
                "default": 8,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

DOCS_READ = {
    "name": "docs_read",
    "description": (
        "Read one or more sections, page ranges, or whole documents in a single "
        "call. Batch cross-document comparisons here instead of issuing multiple "
        "reads. Each target is '<doc> > <section path>', a page range like "
        "'<doc> > p37-41', or '<doc>' for the whole document, where <doc> is a "
        "filename or document ID. A bare section path needs doc=."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "targets": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 10,
                "description": "One to ten read targets (section path, page range, or filename).",
            },
            "doc": {
                "type": "string",
                "description": "Default document for bare section paths.",
            },
        },
        "required": ["targets"],
        "additionalProperties": False,
    },
}

DOCS_OUTLINE = {
    "name": "docs_outline",
    "description": (
        "Heading tree for one document, with page numbers and character counts "
        "so you can judge read cost before calling docs_read. Use for large "
        "documents whose corpus-map entry is coarse."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "doc": {
                "type": "string",
                "description": "Filename or unambiguous substring.",
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum heading depth to include (default 3).",
                "default": 3,
            },
        },
        "required": ["doc"],
        "additionalProperties": False,
    },
}

TOOLS = [DOCS_SEARCH, DOCS_READ, DOCS_OUTLINE]
TOOL_NAMES = [t["name"] for t in TOOLS]


def tools_json() -> list[dict]:
    return TOOLS
