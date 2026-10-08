# LocalRök architecture

LocalRök is a general document assistant. It separates document processing,
retrieval, and model inference so the same index can serve a command-line client,
MCP client, or browser chat.

## Offline document processing

Sources in `corpus/source/` are converted with Docling. Canonical DoclingDocument
JSON preserves structure and provenance. The pipeline removes repeated running
headers, produces coherent sections and small search units, exports Markdown,
extracts figures, and builds a SQLite FTS5 index. The generated `CORPUS.md` gives
an agent a compact map of the collection.

Metadata is inferred from generic titles and version labels. A leading identifier
such as `DOC-POLICY-001` can be retained; no publisher or industry is assumed.
Unknown types and missing page provenance remain explicit.

Device selection is configurable: automatic acceleration, CPU, Apple MPS, or
NVIDIA CUDA. Optional chart, formula, and code enrichment use Docling models and
are independent of the chat model. Chart extraction is bounded by generation and
batch limits; it currently relies on a pinned Docling internal API.

## Retrieval

SQLite FTS5/BM25 ranks small text, table, caption, label, and chart-data units.
Sections provide the larger read boundary. Search returns evidence windows with
`complete_section` or `fragment` completeness labels and source locations.

The three tools are `docs_search`, `docs_read`, and `docs_outline`. Their common
implementation is used by the CLI, MCP stdio server, and FastAPI HTTP routes.
Every call is logged with its arguments and returned evidence.

## Answer generation

The browser's Pi agent loop calls local Ollama through FastAPI's streaming
OpenAI-compatible proxy. It chooses when to search and read. The system prompt
contains citation rules and the current corpus map; there is no automatic vector
retrieval stage.

`llm` configuration is the single source for endpoint, model, context window, and
output limit. Server status checks verify the selected model among all installed
models. Thinking behavior must match the chosen model and Ollama version.

## Context and sources

Read limits are per call. Prompt, read, and output budgets must fit the configured
context window. Ollama must also be configured for that actual context size.
Conversation history consumes additional space; start a new chat when the budget
is exhausted.

Citations refer only to returned sections and real provenance. Specific values
and instructions found in fragments require a coherent read. Extracted chart
values are attributed to their figures and never treated as printed limits.
