# Implementation reference

The source code and `config.toml` define the supported interfaces and defaults.
This document describes the boundaries that new changes should preserve.

## Processing and storage

1. Convert supported source files to DoclingDocument JSON.
2. Remove repeated running furniture without discarding real headings.
3. Build sections and smaller search units with source provenance.
4. Extract figure images, captions, labels, and optional chart CSV.
5. Export a human-readable Markdown view.
6. Index document metadata, sections, units, and figures in SQLite.
7. Produce structure reports and a budgeted corpus map.

Source hashes and conversion versions support incremental imports. Cached JSON
can be reindexed without reloading conversion models. Schema changes require a
rebuild. Conversion-option changes require `ingest --rebuild`.

## Shared tools

`localrok.tools.call_tool` dispatches search, read, and outline operations and logs
results. MCP and HTTP wrappers must preserve the same tool behavior. Search
completeness and omitted-read hints must remain accurate.

Identifiers use generic patterns; document types come from title keywords.
Missing metadata must be reported honestly. Do not introduce organization-specific
rules into the default pipeline.

## Local inference

Ollama serves the configured model. FastAPI forwards streamed and non-streamed
OpenAI-compatible requests and exposes server/model status. The browser receives
model and token settings from the API, rather than duplicating them in UI code.
A missing server and a missing model produce different actionable messages.

## Validation

Use synthetic fixtures for document metadata, provenance, search ranking, read
budgets, figures, and device selection. Test model status and both proxy modes
without depending on a running model server. Separately smoke-test an actual
local model for tool calling, thinking, and citations. Optional evaluations use
question files supplied for the user's own collection.
