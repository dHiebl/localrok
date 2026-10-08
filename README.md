# LocalRök

*Local RAG with agent-directed search and verifiable sources.*

A local document assistant powered by Ollama. LocalRök converts your documents into
structured content, searches a SQLite full-text index, and lets an agent read
relevant sections before answering with source citations.

Works with PDF, DOCX, HTML, PPTX, and Markdown. No embeddings or vector database
are required. Documents and generated artifacts stay on your machine.

## Setup

Requires Python 3.12+, uv, Node.js 22.19+, npm, and Ollama.

The displayed name is LocalRök. The package, imports, and CLI use `localrok`,
and environment variables use the `LOCALROK_` prefix.

```bash
uv sync --frozen --group web --group dev
npm ci --prefix web
```

Open the Ollama application, or run `ollama serve`. Use `ollama list` to see your
installed models. Set `llm.model` in `config.toml` to a model that supports tool
calling. The default is `qwen3.5:9b-mlx`; change it if that model is unavailable or
does not support tools on your installed Ollama version. Download models with
`ollama pull <model>` when needed. No API key is needed for local Ollama.

`llm.base_url` defaults to `http://127.0.0.1:11434/v1`. Set the actual Ollama context
size to match `llm.context_window`; the frontend value alone does not change the
server allocation. See [Ollama setup](docs/ollama.md).

## Add documents

Put documents in `corpus/source/`, then run:

```bash
uv run localrok ingest
uv run localrok stats
uv run localrok search "leave policy"
uv run localrok outline "employee-handbook.pdf"
uv run localrok read "employee-handbook.pdf > 3 Leave policy"
uv run localrok doctor
```

Derived JSON, Markdown, figures, the index, and the live corpus map go to
`corpus/derived/`. The source documents are not modified. The initial collection
is empty; there are no bundled organization-specific documents or questions.

Ingestion uses Docling's local models independently of Ollama. `ingest.device`
selects `auto`, `cpu`, `mps`, `cuda`, or `cuda:N`. Auto selects an available
accelerator; individual Docling stages may use CPU when an accelerator is
unsupported. Inference and ingestion share memory on Apple Silicon, so run large
imports separately from chat. Docling downloads model artifacts on first use.

Chart-value extraction, formula enrichment, and code enrichment are optional and
disabled by default. Captions, figure images, and extracted labels are retained.
Enable chart values with `uv run localrok ingest --charts`; use `--rebuild` to reconvert
already indexed documents after changing conversion options. Chart extraction is
resource intensive and its numbers must be checked against the original figure.

Unchanged files are skipped. `--rebuild` forces reconversion. `localrok clean` lists
derived artifacts and asks before deleting them; `--keep-docling` retains the
canonical JSON for reindexing, and `--yes` skips the confirmation. Source files
are always retained.

## Browser chat

Run these in separate terminals:

```bash
uv run --group web python web/api.py
```

```bash
npm run dev --prefix web
```

Open http://127.0.0.1:3000. FastAPI listens on loopback port 8080; Vite proxies the
document tools, figures, and streamed model requests. The header distinguishes an
unreachable Ollama server from a missing model. Restart the API after changing
configuration. All model settings come from `config.toml`.

## Document tools

`uv run localrok serve` exposes `docs_search`, `docs_read`, and `docs_outline` through
MCP stdio. Pi's built-in MCP client uses `.pi/mcp.json` with direct tool exposure.
Other MCP clients can use the same `uv run localrok serve` command. The browser
uses HTTP wrappers around the same implementations.

The assistant prompt is `AGENTS.md`. A generated corpus map supplies the available
filenames, titles, section previews, and structure quality. Answers cite only
returned evidence; missing information should produce an explicit limitation.

## Configuration and validation

Settings live in `config.toml`. Override them with `LOCALROK_<SECTION>_<KEY>` variables,
for example `LOCALROK_LLM_MODEL` or `LOCALROK_INGEST_DEVICE`. `LOCALROK_CONFIG` selects another
configuration file. Read budgets are estimates and reserve space for the prompt
and output; long conversations may require a new chat.

```bash
uv run pytest
uv run ruff check .
npm run check --prefix web
npm run build --prefix web
```

Optional evaluation harnesses accept a question file you create for your own
collection. No benchmark answers are bundled. See [Evaluation](docs/evaluation.md).
Pi 1.1.0 is the tested CLI version for agent evaluation. MCP support is built in;
no MCP extension is required. The browser and document server do not require the
Pi executable. See [Pi setup](docs/ollama.md#optional-pi-cli).

Tool calls include full retrieved text in `corpus/derived/tool_calls.jsonl`. Keep
that file with your private document outputs; it grows until you remove it.
Dependency downloads require network access. Local processing uses cached models.

## Layout

| Path | Purpose |
|---|---|
| `src/localrok/` | Conversion, indexing, search, reading, and configuration |
| `server/mcp_server.py` | MCP stdio server |
| `web/` | Browser chat and FastAPI proxy |
| `prompts/` | Citation guidance |
| `docs/` | Architecture, setup, and evaluation |
| `tests/` | Synthetic document and integration tests |
| `eval/` | Optional evaluation harnesses |
