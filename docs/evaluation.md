# Optional evaluation

The project includes no predefined question set. Create questions and expected
source locations for your own collection. Review answers and conditions against
the actual documents before interpreting scores.

A TOML question file uses this schema:

```toml
[[question]]
id = "Q1"
text = "Your question"
category = "lookup"
queries = ["exact search phrase", "alternative search phrase"]
gold = [{ doc = "your-document.pdf", section = "Relevant section" }]
expect_contains = ["expected answer text"]
expect_refusal = false
```

`gold` locations can include `page` when real provenance exists. Use an explicit
refusal question to test whether the agent acknowledges missing evidence.

```bash
uv run python eval/retrieval.py --questions /path/to/questions.toml
uv run python eval/run.py --questions /path/to/questions.toml
```

Retrieval evaluation needs the local index only. Agent evaluation needs Pi 1.1.0,
a configured Ollama provider, and the native `.pi/mcp.json` configuration. See
[Pi setup](ollama.md#optional-pi-cli). No MCP extension or warm-up session is needed.

LocalRök evaluations load only Pi's built-in MCP extension and allow the three
`mcp__localrok__docs_*` tools configured in `config.toml`. Evidence scoring recognizes
these names while results retain the actual names Pi called. `--baseline` disables
MCP and document-assistant instructions, and uses only file-search tools over
derived Markdown. Calls outside either allowlist abort the evaluation. `--only Q1`
selects one question. Headless runs trust this project's configuration for the run
and disable automatic network activity; inference still calls the local Ollama
endpoint.

Agent results record model settings, dependency versions, source fingerprints,
returned evidence, timing, and token usage. The report includes a worksheet for
human grading. Model-produced answers must not grade themselves.
