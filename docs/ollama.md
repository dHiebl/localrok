# Local Ollama setup

Open Ollama or run `ollama serve`. Use `ollama list` to inspect installed models.
Set `llm.model` in `config.toml` to the exact installed model tag. LocalRök requires
structured tool calling; verify that the selected model and runtime support it.

The default `qwen3.5:9b-mlx` is a local model choice, not a requirement. A different
installed tool-capable model can be selected with `LOCALROK_LLM_MODEL`. The OpenAI-
compatible endpoint defaults to `http://127.0.0.1:11434/v1`. No local API key is
required; clients that require a key can use the dummy value `ollama`.

## Context size

`llm.context_window` describes the allocated context, not merely the model's
maximum capability. Configure the same size in Ollama. For example, create a
model alias using this Modelfile, replacing the base model when necessary:

```text
FROM qwen3.5:9b-mlx
PARAMETER num_ctx 32768
```

```bash
ollama create localrok-local -f Modelfile
```

Then set `llm.model = "localrok-local"`. Alternatively configure Ollama's server
context allocation using `OLLAMA_CONTEXT_LENGTH`. Use `ollama ps` to inspect the
loaded model's context. Start with conservative settings and measure memory and
latency before increasing context or model size.

Thinking support varies by model. LocalRök uses the OpenAI-compatible reasoning
field and defaults to thinking off. Validate the behavior when selecting a model
or changing the installed runtime. Chart images are displayed from document tool
results; query-time vision is not required.

## Optional Pi CLI

The browser uses Pi libraries directly and does not require the Pi executable.
For CLI evaluation, install the tested Pi version with built-in MCP support:

```bash
npm install --global @earendil-works/pi-coding-agent@1.1.0
pi --version
```

Configure the Ollama provider in `~/.pi/agent/models.json` (or `models.json` inside
`PI_CODING_AGENT_DIR` if set). Merge this entry with any existing providers:

```json
{
  "providers": {
    "ollama": {
      "baseUrl": "http://127.0.0.1:11434/v1",
      "api": "openai-completions",
      "apiKey": "ollama",
      "models": [{
        "id": "qwen3.5:9b-mlx",
        "contextWindow": 32768,
        "maxTokens": 4096,
        "reasoning": true,
        "thinkingLevelMap": { "off": "none", "minimal": "low", "xhigh": "high" },
        "compat": {
          "supportsDeveloperRole": false,
          "supportsStore": false,
          "supportsReasoningEffort": true,
          "maxTokensField": "max_tokens",
          "thinkingFormat": "openai"
        }
      }]
    }
  }
}
```

Keep this entry consistent with LocalRök's selected model and context/output
budgets. From the project directory, start document chat:

```bash
pi --approve --offline --provider ollama --model qwen3.5:9b-mlx --thinking off --tools 'mcp__localrok__*'
```

`--approve` lets Pi load this project's configuration for that run. Native Pi reads
`.pi/mcp.json`, exposes the three tools directly as `mcp__localrok__docs_search`,
`mcp__localrok__docs_read`, and `mcp__localrok__docs_outline`, and does not need a
warm-up session or a separate MCP extension. Remove any previously installed MCP
extension that replaces Pi's `/mcp` command before using built-in MCP.
Use `/mcp` inside the interactive session to check the server connection and tools.

The server caps MCP text at 19,000 UTF-8 bytes so Pi's 20 KiB output truncation
cannot hide evidence. Longer results carry explicit partial/omitted notices;
request smaller sections or page ranges to establish the missing context.
