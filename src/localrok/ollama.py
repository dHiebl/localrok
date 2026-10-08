"""Read-only checks for the configured local Ollama server and model."""

from __future__ import annotations

import json
import urllib.request

from localrok.config import LlmConfig


def check_ollama(config: LlmConfig) -> dict[str, str]:
    url = f"{config.server_url}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            payload = json.loads(response.read())
        models = payload["models"]
        if not isinstance(models, list):
            raise ValueError("expected a model list")
        names = {m.get("name", m.get("model")) for m in models if isinstance(m, dict)}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {
            "status": "down",
            "model": config.model,
            "detail": f"Ollama is not reachable at {config.server_url}. "
            "Open Ollama or run `ollama serve`.",
            "error": str(exc),
        }
    if config.model not in names:
        return {
            "status": "missing_model",
            "model": config.model,
            "detail": f"Ollama is running, but {config.model!r} is not installed. "
            "Choose an installed model in config.toml or download this model with Ollama.",
        }
    return {"status": "up", "model": config.model, "detail": ""}
