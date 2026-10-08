"""Load portable settings from config.toml and LOCALROK_ environment overrides."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


@dataclass
class PathsConfig:
    source: Path
    derived: Path

    @property
    def doclingdoc(self) -> Path:
        return self.derived / "doclingdoc"

    @property
    def markdown(self) -> Path:
        return self.derived / "markdown"

    @property
    def figures(self) -> Path:
        return self.derived / "figures"

    @property
    def db(self) -> Path:
        return self.derived / "localrok.db"

    @property
    def corpus_md(self) -> Path:
        return self.derived / "CORPUS.md"

    @property
    def structure_report(self) -> Path:
        return self.derived / "structure_report.md"

    @property
    def tool_log(self) -> Path:
        return self.derived / "tool_calls.jsonl"


@dataclass
class IngestConfig:
    device: str = "auto"
    gpu_ids: list[int] = field(default_factory=list)
    gpu_min_free_mib: int = 8192
    gpu_min_free_mib_charts: int = 20480
    unit_max_tokens: int = 384
    images_scale: float = 2.0
    figure_max_edge_px: int = 1600
    do_charts: bool = False
    do_formulas: bool = False
    do_code: bool = False
    chart_max_new_tokens: int = 512
    chart_batch_size: int = 1


@dataclass
class SearchConfig:
    default_limit: int = 8
    max_limit: int = 20
    window_max_chars: int = 1200
    units_per_section_cap: int = 2
    heading_weight: float = 2.0


@dataclass
class ReadConfig:
    max_tokens: int = 5000
    chars_per_token: float = 3.5


@dataclass
class PromptConfig:
    corpus_map_max_tokens: int = 4000
    max_prompt_tokens: int = 8000
    corpus_chars_per_token: int = 4


@dataclass
class LlmConfig:
    base_url: str = "http://127.0.0.1:11434/v1"
    model: str = "qwen3.5:9b-mlx"
    context_window: int = 32768
    max_output_tokens: int = 4096

    @property
    def server_url(self) -> str:
        return self.base_url.rstrip("/").removesuffix("/v1")


@dataclass
class AgentConfig:
    pi_bin: str = "pi"
    provider: str = "ollama"
    thinking_level: str = "off"
    tools: list[str] = field(
        default_factory=lambda: [
            "mcp__localrok__docs_search",
            "mcp__localrok__docs_read",
            "mcp__localrok__docs_outline",
        ]
    )
    baseline_tools: list[str] = field(default_factory=lambda: ["grep", "find", "read", "ls"])


@dataclass
class Config:
    root: Path
    paths: PathsConfig
    ingest: IngestConfig
    search: SearchConfig
    read: ReadConfig
    prompt: PromptConfig
    llm: LlmConfig
    agent: AgentConfig


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _coerce(current: Any, raw: str) -> Any:
    if isinstance(current, bool):
        return _as_bool(raw)
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        return [part.strip() for part in raw.split(",") if part.strip()]
    return raw


def _apply_env(section: dict[str, Any], prefix: str) -> dict[str, Any]:
    out = dict(section)
    for key, value in section.items():
        env_key = f"LOCALROK_{prefix}_{key}".upper()
        if env_key in os.environ:
            out[key] = _coerce(value, os.environ[env_key])
    return out


def _section(raw: dict[str, Any], name: str, cls):
    values = _apply_env(asdict(cls()) | raw.get(name, {}), name.upper())
    if name == "ingest":
        values["gpu_ids"] = [int(i) for i in values["gpu_ids"]]
    return cls(**values)


def load_config(root: Path | None = None, config_path: Path | None = None) -> Config:
    root = (root or project_root()).resolve()
    path = config_path or Path(os.environ.get("LOCALROK_CONFIG", root / "config.toml"))
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    paths = _apply_env(
        {"source": "corpus/source", "derived": "corpus/derived"} | raw.get("paths", {}), "PATHS"
    )
    source, derived = Path(paths["source"]), Path(paths["derived"])
    config = Config(
        root=root,
        paths=PathsConfig(
            source=source if source.is_absolute() else root / source,
            derived=derived if derived.is_absolute() else root / derived,
        ),
        ingest=_section(raw, "ingest", IngestConfig),
        search=_section(raw, "search", SearchConfig),
        read=_section(raw, "read", ReadConfig),
        prompt=_section(raw, "prompt", PromptConfig),
        llm=_section(raw, "llm", LlmConfig),
        agent=_section(raw, "agent", AgentConfig),
    )
    base = config.llm.base_url.rstrip("/")
    parsed = urlparse(base)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("llm.base_url must be an HTTP endpoint without a query or fragment")
    config.llm.base_url = base if base.endswith("/v1") else base + "/v1"
    config.llm.model = config.llm.model.strip()
    if not config.llm.model:
        raise ValueError("llm.model must name a locally installed Ollama model")
    if not re.fullmatch(r"auto|cpu|mps|cuda(?::[0-9]+)?", config.ingest.device):
        raise ValueError("ingest.device must be auto, cpu, mps, cuda, or cuda:N")
    if config.llm.max_output_tokens <= 0 or config.llm.context_window <= 0:
        raise ValueError("LLM token budgets must be positive")
    reserved = config.prompt.max_prompt_tokens + config.llm.max_output_tokens
    if config.read.max_tokens <= 0 or config.read.max_tokens + reserved > config.llm.context_window:
        raise ValueError("Prompt, read, and output budgets exceed llm.context_window")
    return config
