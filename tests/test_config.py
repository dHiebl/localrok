import pytest

from localrok.config import load_config


def test_environment_overrides_apply_even_when_sections_are_omitted(tmp_path, monkeypatch):
    (tmp_path / "config.toml").write_text("")
    monkeypatch.setenv("LOCALROK_LLM_MODEL", "example:latest")
    monkeypatch.setenv("LOCALROK_LLM_BASE_URL", "http://localhost:9919/")
    monkeypatch.setenv("LOCALROK_INGEST_GPU_IDS", "0,2")
    monkeypatch.setenv("LOCALROK_INGEST_DO_CHARTS", "true")
    config = load_config(root=tmp_path)
    assert config.llm.model == "example:latest"
    assert config.llm.base_url == "http://localhost:9919/v1"
    assert config.llm.server_url == "http://localhost:9919"
    assert config.ingest.gpu_ids == [0, 2]
    assert config.ingest.do_charts is True
    assert config.paths.source == tmp_path / "corpus/source"


def test_context_budget_reserves_prompt_and_answer(tmp_path):
    (tmp_path / "config.toml").write_text("[llm]\ncontext_window = 8192\n")
    with pytest.raises(ValueError, match="budgets exceed"):
        load_config(root=tmp_path)
