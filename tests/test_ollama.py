import json
from unittest.mock import patch

import pytest

from localrok.config import LlmConfig
from localrok.ollama import check_ollama


class Response:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self):
        return self.body


def test_selected_model_need_not_be_first():
    config = LlmConfig(model="chosen:latest")
    body = json.dumps({"models": [{"name": "other:latest"}, {"name": config.model}]}).encode()
    with patch("localrok.ollama.urllib.request.urlopen", return_value=Response(body)) as request:
        assert check_ollama(config)["status"] == "up"
        request.assert_called_once_with("http://127.0.0.1:11434/api/tags", timeout=2)


def test_missing_model_is_distinct_from_server_down():
    with patch("localrok.ollama.urllib.request.urlopen", return_value=Response(b'{"models":[]}')):
        result = check_ollama(LlmConfig())
    assert result["status"] == "missing_model"
    assert "config.toml" in result["detail"]


@pytest.mark.parametrize("body", [b"invalid", b"{}", b'{"models":null}'])
def test_invalid_response_is_actionable(body):
    with patch("localrok.ollama.urllib.request.urlopen", return_value=Response(body)):
        result = check_ollama(LlmConfig())
    assert result["status"] == "down"
    assert "ollama serve" in result["detail"]


def test_unreachable_server():
    with patch("localrok.ollama.urllib.request.urlopen", side_effect=OSError("offline")):
        assert check_ollama(LlmConfig())["status"] == "down"
