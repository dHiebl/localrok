"""HTTP tool and Ollama proxy tests use synthetic data and mock transports."""

from dataclasses import replace

import api
import httpx
import pytest
from fastapi.testclient import TestClient

from localrok.config import PathsConfig
from localrok.db import connect
from tests.helpers import populate_sample


@pytest.fixture(autouse=True)
def isolated_api(tmp_path, monkeypatch):
    paths = PathsConfig(source=tmp_path / "source", derived=tmp_path / "derived")
    paths.source.mkdir()
    cfg = replace(api._cfg, paths=paths)
    monkeypatch.setattr(api, "_cfg", cfg)
    conn = connect(paths.db)
    populate_sample(conn)
    conn.close()


def _status(monkeypatch, status="up", detail=""):
    async def result():
        return {"status": status, "model": api._cfg.llm.model, "detail": detail}

    monkeypatch.setattr(api, "model_status", result)


def _transport(monkeypatch, handler):
    client_class = httpx.AsyncClient
    monkeypatch.setattr(
        api.httpx,
        "AsyncClient",
        lambda **kwargs: client_class(transport=httpx.MockTransport(handler), **kwargs),
    )


def test_health_and_document_tools():
    client = TestClient(api.app)
    assert client.get("/api/health").json() == {"status": "ok"}
    stats = client.get("/api/stats").json()
    assert stats["documents"] == 2
    prompt = client.get("/api/system-prompt").json()
    assert "docs_search" in prompt["text"]
    assert prompt["context_window"] == api._cfg.llm.context_window
    assert prompt["max_output_tokens"] == api._cfg.llm.max_output_tokens
    assert "device-manual" in prompt["text"] or "Live corpus map" in prompt["text"]
    hit = client.post("/api/docs_search", json={"query": "J5"}).json()["text"]
    assert "device-manual-v4.0.0.pdf" in hit
    read = client.post("/api/docs_read", json={"targets": ["device > p21"]}).json()["text"]
    assert "21.0 V" in read
    assert client.post("/api/docs_outline", json={"doc": "device"}).status_code == 200


def test_model_endpoint_checks_selected_model(monkeypatch):
    monkeypatch.setattr(
        api,
        "check_ollama",
        lambda cfg: {
            "status": "missing_model",
            "model": cfg.model,
            "detail": "Choose an installed model",
        },
    )
    result = TestClient(api.app).get("/api/model").json()
    assert result["status"] == "missing_model"
    assert result["model"] == api._cfg.llm.model


@pytest.mark.parametrize("status", ["down", "missing_model"])
def test_chat_reports_unavailable_model(monkeypatch, status):
    _status(monkeypatch, status, "Ollama or the selected model is unavailable")
    result = TestClient(api.app).post("/v1/chat/completions", json={"stream": True})
    assert result.status_code == 503
    assert "unavailable" in result.json()["error"]["message"]


def test_nonstream_proxy_uses_configured_endpoint_and_strips_credentials(monkeypatch):
    _status(monkeypatch)
    monkeypatch.setattr(
        api,
        "_cfg",
        replace(
            api._cfg,
            llm=replace(
                api._cfg.llm,
                base_url="http://127.0.0.1:9919/v1",
            ),
        ),
    )
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Answer"}}]})

    _transport(monkeypatch, handler)
    result = TestClient(api.app).post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer client-placeholder"},
        json={"model": api._cfg.llm.model, "messages": []},
    )
    assert result.status_code == 200
    assert result.json()["choices"][0]["message"]["content"] == "Answer"
    assert str(seen[0].url) == "http://127.0.0.1:9919/v1/chat/completions"
    assert "authorization" not in seen[0].headers


def test_stream_proxy_preserves_tool_events_and_completion(monkeypatch):
    _status(monkeypatch)
    events = (
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call1",'
        b'"type":"function","function":{"name":"docs_search","arguments":"{}"}}]}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    _transport(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            content=events,
            headers={"content-type": "text/event-stream"},
        ),
    )
    result = TestClient(api.app).post("/v1/chat/completions", json={"stream": True})
    assert result.status_code == 200
    assert result.content == events
    assert result.headers["content-type"].startswith("text/event-stream")


@pytest.mark.parametrize("stream", [False, True])
def test_upstream_error_is_returned_without_rewriting(monkeypatch, stream):
    _status(monkeypatch)
    _transport(
        monkeypatch,
        lambda request: httpx.Response(
            400,
            json={"error": {"message": "model does not support tools"}},
        ),
    )
    result = TestClient(api.app).post("/v1/chat/completions", json={"stream": stream})
    assert result.status_code == 400
    assert result.json()["error"]["message"] == "model does not support tools"


@pytest.mark.parametrize("stream", [False, True])
def test_connection_failure_returns_actionable_error(monkeypatch, stream):
    _status(monkeypatch)

    def handler(request):
        raise httpx.ConnectError("offline", request=request)

    _transport(monkeypatch, handler)
    result = TestClient(api.app).post("/v1/chat/completions", json={"stream": stream})
    assert result.status_code == 503
    assert "ollama serve" in result.json()["error"]["message"]
