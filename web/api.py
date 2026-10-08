"""Thin REST wrapper around LocalRök's search/read/outline — same functions as MCP/CLI.

The browser cannot speak MCP stdio. This FastAPI process is the HTTP face of
those three tools and a streaming proxy to the local Ollama endpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (ROOT, SRC):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from localrok.config import load_config  # noqa: E402
from localrok.db import IndexUnavailable, connect  # noqa: E402
from localrok.ollama import check_ollama  # noqa: E402
from localrok.tools import call_tool  # noqa: E402

AGENTS_MD = ROOT / "AGENTS.md"
_cfg = load_config(root=ROOT)
MODEL_DOWN = "Ollama is not reachable. Open Ollama or run `ollama serve`."


def _model_info() -> dict[str, Any]:
    return {
        "model": _cfg.llm.model,
        "thinking_level": _cfg.agent.thinking_level,
        "context_window": _cfg.llm.context_window,
        "max_output_tokens": _cfg.llm.max_output_tokens,
    }


app = FastAPI(title="LocalRök API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _text_ok(text: str) -> dict[str, str]:
    return {"text": text}


class SearchBody(BaseModel):
    query: str
    doc: str | None = None
    section: str | None = None
    exact: bool = False
    limit: int = Field(default=8, ge=1, le=20)


class ReadBody(BaseModel):
    targets: list[str]
    doc: str | None = None


class OutlineBody(BaseModel):
    doc: str
    max_depth: int = Field(default=3, ge=1, le=8)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _openai_error(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": "api_error"}},
    )


async def model_status() -> dict[str, str]:
    return await run_in_threadpool(check_ollama, _cfg.llm)


@app.get("/api/model")
async def model() -> dict[str, str]:
    return await model_status()


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    try:
        conn = connect(_cfg.paths.db, create=False)
    except IndexUnavailable:
        return {
            "documents": 0,
            "sections": 0,
            "units": 0,
            "docs": [],
            **_model_info(),
        }
    try:
        rows = conn.execute(
            """
            SELECT filename, page_count, structure_score, doc_type,
                   (SELECT COUNT(*) FROM sections s WHERE s.doc_id = d.doc_id) AS sections,
                   (SELECT COUNT(*) FROM units u WHERE u.doc_id = d.doc_id) AS units
            FROM documents d
            ORDER BY filename
            """
        ).fetchall()
        n_units = conn.execute("SELECT COUNT(*) AS n FROM units").fetchone()["n"]
        n_secs = conn.execute("SELECT COUNT(*) AS n FROM sections").fetchone()["n"]
        return {
            "documents": len(rows),
            "sections": n_secs,
            "units": n_units,
            "docs": [
                {
                    "filename": r["filename"],
                    "sections": r["sections"],
                    "units": r["units"],
                    "structure_score": r["structure_score"],
                    "doc_type": r["doc_type"],
                }
                for r in rows
            ],
            **_model_info(),
        }
    finally:
        conn.close()


@app.get("/api/system-prompt")
def system_prompt() -> dict[str, Any]:
    agents = (
        AGENTS_MD.read_text(encoding="utf-8")
        if AGENTS_MD.exists()
        else (
            "You are LocalRök. Search with docs_search, then docs_read before answering. "
            "Cite [doc §id]."
        )
    )
    corpus_path = _cfg.paths.corpus_md
    if corpus_path.exists():
        corpus = corpus_path.read_text(encoding="utf-8")
    else:
        corpus = "(No CORPUS.md yet — run `localrok ingest`.)"
    text = agents.rstrip() + "\n\n## Live corpus map\n\n" + corpus
    budget = int(_cfg.prompt.max_prompt_tokens * _cfg.read.chars_per_token)
    if len(text) > budget:
        raise ValueError("System prompt exceeds prompt.max_prompt_tokens; reduce the corpus map")
    return {"text": text, **_model_info()}


@app.post("/api/docs_search")
def docs_search(body: SearchBody) -> dict[str, str]:
    return _text_ok(call_tool(_cfg, "docs_search", **body.model_dump()))


@app.post("/api/docs_read")
def docs_read(body: ReadBody) -> dict[str, str]:
    return _text_ok(call_tool(_cfg, "docs_read", **body.model_dump()))


@app.post("/api/docs_outline")
def docs_outline(body: OutlineBody) -> dict[str, str]:
    return _text_ok(call_tool(_cfg, "docs_outline", **body.model_dump()))


def _ollama_headers(request: Request) -> dict[str, str]:
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower()
        not in ("host", "connection", "transfer-encoding", "content-length", "authorization")
    }
    return headers


def _is_chat(path: str, method: str) -> bool:
    return method.upper() == "POST" and path.rstrip("/").endswith("chat/completions")


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def ollama_proxy(path: str, request: Request):
    if _is_chat(path, request.method):
        status = await model_status()
        if status["status"] != "up":
            return _openai_error(503, status["detail"])

    url = f"{_cfg.llm.base_url}/{path}"
    body = await request.body()
    stream = False
    if body:
        try:
            parsed = json.loads(body)
            stream = bool(parsed.get("stream")) if isinstance(parsed, dict) else False
        except json.JSONDecodeError:
            stream = False

    timeout = httpx.Timeout(300.0, connect=10.0)
    headers = _ollama_headers(request)

    if stream:
        client = httpx.AsyncClient(timeout=timeout)
        try:
            req = client.build_request(
                request.method,
                url,
                content=body,
                headers=headers,
                params=dict(request.query_params),
            )
            resp = await client.send(req, stream=True)
        except httpx.HTTPError:
            await client.aclose()
            return _openai_error(503, MODEL_DOWN)
        if resp.status_code >= 400:
            raw = await resp.aread()
            media = resp.headers.get("content-type", "application/json")
            await resp.aclose()
            await client.aclose()
            return Response(content=raw, status_code=resp.status_code, media_type=media)

        async def generate():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()

        return StreamingResponse(
            generate(),
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "text/event-stream"),
        )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(
                request.method,
                url,
                content=body,
                headers=headers,
                params=dict(request.query_params),
            )
    except httpx.HTTPError:
        return _openai_error(503, MODEL_DOWN)
    if resp.status_code >= 400:
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


if _cfg.paths.figures.exists():
    app.mount("/figures", StaticFiles(directory=str(_cfg.paths.figures)), name="figures")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)
