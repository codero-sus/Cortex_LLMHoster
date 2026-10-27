from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

from cortex_llmhoster.embedded_backend import BackendOptions, create_backend_app


class FakeLlama:
    def __init__(self) -> None:
        self.chat_calls: list[dict[str, object]] = []
        self.completion_calls: list[dict[str, object]] = []
        self.embedding_calls: list[dict[str, object]] = []
        self.closed = False

    def create_chat_completion(self, **kwargs: object) -> dict[str, object]:
        self.chat_calls.append(kwargs)
        return {"id": "chat-id", "model": "engine-name", "choices": []}

    def create_completion(self, **kwargs: object) -> dict[str, object]:
        self.completion_calls.append(kwargs)
        return {"id": "completion-id", "model": "engine-name", "choices": []}

    def create_embedding(self, **kwargs: object) -> dict[str, object]:
        self.embedding_calls.append(kwargs)
        return {"model": "engine-name", "data": []}

    def close(self) -> None:
        self.closed = True


@asynccontextmanager
async def open_backend_client(
    model_path: Path,
    engine: FakeLlama,
    *,
    embedding: bool = False,
) -> AsyncIterator[tuple[httpx.AsyncClient, FakeLlama]]:
    options = BackendOptions(
        model_path=str(model_path),
        model_alias="cortex-alias",
        embedding=embedding,
    )
    app = create_backend_app(options, llama_factory=lambda **_: engine)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://embedded.test",
        ) as client,
    ):
        yield client, engine


def make_model_file(tmp_path: Path) -> Path:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"test fixture")
    return model_path


@pytest.mark.asyncio
async def test_embedded_backend_serves_chat_completion_and_health(tmp_path: Path) -> None:
    engine = FakeLlama()
    model_path = make_model_file(tmp_path)

    async with open_backend_client(model_path, engine) as (client, _):
        health = await client.get("/health")
        chat = await client.post(
            "/v1/chat/completions",
            json={
                "model": "caller-selected-name",
                "messages": [{"role": "user", "content": "hello"}],
                "max_completion_tokens": 32,
                "frequency_penalty": 0.25,
                "presence_penalty": 0.5,
            },
        )
        completion = await client.post(
            "/v1/completions",
            json={"model": "caller-selected-name", "prompt": "hello"},
        )

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert health.headers["cache-control"] == "no-store"
    assert chat.status_code == 200
    assert chat.json()["model"] == "cortex-alias"
    assert engine.chat_calls == [
        {
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 32,
            "frequency_penalty": 0.25,
            "presence_penalty": 0.5,
        }
    ]
    assert completion.status_code == 200
    assert completion.json()["model"] == "cortex-alias"
    assert engine.completion_calls == [{"prompt": "hello"}]
    assert engine.closed is True


@pytest.mark.asyncio
async def test_embedded_backend_rejects_disabled_embeddings_and_invalid_json(
    tmp_path: Path,
) -> None:
    engine = FakeLlama()
    model_path = make_model_file(tmp_path)

    async with open_backend_client(model_path, engine) as (client, _):
        embeddings = await client.post("/v1/embeddings", json={"input": "hello"})
        invalid_json = await client.post(
            "/v1/chat/completions",
            content=b"[]",
            headers={"content-type": "application/json"},
        )
        invalid_length = await client.post(
            "/v1/chat/completions",
            content=b"{}",
            headers={"content-type": "application/json", "content-length": "not-a-number"},
        )
        oversized_body = await client.post(
            "/v1/chat/completions",
            content=b"{}",
            headers={
                "content-type": "application/json",
                "content-length": str(16 * 1024 * 1024 + 1),
            },
        )
        invalid_stream_type = await client.post(
            "/v1/chat/completions",
            json={"messages": [], "stream": "true"},
        )

    assert embeddings.status_code == 422
    assert embeddings.json()["error"]["code"] == "unsupported_model_capability"
    assert invalid_json.status_code == 400
    assert invalid_length.status_code == 400
    assert oversized_body.status_code == 413
    assert invalid_stream_type.status_code == 400
    assert engine.embedding_calls == []


@pytest.mark.asyncio
async def test_embedded_backend_enables_embeddings_and_streams_openai_sse(tmp_path: Path) -> None:
    engine = FakeLlama()
    engine.create_chat_completion = lambda **kwargs: iter(  # type: ignore[method-assign]
        [
            {"id": "stream-1", "model": "engine-name", "choices": [{"delta": {"content": "hi"}}]},
            {
                "id": "stream-1",
                "model": "engine-name",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
            },
        ]
    )
    model_path = make_model_file(tmp_path)

    async with open_backend_client(model_path, engine, embedding=True) as (client, _):
        embeddings = await client.post("/v1/embeddings", json={"model": "x", "input": "hi"})
        stream = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )

    assert embeddings.status_code == 200
    assert embeddings.json()["model"] == "cortex-alias"
    assert engine.embedding_calls == [{"input": "hi"}]
    assert stream.status_code == 200
    assert stream.headers["content-type"].startswith("text/event-stream")
    assert '"model":"cortex-alias"' in stream.text
    assert "data: [DONE]" in stream.text


@pytest.mark.asyncio
async def test_embedded_backend_sanitizes_model_parameter_errors(tmp_path: Path) -> None:
    engine = FakeLlama()

    def fail_on_invalid_input(**_: object) -> dict[str, object]:
        raise ValueError("private engine detail")

    engine.create_chat_completion = fail_on_invalid_input  # type: ignore[method-assign]
    model_path = make_model_file(tmp_path)

    async with open_backend_client(model_path, engine) as (client, _):
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
    assert "private engine detail" not in response.text


@pytest.mark.asyncio
async def test_embedded_backend_requires_a_real_gguf_file(tmp_path: Path) -> None:
    non_gguf = tmp_path / "model.bin"
    non_gguf.write_bytes(b"fixture")
    app = create_backend_app(
        BackendOptions(model_path=str(non_gguf), model_alias="broken"),
        llama_factory=lambda **_: FakeLlama(),
    )

    with pytest.raises(RuntimeError, match="existing GGUF model file"):
        async with app.router.lifespan_context(app):
            pass


@pytest.mark.asyncio
async def test_embedded_backend_explains_missing_optional_binding(
    tmp_path: Path, monkeypatch
) -> None:
    model_path = make_model_file(tmp_path)
    app = create_backend_app(
        BackendOptions(model_path=str(model_path), model_alias="missing-binding")
    )
    monkeypatch.setitem(sys.modules, "llama_cpp", None)

    with pytest.raises(RuntimeError, match=r"Install Cortex with the \[embedded\] extra"):
        async with app.router.lifespan_context(app):
            pass
