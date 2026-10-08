from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import orjson
import pytest

from cortex_llmhoster.app import _request_timeout, _response_headers, create_app
from cortex_llmhoster.config import ModelConfig, Settings


def test_http_timeout_profiles_are_reused_without_freezing_config_values() -> None:
    first = _request_timeout(1.0, 30.0, 5.0, 2.0)
    repeated = _request_timeout(1.0, 30.0, 5.0, 2.0)
    updated = _request_timeout(1.0, 60.0, 5.0, 2.0)

    assert repeated is first
    assert updated is not first
    assert updated.read == 60.0


def test_response_headers_filter_standard_and_connection_nominated_hop_headers() -> None:
    response = httpx.Response(
        200,
        headers={
            "content-type": "application/json",
            "connection": "x-internal, keep-alive",
            "x-internal": "discard",
            "x-visible": "keep",
        },
    )

    assert _response_headers(response) == {
        "content-type": "application/json",
        "x-visible": "keep",
    }


@pytest.mark.asyncio
async def test_hardware_probe_does_not_block_health_startup(monkeypatch) -> None:
    probe_started = threading.Event()
    release_probe = threading.Event()
    detected_hardware = {
        "platform": "test",
        "cpu_count": 2,
        "recommended_threads": 2,
        "ram_total_gb": 4,
        "ram_available_gb": 2,
        "gpu": "test accelerator",
    }

    def slow_hardware_probe() -> dict[str, object]:
        probe_started.set()
        release_probe.wait(timeout=5)
        return detected_hardware

    monkeypatch.setattr("cortex_llmhoster.app.detect_hardware", slow_hardware_probe)
    app = create_app(Settings())
    client_ready = asyncio.Event()
    observed: dict[str, object] = {}

    async def exercise_health() -> None:
        async with open_client(app) as client:
            observed["initial_gpu"] = app.state.runtime_manager.hardware["gpu"]
            client_ready.set()
            response = await client.get("/health")
            observed["health_status"] = response.status_code
            for _ in range(100):
                if app.state.runtime_manager.hardware["gpu"] == detected_hardware["gpu"]:
                    break
                await asyncio.sleep(0.01)
            observed["detected_gpu"] = app.state.runtime_manager.hardware["gpu"]

    health_task = asyncio.create_task(exercise_health())
    startup_nonblocking = False
    try:
        await asyncio.wait_for(client_ready.wait(), timeout=1)
        startup_nonblocking = True
    except TimeoutError:
        pass
    finally:
        release_probe.set()

    await asyncio.wait_for(health_task, timeout=5)
    assert startup_nonblocking
    assert probe_started.is_set()
    assert observed["initial_gpu"] == "Detecting hardware… (CPU inference is available)"
    assert observed["health_status"] == 200
    assert observed["detected_gpu"] == "test accelerator"


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "models": {
            "public-name": ModelConfig(
                id="public-name",
                upstream_model="runtime-model",
                model_path="/models/test.gguf",
            )
        },
        "default_model": "public-name",
        "max_json_body_bytes": 1024 * 1024,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


@asynccontextmanager
async def open_client(app, *, fake_local_server: bool = True) -> AsyncIterator[httpx.AsyncClient]:
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://cortex.test",
        ) as client,
    ):
        if fake_local_server:
            # Test the API adapter without launching a real llama-server process.
            app.state.runtime_manager.local_base_url = lambda model: "http://127.0.0.1:12345/v1"
        yield client


@pytest.mark.asyncio
async def test_chat_request_is_routed_and_alias_is_rewritten() -> None:
    observed: dict[str, object] = {}

    async def upstream(request: httpx.Request) -> httpx.Response:
        observed["url"] = str(request.url)
        observed["auth"] = request.headers.get("authorization")
        observed["organization"] = request.headers.get("openai-organization")
        observed["body"] = orjson.loads(await request.aread())
        return httpx.Response(200, json={"id": "answer", "choices": []})

    app = create_app(make_settings(api_key=None), transport=httpx.MockTransport(upstream))
    async with open_client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"openai-organization": "team-a"},
            json={"model": "public-name", "messages": [{"role": "user", "content": "Hi"}]},
        )

    assert response.status_code == 200
    assert response.json()["id"] == "answer"
    assert observed["url"] == "http://127.0.0.1:12345/v1/chat/completions"
    assert observed["auth"] is None
    assert observed["organization"] == "team-a"
    assert observed["body"]["model"] == "runtime-model"  # type: ignore[index]


@pytest.mark.asyncio
async def test_inference_capacity_is_bounded_and_released_after_streaming() -> None:
    first_request_started = asyncio.Event()
    release_first_request = asyncio.Event()
    upstream_calls = 0

    async def upstream(_: httpx.Request) -> httpx.Response:
        nonlocal upstream_calls
        upstream_calls += 1
        if upstream_calls == 1:
            first_request_started.set()
            await release_first_request.wait()
        return httpx.Response(200, json={"id": "answer", "choices": []})

    app = create_app(
        make_settings(max_inference_requests=1),
        transport=httpx.MockTransport(upstream),
    )
    async with open_client(app) as client:
        first_task = asyncio.create_task(
            client.post("/v1/chat/completions", json={"model": "public-name"})
        )
        await asyncio.wait_for(first_request_started.wait(), timeout=1)

        rejected = await client.post(
            "/v1/chat/completions",
            content=b"{invalid-json",
            headers={"content-type": "application/json"},
        )
        model_list = await client.get("/v1/models")
        assert rejected.status_code == 503
        assert rejected.json()["error"]["code"] == "server_capacity_exceeded"
        assert rejected.headers["retry-after"] == "1"
        assert model_list.status_code == 200

        release_first_request.set()
        completed = await first_task
        retried = await client.post("/v1/chat/completions", json={"model": "public-name"})

    assert completed.status_code == 200
    assert retried.status_code == 200
    assert upstream_calls == 2


@pytest.mark.asyncio
async def test_streaming_response_is_relayed_without_changing_sse_bytes() -> None:
    sent_chunks = [b'data: {"delta":1}\n\n', b"data: [DONE]\n\n"]

    class ChunkStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for chunk in sent_chunks:
                yield chunk

        async def aclose(self) -> None:
            return None

    async def upstream(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=ChunkStream(),
        )

    app = create_app(make_settings(api_key=None), transport=httpx.MockTransport(upstream))
    async with open_client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "public-name", "messages": [], "stream": True},
        )

    assert response.status_code == 200
    assert response.content == b"".join(sent_chunks)
    assert response.headers["content-type"] == "text/event-stream"
    assert response.headers["x-accel-buffering"] == "no"


@pytest.mark.asyncio
async def test_binary_upload_is_forwarded_and_query_can_select_model() -> None:
    observed: dict[str, object] = {}
    upload = b"--boundary\r\nfile-content\x00\xff\r\n--boundary--\r\n"

    async def upstream(request: httpx.Request) -> httpx.Response:
        observed["url"] = str(request.url)
        observed["body"] = await request.aread()
        observed["content_type"] = request.headers.get("content-type")
        return httpx.Response(204)

    app = create_app(make_settings(api_key=None), transport=httpx.MockTransport(upstream))
    async with open_client(app) as client:
        response = await client.post(
            "/v1/audio/transcriptions?%6dodel=public-name&lang=en",
            content=upload,
            headers={"content-type": "multipart/form-data; boundary=boundary"},
        )

    assert response.status_code == 204
    assert observed["url"] == (
        "http://127.0.0.1:12345/v1/audio/transcriptions?model=runtime-model&lang=en"
    )
    assert observed["body"] == upload
    assert observed["content_type"] == "multipart/form-data; boundary=boundary"


@pytest.mark.asyncio
async def test_json_default_model_is_injected_when_missing() -> None:
    async def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=orjson.loads(await request.aread()))

    app = create_app(make_settings(api_key=None), transport=httpx.MockTransport(upstream))
    async with open_client(app) as client:
        response = await client.post("/v1/embeddings", json={"input": "hello"})

    assert response.status_code == 200
    assert response.json()["model"] == "runtime-model"


@pytest.mark.asyncio
async def test_head_response_is_forwarded_without_a_body() -> None:
    async def upstream(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"would-be-response-body")

    app = create_app(make_settings(api_key=None), transport=httpx.MockTransport(upstream))
    async with open_client(app) as client:
        response = await client.head("/v1/models/runtime")

    assert response.status_code == 200
    assert response.content == b""


@pytest.mark.asyncio
async def test_non_inference_json_endpoint_is_not_modified() -> None:
    observed: dict[str, object] = {}

    async def upstream(request: httpx.Request) -> httpx.Response:
        observed["body"] = orjson.loads(await request.aread())
        return httpx.Response(200)

    app = create_app(make_settings(api_key=None), transport=httpx.MockTransport(upstream))
    async with open_client(app) as client:
        response = await client.post("/v1/assistants", json={"name": "helper"})

    assert response.status_code == 200
    assert observed["body"] == {"name": "helper"}


@pytest.mark.asyncio
async def test_unknown_model_returns_openai_error_without_calling_upstream() -> None:
    called = False

    async def upstream(_: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200)

    app = create_app(make_settings(api_key=None), transport=httpx.MockTransport(upstream))
    async with open_client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "does-not-exist", "messages": []},
        )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"
    assert called is False


@pytest.mark.asyncio
async def test_api_auth_protects_models_and_inference_routes() -> None:
    app = create_app(make_settings(api_key="cortex-secret"))
    async with open_client(app) as client:
        model_response = await client.get("/v1/models")
        inference_response = await client.post("/v1/chat/completions", json={})
        authorized = await client.get(
            "/v1/models", headers={"Authorization": "Bearer cortex-secret"}
        )

    assert model_response.status_code == 401
    assert model_response.headers["www-authenticate"] == "Bearer"
    assert inference_response.status_code == 401
    assert [item["id"] for item in authorized.json()["data"]] == ["public-name"]


@pytest.mark.asyncio
async def test_json_body_limit_is_enforced_before_model_server_call() -> None:
    called = False

    async def upstream(_: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200)

    app = create_app(
        make_settings(api_key=None, max_json_body_bytes=1024),
        transport=httpx.MockTransport(upstream),
    )
    async with open_client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=b'{"model":"public-name","prompt":"' + b"x" * 1200 + b'"}',
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"
    assert called is False


@pytest.mark.asyncio
async def test_chunked_json_limit_is_enforced_without_buffering_unbounded_body() -> None:
    called = False

    async def upstream(_: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200)

    async def chunks():
        yield b'{"model":"public-name","prompt":"'
        yield b"x" * 2048
        yield b'"}'

    app = create_app(
        make_settings(api_key=None, max_json_body_bytes=1024),
        transport=httpx.MockTransport(upstream),
    )
    async with open_client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=chunks(),
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 413
    assert called is False


@pytest.mark.asyncio
async def test_local_runtime_connection_failure_returns_service_unavailable() -> None:
    async def upstream(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable", request=request)

    app = create_app(make_settings(api_key=None), transport=httpx.MockTransport(upstream))
    async with open_client(app) as client:
        response = await client.post("/v1/chat/completions", json={"model": "public-name"})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "model_server_unavailable"


@pytest.mark.asyncio
async def test_health_and_readiness_are_lightweight() -> None:
    app = create_app(Settings())
    async with open_client(app) as client:
        health_response = await client.get("/health")
        readiness_response = await client.get("/ready")

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}
    assert readiness_response.status_code == 503
    assert readiness_response.json()["reason"] == "no_models_configured"


@pytest.mark.asyncio
async def test_readiness_requires_a_running_local_model_without_a_default() -> None:
    settings = Settings(
        models={
            "public-name": ModelConfig(
                id="public-name",
                upstream_model="runtime-model",
                model_path="/models/test.gguf",
            )
        }
    )
    app = create_app(settings)
    async with open_client(app, fake_local_server=False) as client:
        response = await client.get("/ready")

    assert response.status_code == 503
    assert response.json()["reason"] == "no_local_models_running"
