from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

from cortex_llmhoster.app import create_app
from cortex_llmhoster.config import ModelConfig, Settings


@asynccontextmanager
async def open_client(app) -> AsyncIterator[httpx.AsyncClient]:
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://cortex.test",
        ) as client,
    ):
        yield client


def fake_server_file(tmp_path: Path) -> Path:
    executable = tmp_path / "llama-server"
    executable.write_text(
        "#!/usr/bin/env python3\nimport time\nprint('stub server started', flush=True)\ntime.sleep(60)\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


@pytest.mark.asyncio
async def test_local_model_is_managed_and_served_through_cortex(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "tiny.gguf"
    model_path.write_bytes(b"stub model")
    fake_server = fake_server_file(tmp_path)
    observed: dict[str, object] = {}

    async def mock_upstream(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        observed["url"] = str(request.url)
        observed["body"] = await request.aread()
        return httpx.Response(200, json={"id": "local-result", "choices": []})

    monkeypatch.setattr(
        "cortex_llmhoster.app.detect_hardware",
        lambda: {
            "platform": "test",
            "cpu_count": 2,
            "recommended_threads": 2,
            "ram_total_gb": 4,
            "ram_available_gb": 2,
            "gpu": "test CPU",
        },
    )
    settings = Settings(
        models={
            "tiny-local": ModelConfig(
                id="tiny-local",
                upstream_model="tiny-local",
                runtime="llama.cpp",
                model_path=str(model_path),
                gpu_layers=0,
                context_size=512,
                batch_size=64,
                ubatch_size=32,
            )
        },
        default_model="tiny-local",
        api_key="dashboard-secret",
        llama_server_path=str(fake_server),
    )
    app = create_app(settings, transport=httpx.MockTransport(mock_upstream))

    async with open_client(app) as client:
        manager = app.state.runtime_manager
        for _ in range(60):
            if manager.state("tiny-local")["status"] == "running":
                break
            await asyncio.sleep(0.05)
        assert manager.state("tiny-local")["status"] == "running"

        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer dashboard-secret"},
            json={
                "model": "tiny-local",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )
        assert response.status_code == 200
        assert response.json()["id"] == "local-result"
        assert str(observed["url"]).startswith("http://127.0.0.1:")
        assert str(observed["url"]).endswith("/v1/chat/completions")

        status = await client.get(
            "/admin/api/runtime", headers={"Authorization": "Bearer dashboard-secret"}
        )
        assert status.status_code == 200
        assert status.json()["models"]["tiny-local"]["status"] == "running"

        stopped = await client.post(
            "/admin/api/models/stop",
            headers={"Authorization": "Bearer dashboard-secret"},
            json={"id": "tiny-local"},
        )
        assert stopped.status_code == 200
        assert stopped.json()["runtime"]["status"] == "stopped"

        readiness = await client.get("/ready")
        assert readiness.status_code == 503
        assert readiness.json()["reason"] == "default_local_model_not_running"
