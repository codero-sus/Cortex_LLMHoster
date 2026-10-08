from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

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


def admin_headers() -> dict[str, str]:
    return {"Authorization": "Bearer dashboard-secret"}


def configured_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "models": {
            "public-model": ModelConfig(
                id="public-model",
                upstream_model="engine-alias",
                model_path="/models/not-present.gguf",
            )
        },
        "default_model": "public-model",
        "api_key": "dashboard-secret",
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_dashboard_is_served_and_admin_api_requires_key(tmp_path) -> None:
    app = create_app(Settings(), admin_config_path=tmp_path / "cortex.local.json")
    async with open_client(app) as client:
        page = await client.get("/")
        state = await client.get("/admin/api/state")

    assert page.status_code == 200
    assert "Cortex — Local LLM Hoster" in page.text
    assert "Chat playground" in page.text
    assert "ULTRA (+1)" in page.text
    assert page.headers["x-frame-options"] == "DENY"
    assert state.status_code == 503
    assert state.json()["error"]["code"] == "admin_auth_not_configured"


@pytest.mark.asyncio
async def test_authenticated_state_and_metrics_are_available(tmp_path) -> None:
    app = create_app(configured_settings(), admin_config_path=tmp_path / "config.json")
    async with open_client(app) as client:
        state = await client.get("/admin/api/state", headers=admin_headers())
        metrics = await client.get("/admin/api/metrics", headers=admin_headers())
        denied = await client.get("/admin/api/state")

    assert state.status_code == 200
    assert state.json()["models"][0]["id"] == "public-model"
    assert state.json()["models"][0]["runtime"] == "llama.cpp"
    assert "dashboard-secret" not in state.text
    assert metrics.status_code == 200
    assert metrics.json()["in_flight"] == 0
    assert denied.status_code == 401


@pytest.mark.asyncio
async def test_config_update_is_atomic_live_and_secret_free(tmp_path) -> None:
    local_config = tmp_path / "cortex.local.json"
    app = create_app(configured_settings(), admin_config_path=local_config)
    payload = {
        "server": {
            "default_model": "fast-alias",
            "max_json_body_bytes": 2_000_000,
            "max_inference_requests": 4,
        },
        "models": [
            {
                "id": "fast-alias",
                "model_path": "/models/fast-q4.gguf",
                "upstream_model": "fast-alias-runtime",
                "default": True,
            }
        ],
    }

    async with open_client(app) as client:
        saved = await client.put("/admin/api/config", headers=admin_headers(), json=payload)
        state = await client.get("/admin/api/state", headers=admin_headers())

    assert saved.status_code == 200
    assert saved.json()["ok"] is True
    assert state.json()["server"]["default_model"] == "fast-alias"
    assert state.json()["server"]["max_inference_requests"] == 4
    assert "max_inference_requests" not in saved.json()["restart_required"]
    assert app.state.inference_limiter.limit == 4
    assert state.json()["models"][0]["upstream_model"] == "fast-alias-runtime"
    assert app.state.settings.api_key == "dashboard-secret"
    stored = local_config.read_text(encoding="utf-8")
    assert "fast-q4.gguf" in stored
    assert "dashboard-secret" not in stored
    assert json.loads(stored)["models"][0]["default"] is True


@pytest.mark.asyncio
async def test_local_model_fields_round_trip_through_dashboard(tmp_path) -> None:
    local_config = tmp_path / "cortex.local.json"
    app = create_app(configured_settings(), admin_config_path=local_config)
    payload = {
        "server": {"default_model": "tiny-local", "workers": 1},
        "models": [
            {
                "id": "tiny-local",
                "runtime": "llama.cpp",
                "optimization_level": 1,
                "model_path": "/models/tiny-q4.gguf",
                "mmproj_path": "/models/projector.gguf",
                "gpu_layers": -1,
                "context_size": 1024,
                "batch_size": 128,
                "ubatch_size": 32,
                "parallel": 1,
                "embedding": True,
                "default": True,
            }
        ],
    }

    async with open_client(app) as client:
        saved = await client.put("/admin/api/config", headers=admin_headers(), json=payload)
        state = await client.get("/admin/api/state", headers=admin_headers())
        exported = await client.get("/admin/api/config/export", headers=admin_headers())

    assert saved.status_code == 200
    local_model = state.json()["models"][0]
    assert local_model["runtime"] == "llama.cpp"
    assert local_model["optimization_level"] == 1
    assert local_model["gpu_layers"] == -1
    assert local_model["context_size"] == 1024
    assert local_model["embedding"] is True
    assert local_model["mmproj_path"] == "/models/projector.gguf"
    assert state.json()["runtime"]["models"]["tiny-local"]["status"] == "error"
    stored_config = json.loads(local_config.read_text(encoding="utf-8"))
    assert "dashboard-secret" not in json.dumps(stored_config)
    assert stored_config["models"][0]["optimization_level"] == 1
    assert exported.status_code == 200
    assert exported.json()["models"][0]["runtime"] == "llama.cpp"
    assert exported.json()["models"][0]["optimization_level"] == 1
    assert exported.json()["models"][0]["embedding"] is True


@pytest.mark.asyncio
async def test_invalid_config_does_not_replace_live_config_or_write_file(tmp_path) -> None:
    local_config = tmp_path / "cortex.local.json"
    app = create_app(configured_settings(), admin_config_path=local_config)
    payload = {
        "server": {"default_model": "missing"},
        "models": [
            {
                "id": "valid",
                "model_path": "/models/valid.gguf",
                "default": True,
            }
        ],
    }
    async with open_client(app) as client:
        response = await client.put("/admin/api/config", headers=admin_headers(), json=payload)

    assert response.status_code == 422
    assert app.state.settings.default_model == "public-model"
    assert not local_config.exists()


def test_saved_ui_overlay_is_loaded_on_restart(tmp_path) -> None:
    base_config = tmp_path / "base.toml"
    base_config.write_text("[server]\n", encoding="utf-8")
    local_config = tmp_path / "cortex.local.json"
    local_config.write_text(
        json.dumps(
            {
                "server": {"default_model": "from-ui", "api_key_env": "CORTEX_API_KEY"},
                "models": [
                    {
                        "id": "from-ui",
                        "runtime": "llama.cpp",
                        "model_path": "/models/from-ui.gguf",
                        "upstream_model": "runtime-name",
                        "default": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    settings = Settings.load(
        base_config,
        environ={
            "CORTEX_ADMIN_CONFIG": str(local_config),
            "CORTEX_API_KEY": "dashboard-secret",
        },
    )

    assert settings.default_model == "from-ui"
    assert settings.models["from-ui"].model_path == "/models/from-ui.gguf"
    assert settings.api_key == "dashboard-secret"


@pytest.mark.asyncio
async def test_local_model_setup_check_uses_gguf_and_llama_server_paths(tmp_path) -> None:
    model_path = tmp_path / "tiny.gguf"
    model_path.write_bytes(b"stub model")
    executable = tmp_path / "llama-server"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    app = create_app(
        configured_settings(llama_server_path=str(executable)),
        admin_config_path=tmp_path / "config.json",
    )
    payload = {
        "id": "new-local",
        "runtime": "llama.cpp",
        "model_path": str(model_path),
        "upstream_model": "new-local",
    }
    async with open_client(app) as client:
        response = await client.post(
            "/admin/api/models/test", headers=admin_headers(), json=payload
        )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["model_file_exists"] is True
    assert response.json()["runner"] == str(executable)
    assert "stub model" not in response.text
