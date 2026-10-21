from __future__ import annotations

import asyncio
import base64
import json
import stat
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import orjson
import pytest
from starlette.requests import ClientDisconnect, Request

from cortex_llmhoster.app import create_app
from cortex_llmhoster.config import ModelConfig, Settings
from cortex_llmhoster.dashboard import (
    LoginAttemptLimiter,
    _inline_script_hashes,
    _read_json,
    _write_json,
)


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


@pytest.mark.asyncio
async def test_admin_json_reader_reports_client_disconnect_as_a_bad_request() -> None:
    async def disconnected_receive():
        raise ClientDisconnect

    request = Request(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/admin/api/config",
            "raw_path": b"/admin/api/config",
            "query_string": b"",
            "headers": [],
            "client": ("testclient", 123),
            "server": ("testserver", 80),
        },
        receive=disconnected_receive,
    )

    payload, error = await _read_json(request)

    assert payload is None
    assert error is not None
    assert error.status_code == 400
    assert orjson.loads(error.body)["error"]["code"] == "request_body_interrupted"


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


def update_metadata_transport(version: str) -> httpx.MockTransport:
    async def upstream(request: httpx.Request) -> httpx.Response:
        if "/branches/" in request.url.path:
            return httpx.Response(
                200,
                json={"commit": {"sha": "b" * 40}},
            )
        if request.url.path.endswith("/contents/pyproject.toml"):
            content = f'[project]\nversion = "{version}"\n'.encode()
            return httpx.Response(
                200,
                json={"content": base64.b64encode(content).decode("ascii")},
            )
        return httpx.Response(404)

    return httpx.MockTransport(upstream)


@pytest.mark.asyncio
async def test_updater_routes_require_a_key_when_authentication_is_enabled(tmp_path) -> None:
    app = create_app(
        Settings(api_auth_enabled=True), admin_config_path=tmp_path / "cortex.local.json"
    )
    async with open_client(app) as client:
        response = await client.get("/admin/api/update/check")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "admin_auth_not_configured"


@pytest.mark.asyncio
async def test_authenticated_updater_checks_and_installs_pinned_branch(
    tmp_path, monkeypatch
) -> None:
    app = create_app(
        configured_settings(),
        transport=update_metadata_transport("0.3.0"),
        admin_config_path=tmp_path / "config.json",
    )
    installed: list[tuple[str, str]] = []

    async def fake_install(commit_sha: str, *, api_key_env: str) -> str:
        installed.append((commit_sha, api_key_env))
        return "Successfully installed cortex-llmhoster"

    monkeypatch.setattr("cortex_llmhoster.dashboard.install_from_commit", fake_install)
    async with open_client(app) as client:
        checked = await client.get("/admin/api/update/check", headers=admin_headers())
        result = await client.post(
            "/admin/api/update/install",
            headers=admin_headers(),
            json={"commit_sha": "b" * 40},
        )

    assert checked.status_code == 200
    assert checked.json()["update_available"] is True
    assert checked.json()["latest_version"] == "0.3.0"
    assert result.status_code == 200
    assert result.json()["installed_version"] == "0.3.0"
    assert result.json()["restart_required"] is True
    assert installed == [("b" * 40, "CORTEX_API_KEY")]


@pytest.mark.asyncio
async def test_updater_refuses_multiworker_installation(tmp_path) -> None:
    app = create_app(
        configured_settings(workers=2),
        admin_config_path=tmp_path / "config.json",
    )
    async with open_client(app) as client:
        response = await client.post(
            "/admin/api/update/install",
            headers=admin_headers(),
        )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "update_requires_single_worker"


@pytest.mark.asyncio
async def test_updater_will_not_install_a_commit_other_than_the_confirmed_one(
    tmp_path, monkeypatch
) -> None:
    app = create_app(
        configured_settings(),
        transport=update_metadata_transport("0.3.0"),
        admin_config_path=tmp_path / "config.json",
    )

    async def unexpected_install(_: str) -> str:
        raise AssertionError("a changed source must not be installed")

    monkeypatch.setattr("cortex_llmhoster.dashboard.install_from_commit", unexpected_install)
    async with open_client(app) as client:
        response = await client.post(
            "/admin/api/update/install",
            headers=admin_headers(),
            json={"commit_sha": "c" * 40},
        )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "update_source_changed"


@pytest.mark.asyncio
async def test_dashboard_is_served_and_auth_enabled_requires_key(tmp_path) -> None:
    app = create_app(
        Settings(api_auth_enabled=True), admin_config_path=tmp_path / "cortex.local.json"
    )
    async with open_client(app) as client:
        page = await client.get("/")
        state = await client.get("/admin/api/state")

    assert page.status_code == 200
    content_security_policy = page.headers["content-security-policy"]
    script_policy = content_security_policy.split("script-src ", 1)[1].split(";", 1)[0]
    assert "'unsafe-inline'" not in script_policy
    script_hashes = _inline_script_hashes(page.text)
    assert script_hashes
    assert all(source_hash in script_policy for source_hash in script_hashes)
    assert "script-src-attr 'none'" in content_security_policy
    assert "form-action 'self'" in content_security_policy
    assert "Cortex LLMHoster — Free Local Model Hoster" in page.text
    assert "https://github.com/codero-sus" in page.text
    assert "https://the-codero.web.app/" in page.text
    assert "Chat playground" in page.text
    assert "Use ULTRA tuning (optional)" in page.text
    assert "Choose supported tasks (optional)" in page.text
    assert 'id="browse-model-file"' in page.text
    assert 'id="model-file-input"' in page.text
    assert "Browse this device…" in page.text
    assert "Use a GGUF already on this Cortex server" in page.text
    assert "/admin/api/models/files/upload" in page.text
    assert "xhr.send(file)" in page.text
    assert 'id="connection-action"' in page.text
    assert "Server offline" in page.text
    assert "Open docs" in page.text
    assert "error.status=401" in page.text
    assert "event.isComposing" in page.text
    assert 'id="model-file-select"' in page.text
    assert 'id="manual-model-path-toggle"' in page.text
    assert "Enter a model path manually" in page.text
    assert "/admin/api/models/files" in page.text
    assert "Auto-detect tasks" in page.text
    assert "/admin/api/models/detect-capabilities" in page.text
    assert 'data-page="docs"' in page.text
    assert "Cortex LLMHoster Docs" in page.text
    assert "2PY2=/absolute/path/to/python3.12" in page.text
    assert 'name="capability_choice"' in page.text
    assert 'name="custom_capabilities"' in page.text
    assert "Text-to-image · T2I" in page.text
    assert "Image-to-video · I2V" in page.text
    assert "Speech-to-speech" in page.text
    assert "Search models" in page.text
    assert "Make default" in page.text
    assert "stop-all-models" in page.text
    assert "Balanced (Auto)" in page.text
    for mode in ("Low−", "Low+", "Medium−", "Medium+", "High−", "High+"):
        assert mode in page.text
    assert "{python}" in page.text
    assert 'id="server-profile-field"' in page.text
    assert "vLLM · GPU throughput and batching" in page.text
    assert "SGLang · high-throughput serving" in page.text
    assert "Cortex native C++ GGUF backend" in page.text
    assert "cortex-cpp" in page.text
    assert "conversion_outtype" in page.text
    assert "allow_unsafe_legacy_checkpoint_conversion" in page.text
    assert "Pickle-based checkpoints can execute code" in page.text
    assert "CORTEX_CPP_HOST" in page.text
    assert 'id="frequency-penalty"' in page.text
    assert 'id="presence-penalty"' in page.text
    assert "frequency_penalty:Number" in page.text
    assert "presence_penalty:Number" in page.text
    assert "Hugging Face TGI · Transformers serving" in page.text
    assert "MLX-LM · Apple Silicon" in page.text
    assert "Templates only prefill commands" in page.text
    assert 'aria-labelledby="dialog-title"' in page.text
    assert "Local admin username" in page.text
    assert "Use API key instead" in page.text
    assert "Authentication-enabled mode requires a local API key." in page.text
    assert "No-auth mode (use instead of setting CORTEX_API_KEY)" in page.text
    assert "--no-auth" in page.text
    assert "When off, no API key or login is required" in page.text
    assert 'placeholder="Local API key"' in page.text
    assert "requestAnimationFrame" in page.text
    assert "setAttribute('aria-current','page')" in page.text
    assert page.headers["x-frame-options"] == "DENY"
    assert state.status_code == 503
    assert state.json()["error"]["code"] == "admin_auth_not_configured"


@pytest.mark.asyncio
async def test_admin_api_responses_are_never_cached(tmp_path) -> None:
    app = create_app(configured_settings(), admin_config_path=tmp_path / "config.json")
    async with open_client(app) as client:
        responses = [
            await client.get("/admin/api/auth/options"),
            await client.get("/admin/api/state", headers=admin_headers()),
            await client.get("/admin/api/metrics", headers=admin_headers()),
            await client.get("/admin/api/config/export", headers=admin_headers()),
            await client.get("/admin/api/state"),
        ]

    assert [response.status_code for response in responses] == [200, 200, 200, 200, 401]
    for response in responses:
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["pragma"] == "no-cache"
        assert response.headers["x-content-type-options"] == "nosniff"


@pytest.mark.asyncio
async def test_disabled_auth_allows_admin_and_inference_without_a_key(tmp_path) -> None:
    app = create_app(Settings(api_auth_enabled=False), admin_config_path=tmp_path / "config.json")
    async with open_client(app) as client:
        state = await client.get("/admin/api/state")
        models = await client.get("/v1/models")

    assert state.status_code == 200
    assert state.json()["server"]["api_auth_enabled"] is False
    assert models.status_code == 200
    assert models.json()["data"] == []


@pytest.mark.parametrize("content_length", ("-1", "+1", "1.0", "1, 1", "not-a-number", ""))
@pytest.mark.asyncio
async def test_admin_json_reader_rejects_invalid_content_length(
    tmp_path, content_length: str
) -> None:
    app = create_app(configured_settings(), admin_config_path=tmp_path / "config.json")
    async with open_client(app) as client:
        response = await client.post(
            "/admin/api/auth/login",
            content=b"{}",
            headers={"content-type": "application/json", "content-length": content_length},
        )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Invalid Content-Length header."


@pytest.mark.asyncio
async def test_login_endpoint_throttles_repeated_invalid_credentials(tmp_path) -> None:
    app = create_app(configured_settings(), admin_config_path=tmp_path / "config.json")
    async with open_client(app) as client:
        failures = [
            await client.post("/admin/api/auth/login", json={"api_key": "wrong-key"})
            for _ in range(10)
        ]
        blocked = await client.post("/admin/api/auth/login", json={"api_key": "dashboard-secret"})

    assert all(response.status_code == 401 for response in failures)
    assert blocked.status_code == 429
    assert blocked.json()["error"]["code"] == "login_rate_limited"
    assert int(blocked.headers["retry-after"]) >= 1


@pytest.mark.asyncio
async def test_login_rate_limit_reserves_concurrent_attempts_before_body_parsing(tmp_path) -> None:
    app = create_app(configured_settings(), admin_config_path=tmp_path / "config.json")
    async with open_client(app) as client:
        responses = await asyncio.gather(
            *(
                client.post("/admin/api/auth/login", json={"api_key": "wrong-key"})
                for _ in range(20)
            )
        )

    assert sum(response.status_code == 401 for response in responses) == 10
    assert sum(response.status_code == 429 for response in responses) == 10
    assert all(
        response.json()["error"]["code"] == "login_rate_limited"
        for response in responses
        if response.status_code == 429
    )


def test_login_attempt_limiter_expires_old_failures_and_bounds_client_storage() -> None:
    limiter = LoginAttemptLimiter(max_attempts=2, window_seconds=10, max_clients=2)
    limiter.record_failure("client-a", now=1)
    assert limiter.retry_after("client-a", now=1) is None
    limiter.record_failure("client-a", now=2)
    assert limiter.retry_after("client-a", now=2) == 9
    assert limiter.retry_after("client-a", now=11) is None

    limiter.record_failure("client-b", now=20)
    limiter.record_failure("client-c", now=21)
    assert len(limiter._failures) == 2
    assert limiter.retry_after("client-a", now=21) is None


@pytest.mark.asyncio
async def test_dashboard_login_sets_persistent_http_only_session_cookie(tmp_path) -> None:
    app = create_app(configured_settings(), admin_config_path=tmp_path / "config.json")
    async with open_client(app) as client:
        options = await client.get("/admin/api/auth/options")
        invalid = await client.post("/admin/api/auth/login", json={"api_key": "not-the-key"})
        login = await client.post("/admin/api/auth/login", json={"api_key": "dashboard-secret"})
        state = await client.get("/admin/api/state")
        inference_models = await client.get("/v1/models")
        logout = await client.post(
            "/admin/api/auth/logout", headers={"origin": "http://cortex.test"}
        )
        denied_after_logout = await client.get("/admin/api/state")

    cookie = login.headers["set-cookie"]
    assert options.status_code == 200
    assert options.json() == {
        "api_auth_enabled": True,
        "account_login_enabled": False,
        "api_key_login_enabled": True,
    }
    assert invalid.status_code == 401
    assert login.status_code == 200
    assert "httponly" in cookie.lower()
    assert "max-age=2592000" in cookie.lower()
    assert "samesite=lax" in cookie.lower()
    assert "dashboard-secret" not in cookie
    assert login.headers["cache-control"] == "no-store"
    assert state.status_code == 200
    assert inference_models.status_code == 200
    assert logout.status_code == 200
    assert denied_after_logout.status_code == 401


@pytest.mark.asyncio
async def test_account_login_issues_session_and_bearer_api_key_remains_supported(tmp_path) -> None:
    app = create_app(
        configured_settings(
            admin_username="local-admin",
            admin_password="correct-horse-battery-staple",
        ),
        admin_config_path=tmp_path / "config.json",
    )
    async with open_client(app) as client:
        options = await client.get("/admin/api/auth/options")
        invalid = await client.post(
            "/admin/api/auth/login",
            json={"username": "local-admin", "password": "incorrect"},
        )
        login = await client.post(
            "/admin/api/auth/login",
            json={"username": "local-admin", "password": "correct-horse-battery-staple"},
        )
        state = await client.get("/admin/api/state")
        inference_models = await client.get("/v1/models")
        await client.post("/admin/api/auth/logout", headers={"origin": "http://cortex.test"})
        bearer_state = await client.get("/admin/api/state", headers=admin_headers())

    cookie = login.headers["set-cookie"]
    assert options.status_code == 200
    assert options.json() == {
        "api_auth_enabled": True,
        "account_login_enabled": True,
        "api_key_login_enabled": True,
    }
    assert invalid.status_code == 401
    assert invalid.json()["error"]["message"] == "The account credentials or API key are invalid."
    assert login.status_code == 200
    assert "httponly" in cookie.lower()
    assert "local-admin" not in cookie
    assert "correct-horse-battery-staple" not in cookie
    assert state.status_code == 200
    assert "correct-horse-battery-staple" not in state.text
    assert "admin_password" not in state.json()["server"]
    assert inference_models.status_code == 200
    assert bearer_state.status_code == 200


@pytest.mark.asyncio
async def test_https_dashboard_cookie_supports_embedded_preview_and_partitioning(tmp_path) -> None:
    app = create_app(configured_settings(), admin_config_path=tmp_path / "config.json")
    async with open_client(app) as client:
        response = await client.post(
            "/admin/api/auth/login",
            json={"api_key": "dashboard-secret"},
            headers={"x-forwarded-proto": "https"},
        )

    cookie = response.headers["set-cookie"].lower()
    assert response.status_code == 200
    assert "secure" in cookie
    assert "httponly" in cookie
    assert "samesite=none" in cookie
    assert "partitioned" in cookie


@pytest.mark.asyncio
async def test_cross_origin_cookie_mutation_is_rejected(tmp_path) -> None:
    app = create_app(configured_settings(), admin_config_path=tmp_path / "config.json")
    async with open_client(app) as client:
        login = await client.post("/admin/api/auth/login", json={"api_key": "dashboard-secret"})
        csrf = await client.post(
            "/admin/api/auth/logout", headers={"origin": "https://attacker.invalid"}
        )
        still_authenticated = await client.get("/admin/api/state")

    assert login.status_code == 200
    assert csrf.status_code == 403
    assert csrf.json()["error"]["code"] == "csrf_origin_rejected"
    assert still_authenticated.status_code == 200


@pytest.mark.parametrize(
    ("headers", "expected_status"),
    [
        ({"origin": "http://cortex.test"}, 200),
        ({"origin": "http://CORTEX.TEST:80"}, 200),
        ({"referer": "http://cortex.test/settings"}, 200),
        ({"sec-fetch-site": "same-origin"}, 200),
        (
            {
                "origin": "https://preview.cortex.test",
                "x-forwarded-proto": "https",
                "x-forwarded-host": "preview.cortex.test",
            },
            200,
        ),
        ({}, 403),
        ({"sec-fetch-site": "same-site"}, 403),
        ({"sec-fetch-site": "cross-site"}, 403),
        ({"origin": "null"}, 403),
        ({"origin": "http://cortex.test/settings"}, 403),
        ({"origin": "http://attacker.invalid"}, 403),
        ({"referer": "https://attacker.invalid/"}, 403),
        ({"origin": "http://[invalid"}, 403),
    ],
)
@pytest.mark.asyncio
async def test_cookie_mutations_require_verifiable_same_origin_metadata(
    tmp_path, headers: dict[str, str], expected_status: int
) -> None:
    app = create_app(configured_settings(), admin_config_path=tmp_path / "config.json")
    async with open_client(app) as client:
        login = await client.post("/admin/api/auth/login", json={"api_key": "dashboard-secret"})
        mutation = await client.post("/admin/api/auth/logout", headers=headers)

    assert login.status_code == 200
    assert mutation.status_code == expected_status
    if expected_status == 403:
        assert mutation.json()["error"]["code"] == "csrf_origin_rejected"


def test_atomic_config_write_ignores_predictable_temp_symlinks_and_sets_private_mode(
    tmp_path,
) -> None:
    destination = tmp_path / "cortex.local.json"
    victim = tmp_path / "must-not-be-overwritten.txt"
    victim.write_text("keep this file", encoding="utf-8")
    predictable_temp = tmp_path / f".{destination.name}.tmp"
    try:
        predictable_temp.symlink_to(victim)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    _write_json(destination, {"server": {"port": 9000}})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"server": {"port": 9000}}
    assert victim.read_text(encoding="utf-8") == "keep this file"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert predictable_temp.is_symlink()


@pytest.mark.asyncio
async def test_reset_config_validates_base_before_removing_saved_overlay(
    tmp_path, monkeypatch
) -> None:
    base_config = tmp_path / "broken.toml"
    base_config.write_text("[server\nport = 9000\n", encoding="utf-8")
    monkeypatch.setenv("CORTEX_CONFIG", str(base_config))
    monkeypatch.delenv("CORTEX_API_KEY", raising=False)
    overlay = tmp_path / "cortex.local.json"
    original_overlay = b'{"server":{"port":9100},"models":[]}\n'
    overlay.write_bytes(original_overlay)
    app = create_app(configured_settings(), admin_config_path=overlay)

    async with open_client(app) as client:
        response = await client.post("/admin/api/config/reset", headers=admin_headers())

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "config_reset_failed"
    assert overlay.read_bytes() == original_overlay
    assert app.state.settings.port == configured_settings().port


@pytest.mark.asyncio
async def test_reset_config_removes_overlay_after_valid_base_is_loaded(
    tmp_path, monkeypatch
) -> None:
    base_config = tmp_path / "base.toml"
    base_config.write_text("[server]\nport = 9123\napi_auth_enabled = false\n", encoding="utf-8")
    monkeypatch.setenv("CORTEX_CONFIG", str(base_config))
    monkeypatch.delenv("CORTEX_API_KEY", raising=False)
    overlay = tmp_path / "cortex.local.json"
    overlay.write_text('{"server":{"port":9100},"models":[]}', encoding="utf-8")
    app = create_app(configured_settings(), admin_config_path=overlay)

    async with open_client(app) as client:
        response = await client.post("/admin/api/config/reset", headers=admin_headers())

    assert response.status_code == 200
    assert response.json()["server"]["port"] == 9123
    assert not overlay.exists()
    assert app.state.settings.port == 9123


@pytest.mark.asyncio
async def test_auth_toggle_persists_without_writing_the_local_secret(tmp_path) -> None:
    settings = configured_settings(api_auth_enabled=False)
    config_path = tmp_path / "config.json"
    app = create_app(settings, admin_config_path=config_path)
    async with open_client(app) as client:
        current = await client.get("/admin/api/state")
        payload = current.json()
        payload["server"]["api_auth_enabled"] = True
        saved = await client.put("/admin/api/config", json=payload)
        denied = await client.get("/admin/api/state")
        login = await client.post("/admin/api/auth/login", json={"api_key": "dashboard-secret"})
        authorized = await client.get("/admin/api/state")

    saved_json = config_path.read_text(encoding="utf-8")
    assert current.status_code == 200
    assert saved.status_code == 200
    assert denied.status_code == 401
    assert login.status_code == 200
    assert authorized.status_code == 200
    assert '"api_auth_enabled": true' in saved_json
    assert "dashboard-secret" not in saved_json


@pytest.mark.asyncio
async def test_model_cookbook_curated_offline_catalog_remains_available(tmp_path) -> None:
    app = create_app(Settings(), admin_config_path=tmp_path / "config.json")
    async with open_client(app) as client:
        response = await client.get("/admin/api/cookbook")

    payload = response.json()
    assert response.status_code == 200
    assert len(payload["models"]) >= 45
    assert "no Hub API calls" in payload["source"]
    assert all(model["repo"] and model["task"] and model["format"] for model in payload["models"])
    assert all("/" in model["repo"] for model in payload["models"])


@pytest.mark.asyncio
async def test_live_hugging_face_catalog_filters_caches_and_paginates(tmp_path) -> None:
    received: list[httpx.Request] = []

    async def hub_response(request: httpx.Request) -> httpx.Response:
        received.append(request)
        if request.url.params.get("cursor") == "page-2":
            return httpx.Response(
                200,
                json=[
                    {
                        "id": "community/llama-3b-GGUF",
                        "pipeline_tag": "text-generation",
                        "library_name": "gguf",
                        "tags": ["gguf", "3b"],
                        "downloads": 20,
                        "likes": 3,
                    }
                ],
            )
        return httpx.Response(
            200,
            json=[
                {
                    "id": "community/llama-8b-GGUF",
                    "pipeline_tag": "text-generation",
                    "library_name": "gguf",
                    "tags": ["gguf", "8b", "license:apache-2.0"],
                    "downloads": 1500,
                    "likes": 45,
                    "cardData": {"description": "A public local model."},
                }
            ],
            headers={"Link": '<https://huggingface.co/api/models?cursor=page-2>; rel="next"'},
        )

    app = create_app(
        Settings(),
        transport=httpx.MockTransport(hub_response),
        admin_config_path=tmp_path / "config.json",
    )
    query = (
        "/admin/api/cookbook?source=huggingface&q=llama&task=text-generation"
        "&format=gguf&sort=downloads"
    )
    async with open_client(app) as client:
        first = await client.get(query)
        cached = await client.get(query)
        next_page = await client.get(f"{query}&cursor=page-2")

    assert first.status_code == 200
    assert first.json()["live"] is True
    assert first.json()["models"][0]["repo"] == "community/llama-8b-GGUF"
    assert first.json()["models"][0]["scale"] == "8B"
    assert first.json()["models"][0]["license"] == "apache-2.0"
    assert first.json()["next_cursor"] == "page-2"
    assert cached.json() == first.json()
    assert next_page.status_code == 200
    assert next_page.json()["models"][0]["repo"] == "community/llama-3b-GGUF"
    assert len(received) == 2
    assert received[0].url.host == "huggingface.co"
    assert received[0].url.params["search"] == "llama"
    assert received[0].url.params["pipeline_tag"] == "text-generation"
    assert received[0].url.params["filter"] == "gguf"


@pytest.mark.asyncio
async def test_live_hugging_face_catalog_is_bounded_and_has_offline_fallback(tmp_path) -> None:
    async def hub_unavailable(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    app = create_app(
        Settings(),
        transport=httpx.MockTransport(hub_unavailable),
        admin_config_path=tmp_path / "config.json",
    )
    async with open_client(app) as client:
        invalid_task = await client.get("/admin/api/cookbook?source=huggingface&task=not-a-task")
        long_query = await client.get(f"/admin/api/cookbook?source=huggingface&q={'x' * 121}")
        unavailable = await client.get("/admin/api/cookbook?source=huggingface")
        curated = await client.get("/admin/api/cookbook?source=curated")

    assert invalid_task.status_code == 400
    assert long_query.status_code == 400
    assert unavailable.status_code == 503
    assert unavailable.json()["error"]["code"] == "cookbook_hub_unavailable"
    assert curated.status_code == 200
    assert curated.json()["live"] is False
    assert len(curated.json()["models"]) >= 45


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
    assert state.json()["api"] == {
        "format": "openai",
        "version": 1,
        "base_path": "/v1",
        "models_path": "/models",
        "models_endpoint": "/v1/models",
    }
    assert "dashboard-secret" not in state.text
    assert metrics.status_code == 200
    assert metrics.json()["in_flight"] == 0
    assert denied.status_code == 401


@pytest.mark.asyncio
async def test_dashboard_rejects_remote_model_base_urls(tmp_path) -> None:
    local_config = tmp_path / "cortex.local.json"
    app = create_app(configured_settings(), admin_config_path=local_config)
    payload = {
        "server": {"default_model": "remote"},
        "models": [
            {
                "id": "remote",
                "runtime": "command",
                "base_url": "https://example.invalid/v1",
                "default": True,
            }
        ],
    }

    async with open_client(app) as client:
        response = await client.put("/admin/api/config", headers=admin_headers(), json=payload)

    assert response.status_code == 422
    assert "must be local processes" in response.json()["error"]["message"]
    assert app.state.settings.default_model == "public-model"
    assert not local_config.exists()


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
                "optimization_preset": "medium_plus",
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
    assert local_model["optimization_level"] == 0
    assert local_model["optimization_preset"] == "medium_plus"
    assert local_model["gpu_layers"] == -1
    assert local_model["context_size"] == 1024
    assert local_model["embedding"] is True
    assert local_model["mmproj_path"] == "/models/projector.gguf"
    assert state.json()["runtime"]["models"]["tiny-local"]["status"] == "error"
    stored_config = json.loads(local_config.read_text(encoding="utf-8"))
    assert "dashboard-secret" not in json.dumps(stored_config)
    assert stored_config["models"][0]["optimization_level"] == 0
    assert stored_config["models"][0]["optimization_preset"] == "medium_plus"
    assert exported.status_code == 200
    assert exported.json()["models"][0]["runtime"] == "llama.cpp"
    assert exported.json()["models"][0]["optimization_level"] == 0
    assert exported.json()["models"][0]["optimization_preset"] == "medium_plus"
    assert exported.json()["models"][0]["embedding"] is True


@pytest.mark.asyncio
async def test_cortex_cpp_conversion_settings_round_trip_through_dashboard(tmp_path) -> None:
    local_config = tmp_path / "cortex.local.json"
    app = create_app(configured_settings(), admin_config_path=local_config)
    payload = {
        "server": {"default_model": "native-local", "workers": 1},
        "models": [
            {
                "id": "native-local",
                "runtime": "cortex-cpp",
                "model_path": "/models/hf-checkpoint",
                "conversion_outtype": "q8_0",
                "allow_unsafe_legacy_checkpoint_conversion": True,
                "default": True,
            }
        ],
    }

    async with open_client(app) as client:
        saved = await client.put("/admin/api/config", headers=admin_headers(), json=payload)
        state = await client.get("/admin/api/state", headers=admin_headers())
        exported = await client.get("/admin/api/config/export", headers=admin_headers())

    assert saved.status_code == 200
    assert state.json()["models"][0]["conversion_outtype"] == "q8_0"
    assert state.json()["models"][0]["allow_unsafe_legacy_checkpoint_conversion"] is True
    assert exported.json()["models"][0]["allow_unsafe_legacy_checkpoint_conversion"] is True
    stored_model = json.loads(local_config.read_text(encoding="utf-8"))["models"][0]
    assert stored_model["conversion_outtype"] == "q8_0"
    assert stored_model["allow_unsafe_legacy_checkpoint_conversion"] is True


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


@pytest.mark.asyncio
async def test_command_runtime_capabilities_and_argv_round_trip_through_dashboard(tmp_path) -> None:
    local_config = tmp_path / "cortex.local.json"
    app = create_app(configured_settings(), admin_config_path=local_config)
    command = [
        "python",
        "local_api.py",
        "--model",
        "{model_path}",
        "--host",
        "{host}",
        "--port",
        "{port}",
    ]
    payload = {
        "server": {"default_model": "public-model", "workers": 1},
        "models": [
            {
                "id": "public-model",
                "runtime": "llama.cpp",
                "model_path": "/models/not-present.gguf",
                "capabilities": ["text_generation"],
                "default": True,
            },
            {
                "id": "local-video",
                "runtime": "command",
                "model_path": "/models/video-folder",
                "api_base_path": "/compatible/v1",
                "health_path": "/ready",
                "capabilities": ["video_understanding", "video_generation", "ocr"],
                "runtime_command": command,
                "default": False,
            },
        ],
    }

    async with open_client(app) as client:
        saved = await client.put("/admin/api/config", headers=admin_headers(), json=payload)
        state = await client.get("/admin/api/state", headers=admin_headers())
        exported = await client.get("/admin/api/config/export", headers=admin_headers())

    assert saved.status_code == 200
    model = next(row for row in state.json()["models"] if row["id"] == "local-video")
    assert model["runtime"] == "command"
    assert model["capabilities"] == ["video_understanding", "video_generation", "ocr"]
    assert model["runtime_command"] == command
    assert model["api_base_path"] == "/compatible/v1"
    assert exported.json()["models"][1]["runtime_command"] == command
    assert json.loads(local_config.read_text(encoding="utf-8"))["models"][1]["capabilities"] == [
        "video_understanding",
        "video_generation",
        "ocr",
    ]


@pytest.mark.asyncio
async def test_model_file_picker_requires_admin_auth_and_lists_local_gguf_files(
    tmp_path, monkeypatch
) -> None:
    model_directory = tmp_path / "models"
    model_directory.mkdir()
    model_file = model_directory / "nomic-embed-text-v1.5.Q4_K_M.gguf"
    model_file.write_bytes(b"GGUF local model fixture")
    monkeypatch.setenv("CORTEX_MODELS_DIR", str(model_directory))
    app = create_app(configured_settings(), admin_config_path=tmp_path / "config.json")

    async with open_client(app) as client:
        denied = await client.get("/admin/api/models/files")
        response = await client.get("/admin/api/models/files", headers=admin_headers())

    assert denied.status_code == 401
    assert response.status_code == 200
    assert response.json()["files"] == [
        {
            "path": str(model_file),
            "name": model_file.name,
            "size_bytes": model_file.stat().st_size,
        }
    ]


@pytest.mark.asyncio
async def test_model_file_upload_requires_admin_auth_and_streams_to_local_storage(
    tmp_path, monkeypatch
) -> None:
    model_directory = tmp_path / "models"
    monkeypatch.setenv("CORTEX_MODELS_DIR", str(model_directory))
    app = create_app(configured_settings(), admin_config_path=tmp_path / "config.json")
    content = b"GGUF local model fixture"
    url = "/admin/api/models/files/upload"
    headers = {**admin_headers(), "Content-Type": "application/octet-stream"}

    async with open_client(app) as client:
        denied = await client.post(url, params={"filename": "local.gguf"}, content=content)
        uploaded = await client.post(
            url, params={"filename": "local.gguf"}, content=content, headers=headers
        )

    assert denied.status_code == 401
    assert uploaded.status_code == 200
    result = uploaded.json()
    saved_path = model_directory / "local.gguf"
    assert result["path"] == str(saved_path.resolve())
    assert result["name"] == "local.gguf"
    assert result["size_bytes"] == len(content)
    assert saved_path.read_bytes() == content


@pytest.mark.asyncio
@pytest.mark.parametrize("content_length", ("-1", "+1", "1.0", "1, 1"))
async def test_model_file_upload_rejects_invalid_content_length(tmp_path, content_length) -> None:
    app = create_app(configured_settings(), admin_config_path=tmp_path / "config.json")
    headers = {
        **admin_headers(),
        "Content-Type": "application/octet-stream",
        "Content-Length": content_length,
    }

    async with open_client(app) as client:
        response = await client.post(
            "/admin/api/models/files/upload",
            params={"filename": "local.gguf"},
            content=b"GGUF fixture",
            headers=headers,
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_content_length"


@pytest.mark.asyncio
async def test_model_file_upload_rejects_unsafe_names_and_non_gguf_data(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("CORTEX_MODELS_DIR", str(tmp_path / "models"))
    app = create_app(configured_settings(), admin_config_path=tmp_path / "config.json")
    headers = {**admin_headers(), "Content-Type": "application/octet-stream"}

    async with open_client(app) as client:
        unsafe_name = await client.post(
            "/admin/api/models/files/upload",
            params={"filename": "../escape.gguf"},
            content=b"GGUF fixture",
            headers=headers,
        )
        invalid_header = await client.post(
            "/admin/api/models/files/upload",
            params={"filename": "not-a-model.gguf"},
            content=b"not a gguf model",
            headers=headers,
        )

    assert unsafe_name.status_code == 400
    assert unsafe_name.json()["error"]["code"] == "invalid_model_filename"
    assert invalid_header.status_code == 422
    assert invalid_header.json()["error"]["code"] == "invalid_gguf_file"
    assert list((tmp_path / "models").glob("*.gguf")) == []
    assert list((tmp_path / "models").glob("*.partial")) == []


@pytest.mark.asyncio
async def test_model_file_upload_avoids_overwriting_same_named_model(tmp_path, monkeypatch) -> None:
    model_directory = tmp_path / "models"
    monkeypatch.setenv("CORTEX_MODELS_DIR", str(model_directory))
    app = create_app(configured_settings(), admin_config_path=tmp_path / "config.json")
    headers = {**admin_headers(), "Content-Type": "application/octet-stream"}

    async with open_client(app) as client:
        first = await client.post(
            "/admin/api/models/files/upload",
            params={"filename": "same.gguf"},
            content=b"GGUF first model",
            headers=headers,
        )
        second = await client.post(
            "/admin/api/models/files/upload",
            params={"filename": "same.gguf"},
            content=b"GGUF second model",
            headers=headers,
        )

    assert first.status_code == second.status_code == 200
    assert first.json()["name"] == "same.gguf"
    assert second.json()["name"] == "same (2).gguf"
    assert (model_directory / "same.gguf").read_bytes() == b"GGUF first model"
    assert (model_directory / "same (2).gguf").read_bytes() == b"GGUF second model"


@pytest.mark.asyncio
async def test_capability_detection_suggests_tasks_from_local_gguf_metadata(tmp_path) -> None:
    model_file = tmp_path / "llava-vision-instruct.Q4_K_M.gguf"
    model_file.write_bytes(b"GGUF general.architecture=llava vision instruct")
    app = create_app(configured_settings(), admin_config_path=tmp_path / "config.json")

    async with open_client(app) as client:
        denied = await client.post(
            "/admin/api/models/detect-capabilities", json={"model_path": str(model_file)}
        )
        response = await client.post(
            "/admin/api/models/detect-capabilities",
            headers=admin_headers(),
            json={"model_path": str(model_file), "model_id": "local-vision"},
        )

    assert denied.status_code == 401
    assert response.status_code == 200
    assert {"vision_understanding", "text_generation"}.issubset(response.json()["capabilities"])
    assert "best-effort" in response.json()["note"].lower()
