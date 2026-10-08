"""Lightweight web dashboard and authenticated configuration API."""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import orjson
from starlette.requests import Request
from starlette.responses import HTMLResponse, Response

from . import __version__
from .config import ConfigurationError, ModelConfig, Settings
from .responses import OrjsonResponse
from .responses import error_response as _error

_UI_PATH = Path(__file__).parent / "static" / "index.html"
_UI_HTML = _UI_PATH.read_text(encoding="utf-8") if _UI_PATH.is_file() else ""
_RESTART_FIELDS = (
    "host",
    "port",
    "workers",
    "log_level",
    "access_log",
    "max_connections",
    "max_keepalive_connections",
    "keepalive_expiry",
)
_SERVER_FIELDS = (
    "host",
    "port",
    "workers",
    "log_level",
    "access_log",
    "llama_server_path",
    "default_model",
    "max_json_body_bytes",
    "connect_timeout",
    "read_timeout",
    "write_timeout",
    "pool_timeout",
    "max_connections",
    "max_keepalive_connections",
    "keepalive_expiry",
    "max_inference_requests",
)


class InferenceMetrics:
    """Low-overhead per-worker inference API counters for the dashboard."""

    def __init__(self) -> None:
        self.started_at = time.monotonic()
        self.in_flight = 0
        self.total_requests = 0
        self.failed_requests = 0
        self.completed_requests = 0
        self.total_latency_ms = 0.0
        self.status_codes: Counter[str] = Counter()
        self.models: Counter[str] = Counter()

    def start(self, model_id: str) -> float:
        self.in_flight += 1
        self.total_requests += 1
        self.models[model_id] += 1
        return time.perf_counter()

    def finish(self, started_at: float, status_code: int) -> None:
        self.in_flight = max(0, self.in_flight - 1)
        self.completed_requests += 1
        self.total_latency_ms += (time.perf_counter() - started_at) * 1000
        self.status_codes[str(status_code)] += 1
        if status_code >= 500:
            self.failed_requests += 1

    def snapshot(self) -> dict[str, Any]:
        avg_latency = (
            self.total_latency_ms / self.completed_requests if self.completed_requests else 0.0
        )
        return {
            "uptime_seconds": round(time.monotonic() - self.started_at, 1),
            "in_flight": self.in_flight,
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
            "average_latency_ms": round(avg_latency, 2),
            "status_codes": dict(self.status_codes),
            "requests_by_model": dict(self.models),
        }

    def reset(self) -> None:
        self.total_requests = 0
        self.failed_requests = 0
        self.completed_requests = 0
        self.total_latency_ms = 0.0
        self.status_codes.clear()
        self.models.clear()


def _resize_inference_limiter(request: Request, settings: Settings) -> None:
    request.app.state.inference_limiter.resize(settings.max_inference_requests)


def _admin_auth(request: Request, settings: Settings) -> Response | None:
    if settings.api_key is None:
        return _error(
            "Admin APIs are disabled until CORTEX_API_KEY is configured.",
            status_code=503,
            error_type="authentication_error",
            code="admin_auth_not_configured",
        )
    authorization = request.headers.get("authorization", "")
    scheme, separator, supplied = authorization.partition(" ")
    if (
        not separator
        or scheme.lower() != "bearer"
        or not hmac.compare_digest(supplied.encode("utf-8"), settings.api_key.encode("utf-8"))
    ):
        return _error(
            "A valid bearer token is required.",
            status_code=401,
            error_type="authentication_error",
            code="invalid_api_key",
            headers={"www-authenticate": "Bearer"},
        )
    return None


def _public_settings(settings: Settings) -> dict[str, Any]:
    return {
        "server": {
            **{name: getattr(settings, name) for name in _SERVER_FIELDS},
            "api_auth_enabled": settings.api_key is not None,
            "api_key_env": settings.api_key_env,
        },
        "models": [
            {
                "id": model.id,
                "upstream_model": model.upstream_model,
                "default": model.id == settings.default_model,
                "runtime": model.runtime,
                "optimization_level": model.optimization_level,
                "model_path": model.model_path,
                "threads": model.threads,
                "threads_batch": model.threads_batch,
                "gpu_layers": model.gpu_layers,
                "context_size": model.context_size,
                "batch_size": model.batch_size,
                "ubatch_size": model.ubatch_size,
                "parallel": model.parallel,
                "flash_attn": model.flash_attn,
                "cache_type_k": model.cache_type_k,
                "cache_type_v": model.cache_type_v,
                "use_mmap": model.use_mmap,
                "mlock": model.mlock,
                "embedding": model.embedding,
                "mmproj_path": model.mmproj_path,
            }
            for model in settings.models.values()
        ],
    }


def _persistable_settings(settings: Settings) -> dict[str, Any]:
    """Serialize references to secrets, never resolved secret values."""

    server = {name: getattr(settings, name) for name in _SERVER_FIELDS}
    server["api_key_env"] = settings.api_key_env
    models = [
        {
            "id": model.id,
            "upstream_model": model.upstream_model,
            "runtime": model.runtime,
            "optimization_level": model.optimization_level,
            "model_path": model.model_path,
            "threads": model.threads,
            "threads_batch": model.threads_batch,
            "gpu_layers": model.gpu_layers,
            "context_size": model.context_size,
            "batch_size": model.batch_size,
            "ubatch_size": model.ubatch_size,
            "parallel": model.parallel,
            "flash_attn": model.flash_attn,
            "cache_type_k": model.cache_type_k,
            "cache_type_v": model.cache_type_v,
            "use_mmap": model.use_mmap,
            "mlock": model.mlock,
            "embedding": model.embedding,
            "mmproj_path": model.mmproj_path,
            "default": model.id == settings.default_model,
        }
        for model in settings.models.values()
    ]
    return {"server": server, "models": models}


def _env_overrides() -> list[str]:
    env_map = {
        "CORTEX_HOST": "host",
        "CORTEX_PORT": "port",
        "CORTEX_WORKERS": "workers",
        "CORTEX_LOG_LEVEL": "log_level",
        "CORTEX_ACCESS_LOG": "access_log",
        "CORTEX_LLAMA_SERVER": "llama_server_path",
        "CORTEX_DEFAULT_MODEL": "default_model",
        "CORTEX_MAX_JSON_BODY_BYTES": "max_json_body_bytes",
        "CORTEX_CONNECT_TIMEOUT": "connect_timeout",
        "CORTEX_READ_TIMEOUT": "read_timeout",
        "CORTEX_WRITE_TIMEOUT": "write_timeout",
        "CORTEX_POOL_TIMEOUT": "pool_timeout",
        "CORTEX_MAX_CONNECTIONS": "max_connections",
        "CORTEX_MAX_KEEPALIVE_CONNECTIONS": "max_keepalive_connections",
        "CORTEX_KEEPALIVE_EXPIRY": "keepalive_expiry",
        "CORTEX_MAX_INFERENCE_REQUESTS": "max_inference_requests",
    }
    return [field for variable, field in env_map.items() if variable in os.environ]


async def _read_json(request: Request, limit: int = 256 * 1024) -> tuple[Any, Response | None]:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > limit:
                return None, _error("Admin request body is too large.", status_code=413)
        except ValueError:
            return None, _error("Invalid Content-Length header.", status_code=400)

    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > limit:
            return None, _error("Admin request body is too large.", status_code=413)
        body.extend(chunk)
    try:
        return orjson.loads(body), None
    except orjson.JSONDecodeError:
        return None, _error("Request body must be valid JSON.", status_code=400)


def _candidate_settings(payload: Any, current: Settings) -> Settings:
    if not isinstance(payload, Mapping):
        raise ConfigurationError("Configuration payload must be an object.")
    raw_server = payload.get("server")
    raw_models = payload.get("models")
    if not isinstance(raw_server, Mapping) or not isinstance(raw_models, list):
        raise ConfigurationError("Configuration requires a server object and models array.")

    server: dict[str, Any] = {"api_key_env": current.api_key_env}
    for field in _SERVER_FIELDS:
        if field in raw_server:
            server[field] = raw_server[field]
    models: list[dict[str, Any]] = []
    for index, raw_model in enumerate(raw_models):
        if not isinstance(raw_model, Mapping):
            raise ConfigurationError(f"models[{index}] must be an object.")
        models.append(
            {
                "id": raw_model.get("id"),
                "upstream_model": raw_model.get("upstream_model") or raw_model.get("id"),
                "runtime": raw_model.get("runtime", "llama.cpp"),
                "optimization_level": raw_model.get(
                    "optimization_level", raw_model.get("optimization_profile", 0)
                ),
                "model_path": raw_model.get("model_path"),
                "threads": raw_model.get("threads", 0),
                "threads_batch": raw_model.get("threads_batch", 0),
                "gpu_layers": raw_model.get("gpu_layers", 0),
                "context_size": raw_model.get("context_size", 2048),
                "batch_size": raw_model.get("batch_size", 256),
                "ubatch_size": raw_model.get("ubatch_size", 64),
                "parallel": raw_model.get("parallel", 1),
                "flash_attn": raw_model.get("flash_attn", "auto"),
                "cache_type_k": raw_model.get("cache_type_k", "f16"),
                "cache_type_v": raw_model.get("cache_type_v", "f16"),
                "use_mmap": raw_model.get("use_mmap", True),
                "mlock": raw_model.get("mlock", False),
                "embedding": raw_model.get("embedding", False),
                "mmproj_path": raw_model.get("mmproj_path"),
                "default": raw_model.get("default", False),
            }
        )

    parse_env = dict(os.environ)
    # Admin edits are the persisted source for the model registry. Secrets remain
    # environment-only and are resolved by Settings.from_mapping.
    parse_env.pop("CORTEX_MODELS", None)
    candidate = Settings.from_mapping({"server": server, "models": models}, environ=parse_env)
    # Programmatic create_app(Settings(...)) callers may supply a key directly;
    # keep it in process memory without ever serializing it to the overlay.
    if candidate.api_key is None and current.api_key is not None:
        candidate = replace(candidate, api_key=current.api_key)
    return candidate


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    encoded = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


async def dashboard(_: Request) -> Response:
    if not _UI_HTML:
        return Response("Dashboard assets are missing.", status_code=503)
    return HTMLResponse(
        _UI_HTML,
        headers={
            "cache-control": "no-cache",
            "x-frame-options": "DENY",
            "x-content-type-options": "nosniff",
            "referrer-policy": "no-referrer",
            "content-security-policy": (
                "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
                "img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
            ),
        },
    )


async def admin_state(request: Request) -> Response:
    settings: Settings = request.app.state.settings
    auth = _admin_auth(request, settings)
    if auth is not None:
        return auth
    return OrjsonResponse(
        {
            **_public_settings(settings),
            "metrics": request.app.state.metrics.snapshot(),
            "runtime": {
                "hardware": request.app.state.runtime_manager.hardware,
                "models": request.app.state.runtime_manager.states(),
            },
            "version": __version__,
            "env_overrides": _env_overrides(),
            "admin_config_path": request.app.state.admin_config_path.name,
        }
    )


async def admin_metrics(request: Request) -> Response:
    auth = _admin_auth(request, request.app.state.settings)
    if auth is not None:
        return auth
    return OrjsonResponse(request.app.state.metrics.snapshot())


async def reset_metrics(request: Request) -> Response:
    auth = _admin_auth(request, request.app.state.settings)
    if auth is not None:
        return auth
    request.app.state.metrics.reset()
    return OrjsonResponse({"ok": True})


async def update_config(request: Request) -> Response:
    settings: Settings = request.app.state.settings
    auth = _admin_auth(request, settings)
    if auth is not None:
        return auth
    payload, body_error = await _read_json(request)
    if body_error is not None:
        return body_error

    async with request.app.state.admin_lock:
        settings = request.app.state.settings
        try:
            candidate = _candidate_settings(payload, settings)
        except ConfigurationError as exc:
            return _error(str(exc), status_code=422, code="invalid_configuration")

        previous = settings
        restart_required = [
            field
            for field in _RESTART_FIELDS
            if getattr(previous, field) != getattr(candidate, field)
        ]
        path: Path = request.app.state.admin_config_path
        try:
            await asyncio.to_thread(_write_json, path, _persistable_settings(candidate))
        except OSError:
            return _error(
                "Could not persist configuration. Check the config directory permissions.",
                status_code=500,
                error_type="server_error",
                code="config_write_failed",
            )
        await request.app.state.runtime_manager.apply_settings(candidate)
        _resize_inference_limiter(request, candidate)
        request.app.state.settings = candidate

    return OrjsonResponse(
        {
            "ok": True,
            "restart_required": restart_required,
            **_public_settings(candidate),
        }
    )


async def reload_config(request: Request) -> Response:
    auth = _admin_auth(request, request.app.state.settings)
    if auth is not None:
        return auth
    async with request.app.state.admin_lock:
        env = dict(os.environ)
        env["CORTEX_ADMIN_CONFIG"] = str(request.app.state.admin_config_path)
        try:
            candidate = Settings.load(environ=env)
        except ConfigurationError as exc:
            return _error(str(exc), status_code=422, code="invalid_configuration")
        await request.app.state.runtime_manager.apply_settings(candidate)
        _resize_inference_limiter(request, candidate)
        request.app.state.settings = candidate
    return OrjsonResponse({"ok": True, **_public_settings(candidate)})


async def reset_config(request: Request) -> Response:
    previous: Settings = request.app.state.settings
    auth = _admin_auth(request, previous)
    if auth is not None:
        return auth
    async with request.app.state.admin_lock:
        path: Path = request.app.state.admin_config_path
        try:
            await asyncio.to_thread(path.unlink, missing_ok=True)
            env = dict(os.environ)
            env["CORTEX_ADMIN_CONFIG"] = str(path)
            candidate = Settings.load(environ=env)
        except (OSError, ConfigurationError) as exc:
            return _error(str(exc), status_code=500, code="config_reset_failed")
        restart_required = [
            field
            for field in _RESTART_FIELDS
            if getattr(previous, field) != getattr(candidate, field)
        ]
        await request.app.state.runtime_manager.apply_settings(candidate)
        _resize_inference_limiter(request, candidate)
        request.app.state.settings = candidate
    return OrjsonResponse(
        {"ok": True, "restart_required": restart_required, **_public_settings(candidate)}
    )


async def download_config(request: Request) -> Response:
    auth = _admin_auth(request, request.app.state.settings)
    if auth is not None:
        return auth
    return OrjsonResponse(
        _persistable_settings(request.app.state.settings),
        headers={"content-disposition": 'attachment; filename="cortex-config.json"'},
    )


async def runtime_status(request: Request) -> Response:
    auth = _admin_auth(request, request.app.state.settings)
    if auth is not None:
        return auth
    manager = request.app.state.runtime_manager
    return OrjsonResponse({"hardware": manager.hardware, "models": manager.states()})


async def runtime_action(request: Request, action: str) -> Response:
    auth = _admin_auth(request, request.app.state.settings)
    if auth is not None:
        return auth
    model_id = request.path_params.get("model_id")
    if model_id is None:
        payload, body_error = await _read_json(request, limit=4096)
        if body_error is not None:
            return body_error
        if not isinstance(payload, Mapping) or not isinstance(payload.get("id"), str):
            return _error("Runtime action requires a model id.", status_code=400)
        model_id = payload["id"]

    async with request.app.state.admin_lock:
        settings: Settings = request.app.state.settings
        model = settings.models.get(model_id)
        if model is None:
            return _error("Unknown model.", status_code=404, code="model_not_found")
        if model.runtime != "llama.cpp":
            return _error("Only local llama.cpp models have process controls.", status_code=422)

        manager = request.app.state.runtime_manager
        try:
            if action == "start":
                state = await manager.start(model_id)
            elif action == "stop":
                state = await manager.stop(model_id)
            else:
                state = await manager.restart(model_id)
        except (KeyError, ValueError) as exc:
            return _error(str(exc), status_code=422, code="runtime_action_failed")
        return OrjsonResponse({"ok": state.get("status") != "error", "runtime": state})


async def start_model(request: Request) -> Response:
    return await runtime_action(request, "start")


async def stop_model(request: Request) -> Response:
    return await runtime_action(request, "stop")


async def restart_model(request: Request) -> Response:
    return await runtime_action(request, "restart")


async def test_model(request: Request) -> Response:
    auth = _admin_auth(request, request.app.state.settings)
    if auth is not None:
        return auth
    payload, body_error = await _read_json(request, limit=64 * 1024)
    if body_error is not None:
        return body_error
    if not isinstance(payload, Mapping):
        return _error("Model test payload must be an object.", status_code=400)

    model_data = dict(payload)
    model_data.setdefault("id", "test-model")
    model_data.setdefault("upstream_model", model_data["id"])
    try:
        test_env = dict(os.environ)
        test_env.pop("CORTEX_MODELS", None)
        test_env.pop("CORTEX_DEFAULT_MODEL", None)
        parsed = Settings.from_mapping(
            {
                "server": {"default_model": model_data["id"]},
                "models": [{**model_data, "default": True}],
            },
            environ=test_env,
        )
        model: ModelConfig = parsed.models[model_data["id"]]
    except (ConfigurationError, KeyError) as exc:
        return _error(str(exc), status_code=422, code="invalid_model")

    started = time.perf_counter()
    result = await asyncio.to_thread(request.app.state.runtime_manager.test, model)
    result["check_duration_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return OrjsonResponse(result)
