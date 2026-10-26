"""Lightweight web dashboard and authenticated configuration API."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import math
import os
import re
import tempfile
import time
import uuid
from collections import Counter, OrderedDict, deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import ceil
from pathlib import Path
from typing import Any

import httpx
import orjson
from starlette.requests import ClientDisconnect, Request
from starlette.responses import HTMLResponse, Response

from . import __version__
from .auth import (
    SESSION_COOKIE_NAME,
    SESSION_TTL_SECONDS,
    create_session_token,
    is_authenticated,
    session_signing_key,
    verify_admin_credentials,
)
from .config import LOCAL_RUNTIMES, ConfigurationError, ModelConfig, Settings
from .cookbook import HF_TASK_LABELS, cookbook_models, search_hugging_face
from .http_utils import content_length_from_headers
from .responses import OrjsonResponse
from .responses import error_response as _error
from .updater import UpdateError, check_for_updates, install_from_commit

_UI_PATH = Path(__file__).parent / "static" / "index.html"
_UI_HTML = _UI_PATH.read_text(encoding="utf-8") if _UI_PATH.is_file() else ""


def _inline_script_hashes(document: str) -> tuple[str, ...]:
    """Compute CSP source hashes for the dashboard's static inline scripts."""

    hashes: list[str] = []
    pattern = re.compile(r"<script\b([^>]*)>(.*?)</script\s*>", re.IGNORECASE | re.DOTALL)
    for attributes, source in pattern.findall(document):
        if re.search(r"(?:^|\s)src(?:\s|=|$)", attributes, re.IGNORECASE):
            continue
        digest = base64.b64encode(hashlib.sha256(source.encode("utf-8")).digest()).decode("ascii")
        source_hash = f"'sha256-{digest}'"
        if source_hash not in hashes:
            hashes.append(source_hash)
    return tuple(hashes)


_SCRIPT_CSP_SOURCES = " ".join(("'self'", *_inline_script_hashes(_UI_HTML)))
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
    "api_auth_enabled",
    "default_model",
    "max_json_body_bytes",
    "max_media_body_bytes",
    "connect_timeout",
    "read_timeout",
    "write_timeout",
    "pool_timeout",
    "max_connections",
    "max_keepalive_connections",
    "keepalive_expiry",
    "max_inference_requests",
)


class LoginAttemptLimiter:
    """Bound repeated failed sign-ins per client without retaining credentials."""

    def __init__(
        self, *, max_attempts: int = 10, window_seconds: float = 60.0, max_clients: int = 4096
    ) -> None:
        if (
            max_attempts < 1
            or max_clients < 1
            or not math.isfinite(window_seconds)
            or window_seconds <= 0
        ):
            raise ValueError("Login rate-limit values must be positive and finite.")
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self.max_clients = max_clients
        self._failures: OrderedDict[str, deque[float]] = OrderedDict()

    def _active_failures(self, client_key: str, now: float) -> deque[float] | None:
        attempts = self._failures.get(client_key)
        if attempts is None:
            return None
        cutoff = now - self.window_seconds
        while attempts and attempts[0] <= cutoff:
            attempts.popleft()
        if not attempts:
            self._failures.pop(client_key, None)
            return None
        self._failures.move_to_end(client_key)
        return attempts

    def retry_after(self, client_key: str, *, now: float | None = None) -> int | None:
        current = time.monotonic() if now is None else now
        attempts = self._active_failures(client_key, current)
        if attempts is None or len(attempts) < self.max_attempts:
            return None
        return max(1, ceil(self.window_seconds - (current - attempts[0])))

    def record_failure(self, client_key: str, *, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        attempts = self._active_failures(client_key, current)
        if attempts is None:
            attempts = deque()
            self._failures[client_key] = attempts
        attempts.append(current)
        self._failures.move_to_end(client_key)
        while len(self._failures) > self.max_clients:
            self._failures.popitem(last=False)

    def clear(self, client_key: str) -> None:
        self._failures.pop(client_key, None)


@dataclass(slots=True)
class _InferenceSample:
    model_id: str
    started_at: float
    generation: int
    first_token_at: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(slots=True)
class _GenerationStats:
    usage_requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    first_token_samples: int = 0
    total_first_token_latency_ms: float = 0.0
    throughput_samples: int = 0
    throughput_completion_tokens: int = 0
    generation_seconds: float = 0.0

    def record(self, sample: _InferenceSample, finished_at: float) -> None:
        if sample.prompt_tokens is not None or sample.completion_tokens is not None:
            self.usage_requests += 1
            self.prompt_tokens += sample.prompt_tokens or 0
            self.completion_tokens += sample.completion_tokens or 0

        if sample.first_token_at is None:
            return
        self.first_token_samples += 1
        self.total_first_token_latency_ms += max(
            0.0, sample.first_token_at - sample.started_at
        ) * 1000
        completion_tokens = sample.completion_tokens or 0
        generation_seconds = max(0.0, finished_at - sample.first_token_at)
        if completion_tokens and generation_seconds > 0:
            self.throughput_samples += 1
            self.throughput_completion_tokens += completion_tokens
            self.generation_seconds += generation_seconds

    def snapshot(self) -> dict[str, int | float]:
        average_first_token_ms = (
            self.total_first_token_latency_ms / self.first_token_samples
            if self.first_token_samples
            else 0.0
        )
        completion_tokens_per_second = (
            self.throughput_completion_tokens / self.generation_seconds
            if self.generation_seconds
            else 0.0
        )
        return {
            "usage_requests": self.usage_requests,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "first_token_samples": self.first_token_samples,
            "average_first_token_latency_ms": round(average_first_token_ms, 2),
            "throughput_samples": self.throughput_samples,
            "completion_tokens_per_second": round(completion_tokens_per_second, 2),
        }


class InferenceMetrics:
    """Low-overhead per-worker inference and model-provided token counters."""

    def __init__(self) -> None:
        self.started_at = time.monotonic()
        self.in_flight = 0
        self.total_requests = 0
        self.failed_requests = 0
        self.completed_requests = 0
        self.total_latency_ms = 0.0
        self.status_codes: Counter[str] = Counter()
        self.models: Counter[str] = Counter()
        self._generation = 0
        self._generation_stats = _GenerationStats()
        self._generation_by_model: dict[str, _GenerationStats] = {}

    def start(self, model_id: str) -> _InferenceSample:
        self.in_flight += 1
        self.total_requests += 1
        self.models[model_id] += 1
        return _InferenceSample(model_id, time.perf_counter(), self._generation)

    def observe_first_token(self, sample: _InferenceSample, *, at: float | None = None) -> None:
        if sample.first_token_at is None:
            sample.first_token_at = time.perf_counter() if at is None else at

    def observe_usage(
        self,
        sample: _InferenceSample,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool) and prompt_tokens >= 0:
            sample.prompt_tokens = prompt_tokens
        if (
            isinstance(completion_tokens, int)
            and not isinstance(completion_tokens, bool)
            and completion_tokens >= 0
        ):
            sample.completion_tokens = completion_tokens

    def finish(
        self, sample: _InferenceSample, status_code: int, *, at: float | None = None
    ) -> None:
        finished_at = time.perf_counter() if at is None else at
        self.in_flight = max(0, self.in_flight - 1)
        # A reset starts a new reporting window. In-flight requests from the old
        # window still leave the active count, but must not repopulate cleared totals.
        if sample.generation != self._generation:
            return
        self.completed_requests += 1
        self.total_latency_ms += max(0.0, finished_at - sample.started_at) * 1000
        self.status_codes[str(status_code)] += 1
        if status_code >= 400:
            self.failed_requests += 1
            return
        self._generation_stats.record(sample, finished_at)
        if (
            sample.prompt_tokens is not None
            or sample.completion_tokens is not None
            or sample.first_token_at is not None
        ):
            self._generation_by_model.setdefault(sample.model_id, _GenerationStats()).record(
                sample, finished_at
            )

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
            "generation_metrics": self._generation_stats.snapshot(),
            "generation_metrics_by_model": {
                model_id: stats.snapshot()
                for model_id, stats in sorted(self._generation_by_model.items())
            },
        }

    def reset(self) -> None:
        self._generation += 1
        self.total_requests = 0
        self.failed_requests = 0
        self.completed_requests = 0
        self.total_latency_ms = 0.0
        self.status_codes.clear()
        self.models.clear()
        self._generation_stats = _GenerationStats()
        self._generation_by_model.clear()


class InferenceResponseTelemetry:
    """Extract bounded, prompt-free usage metadata from JSON and SSE responses."""

    _MAX_JSON_BYTES = 64 * 1024
    _MAX_SSE_LINE_BYTES = 64 * 1024
    _MAX_SSE_EVENT_BYTES = 256 * 1024

    def __init__(self, metrics: InferenceMetrics, sample: _InferenceSample, content_type: str) -> None:
        self._metrics = metrics
        self._sample = sample
        self._streaming = "text/event-stream" in content_type.lower()
        self._line_buffer = bytearray()
        self._event_data = bytearray()
        self._event_oversized = False
        self._discarding_line = False
        self._json_body = (
            bytearray()
            if not self._streaming and "json" in content_type.lower()
            else None
        )
        self._finished = False

    def feed(self, chunk: bytes) -> None:
        if self._finished or not chunk:
            return
        if not self._streaming:
            if self._json_body is not None:
                if len(self._json_body) + len(chunk) > self._MAX_JSON_BYTES:
                    self._json_body = None
                else:
                    self._json_body.extend(chunk)
            return

        if self._discarding_line:
            newline = chunk.find(b"\n")
            if newline < 0:
                return
            chunk = chunk[newline + 1 :]
            self._discarding_line = False
        if chunk:
            self._line_buffer.extend(chunk)
        cursor = 0
        while True:
            newline = self._line_buffer.find(b"\n", cursor)
            if newline < 0:
                break
            if newline - cursor > self._MAX_SSE_LINE_BYTES:
                self._mark_event_oversized()
            else:
                line = bytes(self._line_buffer[cursor:newline])
                if line.endswith(b"\r"):
                    line = line[:-1]
                self._consume_sse_line(line)
            cursor = newline + 1
        if cursor:
            del self._line_buffer[:cursor]
        if len(self._line_buffer) > self._MAX_SSE_LINE_BYTES:
            self._line_buffer.clear()
            self._discarding_line = True
            self._mark_event_oversized()

    def finish(self, status_code: int) -> None:
        if self._finished:
            return
        self._finished = True
        if status_code >= 400:
            return
        if self._streaming:
            if self._line_buffer and not self._discarding_line:
                line = bytes(self._line_buffer)
                if line.endswith(b"\r"):
                    line = line[:-1]
                if len(line) <= self._MAX_SSE_LINE_BYTES:
                    self._consume_sse_line(line)
                else:
                    self._mark_event_oversized()
            self._line_buffer.clear()
            if self._event_data or self._event_oversized:
                self._finish_sse_event()
            return
        if self._json_body is None:
            return
        try:
            payload = orjson.loads(bytes(self._json_body))
        except orjson.JSONDecodeError:
            return
        self._observe_payload(payload, observe_timing=False)

    def _consume_sse_line(self, line: bytes) -> None:
        if not line:
            self._finish_sse_event()
            return
        if not line.startswith(b"data:") or self._event_oversized:
            return
        data = line[5:]
        if data.startswith(b" "):
            data = data[1:]
        if len(self._event_data) + len(data) + 1 > self._MAX_SSE_EVENT_BYTES:
            self._mark_event_oversized()
            return
        self._event_data.extend(data)
        self._event_data.extend(b"\n")

    def _finish_sse_event(self) -> None:
        if not self._event_oversized and self._event_data:
            payload = bytes(self._event_data).rstrip(b"\n")
            if payload and payload != b"[DONE]":
                try:
                    packet = orjson.loads(payload)
                except orjson.JSONDecodeError:
                    packet = None
                self._observe_payload(packet, observe_timing=True)
        self._event_data.clear()
        self._event_oversized = False

    def _mark_event_oversized(self) -> None:
        self._event_oversized = True
        self._event_data.clear()

    def _observe_payload(self, payload: object, *, observe_timing: bool) -> None:
        if not isinstance(payload, Mapping):
            return
        usage = payload.get("usage")
        if not isinstance(usage, Mapping):
            response = payload.get("response")
            usage = response.get("usage") if isinstance(response, Mapping) else None
        if isinstance(usage, Mapping):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            if prompt_tokens is None:
                prompt_tokens = usage.get("input_tokens")
            if completion_tokens is None:
                completion_tokens = usage.get("output_tokens")
            self._metrics.observe_usage(
                self._sample,
                prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
                completion_tokens=(
                    completion_tokens if isinstance(completion_tokens, int) else None
                ),
            )
        if not observe_timing:
            return
        choices = payload.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, Mapping):
                    continue
                delta = choice.get("delta")
                if isinstance(delta, Mapping) and self._has_text(delta.get("content")):
                    self._metrics.observe_first_token(self._sample)
                    return
                if self._has_text(choice.get("text")):
                    self._metrics.observe_first_token(self._sample)
                    return
        event_type = payload.get("type")
        if event_type in {"response.output_text.delta", "response.refusal.delta"} and self._has_text(
            payload.get("delta")
        ):
            self._metrics.observe_first_token(self._sample)

    @staticmethod
    def _has_text(value: object) -> bool:
        if isinstance(value, str):
            return bool(value)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item:
                    return True
                if isinstance(item, Mapping) and any(
                    isinstance(item.get(key), str) and item.get(key)
                    for key in ("text", "content")
                ):
                    return True
        return False


def _resize_inference_limiter(request: Request, settings: Settings) -> None:
    request.app.state.inference_limiter.resize(settings.max_inference_requests)


def _admin_auth(request: Request, settings: Settings) -> Response | None:
    if not settings.api_auth_enabled:
        return None
    if settings.api_key is None:
        return _error(
            "API authentication is enabled but CORTEX_API_KEY is not configured.",
            status_code=503,
            error_type="authentication_error",
            code="admin_auth_not_configured",
        )
    session_secret = session_signing_key(
        settings.api_key,
        settings.admin_username,
        settings.admin_password,
    )
    if is_authenticated(
        settings.api_key,
        request.headers.get("authorization", ""),
        request.cookies.get(SESSION_COOKIE_NAME),
        session_secret=session_secret,
    ):
        return None
    return _error(
        "A valid dashboard account session or bearer API key is required.",
        status_code=401,
        error_type="authentication_error",
        code="invalid_api_key",
        headers={"www-authenticate": "Bearer"},
    )


def _public_settings(settings: Settings) -> dict[str, Any]:
    return {
        "server": {
            **{name: getattr(settings, name) for name in _SERVER_FIELDS},
            "api_auth_enabled": bool(settings.api_auth_enabled),
            "api_key_env": settings.api_key_env,
        },
        "models": [
            {
                "id": model.id,
                "upstream_model": model.upstream_model,
                "default": model.id == settings.default_model,
                "runtime": model.runtime,
                "capabilities": list(model.effective_capabilities),
                "runtime_command": list(model.runtime_command),
                "api_base_path": model.api_base_path,
                "health_path": model.health_path,
                "optimization_level": model.optimization_level,
                "optimization_preset": model.optimization_preset,
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
                "conversion_outtype": model.conversion_outtype,
                "allow_requantize": model.allow_requantize,
                "allow_unsafe_legacy_checkpoint_conversion": model.allow_unsafe_legacy_checkpoint_conversion,
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
            "capabilities": list(model.effective_capabilities),
            "runtime_command": list(model.runtime_command),
            "api_base_path": model.api_base_path,
            "health_path": model.health_path,
            "optimization_level": model.optimization_level,
            "optimization_preset": model.optimization_preset,
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
            "conversion_outtype": model.conversion_outtype,
            "allow_requantize": model.allow_requantize,
            "allow_unsafe_legacy_checkpoint_conversion": model.allow_unsafe_legacy_checkpoint_conversion,
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
        "CORTEX_API_AUTH_ENABLED": "api_auth_enabled",
        "CORTEX_DEFAULT_MODEL": "default_model",
        "CORTEX_MAX_JSON_BODY_BYTES": "max_json_body_bytes",
        "CORTEX_MAX_MEDIA_BODY_BYTES": "max_media_body_bytes",
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
    try:
        declared_length = content_length_from_headers(request.headers)
    except ValueError:
        return None, _error("Invalid Content-Length header.", status_code=400)
    if declared_length is not None and declared_length > limit:
        return None, _error("Admin request body is too large.", status_code=413)

    body = bytearray()
    try:
        async for chunk in request.stream():
            if len(body) + len(chunk) > limit:
                return None, _error("Admin request body is too large.", status_code=413)
            body.extend(chunk)
    except ClientDisconnect:
        return None, _error(
            "The request body was interrupted.",
            status_code=400,
            code="request_body_interrupted",
        )
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
        if "base_url" in raw_model:
            raise ConfigurationError(
                f"models[{index}].base_url is not supported; runtimes must be local processes."
            )
        models.append(
            {
                "id": raw_model.get("id"),
                "upstream_model": raw_model.get("upstream_model") or raw_model.get("id"),
                "runtime": raw_model.get("runtime", "llama.cpp"),
                "capabilities": raw_model.get("capabilities", ["text_generation"]),
                "runtime_command": raw_model.get("runtime_command", []),
                "api_base_path": raw_model.get("api_base_path", "/v1"),
                "health_path": raw_model.get("health_path", "/health"),
                "optimization_level": raw_model.get(
                    "optimization_level", raw_model.get("optimization_profile", 0)
                ),
                "optimization_preset": raw_model.get("optimization_preset"),
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
                "conversion_outtype": raw_model.get("conversion_outtype", "f16"),
                "allow_requantize": raw_model.get("allow_requantize", False),
                "allow_unsafe_legacy_checkpoint_conversion": raw_model.get(
                    "allow_unsafe_legacy_checkpoint_conversion", False
                ),
                "mmproj_path": raw_model.get("mmproj_path"),
                "default": raw_model.get("default", False),
            }
        )

    parse_env = dict(os.environ)
    # Preserve programmatic keys only in this process-local candidate; never write
    # them to the saved configuration overlay.
    if "CORTEX_API_KEY" not in parse_env and current.api_key is not None:
        parse_env["CORTEX_API_KEY"] = current.api_key
    if "CORTEX_ADMIN_USERNAME" not in parse_env and current.admin_username is not None:
        parse_env["CORTEX_ADMIN_USERNAME"] = current.admin_username
        parse_env["CORTEX_ADMIN_PASSWORD"] = current.admin_password or ""
    # Admin edits are the persisted source for the model registry. Secrets remain
    # environment-only and are resolved by Settings.from_mapping.
    parse_env.pop("CORTEX_MODELS", None)
    candidate = Settings.from_mapping({"server": server, "models": models}, environ=parse_env)
    # Programmatic create_app(Settings(...)) callers may supply a key directly;
    # keep it in process memory without ever serializing it to the overlay.
    if candidate.api_key is None and current.api_key is not None:
        candidate = replace(candidate, api_key=current.api_key)
    if candidate.admin_username is None and current.admin_username is not None:
        candidate = replace(
            candidate,
            admin_username=current.admin_username,
            admin_password=current.admin_password,
        )
    return candidate


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically write a private JSON overlay without following temp-file symlinks."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    if os.name == "posix":
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Some filesystems do not allow syncing directories. The data file
            # itself has already been flushed and atomically replaced.
            pass


def _request_is_https(request: Request) -> bool:
    forwarded_proto = request.headers.get("x-forwarded-proto", "").split(",", 1)[0]
    return request.url.scheme == "https" or forwarded_proto.strip().lower() == "https"


def _cookie_samesite(secure: bool) -> str:
    # A secure, partitioned None cookie also works when a live preview embeds
    # Cortex under a different top-level site. Plain HTTP stays on Lax.
    return "none" if secure else "lax"


def _mark_cookie_partitioned(response: Response, secure: bool) -> None:
    if secure:
        response.headers["set-cookie"] = f"{response.headers['set-cookie']}; Partitioned"


async def auth_options(request: Request) -> Response:
    """Return non-secret login capabilities so the dashboard selects a valid sign-in mode."""

    settings: Settings = request.app.state.settings
    response = OrjsonResponse(
        {
            "api_auth_enabled": bool(settings.api_auth_enabled),
            "account_login_enabled": (
                settings.api_auth_enabled
                and settings.api_key is not None
                and settings.admin_username is not None
                and settings.admin_password is not None
            ),
            "api_key_login_enabled": bool(
                settings.api_auth_enabled and settings.api_key is not None
            ),
        }
    )
    response.headers["cache-control"] = "no-store"
    return response


async def auth_login(request: Request) -> Response:
    settings: Settings = request.app.state.settings
    if not settings.api_auth_enabled:
        response = OrjsonResponse({"ok": True, "api_auth_enabled": False})
        secure = _request_is_https(request)
        response.delete_cookie(
            SESSION_COOKIE_NAME,
            path="/",
            secure=secure,
            httponly=True,
            samesite=_cookie_samesite(secure),
        )
        _mark_cookie_partitioned(response, secure)
        response.headers["cache-control"] = "no-store"
        return response
    if settings.api_key is None:
        return _error(
            "API authentication is enabled but CORTEX_API_KEY is not configured.",
            status_code=503,
            error_type="authentication_error",
            code="admin_auth_not_configured",
        )
    client_key = request.client.host if request.client is not None else "unknown"
    limiter: LoginAttemptLimiter | None = getattr(request.app.state, "login_attempt_limiter", None)
    retry_after = limiter.retry_after(client_key) if limiter is not None else None
    if retry_after is not None:
        return _error(
            "Too many failed sign-in attempts. Wait before trying again.",
            status_code=429,
            error_type="authentication_error",
            code="login_rate_limited",
            headers={"retry-after": str(retry_after)},
        )
    if limiter is not None:
        # Reserve the attempt before awaiting the request body so concurrent
        # sign-ins cannot all pass the limit check at once.
        limiter.record_failure(client_key)
    payload, body_error = await _read_json(request, limit=4096)
    if body_error is not None:
        return body_error
    credentials = payload if isinstance(payload, Mapping) else {}
    supplied_key = credentials.get("api_key")
    api_key_valid = (
        settings.api_key is not None
        and isinstance(supplied_key, str)
        and hmac.compare_digest(supplied_key.encode("utf-8"), settings.api_key.encode("utf-8"))
    )
    account_valid = verify_admin_credentials(
        credentials.get("username"),
        credentials.get("password"),
        settings.admin_username,
        settings.admin_password,
    )
    if not (api_key_valid or account_valid):
        return _error(
            "The account credentials or API key are invalid.",
            status_code=401,
            error_type="authentication_error",
            code="invalid_api_key",
            headers={"www-authenticate": "Bearer"},
        )
    if limiter is not None:
        limiter.clear(client_key)

    signing_secret = session_signing_key(
        settings.api_key,
        settings.admin_username,
        settings.admin_password,
    )
    if signing_secret is None:
        return _error(
            "Account sessions are unavailable because no session signing secret is configured.",
            status_code=503,
            error_type="authentication_error",
            code="admin_auth_not_configured",
        )

    response = OrjsonResponse({"ok": True, "api_auth_enabled": True})
    secure = _request_is_https(request)
    response.set_cookie(
        SESSION_COOKIE_NAME,
        create_session_token(signing_secret),
        max_age=SESSION_TTL_SECONDS,
        path="/",
        secure=secure,
        httponly=True,
        samesite=_cookie_samesite(secure),
    )
    _mark_cookie_partitioned(response, secure)
    response.headers["cache-control"] = "no-store"
    return response


async def auth_logout(request: Request) -> Response:
    response = OrjsonResponse({"ok": True})
    secure = _request_is_https(request)
    response.delete_cookie(
        SESSION_COOKIE_NAME,
        path="/",
        secure=secure,
        httponly=True,
        samesite=_cookie_samesite(secure),
    )
    _mark_cookie_partitioned(response, secure)
    response.headers["cache-control"] = "no-store"
    return response


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
                f"default-src 'self'; script-src {_SCRIPT_CSP_SOURCES}; "
                "script-src-attr 'none'; style-src 'self' 'unsafe-inline'; connect-src 'self'; "
                "img-src 'self' data:; object-src 'none'; base-uri 'none'; "
                "form-action 'self'; frame-ancestors 'none'"
            ),
        },
    )


async def model_cookbook(request: Request) -> Response:
    auth = _admin_auth(request, request.app.state.settings)
    if auth is not None:
        return auth

    params = request.query_params
    source = params.get("source", "curated")
    if source == "curated":
        return OrjsonResponse(
            {
                "models": cookbook_models(),
                "source": "Curated offline list; no Hub API calls are made in this view.",
                "live": False,
                "next_cursor": None,
            }
        )
    if source != "huggingface":
        return _error("source must be 'curated' or 'huggingface'.", status_code=400)

    query = params.get("q", "").strip()
    task = params.get("task", "").strip()
    file_format = params.get("format", "gguf").strip().lower()
    sort = params.get("sort", "downloads").strip().lower()
    cursor = params.get("cursor", "")
    if len(query) > 120 or any(ord(character) < 32 for character in query):
        return _error("Search text must be at most 120 printable characters.", status_code=400)
    if task and task not in HF_TASK_LABELS:
        return _error("Choose a supported Hugging Face task filter.", status_code=400)
    if file_format not in {"all", "gguf"}:
        return _error("format must be 'all' or 'gguf'.", status_code=400)
    if sort not in {"updated", "downloads", "likes"}:
        return _error("sort must be 'downloads', 'likes', or 'updated'.", status_code=400)
    if len(cursor) > 1024 or any(ord(character) < 32 for character in cursor):
        return _error("Invalid Hugging Face pagination cursor.", status_code=400)

    cache_key = (query, task, file_format, sort, cursor)
    now = time.monotonic()
    cache = getattr(request.app.state, "hf_cookbook_cache", {})
    request.app.state.hf_cookbook_cache = cache
    cached = cache.get(cache_key)
    if cached and cached[0] > now:
        return OrjsonResponse(cached[1])

    try:
        models, next_cursor = await search_hugging_face(
            request.app.state.http_client,
            query=query,
            task=task,
            file_format=file_format,
            sort=sort,
            cursor=cursor,
        )
    except (httpx.HTTPError, TypeError, ValueError):
        return _error(
            "Live Hugging Face search is unavailable. Try again or switch to the curated offline list.",
            status_code=503,
            error_type="server_error",
            code="cookbook_hub_unavailable",
        )

    result = {
        "models": models,
        "next_cursor": next_cursor,
        "live": True,
        "source": "Hugging Face public model metadata · cached for 5 minutes · weights are not downloaded.",
    }
    cache[cache_key] = (now + 300.0, result)
    if len(cache) > 64:
        expired_keys = sorted(cache, key=lambda key: cache[key][0])[: len(cache) - 64]
        for expired_key in expired_keys:
            cache.pop(expired_key, None)
    return OrjsonResponse(result)


async def admin_state(request: Request) -> Response:
    settings: Settings = request.app.state.settings
    auth = _admin_auth(request, settings)
    if auth is not None:
        return auth
    return OrjsonResponse(
        {
            **_public_settings(settings),
            "api": request.app.state.api_format.public_dict(),
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


async def check_updates(request: Request) -> Response:
    auth = _admin_auth(request, request.app.state.settings)
    if auth is not None:
        return auth
    try:
        update = await check_for_updates(request.app.state.http_client, __version__)
    except UpdateError as exc:
        return _error(
            str(exc),
            status_code=503,
            error_type="server_error",
            code="update_check_failed",
        )
    return OrjsonResponse(update.as_dict())


async def install_update(request: Request) -> Response:
    settings: Settings = request.app.state.settings
    auth = _admin_auth(request, settings)
    if auth is not None:
        return auth
    if request.app.state.worker_count != 1:
        return _error(
            "The updater requires one Cortex worker to avoid concurrent package installation.",
            status_code=409,
            code="update_requires_single_worker",
        )
    payload, body_error = await _read_json(request)
    if body_error is not None:
        return body_error
    confirmed_commit = payload.get("commit_sha") if isinstance(payload, Mapping) else None
    if not isinstance(confirmed_commit, str):
        return _error("Confirm the commit ID shown by the update check.", status_code=400)

    async with request.app.state.update_lock:
        try:
            update = await check_for_updates(request.app.state.http_client, __version__)
            if not update.update_available:
                return _error(
                    f"Cortex {__version__} is already at the latest branch version "
                    f"({update.latest_version}).",
                    status_code=409,
                    code="no_update_available",
                )
            if update.commit_sha != confirmed_commit:
                return _error(
                    "The update branch changed after your check. Check again and confirm the new commit.",
                    status_code=409,
                    code="update_source_changed",
                )
            output = await install_from_commit(
                confirmed_commit,
                api_key_env=settings.api_key_env,
            )
        except UpdateError as exc:
            return _error(
                str(exc),
                status_code=502,
                error_type="server_error",
                code="update_install_failed",
            )

    return OrjsonResponse(
        {
            "ok": True,
            "current_version": update.current_version,
            "installed_version": update.latest_version,
            "source_repository": update.source_repository,
            "source_ref": update.source_ref,
            "commit_sha": update.commit_sha,
            "restart_required": True,
            "message": "The update was installed. Restart Cortex to load the new version.",
            "installer_output": output,
        }
    )


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
            # Validate the configuration that will remain before deleting the saved
            # overlay. A malformed base TOML must not erase a working override.
            validation_path = path.with_name(f".{path.name}.reset-check-{uuid.uuid4().hex}")
            env = dict(os.environ)
            env["CORTEX_ADMIN_CONFIG"] = str(validation_path)
            candidate = Settings.load(environ=env)
            await asyncio.to_thread(path.unlink, missing_ok=True)
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
        if model.runtime not in LOCAL_RUNTIMES:
            return _error("This model runtime is not managed by Cortex.", status_code=422)

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


async def stop_all_models(request: Request) -> Response:
    auth = _admin_auth(request, request.app.state.settings)
    if auth is not None:
        return auth
    async with request.app.state.admin_lock:
        settings: Settings = request.app.state.settings
        model_ids = [
            model_id
            for model_id, model in settings.models.items()
            if model.runtime in LOCAL_RUNTIMES
        ]
        manager = request.app.state.runtime_manager
        results = await asyncio.gather(
            *(manager.stop(model_id) for model_id in model_ids), return_exceptions=True
        )
        model_states: dict[str, dict[str, Any]] = {}
        for model_id, result in zip(model_ids, results, strict=True):
            if isinstance(result, BaseException):
                model_states[model_id] = {
                    "id": model_id,
                    "status": "error",
                    "error": str(result),
                }
            else:
                model_states[model_id] = result
        states = list(model_states.values())
    return OrjsonResponse(
        {
            "ok": all(state.get("status") == "stopped" for state in states),
            "stopped": [
                model_id
                for model_id, state in model_states.items()
                if state.get("status") == "stopped"
            ],
            "models": model_states,
        }
    )


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
