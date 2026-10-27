"""Local text/multimodal model hosting with streamed OpenAI-compatible routes."""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import re
import ssl
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from time import time
from urllib.parse import parse_qsl, quote, urlencode, urlsplit
from uuid import uuid4

import httpx
import orjson
from starlette.applications import Starlette
from starlette.datastructures import Headers, UploadFile
from starlette.middleware import Middleware
from starlette.requests import ClientDisconnect, Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from .api_config import APIFormatConfig
from .auth import SESSION_COOKIE_NAME, is_authenticated, session_signing_key
from .config import ConfigurationError, ModelConfig, Settings, resolve_admin_config_path
from .dashboard import (
    InferenceMetrics,
    InferenceResponseTelemetry,
    LoginAttemptLimiter,
    admin_metrics,
    admin_state,
    auth_login,
    auth_logout,
    auth_options,
    check_updates,
    dashboard,
    download_config,
    install_update,
    model_cookbook,
    reload_config,
    reset_config,
    reset_metrics,
    restart_model,
    runtime_status,
    start_model,
    stop_all_models,
    stop_model,
    test_model,
    update_config,
)
from .dashboard_models import detect_model_capabilities_route, model_files, upload_model_file
from .http_utils import content_length_from_headers
from .responses import OrjsonResponse
from .responses import error_response as _error
from .runtime import LocalRuntimeManager, automatic_threads, detect_hardware

logger = logging.getLogger("cortex_llmhoster.server")


@lru_cache(maxsize=32)
def _request_timeout(connect: float, read: float, write: float, pool: float) -> httpx.Timeout:
    """Reuse HTTPX timeout profiles across inference requests."""

    return httpx.Timeout(connect=connect, read=read, write=write, pool=pool)


def _initial_hardware_snapshot() -> dict[str, object]:
    """Return a cheap CPU summary while optional GPU probing runs in the background."""

    cpu_count = os.cpu_count() or 1
    return {
        "platform": platform.platform(),
        "cpu_count": cpu_count,
        "recommended_threads": automatic_threads(),
        "ram_total_gb": None,
        "ram_available_gb": None,
        "gpu": "Detecting hardware… (CPU inference is available)",
    }


def _auth_error(request: Request, settings: Settings) -> Response | None:
    if not settings.api_auth_enabled:
        return None
    expected = settings.api_key
    if expected is None:
        return _error(
            "API authentication is enabled but CORTEX_API_KEY is not configured.",
            status_code=503,
            error_type="authentication_error",
            code="admin_auth_not_configured",
        )
    session_secret = session_signing_key(
        expected,
        settings.admin_username,
        settings.admin_password,
    )
    if is_authenticated(
        expected,
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


def _select_model(
    settings: Settings,
    requested_id: object,
    *,
    required_capabilities: set[str] | None = None,
    runtime_manager: LocalRuntimeManager | None = None,
) -> tuple[ModelConfig | None, Response | None]:
    if requested_id is None:
        requested_id = settings.default_model
    if requested_id is None:
        return None, _error(
            "Specify a model or configure a default model.",
            status_code=400,
            param="model",
        )
    if not isinstance(requested_id, str) or not requested_id:
        return None, _error(
            "The model field must be a non-empty string.",
            status_code=400,
            param="model",
        )

    model = settings.models.get(requested_id)
    if model is None and requested_id.casefold() == "auto":
        required = required_capabilities or set()
        candidates = [
            candidate
            for candidate in settings.models.values()
            if not _missing_capabilities(required, candidate.capability_set)
        ]
        if not candidates:
            requested = ", ".join(sorted(required)) or "any declared task"
            return None, _error(
                f"No configured model supports the request capability: {requested}.",
                status_code=422,
                error_type="invalid_request_error",
                code="no_compatible_model",
                param="model",
            )

        def route_rank(candidate: ModelConfig) -> tuple[bool, bool]:
            is_running = (
                runtime_manager is not None
                and runtime_manager.local_base_url(candidate) is not None
            )
            return (not is_running, candidate.id != settings.default_model)

        model = min(candidates, key=route_rank)

    if model is None:
        return None, _error(
            f"Unknown model: {requested_id}.",
            status_code=404,
            error_type="invalid_request_error",
            code="model_not_found",
            param="model",
        )
    return model, None


def _upstream_query(request: Request, model: ModelConfig) -> str:
    """Keep the original query string byte-for-byte unless a model alias changes."""

    raw_query = request.scope.get("query_string", b"")
    if not raw_query:
        return ""

    original = raw_query.decode("ascii", errors="replace")
    pairs = parse_qsl(original, keep_blank_values=True)
    if model.id == model.upstream_model and not any(
        key == "model" and value.casefold() == "auto" for key, value in pairs
    ):
        return original

    changed = False
    rewritten: list[tuple[str, str]] = []
    for key, value in pairs:
        if key == "model" and (
            value == model.id or (value.casefold() == "auto" and model.id != "auto")
        ):
            rewritten.append((key, model.upstream_model))
            changed = changed or value != model.upstream_model
        else:
            rewritten.append((key, value))
    return urlencode(rewritten, doseq=True) if changed else original


def _local_model_url(request: Request, model: ModelConfig, base_url: str) -> str:
    route_path = request.path_params.get("path", "")
    segments = route_path.split("/")
    if not route_path or any(segment in {".", ".."} for segment in segments):
        raise ValueError("Invalid API path.")

    # This base URL is provided by Cortex's loopback-only llama-server manager.
    encoded_path = quote(route_path, safe="/-._~!$&'()*+,;=:@")
    url = f"{base_url}/" + encoded_path
    query = _upstream_query(request, model)
    if query:
        url += "?" + query
    return url


_MODEL_BODY_ENDPOINTS = frozenset(
    {
        "audio/speech",
        "audio/transcriptions",
        "audio/translations",
        "chat/completions",
        "completions",
        "embeddings",
        "fine_tuning/jobs",
        "images/edits",
        "images/generations",
        "images/variations",
        "moderations",
        "moderation",
        "content-moderation",
        "reranking",
        "reranker",
        "ocr",
        "ocr/parse",
        "responses",
        "rerank",
        "audio/analysis",
        "audio-to-text",
        "audio-to-audio",
        "audio-understanding",
        "audio-generation",
        "audio-perception",
        "audio-classification",
        "audio-translation",
        "automatic-speech-recognition",
        "speech-recognition",
        "video/analysis",
        "video/generations",
        "video/understanding",
        "videos/analysis",
        "videos/generations",
        "videos/understanding",
        "image-to-image",
        "image-to-text",
        "text-to-image",
        "text-to-text",
        "text-to-video",
        "image-to-video",
        "video-to-text",
        "video-to-video",
        "video-summarization",
        "video-captioning",
        "video-generation",
        "video-understanding",
        "image-understanding",
        "vision-understanding",
        "vision",
        "llm",
        "text-generation",
        "audio-transcription",
        "image-generation",
        "object-detection",
        "image-segmentation",
        "visual-question-answering",
        "depth-estimation",
        "image-captioning",
        "image-classification",
        "text-recognition",
        "document-ocr",
        "text-summarization",
        "text-translation",
        "text-classification",
        "text-embedding",
        "image-embedding",
        "image-editing",
        "speech-to-text",
        "text-to-audio",
        "text-to-speech",
        "speech-to-speech",
        "t2t",
        "t2i",
        "i2i",
        "t2v",
        "i2v",
        "v2v",
        "stt",
        "tts",
        "asr",
        "vlm",
    }
)

_ENDPOINT_CAPABILITIES = {
    "audio/speech": "audio_generation",
    "audio/transcriptions": "audio_transcription",
    "audio/translations": "audio_transcription",
    "embeddings": "embeddings",
    "images/edits": "image_generation",
    "images/generations": "image_generation",
    "images/variations": "image_generation",
    "moderations": "moderation",
    "rerank": "reranking",
    "video/generations": "video_generation",
    "videos/generations": "video_generation",
    "videos/understanding": "video_understanding",
    "video/understanding": "video_understanding",
    "video/analysis": "video_understanding",
    "videos/analysis": "video_understanding",
    "ocr": "ocr",
    "ocr/parse": "ocr",
    "audio/analysis": "audio_understanding",
    "audio-perception": "audio_understanding",
    "audio-classification": "audio_understanding",
    "audio-generation": "audio_generation",
    "audio-to-text": "audio_transcription",
    "audio-translation": "audio_transcription",
    "automatic-speech-recognition": "audio_transcription",
    "speech-recognition": "audio_transcription",
    "text-generation": "text_generation",
    "llm": "text_generation",
    "vision": "vision_understanding",
    "vision-understanding": "vision_understanding",
    "object-detection": "vision_understanding",
    "image-segmentation": "vision_understanding",
    "visual-question-answering": "vision_understanding",
    "depth-estimation": "vision_understanding",
    "video-generation": "video_generation",
    "video-understanding": "video_understanding",
    "moderation": "moderation",
    "content-moderation": "moderation",
    "reranking": "reranking",
    "reranker": "reranking",
    "audio-transcription": "audio_transcription",
    "audio-understanding": "audio_understanding",
    "audio-to-audio": "audio_generation",
    "image-to-image": "image_generation",
    "image-to-text": "vision_understanding",
    "text-to-image": "image_generation",
    "text-to-text": "text_generation",
    "text-to-video": "video_generation",
    "image-to-video": "video_generation",
    "video-to-text": "video_understanding",
    "video-to-video": "video_generation",
    "video-summarization": "video_understanding",
    "video-captioning": "video_understanding",
    "image-understanding": "vision_understanding",
    "image-captioning": "vision_understanding",
    "image-classification": "vision_understanding",
    "text-recognition": "ocr",
    "document-ocr": "ocr",
    "text-summarization": "text_generation",
    "text-translation": "text_generation",
    "text-classification": "text_generation",
    "text-embedding": "embeddings",
    "image-embedding": "embeddings",
    "image-generation": "image_generation",
    "image-editing": "image_generation",
    "speech-to-text": "audio_transcription",
    "text-to-audio": "audio_generation",
    "text-to-speech": "audio_generation",
    "speech-to-speech": "audio_generation",
    "t2t": "text_generation",
    "t2i": "image_generation",
    "i2i": "image_generation",
    "t2v": "video_generation",
    "i2v": "video_generation",
    "v2v": "video_generation",
    "stt": "audio_transcription",
    "tts": "audio_generation",
    "asr": "audio_transcription",
    "vlm": "vision_understanding",
}

_CAPABILITY_ALTERNATIVES = {
    "text_generation": frozenset(
        {
            "text_generation",
            "text_to_text",
            "llm",
            "language_model",
            "chat",
            "completion",
            "t2t",
            "text_summarization",
            "text_translation",
            "text_classification",
        }
    ),
    "vision_understanding": frozenset(
        {
            "vision_understanding",
            "vision",
            "vlm",
            "image_understanding",
            "image_to_text",
            "image_captioning",
            "image_classification",
            "object_detection",
            "image_segmentation",
            "visual_question_answering",
            "depth_estimation",
        }
    ),
    "ocr": frozenset({"ocr", "text_recognition", "document_ocr"}),
    "image_generation": frozenset(
        {
            "image_generation",
            "text_to_image",
            "image_to_image",
            "image_editing",
            "t2i",
            "i2i",
        }
    ),
    "audio_transcription": frozenset(
        {
            "audio_transcription",
            "speech_to_text",
            "audio_to_text",
            "audio_translation",
            "automatic_speech_recognition",
            "speech_recognition",
            "asr",
            "stt",
        }
    ),
    "audio_understanding": frozenset(
        {
            "audio_understanding",
            "audio_perception",
            "audio_classification",
            "sound_classification",
            "speech_to_text",
            "audio_to_text",
            "asr",
            "stt",
        }
    ),
    "audio_generation": frozenset(
        {
            "audio_generation",
            "text_to_speech",
            "speech_generation",
            "speech_synthesis",
            "speech_to_speech",
            "audio_to_audio",
            "music_generation",
            "tts",
        }
    ),
    "video_understanding": frozenset(
        {
            "video_understanding",
            "video_to_text",
            "video_summarization",
            "video_captioning",
            "video_question_answering",
            "video_classification",
            "action_recognition",
        }
    ),
    "video_generation": frozenset(
        {
            "video_generation",
            "text_to_video",
            "image_to_video",
            "video_to_video",
            "t2v",
            "i2v",
            "v2v",
        }
    ),
    "embeddings": frozenset(
        {"embeddings", "text_embedding", "image_embedding", "feature_extraction"}
    ),
    "reranking": frozenset({"reranking", "reranker"}),
    "moderation": frozenset({"moderation", "content_moderation"}),
}


def _missing_capabilities(required: set[str], available: frozenset[str]) -> set[str]:
    normalized_available = {
        capability.replace("/", "_").replace("-", "_") for capability in available
    }
    return {
        capability
        for capability in required
        if not (
            _CAPABILITY_ALTERNATIVES.get(capability, frozenset({capability})) & normalized_available
        )
    }


def _message_capabilities(value: object) -> set[str]:
    """Infer common input/output modalities from OpenAI-style JSON payloads."""

    found: set[str] = set()
    if isinstance(value, dict):
        media_type = value.get("type")
        if isinstance(media_type, str):
            normalized_type = media_type.lower().replace("-", "_")
            if normalized_type in {"image", "image_url", "input_image"}:
                found.add("vision_understanding")
            elif normalized_type in {"input_audio", "audio_url", "input_audio_buffer"}:
                found.add("audio_understanding")
            elif normalized_type in {"video", "video_url", "input_video"}:
                found.add("video_understanding")
        for key, nested in value.items():
            normalized_key = str(key).lower().replace("-", "_")
            if normalized_key in {"image", "image_url", "images", "input_image"}:
                found.add("vision_understanding")
            elif normalized_key in {"input_audio", "audio_url", "audio_input"}:
                found.add("audio_understanding")
            elif normalized_key in {"video", "video_url", "videos", "input_video"}:
                found.add("video_understanding")
            elif normalized_key in {"modalities", "output_modalities"} and isinstance(nested, list):
                modalities = {str(item).lower() for item in nested}
                if "audio" in modalities:
                    found.add("audio_generation")
                if "image" in modalities:
                    found.add("image_generation")
                if "video" in modalities:
                    found.add("video_generation")
            found.update(_message_capabilities(nested))
    elif isinstance(value, list):
        for item in value:
            found.update(_message_capabilities(item))
    # Do not infer a media request from arbitrary prose: a text prompt may
    # mention a filename or data-URI example without attaching that media.
    return found


def _required_capabilities(
    route_path: str,
    body: dict[str, object] | None,
    declared_capabilities: frozenset[str] = frozenset(),
) -> set[str]:
    normalized_path = route_path.strip("/").lower()
    required = _ENDPOINT_CAPABILITIES.get(normalized_path)
    if required is None:
        required = _ENDPOINT_CAPABILITIES.get(normalized_path.replace("_", "-"))
    if required is not None:
        return {required}
    if normalized_path in {"chat/completions", "completions", "responses"}:
        detected = _message_capabilities(body or {})
        return detected or {"text_generation"}
    if normalized_path.startswith("ocr/"):
        return {"ocr"}
    custom_capability = normalized_path.replace("-", "_").replace("/", "_")
    if custom_capability in declared_capabilities:
        return {custom_capability}
    return set()


_FORWARD_REQUEST_HEADERS = (
    "accept",
    "content-type",
    "user-agent",
    "openai-organization",
    "openai-project",
    "openai-beta",
    "idempotency-key",
    "x-request-id",
    "x-client-request-id",
    "traceparent",
)

_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def _response_headers(upstream: httpx.Response) -> dict[str, str]:
    # Avoid allocating an exclusion set unless the response nominates extra
    # hop-by-hop headers; Connection is absent in the common case.
    connection = upstream.headers.get("connection")
    if connection:
        excluded = _HOP_BY_HOP_HEADERS | {
            token.strip().lower() for token in connection.split(",") if token.strip()
        }
    else:
        excluded = _HOP_BY_HOP_HEADERS
    # The local runtime does not own Cortex's browser-origin cookie namespace.
    return {
        name: value
        for name, value in upstream.headers.items()
        if name.lower() not in excluded and name.lower() != "set-cookie"
    }


def _model_timeout() -> OrjsonResponse:
    return _error(
        "The local model server did not respond before the request timeout.",
        status_code=504,
        error_type="server_error",
        code="model_server_timeout",
    )


def _local_model_unavailable() -> OrjsonResponse:
    return _error(
        "The local model server is temporarily unavailable.",
        status_code=502,
        error_type="server_error",
        code="model_server_unavailable",
    )


async def _read_limited_json_body(
    request: Request, limit: int
) -> tuple[bytes | None, Response | None]:
    try:
        declared_length = content_length_from_headers(request.headers)
    except ValueError:
        return None, _error("Invalid Content-Length header.", status_code=400)
    if declared_length is not None and declared_length > limit:
        return None, _error(
            f"JSON request body exceeds the {limit}-byte limit.",
            status_code=413,
            code="request_too_large",
        )

    body = bytearray()
    try:
        async for chunk in request.stream():
            if len(body) + len(chunk) > limit:
                return None, _error(
                    f"JSON request body exceeds the {limit}-byte limit.",
                    status_code=413,
                    code="request_too_large",
                )
            body.extend(chunk)
    except ClientDisconnect:
        return None, _error(
            "The request body was interrupted.",
            status_code=400,
            code="request_body_interrupted",
        )
    return bytes(body), None


class _MediaBodyTooLarge(Exception):
    """Signal that a streamed media upload exceeded the configured limit."""


def _media_size_error(limit: int) -> Response:
    return _error(
        f"Media request body exceeds the {limit}-byte limit.",
        status_code=413,
        code="request_too_large",
    )


async def _limited_media_stream(request: Request, limit: int):
    """Stream a raw media body with a byte ceiling; do not buffer it in memory."""

    try:
        declared_length = content_length_from_headers(request.headers)
    except ValueError:
        return None, _error("Invalid Content-Length header.", status_code=400)
    if declared_length is not None and declared_length > limit:
        return None, _media_size_error(limit)

    async def stream():
        received = 0
        async for chunk in request.stream():
            received += len(chunk)
            if received > limit:
                raise _MediaBodyTooLarge
            yield chunk

    return stream(), None


async def _read_limited_multipart_form(request: Request, limit: int):
    """Parse a bounded multipart form, spooling media files instead of buffering RAM."""

    try:
        declared_length = content_length_from_headers(request.headers)
    except ValueError:
        return None, _error("Invalid Content-Length header.", status_code=400)
    if declared_length is not None and declared_length > limit:
        return None, _error(
            f"Media request body exceeds the {limit}-byte limit.",
            status_code=413,
            code="request_too_large",
        )

    received = 0
    original_receive = request.receive

    async def limited_receive():
        nonlocal received
        message = await original_receive()
        if message.get("type") == "http.request":
            received += len(message.get("body", b""))
            if received > limit:
                raise _MediaBodyTooLarge
        return message

    parsing_request = Request(request.scope, receive=limited_receive)
    try:
        form = await parsing_request.form(max_files=16, max_fields=100, max_part_size=1024 * 1024)
    except _MediaBodyTooLarge:
        return None, _error(
            f"Media request body exceeds the {limit}-byte limit.",
            status_code=413,
            code="request_too_large",
        )
    except ClientDisconnect:
        return None, _error(
            "The request body was interrupted.",
            status_code=400,
            code="request_body_interrupted",
        )
    except Exception as exc:  # noqa: BLE001 - malformed multipart should be a 400
        logger.debug("Could not parse multipart media request: %s", exc)
        return None, _error("Request body is not valid multipart form data.", status_code=400)
    return form, None


async def health(_: Request) -> Response:
    """Liveness endpoint; it does not make a slow downstream health check."""

    return OrjsonResponse({"status": "ok"}, headers={"cache-control": "no-store"})


async def readiness(request: Request) -> Response:
    settings: Settings = request.app.state.settings
    if not settings.models:
        return OrjsonResponse(
            {"status": "not_ready", "reason": "no_models_configured"},
            status_code=503,
            headers={"cache-control": "no-store"},
        )
    default_model = settings.models.get(settings.default_model or "")
    if default_model is not None:
        runtime_state = request.app.state.runtime_manager.state(default_model.id)
        if runtime_state["status"] != "running":
            return OrjsonResponse(
                {
                    "status": "not_ready",
                    "reason": "default_local_model_not_running",
                    "model": default_model.id,
                    "runtime_status": runtime_state["status"],
                },
                status_code=503,
                headers={"cache-control": "no-store"},
            )
    elif not any(
        request.app.state.runtime_manager.local_base_url(model) is not None
        for model in settings.models.values()
    ):
        return OrjsonResponse(
            {"status": "not_ready", "reason": "no_local_models_running"},
            status_code=503,
            headers={"cache-control": "no-store"},
        )
    return OrjsonResponse(
        {"status": "ready", "models": len(settings.models)},
        headers={"cache-control": "no-store"},
    )


async def list_models(request: Request) -> Response:
    settings: Settings = request.app.state.settings
    auth_response = _auth_error(request, settings)
    if auth_response is not None:
        return auth_response

    created = request.app.state.models_created_at
    models = [
        {
            "id": model.id,
            "object": "model",
            "created": created,
            "owned_by": "cortex-llmhoster",
            "runtime": model.runtime,
            "capabilities": list(model.effective_capabilities),
        }
        for model in settings.models.values()
    ]
    if settings.models and "auto" not in settings.models:
        capabilities = sorted(
            {
                capability
                for model in settings.models.values()
                for capability in model.effective_capabilities
            }
        )
        models.append(
            {
                "id": "auto",
                "object": "model",
                "created": created,
                "owned_by": "cortex-llmhoster",
                "runtime": "cortex-auto-router",
                "capabilities": capabilities,
            }
        )
    return OrjsonResponse({"object": "list", "data": models})


async def inference(request: Request) -> Response:
    settings: Settings = request.app.state.settings
    auth_response = _auth_error(request, settings)
    if auth_response is not None:
        return auth_response

    json_body: dict[str, object] | None = None
    body: bytes | None = None
    multipart_form = None
    multipart_items: list[tuple[str, object]] = []
    request_content = None
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    is_json = content_type == "application/json" or content_type.endswith("+json")
    is_multipart = content_type == "multipart/form-data"

    try:
        if is_json:
            body, body_error = await _read_limited_json_body(request, settings.max_json_body_bytes)
            if body_error is not None:
                return body_error
            if body is None:
                return _error(
                    "The request body could not be read.",
                    status_code=500,
                    code="request_body_read_failed",
                )
            if body:
                try:
                    parsed_body = orjson.loads(body)
                except orjson.JSONDecodeError:
                    return _error("Request body is not valid JSON.", status_code=400)
                if not isinstance(parsed_body, dict):
                    return _error(
                        "JSON request body must be an object.",
                        status_code=400,
                        code="invalid_request_body",
                    )
                json_body = parsed_body
        elif is_multipart and request.query_params.get("model") is None:
            multipart_form, body_error = await _read_limited_multipart_form(
                request, settings.max_media_body_bytes
            )
            if body_error is not None:
                return body_error
            multipart_items = list(multipart_form.multi_items())

        body_model: object = json_body.get("model") if json_body is not None else None
        if body_model is None and multipart_form is not None:
            body_model = next(
                (
                    value
                    for key, value in multipart_items
                    if key == "model" and isinstance(value, str)
                ),
                None,
            )
        requested_model = body_model
        if requested_model is None:
            requested_model = request.query_params.get("model")
        route_path = request.path_params.get("path", "").strip("/")
        all_capabilities = frozenset(
            capability
            for configured_model in settings.models.values()
            for capability in configured_model.effective_capabilities
        )
        routing_capabilities = _required_capabilities(route_path, json_body, all_capabilities)
        model, model_error = _select_model(
            settings,
            requested_model,
            required_capabilities=routing_capabilities,
            runtime_manager=request.app.state.runtime_manager,
        )
        if model_error is not None:
            return model_error
        if model is None:
            return _error(
                "The configured model could not be selected.",
                status_code=500,
                code="model_selection_failed",
            )

        required_capabilities = _required_capabilities(route_path, json_body, model.capability_set)
        missing_capabilities = _missing_capabilities(required_capabilities, model.capability_set)
        if missing_capabilities:
            missing = ", ".join(sorted(missing_capabilities))
            available = ", ".join(model.effective_capabilities) or "none declared"
            return _error(
                f"Model {model.id!r} does not declare required capability: {missing}. "
                f"Available capabilities: {available}.",
                status_code=422,
                error_type="invalid_request_error",
                code="unsupported_model_capability",
                param="model",
            )

        should_inject_model = body_model is None and (
            route_path in _MODEL_BODY_ENDPOINTS or bool(required_capabilities)
        )
        should_rewrite_model = body_model is not None and body_model != model.upstream_model
        if json_body is not None and (should_inject_model or should_rewrite_model):
            rewritten = dict(json_body)
            rewritten["model"] = model.upstream_model
            body = orjson.dumps(rewritten)

        if multipart_form is None and body is None:
            request_content, body_error = await _limited_media_stream(
                request, settings.max_media_body_bytes
            )
            if body_error is not None:
                return body_error

        base_url = request.app.state.runtime_manager.local_base_url(model)
        if base_url is None:
            runtime_state = request.app.state.runtime_manager.state(model.id)
            detail = runtime_state.get("error") or (
                "Start this local model from the Cortex dashboard and wait until it is ready."
            )
            return _error(
                detail,
                status_code=503,
                error_type="server_error",
                code="local_model_not_ready",
            )
        try:
            target_url = _local_model_url(request, model, base_url)
        except ValueError:
            return _error("Invalid API path.", status_code=400)

        outgoing_headers = {
            name: request.headers[name]
            for name in _FORWARD_REQUEST_HEADERS
            if name in request.headers
            and not (multipart_form is not None and name == "content-type")
        }
        outgoing_headers["x-request-id"] = request.state.request_id
        client: httpx.AsyncClient = request.app.state.http_client
        metrics: InferenceMetrics = request.app.state.metrics
        metrics_started_at = metrics.start(model.id)
        try:
            request_kwargs: dict[str, object]
            if multipart_form is not None:
                form_fields: dict[str, str | list[str]] = {}
                form_files: list[tuple[str, tuple[object, ...]]] = []
                found_model = False
                for key, value in multipart_items:
                    if isinstance(value, UploadFile):
                        form_files.append(
                            (
                                key,
                                (
                                    value.filename or "upload.bin",
                                    value.file,
                                    value.content_type or "application/octet-stream",
                                ),
                            )
                        )
                    else:
                        text_value = str(value)
                        if key == "model":
                            found_model = True
                            if should_inject_model or should_rewrite_model:
                                text_value = model.upstream_model
                        if key not in form_fields:
                            form_fields[key] = text_value
                        else:
                            previous = form_fields[key]
                            if isinstance(previous, list):
                                previous.append(text_value)
                            else:
                                form_fields[key] = [previous, text_value]
                if should_inject_model and not found_model:
                    form_fields["model"] = model.upstream_model
                request_kwargs = {"data": form_fields, "files": form_files}
            else:
                request_kwargs = {"content": body if body is not None else request_content}

            upstream_request = client.build_request(
                request.method,
                target_url,
                headers=outgoing_headers,
                timeout=_request_timeout(
                    settings.connect_timeout,
                    settings.read_timeout,
                    settings.write_timeout,
                    settings.pool_timeout,
                ),
                **request_kwargs,
            )
            upstream_response = await client.send(upstream_request, stream=True)
        except asyncio.CancelledError:
            metrics.finish(metrics_started_at, 499)
            raise
        except ClientDisconnect:
            metrics.finish(metrics_started_at, 499)
            return _error(
                "The request body was interrupted.",
                status_code=400,
                code="request_body_interrupted",
            )
        except _MediaBodyTooLarge:
            metrics.finish(metrics_started_at, 413)
            return _media_size_error(settings.max_media_body_bytes)
        except httpx.PoolTimeout:
            metrics.finish(metrics_started_at, 503)
            return _error(
                "Cortex is at capacity; retry the request shortly.",
                status_code=503,
                error_type="server_error",
                code="server_capacity_exceeded",
            )
        except httpx.TimeoutException:
            metrics.finish(metrics_started_at, 504)
            return _model_timeout()
        except httpx.RequestError:
            metrics.finish(metrics_started_at, 502)
            logger.warning(
                "Could not reach local runtime for model %s (request_id=%s)",
                model.id,
                request.state.request_id,
            )
            return _local_model_unavailable()
        except Exception:
            metrics.finish(metrics_started_at, 500)
            logger.exception(
                "Unexpected failure handling local-runtime request for model %s (request_id=%s)",
                model.id,
                request.state.request_id,
            )
            return _error(
                "Cortex encountered an unexpected error while processing the request.",
                status_code=500,
                error_type="server_error",
                code="model_request_failed",
            )

        response_telemetry = InferenceResponseTelemetry(
            metrics, metrics_started_at, upstream_response.headers.get("content-type", "")
        )

        async def relay():
            final_status = upstream_response.status_code
            try:
                # Mock/custom transports may return an eagerly-buffered response even
                # though the production client requested a streamed response.
                if upstream_response.is_stream_consumed:
                    if upstream_response.content:
                        response_telemetry.feed(upstream_response.content)
                        yield upstream_response.content
                else:
                    async for chunk in upstream_response.aiter_raw():
                        if chunk:
                            response_telemetry.feed(chunk)
                            yield chunk
            except asyncio.CancelledError:
                final_status = 499
                raise
            except httpx.HTTPError:
                final_status = 502
                logger.warning(
                    "Local runtime stream failed for model %s (request_id=%s)",
                    model.id,
                    request.state.request_id,
                )
                raise
            except Exception:
                final_status = 502
                logger.exception(
                    "Unexpected local-runtime stream failure for model %s (request_id=%s)",
                    model.id,
                    request.state.request_id,
                )
                raise
            finally:
                try:
                    response_telemetry.finish(final_status)
                finally:
                    try:
                        await upstream_response.aclose()
                    finally:
                        metrics.finish(metrics_started_at, final_status)

        headers = _response_headers(upstream_response)
        headers["x-cortex-selected-model"] = model.id
        if "text/event-stream" in upstream_response.headers.get("content-type", "").lower():
            headers.setdefault("cache-control", "no-cache")
            headers["x-accel-buffering"] = "no"

        if request.method == "HEAD":
            await upstream_response.aclose()
            response_telemetry.finish(upstream_response.status_code)
            metrics.finish(metrics_started_at, upstream_response.status_code)
            return Response(status_code=upstream_response.status_code, headers=headers)

        return StreamingResponse(
            relay(),
            status_code=upstream_response.status_code,
            headers=headers,
        )
    finally:
        if multipart_form is not None:
            await multipart_form.close()


class InferenceRequestLimiter:
    """Event-loop-local counter that rejects excess requests without queuing."""

    __slots__ = ("active", "limit")

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("Inference request limit must be at least 1.")
        self.limit = limit
        self.active = 0

    def try_acquire(self) -> bool:
        if self.active >= self.limit:
            return False
        self.active += 1
        return True

    def release(self) -> None:
        if self.active < 1:
            raise RuntimeError("Inference request limiter released without an active request.")
        self.active -= 1

    def resize(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("Inference request limit must be at least 1.")
        self.limit = limit


_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,128}")


class RequestIDMiddleware:
    """Attach a safe correlation ID to every HTTP response and downstream call."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        supplied = headers.get("x-request-id", "")
        request_id = supplied if _REQUEST_ID_PATTERN.fullmatch(supplied) else uuid4().hex
        state = scope.setdefault("state", {})
        if not isinstance(state, dict):
            state = {}
            scope["state"] = state
        state["request_id"] = request_id

        async def send_with_request_id(message) -> None:
            if message.get("type") == "http.response.start":
                response_headers = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() != b"x-request-id"
                ]
                response_headers.append((b"x-request-id", request_id.encode("ascii")))
                message = {**message, "headers": response_headers}
            await send(message)

        await self.app(scope, receive, send_with_request_id)


class RequestIDStarlette(Starlette):
    """Run request-ID handling outside Starlette's server-error responses."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._request_id_application = RequestIDMiddleware(super().__call__)

    async def __call__(self, scope, receive, send) -> None:
        await self._request_id_application(scope, receive, send)


class InferenceResponseHeadersMiddleware:
    """Prevent shared caches from retaining prompts, outputs, or model listings."""

    def __init__(self, app, *, api_base_path: str, models_endpoint: str) -> None:
        self.app = app
        normalized_base = api_base_path.rstrip("/")
        self._api_prefix = f"{normalized_base}/" if normalized_base else "/"
        self._models_endpoint = models_endpoint

    async def __call__(self, scope, receive, send) -> None:
        path = scope.get("path", "")
        if scope.get("type") != "http" or not (
            path.startswith(self._api_prefix) or path == self._models_endpoint
        ):
            await self.app(scope, receive, send)
            return

        async def send_uncached(message) -> None:
            if message.get("type") == "http.response.start":
                headers = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() not in {b"cache-control", b"pragma", b"x-content-type-options"}
                ]
                headers.extend(
                    (
                        (b"cache-control", b"no-store"),
                        (b"pragma", b"no-cache"),
                        (b"x-content-type-options", b"nosniff"),
                    )
                )
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_uncached)


class InferenceCapacityMiddleware:
    """Bound active inference work before the route consumes request bodies."""

    def __init__(
        self,
        app,
        limiter: InferenceRequestLimiter,
        api_base_path: str,
        models_endpoint: str,
    ) -> None:
        self.app = app
        self._limiter = limiter
        normalized_base = api_base_path.rstrip("/")
        self._api_prefix = f"{normalized_base}/" if normalized_base else "/"
        self._models_endpoint = models_endpoint

    async def __call__(self, scope, receive, send) -> None:
        path = scope.get("path", "")
        in_api = path.startswith(self._api_prefix)
        if (
            scope.get("type") != "http"
            or not in_api
            or (path == self._models_endpoint and scope.get("method") == "GET")
        ):
            await self.app(scope, receive, send)
            return

        # Reject overload before the inference handler parses payloads or opens
        # more local requests. The model server remains responsible for parallel
        # decoding within this application-level ceiling.
        if not self._limiter.try_acquire():
            response = _error(
                "Cortex is at capacity; retry the request shortly.",
                status_code=503,
                error_type="server_error",
                code="server_capacity_exceeded",
                headers={"retry-after": "1"},
            )
            await response(scope, receive, send)
            return

        try:
            await self.app(scope, receive, send)
        finally:
            self._limiter.release()


class AdminResponseHeadersMiddleware:
    """Prevent browser/proxy caches from retaining authenticated dashboard data."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        path = scope.get("path", "")
        if scope.get("type") != "http" or not path.startswith("/admin/api/"):
            await self.app(scope, receive, send)
            return

        async def send_uncached(message) -> None:
            if message.get("type") == "http.response.start":
                headers = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() not in {b"cache-control", b"pragma", b"x-content-type-options"}
                ]
                headers.extend(
                    (
                        (b"cache-control", b"no-store"),
                        (b"pragma", b"no-cache"),
                        (b"x-content-type-options", b"nosniff"),
                    )
                )
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_uncached)


class SessionOriginMiddleware:
    """Require verifiable same-origin metadata for unsafe session-cookie requests."""

    _SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

    def __init__(self, app, *, cookie_name: str) -> None:
        self.app = app
        self.cookie_name = f"{cookie_name}="

    @staticmethod
    def _origin_key(scheme: str, authority: str) -> tuple[str, str, int] | None:
        scheme = scheme.strip().lower()
        if (
            scheme not in {"http", "https"}
            or not authority
            or authority.endswith(":")
            or "\\" in authority
            or any(c.isspace() for c in authority)
        ):
            return None
        try:
            parsed = urlsplit(f"{scheme}://{authority}")
            if (
                parsed.username is not None
                or parsed.password is not None
                or parsed.path not in {"", "/"}
                or parsed.query
                or parsed.fragment
                or not parsed.hostname
            ):
                return None
            raw_hostname = parsed.hostname
            if raw_hostname.endswith(".."):
                return None
            normalized_hostname = raw_hostname.removesuffix(".")
            if not normalized_hostname:
                return None
            hostname = normalized_hostname.encode("idna").decode("ascii").lower()
            explicit_port = parsed.port
            port = (
                explicit_port if explicit_port is not None else (443 if scheme == "https" else 80)
            )
        except (UnicodeError, ValueError):
            return None
        return scheme, hostname, port

    @classmethod
    def _matches_origin(
        cls,
        candidate: str,
        request_scheme: str,
        request_host: str,
        *,
        allow_path: bool,
    ) -> bool:
        try:
            parsed = urlsplit(candidate)
        except ValueError:
            return False
        if not parsed.scheme or not parsed.netloc:
            return False
        if not allow_path and (parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
            return False
        candidate_key = cls._origin_key(parsed.scheme, parsed.netloc)
        request_key = cls._origin_key(request_scheme, request_host)
        return candidate_key is not None and candidate_key == request_key

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") == "http" and scope.get("method") not in self._SAFE_METHODS:
            headers = Headers(scope=scope)
            cookies = headers.get("cookie", "").split(";")
            has_session = any(part.strip().startswith(self.cookie_name) for part in cookies)
            if has_session:
                forwarded_scheme = headers.get("x-forwarded-proto", "").split(",", 1)[0]
                request_scheme = (forwarded_scheme or scope.get("scheme", "http")).strip()
                forwarded_host = headers.get("x-forwarded-host", "").split(",", 1)[0]
                request_host = (forwarded_host or headers.get("host", "")).strip()
                origin = headers.get("origin")
                if origin:
                    same_origin = self._matches_origin(
                        origin, request_scheme, request_host, allow_path=False
                    )
                else:
                    referer = headers.get("referer")
                    if referer:
                        same_origin = self._matches_origin(
                            referer, request_scheme, request_host, allow_path=True
                        )
                    else:
                        # Without Origin or Referer, only explicit browser metadata can
                        # safely establish same-origin. Missing metadata is not trusted.
                        same_origin = headers.get("sec-fetch-site", "").lower() == "same-origin"
                if not same_origin:
                    response = _error(
                        "Cross-origin or unverifiable session requests are not allowed.",
                        status_code=403,
                        error_type="authentication_error",
                        code="csrf_origin_rejected",
                    )
                    await response(scope, receive, send)
                    return
        await self.app(scope, receive, send)


def create_app(
    settings: Settings | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    admin_config_path: str | Path | None = None,
    api_config_path: str | Path | None = None,
) -> Starlette:
    """Build an ASGI app. ``transport`` is injectable for tests and adapters."""

    watch_config_files = settings is None
    runtime_settings = settings if settings is not None else Settings.load()
    api_format = APIFormatConfig.load(api_config_path)

    @asynccontextmanager
    async def lifespan(app: Starlette):
        limits = httpx.Limits(
            max_connections=runtime_settings.max_connections,
            max_keepalive_connections=runtime_settings.max_keepalive_connections,
            keepalive_expiry=runtime_settings.keepalive_expiry,
        )
        timeout = _request_timeout(
            runtime_settings.connect_timeout,
            runtime_settings.read_timeout,
            runtime_settings.write_timeout,
            runtime_settings.pool_timeout,
        )
        app.state.http_client = httpx.AsyncClient(
            limits=limits,
            timeout=timeout,
            transport=transport,
            verify=ssl.create_default_context(),
            follow_redirects=False,
            trust_env=False,
        )
        app.state.models_created_at = int(time())
        runtime_manager = LocalRuntimeManager(
            app.state.settings,
            app.state.http_client,
            hardware=_initial_hardware_snapshot(),
        )
        app.state.runtime_manager = runtime_manager

        async def refresh_hardware() -> None:
            try:
                hardware = await asyncio.to_thread(detect_hardware)
            except Exception as exc:  # noqa: BLE001 - hardware detection is optional
                logger.warning("Could not detect host hardware: %s", exc)
                runtime_manager.hardware = {
                    **runtime_manager.hardware,
                    "gpu": "Hardware detection unavailable (CPU inference is available)",
                }
                return
            runtime_manager.hardware = hardware

        await runtime_manager.start_configured()
        hardware_task = asyncio.create_task(refresh_hardware(), name="cortex-hardware-detection")
        app.state.hardware_task = hardware_task
        watcher: asyncio.Task[None] | None = None
        if watch_config_files:

            async def watch_files() -> None:
                last_revision: tuple[tuple[str, int, int] | None, ...] | None = None
                toml_path = Path(os.environ.get("CORTEX_CONFIG", "cortex.toml"))
                admin_path: Path = app.state.admin_config_path
                while True:
                    await asyncio.sleep(1.0)
                    revision_items: list[tuple[str, int, int] | None] = []
                    for watched_path in (toml_path, admin_path):
                        try:
                            stat = watched_path.stat()
                            revision_items.append(
                                (str(watched_path), stat.st_mtime_ns, stat.st_size)
                            )
                        except OSError:
                            revision_items.append(None)
                    revision = tuple(revision_items)
                    if revision == last_revision:
                        continue
                    last_revision = revision
                    async with app.state.admin_lock:
                        env = dict(os.environ)
                        env["CORTEX_ADMIN_CONFIG"] = str(admin_path)
                        try:
                            updated = await asyncio.to_thread(Settings.load, environ=env)
                        except (ConfigurationError, OSError) as exc:
                            logger.warning("Could not reload changed Cortex config: %s", exc)
                            continue
                        await runtime_manager.apply_settings(updated)
                        app.state.inference_limiter.resize(updated.max_inference_requests)
                        app.state.settings = updated

            watcher = asyncio.create_task(watch_files(), name="cortex-config-watcher")
        try:
            yield
        finally:
            if watcher is not None:
                watcher.cancel()
                try:
                    await watcher
                except asyncio.CancelledError:
                    pass
            await runtime_manager.close()
            await asyncio.gather(hardware_task, return_exceptions=True)
            await app.state.http_client.aclose()

    routes = [
        Route("/", dashboard, methods=["GET"]),
        Route("/admin/api/auth/options", auth_options, methods=["GET"]),
        Route("/admin/api/auth/login", auth_login, methods=["POST"]),
        Route("/admin/api/auth/logout", auth_logout, methods=["POST"]),
        Route("/admin/api/state", admin_state, methods=["GET"]),
        Route("/admin/api/cookbook", model_cookbook, methods=["GET"]),
        Route("/admin/api/metrics", admin_metrics, methods=["GET"]),
        Route("/admin/api/metrics/reset", reset_metrics, methods=["POST"]),
        Route("/admin/api/update/check", check_updates, methods=["GET"]),
        Route("/admin/api/update/install", install_update, methods=["POST"]),
        Route("/admin/api/config", update_config, methods=["PUT"]),
        Route("/admin/api/config/reload", reload_config, methods=["POST"]),
        Route("/admin/api/config/reset", reset_config, methods=["POST"]),
        Route("/admin/api/config/export", download_config, methods=["GET"]),
        Route("/admin/api/models/test", test_model, methods=["POST"]),
        Route("/admin/api/models/files", model_files, methods=["GET"]),
        Route("/admin/api/models/files/upload", upload_model_file, methods=["POST"]),
        Route(
            "/admin/api/models/detect-capabilities",
            detect_model_capabilities_route,
            methods=["POST"],
        ),
        Route("/admin/api/runtime", runtime_status, methods=["GET"]),
        Route("/admin/api/models/start", start_model, methods=["POST"]),
        Route("/admin/api/models/stop", stop_model, methods=["POST"]),
        Route("/admin/api/models/stop-all", stop_all_models, methods=["POST"]),
        Route("/admin/api/models/restart", restart_model, methods=["POST"]),
        Route("/health", health, methods=["GET"]),
        Route("/ready", readiness, methods=["GET"]),
        Route(f"{api_format.models_endpoint}", list_models, methods=["GET"]),
        Route(
            f"{api_format.base_path}/{{path:path}}",
            inference,
            methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
        ),
    ]
    inference_limiter = InferenceRequestLimiter(runtime_settings.max_inference_requests)
    app = RequestIDStarlette(
        routes=routes,
        lifespan=lifespan,
        middleware=[
            Middleware(SessionOriginMiddleware, cookie_name=SESSION_COOKIE_NAME),
            Middleware(AdminResponseHeadersMiddleware),
            Middleware(
                InferenceResponseHeadersMiddleware,
                api_base_path=api_format.base_path,
                models_endpoint=api_format.models_endpoint,
            ),
            Middleware(
                InferenceCapacityMiddleware,
                limiter=inference_limiter,
                api_base_path=api_format.base_path,
                models_endpoint=api_format.models_endpoint,
            ),
        ],
    )
    app.state.inference_limiter = inference_limiter
    app.state.login_attempt_limiter = LoginAttemptLimiter()
    app.state.settings = runtime_settings
    app.state.worker_count = runtime_settings.workers
    app.state.metrics = InferenceMetrics()
    app.state.admin_lock = asyncio.Lock()
    app.state.update_lock = asyncio.Lock()
    app.state.admin_config_path = (
        Path(admin_config_path) if admin_config_path is not None else resolve_admin_config_path()
    )
    app.state.api_format = api_format
    return app


# Uvicorn imports this object when launched by the console script.
app = create_app()
