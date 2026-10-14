"""Authenticated model-file discovery and task-suggestion dashboard handlers."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from starlette.requests import Request
from starlette.responses import Response

from .config import Settings
from .model_discovery import detect_model_capabilities, discover_local_model_files
from .responses import OrjsonResponse
from .responses import error_response as _error


async def model_files(request: Request) -> Response:
    """Return known local GGUF files for the model editor's file picker."""

    from .dashboard import _admin_auth

    settings: Settings = request.app.state.settings
    auth = _admin_auth(request, settings)
    if auth is not None:
        return auth
    configured_paths = [model.model_path for model in settings.models.values()]
    result = await asyncio.to_thread(discover_local_model_files, configured_paths)
    return OrjsonResponse(result)


async def detect_model_capabilities_route(request: Request) -> Response:
    """Suggest task checkboxes based on a selected local model and its metadata."""

    from .dashboard import _admin_auth, _read_json

    auth = _admin_auth(request, request.app.state.settings)
    if auth is not None:
        return auth
    payload, body_error = await _read_json(request, limit=64 * 1024)
    if body_error is not None:
        return body_error
    if not isinstance(payload, Mapping):
        return _error("Capability detection payload must be an object.", status_code=400)
    model_path = payload.get("model_path")
    model_id = payload.get("model_id", "")
    mmproj_path = payload.get("mmproj_path")
    if not isinstance(model_path, str) or not model_path.strip():
        return _error("A model_path is required.", status_code=400)
    if len(model_path) > 4096:
        return _error("The model_path is too long.", status_code=400)
    if not isinstance(model_id, str) or not isinstance(mmproj_path, (str, type(None))):
        return _error("Model identity and projector path must be strings.", status_code=400)
    if len(model_id) > 512 or (mmproj_path is not None and len(mmproj_path) > 4096):
        return _error("The model identity or projector path is too long.", status_code=400)
    try:
        result: dict[str, Any] = await asyncio.to_thread(
            detect_model_capabilities,
            model_path,
            model_id=model_id[:512],
            mmproj_path=mmproj_path[:4096] if mmproj_path else None,
        )
    except (OSError, ValueError) as exc:
        return _error(str(exc), status_code=422, code="model_detection_failed")
    return OrjsonResponse(result)
