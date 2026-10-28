"""Authenticated local model-file selection, upload, and task-suggestion handlers."""

from __future__ import annotations

import asyncio
import errno
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from starlette.requests import ClientDisconnect, Request
from starlette.responses import Response

from .config import Settings
from .http_utils import content_length_from_headers
from .model_discovery import detect_model_capabilities, discover_local_model_files
from .responses import OrjsonResponse
from .responses import error_response as _error

_UPLOAD_WRITE_CHUNK_BYTES = 8 * 1024 * 1024
_GGUF_MAGIC = b"GGUF"


class _UploadRejected(Exception):
    """A validation or storage condition that should become a safe HTTP response."""

    def __init__(self, message: str, *, status_code: int, code: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


def _safe_model_filename(value: str) -> str | None:
    """Accept one plain GGUF filename, never a client-supplied filesystem path."""

    if (
        not value
        or value in {".", ".."}
        or value != value.strip()
        or any(character in value for character in '<>:/\\|?*"')
        or any(ord(character) < 32 for character in value)
        or value.endswith((".", " "))
        or Path(value).suffix.casefold() != ".gguf"
    ):
        return None
    try:
        if len(value.encode("utf-8", errors="strict")) > 240:
            return None
    except UnicodeEncodeError:
        return None
    if value.split(".", 1)[0].rstrip(" .").upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{number}" for number in range(1, 10)),
        *(f"LPT{number}" for number in range(1, 10)),
    }:
        return None
    return value


def _writable_model_directory() -> Path:
    """Create and return the first writable configured/default model directory."""

    configured = os.environ.get("CORTEX_MODELS_DIR", "")
    candidates = [
        Path(value.strip()).expanduser() for value in configured.split(os.pathsep) if value.strip()
    ]
    candidates.extend((Path.home() / "models", Path.cwd() / "models"))
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            root = candidate.resolve(strict=True)
            if root in seen or not root.is_dir():
                continue
            seen.add(root)
            descriptor, probe = tempfile.mkstemp(prefix=".cortex-write-test-", dir=root)
            os.close(descriptor)
            Path(probe).unlink()
            return root
        except (OSError, RuntimeError):
            continue
    raise OSError(
        "No writable model folder is available. Set CORTEX_MODELS_DIR to a writable folder."
    )


def _publish_uploaded_model(temp_path: Path, directory: Path, filename: str) -> Path:
    """Atomically publish without overwriting an existing model with the same name."""

    original = Path(filename)
    stem, suffix = original.stem, original.suffix
    unsupported_links = {
        errno.EACCES,
        errno.EPERM,
        errno.ENOSYS,
        getattr(errno, "ENOTSUP", errno.EPERM),
        getattr(errno, "EOPNOTSUPP", errno.EPERM),
    }
    for index in range(10_000):
        name = filename if index == 0 else f"{stem} ({index + 1}){suffix}"
        destination = directory / name
        try:
            os.link(temp_path, destination)
        except FileExistsError:
            continue
        except OSError as exc:
            if exc.errno not in unsupported_links:
                raise
            # Some Windows volumes (for example, exFAT) do not support hard
            # links. Reserve the name exclusively, then atomically replace our
            # zero-byte reservation with the completed temporary file.
            try:
                descriptor = os.open(
                    destination,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError:
                continue
            os.close(descriptor)
            try:
                os.replace(temp_path, destination)
            except BaseException:
                destination.unlink(missing_ok=True)
                raise
            return destination
        else:
            temp_path.unlink()
            return destination
    raise OSError("Could not choose a unique filename for the uploaded model.")


def _flush_and_sync(handle) -> None:
    handle.flush()
    os.fsync(handle.fileno())


async def model_files(request: Request) -> Response:
    """Return known local GGUF files for the model editor's existing-file picker."""

    from .dashboard import _admin_auth

    settings: Settings = request.app.state.settings
    auth = _admin_auth(request, settings)
    if auth is not None:
        return auth
    configured_paths = [model.model_path for model in settings.models.values()]
    result = await asyncio.to_thread(discover_local_model_files, configured_paths)
    return OrjsonResponse(result)


async def upload_model_file(request: Request) -> Response:
    """Stream a browser-selected GGUF to local model storage with bounded memory."""

    from .dashboard import _admin_auth

    auth = _admin_auth(request, request.app.state.settings)
    if auth is not None:
        return auth

    filename = _safe_model_filename(request.query_params.get("filename", ""))
    if filename is None:
        return _error(
            "Choose a file with a valid .gguf filename.",
            status_code=400,
            code="invalid_model_filename",
        )
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != "application/octet-stream":
        return _error(
            "Model-file uploads must use application/octet-stream.",
            status_code=415,
            code="invalid_model_upload_type",
        )

    try:
        expected_length = content_length_from_headers(request.headers)
    except ValueError:
        return _error(
            "Invalid Content-Length header.", status_code=400, code="invalid_content_length"
        )
    if expected_length is not None and expected_length < len(_GGUF_MAGIC):
        return _error("The selected file is too small to be a GGUF model.", status_code=422)

    try:
        directory = await asyncio.to_thread(_writable_model_directory)
        available_bytes = await asyncio.to_thread(shutil.disk_usage, directory)
    except OSError as exc:
        return _error(
            f"Could not prepare local model storage: {exc}",
            status_code=500,
            code="model_storage_unavailable",
        )
    if expected_length is not None and expected_length > available_bytes.free:
        return _error(
            f"The model needs {expected_length} bytes, but the model drive has only "
            f"{available_bytes.free} bytes free.",
            status_code=507,
            code="insufficient_model_storage",
        )

    temp_path: Path | None = None
    target = None
    received = 0
    header = bytearray()
    buffer = bytearray()
    try:
        descriptor, temp_name = await asyncio.to_thread(
            tempfile.mkstemp,
            prefix=".cortex-model-upload-",
            suffix=".partial",
            dir=directory,
        )
        temp_path = Path(temp_name)
        target = os.fdopen(descriptor, "wb")

        async for chunk in request.stream():
            if not chunk:
                continue
            received += len(chunk)
            if received > available_bytes.free:
                raise _UploadRejected(
                    "The model drive ran out of free space during upload.",
                    status_code=507,
                    code="insufficient_model_storage",
                )
            if expected_length is not None and received > expected_length:
                raise _UploadRejected(
                    "The uploaded data is longer than its Content-Length.",
                    status_code=400,
                    code="invalid_model_upload_length",
                )
            if len(header) < len(_GGUF_MAGIC):
                header.extend(chunk[: len(_GGUF_MAGIC) - len(header)])
                if len(header) == len(_GGUF_MAGIC) and bytes(header) != _GGUF_MAGIC:
                    raise _UploadRejected(
                        "The selected file does not have a GGUF header.",
                        status_code=422,
                        code="invalid_gguf_file",
                    )
            # ASGI servers may yield very large chunks; do not duplicate one
            # into the bounded write buffer before flushing it to disk.
            chunk_view = memoryview(chunk)
            offset = 0
            while offset < len(chunk_view):
                remaining = len(chunk_view) - offset
                capacity = _UPLOAD_WRITE_CHUNK_BYTES - len(buffer)
                if buffer or remaining < _UPLOAD_WRITE_CHUNK_BYTES:
                    count = min(capacity, remaining)
                    buffer.extend(chunk_view[offset : offset + count])
                    offset += count
                    if len(buffer) == _UPLOAD_WRITE_CHUNK_BYTES:
                        await asyncio.to_thread(target.write, buffer)
                        buffer.clear()
                else:
                    end = offset + _UPLOAD_WRITE_CHUNK_BYTES
                    await asyncio.to_thread(target.write, chunk_view[offset:end])
                    offset = end

        if len(header) < len(_GGUF_MAGIC):
            raise _UploadRejected(
                "The selected file is too small to be a GGUF model.",
                status_code=422,
                code="invalid_gguf_file",
            )
        if expected_length is not None and received != expected_length:
            raise _UploadRejected(
                "The model upload ended before the full file was received.",
                status_code=400,
                code="incomplete_model_upload",
            )
        if buffer:
            await asyncio.to_thread(target.write, buffer)
            buffer.clear()
        await asyncio.to_thread(_flush_and_sync, target)
        await asyncio.to_thread(target.close)
        target = None
        destination = await asyncio.to_thread(
            _publish_uploaded_model, temp_path, directory, filename
        )
        temp_path = None
        return OrjsonResponse(
            {
                "path": str(destination.resolve(strict=True)),
                "name": destination.name,
                "size_bytes": received,
            }
        )
    except _UploadRejected as exc:
        return _error(str(exc), status_code=exc.status_code, code=exc.code)
    except ClientDisconnect:
        return _error(
            "The model upload was interrupted. No partial model was kept.",
            status_code=400,
            code="model_upload_interrupted",
        )
    except OSError as exc:
        return _error(
            f"Could not store the model file: {exc}",
            status_code=507,
            code="model_storage_write_failed",
        )
    finally:
        if target is not None:
            try:
                await asyncio.to_thread(target.close)
            except OSError:
                pass
        if temp_path is not None:
            try:
                await asyncio.to_thread(temp_path.unlink, missing_ok=True)
            except OSError:
                pass


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
