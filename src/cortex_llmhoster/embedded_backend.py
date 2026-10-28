"""Cortex-owned OpenAI-compatible GGUF backend using llama-cpp-python in-process."""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Iterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson
from starlette.applications import Starlette
from starlette.requests import ClientDisconnect, Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from .http_utils import content_length_from_headers
from .responses import OrjsonResponse
from .responses import error_response as _error

logger = logging.getLogger("cortex_llmhoster.embedded_backend")
_MAX_JSON_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class BackendOptions:
    """Settings passed from Cortex's supervised process to the embedded engine."""

    model_path: str
    model_alias: str
    api_base_path: str = "/v1"
    health_path: str = "/health"
    context_size: int = 2048
    threads: int = 0
    threads_batch: int = 0
    gpu_layers: int = 0
    batch_size: int = 256
    use_mmap: bool = True
    mlock: bool = False
    embedding: bool = False


def _next_item(iterator: Iterator[Any]) -> tuple[bool, Any]:
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


async def _read_json(request: Request) -> tuple[dict[str, Any] | None, Response | None]:
    try:
        declared_length = content_length_from_headers(request.headers)
    except ValueError:
        return None, _error("Invalid Content-Length header.", status_code=400)
    if declared_length is not None and declared_length > _MAX_JSON_BYTES:
        return None, _error("JSON request body is too large.", status_code=413)

    body = bytearray()
    try:
        async for chunk in request.stream():
            if len(body) + len(chunk) > _MAX_JSON_BYTES:
                return None, _error("JSON request body is too large.", status_code=413)
            body.extend(chunk)
    except ClientDisconnect:
        return None, _error(
            "The request body was interrupted.",
            status_code=400,
            code="request_body_interrupted",
        )
    try:
        payload = orjson.loads(body)
    except orjson.JSONDecodeError:
        return None, _error("Request body must be valid JSON.", status_code=400)
    if not isinstance(payload, dict):
        return None, _error("JSON request body must be an object.", status_code=400)
    return payload, None


def _set_model_name(result: object, model_alias: str) -> object:
    if isinstance(result, dict):
        response = dict(result)
        response["model"] = model_alias
        return response
    return result


def create_backend_app(
    options: BackendOptions,
    *,
    llama_factory=None,
) -> Starlette:
    """Build the private local API app; the optional factory keeps tests hardware-free."""

    @asynccontextmanager
    async def lifespan(app: Starlette):
        factory = llama_factory
        if factory is None:
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise RuntimeError(
                    "Cortex's built-in GGUF backend needs the optional llama-cpp-python "
                    "dependency. Install Cortex with the [embedded] extra."
                ) from exc
            factory = Llama

        model_path = Path(options.model_path).expanduser().resolve(strict=True)
        if not model_path.is_file() or model_path.suffix.lower() != ".gguf":
            raise RuntimeError("The built-in backend requires an existing GGUF model file.")
        app.state.llama = await asyncio.to_thread(
            factory,
            model_path=str(model_path),
            n_ctx=options.context_size,
            n_threads=options.threads or None,
            n_threads_batch=options.threads_batch or options.threads or None,
            n_gpu_layers=options.gpu_layers,
            n_batch=options.batch_size,
            use_mmap=options.use_mmap,
            use_mlock=options.mlock,
            embedding=options.embedding,
            verbose=False,
        )
        app.state.inference_lock = asyncio.Lock()
        try:
            yield
        finally:
            engine = app.state.llama
            close = getattr(engine, "close", None)
            if callable(close):
                await asyncio.to_thread(close)

    async def health(_: Request) -> Response:
        return OrjsonResponse({"status": "ok"}, headers={"cache-control": "no-store"})

    def make_inference_handler(operation: str):
        async def infer(request: Request) -> Response:
            payload, body_error = await _read_json(request)
            if body_error is not None:
                return body_error
            if payload is None:
                return _error(
                    "The request body could not be read.",
                    status_code=500,
                    code="request_body_read_failed",
                )
            if operation == "embeddings" and not options.embedding:
                return _error(
                    "Embeddings are disabled for this embedded model.",
                    status_code=422,
                    code="unsupported_model_capability",
                )

            streaming = payload.get("stream", False)
            if not isinstance(streaming, bool):
                return _error("The stream field must be a boolean.", status_code=400)
            if operation == "embeddings" and streaming:
                return _error("Embedding responses cannot be streamed.", status_code=400)
            arguments = dict(payload)
            arguments.pop("model", None)
            arguments.pop("stream", None)
            if "max_completion_tokens" in arguments and "max_tokens" not in arguments:
                arguments["max_tokens"] = arguments.pop("max_completion_tokens")
            method_name = {
                "chat": "create_chat_completion",
                "completion": "create_completion",
                "embeddings": "create_embedding",
            }[operation]
            method = getattr(request.app.state.llama, method_name, None)
            if not callable(method):
                return _error(
                    f"This installed llama-cpp-python version does not support {operation}.",
                    status_code=501,
                    code="backend_operation_unavailable",
                )

            if streaming:

                async def stream_response():
                    async with request.app.state.inference_lock:
                        try:
                            iterator = await asyncio.to_thread(method, stream=True, **arguments)
                            while True:
                                has_item, item = await asyncio.to_thread(_next_item, iterator)
                                if not has_item:
                                    break
                                item = _set_model_name(item, options.model_alias)
                                yield b"data: " + orjson.dumps(item) + b"\n\n"
                            yield b"data: [DONE]\n\n"
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            logger.exception(
                                "Embedded GGUF stream failed for %s", options.model_alias
                            )
                            error = {
                                "error": {
                                    "message": "The embedded model stream failed.",
                                    "type": "server_error",
                                    "code": "embedded_stream_failed",
                                }
                            }
                            yield b"data: " + orjson.dumps(error) + b"\n\n"
                            yield b"data: [DONE]\n\n"

                return StreamingResponse(
                    stream_response(),
                    media_type="text/event-stream",
                    headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
                )

            try:
                async with request.app.state.inference_lock:
                    result = await asyncio.to_thread(method, **arguments)
            except asyncio.CancelledError:
                raise
            except (TypeError, ValueError):
                logger.info("Rejected invalid embedded %s request", operation)
                return _error(
                    f"Invalid {operation} request parameters.",
                    status_code=400,
                    error_type="invalid_request_error",
                    code="invalid_request",
                )
            except Exception:
                logger.exception("Embedded GGUF inference failed for %s", options.model_alias)
                return _error(
                    "The embedded model could not complete the request.",
                    status_code=500,
                    error_type="server_error",
                    code="embedded_inference_failed",
                )
            if not isinstance(result, Mapping):
                return _error(
                    "The embedded backend returned an invalid response.",
                    status_code=502,
                    code="invalid_backend_response",
                )
            return OrjsonResponse(_set_model_name(result, options.model_alias))

        return infer

    base_path = options.api_base_path.rstrip("/")
    routes = [
        Route(options.health_path, health, methods=["GET"]),
        Route(f"{base_path}/chat/completions", make_inference_handler("chat"), methods=["POST"]),
        Route(f"{base_path}/completions", make_inference_handler("completion"), methods=["POST"]),
        Route(f"{base_path}/embeddings", make_inference_handler("embeddings"), methods=["POST"]),
    ]
    return Starlette(routes=routes, lifespan=lifespan)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cortex's private embedded GGUF backend.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--host", choices=("127.0.0.1",), default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--api-base-path", default="/v1")
    parser.add_argument("--health-path", default="/health")
    parser.add_argument("--context-size", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--threads-batch", type=int, default=0)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-mmap", action="store_true")
    parser.add_argument("--mlock", action="store_true")
    parser.add_argument("--embedding", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")
    if args.context_size < 128 or args.batch_size < 1:
        parser.error("Context size and batch size must be positive.")

    options = BackendOptions(
        model_path=args.model_path,
        model_alias=args.model_alias,
        api_base_path=args.api_base_path,
        health_path=args.health_path,
        context_size=args.context_size,
        threads=args.threads,
        threads_batch=args.threads_batch,
        gpu_layers=args.gpu_layers,
        batch_size=args.batch_size,
        use_mmap=not args.no_mmap,
        mlock=args.mlock,
        embedding=args.embedding,
    )
    from uvicorn import run

    run(
        create_backend_app(options),
        host=args.host,
        port=args.port,
        workers=1,
        access_log=False,
        log_level="warning",
        server_header=False,
        date_header=False,
    )


if __name__ == "__main__":
    main()
