"""Command-line entry point."""

from __future__ import annotations

import argparse
import importlib.util
import os

from .config import LOCAL_RUNTIMES, ConfigurationError, Settings


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cortex-llmhoster",
        description="Local text and multimodal model hosting with an OpenAI-compatible API.",
    )
    parser.add_argument("--config", help="TOML config file (default: ./cortex.toml)")
    parser.add_argument("--api-config", help="OpenAI-compatible API route config JSON.")
    parser.add_argument("--host", help="Listen address (overrides TOML/environment).")
    parser.add_argument("--port", type=int, help="Listen port (overrides TOML/environment).")
    parser.add_argument(
        "--workers",
        type=int,
        help="Uvicorn workers; use 1 with local models to avoid duplicate model loads.",
    )
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        help="Server log level (overrides TOML/environment).",
    )
    access_logs = parser.add_mutually_exclusive_group()
    access_logs.add_argument(
        "--access-log",
        dest="access_log",
        action="store_true",
        help="Enable per-request access logs.",
    )
    access_logs.add_argument(
        "--no-access-log",
        dest="access_log",
        action="store_false",
        help="Disable per-request access logs for lower overhead.",
    )
    parser.set_defaults(access_log=None)
    args = parser.parse_args()

    if args.config:
        os.environ["CORTEX_CONFIG"] = args.config
    if args.api_config:
        os.environ["CORTEX_API_CONFIG"] = args.api_config

    try:
        settings = Settings.load()
    except ConfigurationError as exc:
        parser.error(str(exc))

    host = args.host or settings.host
    port = args.port if args.port is not None else settings.port
    workers = args.workers if args.workers is not None else settings.workers
    if port < 1 or port > 65535:
        parser.error("--port must be between 1 and 65535.")
    if workers < 1:
        parser.error("--workers must be at least 1.")
    if workers > 1 and any(model.runtime in LOCAL_RUNTIMES for model in settings.models.values()):
        parser.error(
            "Locally managed model runtimes require --workers 1 so model weights are not "
            "loaded once per worker. Use the runtime's own concurrency settings."
        )
    # Uvicorn worker processes reload the app in child interpreters. Keep the
    # resolved CLI override visible so process-local features can detect the
    # actual deployment worker count, not only the TOML value.
    os.environ["CORTEX_WORKERS"] = str(workers)

    # Prefer uvloop + httptools where the platform provides them, while retaining
    # a dependency-light fallback for development and Windows.
    loop = "uvloop" if importlib.util.find_spec("uvloop") else "asyncio"
    protocol = "httptools" if importlib.util.find_spec("httptools") else "h11"

    import uvicorn

    uvicorn.run(
        "cortex_llmhoster.app:app",
        host=host,
        port=port,
        workers=workers,
        loop=loop,
        http=protocol,
        log_level=args.log_level or settings.log_level,
        access_log=settings.access_log if args.access_log is None else args.access_log,
        server_header=False,
        date_header=False,
    )
