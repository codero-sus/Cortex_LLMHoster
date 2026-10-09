from __future__ import annotations

import os
import sys

import uvicorn

from cortex_llmhoster import cli
from cortex_llmhoster.config import Settings


def test_cli_passes_resolved_listener_and_workers_to_uvicorn(tmp_path, monkeypatch) -> None:
    config = tmp_path / "cortex.toml"
    config.write_text(
        """
[server]
host = "127.0.0.1"
port = 9101
workers = 2

""",
        encoding="utf-8",
    )
    observed: dict[str, object] = {}
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: observed.update(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cortex-llmhoster",
            "--config",
            str(config),
            "--host",
            "0.0.0.0",
            "--port",
            "9123",
            "--workers",
            "3",
        ],
    )

    cli.main()

    assert observed["host"] == "0.0.0.0"
    assert observed["port"] == 9123
    assert observed["workers"] == 3


def test_cli_uses_default_port_8624(monkeypatch) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(cli.Settings, "load", lambda: Settings())
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: observed.update(kwargs))
    monkeypatch.setattr(sys, "argv", ["cortex-llmhoster"])

    cli.main()

    assert observed["port"] == 8624


def test_cli_accepts_api_config_override(monkeypatch) -> None:
    observed: dict[str, object] = {}
    api_config = "/tmp/cortex-api.json"
    previous_api_config = os.environ.pop("CORTEX_API_CONFIG", None)
    monkeypatch.setattr(cli.Settings, "load", lambda: Settings())
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: observed.update(kwargs))
    monkeypatch.setattr(sys, "argv", ["cortex-llmhoster", "--api-config", api_config])

    try:
        cli.main()
        assert os.environ["CORTEX_API_CONFIG"] == api_config
        assert observed["port"] == 8624
    finally:
        os.environ.pop("CORTEX_API_CONFIG", None)
        if previous_api_config is not None:
            os.environ["CORTEX_API_CONFIG"] = previous_api_config
