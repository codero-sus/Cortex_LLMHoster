from __future__ import annotations

import sys

import uvicorn

from cortex_llmhoster import cli


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
