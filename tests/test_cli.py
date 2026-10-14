from __future__ import annotations

import os
import sys

import pytest
import uvicorn

from cortex_llmhoster import cli
from cortex_llmhoster.config import Settings


def test_cli_passes_resolved_listener_and_workers_to_uvicorn(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CORTEX_WORKERS", "2")
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
    assert os.environ["CORTEX_WORKERS"] == "3"


@pytest.mark.parametrize("flag", ["--no-api-auth", "--no-auth"])
def test_cli_no_api_auth_flag_runs_without_a_key(flag, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.python_runtime, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(cli, "reexec_with_selected_python", lambda: None)
    monkeypatch.setenv("CORTEX_API_AUTH_ENABLED", "true")
    monkeypatch.delenv("CORTEX_API_KEY", raising=False)
    monkeypatch.setenv("CORTEX_ADMIN_USERNAME", "local-admin")
    monkeypatch.setenv("CORTEX_ADMIN_PASSWORD", "local-password")
    monkeypatch.delenv("CORTEX_WORKERS", raising=False)
    observed: dict[str, object] = {}

    def load_settings() -> Settings:
        settings = Settings.from_mapping({}, environ=os.environ)
        observed["auth_enabled"] = settings.api_auth_enabled
        return settings

    monkeypatch.setattr(cli.Settings, "load", load_settings)
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: observed.update(kwargs))
    monkeypatch.setattr(sys, "argv", ["cortex-llmhoster", flag])

    cli.main()

    assert observed["auth_enabled"] is False
    assert os.environ["CORTEX_API_AUTH_ENABLED"] == "false"
    assert observed["port"] == 8624


def test_cli_uses_default_port_8624(monkeypatch) -> None:
    monkeypatch.delenv("CORTEX_WORKERS", raising=False)
    observed: dict[str, object] = {}
    monkeypatch.setattr(cli.Settings, "load", lambda: Settings())
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: observed.update(kwargs))
    monkeypatch.setattr(sys, "argv", ["cortex-llmhoster"])

    cli.main()

    assert observed["port"] == 8624


def test_cli_loads_project_dotenv_before_python_selection(tmp_path, monkeypatch) -> None:
    python_path = tmp_path / "portable python"
    python_path.write_text("placeholder", encoding="utf-8")
    (tmp_path / ".env").write_text(
        f"2PY2={python_path}\n"
        "CORTEX_ADMIN_USERNAME=test-admin\n"
        "CORTEX_ADMIN_PASSWORD=test-password\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli.python_runtime, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(cli, "reexec_with_selected_python", lambda: None)
    monkeypatch.setattr(cli.Settings, "load", lambda: Settings())
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["cortex-llmhoster"])
    monkeypatch.delenv("2PY2", raising=False)
    monkeypatch.delenv("CORTEX_ADMIN_USERNAME", raising=False)
    monkeypatch.delenv("CORTEX_ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("CORTEX_WORKERS", raising=False)

    try:
        cli.main()
        assert os.environ["2PY2"] == str(python_path)
        assert os.environ["CORTEX_ADMIN_USERNAME"] == "test-admin"
        assert os.environ["CORTEX_ADMIN_PASSWORD"] == "test-password"
    finally:
        os.environ.pop("2PY2", None)
        os.environ.pop("CORTEX_ADMIN_USERNAME", None)
        os.environ.pop("CORTEX_ADMIN_PASSWORD", None)


def test_cli_accepts_api_config_override(monkeypatch) -> None:
    monkeypatch.delenv("CORTEX_WORKERS", raising=False)
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
