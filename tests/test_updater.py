from __future__ import annotations

import base64
import sys
from pathlib import Path

import httpx
import pytest

from cortex_llmhoster.updater import UPDATE_REF, UpdateError, check_for_updates, install_from_commit

COMMIT_SHA = "a" * 40


def update_transport(version: str) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.github.com"
        assert "authorization" not in request.headers
        if "/branches/" in request.url.path:
            return httpx.Response(
                200,
                json={"name": UPDATE_REF, "commit": {"sha": COMMIT_SHA}},
            )
        if request.url.path.endswith("/contents/pyproject.toml"):
            source = f'[project]\nversion = "{version}"\n'.encode()
            return httpx.Response(
                200,
                json={"content": base64.b64encode(source).decode("ascii")},
            )
        return httpx.Response(404, json={"message": "not found"})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_update_check_reads_pinned_branch_version_without_credentials() -> None:
    async with httpx.AsyncClient(transport=update_transport("0.2.0")) as client:
        update = await check_for_updates(client, "0.1.0")

    assert update.current_version == "0.1.0"
    assert update.latest_version == "0.2.0"
    assert update.source_ref == UPDATE_REF
    assert update.commit_sha == COMMIT_SHA
    assert update.update_available is True


@pytest.mark.asyncio
async def test_update_check_reports_same_version_as_current() -> None:
    async with httpx.AsyncClient(transport=update_transport("0.2.0")) as client:
        update = await check_for_updates(client, "0.2.0")

    assert update.update_available is False


@pytest.mark.asyncio
async def test_update_check_handles_unavailable_branch() -> None:
    async def not_found(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(not_found)) as client:
        with pytest.raises(UpdateError, match="branch or project file was not found"):
            await check_for_updates(client, "0.1.0")


@pytest.mark.asyncio
async def test_installer_uses_running_interpreter_and_immutable_archive(monkeypatch) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setenv("CORTEX_API_KEY", "dashboard-key")
    monkeypatch.setenv("CORTEX_ADMIN_PASSWORD", "dashboard-password")
    monkeypatch.setenv("CORTEX_CUSTOM_KEY", "custom-dashboard-key")

    class FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"Successfully installed cortex-llmhoster", b""

    async def fake_create_process(*args: object, **kwargs: object) -> FakeProcess:
        calls["args"] = args
        calls["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(
        "cortex_llmhoster.updater.asyncio.create_subprocess_exec", fake_create_process
    )
    output = await install_from_commit(COMMIT_SHA, api_key_env="CORTEX_CUSTOM_KEY")

    args = calls["args"]
    assert args[:3] == (sys.executable, "-m", "pip")
    assert args[3:6] == ("install", "--upgrade", "--no-input")
    assert args[-1] == (f"https://github.com/codero-sus/Cortex_LLMHoster/archive/{COMMIT_SHA}.zip")
    child_environment = calls["kwargs"]["env"]
    assert child_environment["PIP_DISABLE_PIP_VERSION_CHECK"] == "1"
    assert "PATH" in child_environment
    assert "CORTEX_API_KEY" not in child_environment
    assert "CORTEX_ADMIN_PASSWORD" not in child_environment
    assert "CORTEX_CUSTOM_KEY" not in child_environment
    assert "Successfully installed" in output


@pytest.mark.asyncio
async def test_installer_uses_selected_python_for_pip(monkeypatch) -> None:
    calls: dict[str, object] = {}
    selected_python = "/portable/python/bin/python3"

    class FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"update complete", b""

    async def fake_create_process(*args: object, **kwargs: object) -> FakeProcess:
        calls["args"] = args
        return FakeProcess()

    monkeypatch.setattr(
        "cortex_llmhoster.updater.resolve_python_executable", lambda: selected_python
    )
    monkeypatch.setattr(
        "cortex_llmhoster.updater.asyncio.create_subprocess_exec", fake_create_process
    )

    await install_from_commit(COMMIT_SHA)

    assert calls["args"][0] == selected_python


@pytest.mark.asyncio
async def test_installer_rejects_untrusted_commit_id_before_starting_pip() -> None:
    with pytest.raises(UpdateError, match="invalid update commit ID"):
        await install_from_commit("https://example.invalid/repo.zip")


def test_root_updaters_clone_the_pinned_branch_without_python_installs() -> None:
    project_root = Path(__file__).resolve().parents[1]
    expected_branch = "arena/b3072048-cortex-llmhoster"
    expected_url = "https://github.com/codero-sus/Cortex_LLMHoster.git"

    for filename in ("updater.sh", "updater.bat"):
        script = (project_root / filename).read_text(encoding="utf-8")
        clone_position = script.index("git clone --branch")
        backup_marker = 'find "$PROJECT_ROOT"' if filename.endswith(".sh") else "robocopy"
        assert script.index(backup_marker) < clone_position
        assert expected_branch in script
        assert expected_url in script
        assert "refs/heads/" in script
        assert "git merge --ff-only" in script
        assert "-m pip install" not in script
        assert "python_runtime" not in script
        for cache_dir in (".pytest_cache", "__pycache__", "coverage", ".cache"):
            assert cache_dir in script
