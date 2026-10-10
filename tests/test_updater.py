from __future__ import annotations

import base64
import sys

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
    output = await install_from_commit(COMMIT_SHA)

    args = calls["args"]
    assert args[:3] == (sys.executable, "-m", "pip")
    assert args[3:6] == ("install", "--upgrade", "--no-input")
    assert args[-1] == (f"https://github.com/codero-sus/Cortex_LLMHoster/archive/{COMMIT_SHA}.zip")
    assert calls["kwargs"]["env"]["PIP_DISABLE_PIP_VERSION_CHECK"] == "1"
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
