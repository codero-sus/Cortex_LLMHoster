from __future__ import annotations

import hashlib
import io
import json
import urllib.error
import zipfile
from pathlib import Path
from typing import Any, Self

import pytest

import cortex_llmhoster.native_installer as installer
from cortex_llmhoster.native_installer import (
    RuntimeIntegrityError,
    RuntimeReleaseUnavailableError,
    auto_install_enabled,
    install_latest_runtime_binary,
    release_target,
)


class MemoryResponse:
    def __init__(self, content: bytes, url: str) -> None:
        self._stream = io.BytesIO(content)
        self.headers = {"Content-Length": str(len(content))}
        self.status = 200
        self._url = url

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def geturl(self) -> str:
        return self._url

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def close(self) -> None:
        self._stream.close()


def release_fixture(
    *,
    target: str = "linux-x86_64",
    corrupt_digest: bool = False,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    suffix = ".exe" if target.startswith("windows-") else ""
    binary_names = [
        f"cortex-gguf-host{suffix}",
        f"llama-server{suffix}",
        f"llama-quantize{suffix}",
    ]
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for binary_name in binary_names:
            bundle.writestr(binary_name, f"test executable: {binary_name}")
        if target.startswith("linux-"):
            bundle.writestr("libgomp.so.1", b"test OpenMP runtime")
        elif target.startswith("windows-"):
            bundle.writestr("vcomp140.dll", b"test MSVC OpenMP runtime")
        bundle.writestr("unrequested.txt", "not extracted")
    archive_content = archive_buffer.getvalue()
    archive_name = f"cortex-cpp-{target}.zip"
    archive_url = (
        "https://github.com/codero-sus/Cortex_LLMHoster/releases/download/"
        f"cortex-cpp/{archive_name}"
    )
    digest = hashlib.sha256(archive_content).hexdigest()
    if corrupt_digest:
        digest = "0" * 64
    manifest = {
        "schema_version": 1,
        "release_tag": "cortex-cpp",
        "assets": {
            target: {
                "name": archive_name,
                "size": len(archive_content),
                "sha256": digest,
                "binaries": binary_names,
                "runtime_libraries": (
                    ["libgomp.so.1"]
                    if target.startswith("linux-")
                    else ["vcomp140.dll"]
                    if target.startswith("windows-")
                    else []
                ),
            }
        },
    }
    manifest_content = json.dumps(manifest).encode()
    manifest_url = (
        "https://github.com/codero-sus/Cortex_LLMHoster/releases/download/"
        "cortex-cpp/cortex-cpp-manifest.json"
    )
    release = {
        "tag_name": "cortex-cpp",
        "assets": [
            {
                "name": "cortex-cpp-manifest.json",
                "size": len(manifest_content),
                "browser_download_url": manifest_url,
            },
            {
                "name": archive_name,
                "size": len(archive_content),
                "browser_download_url": archive_url,
            },
        ],
    }
    responses = {
        installer.RELEASE_API_URL: json.dumps(release).encode(),
        manifest_url: manifest_content,
        archive_url: archive_content,
    }
    return release, responses


def fixture_opener(responses: dict[str, bytes]):
    def open_request(request: Any, *, timeout: int) -> MemoryResponse:
        del timeout
        url = request.full_url
        if url not in responses:
            raise AssertionError(f"Unexpected URL requested: {url}")
        return MemoryResponse(responses[url], url)

    return open_request


def test_release_target_normalizes_supported_platform_names() -> None:
    assert release_target("Linux", "AMD64") == "linux-x86_64"
    assert release_target("Windows", "x86_64") == "windows-x86_64"
    assert release_target("Darwin", "arm64") == "macos-arm64"
    assert release_target("FreeBSD", "x86_64") is None
    assert release_target("Linux", "armv7l") is None
    assert release_target("Linux", "aarch64") is None


def test_auto_install_is_enabled_by_default_and_can_be_disabled(monkeypatch) -> None:
    monkeypatch.delenv("CORTEX_AUTO_INSTALL_RUNTIMES", raising=False)
    assert auto_install_enabled()
    monkeypatch.setenv("CORTEX_AUTO_INSTALL_RUNTIMES", "false")
    assert not auto_install_enabled()
    monkeypatch.setenv("CORTEX_AUTO_INSTALL_RUNTIMES", "yes")
    assert auto_install_enabled()


def test_installs_verified_runtime_bundle_and_marks_executables_ready(tmp_path: Path) -> None:
    _, responses = release_fixture()
    binary = install_latest_runtime_binary(
        "cortex-gguf-host",
        cache_dir=tmp_path,
        target="linux-x86_64",
        opener=fixture_opener(responses),
    )

    assert binary.name == "cortex-gguf-host"
    assert binary.read_text() == "test executable: cortex-gguf-host"
    assert binary.stat().st_mode & 0o111
    assert (binary.parent / "llama-server").read_text() == "test executable: llama-server"
    assert (binary.parent / "llama-quantize").read_text() == "test executable: llama-quantize"
    runtime_library = binary.parent / "libgomp.so.1"
    assert runtime_library.read_bytes() == b"test OpenMP runtime"
    assert runtime_library.stat().st_mode & 0o111
    assert {path.name for path in binary.parent.iterdir()} == {
        "cortex-gguf-host",
        "llama-server",
        "llama-quantize",
        "libgomp.so.1",
    }


def test_windows_bundle_installs_openmp_runtime_beside_executables(tmp_path: Path) -> None:
    _, responses = release_fixture(target="windows-x86_64")
    server = install_latest_runtime_binary(
        "llama-server",
        cache_dir=tmp_path,
        target="windows-x86_64",
        opener=fixture_opener(responses),
    )

    assert server.name == "llama-server.exe"
    assert (server.parent / "cortex-gguf-host.exe").is_file()
    assert (server.parent / "llama-quantize.exe").is_file()
    assert (server.parent / "vcomp140.dll").read_bytes() == b"test MSVC OpenMP runtime"


def test_macos_bundle_installs_with_system_runtime_dependencies_only(tmp_path: Path) -> None:
    _, responses = release_fixture(target="macos-arm64")
    host = install_latest_runtime_binary(
        "cortex-gguf-host",
        cache_dir=tmp_path,
        target="macos-arm64",
        opener=fixture_opener(responses),
    )

    assert host.name == "cortex-gguf-host"
    assert host.stat().st_mode & 0o111
    assert not (host.parent / "libgomp.so.1").exists()
    assert not (host.parent / "vcomp140.dll").exists()


def test_uses_cached_binary_if_the_github_release_is_temporarily_unreachable(
    tmp_path: Path,
) -> None:
    _, responses = release_fixture()
    first = install_latest_runtime_binary(
        "cortex-gguf-host",
        cache_dir=tmp_path,
        target="linux-x86_64",
        opener=fixture_opener(responses),
    )

    def unavailable(*_: object, **__: object) -> None:
        raise urllib.error.URLError("offline")

    cached = install_latest_runtime_binary(
        "cortex-gguf-host",
        cache_dir=tmp_path,
        target="linux-x86_64",
        opener=unavailable,
    )
    assert cached == first


def test_rejects_an_archive_that_does_not_match_the_manifest_hash(tmp_path: Path) -> None:
    _, responses = release_fixture(corrupt_digest=True)

    with pytest.raises(RuntimeIntegrityError, match="SHA-256/size check"):
        install_latest_runtime_binary(
            "cortex-gguf-host",
            cache_dir=tmp_path,
            target="linux-x86_64",
            opener=fixture_opener(responses),
        )

    assert not (tmp_path / "cortex-cpp" / "linux-x86_64").exists()


def test_uses_last_verified_binary_if_a_rolling_release_is_incomplete(tmp_path: Path) -> None:
    _, good_responses = release_fixture()
    verified = install_latest_runtime_binary(
        "cortex-gguf-host",
        cache_dir=tmp_path,
        target="linux-x86_64",
        opener=fixture_opener(good_responses),
    )
    _, interrupted_responses = release_fixture(corrupt_digest=True)

    fallback = install_latest_runtime_binary(
        "cortex-gguf-host",
        cache_dir=tmp_path,
        target="linux-x86_64",
        opener=fixture_opener(interrupted_responses),
    )

    assert fallback == verified


def test_rejects_a_release_with_a_different_tag(tmp_path: Path) -> None:
    release, responses = release_fixture()
    release["tag_name"] = "v1.0.0"
    responses[installer.RELEASE_API_URL] = json.dumps(release).encode()

    with pytest.raises(RuntimeIntegrityError, match="required 'cortex-cpp' tag"):
        install_latest_runtime_binary(
            "cortex-gguf-host",
            cache_dir=tmp_path,
            target="linux-x86_64",
            opener=fixture_opener(responses),
        )


def test_missing_platform_bundle_fails_clearly(tmp_path: Path) -> None:
    _, responses = release_fixture(target="windows-x86_64")

    with pytest.raises(RuntimeReleaseUnavailableError, match="does not contain a linux-x86_64"):
        install_latest_runtime_binary(
            "cortex-gguf-host",
            cache_dir=tmp_path,
            target="linux-x86_64",
            opener=fixture_opener(responses),
        )


def test_rejects_untrusted_github_asset_redirect(tmp_path: Path) -> None:
    _, responses = release_fixture()

    class RedirectingOpener:
        def __call__(self, request: Any, *, timeout: int) -> MemoryResponse:
            del timeout
            if request.full_url == installer.RELEASE_API_URL:
                response = MemoryResponse(responses[request.full_url], request.full_url)
                return response
            if request.full_url.endswith("cortex-cpp-manifest.json"):
                return MemoryResponse(b"{}", "https://evil.example/manifest.json")
            raise AssertionError(f"Unexpected URL requested: {request.full_url}")

    with pytest.raises(RuntimeIntegrityError, match="untrusted URL"):
        install_latest_runtime_binary(
            "cortex-gguf-host",
            cache_dir=tmp_path,
            target="linux-x86_64",
            opener=RedirectingOpener(),
        )
