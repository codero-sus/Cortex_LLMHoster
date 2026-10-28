"""Install the official, tag-pinned native runtime bundles on demand.

Downloads are limited to assets published by this repository's ``cortex-cpp``
GitHub release. The release manifest records a SHA-256 for each platform bundle;
only the declared runtime executables and approved support libraries are
extracted into the per-user cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import stat
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

RELEASE_TAG = "cortex-cpp"
RELEASE_REPOSITORY = "codero-sus/Cortex_LLMHoster"
RELEASE_API_URL = f"https://api.github.com/repos/{RELEASE_REPOSITORY}/releases/tags/{RELEASE_TAG}"
MANIFEST_ASSET_NAME = "cortex-cpp-manifest.json"
SUPPORTED_RELEASE_TARGETS = frozenset(
    {"linux-x86_64", "windows-x86_64", "macos-x86_64", "macos-arm64"}
)
_SUPPORTED_BINARIES = frozenset({"cortex-gguf-host", "llama-server", "llama-quantize"})
_ALLOWED_RUNTIME_LIBRARIES = frozenset({"libgomp.so.1", "vcomp140.dll"})
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
_MAX_BINARY_BYTES = 512 * 1024 * 1024
_MAX_RUNTIME_LIBRARY_BYTES = 128 * 1024 * 1024
_HTTP_TIMEOUT_SECONDS = 30
_RELEASE_CACHE_TTL_SECONDS = 300
_GITHUB_ASSET_HOSTS = frozenset(
    {
        "github.com",
        "release-assets.githubusercontent.com",
        "objects.githubusercontent.com",
    }
)
_RELEASE_API_HOSTS = frozenset({"api.github.com"})


class RuntimeInstallError(RuntimeError):
    """Base error for release lookup or native runtime installation failures."""


class RuntimeReleaseUnavailableError(RuntimeInstallError):
    """The rolling release or a compatible platform bundle is unavailable."""


class RuntimeIntegrityError(RuntimeInstallError):
    """The release manifest or downloaded bundle did not pass validation."""


UrlOpener = Callable[..., Any]

_release_cache_lock = threading.Lock()
_release_cache: tuple[float, dict[str, Any], dict[str, dict[str, Any]]] | None = None


def release_target(system: str | None = None, machine: str | None = None) -> str | None:
    """Return the release asset key for a supported OS/CPU pair, if any."""

    system_name = (system or platform.system()).strip().casefold()
    machine_name = (machine or platform.machine()).strip().casefold()
    if machine_name in {"amd64", "x64"}:
        architecture = "x86_64"
    elif machine_name in {"aarch64", "arm64"}:
        architecture = "arm64"
    elif machine_name in {"x86_64", "x86-64"}:
        architecture = "x86_64"
    else:
        return None

    if system_name == "linux":
        operating_system = "linux"
    elif system_name in {"windows", "win32"}:
        operating_system = "windows"
    elif system_name in {"darwin", "macos", "mac os x"}:
        operating_system = "macos"
    else:
        return None

    target = f"{operating_system}-{architecture}"
    return target if target in SUPPORTED_RELEASE_TARGETS else None


def auto_install_enabled() -> bool:
    """Return whether runtime binaries may be fetched on first use (default on)."""

    value = os.environ.get("CORTEX_AUTO_INSTALL_RUNTIMES", "true").strip().casefold()
    return value not in {"0", "false", "no", "off", "disabled"}


def runtime_cache_dir() -> Path:
    """Return the writable, per-user root used for downloaded runtime bundles."""

    override = os.environ.get("CORTEX_RUNTIME_BIN_DIR")
    if override:
        return Path(override).expanduser()

    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif platform.system() == "Darwin":
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / "Library" / "Caches"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "cortex-llmhoster" / "runtimes"


def _asset_filename(binary_name: str, target: str) -> str:
    if binary_name not in _SUPPORTED_BINARIES:
        raise RuntimeInstallError(f"Unsupported native runtime executable: {binary_name}")
    suffix = ".exe" if target.startswith("windows-") else ""
    return f"{binary_name}{suffix}"


def _trusted_url(url: str, *, api: bool) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
        host = (parsed.hostname or "").casefold()
        if parsed.scheme != "https" or parsed.username or parsed.password:
            return False
        if api:
            return host in _RELEASE_API_HOSTS
        return host in _GITHUB_ASSET_HOSTS
    except ValueError:
        return False


def _asset_download_url(asset: dict[str, Any], expected_name: str) -> str:
    name = asset.get("name")
    url = asset.get("browser_download_url")
    expected_prefix = f"https://github.com/{RELEASE_REPOSITORY}/releases/download/{RELEASE_TAG}/"
    if (
        name != expected_name
        or not isinstance(url, str)
        or not url.startswith(expected_prefix)
        or not _trusted_url(url, api=False)
    ):
        raise RuntimeIntegrityError(
            f"The release contains an invalid asset entry for {expected_name}."
        )
    return url


def _make_request(url: str, *, api: bool) -> urllib.request.Request:
    if not _trusted_url(url, api=api):
        raise RuntimeIntegrityError(
            "Refusing to contact a non-HTTPS or untrusted runtime release URL."
        )
    headers = {
        "User-Agent": "Cortex-LLMHoster-runtime-installer",
        "Accept": "application/vnd.github+json" if api else "application/octet-stream",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    return urllib.request.Request(url, headers=headers)


def _open_url(url: str, *, api: bool, opener: UrlOpener | None) -> Any:
    request = _make_request(url, api=api)
    open_request = opener or urllib.request.urlopen
    try:
        response = open_request(request, timeout=_HTTP_TIMEOUT_SECONDS)
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        raise RuntimeReleaseUnavailableError(
            f"Could not reach the GitHub runtime release: {exc}"
        ) from exc

    try:
        final_url = response.geturl()
    except (AttributeError, OSError) as exc:
        response.close()
        raise RuntimeIntegrityError(
            "The GitHub runtime response did not provide a final URL."
        ) from exc
    if not _trusted_url(final_url, api=api):
        response.close()
        raise RuntimeIntegrityError("The GitHub runtime release redirected to an untrusted URL.")
    status = getattr(response, "status", None)
    if status is None and hasattr(response, "getcode"):
        status = response.getcode()
    if status is not None and status != 200:
        response.close()
        raise RuntimeReleaseUnavailableError(f"GitHub runtime download returned HTTP {status}.")
    return response


def _content_length(response: Any) -> int | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get("Content-Length")
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _read_small_response(
    url: str,
    *,
    api: bool,
    maximum_bytes: int,
    opener: UrlOpener | None,
) -> bytes:
    response = _open_url(url, api=api, opener=opener)
    with response:
        content_length = _content_length(response)
        if content_length is not None and content_length > maximum_bytes:
            raise RuntimeIntegrityError("GitHub runtime metadata exceeded the permitted size.")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = response.read(min(64 * 1024, maximum_bytes - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum_bytes:
                raise RuntimeIntegrityError("GitHub runtime metadata exceeded the permitted size.")
            chunks.append(chunk)
        return b"".join(chunks)


def _download_to_file(
    url: str,
    destination: Path,
    *,
    maximum_bytes: int,
    opener: UrlOpener | None,
) -> tuple[int, str]:
    response = _open_url(url, api=False, opener=opener)
    with response:
        content_length = _content_length(response)
        if content_length is not None and content_length > maximum_bytes:
            raise RuntimeIntegrityError(
                "The native runtime archive is larger than the safe download limit."
            )
        digest = hashlib.sha256()
        total = 0
        with destination.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum_bytes:
                    raise RuntimeIntegrityError(
                        "The native runtime archive is larger than the safe download limit."
                    )
                digest.update(chunk)
                output.write(chunk)
        return total, digest.hexdigest()


def _parse_json(data: bytes, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeIntegrityError(f"The GitHub {label} is not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise RuntimeIntegrityError(f"The GitHub {label} must be a JSON object.")
    return parsed


def _fetch_release_manifest(
    opener: UrlOpener | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    release_data = _parse_json(
        _read_small_response(
            RELEASE_API_URL,
            api=True,
            maximum_bytes=_MAX_MANIFEST_BYTES,
            opener=opener,
        ),
        "release record",
    )
    if release_data.get("tag_name") != RELEASE_TAG:
        raise RuntimeIntegrityError(
            f"GitHub returned a release other than the required {RELEASE_TAG!r} tag."
        )

    assets_list = release_data.get("assets")
    if not isinstance(assets_list, list):
        raise RuntimeIntegrityError("The GitHub runtime release has no valid asset list.")
    release_assets: dict[str, dict[str, Any]] = {}
    for asset in assets_list:
        if isinstance(asset, dict) and isinstance(asset.get("name"), str):
            release_assets[asset["name"]] = asset

    manifest_asset = release_assets.get(MANIFEST_ASSET_NAME)
    if manifest_asset is None:
        raise RuntimeReleaseUnavailableError(
            f"The {RELEASE_TAG!r} release has no {MANIFEST_ASSET_NAME} asset yet."
        )
    manifest_url = _asset_download_url(manifest_asset, MANIFEST_ASSET_NAME)
    manifest = _parse_json(
        _read_small_response(
            manifest_url,
            api=False,
            maximum_bytes=_MAX_MANIFEST_BYTES,
            opener=opener,
        ),
        "runtime manifest",
    )
    if manifest.get("schema_version") != 1 or manifest.get("release_tag") != RELEASE_TAG:
        raise RuntimeIntegrityError("The GitHub runtime manifest has an unsupported schema or tag.")
    if not isinstance(manifest.get("assets"), dict):
        raise RuntimeIntegrityError("The GitHub runtime manifest has no platform asset map.")
    return manifest, release_assets


def _get_release_manifest(
    opener: UrlOpener | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    global _release_cache
    if opener is not None:
        return _fetch_release_manifest(opener)

    with _release_cache_lock:
        now = time.monotonic()
        if _release_cache is not None and now - _release_cache[0] < _RELEASE_CACHE_TTL_SECONDS:
            return _release_cache[1], _release_cache[2]
        manifest, release_assets = _fetch_release_manifest(None)
        _release_cache = (now, manifest, release_assets)
        return manifest, release_assets


def _is_usable_binary(path: Path, target: str) -> bool:
    if not path.is_file():
        return False
    return target.startswith("windows-") or os.access(path, os.X_OK)


def _required_runtime_libraries(target: str) -> tuple[str, ...]:
    if target.startswith("linux-"):
        return ("libgomp.so.1",)
    if target.startswith("windows-"):
        return ("vcomp140.dll",)
    return ()


def _is_usable_bundle(version_root: Path, filename: str, target: str) -> bool:
    return _is_usable_binary(version_root / filename, target) and all(
        (version_root / library).is_file() for library in _required_runtime_libraries(target)
    )


def _find_cached_binary(cache_root: Path, target: str, filename: str) -> Path | None:
    target_root = cache_root / RELEASE_TAG / target
    try:
        versions = sorted(
            (path for path in target_root.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
    except OSError:
        return None
    for version in versions:
        candidate = version / filename
        if _is_usable_bundle(version, filename, target):
            return candidate.resolve()
    return None


def _extract_archive_member(
    archive: zipfile.ZipFile,
    filename: str,
    destination: Path,
    *,
    maximum_bytes: int,
    required: bool,
) -> bool:
    try:
        member = archive.getinfo(filename)
    except KeyError as exc:
        if required:
            raise RuntimeIntegrityError(
                f"The release bundle is missing its expected {filename} file."
            ) from exc
        return False
    if member.is_dir() or member.file_size <= 0 or member.file_size > maximum_bytes:
        raise RuntimeIntegrityError(f"The native runtime bundle has an invalid {filename} size.")
    mode = member.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise RuntimeIntegrityError("The native runtime bundle may not install symbolic links.")
    with archive.open(member, "r") as source, destination.open("wb") as output:
        shutil.copyfileobj(source, output, length=1024 * 1024)
    if destination.stat().st_size != member.file_size:
        raise RuntimeIntegrityError(f"The extracted {filename} has an unexpected size.")
    return True


def _extract_runtime_bundle(
    archive_path: Path,
    target: str,
    destination: Path,
    binaries: list[str],
    runtime_libraries: list[str],
) -> None:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for filename in binaries:
                extracted = destination / filename
                _extract_archive_member(
                    archive,
                    filename,
                    extracted,
                    maximum_bytes=_MAX_BINARY_BYTES,
                    required=True,
                )
                if not target.startswith("windows-"):
                    extracted.chmod(0o755)
            for filename in runtime_libraries:
                extracted_library = destination / filename
                _extract_archive_member(
                    archive,
                    filename,
                    extracted_library,
                    maximum_bytes=_MAX_RUNTIME_LIBRARY_BYTES,
                    required=True,
                )
                if target.startswith("linux-"):
                    extracted_library.chmod(0o755)
    except RuntimeIntegrityError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile, NotImplementedError) as exc:
        raise RuntimeIntegrityError(
            f"Could not safely extract the native runtime bundle: {exc}"
        ) from exc


def install_latest_runtime_binary(
    binary_name: str,
    *,
    cache_dir: str | Path | None = None,
    target: str | None = None,
    opener: UrlOpener | None = None,
) -> Path:
    """Return a verified binary from the latest ``cortex-cpp`` tagged release.

    If GitHub is unavailable or a rolling release update is incomplete, the
    last verified bundle for this platform is reused. ``cache_dir`` and
    ``opener`` are injectable for deterministic tests.
    """

    selected_target = target or release_target()
    if selected_target not in SUPPORTED_RELEASE_TARGETS:
        raise RuntimeReleaseUnavailableError(
            f"No prebuilt Cortex C++ runtime is published for this platform ({selected_target or 'unknown'})."
        )
    filename = _asset_filename(binary_name, selected_target)
    cache_root = Path(cache_dir).expanduser() if cache_dir is not None else runtime_cache_dir()

    try:
        manifest, release_assets = _get_release_manifest(opener)
        platform_assets = manifest["assets"].get(selected_target)
        if not isinstance(platform_assets, dict):
            raise RuntimeReleaseUnavailableError(
                f"The {RELEASE_TAG!r} release does not contain a {selected_target} runtime bundle."
            )
        archive_name = f"cortex-cpp-{selected_target}.zip"
        expected_binaries = sorted(
            _asset_filename(name, selected_target) for name in _SUPPORTED_BINARIES
        )
        manifest_binaries = platform_assets.get("binaries")
        if (
            not isinstance(manifest_binaries, list)
            or not all(isinstance(name, str) for name in manifest_binaries)
            or sorted(manifest_binaries) != expected_binaries
        ):
            raise RuntimeIntegrityError(
                f"The {selected_target} runtime manifest has an incomplete executable list."
            )
        runtime_libraries = platform_assets.get("runtime_libraries", [])
        if (
            not isinstance(runtime_libraries, list)
            or not all(isinstance(name, str) for name in runtime_libraries)
            or sorted(runtime_libraries) != sorted(_required_runtime_libraries(selected_target))
            or any(name not in _ALLOWED_RUNTIME_LIBRARIES for name in runtime_libraries)
        ):
            raise RuntimeIntegrityError(
                f"The {selected_target} runtime manifest has an invalid runtime library list."
            )
        expected_hash = platform_assets.get("sha256")
        expected_size = platform_assets.get("size")
        if not isinstance(expected_hash, str) or not _SHA256_PATTERN.fullmatch(expected_hash):
            raise RuntimeIntegrityError(
                f"The {selected_target} runtime has an invalid SHA-256 entry."
            )
        if (
            not isinstance(expected_size, int)
            or expected_size <= 0
            or expected_size > _MAX_ARCHIVE_BYTES
        ):
            raise RuntimeIntegrityError(
                f"The {selected_target} runtime has an invalid archive size."
            )
        if platform_assets.get("name") != archive_name:
            raise RuntimeIntegrityError(
                f"The {selected_target} runtime has an unexpected asset name."
            )

        archive_asset = release_assets.get(archive_name)
        if archive_asset is None:
            raise RuntimeReleaseUnavailableError(
                f"The {RELEASE_TAG!r} release is missing {archive_name}."
            )
        api_size = archive_asset.get("size")
        if not isinstance(api_size, int) or api_size != expected_size:
            raise RuntimeIntegrityError(
                "The runtime manifest size does not match the GitHub asset metadata."
            )
        archive_url = _asset_download_url(archive_asset, archive_name)
        version_root = cache_root / RELEASE_TAG / selected_target / expected_hash[:20]
        installed_binary = version_root / filename
        if _is_usable_bundle(version_root, filename, selected_target):
            return installed_binary.resolve()

        cache_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".cortex-runtime-", dir=cache_root) as temp_name:
            temp_dir = Path(temp_name)
            archive_path = temp_dir / archive_name
            downloaded_size, actual_hash = _download_to_file(
                archive_url,
                archive_path,
                maximum_bytes=_MAX_ARCHIVE_BYTES,
                opener=opener,
            )
            if downloaded_size != expected_size or actual_hash != expected_hash:
                raise RuntimeIntegrityError(
                    f"The {selected_target} runtime archive failed its SHA-256/size check."
                )
            staged_version = temp_dir / "bundle"
            staged_version.mkdir()
            _extract_runtime_bundle(
                archive_path,
                selected_target,
                staged_version,
                manifest_binaries,
                runtime_libraries,
            )
            version_root.parent.mkdir(parents=True, exist_ok=True)
            if not version_root.exists():
                os.replace(staged_version, version_root)
            else:
                version_root.mkdir(parents=True, exist_ok=True)
                for staged_file in staged_version.iterdir():
                    os.replace(staged_file, version_root / staged_file.name)
        return installed_binary.resolve()
    except (RuntimeReleaseUnavailableError, RuntimeIntegrityError):
        cached = _find_cached_binary(cache_root, selected_target, filename)
        if cached is not None:
            return cached
        raise
    except OSError as exc:
        cached = _find_cached_binary(cache_root, selected_target, filename)
        if cached is not None:
            return cached
        raise RuntimeInstallError(f"Could not install the {binary_name} runtime: {exc}") from exc
