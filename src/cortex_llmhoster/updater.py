"""Manual, branch-pinned software update checks and installation."""

from __future__ import annotations

import asyncio
import base64
import os
import re
import sys
import tomllib
from dataclasses import dataclass
from urllib.parse import quote

import httpx

REPOSITORY = "codero-sus/Cortex_LLMHoster"
UPDATE_REF = "arena/b3072048-cortex-llmhoster"
GITHUB_API = f"https://api.github.com/repos/{REPOSITORY}"
_GITHUB_HEADERS = {
    "accept": "application/vnd.github+json",
    "user-agent": "Cortex-LLMHoster-Updater",
}
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_VERSION_PATTERN = re.compile(r"\d+(?:\.\d+){1,3}")


class UpdateError(RuntimeError):
    """A safe, user-displayable updater failure."""


@dataclass(frozen=True, slots=True)
class UpdateInfo:
    current_version: str
    latest_version: str
    source_repository: str
    source_ref: str
    commit_sha: str
    update_available: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "current_version": self.current_version,
            "latest_version": self.latest_version,
            "source_repository": self.source_repository,
            "source_ref": self.source_ref,
            "commit_sha": self.commit_sha,
            "update_available": self.update_available,
        }


def _version_key(version: str) -> tuple[int, int, int, int]:
    if not _VERSION_PATTERN.fullmatch(version):
        raise UpdateError("The update source contains an unsupported version format.")
    parts = tuple(int(part) for part in version.split("."))
    return (parts + (0, 0, 0, 0))[:4]


async def _github_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, str] | None = None,
) -> dict[str, object]:
    try:
        response = await client.get(
            url,
            params=params,
            headers=_GITHUB_HEADERS,
            timeout=httpx.Timeout(10.0, connect=4.0),
        )
    except httpx.HTTPError as exc:
        raise UpdateError("Could not reach GitHub to check for updates.") from exc

    if response.status_code == 404:
        raise UpdateError("The configured updater branch or project file was not found.")
    if response.status_code == 403 and response.headers.get("x-ratelimit-remaining") == "0":
        raise UpdateError(
            "GitHub's unauthenticated update-check rate limit was reached. Try later."
        )
    if not response.is_success:
        raise UpdateError(f"GitHub returned HTTP {response.status_code} while checking updates.")
    try:
        payload = response.json()
    except ValueError as exc:
        raise UpdateError("GitHub returned an invalid update-check response.") from exc
    if not isinstance(payload, dict):
        raise UpdateError("GitHub returned an invalid update-check response.")
    return payload


async def check_for_updates(
    client: httpx.AsyncClient,
    current_version: str,
) -> UpdateInfo:
    """Read the pinned branch head and its package version from GitHub's API."""

    branch_url = f"{GITHUB_API}/branches/{quote(UPDATE_REF, safe='')}"
    branch_data = await _github_json(client, branch_url)
    commit = branch_data.get("commit")
    commit_sha = commit.get("sha") if isinstance(commit, dict) else None
    if not isinstance(commit_sha, str) or not _COMMIT_PATTERN.fullmatch(commit_sha):
        raise UpdateError("GitHub returned an invalid update commit ID.")

    contents_url = f"{GITHUB_API}/contents/pyproject.toml"
    project_data = await _github_json(client, contents_url, params={"ref": commit_sha})
    encoded_content = project_data.get("content")
    if not isinstance(encoded_content, str):
        raise UpdateError("The update source did not include package metadata.")
    try:
        project_toml = base64.b64decode(encoded_content, validate=False).decode("utf-8")
        project = tomllib.loads(project_toml)
        project_section = project.get("project")
        latest_version = (
            project_section.get("version") if isinstance(project_section, dict) else None
        )
    except (ValueError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise UpdateError("The update source contains invalid package metadata.") from exc
    if not isinstance(latest_version, str):
        raise UpdateError("The update source does not declare a package version.")

    current_key = _version_key(current_version)
    latest_key = _version_key(latest_version)
    return UpdateInfo(
        current_version=current_version,
        latest_version=latest_version,
        source_repository=REPOSITORY,
        source_ref=UPDATE_REF,
        commit_sha=commit_sha,
        update_available=latest_key > current_key,
    )


async def _stop_installer(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(process.wait(), timeout=5.0)
    except TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        await process.wait()


async def install_from_commit(commit_sha: str, *, timeout_seconds: float = 300.0) -> str:
    """Install an immutable archive from the pinned project repository via pip."""

    if not _COMMIT_PATTERN.fullmatch(commit_sha):
        raise UpdateError("Refusing to install an invalid update commit ID.")

    archive_url = f"https://github.com/{REPOSITORY}/archive/{commit_sha}.zip"
    env = os.environ.copy()
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--no-input",
            archive_url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
    except (OSError, ValueError) as exc:
        raise UpdateError(f"Could not start pip for the update: {exc}") from exc

    try:
        output, _ = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
    except TimeoutError as exc:
        await _stop_installer(process)
        raise UpdateError(
            "The update installation timed out; check the terminal and try again."
        ) from exc
    except asyncio.CancelledError:
        await _stop_installer(process)
        raise

    output_text = output.decode("utf-8", errors="replace") if output else ""
    if process.returncode != 0:
        details = output_text[-6000:].strip()
        message = "pip could not install the update. Check network access and package permissions."
        if details:
            message = f"{message}\n{details}"
        raise UpdateError(message)
    return output_text[-6000:].strip()
