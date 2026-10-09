"""Configuration for Cortex's user-facing OpenAI-compatible API routes."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ConfigurationError

_API_CONFIG_ENV = "CORTEX_API_CONFIG"
_API_CONFIG_FILENAME = "api.json"
_ALLOWED_KEYS = {"format", "version", "base_path", "models_path", "examples"}
_RESERVED_PATHS = ("/admin", "/health", "/ready")


def default_api_config_path() -> Path:
    """Find the repository config, falling back to the packaged default."""

    source_config = Path(__file__).resolve().parents[1] / "config" / _API_CONFIG_FILENAME
    if source_config.is_file():
        return source_config
    return Path(__file__).resolve().parent / _API_CONFIG_FILENAME


def _route_path(value: object, field: str, *, trim_trailing_slash: bool = False) -> str:
    if not isinstance(value, str) or not value.startswith("/"):
        raise ConfigurationError(f"API config {field} must be an absolute route path.")
    path = value.rstrip("/") if trim_trailing_slash else value
    if (
        not path
        or path == "/"
        or path.startswith("//")
        or "//" in path
        or any(char in path for char in "?#\\%")
        or any(segment in {".", ".."} for segment in path.split("/"))
    ):
        raise ConfigurationError(f"API config {field} is not a safe route path.")
    return path


@dataclass(frozen=True, slots=True)
class APIFormatConfig:
    """The supported public API dialect and its configurable route prefix."""

    format: str = "openai"
    version: int = 1
    base_path: str = "/v1"
    models_path: str = "/models"

    @classmethod
    def load(cls, path: str | Path | None = None) -> APIFormatConfig:
        """Load and validate ``src/config/api.json`` or an override path."""

        configured = os.environ.get(_API_CONFIG_ENV) if path is None else None
        selected_path = (
            Path(path or configured).expanduser()
            if path or configured
            else default_api_config_path()
        )
        try:
            data = json.loads(selected_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ConfigurationError(f"Could not read API config {selected_path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigurationError(
                f"API config {selected_path} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(data, Mapping):
            raise ConfigurationError("API config must contain a JSON object.")
        unknown_keys = set(data) - _ALLOWED_KEYS
        if unknown_keys:
            raise ConfigurationError(
                f"API config contains unsupported keys: {', '.join(sorted(unknown_keys))}."
            )

        api_format = data.get("format", "openai")
        if not isinstance(api_format, str) or api_format.strip().lower() != "openai":
            raise ConfigurationError("Only the 'openai' API format is currently supported.")
        version = data.get("version", 1)
        if isinstance(version, bool) or version != 1:
            raise ConfigurationError("API config version must be 1.")

        base_path = _route_path(data.get("base_path", "/v1"), "base_path", trim_trailing_slash=True)
        if any(
            base_path == reserved
            or base_path.startswith(f"{reserved}/")
            or reserved.startswith(f"{base_path}/")
            for reserved in _RESERVED_PATHS
        ):
            raise ConfigurationError("API config base_path conflicts with a reserved Cortex route.")
        models_path = _route_path(data.get("models_path", "/models"), "models_path")
        if models_path == "/" or models_path.startswith("/{"):
            raise ConfigurationError("API config models_path must be a static route path.")
        if not isinstance(data.get("examples", {}), Mapping):
            raise ConfigurationError("API config examples must be a JSON object when provided.")

        return cls(
            format="openai",
            version=1,
            base_path=base_path,
            models_path=models_path,
        )

    @property
    def models_endpoint(self) -> str:
        return f"{self.base_path}{self.models_path}"

    def public_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "base_path": self.base_path,
            "models_path": self.models_path,
            "models_endpoint": self.models_endpoint,
        }
