"""Configuration loading and validation for the Cortex local LLM hoster."""

from __future__ import annotations

import json
import math
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType


class ConfigurationError(ValueError):
    """Raised when a Cortex configuration is invalid."""


def resolve_admin_config_path(
    config_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the local JSON overlay used by the web configuration editor."""

    env = os.environ if environ is None else environ
    selected_config = config_path or env.get("CORTEX_CONFIG", "cortex.toml")
    configured_path = env.get("CORTEX_ADMIN_CONFIG")
    if configured_path:
        return Path(configured_path)
    return Path(selected_config).with_name("cortex.local.json")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """A local GGUF model and its llama.cpp serving configuration."""

    id: str
    upstream_model: str
    runtime: str = "llama.cpp"
    model_path: str | None = None
    threads: int = 0
    threads_batch: int = 0
    gpu_layers: int = 0
    context_size: int = 2048
    batch_size: int = 256
    ubatch_size: int = 64
    parallel: int = 1
    flash_attn: str = "auto"
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    use_mmap: bool = True
    mlock: bool = False
    embedding: bool = False
    mmproj_path: str | None = None
    optimization_level: int = 0


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable runtime settings shared by every request in a worker."""

    models: Mapping[str, ModelConfig] = field(default_factory=dict)
    default_model: str | None = None
    api_key: str | None = None
    api_key_env: str = "CORTEX_API_KEY"
    host: str = "0.0.0.0"
    port: int = 8624
    workers: int = 1
    log_level: str = "warning"
    access_log: bool = False
    llama_server_path: str | None = None
    max_json_body_bytes: int = 16 * 1024 * 1024
    max_inference_requests: int = 8
    connect_timeout: float = 5.0
    read_timeout: float = 300.0
    write_timeout: float = 60.0
    pool_timeout: float = 5.0
    max_connections: int = 512
    max_keepalive_connections: int = 128
    keepalive_expiry: float = 30.0

    @classmethod
    def load(
        cls,
        config_path: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> Settings:
        """Load TOML settings, local UI overrides, then environment overrides.

        If no TOML file is present, an empty model registry is allowed so the
        service can start and report an unhealthy readiness state. The web UI
        writes its settings to a separate JSON overlay, leaving the user's TOML
        file untouched and keeping credentials environment-only.
        """

        env = os.environ if environ is None else environ
        explicit_path = config_path is not None or "CORTEX_CONFIG" in env
        selected_path = config_path or env.get("CORTEX_CONFIG", "cortex.toml")
        path = Path(selected_path)

        if path.is_file():
            try:
                with path.open("rb") as config_file:
                    data = tomllib.load(config_file)
            except (OSError, tomllib.TOMLDecodeError) as exc:
                raise ConfigurationError(f"Could not read {path}: {exc}") from exc
        elif explicit_path:
            raise ConfigurationError(f"Configuration file not found: {path}")
        else:
            data = {}

        local_path = resolve_admin_config_path(config_path, env)
        local_override = False
        if local_path.is_file():
            try:
                with local_path.open("r", encoding="utf-8") as config_file:
                    local_data = json.load(config_file)
            except (OSError, json.JSONDecodeError) as exc:
                raise ConfigurationError(f"Could not read {local_path}: {exc}") from exc
            if not isinstance(local_data, dict):
                raise ConfigurationError(f"{local_path} must contain a JSON object.")
            for section in ("server", "models"):
                if section in local_data:
                    data[section] = local_data[section]
            local_override = "models" in local_data

        parse_env = dict(env)
        # The UI's explicit saved registry supersedes the one-time JSON env list.
        if local_override:
            parse_env.pop("CORTEX_MODELS", None)
        return cls.from_mapping(data, environ=parse_env)

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, object],
        environ: Mapping[str, str] | None = None,
    ) -> Settings:
        """Validate settings from TOML/JSON-compatible dictionaries."""

        env = os.environ if environ is None else environ
        if not isinstance(data, Mapping):
            raise ConfigurationError("The configuration must be an object/table.")

        server = data.get("server", {})
        if not isinstance(server, Mapping):
            raise ConfigurationError("[server] must be a table/object.")

        if "CORTEX_MODELS" in env:
            try:
                model_rows = json.loads(env["CORTEX_MODELS"])
            except json.JSONDecodeError as exc:
                raise ConfigurationError(f"CORTEX_MODELS is not valid JSON: {exc}") from exc
        else:
            model_rows = data.get("models", [])
        if not isinstance(model_rows, list):
            raise ConfigurationError("models must be an array of [[models]] tables.")

        models: dict[str, ModelConfig] = {}
        declared_defaults: list[str] = []
        for index, row in enumerate(model_rows):
            prefix = f"models[{index}]"
            if not isinstance(row, Mapping):
                raise ConfigurationError(f"{prefix} must be a table/object.")

            model_id = row.get("id")
            if not isinstance(model_id, str) or not model_id.strip():
                raise ConfigurationError(f"{prefix}.id must be a non-empty string.")
            if model_id in models:
                raise ConfigurationError(f"Duplicate model id: {model_id!r}.")

            runtime = row.get("runtime", "llama.cpp")
            if runtime != "llama.cpp":
                raise ConfigurationError(
                    f"{prefix}.runtime must be 'llama.cpp'; Cortex hosts local GGUF models only."
                )
            raw_optimization_level = row.get("optimization_level")
            if raw_optimization_level is None:
                raw_optimization_level = row.get("optimization_profile", 0)
            profile_levels = {"efficient": -1, "eco": -1, "balanced": 0, "ultra": 1}
            if isinstance(raw_optimization_level, bool):
                raise ConfigurationError(f"{prefix}.optimization_level must be -1, 0, or 1.")
            if isinstance(raw_optimization_level, str):
                normalized_level = raw_optimization_level.strip().lower()
                if normalized_level in profile_levels:
                    optimization_level = profile_levels[normalized_level]
                else:
                    try:
                        optimization_level = int(normalized_level)
                    except ValueError as exc:
                        raise ConfigurationError(
                            f"{prefix}.optimization_level must be -1, 0, or 1."
                        ) from exc
            elif isinstance(raw_optimization_level, int):
                optimization_level = raw_optimization_level
            elif isinstance(raw_optimization_level, float) and raw_optimization_level.is_integer():
                optimization_level = int(raw_optimization_level)
            else:
                raise ConfigurationError(f"{prefix}.optimization_level must be -1, 0, or 1.")
            if optimization_level not in {-1, 0, 1}:
                raise ConfigurationError(f"{prefix}.optimization_level must be -1, 0, or 1.")

            model_path = row.get("model_path")
            if not isinstance(model_path, str) or not model_path.strip():
                raise ConfigurationError(f"{prefix}.model_path is required for local GGUF models.")
            if Path(model_path).suffix.lower() != ".gguf":
                raise ConfigurationError(f"{prefix}.model_path must point to a GGUF file.")

            upstream_model = row.get("upstream_model", model_id)
            if not isinstance(upstream_model, str) or not upstream_model.strip():
                raise ConfigurationError(f"{prefix}.upstream_model must be a non-empty string.")

            def model_int(
                name: str,
                default: int,
                minimum: int,
                row: Mapping[str, object] = row,
                prefix: str = prefix,
            ) -> int:
                value = row.get(name, default)
                if isinstance(value, bool):
                    raise ConfigurationError(f"{prefix}.{name} must be an integer.")
                try:
                    result = int(value)
                except (TypeError, ValueError) as exc:
                    raise ConfigurationError(f"{prefix}.{name} must be an integer.") from exc
                if result < minimum:
                    raise ConfigurationError(f"{prefix}.{name} must be at least {minimum}.")
                return result

            threads = model_int("threads", 0, 0)
            threads_batch = model_int("threads_batch", threads, 0)
            gpu_layers = model_int("gpu_layers", 0, -1)
            context_size = model_int("context_size", 2048, 128)
            batch_size = model_int("batch_size", 256, 1)
            ubatch_size = model_int("ubatch_size", 64, 1)
            if ubatch_size > batch_size:
                raise ConfigurationError(f"{prefix}.ubatch_size cannot exceed batch_size.")
            parallel = model_int("parallel", 1, 1)
            flash_attn = row.get("flash_attn", "auto")
            if not isinstance(flash_attn, str) or flash_attn not in {"auto", "on", "off"}:
                raise ConfigurationError(f"{prefix}.flash_attn must be auto, on, or off.")
            cache_types = {"f32", "f16", "bf16", "q8_0", "q5_0", "q5_1", "q4_0", "q4_1"}
            cache_type_k = row.get("cache_type_k", "f16")
            cache_type_v = row.get("cache_type_v", "f16")
            if (
                not isinstance(cache_type_k, str)
                or not isinstance(cache_type_v, str)
                or cache_type_k not in cache_types
                or cache_type_v not in cache_types
            ):
                raise ConfigurationError(
                    f"{prefix} cache types must be f32, f16, bf16, q8_0, q5_0, q5_1, q4_0, or q4_1."
                )
            use_mmap = row.get("use_mmap", True)
            mlock = row.get("mlock", False)
            embedding = row.get("embedding", False)
            mmproj_path = row.get("mmproj_path")
            if not all(isinstance(value, bool) for value in (use_mmap, mlock, embedding)):
                raise ConfigurationError(
                    f"{prefix}.use_mmap, {prefix}.mlock, and {prefix}.embedding must be booleans."
                )
            if mmproj_path is not None and not isinstance(mmproj_path, str):
                raise ConfigurationError(f"{prefix}.mmproj_path must be a path string.")

            models[model_id] = ModelConfig(
                id=model_id,
                upstream_model=upstream_model,
                runtime=runtime,
                optimization_level=optimization_level,
                model_path=model_path,
                threads=threads,
                threads_batch=threads_batch,
                gpu_layers=gpu_layers,
                context_size=context_size,
                batch_size=batch_size,
                ubatch_size=ubatch_size,
                parallel=parallel,
                flash_attn=flash_attn,
                cache_type_k=cache_type_k,
                cache_type_v=cache_type_v,
                use_mmap=use_mmap,
                mlock=mlock,
                embedding=embedding,
                mmproj_path=mmproj_path,
            )
            is_default = row.get("default", False)
            if not isinstance(is_default, bool):
                raise ConfigurationError(f"{prefix}.default must be true or false.")
            if is_default:
                declared_defaults.append(model_id)

        if len(declared_defaults) > 1:
            raise ConfigurationError("Only one model may set default = true.")

        configured_default = server.get("default_model")
        env_default = env.get("CORTEX_DEFAULT_MODEL")
        default_model = env_default if env_default is not None else configured_default
        if default_model is not None and not isinstance(default_model, str):
            raise ConfigurationError("server.default_model must be a model id string.")
        if default_model is None and declared_defaults:
            default_model = declared_defaults[0]
        if default_model is None and len(models) == 1:
            default_model = next(iter(models))
        if default_model is not None and default_model not in models:
            raise ConfigurationError(
                f"Default model {default_model!r} is not present in the model registry."
            )
        if declared_defaults and default_model != declared_defaults[0]:
            raise ConfigurationError(
                "server.default_model conflicts with the model marked default = true."
            )

        api_key_env = env.get("CORTEX_API_KEY_ENV", server.get("api_key_env", "CORTEX_API_KEY"))
        if not isinstance(api_key_env, str):
            raise ConfigurationError("server.api_key_env must be an environment name.")
        api_key = env.get("CORTEX_API_KEY")
        if api_key is None and api_key_env:
            api_key = env.get(api_key_env)
        if api_key == "":
            raise ConfigurationError("The Cortex API key must not be empty.")

        def setting(name: str, env_name: str, default: float, cast: type):
            value = env.get(env_name, server.get(name, default))
            try:
                converted = cast(value)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(f"{env_name} / server.{name} is invalid.") from exc
            return converted

        host = env.get("CORTEX_HOST", server.get("host", "0.0.0.0"))
        if not isinstance(host, str) or not host:
            raise ConfigurationError("CORTEX_HOST / server.host must be a non-empty string.")
        port = setting("port", "CORTEX_PORT", 8624, int)
        workers = setting("workers", "CORTEX_WORKERS", 1, int)
        if not 1 <= port <= 65535:
            raise ConfigurationError("port must be between 1 and 65535.")
        if workers < 1:
            raise ConfigurationError("workers must be at least 1.")
        if workers > 1 and any(model.runtime == "llama.cpp" for model in models.values()):
            raise ConfigurationError(
                "Local llama.cpp models require one Cortex worker so model weights are not "
                "loaded once per worker; use llama-server parallelism for concurrent requests."
            )

        log_level = env.get("CORTEX_LOG_LEVEL", server.get("log_level", "warning"))
        if not isinstance(log_level, str) or log_level.lower() not in {
            "critical",
            "error",
            "warning",
            "info",
            "debug",
            "trace",
        }:
            raise ConfigurationError(
                "log_level must be critical, error, warning, info, debug, or trace."
            )
        log_level = log_level.lower()
        raw_access_log: object = env.get("CORTEX_ACCESS_LOG", server.get("access_log", False))
        if isinstance(raw_access_log, bool):
            access_log = raw_access_log
        elif isinstance(raw_access_log, str) and raw_access_log.lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            access_log = True
        elif isinstance(raw_access_log, str) and raw_access_log.lower() in {
            "0",
            "false",
            "no",
            "off",
        }:
            access_log = False
        else:
            raise ConfigurationError("access_log must be a boolean.")

        llama_server_path = env.get("CORTEX_LLAMA_SERVER", server.get("llama_server_path"))
        if llama_server_path == "":
            llama_server_path = None
        if llama_server_path is not None and not isinstance(llama_server_path, str):
            raise ConfigurationError("llama_server_path must be a filesystem path.")

        max_connections = setting("max_connections", "CORTEX_MAX_CONNECTIONS", 512, int)
        max_keepalive_connections = setting(
            "max_keepalive_connections",
            "CORTEX_MAX_KEEPALIVE_CONNECTIONS",
            min(128, max_connections),
            int,
        )
        if max_connections < 1:
            raise ConfigurationError("max_connections must be at least 1.")
        max_inference_requests = setting(
            "max_inference_requests", "CORTEX_MAX_INFERENCE_REQUESTS", min(8, max_connections), int
        )
        if max_inference_requests < 1:
            raise ConfigurationError("max_inference_requests must be at least 1.")
        if max_inference_requests > max_connections:
            raise ConfigurationError("max_inference_requests cannot exceed max_connections.")
        if not 0 <= max_keepalive_connections <= max_connections:
            raise ConfigurationError(
                "max_keepalive_connections must be between 0 and max_connections."
            )

        max_json_body_bytes = setting(
            "max_json_body_bytes", "CORTEX_MAX_JSON_BODY_BYTES", 16 * 1024 * 1024, int
        )
        if max_json_body_bytes < 1024:
            raise ConfigurationError("max_json_body_bytes must be at least 1024.")

        timeouts = {
            "connect_timeout": setting("connect_timeout", "CORTEX_CONNECT_TIMEOUT", 5.0, float),
            "read_timeout": setting("read_timeout", "CORTEX_READ_TIMEOUT", 300.0, float),
            "write_timeout": setting("write_timeout", "CORTEX_WRITE_TIMEOUT", 60.0, float),
            "pool_timeout": setting("pool_timeout", "CORTEX_POOL_TIMEOUT", 5.0, float),
            "keepalive_expiry": setting("keepalive_expiry", "CORTEX_KEEPALIVE_EXPIRY", 30.0, float),
        }
        if any(not math.isfinite(value) or value <= 0 for value in timeouts.values()):
            raise ConfigurationError("Timeout and keepalive settings must be finite and positive.")

        return cls(
            models=MappingProxyType(models),
            default_model=default_model,
            api_key=api_key,
            api_key_env=api_key_env,
            host=host,
            port=port,
            workers=workers,
            log_level=log_level,
            access_log=access_log,
            llama_server_path=llama_server_path,
            max_json_body_bytes=max_json_body_bytes,
            max_inference_requests=max_inference_requests,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            **timeouts,
        )
