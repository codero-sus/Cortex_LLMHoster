from __future__ import annotations

import pytest

from cortex_llmhoster.config import ConfigurationError, Settings


def test_loads_local_gguf_toml_and_reads_api_key_from_environment(tmp_path) -> None:
    config = tmp_path / "cortex.toml"
    config.write_text(
        """
[server]
api_key_env = "CORTEX_KEY"
default_model = "public"
host = "127.0.0.1"
port = 9123
workers = 1
log_level = "debug"
access_log = true
max_connections = 64

[[models]]
id = "public"
runtime = "llama.cpp"
model_path = "/models/qwen-q4.gguf"
upstream_model = "engine-alias"
gpu_layers = -1
context_size = 1536
batch_size = 128
ubatch_size = 32
embedding = true
""",
        encoding="utf-8",
    )

    settings = Settings.load(config, environ={"CORTEX_KEY": "cortex-secret"})

    model = settings.models["public"]
    assert settings.default_model == "public"
    assert settings.api_key == "cortex-secret"
    assert settings.host == "127.0.0.1"
    assert settings.port == 9123
    assert settings.workers == 1
    assert settings.log_level == "debug"
    assert settings.access_log is True
    assert settings.max_connections == 64
    assert settings.max_inference_requests == 8
    assert model.runtime == "llama.cpp"
    assert model.model_path == "/models/qwen-q4.gguf"
    assert model.upstream_model == "engine-alias"
    assert model.gpu_layers == -1
    assert model.context_size == 1536
    assert model.embedding is True


def test_environment_can_define_local_models_without_a_toml_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings.load(
        environ={"CORTEX_MODELS": '[{"id":"only-model","model_path":"/models/only.gguf"}]'},
    )

    assert settings.default_model == "only-model"
    assert list(settings.models) == ["only-model"]
    assert settings.models["only-model"].runtime == "llama.cpp"


def test_missing_explicit_config_is_a_clear_error(tmp_path) -> None:
    with pytest.raises(ConfigurationError, match="Configuration file not found"):
        Settings.load(tmp_path / "missing.toml", environ={})


def test_duplicate_defaults_are_rejected(tmp_path) -> None:
    config = tmp_path / "cortex.toml"
    config.write_text(
        """
[[models]]
id = "one"
model_path = "/models/one.gguf"
default = true

[[models]]
id = "two"
model_path = "/models/two.gguf"
default = true
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="Only one model"):
        Settings.load(config, environ={})


def test_rejects_non_local_model_runtime(tmp_path) -> None:
    config = tmp_path / "cortex.toml"
    config.write_text(
        """
[[models]]
id = "external"
runtime = "openai"
base_url = "https://example.test/v1"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="hosts local GGUF models only"):
        Settings.load(config, environ={})


def test_inference_request_limit_is_configurable_and_bounded_by_connection_pool() -> None:
    settings = Settings.from_mapping(
        {"server": {"max_connections": 4, "max_inference_requests": 3}},
        environ={"CORTEX_MAX_INFERENCE_REQUESTS": "2"},
    )
    assert settings.max_inference_requests == 2
    assert (
        Settings.from_mapping({"server": {"max_connections": 4}}, environ={}).max_inference_requests
        == 4
    )

    with pytest.raises(ConfigurationError, match="must be at least 1"):
        Settings.from_mapping(
            {"server": {"max_inference_requests": 0}},
            environ={},
        )
    with pytest.raises(ConfigurationError, match="cannot exceed max_connections"):
        Settings.from_mapping(
            {"server": {"max_connections": 2, "max_inference_requests": 3}},
            environ={},
        )


def test_local_model_path_must_be_gguf(tmp_path) -> None:
    with pytest.raises(ConfigurationError, match="must point to a GGUF file"):
        Settings.from_mapping(
            {"models": [{"id": "not-gguf", "model_path": "/models/model.bin"}]},
            environ={},
        )
