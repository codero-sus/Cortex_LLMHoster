from __future__ import annotations

import pytest

from cortex_llmhoster.config import ConfigurationError, ModelConfig, Settings


def test_default_listener_is_localhost_on_port_8624() -> None:
    assert Settings().host == "127.0.0.1"
    assert Settings().port == 8624
    settings = Settings.from_mapping({}, environ={})
    assert settings.host == "127.0.0.1"
    assert settings.port == 8624


@pytest.mark.parametrize("host", ("localhost", "0.0.0.0", "::", "2001:db8::1", "münchen.example"))
def test_valid_bind_hosts_are_supported(host: str) -> None:
    settings = Settings.from_mapping({"server": {"host": host}}, environ={})

    assert settings.host == host


@pytest.mark.parametrize(
    "host",
    (
        "",
        " ",
        "localhost:8624",
        "http://localhost",
        "host/path",
        "host name",
        "-bad-host",
        "bad..host",
        "[::1]",
        "host\x00name",
    ),
)
def test_invalid_bind_hosts_fail_during_config_validation(host: str) -> None:
    with pytest.raises(ConfigurationError, match="IP address or valid DNS hostname"):
        Settings.from_mapping({"server": {"host": host}}, environ={})


def test_api_auth_is_switchable_and_requires_a_key_when_enabled() -> None:
    no_key = Settings.from_mapping({}, environ={})
    key_defaults_to_on = Settings.from_mapping({}, environ={"CORTEX_API_KEY": "local-secret"})
    explicit_off = Settings.from_mapping(
        {}, environ={"CORTEX_API_KEY": "local-secret", "CORTEX_API_AUTH_ENABLED": "false"}
    )
    toml_off = Settings.from_mapping({"server": {"api_auth_enabled": False}}, environ={})

    assert no_key.api_auth_enabled is False
    assert Settings(api_key="").api_auth_enabled is False
    assert key_defaults_to_on.api_auth_enabled is True
    assert explicit_off.api_auth_enabled is False
    assert explicit_off.api_key == "local-secret"
    assert toml_off.api_auth_enabled is False

    with pytest.raises(ConfigurationError, match="CORTEX_API_KEY is not configured"):
        Settings.from_mapping({}, environ={"CORTEX_API_AUTH_ENABLED": "true"})
    with pytest.raises(ConfigurationError, match="must be boolean"):
        Settings.from_mapping({}, environ={"CORTEX_API_AUTH_ENABLED": "sometimes"})


@pytest.mark.parametrize(
    "api_key_env",
    (
        "",
        "9CORTEX_KEY",
        "CORTEX-KEY",
        "CORTEX KEY",
        "CORTEX_ADMIN_PASSWORD",
        "cortex_admin_password",
        "CORTEX_API_AUTH_ENABLED",
    ),
)
def test_api_key_environment_name_must_be_a_valid_identifier(api_key_env: str) -> None:
    with pytest.raises(ConfigurationError, match="valid environment name"):
        Settings.from_mapping({"server": {"api_key_env": api_key_env}}, environ={})


@pytest.mark.parametrize(
    "settings_kwargs",
    (
        {"admin_username": 123, "admin_password": "password"},
        {"admin_username": "admin", "admin_password": 123},
    ),
)
def test_direct_settings_reject_non_string_dashboard_credentials(settings_kwargs) -> None:
    with pytest.raises(ConfigurationError, match="must be a string"):
        Settings(**settings_kwargs)


def test_direct_settings_rejects_a_non_string_api_key() -> None:
    with pytest.raises(ConfigurationError, match="CORTEX_API_KEY must be a string"):
        Settings(api_key=123)  # type: ignore[arg-type]


def test_custom_api_key_environment_falls_back_when_default_variable_is_empty() -> None:
    settings = Settings.from_mapping(
        {"server": {"api_key_env": "CORTEX_CUSTOM_KEY"}},
        environ={"CORTEX_API_KEY": "", "CORTEX_CUSTOM_KEY": "custom-secret"},
    )

    assert settings.api_key == "custom-secret"
    assert settings.api_auth_enabled is True


def test_dashboard_account_credentials_are_environment_only_and_pair_with_api_key() -> None:
    settings = Settings.from_mapping(
        {},
        environ={
            "CORTEX_API_KEY": "local-bearer-secret",
            "CORTEX_ADMIN_USERNAME": "  local-admin  ",
            "CORTEX_ADMIN_PASSWORD": "dashboard-password",
        },
    )

    assert settings.api_auth_enabled is True
    assert settings.api_key == "local-bearer-secret"
    assert settings.admin_username == "local-admin"
    assert settings.admin_password == "dashboard-password"

    with pytest.raises(ConfigurationError, match="Set both CORTEX_ADMIN_USERNAME"):
        Settings.from_mapping(
            {}, environ={"CORTEX_API_KEY": "local-secret", "CORTEX_ADMIN_USERNAME": "admin"}
        )
    with pytest.raises(ConfigurationError, match="CORTEX_API_KEY is not configured"):
        Settings.from_mapping(
            {},
            environ={"CORTEX_ADMIN_USERNAME": "admin", "CORTEX_ADMIN_PASSWORD": "password"},
        )

    account_with_auth_disabled = Settings.from_mapping(
        {},
        environ={
            "CORTEX_ADMIN_USERNAME": "admin",
            "CORTEX_ADMIN_PASSWORD": "password",
            "CORTEX_API_AUTH_ENABLED": "false",
        },
    )
    assert account_with_auth_disabled.api_key is None
    assert account_with_auth_disabled.api_auth_enabled is False
    assert account_with_auth_disabled.admin_username == "admin"


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

    with pytest.raises(ConfigurationError, match="runtime must be one of"):
        Settings.load(config, environ={})


@pytest.mark.parametrize("runtime", ("cortex", "cortex-cpp"))
def test_builtin_cortex_runtime_accepts_local_gguf_text_and_embedding_models(runtime: str) -> None:
    settings = Settings.from_mapping(
        {
            "models": [
                {
                    "id": "embedded-gguf",
                    "runtime": runtime,
                    "model_path": "/models/embedded.gguf",
                    "capabilities": ["text_generation", "embeddings"],
                }
            ]
        },
        environ={},
    )

    model = settings.models["embedded-gguf"]
    assert model.runtime == runtime
    assert model.model_path == "/models/embedded.gguf"
    assert model.embedding is True
    assert model.capabilities == ("text_generation", "embeddings")


def test_cortex_cpp_accepts_huggingface_directory_and_conversion_precision() -> None:
    settings = Settings.from_mapping(
        {
            "models": [
                {
                    "id": "hf-native",
                    "runtime": "cortex-cpp",
                    "model_path": "/models/llama-hf-checkpoint",
                    "capabilities": ["text_generation"],
                    "conversion_outtype": "q8_0",
                }
            ]
        },
        environ={},
    )

    model = settings.models["hf-native"]
    assert model.model_path == "/models/llama-hf-checkpoint"
    assert model.conversion_outtype == "q8_0"


@pytest.mark.parametrize("outtype", ("auto", "q4_0", "", 42))
def test_cortex_cpp_rejects_unsupported_conversion_outtype(outtype: object) -> None:
    with pytest.raises(ConfigurationError, match="conversion_outtype"):
        Settings.from_mapping(
            {
                "models": [
                    {
                        "id": "native",
                        "runtime": "cortex-cpp",
                        "model_path": "/models/native-source",
                        "conversion_outtype": outtype,
                    }
                ]
            },
            environ={},
        )


def test_cortex_cpp_accepts_low_bit_weight_quantization_targets() -> None:
    settings = Settings.from_mapping(
        {
            "models": [
                {
                    "id": "native-quantized",
                    "runtime": "cortex-cpp",
                    "model_path": "/models/native-source",
                    "conversion_outtype": "q4_k_m",
                    "allow_requantize": True,
                }
            ]
        },
        environ={},
    )

    model = settings.models["native-quantized"]
    assert model.conversion_outtype == "q4_k_m"
    assert model.allow_requantize is True


def test_super_low_defaults_and_custom_model_tuning_are_preserved() -> None:
    settings = Settings.from_mapping(
        {
            "models": [
                {
                    "id": "default-super-low",
                    "runtime": "cortex-cpp",
                    "model_path": "/models/tiny.gguf",
                    "optimization_preset": "SUPER LOW",
                },
                {
                    "id": "custom-super-low",
                    "runtime": "cortex-cpp",
                    "model_path": "/models/custom.gguf",
                    "optimization_preset": "super_low",
                    "gpu_layers": 3,
                    "context_size": 512,
                    "batch_size": 24,
                    "ubatch_size": 8,
                    "cache_type_k": "q8_0",
                    "cache_type_v": "f16",
                    "parallel": 1,
                    "use_mmap": False,
                    "mlock": True,
                },
            ]
        },
        environ={},
    )

    default = settings.models["default-super-low"]
    assert default.optimization_preset == "super_low"
    assert default.optimization_level == -1
    assert default.gpu_layers == -1
    assert default.context_size == 1024
    assert default.batch_size == 64
    assert default.ubatch_size == 16
    assert default.cache_type_k == default.cache_type_v == "q4_0"
    assert default.use_mmap is True
    assert default.mlock is False

    custom = settings.models["custom-super-low"]
    assert custom.gpu_layers == 3
    assert custom.context_size == 512
    assert custom.batch_size == 24
    assert custom.ubatch_size == 8
    assert custom.cache_type_k == "q8_0"
    assert custom.cache_type_v == "f16"
    assert custom.parallel == 1
    assert custom.use_mmap is False
    assert custom.mlock is True


def test_low_vram_and_native_auto_fit_are_validated_settings() -> None:
    settings = Settings.from_mapping(
        {
            "models": [
                {
                    "id": "native-fit",
                    "runtime": "cortex-cpp",
                    "model_path": "/models/native.gguf",
                    "gpu_layers": -2,
                    "optimization_preset": "low-vram",
                }
            ]
        },
        environ={},
    )

    model = settings.models["native-fit"]
    assert model.gpu_layers == -2
    assert model.optimization_preset == "low_vram"
    assert model.optimization_level == -1

    with pytest.raises(ConfigurationError, match="at least -1"):
        Settings.from_mapping(
            {
                "models": [
                    {
                        "id": "external-fit",
                        "model_path": "/models/external.gguf",
                        "gpu_layers": -2,
                    }
                ]
            },
            environ={},
        )


def test_cortex_cpp_unsafe_legacy_conversion_requires_explicit_boolean_opt_in() -> None:
    base = {
        "models": [
            {
                "id": "native",
                "runtime": "cortex-cpp",
                "model_path": "/models/legacy.ckpt",
            }
        ]
    }
    settings = Settings.from_mapping(base, environ={})
    assert settings.models["native"].allow_unsafe_legacy_checkpoint_conversion is False

    opted_in = Settings.from_mapping(
        {
            "models": [
                {
                    **base["models"][0],
                    "allow_unsafe_legacy_checkpoint_conversion": True,
                }
            ]
        },
        environ={},
    )
    assert opted_in.models["native"].allow_unsafe_legacy_checkpoint_conversion is True

    with pytest.raises(ConfigurationError, match="must be booleans"):
        Settings.from_mapping(
            {
                "models": [
                    {
                        **base["models"][0],
                        "allow_unsafe_legacy_checkpoint_conversion": "true",
                    }
                ]
            },
            environ={},
        )


@pytest.mark.parametrize(
    "extra",
    (
        {"model_path": None},
        {"model_path": "/models/not-gguf/"},
        {"model_path": "/models/vision.gguf", "mmproj_path": "/models/mmproj.gguf"},
        {"model_path": "/models/audio.gguf", "capabilities": ["audio_transcription"]},
    ),
)
def test_legacy_cortex_runtime_rejects_unsupported_model_requirements(extra) -> None:
    model = {
        "id": "embedded-gguf",
        "runtime": "cortex",
        "model_path": "/models/embedded.gguf",
        **extra,
    }
    with pytest.raises(ConfigurationError):
        Settings.from_mapping({"models": [model]}, environ={})


@pytest.mark.parametrize(
    "extra",
    (
        {"model_path": None},
        {"model_path": "/models/vision.gguf", "mmproj_path": "/models/mmproj.gguf"},
        {"model_path": "/models/audio", "capabilities": ["audio_transcription"]},
    ),
)
def test_cortex_cpp_rejects_missing_or_unsupported_model_requirements(extra) -> None:
    model = {"id": "native", "runtime": "cortex-cpp", "model_path": "/models/source", **extra}
    with pytest.raises(ConfigurationError):
        Settings.from_mapping({"models": [model]}, environ={})


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("port", True),
        ("port", 8080.5),
        ("workers", True),
        ("workers", 2.5),
        ("max_connections", True),
        ("max_connections", 10.5),
        ("max_keepalive_connections", True),
        ("max_keepalive_connections", 10.5),
        ("max_inference_requests", True),
        ("max_inference_requests", 10.5),
        ("max_json_body_bytes", True),
        ("max_json_body_bytes", 1024.5),
        ("max_media_body_bytes", True),
        ("max_media_body_bytes", 1024.5),
    ],
)
def test_server_integer_settings_reject_booleans_and_fractional_numbers(
    field: str, value: object
) -> None:
    with pytest.raises(ConfigurationError):
        Settings.from_mapping({"server": {field: value}}, environ={})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (field, 1.5)
        for field in (
            "threads",
            "threads_batch",
            "gpu_layers",
            "context_size",
            "batch_size",
            "ubatch_size",
            "parallel",
        )
    ],
)
def test_model_integer_settings_reject_fractional_numbers(field: str, value: object) -> None:
    with pytest.raises(ConfigurationError, match="must be an integer"):
        Settings.from_mapping(
            {
                "models": [
                    {
                        "id": "local",
                        "model_path": "/models/local.gguf",
                        field: value,
                    }
                ]
            },
            environ={},
        )


@pytest.mark.parametrize(
    "field",
    (
        "connect_timeout",
        "read_timeout",
        "write_timeout",
        "pool_timeout",
        "keepalive_expiry",
    ),
)
def test_timeout_settings_reject_boolean_values(field: str) -> None:
    with pytest.raises(ConfigurationError):
        Settings.from_mapping({"server": {field: True}}, environ={})


def test_invalid_optimization_level_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="optimization_level must be"):
        Settings.from_mapping(
            {
                "models": [
                    {
                        "id": "local",
                        "model_path": "/models/local.gguf",
                        "optimization_level": 2,
                    }
                ]
            },
            environ={},
        )


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("low-", "low_minus"),
        ("low−", "low_minus"),
        ("low+", "low_plus"),
        ("balanced[auto]", "balanced"),
        ("medium-", "medium_minus"),
        ("medium−", "medium_minus"),
        ("medium+", "medium_plus"),
        ("high-", "high_minus"),
        ("high−", "high_minus"),
        ("high+", "high_plus"),
        ("low-vram", "low_vram"),
        ("SUPER LOW", "super_low"),
        ("super-low", "super_low"),
        ("ultra", "ultra"),
    ],
)
def test_optimization_preset_names_and_aliases(configured: str, expected: str) -> None:
    settings = Settings.from_mapping(
        {
            "models": [
                {
                    "id": "local",
                    "model_path": "/models/local.gguf",
                    "optimization_preset": configured,
                }
            ]
        },
        environ={},
    )

    assert settings.models["local"].optimization_preset == expected


def test_legacy_optimization_levels_map_to_named_presets() -> None:
    for level, expected in [(-1, "high_plus"), (0, "balanced"), (1, "ultra")]:
        settings = Settings.from_mapping(
            {
                "models": [
                    {
                        "id": "local",
                        "model_path": "/models/local.gguf",
                        "optimization_level": level,
                    }
                ]
            },
            environ={},
        )
        assert settings.models["local"].optimization_preset == expected


def test_invalid_optimization_preset_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="optimization_preset must"):
        Settings.from_mapping(
            {
                "models": [
                    {
                        "id": "local",
                        "model_path": "/models/local.gguf",
                        "optimization_preset": "fastest-ever",
                    }
                ]
            },
            environ={},
        )


def test_local_model_path_must_be_gguf(tmp_path) -> None:
    with pytest.raises(ConfigurationError, match="must point to a GGUF file"):
        Settings.from_mapping(
            {"models": [{"id": "not-gguf", "model_path": "/models/model.bin"}]},
            environ={},
        )


def test_config_accepts_extensible_local_command_runtime_capabilities() -> None:
    settings = Settings.from_mapping(
        {
            "server": {"default_model": "local-vlm"},
            "models": [
                {
                    "id": "local-vlm",
                    "runtime": "command",
                    "model_path": "/models/vlm-folder",
                    "api_base_path": "/openai/v1",
                    "health_path": "/ready",
                    "capabilities": [
                        "text_generation",
                        "vision-understanding",
                        "audio_transcription",
                        "video-generation",
                        "future_task_v2",
                    ],
                    "runtime_command": [
                        "{python}",
                        "server.py",
                        "--model",
                        "{model_path}",
                        "--host",
                        "{host}",
                        "--port",
                        "{port}",
                    ],
                }
            ],
        },
        environ={},
    )

    model = settings.models["local-vlm"]
    assert settings.default_model == "local-vlm"
    assert model.runtime == "command"
    assert model.model_path == "/models/vlm-folder"
    assert model.api_base_path == "/openai/v1"
    assert model.health_path == "/ready"
    assert "vision_understanding" in model.effective_capabilities
    assert "video_generation" in model.effective_capabilities
    assert "future_task_v2" in model.effective_capabilities
    assert model.runtime_command[0] == "{python}"
    assert model.runtime_command[-2:] == ("--port", "{port}")


def test_effective_capabilities_reuses_declared_tuple_when_no_flags_add_tasks() -> None:
    model = ModelConfig(
        id="text-model",
        upstream_model="text-model",
        capabilities=("text_generation", "embeddings"),
    )

    assert model.effective_capabilities is model.capabilities
    assert model.effective_capabilities is model.effective_capabilities
    assert model.capability_set == frozenset(model.capabilities)
    assert model.capability_set is model.capability_set


def test_command_runtime_requires_loopback_host_and_port_placeholders() -> None:
    with pytest.raises(ConfigurationError, match="must include.*host.*port"):
        Settings.from_mapping(
            {
                "models": [
                    {
                        "id": "bad-command",
                        "runtime": "command",
                        "capabilities": ["ocr"],
                        "runtime_command": ["python", "server.py", "--port", "5000"],
                    }
                ]
            },
            environ={},
        )


def test_command_runtime_does_not_accept_remote_base_urls() -> None:
    with pytest.raises(ConfigurationError, match="base_url.*not supported"):
        Settings.from_mapping(
            {
                "models": [
                    {
                        "id": "not-a-proxy",
                        "runtime": "command",
                        "base_url": "https://example.test/v1",
                        "runtime_command": [
                            "python",
                            "server.py",
                            "--host",
                            "{host}",
                            "--port",
                            "{port}",
                        ],
                    }
                ]
            },
            environ={},
        )
