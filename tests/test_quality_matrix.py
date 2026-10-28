"""Boundary and adversarial regression matrices for security-sensitive inputs."""

from __future__ import annotations

import pytest

from cortex_llmhoster.app import SessionOriginMiddleware
from cortex_llmhoster.auth import create_session_token, valid_session_token
from cortex_llmhoster.config import ConfigurationError, Settings
from cortex_llmhoster.dashboard_models import _safe_model_filename


def _server_case(field: str, value: object, valid: bool) -> tuple[str, object, bool]:
    return field, value, valid


_SERVER_CASES = [
    *[_server_case("port", value, True) for value in (1, 8624, 65535, "9123")],
    *[
        _server_case("port", value, False)
        for value in (0, -1, 65536, True, 8080.5, "not-a-port", None, [])
    ],
    *[_server_case("workers", value, True) for value in (1, 2, 32, "3")],
    *[_server_case("workers", value, False) for value in (0, -1, True, 2.5, "many", None, [])],
    *[_server_case("max_connections", value, True) for value in (1, 8, 512, "64")],
    *[
        _server_case("max_connections", value, False)
        for value in (0, -1, True, 16.5, "many", None, [])
    ],
    *[_server_case("max_keepalive_connections", value, True) for value in (0, 1, 20, "5")],
    *[
        _server_case("max_keepalive_connections", value, False)
        for value in (-1, 21, True, 5.5, "many", None, [])
    ],
    *[_server_case("max_inference_requests", value, True) for value in (1, 8, 20, "5")],
    *[
        _server_case("max_inference_requests", value, False)
        for value in (0, 21, True, 5.5, "many", None, [])
    ],
    *[
        _server_case("max_json_body_bytes", value, True)
        for value in (1024, 4096, 16 * 1024 * 1024, "2048")
    ],
    *[
        _server_case("max_json_body_bytes", value, False)
        for value in (0, 1023, True, 1024.5, "many", None, [])
    ],
    *[
        _server_case("max_media_body_bytes", value, True)
        for value in (1024, 4096, 2 * 1024 * 1024 * 1024, "2048")
    ],
    *[
        _server_case("max_media_body_bytes", value, False)
        for value in (0, 1023, True, 1024.5, "many", None, [])
    ],
]


@pytest.mark.parametrize(("field", "value", "valid"), _SERVER_CASES)
def test_server_numeric_configuration_matrix(field: str, value: object, valid: bool) -> None:
    server: dict[str, object] = {field: value}
    if field in {"max_keepalive_connections", "max_inference_requests"}:
        server["max_connections"] = 20

    if valid:
        settings = Settings.from_mapping({"server": server}, environ={})
        assert getattr(settings, field) == int(value)
    else:
        with pytest.raises(ConfigurationError):
            Settings.from_mapping({"server": server}, environ={})


_MODEL_INTEGER_SPECS = {
    "threads": ((0, 1, 8, "2"), (-1, True, 1.5, "many", None, [])),
    "threads_batch": ((0, 1, 8, "2"), (-1, True, 1.5, "many", None, [])),
    "gpu_layers": ((-1, 0, 32, "2"), (-2, True, 1.5, "many", None, [])),
    "context_size": ((128, 129, 2048, "512"), (0, 127, True, 128.5, "many", None, [])),
    "batch_size": ((1, 2, 128, "64"), (0, True, 1.5, "many", None, [])),
    "ubatch_size": ((1, 2, 64, "32"), (0, True, 1.5, 65, "many", None, [])),
    "parallel": ((1, 2, 16, "4"), (0, True, 1.5, "many", None, [])),
}
_MODEL_CASES = [
    (field, value, valid)
    for field, (valid_values, invalid_values) in _MODEL_INTEGER_SPECS.items()
    for valid, values in ((True, valid_values), (False, invalid_values))
    for value in values
]


@pytest.mark.parametrize(("field", "value", "valid"), _MODEL_CASES)
def test_model_integer_configuration_matrix(field: str, value: object, valid: bool) -> None:
    model: dict[str, object] = {
        "id": "matrix-model",
        "model_path": "/models/matrix-model.gguf",
        field: value,
    }
    if field == "batch_size":
        model["ubatch_size"] = 1
    elif field == "ubatch_size":
        model["batch_size"] = 64

    if valid:
        settings = Settings.from_mapping({"models": [model]}, environ={})
        assert getattr(settings.models["matrix-model"], field) == int(value)
    else:
        with pytest.raises(ConfigurationError):
            Settings.from_mapping({"models": [model]}, environ={})


_BAD_PATHS = (
    "relative/path",
    "v1",
    "/v1?query=1",
    "/v1#fragment",
    "/v1\\hidden",
    "/./v1",
    "/../v1",
    "/v1/./health",
    "/v1/../health",
    None,
    123,
    True,
)
_PATH_CASES = (
    [(field, value, False) for field in ("api_base_path", "health_path") for value in _BAD_PATHS]
    + [
        (field, value, True)
        for field in ("api_base_path", "health_path")
        for value in ("/", "/v1/", "/api/v1", "/模型")
    ]
    + [("api_base_path", "", True)]
)


@pytest.mark.parametrize(("field", "value", "valid"), _PATH_CASES)
def test_local_runtime_url_path_validation_matrix(field: str, value: object, valid: bool) -> None:
    model: dict[str, object] = {
        "id": "matrix-model",
        "model_path": "/models/matrix-model.gguf",
        field: value,
    }
    if valid:
        settings = Settings.from_mapping({"models": [model]}, environ={})
        normalized = getattr(settings.models["matrix-model"], field)
        assert normalized == (value.rstrip("/") if value not in {"", "/"} else "")
    else:
        with pytest.raises(ConfigurationError):
            Settings.from_mapping({"models": [model]}, environ={})


_VALID_CAPABILITIES = (
    ("a", "a"),
    ("model", "model"),
    ("text-generation", "text_generation"),
    ("T2I", "t2i"),
    ("OCR", "ocr"),
    (" OCR ", "ocr"),
    ("future_task_9", "future_task_9"),
    ("task123", "task123"),
    ("x" * 64, "x" * 64),
    ("foo--bar", "foo__bar"),
    ("one-two-three", "one_two_three"),
    ("A1", "a1"),
    ("z_9", "z_9"),
    ("vision_understanding", "vision_understanding"),
    ("audio-generation", "audio_generation"),
    ("video_captioning", "video_captioning"),
    ("moderation", "moderation"),
    ("embeddings", "embeddings"),
)
_INVALID_CAPABILITIES = (
    "",
    " ",
    "9task",
    "-task",
    "_task",
    "task/name",
    "task name",
    "task.",
    "task@name",
    "a" * 65,
    "😊",
    "task\nname",
    None,
    7,
    {},
    True,
)


@pytest.mark.parametrize(("capability", "expected"), _VALID_CAPABILITIES)
def test_custom_capability_normalization_matrix(capability: str, expected: str) -> None:
    settings = Settings.from_mapping(
        {
            "models": [
                {
                    "id": "matrix-model",
                    "model_path": "/models/matrix-model.gguf",
                    "capabilities": [capability],
                }
            ]
        },
        environ={},
    )
    assert settings.models["matrix-model"].capabilities == (expected,)


@pytest.mark.parametrize("capability", _INVALID_CAPABILITIES)
def test_invalid_custom_capability_names_are_rejected(capability: object) -> None:
    with pytest.raises(ConfigurationError):
        Settings.from_mapping(
            {
                "models": [
                    {
                        "id": "matrix-model",
                        "model_path": "/models/matrix-model.gguf",
                        "capabilities": [capability],
                    }
                ]
            },
            environ={},
        )


def test_equivalent_capability_names_are_deduplicated_after_normalization() -> None:
    settings = Settings.from_mapping(
        {
            "models": [
                {
                    "id": "matrix-model",
                    "model_path": "/models/matrix-model.gguf",
                    "capabilities": ["OCR", "ocr", "text-generation", "text_generation"],
                }
            ]
        },
        environ={},
    )
    assert settings.models["matrix-model"].capabilities == ("ocr", "text_generation")


_VALID_MODEL_FILENAMES = (
    "model.gguf",
    "MODEL.GGUF",
    "model-q4_k_m.gguf",
    "model (2).gguf",
    "data_模型.gguf",
    ".hidden.gguf",
    "name with spaces.gguf",
    "a.b.c.gguf",
    "'quoted'.gguf",
    "x" * 235 + ".gguf",
)
_INVALID_MODEL_FILENAMES = (
    "",
    ".",
    "..",
    ".gguf",
    "model.bin",
    "../escape.gguf",
    "folder/model.gguf",
    "folder\\model.gguf",
    "C:\\model.gguf",
    " leading.gguf",
    "trailing .gguf ",
    "trailing-dot.gguf.",
    'double"quote.gguf',
    "question?.gguf",
    "star*.gguf",
    "less<.gguf",
    "greater>.gguf",
    "pipe|.gguf",
    "line\nbreak.gguf",
    "null\x00byte.gguf",
    "x" * 236 + ".gguf",
    "CON.gguf",
    "con.device.gguf",
    "PRN.gguf",
    "AUX.gguf",
    "NUL.gguf",
    "COM1.gguf",
    "COM9.gguf",
    "LPT1.gguf",
    "LPT9.gguf",
    "\ud800.gguf",
)


@pytest.mark.parametrize("filename", _VALID_MODEL_FILENAMES)
def test_safe_model_filename_acceptance_matrix(filename: str) -> None:
    assert _safe_model_filename(filename) == filename


@pytest.mark.parametrize("filename", _INVALID_MODEL_FILENAMES)
def test_safe_model_filename_rejection_matrix(filename: str) -> None:
    assert _safe_model_filename(filename) is None


@pytest.mark.parametrize("position", range(64))
def test_session_signature_tampering_is_rejected_at_every_position(position: int) -> None:
    now = 1_800_000_000
    expires, signature = create_session_token("quality-secret", now=now).split(".")
    replacement = "0" if signature[position] != "0" else "1"
    mutated = f"{expires}.{signature[:position]}{replacement}{signature[position + 1 :]}"
    assert not valid_session_token(mutated, "quality-secret", now=now)


_MALFORMED_SESSION_TOKENS = (
    "",
    " ",
    "token",
    "." + "0" * 64,
    "1.",
    "1." + "0" * 63,
    "1." + "0" * 65,
    "1." + "A" * 64,
    "1." + "g" * 64,
    "-1." + "0" * 64,
    "+1." + "0" * 64,
    "١." + "0" * 64,
    "1." + "0" * 32 + "." + "0" * 31,
    "1." + "0" * 64 + "x",
    "x" * 129,
    "1/2." + "0" * 64,
    "1e3." + "0" * 64,
    " 1." + "0" * 64,
    "1 ." + "0" * 64,
    "999999999999999999999999999999." + "0" * 64,
)


@pytest.mark.parametrize("token", _MALFORMED_SESSION_TOKENS)
def test_malformed_session_token_matrix_is_rejected(token: str) -> None:
    assert not valid_session_token(token, "quality-secret", now=1_800_000_000)


_ORIGIN_KEY_CASES = (
    ("https", "example.com", ("https", "example.com", 443)),
    ("http", "example.com", ("http", "example.com", 80)),
    ("HTTPS", "EXAMPLE.COM:443", ("https", "example.com", 443)),
    ("http", "EXAMPLE.COM:80", ("http", "example.com", 80)),
    ("https", "example.com:8443", ("https", "example.com", 8443)),
    ("http", "example.com:8080", ("http", "example.com", 8080)),
    ("https", "example.com:0", ("https", "example.com", 0)),
    ("https", "example.com.", ("https", "example.com", 443)),
    ("https", "münich.example", ("https", "xn--mnich-kva.example", 443)),
    ("http", "[::1]", ("http", "::1", 80)),
    ("https", "[::1]:9443", ("https", "::1", 9443)),
)
_INVALID_ORIGIN_KEYS = (
    ("file", "example.com"),
    ("null", "example.com"),
    ("https", ""),
    ("https", "."),
    ("https", "example.com.."),
    ("https", "user@example.com"),
    ("https", "example.com/path"),
    ("https", "example.com?query"),
    ("https", "example.com#fragment"),
    ("https", "example.com:bad"),
    ("https", "example.com:99999"),
    ("https", "[invalid"),
    ("https", "example.com with-space"),
    ("https", "example.com\\path"),
    ("https", "@example.com"),
    ("https", "example.com:"),
)


@pytest.mark.parametrize(("scheme", "authority", "expected"), _ORIGIN_KEY_CASES)
def test_origin_normalization_matrix(
    scheme: str, authority: str, expected: tuple[str, str, int]
) -> None:
    assert SessionOriginMiddleware._origin_key(scheme, authority) == expected


@pytest.mark.parametrize(("scheme", "authority"), _INVALID_ORIGIN_KEYS)
def test_malformed_origin_authority_matrix_is_rejected(scheme: str, authority: str) -> None:
    assert SessionOriginMiddleware._origin_key(scheme, authority) is None


_ORIGIN_MATCH_CASES = (
    ("https://example.com", "https", "example.com", False, True),
    ("https://EXAMPLE.COM:443", "https", "example.com", False, True),
    ("http://example.com:80", "http", "example.com", False, True),
    ("http://example.com:81", "http", "example.com", False, False),
    ("http://example.com", "https", "example.com", False, False),
    ("https://other.example", "https", "example.com", False, False),
    ("null", "https", "example.com", False, False),
    ("https://example.com/path", "https", "example.com", False, False),
    ("https://example.com/path", "https", "example.com", True, True),
    ("https://example.com/path?q=1", "https", "example.com", True, True),
    ("https://example.com#fragment", "https", "example.com", True, True),
    ("https://user@example.com", "https", "example.com", False, False),
    ("https://example.com:bad", "https", "example.com", False, False),
    ("https://example.com", "https", "example.com:443", False, True),
    ("https://example.com", "https", "example.com:444", False, False),
    ("https://example.com", "ftp", "example.com", False, False),
    ("http://[::1]", "http", "[::1]", False, True),
    ("http://[::1]:80", "http", "[::1]", False, True),
    ("http://[::1]:8080", "http", "[::1]", False, False),
    ("not a url", "https", "example.com", True, False),
)


@pytest.mark.parametrize(
    ("candidate", "scheme", "host", "allow_path", "expected"),
    _ORIGIN_MATCH_CASES,
)
def test_same_origin_comparison_matrix(
    candidate: str, scheme: str, host: str, allow_path: bool, expected: bool
) -> None:
    assert (
        SessionOriginMiddleware._matches_origin(candidate, scheme, host, allow_path=allow_path)
        is expected
    )
