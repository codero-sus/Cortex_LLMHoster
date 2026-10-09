from __future__ import annotations

import json

import pytest

from cortex_llmhoster.api_config import APIFormatConfig
from cortex_llmhoster.config import ConfigurationError


def test_default_api_config_is_openai_compatible() -> None:
    config = APIFormatConfig.load()

    assert config.format == "openai"
    assert config.version == 1
    assert config.base_path == "/v1"
    assert config.models_endpoint == "/v1/models"


def test_api_config_can_customize_base_and_models_routes(tmp_path) -> None:
    path = tmp_path / "api.json"
    path.write_text(
        json.dumps(
            {
                "format": "openai",
                "version": 1,
                "base_path": "/custom/v1/",
                "models_path": "/available-models",
            }
        ),
        encoding="utf-8",
    )

    config = APIFormatConfig.load(path)

    assert config.base_path == "/custom/v1"
    assert config.models_path == "/available-models"
    assert config.models_endpoint == "/custom/v1/available-models"


def test_api_config_rejects_unsupported_format_and_reserved_routes(tmp_path) -> None:
    path = tmp_path / "api.json"
    path.write_text('{"format":"anthropic"}', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Only the 'openai' API format"):
        APIFormatConfig.load(path)

    path.write_text('{"format":"openai","base_path":"/admin/v1"}', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="reserved Cortex route"):
        APIFormatConfig.load(path)
