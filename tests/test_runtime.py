from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from cortex_llmhoster.config import ConfigurationError, ModelConfig, Settings
from cortex_llmhoster.runtime import (
    LlamaServerManager,
    automatic_threads,
    build_llama_command,
)


def local_model(**overrides: object) -> ModelConfig:
    values: dict[str, object] = {
        "id": "tiny-local",
        "upstream_model": "tiny-local",
        "runtime": "llama.cpp",
        "model_path": "/models/tiny-q4.gguf",
        "embedding": True,
        "mmproj_path": "/models/projector.gguf",
        "gpu_layers": -1,
        "context_size": 2048,
        "batch_size": 256,
        "ubatch_size": 64,
        "use_mmap": False,
        "mlock": True,
    }
    values.update(overrides)
    return ModelConfig(**values)  # type: ignore[arg-type]


def test_automatic_threads_leave_cpu_capacity_for_cortex() -> None:
    assert automatic_threads(1) == 1
    assert automatic_threads(2) == 2
    assert automatic_threads(8) == 7


def test_build_llama_command_uses_args_and_local_low_memory_options() -> None:
    model = local_model()
    command = build_llama_command("/opt/llama-server", model, 8765, cpu_count=4)

    assert command[:3] == ["/opt/llama-server", "--model", "/models/tiny-q4.gguf"]
    assert command[command.index("--alias") + 1] == "tiny-local"
    assert "127.0.0.1" in command
    assert command[command.index("--port") + 1] == "8765"
    assert command[command.index("--threads") + 1] == "3"
    assert command[command.index("--threads-batch") + 1] == "3"
    assert command[command.index("--n-gpu-layers") + 1] == "-1"
    assert command[command.index("--ctx-size") + 1] == "2048"
    assert "--no-mmap" in command
    assert "--mlock" in command
    assert "--embedding" in command
    assert command[command.index("--mmproj") + 1] == "/models/projector.gguf"


def test_ultra_profile_tunes_cpu_and_gpu_commands_without_forcing_device_mode() -> None:
    cpu_model = local_model(optimization_profile="ultra", gpu_layers=0)
    cpu_command = build_llama_command("/opt/llama-server", cpu_model, 8765, cpu_count=4)

    assert cpu_command[cpu_command.index("--n-gpu-layers") + 1] == "0"
    assert cpu_command[cpu_command.index("--batch-size") + 1] == "512"
    assert cpu_command[cpu_command.index("--ubatch-size") + 1] == "128"
    assert cpu_command[cpu_command.index("--cache-type-k") + 1] == "q8_0"
    assert cpu_command[cpu_command.index("--cache-type-v") + 1] == "q8_0"
    assert cpu_command[cpu_command.index("--flash-attn") + 1] == "auto"

    gpu_model = local_model(optimization_profile="ultra", gpu_layers=-1, flash_attn="auto")
    gpu_command = build_llama_command("/opt/llama-server", gpu_model, 8765, cpu_count=4)
    assert gpu_command[gpu_command.index("--n-gpu-layers") + 1] == "-1"
    assert gpu_command[gpu_command.index("--flash-attn") + 1] == "on"

    compatible_override = local_model(
        optimization_profile="ultra",
        gpu_layers=4,
        flash_attn="off",
        batch_size=1024,
        ubatch_size=256,
    )
    override_command = build_llama_command(
        "/opt/llama-server", compatible_override, 8765, cpu_count=4
    )
    assert override_command[override_command.index("--flash-attn") + 1] == "off"
    assert override_command[override_command.index("--batch-size") + 1] == "1024"
    assert override_command[override_command.index("--ubatch-size") + 1] == "256"


def test_local_model_config_accepts_gguf_cpu_and_gpu_controls() -> None:
    settings = Settings.from_mapping(
        {
            "server": {"default_model": "tiny-local"},
            "models": [
                {
                    "id": "tiny-local",
                    "runtime": "llama.cpp",
                    "model_path": "/models/tiny-q4.gguf",
                    "optimization_profile": "ULTRA",
                    "gpu_layers": -1,
                    "context_size": 2048,
                    "batch_size": 256,
                    "ubatch_size": 64,
                    "embedding": True,
                    "mmproj_path": "/models/projector.gguf",
                    "default": True,
                }
            ],
        },
        environ={},
    )

    model = settings.models["tiny-local"]
    assert settings.default_model == "tiny-local"
    assert model.runtime == "llama.cpp"
    assert model.optimization_profile == "ultra"
    assert model.gpu_layers == -1
    assert model.embedding is True
    assert model.mmproj_path == "/models/projector.gguf"


def test_local_runtime_rejects_multiple_cortex_workers() -> None:
    with pytest.raises(ConfigurationError, match="one Cortex worker"):
        Settings.from_mapping(
            {
                "server": {"workers": 2},
                "models": [
                    {
                        "id": "tiny-local",
                        "runtime": "llama.cpp",
                        "model_path": "/models/tiny.gguf",
                    }
                ],
            },
            environ={},
        )


@pytest.mark.asyncio
async def test_manager_starts_and_stops_local_server(tmp_path: Path) -> None:
    model_file = tmp_path / "tiny.gguf"
    model_file.write_bytes(b"not a real model; process manager test only")
    fake_server = tmp_path / "llama-server"
    fake_server.write_text(
        "#!/usr/bin/env python3\nimport time\nprint('fake llama-server online', flush=True)\ntime.sleep(60)\n",
        encoding="utf-8",
    )
    fake_server.chmod(0o755)

    settings = Settings(
        models={
            "tiny-local": local_model(
                model_path=str(model_file),
                mmproj_path=None,
                embedding=False,
                gpu_layers=0,
            )
        },
        default_model="tiny-local",
        llama_server_path=str(fake_server),
    )

    async def fake_health(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(fake_health)) as client:
        manager = LlamaServerManager(settings, client, hardware={"cpu_count": 2})
        await manager.start_configured()
        for _ in range(60):
            if manager.state("tiny-local")["status"] == "running":
                break
            await asyncio.sleep(0.05)

        started = manager.state("tiny-local")
        assert started["status"] == "running"
        assert started["pid"] is not None
        assert started["port"] is not None
        assert started["base_url"] == f"http://127.0.0.1:{started['port']}/v1"
        assert any("fake llama-server online" in line for line in started["logs"])

        stopped = await manager.stop("tiny-local")
        assert stopped["status"] == "stopped"
        assert stopped["pid"] is None
        await manager.close()
