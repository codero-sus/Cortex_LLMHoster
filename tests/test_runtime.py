from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import httpx
import pytest

import cortex_llmhoster.runtime as runtime_module
from cortex_llmhoster.config import ConfigurationError, ModelConfig, Settings
from cortex_llmhoster.native_model import (
    build_native_conversion_command,
    build_native_quantization_command,
    cached_native_conversion,
    create_native_conversion_temp_dir,
    inspect_native_model_source,
    native_conversion_cache_dir,
    native_conversion_cache_key,
    publish_native_conversion,
)
from cortex_llmhoster.runtime import (
    LlamaServerManager,
    LocalRuntimeManager,
    RuntimeSlot,
    _available_cpu_count,
    _resolve_cortex_cpp_binary,
    _runtime_child_environment,
    automatic_threads,
    build_command_runtime_command,
    build_cortex_cpp_runtime_command,
    build_cortex_runtime_command,
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


class _ProcessThatNeverExits:
    returncode: int | None = None


def test_automatic_threads_leave_cpu_capacity_for_cortex() -> None:
    assert automatic_threads(1) == 1
    assert automatic_threads(2) == 2
    assert automatic_threads(8) == 7


def test_detect_hardware_reports_nvidia_vram_totals_and_free_memory(monkeypatch) -> None:
    monkeypatch.setattr(runtime_module.platform, "platform", lambda: "Linux test")
    monkeypatch.setattr(
        runtime_module.shutil,
        "which",
        lambda executable: "/usr/bin/nvidia-smi" if executable == "nvidia-smi" else None,
    )

    class ProbeResult:
        stdout = "NVIDIA RTX 3050, 8192, 6144\n"

    def fake_run(command: list[str], **kwargs: object) -> ProbeResult:
        assert command[1:] == [
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
        assert kwargs["timeout"] == 1.5
        return ProbeResult()

    monkeypatch.setattr(runtime_module.subprocess, "run", fake_run)

    hardware = runtime_module.detect_hardware()

    assert hardware["gpu_vram_total_gb"] == 8.0
    assert hardware["gpu_vram_free_gb"] == 6.0
    assert "RTX 3050" in hardware["gpu"]
    assert "6144/8192 MiB free" in hardware["gpu"]


def test_automatic_threads_caches_os_and_container_probes(monkeypatch) -> None:
    _available_cpu_count.cache_clear()
    calls = {"cpu": 0, "affinity": 0}

    def cpu_count() -> int:
        calls["cpu"] += 1
        return 8

    def affinity(_: int) -> set[int]:
        calls["affinity"] += 1
        return set(range(8))

    def missing_cgroup_file(*_: object, **__: object) -> str:
        raise OSError("no cgroup file")

    monkeypatch.setattr("cortex_llmhoster.runtime.os.cpu_count", cpu_count)
    monkeypatch.setattr("cortex_llmhoster.runtime.os.sched_getaffinity", affinity, raising=False)
    monkeypatch.setattr("cortex_llmhoster.runtime.Path.read_text", missing_cgroup_file)
    try:
        assert automatic_threads() == 7
        assert automatic_threads() == 7
        assert calls == {"cpu": 1, "affinity": 1}
    finally:
        _available_cpu_count.cache_clear()


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


@pytest.mark.asyncio
async def test_cortex_runtime_check_does_not_require_llama_server(tmp_path, monkeypatch) -> None:
    model_file = tmp_path / "embedded.gguf"
    model_file.write_bytes(b"GGUF fixture")
    model = ModelConfig(
        id="embedded",
        upstream_model="embedded",
        runtime="cortex",
        model_path=str(model_file),
    )
    async with httpx.AsyncClient() as client:
        manager = LocalRuntimeManager(Settings(models={model.id: model}), client, hardware={})
        monkeypatch.setattr(
            manager, "_embedded_python_available", lambda: ("/venv/bin/python", True)
        )
        result = manager.test(model)

    assert result["ok"] is True
    assert result["runner"] == "/venv/bin/python"
    assert "llama-server" not in result["message"]


def test_cortex_runtime_command_launches_its_own_python_backend_without_llama_server(
    tmp_path, monkeypatch
) -> None:
    model_file = tmp_path / "embedded.gguf"
    model_file.write_bytes(b"GGUF fixture")
    monkeypatch.setattr(
        "cortex_llmhoster.runtime.resolve_python_executable", lambda: "/venv/bin/python"
    )
    model = ModelConfig(
        id="embedded",
        upstream_model="embedded-alias",
        runtime="cortex",
        model_path=str(model_file),
        api_base_path="/v1",
        health_path="/health",
        threads=4,
        threads_batch=3,
        gpu_layers=-1,
        embedding=True,
        capabilities=("text_generation", "embeddings"),
        use_mmap=False,
        mlock=True,
    )

    command = build_cortex_runtime_command(model, 8765)

    assert command[:3] == ["/venv/bin/python", "-m", "cortex_llmhoster.embedded_backend"]
    assert command[command.index("--model-path") + 1] == str(model_file.resolve())
    assert command[command.index("--model-alias") + 1] == "embedded-alias"
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "8765"
    assert "--embedding" in command
    assert "--no-mmap" in command
    assert "--mlock" in command
    assert "llama-server" not in command


def test_cortex_legacy_super_low_profile_preserves_custom_context_and_batch(
    tmp_path: Path, monkeypatch
) -> None:
    model_file = tmp_path / "embedded.gguf"
    model_file.write_bytes(b"GGUF fixture")
    monkeypatch.setattr(
        "cortex_llmhoster.runtime.resolve_python_executable", lambda: "/venv/bin/python"
    )
    model = ModelConfig(
        id="embedded-super-low",
        upstream_model="embedded-super-low",
        runtime="cortex",
        model_path=str(model_file),
        optimization_preset="super_low",
        context_size=4096,
        batch_size=512,
        use_mmap=False,
        mlock=True,
    )

    command = build_cortex_runtime_command(model, 8765)

    assert command[command.index("--context-size") + 1] == "4096"
    assert command[command.index("--batch-size") + 1] == "512"
    assert "--no-mmap" in command
    assert "--mlock" in command


def test_optimization_presets_balance_quality_and_memory_modes() -> None:
    low = build_llama_command(
        "/opt/llama-server", local_model(optimization_preset="low_minus"), 8765, cpu_count=4
    )
    assert low[low.index("--batch-size") + 1] == "1024"
    assert low[low.index("--ubatch-size") + 1] == "256"
    assert low[low.index("--cache-type-k") + 1] == "f16"

    low_plus = build_llama_command(
        "/opt/llama-server", local_model(optimization_preset="low_plus"), 8765, cpu_count=4
    )
    assert low_plus[low_plus.index("--batch-size") + 1] == "768"
    assert low_plus[low_plus.index("--cache-type-v") + 1] == "f16"

    balanced = build_llama_command(
        "/opt/llama-server",
        local_model(optimization_level=1, optimization_preset="balanced"),
        8765,
        cpu_count=4,
    )
    assert balanced[balanced.index("--batch-size") + 1] == "256"
    assert balanced[balanced.index("--ubatch-size") + 1] == "64"
    assert balanced[balanced.index("--cache-type-k") + 1] == "f16"

    medium_minus = build_llama_command(
        "/opt/llama-server", local_model(optimization_preset="medium_minus"), 8765, cpu_count=4
    )
    medium_plus = build_llama_command(
        "/opt/llama-server", local_model(optimization_preset="medium_plus"), 8765, cpu_count=4
    )
    assert medium_minus[medium_minus.index("--batch-size") + 1] == "384"
    assert medium_minus[medium_minus.index("--cache-type-k") + 1] == "f16"
    assert medium_plus[medium_plus.index("--batch-size") + 1] == "384"
    assert medium_plus[medium_plus.index("--cache-type-k") + 1] == "q8_0"

    high_minus = build_llama_command(
        "/opt/llama-server", local_model(optimization_preset="high_minus"), 8765, cpu_count=4
    )
    high_plus = build_llama_command(
        "/opt/llama-server", local_model(optimization_preset="high_plus"), 8765, cpu_count=4
    )
    assert high_minus[high_minus.index("--batch-size") + 1] == "256"
    assert high_minus[high_minus.index("--cache-type-k") + 1] == "q8_0"
    assert high_plus[high_plus.index("--batch-size") + 1] == "128"
    assert high_plus[high_plus.index("--ubatch-size") + 1] == "32"
    assert high_plus[high_plus.index("--cache-type-v") + 1] == "q8_0"


def test_low_vram_profile_is_conservative_for_llama_server() -> None:
    command = build_llama_command(
        "/opt/llama-server",
        local_model(
            optimization_preset="low_vram",
            context_size=8192,
            batch_size=1024,
            ubatch_size=256,
            parallel=4,
            use_mmap=False,
            mlock=True,
            flash_attn="auto",
        ),
        8765,
        cpu_count=4,
    )

    assert command[command.index("--ctx-size") + 1] == "2048"
    assert command[command.index("--batch-size") + 1] == "128"
    assert command[command.index("--ubatch-size") + 1] == "32"
    assert command[command.index("--parallel") + 1] == "1"
    assert command[command.index("--cache-type-k") + 1] == "q8_0"
    assert command[command.index("--cache-type-v") + 1] == "q8_0"
    assert command[command.index("--flash-attn") + 1] == "on"
    assert command[command.index("--n-gpu-layers") + 1] == "-1"
    assert "--no-mmap" not in command
    assert "--mlock" not in command


def test_super_low_llama_profile_preserves_custom_advanced_settings() -> None:
    command = build_llama_command(
        "/opt/llama-server",
        local_model(
            optimization_preset="super_low",
            context_size=4096,
            batch_size=512,
            ubatch_size=128,
            parallel=4,
            gpu_layers=5,
            flash_attn="auto",
            cache_type_k="q8_0",
            cache_type_v="f16",
            use_mmap=False,
            mlock=True,
        ),
        8765,
        cpu_count=4,
    )

    assert command[command.index("--ctx-size") + 1] == "4096"
    assert command[command.index("--batch-size") + 1] == "512"
    assert command[command.index("--ubatch-size") + 1] == "128"
    assert command[command.index("--parallel") + 1] == "4"
    assert command[command.index("--n-gpu-layers") + 1] == "5"
    assert command[command.index("--flash-attn") + 1] == "on"
    assert command[command.index("--cache-type-k") + 1] == "q8_0"
    assert command[command.index("--cache-type-v") + 1] == "f16"
    assert "--no-mmap" in command
    assert "--mlock" in command


def test_ultra_is_opt_in_and_legacy_levels_are_mapped() -> None:
    cpu_model = local_model(optimization_level=1)
    cpu_command = build_llama_command("/opt/llama-server", cpu_model, 8765, cpu_count=4)
    assert cpu_command[cpu_command.index("--n-gpu-layers") + 1] == "-1"
    assert cpu_command[cpu_command.index("--batch-size") + 1] == "512"
    assert cpu_command[cpu_command.index("--ubatch-size") + 1] == "128"
    assert cpu_command[cpu_command.index("--cache-type-k") + 1] == "q8_0"

    gpu_command = build_llama_command(
        "/opt/llama-server",
        local_model(optimization_preset="ultra", gpu_layers=4, flash_attn="auto"),
        8765,
        cpu_count=4,
    )
    assert gpu_command[gpu_command.index("--n-gpu-layers") + 1] == "4"
    assert gpu_command[gpu_command.index("--flash-attn") + 1] == "on"

    override_command = build_llama_command(
        "/opt/llama-server",
        local_model(
            optimization_preset="ultra",
            gpu_layers=0,
            flash_attn="off",
            batch_size=1024,
            ubatch_size=256,
        ),
        8765,
        cpu_count=4,
    )
    assert override_command[override_command.index("--flash-attn") + 1] == "off"
    assert override_command[override_command.index("--batch-size") + 1] == "1024"
    assert override_command[override_command.index("--ubatch-size") + 1] == "256"

    legacy_efficient = build_llama_command(
        "/opt/llama-server", local_model(optimization_level=-1), 8765, cpu_count=4
    )
    assert legacy_efficient[legacy_efficient.index("--batch-size") + 1] == "128"
    assert legacy_efficient[legacy_efficient.index("--cache-type-k") + 1] == "q8_0"


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
    assert model.optimization_level == 1
    assert model.optimization_preset == "ultra"
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
async def test_runtime_output_reader_handles_unterminated_large_lines() -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        'import sys; sys.stdout.write("x" * 200000); sys.stdout.flush()',
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    slot = RuntimeSlot(model=local_model(), process=process)
    async with httpx.AsyncClient() as client:
        manager = LocalRuntimeManager(Settings(), client, hardware={})
        await asyncio.wait_for(manager._read_output(slot), timeout=5)
    await process.wait()

    assert slot.output
    assert len(slot.output) <= 40
    assert all(len(line) <= 500 for line in slot.output)
    assert all(line for line in slot.output)


@pytest.mark.asyncio
async def test_runtime_start_turns_symlink_loops_into_a_safe_error(tmp_path: Path) -> None:
    loop = tmp_path / "loop"
    try:
        loop.symlink_to(loop)
    except OSError as exc:
        pytest.skip(f"Symlinks are unavailable: {exc}")

    model = ModelConfig(
        id="loop-model",
        upstream_model="loop-model",
        runtime="command",
        model_path=str(loop),
        runtime_command=(sys.executable, "-c", "pass", "--host", "{host}", "--port", "{port}"),
    )
    async with httpx.AsyncClient() as client:
        manager = LocalRuntimeManager(Settings(models={model.id: model}), client, hardware={})
        state = await manager.start(model.id)
        await manager.close()

    assert state["status"] == "error"
    assert "Could not resolve local model path" in state["error"]


@pytest.mark.asyncio
async def test_manager_shutdown_stops_all_runtime_slots_concurrently(monkeypatch) -> None:
    async with httpx.AsyncClient() as client:
        manager = LocalRuntimeManager(Settings(), client, hardware={})
        manager._slots = {
            "first": RuntimeSlot(model=local_model(id="first")),
            "second": RuntimeSlot(model=local_model(id="second")),
        }
        started: set[str] = set()
        all_started = asyncio.Event()
        release = asyncio.Event()

        async def stop(model_id: str) -> dict[str, str]:
            started.add(model_id)
            if len(started) == 2:
                all_started.set()
            await release.wait()
            return {"id": model_id, "status": "stopped"}

        monkeypatch.setattr(manager, "stop", stop)
        closing = asyncio.create_task(manager.close())
        await asyncio.wait_for(all_started.wait(), timeout=1)
        assert not closing.done()
        release.set()
        await closing

    assert started == {"first", "second"}


@pytest.mark.asyncio
async def test_monitor_marks_unhealthy_runtime_unavailable_and_recovers() -> None:
    model = ModelConfig(id="health-model", upstream_model="health-model")
    slot = RuntimeSlot(model=model, status="running", port=12345)
    slot.process = _ProcessThatNeverExits()  # type: ignore[assignment]
    health_calls = 0

    async def health_response(_: httpx.Request) -> httpx.Response:
        nonlocal health_calls
        health_calls += 1
        return httpx.Response(503 if health_calls <= 3 else 200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(health_response)) as client:
        manager = LocalRuntimeManager(Settings(models={model.id: model}), client, hardware={})
        manager._slots[model.id] = slot
        monitor = asyncio.create_task(manager._monitor(model.id, slot))
        for _ in range(80):
            if slot.status == "starting" and slot.error:
                break
            await asyncio.sleep(0.025)
        assert slot.status == "starting"
        assert "health check failed" in str(slot.error).lower()
        assert manager.local_base_url(model) is None

        for _ in range(80):
            if slot.status == "running" and health_calls >= 4:
                break
            await asyncio.sleep(0.025)
        monitor.cancel()
        await asyncio.gather(monitor, return_exceptions=True)

    assert slot.status == "running"
    assert slot.error is None
    assert manager.local_base_url(model) == "http://127.0.0.1:12345/v1"


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


def test_runtime_children_do_not_inherit_cortex_authentication_secrets(monkeypatch) -> None:
    monkeypatch.setenv("CORTEX_API_KEY", "dashboard-key")
    monkeypatch.setenv("CORTEX_ADMIN_PASSWORD", "dashboard-password")
    monkeypatch.setenv("CORTEX_RUNTIME_KEY", "custom-dashboard-key")
    settings = Settings(
        api_key="dashboard-key",
        api_key_env="CORTEX_RUNTIME_KEY",
        admin_username="admin",
        admin_password="dashboard-password",
    )

    child_environment = _runtime_child_environment(settings)

    assert "CORTEX_API_KEY" not in child_environment
    assert "CORTEX_ADMIN_PASSWORD" not in child_environment
    assert "CORTEX_RUNTIME_KEY" not in child_environment
    assert child_environment.get("PATH") == os.environ.get("PATH")


def test_command_runtime_templates_are_argv_and_force_loopback(tmp_path: Path) -> None:
    model = ModelConfig(
        id="vlm-public",
        upstream_model="vlm-engine-name",
        runtime="command",
        model_path=str(tmp_path),
        runtime_command=(
            sys.executable,
            "serve.py",
            "--model",
            "{model_path}",
            "--served-name",
            "{model_alias}",
            "--host",
            "{host}",
            "--port",
            "{port}",
        ),
    )

    command = build_command_runtime_command(model, 8765)

    assert command[0] == sys.executable
    assert command[command.index("--model") + 1] == str(tmp_path.resolve())
    assert command[command.index("--served-name") + 1] == "vlm-engine-name"
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "8765"


def test_command_runtime_uses_selected_python_placeholder(monkeypatch) -> None:
    selected_python = "/portable/python/bin/python3"
    monkeypatch.setattr(
        "cortex_llmhoster.runtime.resolve_python_executable", lambda: selected_python
    )
    model = ModelConfig(
        id="python-command",
        upstream_model="python-engine",
        runtime="command",
        runtime_command=(
            "{python}",
            "serve.py",
            "--host",
            "{host}",
            "--port",
            "{port}",
        ),
    )

    command = build_command_runtime_command(model, 8765)

    assert command == [
        selected_python,
        "serve.py",
        "--host",
        "127.0.0.1",
        "--port",
        "8765",
    ]


def test_command_runtime_skips_unused_path_and_thread_resolution(
    monkeypatch, tmp_path: Path
) -> None:
    model = ModelConfig(
        id="lightweight-command",
        upstream_model="engine-name",
        runtime="command",
        model_path=str(tmp_path / "unused-model-path"),
        runtime_command=(
            sys.executable,
            "serve.py",
            "--host",
            "{host}",
            "--port",
            "{port}",
        ),
    )

    def unexpected_probe(*_: object, **__: object) -> object:
        raise AssertionError("unused runtime placeholders must not trigger expensive probes")

    monkeypatch.setattr("cortex_llmhoster.runtime.automatic_threads", unexpected_probe)
    monkeypatch.setattr("cortex_llmhoster.runtime.Path.resolve", unexpected_probe)

    command = build_command_runtime_command(model, 8765)

    assert command == [sys.executable, "serve.py", "--host", "127.0.0.1", "--port", "8765"]


@pytest.mark.asyncio
async def test_local_command_runtime_is_started_health_checked_and_stopped(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    server_script = tmp_path / "local_api.py"
    server_script.write_text(
        """from http.server import BaseHTTPRequestHandler, HTTPServer
import sys

args = sys.argv
host = args[args.index('--host') + 1]
port = int(args[args.index('--port') + 1])

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == '/health' else 404)
        self.end_headers()
        self.wfile.write(b'ok')
    def log_message(self, *_args):
        pass

HTTPServer((host, port), Handler).serve_forever()
""",
        encoding="utf-8",
    )
    model = ModelConfig(
        id="ocr-local",
        upstream_model="ocr-engine",
        runtime="command",
        model_path=str(model_dir),
        capabilities=("ocr", "vision_understanding"),
        api_base_path="/openai/v1",
        health_path="/health",
        runtime_command=(
            sys.executable,
            str(server_script),
            "--host",
            "{host}",
            "--port",
            "{port}",
        ),
    )
    settings = Settings(models={model.id: model}, default_model=model.id)

    async with httpx.AsyncClient(timeout=2.0) as client:
        manager = LocalRuntimeManager(settings, client)
        await manager.start_configured()
        for _ in range(60):
            if manager.state(model.id)["status"] == "running":
                break
            await asyncio.sleep(0.05)

        started = manager.state(model.id)
        assert started["status"] == "running"
        assert started["runtime"] == "command"
        assert started["capabilities"] == ["ocr", "vision_understanding"]
        assert started["base_url"] == f"http://127.0.0.1:{started['port']}/openai/v1"
        assert started["command"][started["command"].index("--host") + 1] == "127.0.0.1"
        assert (await client.get(f"http://127.0.0.1:{started['port']}/health")).status_code == 200

        stopped = await manager.stop(model.id)
        assert stopped["status"] == "stopped"
        await manager.close()


@pytest.mark.asyncio
async def test_cortex_backend_is_supervised_without_external_server_binary(
    tmp_path: Path, monkeypatch
) -> None:
    model_file = tmp_path / "cortex.gguf"
    model_file.write_bytes(b"GGUF fixture")
    model = ModelConfig(
        id="cortex-local",
        upstream_model="cortex-alias",
        runtime="cortex",
        model_path=str(model_file),
    )
    settings = Settings(models={model.id: model}, default_model=model.id)
    observed: dict[str, object] = {}

    class FakeProcess:
        returncode: int | None = None
        pid = 4321
        stdout = None

        def terminate(self) -> None:
            self.returncode = 0

        def kill(self) -> None:
            self.returncode = -9

        async def wait(self) -> int:
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

    async def fake_create_process(*args: object, **kwargs: object) -> FakeProcess:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(
        "cortex_llmhoster.runtime.resolve_python_executable", lambda: "/venv/bin/python"
    )
    monkeypatch.setattr(
        "cortex_llmhoster.runtime.asyncio.create_subprocess_exec", fake_create_process
    )

    async with httpx.AsyncClient() as client:
        manager = LocalRuntimeManager(settings, client, hardware={})
        monkeypatch.setattr(manager, "_free_port", lambda: 9234)

        async def monitor_forever(*_: object) -> None:
            await asyncio.Event().wait()

        monkeypatch.setattr(manager, "_monitor", monitor_forever)
        await manager.start_configured()

        state = manager.state(model.id)
        command = observed["args"]
        assert state["runtime"] == "cortex"
        assert state["status"] == "starting"
        assert command[0] == "/venv/bin/python"
        assert command[2] == "cortex_llmhoster.embedded_backend"
        assert "llama-server" not in command
        assert observed["kwargs"]["cwd"] == str(tmp_path)

        stopped = await manager.stop(model.id)
        assert stopped["status"] == "stopped"
        await manager.close()


def test_llama_server_resolution_falls_back_to_path_when_unconfigured(monkeypatch) -> None:
    manager = object.__new__(LocalRuntimeManager)
    manager.settings = Settings()
    monkeypatch.setattr(
        "cortex_llmhoster.runtime.shutil.which",
        lambda executable: "/usr/bin/llama-server" if executable == "llama-server" else None,
    )

    assert manager._resolve_binary() == "/usr/bin/llama-server"


def test_cortex_cpp_binary_resolution_honors_override_and_build_tree(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("CORTEX_CPP_HOST", "/opt/custom-cortex-gguf-host")
    monkeypatch.setattr(
        "cortex_llmhoster.runtime._resolve_executable",
        lambda _executable: "/opt/custom-cortex-gguf-host",
    )
    assert _resolve_cortex_cpp_binary() == "/opt/custom-cortex-gguf-host"

    monkeypatch.delenv("CORTEX_CPP_HOST")
    checkout = tmp_path / "checkout"
    runtime_file = checkout / "src" / "cortex_llmhoster" / "runtime.py"
    runtime_file.parent.mkdir(parents=True)
    runtime_file.touch()
    executable = checkout / "native" / "build" / "bin" / "cortex-gguf-host"
    executable.parent.mkdir(parents=True)
    executable.touch()
    executable.chmod(0o755)
    monkeypatch.setattr(runtime_module, "__file__", str(runtime_file))

    assert _resolve_cortex_cpp_binary() == str(executable)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runtime", "binary_name"),
    (("llama.cpp", "llama-server"), ("cortex-cpp", "cortex-gguf-host")),
)
async def test_local_runtime_auto_installs_missing_release_binary(
    runtime: str, binary_name: str, tmp_path: Path, monkeypatch
) -> None:
    model_file = tmp_path / "tiny.gguf"
    model_file.write_bytes(b"GGUF fixture")
    if runtime == "llama.cpp":
        model = local_model(
            id="auto-llama",
            runtime=runtime,
            model_path=str(model_file),
            mmproj_path=None,
            embedding=False,
        )
    else:
        model = ModelConfig(
            id="auto-cortex-cpp",
            upstream_model="auto-cortex-cpp",
            runtime=runtime,
            model_path=str(model_file),
        )
    settings = Settings(models={model.id: model}, default_model=model.id)
    observed: dict[str, object] = {}

    class FakeProcess:
        returncode: int | None = None
        pid = 7442
        stdout = None

        def terminate(self) -> None:
            self.returncode = 0

        def kill(self) -> None:
            self.returncode = -9

        async def wait(self) -> int:
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

    async def fake_create_process(*args: object, **kwargs: object) -> FakeProcess:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return FakeProcess()

    def fake_install(requested_binary: str) -> Path:
        observed["installed"] = requested_binary
        return Path(f"/runtime-cache/{requested_binary}")

    monkeypatch.delenv("CORTEX_AUTO_INSTALL_RUNTIMES", raising=False)
    monkeypatch.delenv("CORTEX_CPP_HOST", raising=False)
    monkeypatch.delenv("CORTEX_CPP_QUANTIZE", raising=False)
    monkeypatch.setattr(runtime_module, "install_latest_runtime_binary", fake_install)
    monkeypatch.setattr(
        "cortex_llmhoster.runtime.asyncio.create_subprocess_exec", fake_create_process
    )
    if runtime == "llama.cpp":
        monkeypatch.setattr(LocalRuntimeManager, "_resolve_binary", lambda _self: None)
    else:
        monkeypatch.setattr(runtime_module, "_resolve_cortex_cpp_binary", lambda: None)

    async with httpx.AsyncClient() as client:
        manager = LocalRuntimeManager(settings, client, hardware={})
        monkeypatch.setattr(manager, "_free_port", lambda: 9236)

        async def monitor_forever(*_: object) -> None:
            await asyncio.Event().wait()

        monkeypatch.setattr(manager, "_monitor", monitor_forever)
        state = await manager.start(model.id)
        assert observed["installed"] == binary_name
        assert state["status"] == "starting"
        assert observed["args"][0] == f"/runtime-cache/{binary_name}"
        await manager.stop(model.id)
        await manager.close()


def test_native_cpp_source_detection_accepts_complete_split_gguf(tmp_path: Path) -> None:
    model_directory = tmp_path / "split-model"
    model_directory.mkdir()
    shards = [model_directory / f"model-{index:05d}-of-00003.gguf" for index in range(1, 4)]
    for index, shard in enumerate(shards, start=1):
        shard.write_bytes(f"shard {index}".encode())

    source = inspect_native_model_source(model_directory)
    assert source.kind == "gguf"
    assert source.source_path == model_directory
    assert source.gguf_path == shards[0]

    selected_second_shard = inspect_native_model_source(shards[1])
    assert selected_second_shard.source_path == model_directory
    assert selected_second_shard.gguf_path == shards[0]


def test_native_cpp_source_detection_rejects_incomplete_split_gguf(tmp_path: Path) -> None:
    from cortex_llmhoster.native_model import NativeModelSourceError

    model_directory = tmp_path / "incomplete-model"
    model_directory.mkdir()
    (model_directory / "model-00001-of-00002.gguf").write_bytes(b"first shard")

    with pytest.raises(NativeModelSourceError, match="shard set.*incomplete"):
        inspect_native_model_source(model_directory)


def test_native_cpp_model_source_detection_and_converter_commands(tmp_path: Path) -> None:
    from cortex_llmhoster.native_model import NativeModelSourceError

    gguf_path = tmp_path / "direct.gguf"
    gguf_path.write_bytes(b"GGUF fixture")
    assert inspect_native_model_source(gguf_path).kind == "gguf"

    hf_directory = tmp_path / "hf-model"
    hf_directory.mkdir()
    (hf_directory / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    (hf_directory / "model.safetensors").write_bytes(b"weights")
    source = inspect_native_model_source(hf_directory)
    assert source.kind == "huggingface"
    assert not source.requires_unsafe_pickle
    assert (
        inspect_native_model_source(hf_directory / "model.safetensors").source_path == hf_directory
    )

    hf_bin_directory = tmp_path / "hf-bin-model"
    hf_bin_directory.mkdir()
    (hf_bin_directory / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    (hf_bin_directory / "pytorch_model.bin").write_bytes(b"weights")
    hf_bin_source = inspect_native_model_source(hf_bin_directory)
    assert hf_bin_source.kind == "huggingface"
    assert not hf_bin_source.requires_unsafe_pickle

    legacy_directory = tmp_path / "legacy-llama"
    legacy_directory.mkdir()
    (legacy_directory / "consolidated.00.pth").write_bytes(b"checkpoint")
    legacy_source = inspect_native_model_source(legacy_directory)
    assert legacy_source.kind == "legacy-llama"
    assert legacy_source.requires_unsafe_pickle
    safe_legacy_file = tmp_path / "consolidated.safetensors"
    safe_legacy_file.write_bytes(b"safe tensor fixture")
    assert not inspect_native_model_source(safe_legacy_file).requires_unsafe_pickle
    checkpoint = tmp_path / "legacy.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_source = inspect_native_model_source(checkpoint)
    assert checkpoint_source.kind == "legacy-llama"
    assert checkpoint_source.requires_unsafe_pickle
    checkpoint_cache_key = native_conversion_cache_key(checkpoint_source, "f16")
    (tmp_path / "tokenizer.model").write_bytes(b"tokenizer revision")
    assert native_conversion_cache_key(checkpoint_source, "f16") != checkpoint_cache_key

    converter_source = tmp_path / "llama.cpp"
    converter = converter_source / "convert_hf_to_gguf.py"
    converter.parent.mkdir(parents=True)
    converter.touch()
    output_dir = tmp_path / "conversion-output"
    command = build_native_conversion_command(
        source,
        converter_source,
        output_dir,
        "/opt/converter-python",
        "q8_0",
        "hf-alias",
    )
    assert command[0] == "/opt/converter-python"
    assert command[1] == str(converter)
    assert command[command.index("--outtype") + 1] == "q8_0"
    assert command[command.index("--model-name") + 1] == "hf-alias"
    assert command[command.index("--split-max-size") + 1] == "4G"
    assert command[-1] == str(hf_directory)

    quantized_path = output_dir / "cortex-model.q4_k_m.gguf"
    quantize_command = build_native_quantization_command(
        "/opt/llama-quantize",
        output_dir / "cortex-model.f16.gguf",
        quantized_path,
        "q4_k_m",
        allow_requantize=True,
    )
    assert quantize_command == [
        "/opt/llama-quantize",
        "--allow-requantize",
        str(output_dir / "cortex-model.f16.gguf"),
        str(quantized_path),
        "Q4_K_M",
    ]

    unknown = tmp_path / "model.onnx"
    unknown.write_bytes(b"not a cortex-cpp format")
    with pytest.raises(NativeModelSourceError, match="accepts GGUF"):
        inspect_native_model_source(unknown)


def test_native_cpp_conversion_cache_publishes_only_complete_gguf(
    tmp_path: Path, monkeypatch
) -> None:
    model_directory = tmp_path / "hf-model"
    model_directory.mkdir()
    (model_directory / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    (model_directory / "model.safetensors").write_bytes(b"weights")
    source = inspect_native_model_source(model_directory)
    monkeypatch.setenv("CORTEX_CPP_CACHE_DIR", str(tmp_path / "cache"))
    key = native_conversion_cache_key(source, "f16")
    cache_dir = native_conversion_cache_dir(key)
    temp_dir = create_native_conversion_temp_dir()
    produced = temp_dir / "cortex-model.f16-00001-of-00002.gguf"
    produced.write_bytes(b"converted shard")

    published = publish_native_conversion(temp_dir, cache_dir, key, "f16")

    assert published.name == produced.name
    assert published.is_file()
    assert cached_native_conversion(cache_dir, key, "f16") == published
    assert cached_native_conversion(cache_dir, "different-key", "f16") is None


def test_cortex_cpp_runtime_command_is_native_and_applies_gguf_tuning(
    tmp_path: Path, monkeypatch
) -> None:
    model_file = tmp_path / "native.gguf"
    model_file.write_bytes(b"GGUF fixture")
    monkeypatch.setattr("cortex_llmhoster.runtime.automatic_threads", lambda: 6)
    model = ModelConfig(
        id="native-local",
        upstream_model="native-alias",
        runtime="cortex-cpp",
        model_path=str(model_file),
        threads=0,
        threads_batch=3,
        gpu_layers=-1,
        context_size=4096,
        batch_size=512,
        ubatch_size=128,
        flash_attn="on",
        cache_type_k="q8_0",
        cache_type_v="q8_0",
        use_mmap=False,
        mlock=True,
        embedding=True,
    )

    command = build_cortex_cpp_runtime_command(model, 8765, "/opt/cortex-gguf-host")

    assert command[0] == "/opt/cortex-gguf-host"
    assert command[command.index("--model-path") + 1] == str(model_file.resolve())
    assert command[command.index("--model-alias") + 1] == "native-alias"
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "8765"
    assert command[command.index("--threads") + 1] == "6"
    assert command[command.index("--threads-batch") + 1] == "3"
    assert command[command.index("--gpu-layers") + 1] == "-1"
    assert command[command.index("--context-size") + 1] == "4096"
    assert command[command.index("--batch-size") + 1] == "512"
    assert command[command.index("--ubatch-size") + 1] == "128"
    assert command[command.index("--cache-type-k") + 1] == "q8_0"
    assert "--no-mmap" in command
    assert "--mlock" in command
    assert "--embedding" in command
    assert "llama-server" not in command
    assert "python" not in command[0].lower()


def test_cortex_cpp_low_vram_profile_uses_native_memory_fit(tmp_path: Path) -> None:
    model_file = tmp_path / "native.gguf"
    model_file.write_bytes(b"GGUF fixture")
    model = ModelConfig(
        id="native-low-vram",
        upstream_model="native-low-vram",
        runtime="cortex-cpp",
        model_path=str(model_file),
        optimization_preset="low_vram",
        gpu_layers=-1,
        context_size=8192,
        batch_size=1024,
        ubatch_size=256,
        parallel=4,
        flash_attn="auto",
        cache_type_k="f16",
        cache_type_v="f16",
        use_mmap=False,
        mlock=True,
    )

    command = build_cortex_cpp_runtime_command(model, 8765, "/opt/cortex-gguf-host")

    assert command[command.index("--gpu-layers") + 1] == "-2"
    assert command[command.index("--context-size") + 1] == "2048"
    assert command[command.index("--batch-size") + 1] == "128"
    assert command[command.index("--ubatch-size") + 1] == "32"
    assert command[command.index("--cache-type-k") + 1] == "q8_0"
    assert command[command.index("--cache-type-v") + 1] == "q8_0"
    assert command[command.index("--flash-attn") + 1] == "on"
    assert "--no-mmap" not in command
    assert "--mlock" not in command


def test_cortex_cpp_super_low_profile_uses_auto_fit_and_q4_cache(tmp_path: Path) -> None:
    model_file = tmp_path / "super-low.gguf"
    model_file.write_bytes(b"GGUF fixture")
    model = Settings.from_mapping(
        {
            "models": [
                {
                    "id": "native-super-low",
                    "runtime": "cortex-cpp",
                    "model_path": str(model_file),
                    "optimization_preset": "super_low",
                }
            ]
        },
        environ={},
    ).models["native-super-low"]

    command = build_cortex_cpp_runtime_command(model, 8765, "/opt/cortex-gguf-host")

    assert command[command.index("--gpu-layers") + 1] == "-2"
    assert command[command.index("--context-size") + 1] == "1024"
    assert command[command.index("--batch-size") + 1] == "64"
    assert command[command.index("--ubatch-size") + 1] == "16"
    assert command[command.index("--cache-type-k") + 1] == "q4_0"
    assert command[command.index("--cache-type-v") + 1] == "q4_0"
    assert command[command.index("--flash-attn") + 1] == "on"


def test_cortex_cpp_super_low_respects_custom_parameters(tmp_path: Path) -> None:
    model_file = tmp_path / "custom-super-low.gguf"
    model_file.write_bytes(b"GGUF fixture")
    model = ModelConfig(
        id="native-super-low-custom",
        upstream_model="native-super-low-custom",
        runtime="cortex-cpp",
        model_path=str(model_file),
        optimization_preset="super_low",
        gpu_layers=2,
        context_size=4096,
        batch_size=512,
        ubatch_size=128,
        flash_attn="off",
        cache_type_k="q8_0",
        cache_type_v="f16",
        use_mmap=False,
        mlock=True,
    )

    command = build_cortex_cpp_runtime_command(model, 8765, "/opt/cortex-gguf-host")

    assert command[command.index("--gpu-layers") + 1] == "2"
    assert command[command.index("--context-size") + 1] == "4096"
    assert command[command.index("--batch-size") + 1] == "512"
    assert command[command.index("--ubatch-size") + 1] == "128"
    assert command[command.index("--flash-attn") + 1] == "off"
    assert command[command.index("--cache-type-k") + 1] == "q8_0"
    assert command[command.index("--cache-type-v") + 1] == "f16"
    assert "--no-mmap" in command
    assert "--mlock" in command


def test_cortex_cpp_ultra_uses_throughput_tuning_explicitly(tmp_path: Path) -> None:
    model_file = tmp_path / "native.gguf"
    model_file.write_bytes(b"GGUF fixture")
    model = ModelConfig(
        id="native-ultra",
        upstream_model="native-ultra",
        runtime="cortex-cpp",
        model_path=str(model_file),
        optimization_preset="ultra",
        gpu_layers=-1,
        context_size=4096,
        batch_size=256,
        ubatch_size=64,
        flash_attn="auto",
    )

    command = build_cortex_cpp_runtime_command(model, 8765, "/opt/cortex-gguf-host")

    assert command[command.index("--context-size") + 1] == "4096"
    assert command[command.index("--batch-size") + 1] == "512"
    assert command[command.index("--ubatch-size") + 1] == "128"
    assert command[command.index("--cache-type-k") + 1] == "q8_0"
    assert command[command.index("--flash-attn") + 1] == "on"


@pytest.mark.asyncio
async def test_cortex_cpp_runtime_check_and_supervision_need_no_python_loader(
    tmp_path: Path, monkeypatch
) -> None:
    model_file = tmp_path / "native.gguf"
    model_file.write_bytes(b"GGUF fixture")
    model = ModelConfig(
        id="native-local",
        upstream_model="native-alias",
        runtime="cortex-cpp",
        model_path=str(model_file),
    )
    settings = Settings(models={model.id: model}, default_model=model.id)
    observed: dict[str, object] = {}

    class FakeProcess:
        returncode: int | None = None
        pid = 5432
        stdout = None

        def terminate(self) -> None:
            self.returncode = 0

        def kill(self) -> None:
            self.returncode = -9

        async def wait(self) -> int:
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

    async def fake_create_process(*args: object, **kwargs: object) -> FakeProcess:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(
        "cortex_llmhoster.runtime._resolve_cortex_cpp_binary", lambda: "/bin/cortex-gguf-host"
    )
    monkeypatch.setattr(
        "cortex_llmhoster.runtime.asyncio.create_subprocess_exec", fake_create_process
    )
    async with httpx.AsyncClient() as client:
        manager = LocalRuntimeManager(settings, client, hardware={})

        async def monitor_forever(*_: object) -> None:
            await asyncio.Event().wait()

        monkeypatch.setattr(manager, "_free_port", lambda: 9235)
        monkeypatch.setattr(manager, "_monitor", monitor_forever)
        readiness = manager.test(model)
        await manager.start_configured()

        state = manager.state(model.id)
        command = observed["args"]
        assert readiness["ok"] is True
        assert readiness["runner"] == "/bin/cortex-gguf-host"
        assert state["runtime"] == "cortex-cpp"
        assert state["status"] == "starting"
        assert command[0] == "/bin/cortex-gguf-host"
        assert "llama-server" not in command
        assert not any(str(argument).endswith("embedded_backend") for argument in command)
        stopped = await manager.stop(model.id)
        assert stopped["status"] == "stopped"
        await manager.close()


@pytest.mark.asyncio
async def test_cortex_cpp_setup_check_warns_about_legacy_pickle_conversion(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = tmp_path / "legacy.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    converter_source = tmp_path / "llama.cpp"
    converter = converter_source / "examples" / "convert_legacy_llama.py"
    converter.parent.mkdir(parents=True)
    converter.touch()
    model = ModelConfig(
        id="legacy-model",
        upstream_model="legacy-model",
        runtime="cortex-cpp",
        model_path=str(checkpoint),
    )
    settings = Settings(models={model.id: model})
    monkeypatch.setattr(
        "cortex_llmhoster.runtime._resolve_cortex_cpp_binary", lambda: "/opt/cortex"
    )
    monkeypatch.setattr(
        "cortex_llmhoster.runtime.resolve_llama_cpp_source_dir", lambda: converter_source
    )
    monkeypatch.setattr(
        "cortex_llmhoster.runtime._resolve_cpp_converter_python", lambda: "/opt/python"
    )

    async with httpx.AsyncClient() as client:
        manager = LocalRuntimeManager(settings, client, hardware={})
        result = manager.test(model)

    assert result["ok"] is False
    assert "trusted checkpoint" in result["message"]


@pytest.mark.asyncio
async def test_cortex_cpp_refuses_legacy_pickle_checkpoint_without_opt_in(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = tmp_path / "legacy.ckpt"
    checkpoint.write_bytes(b"untrusted checkpoint fixture")
    model = ModelConfig(
        id="unsafe-legacy",
        upstream_model="unsafe-legacy",
        runtime="cortex-cpp",
        model_path=str(checkpoint),
    )
    settings = Settings(models={model.id: model}, default_model=model.id)
    monkeypatch.setattr(
        "cortex_llmhoster.runtime._resolve_cortex_cpp_binary", lambda: "/opt/cortex"
    )

    async with httpx.AsyncClient() as client:
        manager = LocalRuntimeManager(settings, client, hardware={})
        state = await manager.start(model.id)
        assert state["status"] == "error"
        assert "trusted LLaMA checkpoint" in state["error"]
        assert manager._slots[model.id].process is None
        await manager.close()


@pytest.mark.asyncio
async def test_cortex_cpp_stop_cancels_and_terminates_active_converter(
    tmp_path: Path, monkeypatch
) -> None:
    model_directory = tmp_path / "hf-model"
    model_directory.mkdir()
    (model_directory / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    (model_directory / "model.safetensors").write_bytes(b"weights")
    converter_source = tmp_path / "llama.cpp"
    converter = converter_source / "convert_hf_to_gguf.py"
    converter.parent.mkdir(parents=True)
    converter.touch()
    model = ModelConfig(
        id="cancellable-conversion",
        upstream_model="hf-alias",
        runtime="cortex-cpp",
        model_path=str(model_directory),
    )
    settings = Settings(models={model.id: model}, default_model=model.id)
    conversion_started = asyncio.Event()
    observed: dict[str, object] = {}

    class BlockingStdout:
        async def readline(self) -> bytes:
            conversion_started.set()
            await asyncio.Event().wait()
            return b""

    class FakeProcess:
        returncode: int | None = None
        pid = 7654
        stdout = BlockingStdout()

        def terminate(self) -> None:
            observed["terminated"] = True
            self.returncode = 0

        def kill(self) -> None:
            observed["killed"] = True
            self.returncode = -9

        async def wait(self) -> int:
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

    async def fake_create_process(*args: object, **kwargs: object) -> FakeProcess:
        observed["command"] = args
        observed["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(
        "cortex_llmhoster.runtime._resolve_cortex_cpp_binary", lambda: "/opt/cortex"
    )
    monkeypatch.setattr(
        "cortex_llmhoster.runtime._resolve_cpp_converter_python", lambda: "/opt/python"
    )
    monkeypatch.setattr(
        "cortex_llmhoster.runtime.resolve_llama_cpp_source_dir", lambda: converter_source
    )
    monkeypatch.setattr(
        "cortex_llmhoster.runtime.asyncio.create_subprocess_exec", fake_create_process
    )

    async with httpx.AsyncClient() as client:
        manager = LocalRuntimeManager(settings, client, hardware={})
        started = await manager.start(model.id)
        assert started["status"] == "starting"
        await asyncio.wait_for(conversion_started.wait(), timeout=1.0)
        assert manager._slots[model.id].preparation_process is not None

        stopped = await manager.stop(model.id)
        assert stopped["status"] == "stopped"
        assert observed["terminated"] is True
        assert manager._slots[model.id].process is None
        await manager.close()


@pytest.mark.asyncio
async def test_cortex_cpp_supervises_conversion_then_starts_native_inference(
    tmp_path: Path, monkeypatch
) -> None:
    model_directory = tmp_path / "hf-model"
    model_directory.mkdir()
    (model_directory / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    (model_directory / "model.safetensors").write_bytes(b"weights")
    converter_source = tmp_path / "llama.cpp"
    converter = converter_source / "convert_hf_to_gguf.py"
    converter.parent.mkdir(parents=True)
    converter.touch()
    source = inspect_native_model_source(model_directory)
    model = ModelConfig(
        id="hf-native",
        upstream_model="hf-native-alias",
        runtime="cortex-cpp",
        model_path=str(model_directory),
        conversion_outtype="f16",
    )
    settings = Settings(models={model.id: model}, default_model=model.id)
    monkeypatch.setenv("CORTEX_CPP_CACHE_DIR", str(tmp_path / "cache"))
    observed: dict[str, object] = {}

    class FakeProcess:
        returncode: int | None = None
        pid = 6543
        stdout = None

        def terminate(self) -> None:
            self.returncode = 0

        def kill(self) -> None:
            self.returncode = -9

        async def wait(self) -> int:
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

    async def fake_create_process(*args: object, **kwargs: object) -> FakeProcess:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return FakeProcess()

    async def fake_convert(slot: RuntimeSlot, command: list[str], _working_directory: str) -> None:
        output_template = Path(command[command.index("--outfile") + 1])
        output_path = output_template.parent / "cortex-model.f16.gguf"
        output_path.write_bytes(b"converted model")
        slot.output.append("fake converter completed")

    async with httpx.AsyncClient() as client:
        manager = LocalRuntimeManager(settings, client, hardware={})
        slot = RuntimeSlot(model=model, status="starting", port=9236)
        manager._slots[model.id] = slot
        monkeypatch.setattr(manager, "_run_native_conversion", fake_convert)
        monkeypatch.setattr(
            "cortex_llmhoster.runtime.asyncio.create_subprocess_exec", fake_create_process
        )

        async def monitor_forever(*_: object) -> None:
            await asyncio.Event().wait()

        monkeypatch.setattr(manager, "_monitor", monitor_forever)
        await manager._prepare_and_start_native_model(
            model,
            slot,
            "/opt/cortex-gguf-host",
            source,
            converter_source,
            "/opt/converter-python",
            None,
        )

        command = observed["args"]
        assert command[0] == "/opt/cortex-gguf-host"
        converted_path = Path(command[command.index("--model-path") + 1])
        assert converted_path.is_file()
        assert converted_path.parent.parent == tmp_path / "cache"
        assert slot.process is not None
        assert slot.monitor_task is not None
        assert "fake converter completed" in slot.output
        assert any("Using native GGUF model" in line for line in slot.output)

        stopped = await manager.stop(model.id)
        assert stopped["status"] == "stopped"
        await manager.close()
