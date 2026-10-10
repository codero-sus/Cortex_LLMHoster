"""Managed local inference process adapters for low-footprint model hosting."""

from __future__ import annotations

import asyncio
import math
import os
import platform
import shutil
import socket
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from string import Formatter
from typing import Any

import httpx

from .config import LOCAL_RUNTIMES, ModelConfig, Settings


@lru_cache(maxsize=1)
def _available_cpu_count() -> int:
    """Cache OS, affinity, and cgroup probes for the lifetime of this process."""

    count = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        try:
            count = min(count, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    quota_files = (
        (Path("/sys/fs/cgroup/cpu.max"), "v2"),
        (Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), "quota"),
    )
    for quota_path, format_name in quota_files:
        try:
            if format_name == "v2":
                quota_text, period_text = quota_path.read_text(encoding="ascii").split()
                if quota_text == "max":
                    continue
                quota, period = int(quota_text), int(period_text)
            else:
                quota = int(quota_path.read_text(encoding="ascii").strip())
                period = int(
                    Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text(encoding="ascii").strip()
                )
            if quota > 0 and period > 0:
                count = min(count, max(1, math.ceil(quota / period)))
                break
        except (OSError, ValueError):
            continue
    return max(1, count)


def automatic_threads(cpu_count: int | None = None) -> int:
    """Choose threads from available CPUs, respecting container limits."""

    count = _available_cpu_count() if cpu_count is None else cpu_count or os.cpu_count() or 1
    return max(1, count - 1) if count > 2 else count


def build_llama_command(
    binary: str,
    model: ModelConfig,
    port: int,
    *,
    cpu_count: int | None = None,
) -> list[str]:
    """Build an argv list (never a shell string) for a local GGUF model."""

    if not model.model_path:
        raise ValueError("Local llama.cpp models require model_path.")
    threads = model.threads or automatic_threads(cpu_count)
    threads_batch = model.threads_batch or threads
    optimization_level = model.optimization_level
    ultra = optimization_level == 1
    if optimization_level == -1:
        batch_size = min(model.batch_size, 128)
        ubatch_size = min(model.ubatch_size, batch_size, 32)
    elif ultra:
        batch_size = max(model.batch_size, 512)
        ubatch_size = min(batch_size, max(model.ubatch_size, 128))
    else:
        batch_size = model.batch_size
        ubatch_size = model.ubatch_size
    flash_attn = model.flash_attn
    if ultra and model.gpu_layers != 0 and flash_attn == "auto":
        flash_attn = "on"
    cache_type_k = "q8_0" if ultra else model.cache_type_k
    cache_type_v = "q8_0" if ultra else model.cache_type_v
    args = [
        binary,
        "--model",
        str(Path(model.model_path).expanduser().resolve()),
        "--alias",
        model.upstream_model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(model.context_size),
        "--threads",
        str(threads),
        "--threads-batch",
        str(threads_batch),
        "--n-gpu-layers",
        str(model.gpu_layers),
        "--batch-size",
        str(batch_size),
        "--ubatch-size",
        str(ubatch_size),
        "--parallel",
        str(model.parallel),
        "--flash-attn",
        flash_attn,
        "--cache-type-k",
        cache_type_k,
        "--cache-type-v",
        cache_type_v,
    ]
    if not model.use_mmap:
        args.append("--no-mmap")
    if model.mlock:
        args.append("--mlock")
    if model.embedding:
        args.append("--embedding")
    if model.mmproj_path:
        args.extend(("--mmproj", str(Path(model.mmproj_path).expanduser().resolve())))
    return args


@lru_cache(maxsize=128)
def _runtime_command_placeholders(command: tuple[str, ...]) -> frozenset[str]:
    formatter = Formatter()
    return frozenset(
        field_name
        for argument in command
        for _, field_name, _, _ in formatter.parse(argument)
        if field_name
    )


def build_command_runtime_command(model: ModelConfig, port: int) -> list[str]:
    """Format argv for a user-selected local OpenAI-compatible runtime process.

    The command is executed without a shell. ``{host}`` is always the loopback
    address, and ``{port}`` is an ephemeral port owned by Cortex.
    """

    if not model.runtime_command:
        raise ValueError("Local command runtime has no executable configured.")
    if not any("{host}" in argument for argument in model.runtime_command) or not any(
        "{port}" in argument for argument in model.runtime_command
    ):
        raise ValueError("Local runtime command must include {host} and {port} placeholders.")

    try:
        placeholders = _runtime_command_placeholders(tuple(model.runtime_command))
    except ValueError as exc:
        raise ValueError(f"Could not format local runtime command: {exc}") from exc

    values = {
        "model_id": model.id,
        "model_alias": model.upstream_model,
        "host": "127.0.0.1",
        "port": str(port),
        "api_base_path": model.api_base_path,
        "health_path": model.health_path,
        "gpu_layers": str(model.gpu_layers),
        "context_size": str(model.context_size),
    }
    if "model_path" in placeholders:
        values["model_path"] = (
            str(Path(model.model_path).expanduser().resolve()) if model.model_path else ""
        )
    if "threads" in placeholders:
        values["threads"] = str(model.threads or automatic_threads())
    try:
        return [argument.format_map(values) for argument in model.runtime_command]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Could not format local runtime command: {exc}") from exc


def _resolve_executable(executable: str) -> str | None:
    expanded = str(Path(executable).expanduser())
    if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
        return expanded
    return shutil.which(expanded)


@dataclass(slots=True)
class RuntimeSlot:
    model: ModelConfig
    status: str = "stopped"
    port: int | None = None
    process: asyncio.subprocess.Process | None = None
    command: list[str] = field(default_factory=list)
    error: str | None = None
    output: deque[str] = field(default_factory=lambda: deque(maxlen=40))
    started_at: float | None = None
    monitor_task: asyncio.Task[None] | None = None
    output_task: asyncio.Task[None] | None = None


class LocalRuntimeManager:
    """Own and supervise local model-serving processes.

    ``llama.cpp`` has a first-class GGUF adapter. The ``command`` adapter can
    start any local runtime that exposes OpenAI-compatible inference endpoints.
    Both bind to loopback and are only reached through Cortex.
    """

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient,
        hardware: dict[str, Any] | None = None,
    ) -> None:
        self.settings = settings
        self.client = client
        self._slots: dict[str, RuntimeSlot] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._settings_lock = asyncio.Lock()
        self.hardware = hardware if hardware is not None else detect_hardware()

    def _slot(self, model_id: str) -> RuntimeSlot:
        slot = self._slots.get(model_id)
        if slot is None:
            model = self.settings.models.get(model_id)
            if model is None:
                raise KeyError(model_id)
            slot = RuntimeSlot(model=model)
            self._slots[model_id] = slot
        return slot

    def _lock(self, model_id: str) -> asyncio.Lock:
        return self._locks.setdefault(model_id, asyncio.Lock())

    def _resolve_binary(self) -> str | None:
        configured = self.settings.llama_server_path
        if configured:
            expanded = str(Path(configured).expanduser())
            if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
                return expanded
            return shutil.which(expanded)
        return shutil.which("llama-server")

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    async def start_configured(self) -> None:
        """Start only the default model to avoid loading every model into RAM/VRAM."""

        default_model = self.settings.models.get(self.settings.default_model or "")
        if default_model is not None and default_model.runtime in LOCAL_RUNTIMES:
            await self.start(default_model.id)

    async def start(self, model_id: str) -> dict[str, Any]:
        async with self._lock(model_id):
            model = self.settings.models.get(model_id)
            if model is None:
                raise KeyError(model_id)
            if model.runtime not in LOCAL_RUNTIMES:
                raise ValueError(f"Unsupported local runtime: {model.runtime}")

            slot = self._slot(model_id)
            if slot.process is not None and slot.process.returncode is None:
                return self.state(model_id)
            await self._cancel_tasks(slot)
            slot.process = None
            slot.port = None
            slot.started_at = None
            slot.model = model
            slot.status = "starting"
            slot.error = None
            slot.output.clear()

            model_path = Path(model.model_path).expanduser().resolve() if model.model_path else None
            if model.runtime == "llama.cpp":
                if model_path is None or not model_path.is_file():
                    slot.status = "error"
                    slot.error = f"GGUF model file not found: {model_path or model.model_path}"
                    return self.state(model_id)
                if model.mmproj_path and not Path(model.mmproj_path).expanduser().is_file():
                    slot.status = "error"
                    slot.error = f"Multimodal projector file not found: {model.mmproj_path}"
                    return self.state(model_id)
                binary = self._resolve_binary()
                if not binary:
                    slot.status = "error"
                    slot.error = (
                        "llama-server was not found. Install a CPU/GPU-enabled llama.cpp build "
                        "or set server.llama_server_path / CORTEX_LLAMA_SERVER."
                    )
                    return self.state(model_id)
                slot.port = self._free_port()
                slot.command = build_llama_command(binary, model, slot.port)
                cwd = str(model_path.parent)
                runtime_name = "llama-server"
            else:
                if model_path is not None and not model_path.exists():
                    slot.status = "error"
                    slot.error = f"Local model path not found: {model_path}"
                    return self.state(model_id)
                slot.port = self._free_port()
                try:
                    slot.command = build_command_runtime_command(model, slot.port)
                except ValueError as exc:
                    slot.status = "error"
                    slot.error = str(exc)
                    return self.state(model_id)
                if not slot.command:
                    slot.status = "error"
                    slot.error = "Local command runtime has no executable configured."
                    return self.state(model_id)
                binary = _resolve_executable(slot.command[0])
                if not binary:
                    slot.status = "error"
                    slot.error = f"Local runtime executable not found: {slot.command[0]}"
                    return self.state(model_id)
                slot.command[0] = binary
                cwd = (
                    str(model_path if model_path.is_dir() else model_path.parent)
                    if model_path is not None
                    else None
                )
                runtime_name = "local runtime"

            try:
                slot.process = await asyncio.create_subprocess_exec(
                    *slot.command,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=cwd,
                )
            except (OSError, ValueError) as exc:
                slot.status = "error"
                slot.error = f"Could not start {runtime_name}: {exc}"
                slot.process = None
                return self.state(model_id)

            slot.started_at = time.monotonic()
            slot.status = "starting"
            slot.output_task = asyncio.create_task(
                self._read_output(slot), name=f"runtime-output-{model_id}"
            )
            slot.monitor_task = asyncio.create_task(
                self._monitor(model_id, slot), name=f"runtime-monitor-{model_id}"
            )
            return self.state(model_id)

    async def _read_output(self, slot: RuntimeSlot) -> None:
        process = slot.process
        if process is None or process.stdout is None:
            return
        try:
            while line := await process.stdout.readline():
                slot.output.append(line.decode("utf-8", errors="replace").rstrip()[-500:])
        except (asyncio.CancelledError, OSError):
            return

    async def _monitor(self, model_id: str, slot: RuntimeSlot) -> None:
        process = slot.process
        if process is None or slot.port is None:
            return
        health_url = f"http://127.0.0.1:{slot.port}{slot.model.health_path}"
        try:
            while process.returncode is None:
                if slot.status != "running":
                    try:
                        response = await self.client.get(health_url, timeout=0.75)
                        if response.is_success:
                            slot.status = "running"
                            slot.error = None
                    except httpx.RequestError:
                        pass
                await asyncio.sleep(0.5)
            code = await process.wait()
            if slot.status != "stopped":
                slot.status = "error"
                slot.error = f"Local runtime exited with code {code}."
        except asyncio.CancelledError:
            return

    async def stop(self, model_id: str) -> dict[str, Any]:
        async with self._lock(model_id):
            slot = self._slots.get(model_id)
            if slot is None:
                return {"id": model_id, "status": "stopped", "logs": []}
            process = slot.process
            if process is not None and process.returncode is None:
                slot.status = "stopping"
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(process.wait(), timeout=6.0)
                except TimeoutError:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    await process.wait()
            await self._cancel_tasks(slot)
            slot.process = None
            slot.status = "stopped"
            slot.port = None
            slot.started_at = None
            return self.state(model_id)

    async def restart(self, model_id: str) -> dict[str, Any]:
        await self.stop(model_id)
        return await self.start(model_id)

    async def _cancel_tasks(self, slot: RuntimeSlot) -> None:
        tasks = [
            task
            for task in (slot.monitor_task, slot.output_task)
            if task is not None and not task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        slot.monitor_task = None
        slot.output_task = None

    async def apply_settings(self, settings: Settings) -> None:
        async with self._settings_lock:
            await self._apply_settings(settings)

    async def _apply_settings(self, settings: Settings) -> None:
        old_settings = self.settings
        old_local = {
            model.id: model
            for model in old_settings.models.values()
            if model.runtime in LOCAL_RUNTIMES
        }
        new_local = {
            model.id: model for model in settings.models.values() if model.runtime in LOCAL_RUNTIMES
        }
        changed = {
            model_id
            for model_id in old_local.keys() & new_local.keys()
            if old_local[model_id] != new_local[model_id]
        }
        if old_settings.llama_server_path != settings.llama_server_path:
            changed.update(
                model_id for model_id, model in new_local.items() if model.runtime == "llama.cpp"
            )
        removed = (old_local.keys() - new_local.keys()) | changed
        to_stop = [
            model_id
            for model_id in removed
            if model_id in self._slots and self._slots[model_id].process is not None
        ]
        old_default = old_settings.models.get(old_settings.default_model or "")
        if (
            old_default is not None
            and old_default.runtime in LOCAL_RUNTIMES
            and old_default.id != settings.default_model
            and old_default.id in self._slots
            and self._slots[old_default.id].process is not None
        ):
            to_stop.append(old_default.id)
        for model_id in dict.fromkeys(to_stop):
            await self.stop(model_id)
        self.settings = settings
        for model_id in removed:
            self._slots.pop(model_id, None)
            self._locks.pop(model_id, None)

        new_default = settings.models.get(settings.default_model or "")
        should_start_default = settings.default_model != old_settings.default_model
        should_start_default = should_start_default or settings.default_model in changed
        if (
            should_start_default
            and new_default is not None
            and new_default.runtime in LOCAL_RUNTIMES
        ):
            await self.start(new_default.id)

    def test(self, model: ModelConfig) -> dict[str, Any]:
        path = Path(model.model_path).expanduser() if model.model_path else None
        exists = path is not None and path.exists()
        size = path.stat().st_size if exists and path is not None and path.is_file() else 0
        projector_exists = (
            model.mmproj_path is None or Path(model.mmproj_path).expanduser().is_file()
        )
        if model.runtime == "llama.cpp":
            runner = self._resolve_binary()
            ready = bool(
                exists
                and path is not None
                and path.is_file()
                and path.suffix.lower() == ".gguf"
                and projector_exists
                and runner
            )
            message = (
                "Local GGUF, optional projector, and llama-server are ready to start."
                if ready
                else "Check the GGUF/projector paths and install/configure llama-server."
            )
        else:
            try:
                command = build_command_runtime_command(model, 1)
            except ValueError as exc:
                command = []
                message = str(exc)
            runner = _resolve_executable(command[0]) if command else None
            model_path_ready = model.model_path is None or exists
            ready = bool(runner and model_path_ready)
            if command:
                message = (
                    "Local command runtime and model path are ready to start."
                    if ready
                    else "Check the runtime executable and local model path."
                )
        return {
            "ok": ready,
            "runtime": model.runtime,
            "runner": runner,
            "model_path_exists": exists,
            "model_file_exists": bool(exists and path is not None and path.is_file()),
            "model_size_bytes": size,
            "mmproj_file_exists": projector_exists,
            "capabilities": list(model.effective_capabilities),
            "message": message,
        }

    def local_base_url(self, model: ModelConfig) -> str | None:
        slot = self._slots.get(model.id)
        if slot is None or slot.status != "running" or slot.port is None:
            return None
        return f"http://127.0.0.1:{slot.port}{model.api_base_path}"

    def state(self, model_id: str) -> dict[str, Any]:
        model = self.settings.models.get(model_id)
        if model is None:
            return {"id": model_id, "status": "removed", "logs": []}
        slot = self._slots.get(model_id)
        if slot is None:
            return {"id": model_id, "status": "stopped", "logs": []}
        uptime = round(time.monotonic() - slot.started_at, 1) if slot.started_at is not None else 0
        return {
            "id": model_id,
            "runtime": model.runtime,
            "capabilities": list(model.effective_capabilities),
            "status": slot.status,
            "error": slot.error,
            "pid": slot.process.pid if slot.process is not None else None,
            "port": slot.port,
            "base_url": self.local_base_url(model),
            "uptime_seconds": uptime,
            "command": list(slot.command),
            "logs": list(slot.output),
        }

    def states(self) -> dict[str, dict[str, Any]]:
        return {model_id: self.state(model_id) for model_id in self.settings.models}

    async def close(self) -> None:
        for model_id in list(self._slots):
            await self.stop(model_id)


# Backwards-compatible name retained for callers from the GGUF-only version.
LlamaServerManager = LocalRuntimeManager


def detect_hardware() -> dict[str, Any]:
    """Lightweight, best-effort hardware summary; GPU support depends on llama.cpp build."""

    result: dict[str, Any] = {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count() or 1,
        "recommended_threads": automatic_threads(),
        "ram_total_gb": None,
        "ram_available_gb": None,
        "gpu": "Not detected (CPU inference remains available)",
    }
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        try:
            values = {}
            for line in meminfo.read_text(encoding="ascii").splitlines():
                key, value = line.split(":", 1)
                if key in {"MemTotal", "MemAvailable"}:
                    values[key] = int(value.strip().split()[0]) * 1024
            if "MemTotal" in values:
                result["ram_total_gb"] = round(values["MemTotal"] / (1024**3), 1)
            if "MemAvailable" in values:
                result["ram_available_gb"] = round(values["MemAvailable"] / (1024**3), 1)
        except (OSError, ValueError):
            pass

    cgroup_memory_pairs = (
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
        (
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
    )
    for limit_path, usage_path in cgroup_memory_pairs:
        try:
            raw_limit = limit_path.read_text(encoding="ascii").strip()
            limit = int(raw_limit) if raw_limit != "max" else 0
            usage = int(usage_path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            continue
        if 0 < limit < 1 << 60:
            host_total = result["ram_total_gb"]
            limit_gb = round(limit / (1024**3), 1)
            result["ram_total_gb"] = min(host_total, limit_gb) if host_total else limit_gb
            result["ram_available_gb"] = round(max(0, limit - usage) / (1024**3), 1)
            break

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            probe = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                check=True,
                text=True,
                timeout=1.5,
            )
            result["gpu"] = (
                "; ".join(line.strip() for line in probe.stdout.splitlines() if line.strip())
                or "NVIDIA GPU detected"
            )
        except (OSError, subprocess.SubprocessError):
            result["gpu"] = "NVIDIA tooling detected; details unavailable"
    elif platform.system() == "Linux":
        vendors = {"0x10de": "NVIDIA", "0x1002": "AMD", "0x8086": "Intel"}
        for vendor_file in sorted(Path("/sys/class/drm").glob("card[0-9]*/device/vendor")):
            try:
                vendor = vendor_file.read_text(encoding="ascii").strip().lower()
            except OSError:
                continue
            if vendor in vendors:
                result["gpu"] = (
                    f"{vendors[vendor]} GPU detected; acceleration depends on the llama.cpp build"
                )
                break
    elif platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        result["gpu"] = "Apple Silicon Metal may be available in the llama.cpp build"
    return result
