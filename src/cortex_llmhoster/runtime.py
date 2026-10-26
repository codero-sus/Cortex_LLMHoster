"""Managed local inference process adapters for low-footprint model hosting."""

from __future__ import annotations

import asyncio
import csv
import io
import logging
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

from .config import (
    CORTEX_CPP_RUNTIME,
    CORTEX_RUNTIME,
    CPP_QUANTIZED_OUTTYPES,
    CPP_QUANTIZER_OUTTYPES,
    LOCAL_RUNTIMES,
    OPTIMIZATION_PRESETS,
    ModelConfig,
    Settings,
)
from .native_installer import (
    RELEASE_TAG,
    RuntimeInstallError,
    auto_install_enabled,
    install_latest_runtime_binary,
    release_target,
)
from .native_model import (
    NativeModelSource,
    NativeModelSourceError,
    build_native_conversion_command,
    build_native_quantization_command,
    cached_native_conversion,
    converter_script,
    create_native_conversion_temp_dir,
    discard_native_conversion_temp,
    find_native_gguf_output,
    inspect_native_model_source,
    native_conversion_cache_dir,
    native_conversion_cache_key,
    publish_native_conversion,
    resolve_llama_cpp_source_dir,
)
from .python_runtime import PythonInterpreterError, resolve_python_executable

logger = logging.getLogger("cortex_llmhoster.runtime")


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
    preset = model.optimization_preset
    if preset not in OPTIMIZATION_PRESETS:
        preset = {-1: "high_plus", 0: "balanced", 1: "ultra"}.get(
            model.optimization_level, "balanced"
        )
    batch_size = model.batch_size
    ubatch_size = model.ubatch_size
    cache_type_k = model.cache_type_k
    cache_type_v = model.cache_type_v
    flash_attn = model.flash_attn
    context_size = model.context_size
    parallel = model.parallel
    use_mmap = model.use_mmap
    mlock = model.mlock
    if preset == "low_minus":
        batch_size = max(batch_size, 1024)
        ubatch_size = min(batch_size, max(ubatch_size, 256))
        cache_type_k = cache_type_v = "f16"
    elif preset == "low_plus":
        batch_size = max(batch_size, 768)
        ubatch_size = min(batch_size, max(ubatch_size, 192))
        cache_type_k = cache_type_v = "f16"
    elif preset == "balanced":
        pass
    elif preset == "medium_minus":
        batch_size = max(batch_size, 384)
        ubatch_size = min(batch_size, max(ubatch_size, 96))
        cache_type_k = cache_type_v = "f16"
    elif preset == "medium_plus":
        batch_size = max(batch_size, 384)
        ubatch_size = min(batch_size, max(ubatch_size, 96))
        cache_type_k = cache_type_v = "q8_0"
    elif preset == "high_minus":
        batch_size = min(batch_size, 256)
        ubatch_size = min(ubatch_size, batch_size, 64)
        cache_type_k = cache_type_v = "q8_0"
    elif preset == "high_plus":
        batch_size = min(batch_size, 128)
        ubatch_size = min(ubatch_size, batch_size, 32)
        cache_type_k = cache_type_v = "q8_0"
    elif preset == "low_vram":
        context_size = min(context_size, 2048)
        batch_size = min(batch_size, context_size, 128)
        ubatch_size = min(ubatch_size, batch_size, 32)
        parallel = 1
        cache_type_k = cache_type_v = "q8_0"
        use_mmap = True
        mlock = False
        if model.gpu_layers != 0 and flash_attn == "auto":
            flash_attn = "on"
    elif preset == "super_low":
        if model.gpu_layers != 0 and flash_attn == "auto":
            flash_attn = "on"
    elif preset == "ultra":
        batch_size = max(batch_size, 512)
        ubatch_size = min(batch_size, max(ubatch_size, 128))
        cache_type_k = cache_type_v = "q8_0"
        if model.gpu_layers != 0 and flash_attn == "auto":
            flash_attn = "on"
    try:
        resolved_model_path = Path(model.model_path).expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Could not resolve local model path: {exc}") from exc
    args = [
        binary,
        "--model",
        str(resolved_model_path),
        "--alias",
        model.upstream_model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(context_size),
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
        str(parallel),
        "--flash-attn",
        flash_attn,
        "--cache-type-k",
        cache_type_k,
        "--cache-type-v",
        cache_type_v,
    ]
    if not use_mmap:
        args.append("--no-mmap")
    if mlock:
        args.append("--mlock")
    if model.embedding:
        args.append("--embedding")
    if model.mmproj_path:
        try:
            projector_path = Path(model.mmproj_path).expanduser().resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(f"Could not resolve projector path: {exc}") from exc
        args.extend(("--mmproj", str(projector_path)))
    return args


def build_cortex_runtime_command(model: ModelConfig, port: int) -> list[str]:
    """Launch Cortex's own Python GGUF backend without a llama-server executable."""

    if model.runtime != CORTEX_RUNTIME:
        raise ValueError("Cortex's built-in backend requires runtime='cortex'.")
    if not model.model_path:
        raise ValueError("Cortex's built-in GGUF backend requires model_path.")
    if model.mmproj_path:
        raise ValueError("Cortex's built-in GGUF backend does not support projector files.")
    try:
        model_path = Path(model.model_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Could not resolve local model path: {exc}") from exc
    if not model_path.is_file() or model_path.suffix.lower() != ".gguf":
        raise ValueError("Cortex's built-in backend requires an existing .gguf model file.")

    threads = model.threads or automatic_threads()
    threads_batch = model.threads_batch or threads
    if model.optimization_preset == "low_vram":
        context_size = min(model.context_size, 2048)
        batch_size = min(model.batch_size, context_size, 128)
        use_mmap = True
        mlock = False
    else:
        context_size = model.context_size
        batch_size = model.batch_size
        use_mmap = model.use_mmap
        mlock = model.mlock
    command = [
        resolve_python_executable(),
        "-m",
        "cortex_llmhoster.embedded_backend",
        "--model-path",
        str(model_path),
        "--model-alias",
        model.upstream_model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--api-base-path",
        model.api_base_path,
        "--health-path",
        model.health_path,
        "--context-size",
        str(context_size),
        "--threads",
        str(threads),
        "--threads-batch",
        str(threads_batch),
        "--gpu-layers",
        str(model.gpu_layers),
        "--batch-size",
        str(batch_size),
    ]
    if not use_mmap:
        command.append("--no-mmap")
    if mlock:
        command.append("--mlock")
    if model.embedding:
        command.append("--embedding")
    return command


def _resolve_cortex_cpp_binary() -> str | None:
    """Find Cortex's native GGUF host without relying on Python model bindings."""

    configured = os.environ.get("CORTEX_CPP_HOST")
    if configured:
        resolved = _resolve_executable(configured)
        return str(Path(resolved).resolve()) if resolved else None

    executable = "cortex-gguf-host.exe" if os.name == "nt" else "cortex-gguf-host"
    project_root = Path(__file__).resolve().parents[2]
    candidates = (
        project_root / "native" / "build" / "bin" / executable,
        project_root / "native" / "build" / executable,
        project_root / "native" / "build" / "bin" / "Release" / executable,
        project_root / "native" / "build" / "Release" / executable,
    )
    for candidate in candidates:
        if candidate.is_file() and (os.name == "nt" or os.access(candidate, os.X_OK)):
            return str(candidate)
    resolved = shutil.which(executable)
    return str(Path(resolved).resolve()) if resolved else None


def _resolve_cortex_cpp_quantizer(host_binary: str | None = None) -> str | None:
    """Find the pinned native GGUF quantizer without needing Python bindings."""

    configured = os.environ.get("CORTEX_CPP_QUANTIZE")
    if configured:
        resolved = _resolve_executable(configured)
        return str(Path(resolved).resolve()) if resolved else None

    executable = "llama-quantize.exe" if os.name == "nt" else "llama-quantize"
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        project_root / "native" / "build" / "bin" / executable,
        project_root / "native" / "build" / executable,
        project_root / "native" / "build" / "bin" / "Release" / executable,
        project_root / "native" / "build" / "Release" / executable,
    ]
    host = host_binary or _resolve_cortex_cpp_binary()
    if host:
        candidates.insert(0, Path(host).parent / executable)
    for candidate in candidates:
        if candidate.is_file() and (os.name == "nt" or os.access(candidate, os.X_OK)):
            return str(candidate)
    resolved = shutil.which(executable)
    return str(Path(resolved).resolve()) if resolved else None


def _requires_native_quantization(source: NativeModelSource | None, outtype: str) -> bool:
    """Use llama-quantize for low-bit sources and any requested GGUF requantization."""

    return outtype in CPP_QUANTIZED_OUTTYPES or (
        source is not None and source.kind == "gguf" and outtype in CPP_QUANTIZER_OUTTYPES
    )


def _resolve_cpp_converter_python() -> str:
    """Resolve an optional Python interpreter used only by upstream converters."""

    configured = os.environ.get("CORTEX_CPP_CONVERTER_PYTHON")
    if configured:
        resolved = _resolve_executable(configured)
        if not resolved:
            raise PythonInterpreterError(
                "CORTEX_CPP_CONVERTER_PYTHON does not point to an executable interpreter."
            )
        return str(Path(resolved).resolve())
    return resolve_python_executable()


def build_cortex_cpp_runtime_command(
    model: ModelConfig,
    port: int,
    binary: str,
    *,
    model_path: str | Path | None = None,
) -> list[str]:
    """Launch the native C++ GGUF process after any source-format conversion."""

    if model.runtime != CORTEX_CPP_RUNTIME:
        raise ValueError("The native backend requires runtime='cortex-cpp'.")
    if not binary:
        raise ValueError("Cortex's native C++ host executable was not found.")
    selected_path = model_path if model_path is not None else model.model_path
    if not selected_path:
        raise ValueError("Cortex's native GGUF backend requires model_path.")
    if model.mmproj_path:
        raise ValueError("Cortex's native GGUF backend does not support projector files.")
    try:
        resolved_model_path = Path(selected_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Could not resolve local model path: {exc}") from exc
    if not resolved_model_path.is_file() or resolved_model_path.suffix.lower() != ".gguf":
        raise ValueError("The native C++ inference process requires a prepared GGUF model file.")

    threads = model.threads or automatic_threads()
    threads_batch = model.threads_batch or threads
    context_size = model.context_size
    batch_size = model.batch_size
    ubatch_size = model.ubatch_size
    gpu_layers = model.gpu_layers
    flash_attn = model.flash_attn
    cache_type_k = model.cache_type_k
    cache_type_v = model.cache_type_v
    use_mmap = model.use_mmap
    mlock = model.mlock
    if model.optimization_preset == "super_low":
        if gpu_layers == -1:
            gpu_layers = -2
        if gpu_layers != 0 and flash_attn == "auto":
            flash_attn = "on"
    elif model.optimization_preset == "low_vram":
        context_size = min(context_size, 2048)
        batch_size = min(batch_size, context_size, 128)
        ubatch_size = min(ubatch_size, batch_size, 32)
        cache_type_k = cache_type_v = "q8_0"
        use_mmap = True
        mlock = False
        if gpu_layers == -1:
            gpu_layers = -2
        if gpu_layers != 0 and flash_attn == "auto":
            flash_attn = "on"
    elif model.optimization_preset == "ultra":
        batch_size = max(batch_size, 512)
        ubatch_size = min(batch_size, max(ubatch_size, 128))
        cache_type_k = cache_type_v = "q8_0"
        if gpu_layers != 0 and flash_attn == "auto":
            flash_attn = "on"
    command = [
        binary,
        "--model-path",
        str(resolved_model_path),
        "--model-alias",
        model.upstream_model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--api-base-path",
        model.api_base_path,
        "--health-path",
        model.health_path,
        "--context-size",
        str(context_size),
        "--threads",
        str(threads),
        "--threads-batch",
        str(threads_batch),
        "--gpu-layers",
        str(gpu_layers),
        "--batch-size",
        str(batch_size),
        "--ubatch-size",
        str(ubatch_size),
        "--flash-attn",
        flash_attn,
        "--cache-type-k",
        cache_type_k,
        "--cache-type-v",
        cache_type_v,
    ]
    if not use_mmap:
        command.append("--no-mmap")
    if mlock:
        command.append("--mlock")
    if model.embedding:
        command.append("--embedding")
    return command


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
        try:
            values["model_path"] = (
                str(Path(model.model_path).expanduser().resolve()) if model.model_path else ""
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(f"Could not resolve local model path: {exc}") from exc
    if "threads" in placeholders:
        values["threads"] = str(model.threads or automatic_threads())
    if "python" in placeholders:
        values["python"] = resolve_python_executable()
    try:
        return [argument.format_map(values) for argument in model.runtime_command]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Could not format local runtime command: {exc}") from exc


def _resolve_executable(executable: str) -> str | None:
    expanded = str(Path(executable).expanduser())
    if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
        return expanded
    return shutil.which(expanded)


def _runtime_child_environment(settings: Settings) -> dict[str, str]:
    """Keep runtime helpers from inheriting Cortex dashboard authentication secrets."""

    environment = os.environ.copy()
    secret_names = {"CORTEX_API_KEY", "CORTEX_ADMIN_PASSWORD"}
    if settings.api_key_env:
        secret_names.add(settings.api_key_env)
    for name in secret_names:
        environment.pop(name, None)
    return environment


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
    preparation_task: asyncio.Task[None] | None = None
    preparation_process: asyncio.subprocess.Process | None = None


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
        self._conversion_locks: dict[str, asyncio.Lock] = {}
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

    def _embedded_python_available(self) -> tuple[str | None, bool]:
        try:
            python_executable = resolve_python_executable()
            probe = subprocess.run(
                [python_executable, "-c", "import llama_cpp"],
                capture_output=True,
                check=False,
                timeout=10,
                env=_runtime_child_environment(self.settings),
            )
        except (OSError, PythonInterpreterError, subprocess.SubprocessError):
            return None, False
        return python_executable, probe.returncode == 0

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    async def _run_native_conversion(
        self,
        slot: RuntimeSlot,
        command: list[str],
        working_directory: str,
    ) -> None:
        """Run a one-time local format conversion without blocking Cortex's event loop."""

        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=working_directory,
            env=_runtime_child_environment(self.settings),
        )
        slot.preparation_process = process
        if process.stdout is not None:
            while line := await process.stdout.readline():
                slot.output.append(line.decode("utf-8", errors="replace").rstrip()[-500:])
        code = await process.wait()
        slot.preparation_process = None
        if code != 0:
            tail = "; ".join(list(slot.output)[-3:])
            detail = f" Last preparation output: {tail}" if tail else ""
            raise RuntimeError(f"The model preparation process exited with code {code}.{detail}")

    async def _prepare_and_start_native_model(
        self,
        model: ModelConfig,
        slot: RuntimeSlot,
        binary: str,
        source: NativeModelSource,
        source_dir: Path | None,
        python_executable: str | None,
        quantizer: str | None,
    ) -> None:
        """Convert a supported source model once, then exec the native inference process."""

        temp_dir: Path | None = None
        try:
            outtype = model.conversion_outtype
            requires_quantization = _requires_native_quantization(source, outtype)
            cache_key = native_conversion_cache_key(source, outtype)
            cache_dir = native_conversion_cache_dir(cache_key)
            converted_model = cached_native_conversion(cache_dir, cache_key, outtype)
            if converted_model is None:
                lock = self._conversion_locks.setdefault(str(source.source_path), asyncio.Lock())
                async with lock:
                    converted_model = cached_native_conversion(cache_dir, cache_key, outtype)
                    if converted_model is None:
                        temp_dir = create_native_conversion_temp_dir()
                        working_directory = str(
                            source.source_path
                            if source.source_path.is_dir()
                            else source.source_path.parent
                        )
                        if source.kind == "gguf":
                            input_model = source.gguf_path or source.source_path
                        else:
                            if source_dir is None or python_executable is None:
                                raise NativeModelSourceError(
                                    "The llama.cpp converter and its optional Python interpreter "
                                    "are required for this source model."
                                )
                            converter_outtype = "f16" if requires_quantization else outtype
                            command = build_native_conversion_command(
                                source,
                                source_dir,
                                temp_dir,
                                python_executable,
                                converter_outtype,
                                model.upstream_model,
                            )
                            slot.command = command
                            slot.output.append(
                                f"Converting {source.kind} model to {converter_outtype} GGUF using llama.cpp."
                            )
                            await self._run_native_conversion(slot, command, working_directory)
                            input_model = find_native_gguf_output(temp_dir, converter_outtype)
                            if input_model is None:
                                raise NativeModelSourceError(
                                    "The upstream converter exited successfully but did not produce a GGUF file."
                                )

                        if requires_quantization:
                            if quantizer is None:
                                raise NativeModelSourceError(
                                    "Build the native llama-quantize target or set CORTEX_CPP_QUANTIZE."
                                )
                            output_model = temp_dir / f"cortex-model.{outtype}.gguf"
                            command = build_native_quantization_command(
                                quantizer,
                                input_model,
                                output_model,
                                outtype,
                                allow_requantize=model.allow_requantize,
                            )
                            slot.command = command
                            slot.output.append(
                                f"Quantizing GGUF weights to {outtype.upper()} with llama-quantize."
                            )
                            await self._run_native_conversion(slot, command, working_directory)

                        converted_model = publish_native_conversion(
                            temp_dir, cache_dir, cache_key, outtype
                        )
                        temp_dir = None

            slot.output.append(f"Using native GGUF model: {converted_model}")
            command = build_cortex_cpp_runtime_command(
                model,
                slot.port or 0,
                binary,
                model_path=converted_model,
            )
            slot.command = command
            slot.process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(converted_model.parent),
                env=_runtime_child_environment(self.settings),
            )
            slot.started_at = time.monotonic()
            slot.output_task = asyncio.create_task(
                self._read_output(slot), name=f"runtime-output-{model.id}"
            )
            slot.monitor_task = asyncio.create_task(
                self._monitor(model.id, slot), name=f"runtime-monitor-{model.id}"
            )
        except asyncio.CancelledError:
            process = slot.preparation_process
            if process is not None and process.returncode is None:
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
            raise
        except (OSError, RuntimeError, ValueError, NativeModelSourceError) as exc:
            slot.status = "error"
            slot.error = f"Cortex native model preparation failed: {exc}"
        finally:
            slot.preparation_process = None
            if temp_dir is not None:
                discard_native_conversion_temp(temp_dir)
            slot.preparation_task = None

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
            if slot.preparation_task is not None and not slot.preparation_task.done():
                return self.state(model_id)
            await self._cancel_tasks(slot)
            slot.process = None
            slot.preparation_process = None
            slot.preparation_task = None
            slot.port = None
            slot.started_at = None
            slot.model = model
            slot.status = "starting"
            slot.error = None
            slot.output.clear()

            try:
                model_path = (
                    Path(model.model_path).expanduser().resolve() if model.model_path else None
                )
            except (OSError, RuntimeError, ValueError) as exc:
                slot.status = "error"
                slot.error = f"Could not resolve local model path: {exc}"
                return self.state(model_id)
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
                if not binary and not self.settings.llama_server_path and auto_install_enabled():
                    try:
                        binary = str(
                            await asyncio.to_thread(install_latest_runtime_binary, "llama-server")
                        )
                    except RuntimeInstallError as exc:
                        slot.status = "error"
                        slot.error = (
                            "llama-server was not found and automatic installation from the "
                            f"GitHub release tag {RELEASE_TAG!r} failed: {exc}"
                        )
                        return self.state(model_id)
                if not binary:
                    slot.status = "error"
                    if self.settings.llama_server_path:
                        slot.error = (
                            "The configured llama-server path was not found. Check "
                            "server.llama_server_path / CORTEX_LLAMA_SERVER."
                        )
                    elif not auto_install_enabled():
                        slot.error = (
                            "llama-server was not found and automatic runtime installation is "
                            "disabled. Set CORTEX_AUTO_INSTALL_RUNTIMES=true, install llama-server "
                            "on PATH, or configure server.llama_server_path."
                        )
                    else:
                        slot.error = (
                            "llama-server was not found. Install a CPU/GPU-enabled llama.cpp build "
                            "or set server.llama_server_path / CORTEX_LLAMA_SERVER."
                        )
                    return self.state(model_id)
                try:
                    slot.port = self._free_port()
                except OSError as exc:
                    slot.status = "error"
                    slot.error = f"Could not reserve a local runtime port: {exc}"
                    return self.state(model_id)
                try:
                    slot.command = build_llama_command(binary, model, slot.port)
                except (OSError, RuntimeError, ValueError) as exc:
                    slot.status = "error"
                    slot.error = str(exc)
                    return self.state(model_id)
                cwd = str(model_path.parent)
                runtime_name = "llama-server"
            elif model.runtime == CORTEX_RUNTIME:
                if model_path is None or not model_path.is_file():
                    slot.status = "error"
                    slot.error = f"GGUF model file not found: {model_path or model.model_path}"
                    return self.state(model_id)
                try:
                    slot.port = self._free_port()
                    slot.command = build_cortex_runtime_command(model, slot.port)
                except (OSError, RuntimeError, ValueError) as exc:
                    slot.status = "error"
                    slot.error = str(exc)
                    return self.state(model_id)
                cwd = str(model_path.parent)
                runtime_name = "Cortex Python GGUF backend"
            elif model.runtime == CORTEX_CPP_RUNTIME:
                if model_path is None:
                    slot.status = "error"
                    slot.error = "Cortex's native C++ backend requires a model_path."
                    return self.state(model_id)
                binary = _resolve_cortex_cpp_binary()
                if not binary and not os.environ.get("CORTEX_CPP_HOST") and auto_install_enabled():
                    try:
                        binary = str(
                            await asyncio.to_thread(
                                install_latest_runtime_binary, "cortex-gguf-host"
                            )
                        )
                    except RuntimeInstallError as exc:
                        slot.status = "error"
                        slot.error = (
                            "Cortex's native C++ host was not found and automatic installation "
                            f"from GitHub release tag {RELEASE_TAG!r} failed: {exc}"
                        )
                        return self.state(model_id)
                if not binary:
                    slot.status = "error"
                    if os.environ.get("CORTEX_CPP_HOST"):
                        slot.error = (
                            "The configured Cortex C++ host was not found. Check CORTEX_CPP_HOST."
                        )
                    elif not auto_install_enabled():
                        slot.error = (
                            "Cortex's native C++ host was not found and automatic runtime "
                            "installation is disabled. Set CORTEX_AUTO_INSTALL_RUNTIMES=true, "
                            "build native/cortex-gguf-host, or set CORTEX_CPP_HOST."
                        )
                    else:
                        slot.error = (
                            "Cortex's native C++ host was not found. Build the native target, "
                            "set CORTEX_CPP_HOST, or check automatic installation from the "
                            f"GitHub release tag {RELEASE_TAG!r}."
                        )
                    return self.state(model_id)
                try:
                    source = inspect_native_model_source(model_path)
                except (OSError, RuntimeError, ValueError) as exc:
                    slot.status = "error"
                    slot.error = str(exc)
                    return self.state(model_id)
                if (
                    source.requires_unsafe_pickle
                    and not model.allow_unsafe_legacy_checkpoint_conversion
                ):
                    slot.status = "error"
                    slot.error = (
                        "Legacy .pt/.pth/.bin/.ckpt conversion can deserialize pickle data. "
                        "Only enable allow_unsafe_legacy_checkpoint_conversion for a trusted "
                        "LLaMA checkpoint."
                    )
                    return self.state(model_id)
                try:
                    slot.port = self._free_port()
                except OSError as exc:
                    slot.status = "error"
                    slot.error = f"Could not reserve a local runtime port: {exc}"
                    return self.state(model_id)

                requires_quantization = _requires_native_quantization(
                    source, model.conversion_outtype
                )
                quantizer = _resolve_cortex_cpp_quantizer(binary) if requires_quantization else None
                if (
                    requires_quantization
                    and quantizer is None
                    and not os.environ.get("CORTEX_CPP_QUANTIZE")
                    and auto_install_enabled()
                ):
                    try:
                        quantizer = str(
                            await asyncio.to_thread(install_latest_runtime_binary, "llama-quantize")
                        )
                    except RuntimeInstallError as exc:
                        slot.status = "error"
                        slot.error = (
                            "The selected output quantization needs llama-quantize, and automatic "
                            f"installation from release tag {RELEASE_TAG!r} failed: {exc}"
                        )
                        return self.state(model_id)
                if requires_quantization and quantizer is None:
                    slot.status = "error"
                    if os.environ.get("CORTEX_CPP_QUANTIZE"):
                        slot.error = (
                            "The configured llama-quantize executable was not found. Check "
                            "CORTEX_CPP_QUANTIZE."
                        )
                    elif not auto_install_enabled():
                        slot.error = (
                            "The selected GGUF quantization needs llama-quantize. Automatic "
                            "installation is disabled; build the llama-quantize target or set "
                            "CORTEX_CPP_QUANTIZE."
                        )
                    else:
                        slot.error = (
                            "The selected GGUF quantization needs llama-quantize. Build the "
                            "llama-quantize target, set CORTEX_CPP_QUANTIZE, or install the "
                            f"runtime bundle from release tag {RELEASE_TAG!r}."
                        )
                    return self.state(model_id)

                if source.kind == "gguf" and not requires_quantization:
                    try:
                        assert source.gguf_path is not None
                        slot.command = build_cortex_cpp_runtime_command(
                            model, slot.port, binary, model_path=source.gguf_path
                        )
                    except (OSError, RuntimeError, ValueError) as exc:
                        slot.status = "error"
                        slot.error = str(exc)
                        return self.state(model_id)
                    cwd = str(source.gguf_path.parent)
                    runtime_name = "Cortex native C++ backend"
                else:
                    source_dir = None
                    python_executable = None
                    if source.kind != "gguf":
                        source_dir = resolve_llama_cpp_source_dir()
                        if source_dir is None:
                            slot.status = "error"
                            slot.error = (
                                "Source-model conversion needs the llama.cpp source fetched by the "
                                "native CMake build. Build native/ or set CORTEX_LLAMA_CPP_SOURCE_DIR."
                            )
                            return self.state(model_id)
                        script = converter_script(source_dir, source.kind)
                        if not script.is_file():
                            slot.status = "error"
                            slot.error = f"The llama.cpp model converter was not found: {script}"
                            return self.state(model_id)
                        try:
                            python_executable = _resolve_cpp_converter_python()
                        except (OSError, PythonInterpreterError, RuntimeError, ValueError) as exc:
                            slot.status = "error"
                            slot.error = (
                                "A Python interpreter is used only for optional source conversion; "
                                f"it is not used for inference. {exc}"
                            )
                            return self.state(model_id)
                    slot.preparation_task = asyncio.create_task(
                        self._prepare_and_start_native_model(
                            model,
                            slot,
                            binary,
                            source,
                            source_dir,
                            python_executable,
                            quantizer,
                        ),
                        name=f"runtime-prepare-{model_id}",
                    )
                    return self.state(model_id)
            else:
                if model_path is not None and not model_path.exists():
                    slot.status = "error"
                    slot.error = f"Local model path not found: {model_path}"
                    return self.state(model_id)
                try:
                    slot.port = self._free_port()
                except OSError as exc:
                    slot.status = "error"
                    slot.error = f"Could not reserve a local runtime port: {exc}"
                    return self.state(model_id)
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
                    env=_runtime_child_environment(self.settings),
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
        pending = bytearray()
        try:
            while chunk := await process.stdout.read(4096):
                pending.extend(chunk)
                while (newline := pending.find(b"\n")) >= 0:
                    line = bytes(pending[:newline])
                    del pending[: newline + 1]
                    slot.output.append(line.decode("utf-8", errors="replace").rstrip()[-500:])
                while len(pending) > 64 * 1024:
                    fragment = bytes(pending[: 64 * 1024])
                    del pending[: 64 * 1024]
                    text = fragment.decode("utf-8", errors="replace").rstrip()
                    slot.output.append(f"{text[-480:]} … [continued]")
            if pending:
                slot.output.append(pending.decode("utf-8", errors="replace").rstrip()[-500:])
        except (asyncio.CancelledError, OSError):
            return

    async def _monitor(self, model_id: str, slot: RuntimeSlot) -> None:
        process = slot.process
        if process is None or slot.port is None:
            return
        health_url = f"http://127.0.0.1:{slot.port}{slot.model.health_path}"
        consecutive_failures = 0
        try:
            while process.returncode is None:
                if slot.status in {"stopping", "stopped"}:
                    await asyncio.sleep(0.1)
                    continue
                healthy = False
                try:
                    response = await self.client.get(health_url, timeout=0.75)
                    healthy = response.is_success
                except httpx.RequestError:
                    pass

                if healthy:
                    consecutive_failures = 0
                    slot.status = "running"
                    slot.error = None
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= 3 and slot.status == "running":
                        slot.status = "starting"
                        slot.error = "Local runtime health check failed; retrying."

                interval = 2.0 if slot.status == "running" and consecutive_failures == 0 else 0.5
                await asyncio.sleep(interval)
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
            preparation_task = slot.preparation_task
            if preparation_task is not None and not preparation_task.done():
                slot.status = "stopping"
                preparation_task.cancel()
                await asyncio.gather(preparation_task, return_exceptions=True)
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
            for task in (slot.monitor_task, slot.output_task, slot.preparation_task)
            if task is not None and not task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        slot.monitor_task = None
        slot.output_task = None
        slot.preparation_task = None
        slot.preparation_process = None

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
        path = None
        exists = False
        size = 0
        if model.model_path:
            try:
                path = Path(model.model_path).expanduser()
                path_stat = path.stat()
                exists = True
                size = path_stat.st_size if path.is_file() else 0
            except (OSError, RuntimeError, ValueError):
                path = None
                exists = False
        try:
            projector_exists = (
                model.mmproj_path is None or Path(model.mmproj_path).expanduser().is_file()
            )
        except (OSError, RuntimeError, ValueError):
            projector_exists = False
        runner_will_install = False
        if model.runtime == "llama.cpp":
            runner = self._resolve_binary()
            runner_will_install = bool(
                not runner
                and not self.settings.llama_server_path
                and auto_install_enabled()
                and release_target()
            )
            ready = bool(
                exists
                and path is not None
                and path.is_file()
                and path.suffix.lower() == ".gguf"
                and projector_exists
                and (runner or runner_will_install)
            )
            if runner:
                message = "Local GGUF, optional projector, and llama-server are ready to start."
            elif runner_will_install:
                message = (
                    "The local GGUF is ready; llama-server will be installed from the latest "
                    f"GitHub release tagged {RELEASE_TAG!r} when the model starts."
                )
            elif not auto_install_enabled():
                message = (
                    "Check the GGUF/projector paths and install/configure llama-server, or enable "
                    "automatic installation with CORTEX_AUTO_INSTALL_RUNTIMES=true."
                )
            else:
                message = "Check the GGUF/projector paths and install/configure llama-server."
        elif model.runtime == CORTEX_RUNTIME:
            runner, package_available = self._embedded_python_available()
            model_path_ready = bool(
                exists and path is not None and path.is_file() and path.suffix.lower() == ".gguf"
            )
            ready = bool(runner and package_available and model_path_ready)
            message = (
                "Cortex's built-in GGUF backend and local model are ready to start."
                if ready
                else "Install cortex-llmhoster[embedded] in the selected Python environment "
                "and check the GGUF model path."
            )
        elif model.runtime == CORTEX_CPP_RUNTIME:
            runner = _resolve_cortex_cpp_binary()
            runner_will_install = bool(
                not runner
                and not os.environ.get("CORTEX_CPP_HOST")
                and auto_install_enabled()
                and release_target()
            )
            source = None
            if model.model_path:
                try:
                    source = inspect_native_model_source(model.model_path)
                except (OSError, RuntimeError, ValueError):
                    source = None
            model_path_ready = source is not None
            requires_quantization = _requires_native_quantization(source, model.conversion_outtype)
            quantizer = _resolve_cortex_cpp_quantizer(runner) if requires_quantization else None
            quantizer_will_install = bool(
                requires_quantization
                and not quantizer
                and not os.environ.get("CORTEX_CPP_QUANTIZE")
                and auto_install_enabled()
                and release_target()
            )
            quantizer_ready = not requires_quantization or bool(quantizer or quantizer_will_install)
            converter_ready = True
            if source is not None and source.kind != "gguf":
                source_dir = resolve_llama_cpp_source_dir()
                converter_ready = bool(
                    source_dir is not None and converter_script(source_dir, source.kind).is_file()
                )
                try:
                    _resolve_cpp_converter_python()
                except (OSError, PythonInterpreterError, RuntimeError, ValueError):
                    converter_ready = False
                if (
                    source.requires_unsafe_pickle
                    and not model.allow_unsafe_legacy_checkpoint_conversion
                ):
                    converter_ready = False
            ready = bool(
                (runner or runner_will_install)
                and model_path_ready
                and converter_ready
                and quantizer_ready
                and model.mmproj_path is None
            )
            if not model_path_ready:
                message = "Select an existing GGUF, supported Hugging Face model directory, or legacy LLaMA checkpoint."
            elif (
                source is not None
                and source.requires_unsafe_pickle
                and not model.allow_unsafe_legacy_checkpoint_conversion
            ):
                message = (
                    "Legacy checkpoint conversion may deserialize pickle data; explicitly enable "
                    "the unsafe-conversion option only for a trusted checkpoint."
                )
            elif requires_quantization and not quantizer_ready:
                message = (
                    "The selected lower-bit output needs the native llama-quantize executable; "
                    "build the llama-quantize target or set CORTEX_CPP_QUANTIZE."
                )
            elif not runner and runner_will_install:
                message = (
                    f"Cortex's native C++ host will be installed from the latest GitHub release "
                    f"tagged {RELEASE_TAG!r} when the model starts."
                )
            elif not runner and os.environ.get("CORTEX_CPP_HOST"):
                message = "The configured Cortex C++ host was not found; check CORTEX_CPP_HOST."
            elif not runner and not auto_install_enabled():
                message = (
                    "Build native/cortex-gguf-host or set CORTEX_CPP_HOST; automatic installation "
                    "is disabled by CORTEX_AUTO_INSTALL_RUNTIMES."
                )
            elif not runner:
                message = "Build native/cortex-gguf-host or set CORTEX_CPP_HOST."
            elif source is not None and source.kind != "gguf" and not converter_ready:
                message = (
                    "Source weights need the llama.cpp converter scripts and a Python interpreter "
                    "for one-time conversion; inference remains native C++."
                )
            elif model.mmproj_path is not None:
                message = "The native C++ GGUF adapter does not support projector files."
            elif source is not None and source.kind == "gguf" and requires_quantization:
                message = (
                    "Cortex's native C++ host and quantizer are ready. Already-quantized input "
                    "may need to be requantized from a higher-precision source."
                )
            elif source is not None and source.kind == "gguf":
                message = "Cortex's native C++ host and GGUF model are ready to start."
            else:
                message = (
                    "Cortex's native C++ host and source model are ready for conversion. "
                    "The converter's optional framework dependencies may still be required."
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
            "runner_will_install": runner_will_install,
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
            "pid": (
                slot.process.pid
                if slot.process is not None
                else slot.preparation_process.pid
                if slot.preparation_process is not None
                else None
            ),
            "port": slot.port,
            "base_url": self.local_base_url(model),
            "uptime_seconds": uptime,
            "command": list(slot.command),
            "logs": list(slot.output),
        }

    def states(self) -> dict[str, dict[str, Any]]:
        return {model_id: self.state(model_id) for model_id in self.settings.models}

    async def close(self) -> None:
        """Stop every managed process concurrently so shutdown time stays bounded."""

        model_ids = list(self._slots)
        results = await asyncio.gather(
            *(self.stop(model_id) for model_id in model_ids), return_exceptions=True
        )
        for model_id, result in zip(model_ids, results, strict=True):
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, Exception):
                logger.error(
                    "Could not stop local runtime %s during shutdown: %s", model_id, result
                )


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
        "gpu_vram_total_gb": None,
        "gpu_vram_free_gb": None,
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
                [
                    nvidia_smi,
                    "--query-gpu=name,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                check=True,
                text=True,
                timeout=1.5,
            )
            summaries: list[str] = []
            total_mib = 0
            free_mib = 0
            for row in csv.reader(io.StringIO(probe.stdout)):
                if len(row) < 3:
                    continue
                try:
                    device_total = int(row[1].strip())
                    device_free = int(row[2].strip())
                except ValueError:
                    continue
                total_mib += device_total
                free_mib += device_free
                summaries.append(f"{row[0].strip()} · {device_free}/{device_total} MiB free")
            result["gpu"] = "; ".join(summaries) or "NVIDIA GPU detected"
            if total_mib > 0:
                result["gpu_vram_total_gb"] = round(total_mib / 1024, 1)
                result["gpu_vram_free_gb"] = round(free_mib / 1024, 1)
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
