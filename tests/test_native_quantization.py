from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from cortex_llmhoster.config import ModelConfig, Settings
from cortex_llmhoster.native_model import inspect_native_model_source
from cortex_llmhoster.runtime import LocalRuntimeManager, RuntimeSlot


@pytest.mark.parametrize("outtype", ("q4_k_m", "q8_0"))
@pytest.mark.asyncio
async def test_cortex_cpp_quantizes_gguf_natively_without_python(
    tmp_path: Path, monkeypatch, outtype: str
) -> None:
    model_file = tmp_path / "teacher.gguf"
    model_file.write_bytes(b"unquantized GGUF fixture")
    source = inspect_native_model_source(model_file)
    model = ModelConfig(
        id="quantized-student",
        upstream_model="quantized-student",
        runtime="cortex-cpp",
        model_path=str(model_file),
        conversion_outtype=outtype,
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
        observed["host_args"] = list(args)
        observed["host_kwargs"] = kwargs
        return FakeProcess()

    async def fake_quantize(
        slot: RuntimeSlot, command: list[str], _working_directory: str
    ) -> None:
        observed["quantize_command"] = command
        Path(command[2]).write_bytes(b"quantized GGUF fixture")
        slot.output.append("fake native quantizer completed")

    async with httpx.AsyncClient() as client:
        manager = LocalRuntimeManager(settings, client, hardware={})
        slot = RuntimeSlot(model=model, status="starting", port=9236)
        manager._slots[model.id] = slot
        monkeypatch.setattr(manager, "_run_native_conversion", fake_quantize)
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
            None,
            None,
            "/opt/llama-quantize",
        )

        quantize_command = observed["quantize_command"]
        assert isinstance(quantize_command, list)
        assert quantize_command[0] == "/opt/llama-quantize"
        assert quantize_command[1] == str(model_file)
        assert quantize_command[-1] == outtype.upper()
        command = observed["host_args"]
        assert isinstance(command, list)
        assert command[0] == "/opt/cortex-gguf-host"
        quantized_model = Path(command[command.index("--model-path") + 1])
        assert quantized_model.name == f"cortex-model.{outtype}.gguf"
        assert quantized_model.read_bytes() == b"quantized GGUF fixture"
        assert "fake native quantizer completed" in slot.output

        await manager.close()
