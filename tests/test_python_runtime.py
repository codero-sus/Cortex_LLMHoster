from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cortex_llmhoster import python_runtime
from cortex_llmhoster.python_runtime import (
    PythonInterpreterError,
    reexec_with_selected_python,
    resolve_python_executable,
)


def executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_python_env_selects_relative_project_path(tmp_path: Path) -> None:
    selected = executable(tmp_path / "portable" / "python")
    (tmp_path / "python.env").write_text("portable/python\n", encoding="utf-8")

    assert resolve_python_executable(project_root=tmp_path, environ={}) == str(selected.resolve())


def test_2py2_environment_override_takes_precedence(tmp_path: Path) -> None:
    from_file = executable(tmp_path / "python-from-file")
    from_env = executable(tmp_path / "python-from-env")
    (tmp_path / "python.env").write_text(str(from_file), encoding="utf-8")

    selected = resolve_python_executable(
        project_root=tmp_path,
        environ={"2PY2": str(from_env)},
    )

    assert selected == str(from_env.resolve())


def test_python_env_accepts_a_quoted_2py2_assignment(tmp_path: Path) -> None:
    selected = executable(tmp_path / "portable python")
    (tmp_path / "python.env").write_text(f'\ufeff2PY2="{selected}"\n', encoding="utf-8")

    assert resolve_python_executable(project_root=tmp_path, environ={}) == str(selected.resolve())


@pytest.mark.skipif(
    os.name == "nt", reason="creating symlinks may require elevated Windows permissions"
)
def test_python_env_preserves_virtualenv_symlink_path(tmp_path: Path) -> None:
    target = executable(tmp_path / "base-python")
    selected = tmp_path / "venv" / "bin" / "python"
    selected.parent.mkdir(parents=True)
    selected.symlink_to(target)
    (tmp_path / "python.env").write_text("venv/bin/python\n", encoding="utf-8")

    assert resolve_python_executable(project_root=tmp_path, environ={}) == str(selected)


def test_python_selection_defaults_to_the_running_interpreter(tmp_path: Path) -> None:
    assert resolve_python_executable(project_root=tmp_path, environ={}) == sys.executable


def test_python_env_rejects_missing_executable(tmp_path: Path) -> None:
    (tmp_path / "python.env").write_text("missing/python\n", encoding="utf-8")

    with pytest.raises(PythonInterpreterError, match="was not found"):
        resolve_python_executable(project_root=tmp_path, environ={})


def test_python_env_rejects_multiple_paths(tmp_path: Path) -> None:
    (tmp_path / "python.env").write_text("python-a\npython-b\n", encoding="utf-8")

    with pytest.raises(PythonInterpreterError, match="must contain one Python executable"):
        resolve_python_executable(project_root=tmp_path, environ={})


def test_cli_reexecs_under_the_selected_interpreter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selected = executable(tmp_path / "portable-python")
    current = executable(tmp_path / "current-python")
    calls: dict[str, object] = {}

    class ReexecCalled(Exception):
        pass

    def capture_execv(path: str, arguments: list[str]) -> None:
        calls["path"] = path
        calls["arguments"] = arguments
        raise ReexecCalled

    monkeypatch.setattr(python_runtime, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "executable", str(current))
    monkeypatch.setattr(sys, "argv", ["cortex-llmhoster", "--port", "9000"])
    monkeypatch.setattr(os, "execv", capture_execv)
    monkeypatch.setenv("2PY2", str(selected))
    monkeypatch.setenv(python_runtime._REEXEC_MARKER, "0")

    with pytest.raises(ReexecCalled):
        reexec_with_selected_python()

    assert calls["path"] == str(selected.resolve())
    assert calls["arguments"] == [
        str(selected.resolve()),
        "-m",
        "cortex_llmhoster",
        "--port",
        "9000",
    ]
    assert os.environ[python_runtime._REEXEC_MARKER] == "1"


def test_windows_cli_reexec_uses_subprocess_and_propagates_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selected = executable(tmp_path / "portable-python.exe")
    current = executable(tmp_path / "current-python.exe")
    calls: dict[str, object] = {}

    def fake_run(arguments: list[str], *, env: dict[str, str], check: bool) -> SimpleNamespace:
        calls["arguments"] = arguments
        calls["env"] = env
        calls["check"] = check
        return SimpleNamespace(returncode=7)

    windows_os = SimpleNamespace(name="nt", environ=os.environ, path=os.path)
    monkeypatch.setattr(python_runtime, "os", windows_os)
    monkeypatch.setattr(python_runtime.subprocess, "run", fake_run)
    monkeypatch.setattr(python_runtime, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "executable", str(current))
    monkeypatch.setattr(sys, "argv", ["cortex-llmhoster", "--port", "9100"])
    monkeypatch.setenv("2PY2", str(selected))
    monkeypatch.setenv(python_runtime._REEXEC_MARKER, "0")

    with pytest.raises(SystemExit) as exit_info:
        reexec_with_selected_python()

    assert exit_info.value.code == 7
    assert calls["arguments"] == [
        str(selected),
        "-m",
        "cortex_llmhoster",
        "--port",
        "9100",
    ]
    assert calls["env"][python_runtime._REEXEC_MARKER] == "1"
    assert calls["check"] is False
