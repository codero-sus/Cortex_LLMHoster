"""Cross-platform selection for Python child processes and CLI startup."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

PYTHON_OVERRIDE_ENV = "2PY2"
PYTHON_OVERRIDE_FILE = "python.env"
_REEXEC_MARKER = "_CORTEX_PYTHON_REEXEC"


class PythonInterpreterError(ValueError):
    """A configured Python executable could not be selected or started."""


def _project_root() -> Path:
    """Find the source project root, falling back to the launch directory."""

    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def _python_env_value(root: Path) -> str | None:
    path = root / PYTHON_OVERRIDE_FILE
    try:
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise PythonInterpreterError(f"Could not read {path}: {exc}") from exc

    if not lines:
        return None
    if len(lines) != 1:
        raise PythonInterpreterError(
            f"{path} must contain one Python executable path (blank lines and comments are allowed)."
        )

    value = lines[0]
    key, separator, assigned_value = value.partition("=")
    if separator and key.strip().lower() in {
        "2py2",
        "python",
        "python_exe",
        "python_executable",
    }:
        value = assigned_value.strip()
    return value or None


def _unquote(value: str, source: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    elif value.startswith(('"', "'")) or value.endswith(('"', "'")):
        raise PythonInterpreterError(f"{source} has unmatched quotes around the Python path.")
    if not value:
        raise PythonInterpreterError(f"{source} does not specify a Python executable path.")
    return os.path.expandvars(os.path.expanduser(value))


def _is_executable_file(path: Path) -> bool:
    return path.is_file() and (os.name == "nt" or os.access(path, os.X_OK))


def _resolve_configured_path(value: str, root: Path, source: str) -> str:
    value = _unquote(value, source)
    candidate = Path(value)
    if not candidate.is_absolute():
        project_candidate = root / candidate
        if _is_executable_file(project_candidate):
            candidate = project_candidate
        else:
            path_match = shutil.which(value)
            candidate = Path(path_match) if path_match else project_candidate

    candidate = Path(os.path.abspath(candidate))
    if not candidate.exists():
        raise PythonInterpreterError(f"Python executable from {source} was not found: {value}")
    if not _is_executable_file(candidate):
        qualifier = "a file" if os.name == "nt" else "an executable file"
        raise PythonInterpreterError(
            f"Python path from {source} must point to {qualifier}: {candidate}"
        )
    # Preserve symlink paths: virtual environments often rely on their own
    # python executable path to establish the correct sys.prefix.
    return str(candidate)


def resolve_python_executable(
    *,
    project_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Return the configured Python, or the interpreter currently running Cortex.

    The ``2PY2`` environment variable takes precedence over ``python.env``.
    Relative configured paths are resolved from the project root; bare executable
    names may also be found on ``PATH``. The default is ``sys.executable``.
    """

    root = Path(project_root).expanduser().resolve() if project_root else _project_root()
    env = os.environ if environ is None else environ
    override = env.get(PYTHON_OVERRIDE_ENV, "").strip()
    if override:
        return _resolve_configured_path(
            override, root, f"environment variable {PYTHON_OVERRIDE_ENV}"
        )

    configured = _python_env_value(root)
    if configured:
        return _resolve_configured_path(configured, root, str(root / PYTHON_OVERRIDE_FILE))

    current = sys.executable
    if not current:
        raise PythonInterpreterError(
            "No Python executable is available; set 2PY2 or add a path to python.env."
        )
    return current


def reexec_with_selected_python() -> None:
    """Restart the CLI under the selected interpreter when it differs."""

    if os.environ.get(_REEXEC_MARKER) == "1":
        return
    selected = resolve_python_executable()
    current = sys.executable
    same_interpreter = bool(
        current
        and os.path.normcase(os.path.abspath(selected))
        == os.path.normcase(os.path.abspath(current))
    )
    if same_interpreter:
        return

    os.environ[_REEXEC_MARKER] = "1"
    arguments = [selected, "-m", "cortex_llmhoster", *sys.argv[1:]]
    try:
        if os.name == "nt":
            completed = subprocess.run(arguments, env=os.environ.copy(), check=False)
            raise SystemExit(completed.returncode)
        os.execv(selected, arguments)
    except OSError as exc:
        os.environ.pop(_REEXEC_MARKER, None)
        raise PythonInterpreterError(
            f"Could not restart Cortex with the selected Python executable {selected}: {exc}"
        ) from exc
