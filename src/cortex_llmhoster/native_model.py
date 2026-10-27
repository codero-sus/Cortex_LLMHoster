"""Source-model detection and GGUF conversion helpers for the native C++ runtime.

llama.cpp's C inference API loads GGUF. Hugging Face and legacy LLaMA assets are
converted once with the pinned upstream conversion scripts, then inference is
performed only by Cortex's native C++ process.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .config import CPP_CONVERSION_OUTTYPES, CPP_QUANTIZER_OUTTYPES

NativeModelKind = Literal["gguf", "huggingface", "legacy-llama"]
CONVERSION_OUTTYPES = CPP_CONVERSION_OUTTYPES
DIRECT_CONVERSION_OUTTYPES = frozenset({"f32", "f16", "q8_0"})
QUANTIZATION_OUTTYPES = CPP_QUANTIZER_OUTTYPES
_WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin", ".pt", ".pth", ".ckpt"})
_GGUF_SHARD_PATTERN = re.compile(
    r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$", re.IGNORECASE
)


class NativeModelSourceError(ValueError):
    """The configured path is not a supported native GGUF model source."""


@dataclass(frozen=True, slots=True)
class NativeModelSource:
    """Detected local model source and the path expected by its converter."""

    kind: NativeModelKind
    configured_path: Path
    source_path: Path
    gguf_path: Path | None = None
    requires_unsafe_pickle: bool = False


def _gguf_shard_metadata(path: Path) -> tuple[str, int, int] | None:
    match = _GGUF_SHARD_PATTERN.fullmatch(path.name)
    if match is None:
        return None
    return match.group("prefix"), int(match.group("index")), int(match.group("count"))


def _resolve_sharded_gguf(path: Path) -> Path | None:
    """Validate a split GGUF set and return its first shard, when the name is standard."""

    metadata = _gguf_shard_metadata(path)
    if metadata is None:
        return None
    prefix, _, count = metadata
    shards: dict[int, Path] = {}
    for candidate in path.parent.iterdir():
        if not candidate.is_file() or candidate.suffix.lower() != ".gguf":
            continue
        candidate_metadata = _gguf_shard_metadata(candidate)
        if candidate_metadata is None:
            continue
        candidate_prefix, index, candidate_count = candidate_metadata
        if candidate_prefix == prefix and candidate_count == count:
            shards[index] = candidate
    if set(shards) != set(range(1, count + 1)):
        raise NativeModelSourceError(
            f"The split GGUF shard set for {path.name} is incomplete; expected "
            f"{count} consecutively numbered shards in {path.parent}."
        )
    return shards[1]


def _resolve_gguf_shard_group(paths: list[Path]) -> Path | None:
    metadata = [_gguf_shard_metadata(path) for path in paths]
    if not metadata or any(item is None for item in metadata):
        return None
    groups = {(item[0], item[2]) for item in metadata if item is not None}
    if len(groups) != 1:
        return None
    prefix, count = next(iter(groups))
    indices = {item[1] for item in metadata if item is not None}
    if len(paths) != count or indices != set(range(1, count + 1)):
        raise NativeModelSourceError(
            f"The split GGUF shard set for {prefix} is incomplete or inconsistent; "
            f"expected {count} consecutively numbered files."
        )
    return next(path for path, item in zip(paths, metadata, strict=True) if item[1] == 1)


def _has_huggingface_weights(directory: Path) -> bool:
    known_names = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    if any((directory / name).is_file() for name in known_names):
        return True
    return any(
        candidate.is_file()
        for pattern in ("model-*.safetensors", "pytorch_model-*.bin")
        for candidate in directory.glob(pattern)
    )


def inspect_native_model_source(model_path: str | Path) -> NativeModelSource:
    """Recognize GGUF, standard Hugging Face, and legacy LLaMA model layouts."""

    try:
        path = Path(model_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise NativeModelSourceError(f"Could not resolve model source path: {exc}") from exc

    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".gguf":
            first_shard = _resolve_sharded_gguf(path)
            if first_shard is not None and first_shard != path:
                return NativeModelSource("gguf", path, path.parent, first_shard)
            return NativeModelSource("gguf", path, path, path)
        if suffix not in _WEIGHT_SUFFIXES:
            raise NativeModelSourceError(
                "Cortex native C++ accepts GGUF, Hugging Face SafeTensors/PyTorch weights, "
                "or supported legacy LLaMA checkpoints."
            )
        if (path.parent / "config.json").is_file() and _has_huggingface_weights(path.parent):
            return NativeModelSource("huggingface", path, path.parent)
        return NativeModelSource(
            "legacy-llama", path, path, requires_unsafe_pickle=suffix != ".safetensors"
        )

    if not path.is_dir():
        raise NativeModelSourceError(
            "The native model path must be a GGUF file or model directory."
        )

    gguf_files = sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix.lower() == ".gguf"
    )
    if len(gguf_files) == 1:
        first_shard = _resolve_sharded_gguf(gguf_files[0])
        return NativeModelSource("gguf", path, path, first_shard or gguf_files[0])
    if len(gguf_files) > 1:
        first_shard = _resolve_gguf_shard_group(gguf_files)
        if first_shard is not None:
            return NativeModelSource("gguf", path, path, first_shard)
        raise NativeModelSourceError(
            "The directory contains multiple GGUF models; set model_path to the intended file."
        )

    if (path / "config.json").is_file() and _has_huggingface_weights(path):
        return NativeModelSource("huggingface", path, path)

    legacy_files = [
        candidate
        for suffix in _WEIGHT_SUFFIXES
        for candidate in path.glob(f"*{suffix}")
        if candidate.is_file()
    ]
    if legacy_files:
        requires_unsafe_pickle = any(
            candidate.suffix.lower() != ".safetensors" for candidate in legacy_files
        )
        return NativeModelSource(
            "legacy-llama", path, path, requires_unsafe_pickle=requires_unsafe_pickle
        )

    raise NativeModelSourceError(
        "No GGUF file or supported model weights were found. For Hugging Face models, "
        "select a directory containing config.json and model.safetensors or pytorch_model.bin."
    )


def resolve_llama_cpp_source_dir() -> Path | None:
    """Find the source tree fetched by CMake, or an explicitly configured checkout."""

    configured = os.environ.get("CORTEX_LLAMA_CPP_SOURCE_DIR")
    if configured:
        try:
            source_dir = Path(configured).expanduser().resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            return None
        return source_dir if source_dir.is_dir() else None

    project_root = Path(__file__).resolve().parents[2]
    candidate = project_root / "native" / "build" / "_deps" / "llama_cpp-src"
    return candidate if candidate.is_dir() else None


def converter_script(source_dir: Path, kind: NativeModelKind) -> Path:
    """Return the pinned llama.cpp converter appropriate for a source format."""

    if kind == "huggingface":
        return source_dir / "convert_hf_to_gguf.py"
    if kind == "legacy-llama":
        return source_dir / "examples" / "convert_legacy_llama.py"
    raise NativeModelSourceError("GGUF sources do not require conversion.")


def build_native_conversion_command(
    source: NativeModelSource,
    source_dir: Path,
    output_dir: Path,
    python_executable: str,
    outtype: str,
    model_name: str,
) -> list[str]:
    """Build an argv-only command for the pinned upstream model converter."""

    if outtype not in DIRECT_CONVERSION_OUTTYPES:
        raise NativeModelSourceError(
            "The upstream converter can emit f16, f32, or q8_0 directly; lower-bit GGUF "
            "formats need llama-quantize."
        )
    script = converter_script(source_dir, source.kind)
    if not script.is_file():
        raise NativeModelSourceError(f"The llama.cpp converter was not found: {script}")

    output_template = output_dir / "cortex-model.{ftype}.gguf"
    command = [
        python_executable,
        str(script),
        "--outfile",
        str(output_template),
        "--outtype",
        outtype,
        "--model-name",
        model_name,
    ]
    if source.kind == "huggingface":
        command.extend(("--split-max-size", "4G"))
    else:
        command.extend(("--concurrency", "4"))
        if source.source_path.is_file():
            command.extend(("--vocab-dir", str(source.source_path.parent)))
    command.append(str(source.source_path))
    return command


def build_native_quantization_command(
    binary: str,
    input_path: Path,
    output_path: Path,
    outtype: str,
    *,
    allow_requantize: bool = False,
) -> list[str]:
    """Build an argv-only command for llama.cpp's native GGUF quantizer."""

    if outtype not in QUANTIZATION_OUTTYPES:
        raise NativeModelSourceError(
            f"Quantization target must be one of: {', '.join(sorted(QUANTIZATION_OUTTYPES))}."
        )
    if not binary:
        raise NativeModelSourceError("The llama.cpp llama-quantize executable was not found.")
    if input_path.resolve() == output_path.resolve():
        raise NativeModelSourceError("Quantization must write to a different output file.")
    command = [binary]
    if allow_requantize:
        command.append("--allow-requantize")
    command.extend((str(input_path), str(output_path), outtype.upper()))
    return command


def native_conversion_cache_key(source: NativeModelSource, outtype: str) -> str:
    """Fingerprint source paths and stat data without reading multi-gigabyte weights."""

    digest = hashlib.sha256()
    digest.update(source.kind.encode("ascii"))
    digest.update(b"\0")
    digest.update(outtype.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(source.source_path).encode("utf-8", errors="surrogateescape"))
    digest.update(b"\0")

    if source.source_path.is_file():
        files = [source.source_path]
        if source.kind == "legacy-llama":
            files = sorted(
                candidate
                for candidate in source.source_path.parent.iterdir()
                if candidate.is_file()
            )
    else:
        ignored_directories = {".git", ".cache", "__pycache__", "node_modules"}
        files = []
        for root, directory_names, file_names in os.walk(source.source_path, followlinks=False):
            directory_names[:] = sorted(
                name for name in directory_names if name not in ignored_directories
            )
            for name in file_names:
                candidate = Path(root) / name
                if candidate.is_file():
                    files.append(candidate)
        files.sort()

    for file_path in files:
        try:
            stat = file_path.stat()
        except OSError as exc:
            raise NativeModelSourceError(
                f"Could not inspect model file {file_path}: {exc}"
            ) from exc
        relative_path = (
            file_path.name
            if source.source_path.is_file()
            else file_path.relative_to(source.source_path).as_posix()
        )
        digest.update(relative_path.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ino}".encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def native_conversion_cache_root() -> Path:
    """Use an external per-user cache so conversions never pollute the checkout."""

    configured = os.environ.get("CORTEX_CPP_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    cache_home = os.environ.get("XDG_CACHE_HOME")
    if cache_home:
        root = Path(cache_home).expanduser()
    elif os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        root = Path(os.environ["LOCALAPPDATA"]).expanduser()
    else:
        root = Path.home() / ".cache"
    return root / "cortex-llmhoster" / "converted"


def native_conversion_cache_dir(key: str) -> Path:
    return native_conversion_cache_root() / key


def _find_gguf_file(directory: Path, outtype: str) -> Path | None:
    direct = directory / f"cortex-model.{outtype}.gguf"
    if direct.is_file() and direct.stat().st_size > 0:
        return direct
    shards = sorted(directory.glob(f"cortex-model.{outtype}-00001-of-*.gguf"))
    return shards[0] if shards and shards[0].stat().st_size > 0 else None


def find_native_gguf_output(directory: Path, outtype: str) -> Path | None:
    """Find the direct GGUF or first shard produced for the requested format."""

    return _find_gguf_file(directory, outtype)


def cached_native_conversion(cache_dir: Path, key: str, outtype: str) -> Path | None:
    """Return a fully published converted GGUF, never a partial conversion."""

    marker = cache_dir / ".complete"
    try:
        if marker.read_text(encoding="ascii").strip() != key:
            return None
    except OSError:
        return None
    return _find_gguf_file(cache_dir, outtype)


def create_native_conversion_temp_dir() -> Path:
    """Create a unique staging directory on the same filesystem as the cache."""

    root = native_conversion_cache_root()
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=".cortex-convert-", dir=root))


def publish_native_conversion(temp_dir: Path, cache_dir: Path, key: str, outtype: str) -> Path:
    """Atomically publish a successful converter output and return its first GGUF shard."""

    converted = _find_gguf_file(temp_dir, outtype)
    if converted is None:
        raise NativeModelSourceError(
            "The upstream converter exited successfully but did not produce a GGUF file."
        )
    (temp_dir / ".complete").write_text(key, encoding="ascii")

    existing = cached_native_conversion(cache_dir, key, outtype)
    if existing is not None:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return existing
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    os.replace(temp_dir, cache_dir)
    result = _find_gguf_file(cache_dir, outtype)
    if result is None:
        raise NativeModelSourceError("The converted GGUF cache could not be verified.")
    return result


def discard_native_conversion_temp(temp_dir: Path) -> None:
    """Remove partial outputs after a failed or cancelled conversion."""

    shutil.rmtree(temp_dir, ignore_errors=True)
