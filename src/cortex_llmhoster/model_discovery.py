"""Local GGUF discovery and best-effort task suggestions for the model editor."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_MAX_DISCOVERED_FILES = 500
_MAX_SCANNED_DIRECTORIES = 10_000
_MAX_SCAN_DEPTH = 5
_METADATA_SAMPLE_BYTES = 2 * 1024 * 1024

_VISION_MARKERS = (
    "vision",
    "vlm",
    "llava",
    "mllama",
    "qwen2vl",
    "qwen2_vl",
    "qwen2.5vl",
    "qwen2.5_vl",
    "qwen3vl",
    "qwen3_vl",
    "internvl",
    "minicpmv",
    "minicpm_v",
    "pixtral",
    "molmo",
    "smolvlm",
    "phi4mm",
    "phi4_mm",
    "idefics",
    "paligemma",
    "gemma3",
    "llama4",
    "qwen_vl",
    "image_understanding",
)
_EMBEDDING_MARKERS = (
    "embedding",
    "nomic_embed",
    "bge",
    "e5_",
    "gte_embed",
    "jina_embed",
    "text_embed",
)
_OCR_MARKERS = ("ocr", "donut", "trocr", "docvqa", "document_qa", "paddleocr")
_AUDIO_TRANSCRIPTION_MARKERS = (
    "whisper",
    "asr",
    "stt",
    "transcription",
    "speech_to_text",
    "speech_recognition",
    "canary",
)
_AUDIO_UNDERSTANDING_MARKERS = (
    "audio_understanding",
    "audio_perception",
    "qwen_audio",
    "ultravox",
    "voxtral",
    "audio_classification",
)
_AUDIO_GENERATION_MARKERS = (
    "tts",
    "text_to_speech",
    "speech_generation",
    "audio_generation",
    "parler",
    "fish_speech",
    "kokoro",
    "orpheus",
    "xtts",
    "vits",
    "bark",
)
_IMAGE_TO_IMAGE_MARKERS = ("image_to_image", "i2i", "img2img", "inpainting", "image_edit")
_TEXT_TO_IMAGE_MARKERS = ("text_to_image", "t2i")
_IMAGE_GENERATION_MARKERS = (
    "stable_diffusion",
    "sdxl",
    "sd3",
    "flux",
    "diffusion",
    "image_generation",
    "image_gen",
)
_IMAGE_TO_VIDEO_MARKERS = ("image_to_video", "i2v")
_TEXT_TO_VIDEO_MARKERS = ("text_to_video", "t2v")
_VIDEO_TO_VIDEO_MARKERS = ("video_to_video", "v2v")
_VIDEO_GENERATION_MARKERS = (
    "video_generation",
    "video_gen",
    "wan2",
    "wan2.1",
    "ltx_video",
    "cogvideo",
    "hunyuan_video",
    "mochi",
    "video_diffusion",
)
_VIDEO_UNDERSTANDING_MARKERS = (
    "video_understanding",
    "video_qa",
    "video_caption",
    "video_summar",
    "video_llm",
)
_CAPABILITY_ORDER = (
    "text_generation",
    "text_summarization",
    "text_translation",
    "text_classification",
    "vision_understanding",
    "ocr",
    "image_captioning",
    "object_detection",
    "image_generation",
    "text_to_image",
    "image_to_image",
    "audio_transcription",
    "audio_understanding",
    "audio_generation",
    "text_to_speech",
    "speech_to_speech",
    "video_understanding",
    "video_generation",
    "text_to_video",
    "image_to_video",
    "video_to_video",
    "embeddings",
    "reranking",
    "moderation",
)


def model_search_roots(configured_model_paths: Iterable[str | None] = ()) -> list[Path]:
    """Return existing, user-relevant directories that may contain local GGUFs."""

    candidates: list[Path] = []
    configured_roots = os.environ.get("CORTEX_MODELS_DIR", "")
    candidates.extend(
        Path(value.strip()).expanduser()
        for value in configured_roots.split(os.pathsep)
        if value.strip()
    )
    candidates.extend(
        (
            Path.cwd() / "models",
            Path.home() / "models",
            Path.home() / ".cache" / "huggingface" / "hub",
        )
    )
    for configured_path in configured_model_paths:
        if not configured_path:
            continue
        path = Path(configured_path).expanduser()
        candidates.append(path if path.is_dir() else path.parent)

    roots: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            root = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if root.is_dir() and root not in seen:
            roots.append(root)
            seen.add(root)
    return roots


def discover_local_model_files(
    configured_model_paths: Iterable[str | None] = (),
) -> dict[str, Any]:
    """List up to 500 GGUF files from known local model folders, without reading weights."""

    roots = model_search_roots(configured_model_paths)
    found: dict[Path, dict[str, Any]] = {}
    scanned_directories = 0
    truncated = False
    for root in roots:
        try:
            walker = os.walk(root, followlinks=False)
            for folder, directory_names, filenames in walker:
                if scanned_directories >= _MAX_SCANNED_DIRECTORIES:
                    truncated = True
                    break
                scanned_directories += 1
                current = Path(folder)
                depth = len(current.relative_to(root).parts)
                if depth >= _MAX_SCAN_DEPTH:
                    directory_names[:] = []
                else:
                    directory_names[:] = sorted(
                        name
                        for name in directory_names
                        if not name.startswith(".") and not (current / name).is_symlink()
                    )
                for filename in filenames:
                    if not filename.lower().endswith(".gguf"):
                        continue
                    candidate = current / filename
                    try:
                        resolved = candidate.resolve(strict=True)
                        if not resolved.is_file():
                            continue
                        stat = resolved.stat()
                    except (OSError, RuntimeError):
                        continue
                    found.setdefault(
                        resolved,
                        {
                            "path": str(resolved),
                            "name": filename,
                            "size_bytes": stat.st_size,
                        },
                    )
                    if len(found) >= _MAX_DISCOVERED_FILES:
                        truncated = True
                        break
                if truncated:
                    break
        except OSError:
            continue
        if truncated:
            break

    files = sorted(found.values(), key=lambda entry: (str(entry["name"]).casefold(), entry["path"]))
    return {
        "files": files,
        "search_roots": [str(root) for root in roots],
        "truncated": truncated,
    }


def _normalized_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _has_marker(normalized_text: str, marker: str) -> bool:
    normalized_marker = _normalized_text(marker)
    if not normalized_marker:
        return False
    if len(normalized_marker) <= 3:
        return f"_{normalized_marker}_" in f"_{normalized_text}_"
    return normalized_marker in normalized_text


def _matches(text: str, markers: Iterable[str]) -> bool:
    normalized = _normalized_text(text)
    return any(_has_marker(normalized, marker) for marker in markers)


def detect_model_capabilities(
    model_path: str,
    *,
    model_id: str = "",
    mmproj_path: str | None = None,
) -> dict[str, Any]:
    """Suggest capability checkboxes from a model name, GGUF metadata, and projector.

    This is intentionally a suggestion, not a claim that a runtime implements a
    task: model tags are inconsistent, so the user reviews the selected tasks.
    Only a small GGUF metadata prefix is read; tensor data and weights are never
    loaded into memory.
    """

    path = Path(model_path).expanduser()
    if not path.exists():
        raise ValueError(f"Model path does not exist: {path}")
    if not (path.is_file() or path.is_dir()):
        raise ValueError("Choose a model file or local model directory.")

    context_parts = [model_id, path.name, *path.parts[-4:]]
    evidence: list[str] = []
    if path.is_file() and path.suffix.lower() == ".gguf":
        try:
            with path.open("rb") as model_file:
                metadata_prefix = model_file.read(_METADATA_SAMPLE_BYTES).decode(
                    "latin-1", errors="ignore"
                )
        except OSError as exc:
            raise ValueError(f"Could not inspect model metadata: {exc}") from exc
        context_parts.append(metadata_prefix)
        if metadata_prefix.startswith("GGUF"):
            evidence.append("GGUF metadata header")
    context = " ".join(context_parts).casefold()
    capabilities: set[str] = set()

    is_embedding = _matches(context, _EMBEDDING_MARKERS)
    is_vision = _matches(context, _VISION_MARKERS)
    is_ocr = _matches(context, _OCR_MARKERS)
    is_audio_transcription = _matches(context, _AUDIO_TRANSCRIPTION_MARKERS)
    is_audio_understanding = _matches(context, _AUDIO_UNDERSTANDING_MARKERS)
    is_audio_generation = _matches(context, _AUDIO_GENERATION_MARKERS)
    is_text_to_image = _matches(context, _TEXT_TO_IMAGE_MARKERS)
    is_image_to_image = _matches(context, _IMAGE_TO_IMAGE_MARKERS)
    is_image_generation = (
        is_text_to_image or is_image_to_image or _matches(context, _IMAGE_GENERATION_MARKERS)
    )
    is_image_to_video = _matches(context, _IMAGE_TO_VIDEO_MARKERS)
    is_text_to_video = _matches(context, _TEXT_TO_VIDEO_MARKERS)
    is_video_to_video = _matches(context, _VIDEO_TO_VIDEO_MARKERS)
    is_video_generation = (
        is_image_to_video
        or is_text_to_video
        or is_video_to_video
        or _matches(context, _VIDEO_GENERATION_MARKERS)
    )
    is_video_understanding = _matches(context, _VIDEO_UNDERSTANDING_MARKERS)
    is_reranker = _matches(context, ("rerank", "reranker", "cross_encoder"))
    is_moderation = _matches(context, ("moderation", "content_moderation", "safety_classifier"))

    projector = Path(mmproj_path).expanduser() if mmproj_path else None
    has_projector = bool(projector and projector.is_file())
    if has_projector:
        is_vision = True
        evidence.append("vision projector file")

    if is_embedding:
        capabilities.add("embeddings")
        evidence.append("embedding model name/metadata")
    if is_vision:
        capabilities.add("vision_understanding")
        evidence.append("vision model name/metadata")
    if is_ocr:
        capabilities.add("ocr")
        evidence.append("OCR model name/metadata")
    if is_audio_transcription:
        capabilities.add("audio_transcription")
        evidence.append("speech recognition/transcription model name/metadata")
    if is_audio_understanding:
        capabilities.add("audio_understanding")
        evidence.append("audio understanding model name/metadata")
    if is_audio_generation:
        capabilities.add("audio_generation")
        evidence.append("audio generation model name/metadata")
        if _matches(context, ("tts", "text_to_speech", "speech_synthesis")):
            capabilities.add("text_to_speech")
    if is_image_generation:
        capabilities.add("image_generation")
        evidence.append("image generation model name/metadata")
    if is_text_to_image:
        capabilities.add("text_to_image")
    if is_image_to_image:
        capabilities.add("image_to_image")
    if is_video_generation:
        capabilities.add("video_generation")
        evidence.append("video generation model name/metadata")
    if is_text_to_video:
        capabilities.add("text_to_video")
    if is_image_to_video:
        capabilities.add("image_to_video")
    if is_video_to_video:
        capabilities.add("video_to_video")
    if is_video_understanding:
        capabilities.add("video_understanding")
        evidence.append("video understanding model name/metadata")
    if is_reranker:
        capabilities.add("reranking")
        evidence.append("reranker model name/metadata")
    if is_moderation:
        capabilities.add("moderation")
        evidence.append("moderation model name/metadata")

    text_task = _matches(
        context,
        (
            "text_generation",
            "text_to_text",
            "llm",
            "instruct",
            "chat",
            "summarization",
            "summarizer",
            "translation",
            "classifier",
        ),
    )
    if is_vision or is_ocr or text_task or not capabilities:
        capabilities.add("text_generation")
    if _matches(context, ("summarization", "summarizer")):
        capabilities.add("text_summarization")
    if _matches(context, ("translation", "translator")):
        capabilities.add("text_translation")
    if _matches(context, ("classification", "classifier")) and not is_vision:
        capabilities.add("text_classification")

    ordered = [capability for capability in _CAPABILITY_ORDER if capability in capabilities]
    return {
        "capabilities": ordered,
        "evidence": list(dict.fromkeys(evidence)) or ["Default text-model suggestion"],
        "note": "Best-effort suggestions from the filename and GGUF metadata. Review them; selection does not add unsupported model/runtime features.",
    }
