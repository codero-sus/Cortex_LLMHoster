from __future__ import annotations

import pytest

from cortex_llmhoster.model_discovery import detect_model_capabilities


@pytest.mark.parametrize(
    ("filename", "expected", "text_generation"),
    [
        ("bge-small-en-v1.5.Q4_K_M.gguf", {"embeddings"}, False),
        ("flux-dev-Q4_K_M.gguf", {"image_generation"}, False),
        ("whisper-small-Q5_0.gguf", {"audio_transcription"}, False),
        (
            "qwen2.5-vl-7b-instruct.Q4_K_M.gguf",
            {"vision_understanding", "text_generation"},
            True,
        ),
        (
            "wan2.1-t2v-Q4_K_M.gguf",
            {"video_generation", "text_to_video"},
            False,
        ),
    ],
)
def test_filename_detection_suggests_relevant_task_checkboxes(
    tmp_path, filename: str, expected: set[str], text_generation: bool
) -> None:
    model_file = tmp_path / filename
    model_file.write_bytes(b"GGUF fixture; weights are not loaded")

    result = detect_model_capabilities(str(model_file))
    capabilities = set(result["capabilities"])

    assert expected <= capabilities
    assert ("text_generation" in capabilities) is text_generation
    assert "best-effort" in result["note"].lower()


def test_vision_projector_suggests_vision_for_a_compatible_text_model(tmp_path) -> None:
    model_file = tmp_path / "llama-3-instruct.Q4_K_M.gguf"
    projector_file = tmp_path / "mmproj-model-f16.gguf"
    model_file.write_bytes(b"GGUF fixture")
    projector_file.write_bytes(b"projector fixture")

    result = detect_model_capabilities(str(model_file), mmproj_path=str(projector_file))

    assert {"vision_understanding", "text_generation"} <= set(result["capabilities"])
    assert "vision projector file" in result["evidence"]


def test_capability_detection_rejects_missing_local_paths(tmp_path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        detect_model_capabilities(str(tmp_path / "missing.gguf"))
