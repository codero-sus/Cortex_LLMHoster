"""Static, public Hugging Face model-cookbook entries; no Hub API is used."""

from __future__ import annotations

# These are hand-curated repository links, not a live Hugging Face query. Model
# availability, file formats, licenses, and access terms can change; check each
# model card before downloading or using weights.
_RAW_MODELS: tuple[tuple[str, str, str, str, str, str], ...] = (
    # name, repo ID, task, format, parameter/size tier, short note
    ("SmolLM2 135M Instruct", "bartowski/SmolLM2-135M-Instruct-GGUF", "Chat", "GGUF", "135M", "Tiny instruction model for experiments and very constrained devices."),
    ("SmolLM2 360M Instruct", "bartowski/SmolLM2-360M-Instruct-GGUF", "Chat", "GGUF", "360M", "Small local assistant; keep prompts and context modest."),
    ("SmolLM2 1.7B Instruct", "bartowski/SmolLM2-1.7B-Instruct-GGUF", "Chat", "GGUF", "1.7B", "Compact general-purpose instruct model."),
    ("Qwen2.5 0.5B Instruct", "bartowski/Qwen2.5-0.5B-Instruct-GGUF", "Chat", "GGUF", "0.5B", "Small multilingual baseline for low-memory machines."),
    ("Qwen2.5 1.5B Instruct", "bartowski/Qwen2.5-1.5B-Instruct-GGUF", "Chat", "GGUF", "1.5B", "Compact general assistant with a broad language mix."),
    ("Qwen2.5 3B Instruct", "bartowski/Qwen2.5-3B-Instruct-GGUF", "Chat", "GGUF", "3B", "Balanced small model for everyday local prompts."),
    ("Qwen2.5 7B Instruct", "bartowski/Qwen2.5-7B-Instruct-GGUF", "Chat", "GGUF", "7B", "General-purpose instruct model; quantization helps reduce RAM use."),
    ("Qwen2.5 14B Instruct", "bartowski/Qwen2.5-14B-Instruct-GGUF", "Chat", "GGUF", "14B", "Larger general assistant; allow room for the context KV cache."),
    ("Qwen2.5 32B Instruct", "bartowski/Qwen2.5-32B-Instruct-GGUF", "Chat", "GGUF", "32B", "Large local model; storage and memory needs are substantial."),
    ("Qwen2.5 Coder 0.5B", "bartowski/Qwen2.5-Coder-0.5B-Instruct-GGUF", "Coding", "GGUF", "0.5B", "Small code-focused assistant for lightweight experiments."),
    ("Qwen2.5 Coder 1.5B", "bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF", "Coding", "GGUF", "1.5B", "Compact code completion and explanation model."),
    ("Qwen2.5 Coder 3B", "bartowski/Qwen2.5-Coder-3B-Instruct-GGUF", "Coding", "GGUF", "3B", "Small code assistant for local development tasks."),
    ("Qwen2.5 Coder 7B", "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF", "Coding", "GGUF", "7B", "Code-oriented model; compare prompt format and quantizations."),
    ("Qwen2.5 Coder 14B", "bartowski/Qwen2.5-Coder-14B-Instruct-GGUF", "Coding", "GGUF", "14B", "Larger code assistant with higher memory requirements."),
    ("Qwen3 0.6B", "bartowski/Qwen3-0.6B-GGUF", "Chat", "GGUF", "0.6B", "Very small Qwen3 family option; verify its chat-template notes."),
    ("Qwen3 1.7B", "bartowski/Qwen3-1.7B-GGUF", "Chat", "GGUF", "1.7B", "Compact Qwen3 model for local chat and reasoning prompts."),
    ("Qwen3 4B", "bartowski/Qwen3-4B-GGUF", "Chat", "GGUF", "4B", "Mid-sized Qwen3 family model; check the repository's recommended template."),
    ("Qwen3 8B", "bartowski/Qwen3-8B-GGUF", "Chat", "GGUF", "8B", "General Qwen3 model; choose a quantization that fits available RAM/VRAM."),
    ("Qwen3 14B", "bartowski/Qwen3-14B-GGUF", "Chat", "GGUF", "14B", "Larger Qwen3 family model; context length affects memory substantially."),
    ("Qwen3 30B A3B", "bartowski/Qwen3-30B-A3B-GGUF", "Chat", "GGUF", "30B MoE", "Mixture-of-experts model; total weight storage still matters."),
    ("Qwen3 32B", "bartowski/Qwen3-32B-GGUF", "Chat", "GGUF", "32B", "Large dense model for machines with substantial memory."),
    ("Phi-3.5 Mini Instruct", "bartowski/Phi-3.5-mini-instruct-GGUF", "Chat", "GGUF", "3.8B", "Compact Microsoft instruct model; review its model-card usage notes."),
    ("Phi-4 Mini Instruct", "bartowski/Phi-4-mini-instruct-GGUF", "Chat", "GGUF", "3.8B", "Small Phi family option; check the repository for quant and template details."),
    ("Mistral 7B Instruct v0.3", "bartowski/Mistral-7B-Instruct-v0.3-GGUF", "Chat", "GGUF", "7B", "Popular compact instruct model; use the prompt format specified by its card."),
    ("DeepSeek R1 Distill Qwen 1.5B", "bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF", "Reasoning", "GGUF", "1.5B", "Small reasoning-distilled model; longer answers use more context and time."),
    ("DeepSeek R1 Distill Qwen 7B", "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF", "Reasoning", "GGUF", "7B", "Reasoning-focused distill; inspect its recommended thinking prompt format."),
    ("DeepSeek R1 Distill Qwen 14B", "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF", "Reasoning", "GGUF", "14B", "Larger reasoning distill with proportionally higher memory needs."),
    ("DeepSeek R1 Distill Qwen 32B", "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF", "Reasoning", "GGUF", "32B", "Large reasoning distill; select quantization and context carefully."),
    ("Granite 3.1 2B Instruct", "bartowski/granite-3.1-2b-instruct-GGUF", "Chat", "GGUF", "2B", "Compact IBM Granite instruct model for local workflows."),
    ("Granite 3.1 8B Instruct", "bartowski/granite-3.1-8b-instruct-GGUF", "Chat", "GGUF", "8B", "Mid-sized Granite model; consult the card for its intended tasks."),
    ("Qwen2.5-VL 3B Instruct", "bartowski/Qwen2.5-VL-3B-Instruct-GGUF", "Vision", "GGUF + projector", "3B", "Vision-language model; use a compatible projector and runtime build."),
    ("Qwen2.5-VL 7B Instruct", "bartowski/Qwen2.5-VL-7B-Instruct-GGUF", "Vision", "GGUF + projector", "7B", "Larger vision-language option; image support requires compatible multimodal assets."),
    ("SmolVLM 500M Instruct", "ggml-org/SmolVLM-500M-Instruct-GGUF", "Vision", "GGUF", "500M", "Small vision-language model; check image-size and projector guidance."),
    ("SmolVLM 2.2B Instruct", "ggml-org/SmolVLM-2.2B-Instruct-GGUF", "Vision", "GGUF", "2.2B", "Compact vision-language model for image description and question answering."),
    ("Whisper tiny", "openai/whisper-tiny", "Audio transcription", "Transformers", "39M", "Very small speech-recognition checkpoint; use a compatible local audio backend."),
    ("Whisper base", "openai/whisper-base", "Audio transcription", "Transformers", "74M", "Small speech-recognition checkpoint for local transcription pipelines."),
    ("Whisper small", "openai/whisper-small", "Audio transcription", "Transformers", "244M", "Medium-size transcription checkpoint; backend and language affect results."),
    ("Whisper medium", "openai/whisper-medium", "Audio transcription", "Transformers", "769M", "Larger transcription checkpoint; use a local Whisper-compatible server."),
    ("Kokoro 82M", "hexgrad/Kokoro-82M", "Audio generation", "Transformers", "82M", "Text-to-speech weights; requires a compatible local TTS runtime and voices."),
    ("Parler TTS Mini", "parler-tts/parler-tts-mini-v1", "Audio generation", "Transformers", "Mini", "Text-to-speech model; review its card for the matching runtime and assets."),
    ("Bark Small", "suno/bark-small", "Audio generation", "Transformers", "Small", "Text/audio generation checkpoint; run with a compatible local Bark backend."),
    ("TrOCR printed text", "microsoft/trocr-base-printed", "OCR", "Transformers", "Base", "Printed-text OCR checkpoint; requires a local Transformers-compatible runtime."),
    ("TrOCR handwritten text", "microsoft/trocr-base-handwritten", "OCR", "Transformers", "Base", "Handwriting OCR checkpoint; pair with the matching processor and backend."),
    ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", "Embeddings", "Transformers", "22M", "Small sentence-embedding model for semantic search and similarity."),
    ("BGE small English v1.5", "BAAI/bge-small-en-v1.5", "Embeddings", "Transformers", "Small", "Compact English embedding checkpoint for retrieval use cases."),
    ("BGE base English v1.5", "BAAI/bge-base-en-v1.5", "Embeddings", "Transformers", "Base", "English embedding checkpoint; compare its model-card retrieval instructions."),
    ("BGE-M3", "BAAI/bge-m3", "Embeddings", "Transformers", "Multi", "Multilingual/multifunctional embedding model; needs a compatible embedding backend."),
    ("BGE Reranker v2 M3", "BAAI/bge-reranker-v2-m3", "Reranking", "Transformers", "Multi", "Reranking checkpoint for a local search pipeline."),
    ("MS MARCO MiniLM cross encoder", "cross-encoder/ms-marco-MiniLM-L-6-v2", "Reranking", "Transformers", "22M", "Compact cross-encoder reranker for passage-ranking workflows."),
    ("Tiny SD", "segmind/tiny-sd", "Image generation", "Diffusers", "Tiny", "Small text-to-image starting point; use a compatible local Diffusers runtime."),
    ("Zeroscope v2 576w", "cerspense/zeroscope_v2_576w", "Video generation", "Diffusers", "1.7B", "Text-to-video checkpoint; memory needs are high and backend support is required."),
)


def cookbook_models() -> list[dict[str, str]]:
    """Return JSON-ready static entries; this intentionally makes no HF requests."""

    return [
        {
            "name": name,
            "repo": repo,
            "task": task,
            "format": model_format,
            "scale": scale,
            "summary": summary,
        }
        for name, repo, task, model_format, scale, summary in _RAW_MODELS
    ]
