# Cortex LLMHoster

Cortex LLMHoster is a **free-to-use local model hoster** for permitted personal,
non-commercial use. It manages llama.cpp GGUF models directly and can supervise
other local inference engines through an OpenAI-compatible command runtime.

**No Ollama required. No cloud inference account or cloud API key required.**
Cortex routes inference to model servers running on your own machine; prompts
and media are not sent to a Cortex cloud service. A `CORTEX_API_KEY` may be set
as a **local bearer secret** to protect this install's API and dashboard—it is
not a provider credential, and Cortex does not forward it to the model runtime.
Cortex has no per-token cloud charge. You still need a computer, local runtime,
model weights, and any required third-party licenses; these are not bundled or
automatically downloaded.

The adaptable runtime layer can host text/chat, embeddings, vision, OCR, audio
understanding/transcription/generation, image generation, and video tasks when
the selected local engine and model actually implement them. No single runtime
or model provides every modality. Cortex routes requests to the chosen local
process; it does not create unsupported model capabilities or silently translate
incompatible APIs.

Efficiency controls include CPU-only operation, optional GPU offload, quantized
GGUF model support, memory mapping, configurable context and batches, KV-cache
tuning, and three optimization levels. Actual speed and memory use depend on the
model, quantization, runtime build, and hardware; Cortex makes no unmeasured
benchmark claims.

## License and permitted use

Cortex is **source-available under a custom restrictive license**, not an
OSI-approved open-source license. You may download, install, and run an
unmodified copy for personal, non-commercial use. Modification, redistribution,
republishing, sale, sublicensing, and commercial use are not permitted without
separate written permission. See [`LICENSE`](LICENSE) for the full terms.
Third-party Python packages, runtimes, and model weights are subject to their
own licenses. The license text is a project license draft, not legal advice;
have counsel review it before relying on it for distribution.

## Highlights

- **Two local runtime modes:** first-class llama.cpp/GGUF hosting plus a
  supervised command adapter for any local engine that exposes OpenAI-compatible
  HTTP endpoints.
- **Task/capability registry:** declare text generation, vision understanding,
  OCR, audio understanding/transcription/generation, image generation, video
  understanding/generation, embeddings, reranking, or future task labels per
  model. Unsupported declared tasks get a clear API error.
- **Multimodal request transport:** supports JSON chat/responses and bounded
  multipart media uploads, including OpenAI-style audio transcription requests.
  Image, audio, video, OCR, and generation endpoints are passed to the matching
  local runtime without uploading data to a cloud inference service.
- **CPU and GPU:** llama.cpp supports `gpu_layers = 0` for CPU-only inference,
  `-1` for maximum GPU offload, or a selected layer count. Other runtimes use
  their own device flags and hardware support.
- **Low-memory controls:** llama.cpp exposes context, batch and micro-batch
  sizes, CPU threads, parallel slots, memory mapping, KV-cache types, and mlock.
- **Memory-aware loading:** only the configured default local model starts
  automatically. Other models can be started and stopped from the dashboard.
- **Lightweight browser UI:** manage models, runtimes and task declarations;
  inspect hardware; chat; use a same-origin modality/API workbench; and
  import/export secret-free configuration.
- **OpenAI-compatible API:** routes chat, completions, responses, embeddings,
  image/audio/video endpoints, and backend-defined paths to the selected local
  process. A route is usable only when that backend implements it.
- **Asynchronous Python control plane:** Starlette and HTTPX, compact `orjson`
  responses, streaming with backpressure, and automatic uvloop/httptools use
  where installed.

## Installation and quick start

Cortex needs Python 3.11+ and a local model runtime with its weights. It does
not require Ollama, a cloud inference account, or a cloud API key. The model
runtime and weights are installed separately and are not downloaded by Cortex.

See the full [installation guide](INSTALLTION.md) for Linux/macOS, Windows
Command Prompt, Windows PowerShell, `pipx`, local runtime setup, efficiency
presets, and first-request examples. If anything goes wrong, see
[Troubleshooting](TROUBLESHOOTING.md).

Quick start from the repository on Linux/macOS:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install .
cp cortex.example.toml cortex.toml
# Edit cortex.toml: set model_path to an existing local GGUF file.
export CORTEX_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python -m cortex_llmhoster
```

Open <http://127.0.0.1:8624/> and enter that generated secret in the dashboard.
It is a local access-control token, not a cloud credential. Cortex binds to
`127.0.0.1` by default; model inference stays on your computer. The software is
free to use only within the personal, non-commercial terms in [`LICENSE`](LICENSE).

## Multimodal models and local runtimes

A model's capabilities describe what its model weights and local engine actually
support; labels do not add missing inference implementations. The command
adapter is the extension point for model families not served by llama.cpp. It
launches a local process, binds it to loopback using an ephemeral port, and
forwards OpenAI-compatible HTTP requests to it. It never accepts a user-supplied
remote base URL.

```toml
[[models]]
id = "local-multimodal"
runtime = "command"
model_path = "/models/my-local-model"
api_base_path = "/v1"
health_path = "/health"
capabilities = [
  "text_generation",
  "vision_understanding",
  "ocr",
  "audio_understanding",
  "audio_transcription",
  "audio_generation",
  "image_generation",
  "video_understanding",
  "video_generation",
]
runtime_command = [
  "python",
  "/opt/my-local-openai-server.py",
  "--model",
  "{model_path}",
  "--host",
  "{host}",
  "--port",
  "{port}",
]
```

`runtime_command` is an argv array executed without a shell. `{host}` and
`{port}` are required and bind to `127.0.0.1`; `{model_path}`, `{model_id}`,
`{model_alias}`, `{api_base_path}`, `{health_path}`, `{gpu_layers}`, `{threads}`,
and `{context_size}` can also be used. Use a local server/wrapper that implements
the desired endpoints and a health path returning HTTP 2xx. `model_path` may be a
file or directory and is optional if the runtime command does not need it.

Common capability-to-route checks are:

| Capability | Cortex API paths |
| --- | --- |
| `text_generation` | `/chat/completions`, `/completions`, text-only `/responses` |
| `vision_understanding` | Image-containing `/chat/completions` and `/responses` |
| `ocr` | `/ocr` and `/ocr/parse` (local backend-defined endpoint) |
| `audio_transcription` | `/audio/transcriptions`, `/audio/translations` (multipart supported) |
| `audio_understanding` | Audio-input chat/responses and `/audio/analysis` |
| `audio_generation` | `/audio/speech` and audio-output chat/responses |
| `image_generation` | `/images/generations`, `/images/edits`, `/images/variations` |
| `video_understanding` | Video-input chat/responses and `/videos/understanding` |
| `video_generation` | `/videos/generations` and `/video/generations` |
| `embeddings`, `reranking`, `moderation` | `/embeddings`, `/rerank`, `/moderations` |

Unknown backend-specific `/v1/*` paths are also passed through, so additional
tasks can be added without changing the Cortex route dispatcher. Declare
capabilities as lowercase underscore task labels. For llama.cpp, text generation
is the default; `embedding = true` enables embeddings, and `mmproj_path` adds
vision understanding for compatible GGUF models. Configure a modality-specific
model/runtime when a task needs different weights or a different engine.

The dashboard's **API & tools** page includes a same-origin request workbench
for JSON and media-file requests. Multipart bodies are parsed/spooled to
temporary storage to read/alias the standard `model` field, then relayed to the
local process; `max_media_body_bytes` (default 2 GiB) bounds uploads. Keep enough
free temporary-disk space for the largest file you intend to test.

## CPU and GPU tuning

The model editor in the UI and `[[models]]` TOML entries expose the following
controls:

| Setting | Purpose |
| --- | --- |
| `gpu_layers` | `0` for CPU-only; `-1` for maximum supported GPU offload; positive values offload that many layers. |
| `optimization_level` | `-1` caps batches at 128/32 for lower memory; `0` uses configured tuning; `1` is ULTRA (batch minima 512/128, q8_0 KV cache, and GPU Flash Attention when set to `auto`). Device selection remains explicit through `gpu_layers`. |
| `threads` / `threads_batch` | CPU inference and prompt-processing threads; `0` selects Cortex's conservative automatic count. |
| `context_size` | Context window in tokens. Reducing it lowers KV-cache memory use. |
| `batch_size` / `ubatch_size` | Prompt-processing batch limits; smaller values can reduce peak memory. |
| `parallel` | llama.cpp request slots. Keep at `1` on low-memory systems. |
| `cache_type_k` / `cache_type_v` | KV-cache precision; quantized types such as `q8_0` can reduce memory use where supported. |
| `use_mmap` | Memory-map model weights (enabled by default). |
| `mlock` | Ask the OS to lock model weights in RAM; this can increase memory pressure and may require elevated permission. |
| `flash_attn` | Select `auto`, `on`, or `off` when supported by the installed build/model. |
| `embedding` | Enable the local embedding endpoint. |
| `mmproj_path` | Optional GGUF multimodal projector file for a compatible model/build. |

For a constrained CPU-only device, start with `gpu_layers = 0`, `parallel = 1`,
a modest `context_size` (for example 1024–2048), and smaller batch sizes. For a
GPU build, try `gpu_layers = -1`, then lower the offload count if VRAM is
insufficient. These are starting points, not universal performance guarantees.

The model editor provides three optimization levels: `-1` (Efficient), `0`
(Balanced), and `1` (ULTRA). Efficient caps batch/micro-batch sizes at 128/32;
Balanced uses the configured values; ULTRA raises their minimums to 512/128,
uses q8_0 KV caches, and enables Flash Attention for GPU offload when set to
`auto`. The levels do not change `gpu_layers` or auto-select an accelerator.
ULTRA can need more scratch RAM/VRAM, q8_0 may affect output quality, and an
unsupported llama.cpp build may reject its options. These are hardware-dependent
presets, not measured throughput claims. Older `optimization_profile =
"balanced"` / `"ultra"` config values remain accepted as levels 0 / 1.

Cortex also bounds active inference requests (`max_inference_requests`; default
8, and never above `max_connections`) before the inference handler parses
payloads or opens more local requests. Lower it on memory-constrained hosts;
once full, Cortex returns `503` with `Retry-After` instead of building a large
queue of parsed prompts and live streams. Change this limit live in the
dashboard and keep it at or below `max_connections`.

Cortex intentionally requires **one Uvicorn worker** when a locally managed
model is configured. Each Cortex worker owns its own runtime processes, so
multiple workers could load duplicate model weights and waste RAM/VRAM. Use the
selected runtime's concurrency settings instead. The resolved CLI/config `host`,
`port`, and `workers` values are passed to Uvicorn.

## Configuration and dashboard

`cortex.example.toml` contains a complete local-hosting example. Each model
needs a unique public `id` and a supported local runtime: `llama.cpp` requires a
`.gguf` file, while `command` starts a local OpenAI-compatible server using
`runtime_command`. Declare supported tasks with `capabilities`. A single model
is selected by default automatically; with multiple models, mark one with
`default = true` or configure `[server].default_model`.

The public API profile lives in `src/config/api.json`; `src/config/example.api.json`
shows its OpenAI-compatible request examples. Set `base_path` and `models_path`
to adapt the exposed routes, or select another config file with
`CORTEX_API_CONFIG` / `--api-config`. The current wire format is OpenAI-compatible;
other dialects require an explicit adapter and are not silently translated.

The dashboard at `/` includes:

- Host CPU/RAM/GPU snapshot and local runtime status.
- Add/edit/remove model configuration, task capabilities, local start/stop/restart,
  and runtime process output.
- A streaming text chat playground and a local API/media workbench.
- llama.cpp tuning, API settings, request metrics, and secret-free config
  import/export.

Dashboard edits are validated and atomically written to `cortex.local.json` (or
the path in `CORTEX_ADMIN_CONFIG`), leaving the startup TOML unchanged. Secret
values are never stored in that JSON file; use environment variable names for
credentials. Active local model processes are restarted when their runtime
settings change.

### Useful environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `CORTEX_CONFIG` | `./cortex.toml` | TOML configuration path. |
| `CORTEX_API_CONFIG` | `src/config/api.json` | OpenAI-compatible API format and route-prefix settings; restart after changes. |
| `CORTEX_ADMIN_CONFIG` | sibling `cortex.local.json` | Dashboard override file path. |
| `CORTEX_HOST` / `CORTEX_PORT` | `127.0.0.1` / `8624` | Local-only listen address by default; use `0.0.0.0` only when intentionally exposing the service. |
| `CORTEX_WORKERS` | `1` | Uvicorn worker count; local model hosting requires one. |
| `CORTEX_LLAMA_SERVER` | `PATH` lookup | Path to the `llama-server` executable. |
| `CORTEX_API_KEY` | unset | Optional local bearer secret for API/admin access; not a cloud-provider key. Admin endpoints stay locked when unset. |
| `CORTEX_DEFAULT_MODEL` | config-derived | Default public model ID. |
| `CORTEX_LOG_LEVEL` / `CORTEX_ACCESS_LOG` | `warning` / `false` | Server logging controls. |
| `CORTEX_MAX_JSON_BODY_BYTES` | `16777216` | Maximum buffered JSON body size. |
| `CORTEX_MAX_MEDIA_BODY_BYTES` | `2147483648` | Maximum multipart audio/image/video upload size; uploads spool to temporary storage. |
| `CORTEX_MAX_INFERENCE_REQUESTS` | Up to `8` | Active inference body/stream limit, capped by `max_connections`; excess requests receive HTTP 503 with `Retry-After`. |

CLI options `--host`, `--port`, `--workers`, `--log-level`, and
`--access-log` override their TOML/environment values. `--api-config` selects the
OpenAI-compatible route config JSON. For locally managed
models, `--workers` must resolve to `1` to avoid duplicate model processes.

## Endpoints

- `GET /` — self-contained dashboard; no build step or third-party assets.
- `GET /health` — liveness check.
- `GET /ready` — readiness check; a local default model must be running.
- `GET /v1/models` — public model IDs and declared capabilities.
- `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings` —
  routes implemented by each selected local runtime.
- `/v1/audio/*`, `/v1/images/*`, `/v1/videos/*`, `/v1/ocr` — multimodal or
  backend-defined routes; availability depends on that model's runtime.
- `/admin/api/*` — authenticated settings, runtime controls, process logs and
  per-worker metrics.

## Maintainer test commands

These commands are for the copyright holder or explicitly authorized maintainers;
they are not permission for users to modify or redistribute Cortex under the
license above.

```bash
python -m pip install -e '.[test]'
pytest
ruff check .
```
