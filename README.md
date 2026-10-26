# Cortex LLMHoster

Cortex LLMHoster is a **free-to-use local model hoster** for permitted personal,
non-commercial use. It includes a Cortex-managed native C++ GGUF host built on
the llama.cpp inference core, can manage an external llama.cpp server, and can
supervise other local inference engines through an OpenAI-compatible command
runtime. A legacy Python-binding GGUF runtime remains for existing
configurations, but is not the recommended model-loading path.

**No Ollama required. No cloud inference account or cloud API key required.**
Cortex routes inference to model servers running on your own machine; prompts
and media are not sent to a Cortex cloud service. `CORTEX_API_AUTH_ENABLED`
explicitly turns local API protection on or off. Authentication defaults on
when a non-empty `CORTEX_API_KEY` or dashboard account username is configured;
when enabled, a non-empty local `CORTEX_API_KEY` is required. A configured
username/password adds account sign-in to the dashboard but does not replace the
API key. Use `CORTEX_API_AUTH_ENABLED=false` or `--no-auth` (alias
`--no-api-auth`) to run without a key and allow unauthenticated local requests.
The API key is a **local bearer secret**, not a provider credential, and Cortex
does not forward it to a model runtime. Successful dashboard sign-in creates a signed, persistent, HttpOnly
session cookie for 30 days; API clients can continue to use the bearer key.
Cortex has no per-token cloud charge. You still need a computer, local model
weights, and any required third-party licenses. Model weights are not bundled or
automatically downloaded; built-in native runtime executables can be installed
on demand from the project's rolling GitHub release.

The adaptable runtime layer can host text/chat, embeddings, vision, OCR, audio
understanding/transcription/generation, image generation, and video tasks when
the selected local engine and model actually implement them. No single runtime
or model provides every modality. Cortex's native C++ GGUF host focuses on
text generation and embeddings; use the external llama.cpp runtime or a suitable
custom server for projector-based vision, audio, video, or other tasks. Cortex
routes requests to the chosen local process; it does not create unsupported model
capabilities or silently translate incompatible APIs.

Efficiency controls include CPU-only operation, optional GPU offload, quantized
GGUF model support, memory mapping, configurable context and batches, KV-cache
tuning, seven standard optimization modes, Low VRAM, and customizable SUPER LOW,
with ULTRA as a separate opt-in.
Actual speed and memory use depend on the model, quantization, runtime build, and
hardware; Cortex makes no unmeasured benchmark claims.

## License and permitted use

Cortex is **source-available under a custom restrictive license**, not an
OSI-approved open-source license. You may download, install, and run an
unmodified copy for personal, non-commercial use. Modification, redistribution,
republishing, sale, sublicensing, and commercial use are not permitted without
separate written permission. See [`LICENSE`](LICENSE) for the full terms.
Third-party Python packages, runtimes, and model weights are subject to their
own licenses. The license text is a project license draft, not legal advice;
have counsel review it before relying on it for distribution.

## Author and credits

Made with ❤️ by [codero-sus on GitHub](https://github.com/codero-sus).
Visit the [project website](https://the-codero.web.app/) for more from Codero-Sus.

## Highlights

- **Four managed runtime modes:** Cortex's native C++ GGUF host (no external
  `llama-server` or Python inference binding), the retained legacy Python
  binding runtime, an external llama.cpp server, and a supervised command
  adapter for other local OpenAI-compatible engines. The C++ host uses the
  llama.cpp inference core; it is not a new inference engine. The dashboard
  includes optional command templates for vLLM, SGLang, TGI, and MLX-LM; they
  prefill arguments but do not install engines. See
  [`SERVER_OPTIONS.md`](SERVER_OPTIONS.md) for choices and use cases.
- **Cross-runtime task-aware routing:** use the virtual `auto` model to dispatch
  requests by declared capabilities across compatible local models/runtimes;
  Cortex prefers an already-running compatible model and reports its selection.
- **Task/capability registry:** choose common text, vision/VLM, OCR, image,
  audio, video, embeddings, reranking, and moderation tasks in the grouped model
  form, or enter custom task IDs. Known routes accept both broad capabilities
  and specific aliases such as LLM/T2T, T2I/I2I, ASR/STT/TTS, and T2V/I2V/V2V.
  Capability labels describe the selected local model/runtime; they do not add
  missing implementations.
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
- **Manual software updater:** the dashboard updater checks the pinned source
  branch when requested. Root-level `updater.sh` and `updater.bat` first back up
  repository JSON files under `backups/`, then run `git clone --branch
  arena/b3072048-cortex-llmhoster https://github.com/codero-sus/Cortex_LLMHoster.git`
  in a temporary folder and fast-forward the checkout from that clone. They need
  Git and network access only—not Python, pip, or a virtual environment—and
  never switch to `main`. Restart Cortex LLMHoster after updating.
- **Local account sign-in:** optionally configure a single dashboard username
  and password. A successful sign-in creates a signed HttpOnly cookie; API
  clients may continue to authenticate with the local bearer key. Sign-in
  attempts are throttled, and admin API responses are marked no-store.
- **Lightweight browser UI:** manage models, runtimes and task declarations;
  inspect hardware; chat; use a same-origin modality/API workbench; browse a
  paginated Hugging Face model cookbook with an offline curated fallback; and
  import/export secret-free configuration. Public search sends no HF token and
  downloads no model weights.
- **OpenAI-compatible API:** routes chat, completions, responses, embeddings,
  image/audio/video endpoints, and backend-defined paths to the selected local
  process. A route is usable only when that backend implements it.
- **Asynchronous Python control plane:** Starlette and HTTPX, compact `orjson`
  responses, streaming with backpressure, and automatic uvloop/httptools use
  where installed.

## Installation and quick start

Cortex needs Python 3.11+ and local model weights. It does not require Ollama,
a cloud inference account, or a cloud API key. When the managed llama.cpp or
Cortex C++ runtime is not already configured, Cortex can auto-install its native
runtime executables; model weights and other third-party runtimes remain a
separate installation.

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
# Edit cortex.toml: set model_path to a local GGUF or supported source model.
export CORTEX_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python -m cortex_llmhoster
```

Open <http://127.0.0.1:8624/>. Without account variables, choose **Use API key
instead** and enter the generated key. For username/password dashboard sign-in,
keep the API key configured and also set both optional account variables before
launching:

```bash
export CORTEX_ADMIN_USERNAME=admin
export CORTEX_ADMIN_PASSWORD='replace-with-a-long-unique-password'
```

The dashboard validates account credentials on the server and stores only a
signed HttpOnly session cookie for 30 days; it does not store passwords or API
keys in browser storage. Account sessions are tied to the configured API key and
account credentials. Authentication-enabled mode always requires
`CORTEX_API_KEY`; configure account credentials too if you want account-based
dashboard sign-in alongside bearer-key access. To run without an API key, unset
it from the environment and disable authentication, for example:
`unset CORTEX_API_KEY && python -m cortex_llmhoster --no-auth` (the older
`--no-api-auth` flag remains supported), or set `CORTEX_API_AUTH_ENABLED=false`
(also available as `api_auth_enabled = false` under `[server]` in `cortex.toml`).
Anonymous access should only be used on a trusted local listener. Cortex binds to `127.0.0.1` by default; model inference
stays on your computer. The software is free to use only within the personal,
non-commercial terms in [`LICENSE`](LICENSE).

### Automatic native runtime installation

When no configured executable, local build, or `PATH` entry is found, Cortex
can download the matching `llama-server`, `cortex-gguf-host`, or `llama-quantize`
binary from this repository's latest release under the exact rolling tag
`cortex-cpp`. Successful push builds on `main` publish ZIP bundles for Linux
x86_64, Windows x86_64, macOS Intel, and Apple Silicon. The manifest SHA-256
and bundle size are checked before the declared binaries and support libraries
are installed in the per-user runtime cache. Linux bundles include `libgomp`,
and Windows bundles include the app-local MSVC OpenMP DLL and redistributable
notice.
Previously installed binaries remain available if GitHub is temporarily
unreachable. The first install (and later release updates) requires access to
GitHub; the install uses Python's standard-library HTTPS and ZIP support and
never invokes pip or a virtual environment. It does not download model weights.

An explicit `CORTEX_LLAMA_SERVER`, `CORTEX_CPP_HOST`, or `CORTEX_CPP_QUANTIZE`
path, a local native build, or a binary already on `PATH` takes precedence as
applicable. Set `CORTEX_AUTO_INSTALL_RUNTIMES=false` to opt out, or set
`CORTEX_RUNTIME_BIN_DIR` to change the runtime cache root. Release builds use
an x86-64/SSE2 baseline (without AVX/AVX2) on Linux, Windows, and Intel macOS;
Apple Silicon builds also enable Metal. The bundled `llama-server` is intended
for API use and does not include the upstream browser UI assets. For a
CUDA/Vulkan-enabled llama.cpp server, install that build separately and point
`CORTEX_LLAMA_SERVER` at it. The C++ Cortex backend remains an in-process C++
inference host and does not depend on `llama-server`.

### Optional custom Python interpreter

Cortex LLMHoster normally keeps using the interpreter that started it. To
select a portable or embedded interpreter for the CLI and local command runtimes,
add a `python.env` file in the project root containing one executable path
(absolute, or relative to the project root). A line such as
`2PY2=/path/to/python` is also accepted. This machine-specific file is ignored
by Git.

Alternatively, copy [`.env.example`](.env.example) to `.env` and set
`2PY2=/absolute/path/to/python`. The CLI loads `.env` for interpreter and local
account settings. The root source updater uses Git only and does not read these
Python settings. `.env` is ignored by Git; only the sample is tracked. A process
environment `2PY2` takes precedence, followed by `2PY2` in `.env`, then
`python.env`, and finally the interpreter that started the command. On Linux or macOS:

```bash
cp .env.example .env
# Edit .env and replace /absolute/path/to/python with your interpreter.
```

```bash
/opt/portable-python/bin/python3 -m pip install .
env '2PY2=/opt/portable-python/bin/python3' python -m cortex_llmhoster
```

Cortex restarts its CLI under the selected interpreter when needed, so install
Cortex and its dependencies into that interpreter first. `{python}` in a
`runtime_command` resolves to it. The root source updater only synchronizes the
Git checkout; if an update introduces Python dependency changes, install them
separately in your selected environment. On Linux/macOS shells where names
beginning with a digit cannot be exported, use the `env '2PY2=…' command` form
shown above or configure `python.env`.

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
  "{python}",
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
`{context_size}`, and `{python}` can also be used. `{python}` resolves to the
selected interpreter described above. Use a local server/wrapper that implements
the desired endpoints and a health path returning HTTP 2xx. `model_path` may be a
file or directory and is optional if the runtime command does not need it. Child
runtimes inherit the normal process environment, but Cortex removes its API key,
configured API-key environment variable, and dashboard password before launch.

The dashboard's **Models** form has grouped task checkboxes and a custom task-ID
field. Capability names are normalized to lowercase underscore IDs (hyphens are
accepted). The built-in route checks accept either the broad capability names
or appropriate task-specific aliases:

| Route capability | Example accepted aliases |
| --- | --- |
| `text_generation` | `text_to_text`, `llm`, `language_model`, `t2t`, `text_summarization`, `text_translation` |
| `vision_understanding` | `vision`, `vlm`, `image_understanding`, `image_to_text`, `image_captioning`, `visual_question_answering` |
| `ocr` | `text_recognition`, `document_ocr` |
| `image_generation` | `text_to_image` / `t2i`, `image_to_image` / `i2i`, `image_editing` |
| `audio_transcription` | `speech_to_text`, `audio_to_text`, `asr`, `stt`, `audio_translation` |
| `audio_understanding` | `audio_perception`, `audio_classification`, `asr`, `stt` |
| `audio_generation` | `text_to_speech` / `tts`, `speech_generation`, `speech_to_speech`, `music_generation` |
| `video_understanding` | `video_to_text`, `video_summarization`, `video_captioning`, `action_recognition` |
| `video_generation` | `text_to_video` / `t2v`, `image_to_video` / `i2v`, `video_to_video` / `v2v` |
| `embeddings`, `reranking`, `moderation` | `text_embedding`, `image_embedding`, `feature_extraction`, `reranker`, `content_moderation` |

The standard routes listed above also have task-oriented path aliases, including
`/text-to-text`, `/text-to-image`, `/image-to-image`, `/image-to-text`,
`/speech-to-text`, `/text-to-speech`, `/text-to-video`, `/image-to-video`, and
`/video-to-video`. They are forwarded unchanged to the selected local runtime.

**Cortex task-aware routing:** send `"model": "auto"` (or select **Auto** in
the dashboard) to route a request to a configured model that declares the
required task. Cortex infers common tasks from the endpoint and attached media,
prefers a compatible model already running, then the configured default. The
virtual `auto` entry appears in `GET /v1/models`; responses include
`X-Cortex-Selected-Model` so clients can see the chosen model. Explicit model IDs
continue to select exactly that model. Auto-routing does not start or stop
models; start the compatible local model in the dashboard first.

Unknown backend-specific `/v1/*` paths are also passed through, so additional
tasks can be added without changing the Cortex route dispatcher. If a custom
capability ID matches an unknown route after converting its slashes and hyphens
to underscores (for example, `extensions_speech_translation` for
`/extensions/speech-translation`), Cortex checks that declaration too. Declare
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
| `gpu_layers` | `0` for CPU-only; `-1` for maximum supported GPU offload; positive values offload that many layers. `-2` is Cortex C++ only and uses the pinned llama.cpp memory fitter to choose a safe offload count. |
| `optimization_preset` | `low_minus`, `low_plus`, `balanced`, `medium_minus`, `medium_plus`, `high_minus`, `high_plus`, `low_vram`, or `super_low`; optional `ultra` enables higher-throughput tuning. `low_vram` caps context at 2048 with 128/32 batches and Q8 KV. `super_low` starts at 1024 context, 64/16 batches and Q4 KV; the dashboard and TOML can customize these starting values, cache, threads, GPU layers, mmap, and mlock. Larger context/batches/parallelism raise memory use. Cortex C++ maps `gpu_layers = -1` to memory-fit for both low-memory presets; external llama.cpp keeps manual layer settings. |
| `optimization_level` | Legacy setting (`-1` Efficient, `0` Balanced, `1` ULTRA); new configurations should use `optimization_preset`. |
| `threads` / `threads_batch` | CPU inference and prompt-processing threads; `0` selects a conservative, container-aware automatic count. CPU affinity/quota detection is cached for the process lifetime. |
| `context_size` | Context window in tokens. Reducing it lowers KV-cache memory use. |
| `batch_size` / `ubatch_size` | Prompt-processing batch limits; smaller values can reduce peak memory. |
| `parallel` | llama.cpp request slots. Keep at `1` on low-memory systems. |
| `cache_type_k` / `cache_type_v` | KV-cache precision; quantized types such as `q8_0` can reduce memory use where supported. |
| `use_mmap` | Memory-map model weights (enabled by default). |
| `mlock` | Ask the OS to lock model weights in RAM; this can increase memory pressure and may require elevated permission. |
| `flash_attn` | Select `auto`, `on`, or `off` when supported by the installed build/model. |
| `embedding` | Enable the local embedding endpoint. |
| `mmproj_path` | Optional GGUF multimodal projector file for a compatible `llama.cpp` build. |
| `conversion_outtype` | For `cortex-cpp`, non-GGUF sources can convert directly to `f16`, `q8_0`, or `f32`; Q2_K–Q6_K use F16 plus `llama-quantize`. GGUF inputs can be requantized to Q8_0 or Q2_K–Q6_K. F16/F32 selection leaves an existing GGUF unchanged. |
| `allow_requantize` | Explicitly allow re-quantizing an already-quantized GGUF. Off by default because repeated quantization can severely reduce quality. |
| `allow_unsafe_legacy_checkpoint_conversion` | Explicit opt-in for trusted legacy LLaMA pickle checkpoints only; pickle deserialization can execute code. |

For a constrained CPU-only device, start with `gpu_layers = 0`, `parallel = 1`,
a modest `context_size` (for example 1024–2048), and smaller batch sizes. For a
GPU build, try `gpu_layers = -1`, then lower the offload count if VRAM is
insufficient. These are starting points, not universal performance guarantees.

The model editor offers the standard quality/memory modes plus **Low VRAM** and **SUPER LOW**.
Low VRAM is deliberately conservative: it caps context at 2048, limits prompt
and micro-batches, keeps one request slot, uses a Q8 KV cache and memory-mapped
weights, and disables memory locking. The native C++ backend can also fit the
number of GPU layers against currently available device memory (`gpu_layers =
-2`; the Low VRAM profile maps `-1` to this mode). Its memory fitter leaves a
safety margin and reports the selected layer count. External `llama.cpp` servers
keep their explicit layer setting for compatibility; reduce `gpu_layers`
manually if they cannot load the model.

**SUPER LOW is the aggressive, customizable profile.** It starts from a 1024-token
context, 64-token batch, 16-token micro-batch, one request slot, Q4 KV cache,
memory mapping, and no mlock. The dashboard pre-fills these values when
you select the mode; all advanced fields remain editable, including context,
batches, parallelism, cache precision, threads, GPU layers, mmap, and mlock.
Increasing context/batches/parallelism increases memory use. Cortex C++ maps
`gpu_layers = -1` to its native memory fitter; external llama.cpp needs an
explicit layer count. Q4 KV and the shorter starting context can reduce output
quality or conversation length; adjust settings or choose a less aggressive
mode if the model or backend rejects them.

**ULTRA is a separate opt-in throughput tune**, not a low-memory guarantee: it
raises batch sizes and uses Q8 KV cache, which can increase memory use. No preset
guarantees speed or quality; results depend on the model, runtime, build, and
hardware. The low-bit weight targets are a separate trade-off: Q4_K_M is a
reasonable starting point, while Q2/Q3 can cause noticeable quality loss.

A 27B model on an RTX 3050 8 GB remains a target to test, not a promise. Common
4-bit weights alone can exceed 8 GB; the native fitter can leave some layers on
CPU, but system RAM, context size, storage, and CPU/GPU transfer speed still
limit performance. Distillation would train a smaller student from teacher
outputs; it does not shrink the teacher at inference. Cortex does not yet include
a student-training/distillation pipeline or target-device quality/throughput
benchmarks, so no comparative advantage is claimed without those measurements.
Legacy `optimization_level` values and `optimization_profile = "balanced"` /
`"ultra"` remain supported.

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
needs a unique public `id` and a supported local runtime: `cortex-cpp` loads
GGUF directly in Cortex's supervised native C++ host, and can convert supported
local Hugging Face/legacy LLaMA sources to cached GGUF (details and limitations
in [`native/README.md`](native/README.md)); legacy `cortex` uses the optional
`llama-cpp-python` binding; `llama.cpp` starts the external `llama-server`
executable; and `command` starts another local OpenAI-compatible server using
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
- Search configured models; add/edit/remove them, switch the default directly
  from a model card, and view runtime process output.
- Start/stop/restart individual runtimes or stop all loaded local models in one
  action to unload their weights and free process memory.
- A streaming text chat playground and a local API/media workbench. Chat can
  optionally request standard streaming token-usage metadata when its backend
  supports it.
- llama.cpp tuning, API settings, and secret-free config import/export. The
  Activity page reports per-model input/output token counts, streamed first-token
  latency, and completion tokens/second when the backend returns usage metadata;
  these are observed workloads, not comparative hardware benchmarks. The local
  counters retain no prompts or generated text and reset on worker restart.

Dashboard edits are validated and atomically written to `cortex.local.json` (or
the path in `CORTEX_ADMIN_CONFIG`), leaving the startup TOML unchanged. Secret
values are never stored in that JSON file; use environment variable names for
credentials. Active local model processes are restarted when their runtime
settings change. The Settings page includes an explicit **Require authenticated
API access** switch. When enabled, dashboard sign-in accepts the configured local
admin account or API key; API clients can use bearer keys when configured. The
resulting HMAC-signed HttpOnly cookie is valid for 30 days and stores no password
or API key. It is signed using the configured API key and, when present, the
account credentials. It uses SameSite=Lax on plain HTTP. Over HTTPS it is Secure,
SameSite=None, and Partitioned so embedded previews can retain the session;
unsafe cookie-authenticated requests also validate their Origin. Authentication
cannot be enabled without a non-empty `CORTEX_API_KEY`; configure the optional
admin account for dashboard username/password sign-in. Turning authentication
off removes the dashboard cookie and allows admin and inference API access
without credentials, so only do this on a trusted local listener. Changing the
API key or either account credential revokes existing dashboard sessions.

The **Model cookbook** searches the public Hugging Face model registry live and
loads results in pages of 30. Search by model/author, filter by task or GGUF
format, and sort by downloads, likes, or update time. Results are metadata only;
Cortex does not download model weights or send an HF token. Search text and
filters are sent to Hugging Face, and responses are cached in memory for five
minutes. A curated offline catalog is available if the Hub is unreachable.
Repository access and files can change; check each model card's license, usage
conditions, quantization, format, and local runtime compatibility before using
a model.

### Useful environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `CORTEX_CONFIG` | `./cortex.toml` | TOML configuration path. |
| `CORTEX_API_CONFIG` | `src/config/api.json` | OpenAI-compatible API format and route-prefix settings; restart after changes. |
| `CORTEX_ADMIN_CONFIG` | sibling `cortex.local.json` | Dashboard override file path. |
| `CORTEX_HOST` / `CORTEX_PORT` | `127.0.0.1` / `8624` | Local-only listen address by default; use `0.0.0.0` only when intentionally exposing the service. |
| `CORTEX_WORKERS` | `1` | Uvicorn worker count; local model hosting requires one. |
| `CORTEX_LLAMA_SERVER` | `PATH` lookup, then auto-install | Optional path to an external `llama-server`; if unset and not on `PATH`, Cortex installs the release binary tagged `cortex-cpp`. |
| `CORTEX_CPP_HOST` | `native/build/bin/cortex-gguf-host`, then `PATH`, then auto-install | Optional path to Cortex's native C++ GGUF host executable. |
| `CORTEX_CPP_QUANTIZE` | `native/build/bin/llama-quantize`, then `PATH`, then auto-install when needed | Optional path to the pinned native quantizer for Q8_0/Q2_K–Q6_K GGUF output formats. |
| `CORTEX_AUTO_INSTALL_RUNTIMES` | `true` | Set to `false` to disable downloads of native binaries from release tag `cortex-cpp`. |
| `CORTEX_RUNTIME_BIN_DIR` | Per-user runtime cache (`~/.cache/cortex-llmhoster/runtimes`, `~/Library/Caches/cortex-llmhoster/runtimes`, or `%LOCALAPPDATA%`) | Override the cache root for verified `llama-server`, `cortex-gguf-host`, and `llama-quantize` release binaries. |
| `CORTEX_LLAMA_CPP_SOURCE_DIR` | Native CMake dependency checkout | llama.cpp source tree containing the optional model-converter scripts. |
| `CORTEX_CPP_CONVERTER_PYTHON` | Selected Cortex Python | Interpreter for one-time optional HF/legacy model conversion; not used for inference. |
| `CORTEX_CPP_CACHE_DIR` | Per-user cache (`~/.cache` or `%LOCALAPPDATA%`) | Directory for converted GGUF cache entries; remove stale files manually. |
| `CORTEX_API_KEY` | unset | Local bearer secret; required when authentication is enabled and never a cloud-provider key. Omit it when authentication is disabled. |
| `CORTEX_ADMIN_USERNAME` / `CORTEX_ADMIN_PASSWORD` | unset | Optional single local dashboard account. Set both alongside the required API key for account sign-in; credentials are never written to `cortex.local.json` or browser storage. |
| `CORTEX_API_AUTH_ENABLED` | on iff a non-empty API key or admin username is configured | Enables/disables auth for admin and inference APIs. When on, a non-empty API key is required; the account is optional. `false` permits unauthenticated requests. The equivalent TOML field is `[server].api_auth_enabled`. |
| `CORTEX_DEFAULT_MODEL` | config-derived | Default public model ID. |
| `CORTEX_LOG_LEVEL` / `CORTEX_ACCESS_LOG` | `warning` / `false` | Server logging controls. |
| `CORTEX_MAX_JSON_BODY_BYTES` | `16777216` | Maximum buffered JSON body size. |
| `CORTEX_MAX_MEDIA_BODY_BYTES` | `2147483648` | Maximum multipart audio/image/video upload size; uploads spool to temporary storage. |
| `CORTEX_MAX_INFERENCE_REQUESTS` | Up to `8` | Active inference body/stream limit, capped by `max_connections`; excess requests receive HTTP 503 with `Retry-After`. |

CLI options `--host`, `--port`, `--workers`, `--log-level`, and
`--access-log` override their TOML/environment values. `--no-auth` (alias
`--no-api-auth`) disables all API/dashboard authentication, allowing a keyless
open local server; `--api-config` selects the OpenAI-compatible route config
JSON. For locally managed models, `--workers` must resolve to `1` to avoid
duplicate model processes.

## Endpoints

- `GET /` — self-contained dashboard; no build step or third-party assets.
- `GET /health` — liveness check.
- `GET /ready` — readiness check; a local default model must be running.
- `GET /v1/models` — public model IDs and declared capabilities.
- `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings` —
  routes implemented by each selected local runtime.
- `/v1/audio/*`, `/v1/images/*`, `/v1/videos/*`, `/v1/ocr` — multimodal or
  backend-defined routes; availability depends on that model's runtime.
- `/v1/text-to-*`, `/v1/image-to-*`, `/v1/video-to-*`, and speech task aliases —
  task-oriented paths forwarded unchanged to the configured local runtime.
- `GET /admin/api/auth/options` — public non-secret login-capability flags used
  to choose account or API-key mode in the sign-in form.
- `/admin/api/auth/login` and `/admin/api/auth/logout` — dashboard session
  creation/removal; login verifies configured local admin credentials or the
  local API key and issues/removes the signed cookie.
- `/admin/api/state`, `/admin/api/cookbook`, and other `/admin/api/*` routes —
  settings, runtime controls, process logs, and per-worker metrics. Streaming
  chat/completion clients may send `"stream_options":{"include_usage":true}`
  to request usage when their backend supports it; usage from non-streaming JSON
  responses is also detected within a small bounded inspection buffer. The
  cookbook accepts `source=huggingface` plus `q`, `task`, `format`, `sort`, and `cursor`
  parameters for paginated Hub search; `source=curated` selects its offline list.
  These routes accept the session cookie or a bearer key when auth is enabled,
  and no key when auth is disabled.

## Maintainer test commands

These commands are for the copyright holder or explicitly authorized maintainers;
they are not permission for users to modify or redistribute Cortex under the
license above.

```bash
python -m pip install -e '.[test]'
pytest
ruff check .
```
