# Cortex LLM Hoster

Cortex is a lightweight **local LLM hoster**. It manages `llama-server` processes
from llama.cpp and serves local GGUF models on CPU-only machines or with a
supported GPU backend. Its Python API provides an OpenAI-compatible interface,
while llama.cpp performs model loading and inference in its optimized native
runtime.

Cortex is designed to make local hosting manageable on modest hardware: select
CPU-only or GPU layer offload, tune context and batch sizes, choose lower-memory
KV caches, and start only the models you need. Actual speed and memory use depend
on the model, quantization, llama.cpp build, and hardware; Cortex does not claim
unmeasured benchmark wins.

## Highlights

- **Local GGUF hosting:** Python supervises local `llama-server` processes,
  captures their recent logs, monitors readiness, and stops them on shutdown.
- **CPU and GPU:** set `gpu_layers = 0` for CPU-only inference, `-1` for maximum
  GPU offload, or choose an explicit number of layers. GPU acceleration requires
  a llama.cpp build compatible with your device (for example CUDA, Metal, or
  Vulkan).
- **Low-memory controls:** tune context, batch and micro-batch sizes, CPU
  threads, parallel slots, memory mapping, KV-cache types, and optional mlock.
- **Memory-aware loading:** only the configured default local model starts
  automatically. Other models can be started and stopped from the dashboard so
  you do not have to keep every model loaded in RAM or VRAM.
- **Lightweight browser UI:** manage GGUF paths, runtime settings, local process
  controls and logs; inspect detected hardware; chat with hosted models; and
  import/export secret-free configuration.
- **OpenAI-compatible API:** serve locally hosted models through `/v1/models`,
  `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` (when enabled), and
  the routes supported by the selected llama.cpp build.
- **Asynchronous Python control plane:** Starlette and HTTPX, compact `orjson`
  responses, streaming with backpressure, and automatic uvloop/httptools use
  where installed.

## Requirements

- Python 3.11 or newer.
- A `llama-server` executable from a llama.cpp build. Pick a build appropriate
  for your CPU and, if applicable, GPU. Keep it on `PATH` or set
  `server.llama_server_path` / `CORTEX_LLAMA_SERVER`.
- A local GGUF model file. Cortex does not download model weights or include
  large models in the Python package.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
cp cortex.example.toml cortex.toml
```

Install llama.cpp separately. Verify that `llama-server --help` works with the
build you intend to use, then edit `cortex.toml` and set `model_path` to a GGUF
file on this machine. For example:

```toml
[server]
host = "0.0.0.0"
port = 8624
workers = 1
api_key_env = "CORTEX_API_KEY"
default_model = "qwen-local"

[[models]]
id = "qwen-local"
runtime = "llama.cpp"
model_path = "/models/qwen-instruct-q4_k_m.gguf"
default = true
# CPU-only: 0. GPU offload: -1 (all possible layers) or a positive layer count.
gpu_layers = 0
# 0 selects a conservative automatic thread count.
threads = 0
context_size = 2048
batch_size = 256
ubatch_size = 64
parallel = 1
cache_type_k = "q8_0"
cache_type_v = "q8_0"
```

Protect the API and dashboard with a bearer key, then start Cortex:

```bash
export CORTEX_API_KEY='replace-with-a-long-random-secret'
python -m cortex_llmhoster
```

The same command is available as `cortex-llmhoster`. Cortex binds to
`0.0.0.0:8624` by default; open `http://localhost:8624/` for the dashboard.
The configured default local model starts automatically. Add other GGUF files
in the **Models** section and use **Start** when you want to load one.

Send a test chat request:

```bash
curl http://localhost:8624/v1/chat/completions \
  -H "Authorization: Bearer $CORTEX_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-local","messages":[{"role":"user","content":"Hi"}],"stream":true}'
```

If `CORTEX_API_KEY` is unset, API authentication is disabled and the admin
management APIs remain locked. Configure the key before using the dashboard or
exposing the server to a network you do not fully trust.

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

Cortex intentionally requires **one Uvicorn worker** when a local model is
configured. Each Cortex worker owns its own process manager, so multiple workers
would load duplicate model weights and waste RAM/VRAM. Use llama.cpp's `parallel`
setting for concurrent requests instead. The resolved CLI/config `host`, `port`,
and `workers` values are passed to Uvicorn.

## Configuration and dashboard

`cortex.example.toml` contains a complete local-hosting example. Each local
model needs a unique public `id`, `runtime = "llama.cpp"`, and a `.gguf`
`model_path`. A single model is selected by default automatically; with multiple
models, mark one with `default = true` or configure `[server].default_model`.

The dashboard at `/` includes:

- Host CPU/RAM/GPU snapshot and local runtime status.
- Add/edit/remove model configuration, local start/stop/restart, and recent
  llama-server output.
- A streaming chat playground.
- Runtime tuning, API settings, request metrics, and secret-free config
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
| `CORTEX_ADMIN_CONFIG` | sibling `cortex.local.json` | Dashboard override file path. |
| `CORTEX_HOST` / `CORTEX_PORT` | `0.0.0.0` / `8624` | Cortex listen address. |
| `CORTEX_WORKERS` | `1` | Uvicorn worker count; local model hosting requires one. |
| `CORTEX_LLAMA_SERVER` | `PATH` lookup | Path to the `llama-server` executable. |
| `CORTEX_API_KEY` | unset | Bearer key for the API and admin endpoints. |
| `CORTEX_DEFAULT_MODEL` | config-derived | Default public model ID. |
| `CORTEX_LOG_LEVEL` / `CORTEX_ACCESS_LOG` | `warning` / `false` | Server logging controls. |
| `CORTEX_MAX_JSON_BODY_BYTES` | `16777216` | Maximum buffered JSON body size. |
| `CORTEX_MAX_INFERENCE_REQUESTS` | Up to `8` | Active inference body/stream limit, capped by `max_connections`; excess requests receive HTTP 503 with `Retry-After`. |

CLI options `--host`, `--port`, `--workers`, `--log-level`, and
`--access-log` override their TOML/environment values. For local llama.cpp
models, `--workers` must resolve to `1` to avoid duplicate model processes.

## Endpoints

- `GET /` — self-contained dashboard; no build step or third-party assets.
- `GET /health` — liveness check.
- `GET /ready` — readiness check; a local default model must be running.
- `GET /v1/models` — public model IDs.
- `/v1/*` — OpenAI-compatible routes provided by the hosted runtime.
- `/admin/api/*` — authenticated settings, runtime controls, process logs and
  per-worker metrics.

## Development

```bash
pip install -e '.[test]'
pytest
ruff check .
```
