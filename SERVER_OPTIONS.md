# Local server and backend options

Cortex is the public API gateway and runtime supervisor. Choose a backend per
model; Cortex does not need a pre-existing llama.cpp server for every option.
All managed children bind to loopback, and Cortex forwards requests to the
selected local OpenAI-compatible API.

## Managed backends

| Option | Configuration value | Best use | Model/runtime notes |
|---|---|---|---|
| **Cortex native C++ GGUF host** | `runtime = "cortex-cpp"` | Native local inference without the external `llama-server` executable or a Python inference binding. | Loads GGUF directly with llama.cpp's C API. Local Hugging Face SafeTensors and standard `pytorch_model.bin` directories can be converted once to cached GGUF using pinned llama.cpp scripts; some legacy LLaMA checkpoints are supported. Converter dependencies are optional and are not used for inference. This is not a new inference engine. |
| **Legacy Cortex Python GGUF backend** | `runtime = "cortex"` | Compatibility for existing Cortex configs that already use `llama-cpp-python`. | Still available for old configs, but new model configs should use `cortex-cpp`. Python is not used by the native backend. |
| **llama.cpp server** | `runtime = "llama.cpp"` | GGUF on CPU, CUDA, Metal, Vulkan, or other builds supported by the chosen llama.cpp binary. | Cortex starts `llama-server` and provides tuning controls, optional projector support, and health monitoring. This option requires the external executable. |
| **Custom local server** | `runtime = "command"` | Hugging Face Transformers models, high-throughput GPU engines, specialized task servers, or a user-maintained adapter. | Cortex starts an argv command (not a shell string), substitutes `{host}` and `{port}`, and checks the configured health path. The process must serve the OpenAI-compatible paths/tasks you declare. |

Cortex also adds a cross-runtime task router above these backends: clients can
send `model="auto"` and Cortex selects a configured model that declares the
endpoint's task, preferring an already-running compatible model. The selected
public model ID is returned in `X-Cortex-Selected-Model`. Unlike a raw single-
server setup, this can route across different local engines. It does not auto-
start models; at least one compatible model must be running.

The native C++ runtime is the default recommendation for local GGUF inference;
HF SafeTensors and standard HF PyTorch directories can also be used when the
optional upstream conversion dependencies are installed. Conversion runs once,
then the supervised `cortex-gguf-host` process handles inference in C++.
Legacy pickle checkpoints require an explicit trusted-file opt-in and are not
universally compatible. A model crash or out-of-memory exit does not take down
Cortex's dashboard/API worker. llama.cpp supplies the inference kernels and
format support; Cortex makes no unbenchmarked speed claim. See
[`native/README.md`](native/README.md) for supported formats, conversion setup,
and device-specific configuration. The native chat and completion routes accept
OpenAI-style `frequency_penalty` and `presence_penalty` values from −2 to 2.

## Common custom-server use cases

These are server choices for the `command` adapter, not bundled engines. Install
the server and its model dependencies separately. The dashboard includes
optional command templates for vLLM, SGLang, TGI, and MLX-LM; selecting one only
prefills an argv list and does not install the engine. Verify the installed
server version, exact flags, model format, API routes, and health endpoint before
starting it. Other engines can be configured with the generic command field.
Commands are examples; check the installed server version for exact flags and
supported model formats.

| Server | Useful when | Common deployment notes |
|---|---|---|
| **vLLM** | You want GPU throughput, continuous batching, and Hugging Face model support. | Primarily GPU-oriented; use a local model directory and expose its OpenAI-compatible API. |
| **SGLang** | You need throughput-oriented serving, structured generation, tool use, or supported multimodal models. | GPU-focused; verify that the selected model and endpoint are supported by your installed version. |
| **Hugging Face TGI** | You already operate a Transformers serving stack and need a managed model process. | Requires the matching TGI release and model format; confirm its OpenAI-compatible routes before using it as a Cortex target. |
| **MLX-LM** | You are serving models on Apple Silicon and want unified-memory acceleration. | macOS/Apple Silicon-specific; use its local OpenAI-compatible server and supported model formats. |
| **LocalAI** | You want one API front end for several locally installed inference engines. | Configure the model in LocalAI first; run it in the foreground on Cortex's assigned loopback port. |
| **Mistral.rs** | You want a Rust-based CPU/GPU backend or its supported quantized/model-family features. | Select a server mode and API path supported by the installed release. |
| **KoboldCpp** | You want a desktop-friendly GGUF server for interactive writing or role-play workloads. | Check API dialect and bind flags; use it only if its server can accept Cortex's host/port placeholders. |
| **Ollama** | You want its model-library workflow and local model management. | Its usual fixed-port service is not automatically attached as an unmanaged remote URL. Use a wrapper that launches a compatible foreground server on the supplied loopback port. |
| **LM Studio** | You want a desktop model picker and interactive testing before serving. | The GUI's already-running fixed-port server is not a managed Cortex child. Use a compatible foreground command/wrapper if you need Cortex lifecycle and health management. |

Cortex intentionally does not accept arbitrary `base_url` values in model
configuration. A command adapter must start a local process and bind the
provided loopback address and ephemeral port. This prevents model configuration
from silently becoming a proxy to an arbitrary network host.

## Example command configurations

### vLLM

```toml
[[models]]
id = "qwen-vllm"
runtime = "command"
model_path = "/models/Qwen3-8B"
capabilities = ["text_generation"]
api_base_path = "/v1"
health_path = "/health"
runtime_command = [
  "vllm", "serve", "{model_path}",
  "--host", "{host}", "--port", "{port}",
  "--served-model-name", "{model_alias}"
]
```

### SGLang

```toml
[[models]]
id = "qwen-sglang"
runtime = "command"
model_path = "/models/Qwen3-8B"
capabilities = ["text_generation"]
api_base_path = "/v1"
health_path = "/health"
runtime_command = [
  "{python}", "-m", "sglang.launch_server",
  "--model-path", "{model_path}",
  "--host", "{host}", "--port", "{port}",
  "--served-model-name", "{model_alias}"
]
```

### MLX-LM

```toml
[[models]]
id = "mlx-local"
runtime = "command"
model_path = "/models/Qwen3-8B-4bit"
capabilities = ["text_generation"]
api_base_path = "/v1"
health_path = "/health"
runtime_command = [
  "mlx_lm.server", "--model", "{model_path}",
  "--host", "{host}", "--port", "{port}"
]
```

### Hugging Face TGI

```toml
[[models]]
id = "tgi-local"
runtime = "command"
model_path = "/models/your-transformers-model"
capabilities = ["text_generation"]
api_base_path = "/v1"
health_path = "/health"
runtime_command = [
  "text-generation-launcher", "--model-id", "{model_path}",
  "--hostname", "{host}", "--port", "{port}"
]
```

If a server uses another health route, set `health_path` to that local endpoint.
If its OpenAI API is not under `/v1`, set `api_base_path` accordingly. A profile
will only work when the installed server supports the declared capabilities and
API routes; Cortex does not translate one backend's API dialect into another.

## Choosing the native C++ backend

Build Cortex's process from the repository using CMake, Git, and a C++17
compiler. The build fetches pinned upstream dependencies; it does not require
Python, pip, a virtual environment, or `llama-server`:

```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build native/build --target cortex-gguf-host --parallel 2
# Optional: build the pinned native weight quantizer for Q8_0 and Q2_K–Q6_K GGUF outputs.
cmake --build native/build --target llama-quantize --parallel 2
```

Choose **Cortex native C++ GGUF backend** in the model editor. Point `model_path`
to a local GGUF file, or to a supported Hugging Face checkpoint directory with
`config.json` and SafeTensors/`pytorch_model.bin` weights. Cortex automatically
discovers `native/build/bin/cortex-gguf-host`; set `CORTEX_CPP_HOST` if it is
installed elsewhere. Source conversion requires the pinned converter scripts
and their optional dependencies; GGUF inference remains native C++. See
[`native/README.md`](native/README.md) for supported layouts, conversion setup,
accelerator builds, and tuning. The adapter supports text generation and
embeddings, not projector-based vision, audio, video, or arbitrary custom tasks.
Use the external llama.cpp runtime or a suitable custom server for those use
cases. Non-streaming native chat/completion replies include token usage; streaming
replies add a final usage-only SSE event when the client requests
`stream_options.include_usage = true`. Cortex Activity aggregates those counts,
first streamed text-token latency, and completion throughput. Other runtimes must
return compatible usage metadata; unsupported backends may ignore or reject the
optional request field.

## Low-end device tuning and limits

The model editor's **Low VRAM** profile caps context at 2048, limits batches to
128/32, uses one request slot, Q8 KV cache, mmap, and no mlock. With Cortex's
native C++ runtime, `gpu_layers = -1` under this profile selects the pinned
llama.cpp memory fitter (`-2` explicitly requests the same auto-fit mode); it
estimates the configured context and available device memory, then chooses a
GPU layer count while preserving a memory margin. Use `gpu_layers = 0` to force
CPU-only inference. External `llama.cpp` retains explicit layer counts for
compatibility, so lower them manually if needed.

The new **SUPER LOW** mode is more aggressive and customizable: it starts at
1024 context, 64/16 batch limits, Q4 KV cache, one request slot, mmap, and no
mlock. Select it to prefill these values, then customize context, batch,
parallelism, cache type, threads, GPU layers, mmap, or mlock. Higher context,
batches, and parallelism need more memory. Q4 KV can affect quality, and short
starting contexts limit long conversations. The C++ backend memory-fits `-1`;
external llama.cpp needs a manual layer count.

For supported local HF/checkpoint sources, select a weight output format in the
Cortex C++ model editor. F16 is the converter baseline; Q4_K_M is a practical
size/quality starting point, while Q2/Q3 can reduce quality noticeably. Existing
GGUF inputs can be quantized to Q8_0 or Q2_K–Q6_K by building
`llama-quantize`; requantizing an already-quantized model is disabled by default
and must be explicitly enabled.
These controls reduce memory requirements, not total model work: 27B on an 8 GB
RTX 3050 is not guaranteed to be fast or fit wholly in VRAM, and still depends
on system RAM, context, quantization, and the model architecture. Distilling a
smaller student requires a separate training workflow; Cortex does not yet train
students or claim measured quality/throughput advantages.

The legacy `runtime = "cortex"` Python-binding backend remains available for
existing configurations. It is not recommended for new installations.
