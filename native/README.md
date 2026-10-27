# Cortex native C++ GGUF backend

`cortex-gguf-host` is Cortex's private, supervised C++ model process. It loads
GGUF directly through the llama.cpp C API and serves the private OpenAI-compatible
routes on loopback. It does **not** use Python for model loading, the
`llama-cpp-python` binding, or the external `llama-server` executable. The
release workflow can package the separate upstream `llama-server` executable
beside this host, but the Cortex C++ backend never shells out to it.

The numerical kernels and GGUF support come from the pinned llama.cpp/ggml core.
This is a native C++ integration and HTTP adapter, not a new tensor engine or a
claim that Cortex outperforms llama.cpp. Native release builds eliminate the
Python inference layer and enable CPU instruction-set tuning; actual speed
remains model-, hardware-, and build-dependent. No performance claim is made
without benchmarks on the target host.

## Build

Requirements: CMake 3.24+, Git, a C/C++17 compiler, and internet access for the
first configure. CMake fetches pinned source revisions of llama.cpp, cpp-httplib,
and nlohmann/json. No dependency is vendored into the Cortex repository.

Linux CPU build (the default build uses `GGML_NATIVE=ON` for the build machine):

```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build native/build --target cortex-gguf-host --parallel 2
# Optional; builds only when requested, for lower-bit GGUF weight conversion.
cmake --build native/build --target llama-quantize --parallel 2
```

For a release-style bundle that also builds upstream `llama-server`, enable
`CORTEX_BUILD_LLAMA_SERVER`. It creates the server and quantizer targets from
the same pinned llama.cpp checkout; all three executables are placed in
`native/build-release/bin`:

```bash
cmake -S native -B native/build-release -DCMAKE_BUILD_TYPE=Release \
  -DCORTEX_BUILD_LLAMA_SERVER=ON -DGGML_NATIVE=OFF
cmake --build native/build-release \
  --target cortex-gguf-host llama-server llama-quantize --parallel 2
```

The checked-in workflow at `.github/workflows/cortex-cpp-release.yml` builds
Linux x86_64, Windows x86_64, Intel macOS, and Apple Silicon bundles. It runs
on relevant pull requests for build validation; successful matching pushes to
`main` publish/update the rolling GitHub release at the exact tag `cortex-cpp`.
Each ZIP includes `cortex-gguf-host`, `llama-server`, `llama-quantize`, and
license notices; Linux bundles also carry `libgomp.so.1`, and Windows bundles
carry the app-local `vcomp140.dll` for OpenMP. The manifest records the bundle
size and SHA-256. Linux, Windows, and Intel macOS
x86_64 builds use an SSE2 baseline without AVX/AVX2; Apple Silicon also enables
Metal. The bundled `llama-server` is API-ready but does not include upstream
browser UI assets. PR builds do not publish releases.

The Python runtime manager checks local overrides/builds/`PATH` first, then
installs the requested binary from that release on first use. The archive is
verified against the tag's manifest and cached per user; previously installed
versions remain usable if GitHub is unreachable. Set
`CORTEX_AUTO_INSTALL_RUNTIMES=false` to disable downloading or
`CORTEX_RUNTIME_BIN_DIR` to override the runtime cache root. Model files are
never downloaded by this installer. For a platform without a release asset,
build locally or provide an executable path (`CORTEX_CPP_HOST`,
`CORTEX_LLAMA_SERVER`, or `CORTEX_CPP_QUANTIZE`).

Use a separate build directory for each hardware target. For example, a CUDA
build requires the CUDA toolkit and a compatible compiler:

```bash
cmake -S native -B native/build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON -DGGML_CUDA=ON
cmake --build native/build-cuda --target cortex-gguf-host --parallel 2
```

For Apple Silicon, configure on macOS with `-DGGML_METAL=ON`; for Vulkan, use
`-DGGML_VULKAN=ON` with the Vulkan SDK installed. Consult the pinned llama.cpp
build documentation for OS-specific prerequisites and supported accelerators.

The host is written to `native/build/bin/cortex-gguf-host` (Windows adds
`.exe`). If it is built elsewhere, set `CORTEX_CPP_HOST` to its full path or
install it on `PATH`. The optional quantizer is written beside it as
`llama-quantize`; Cortex discovers that sibling automatically or accepts an
explicit `CORTEX_CPP_QUANTIZE` path.

## Model formats

The llama.cpp inference C API loads GGUF. `cortex-cpp` accepts GGUF directly;
for supported non-GGUF inputs, Cortex runs a one-time conversion through the
converter scripts from the same pinned llama.cpp source tree, caches the output
GGUF, then starts the same native C++ inference process.

- **GGUF:** load directly. Standard numbered split files (`-00001-of-000NN`)
  are accepted as a complete directory set or by selecting any shard; Cortex
  validates the set and starts llama.cpp from shard 1. No conversion interpreter
  or converter dependencies are needed.
- **Hugging Face SafeTensors:** select a local directory with `config.json` and
  `model.safetensors` (single or sharded). `model_path` may also point at a
  standard weight file inside that directory.
- **Hugging Face PyTorch weights:** standard `pytorch_model.bin` and indexed
  shards are accepted through the upstream converter. Other model families,
  tensor names, and quantization schemes are limited to what the pinned
  llama.cpp converter supports.
- **Legacy LLaMA checkpoints:** selected `.safetensors` and legacy checkpoint
  layouts can use llama.cpp's legacy converter. This is architecture- and
  metadata-dependent; it is not a universal PyTorch checkpoint loader.
- **`.pt`, `.pth`, `.bin`, and `.ckpt`:** these extensions do not define one
  universal layout. Hugging Face `.bin` directories are handled by the HF
  converter. Legacy LLaMA pickle checkpoints require the explicit unsafe
  conversion opt-in in the model editor/TOML. Only use that opt-in for a
  checkpoint from a trusted source; pickle deserialization can execute code.
  Arbitrary training checkpoints (including many `.ckpt` files) must first be
  exported to a supported Hugging Face layout or GGUF.

For non-GGUF conversion, first build the native target so the pinned converter
scripts exist under `native/build/_deps/llama_cpp-src`. Install that upstream
converter's optional Python dependencies in a dedicated interpreter, then set
`CORTEX_CPP_CONVERTER_PYTHON` if it is not the interpreter Cortex normally
selects. `CORTEX_LLAMA_CPP_SOURCE_DIR` can point at another llama.cpp checkout.
Python is used only by the optional import/conversion step; the converted model
is loaded and used for every inference request by `cortex-gguf-host`. Cortex
does not download source weights or install conversion dependencies for you.

Converted/quantized files are cached outside the checkout in the per-user cache
(`~/.cache/cortex-llmhoster/converted` on Linux/macOS or `%LOCALAPPDATA%` on
Windows, honoring `XDG_CACHE_HOME` where set). Set `CORTEX_CPP_CACHE_DIR` to
change that location; source conversion can temporarily require the original
weights, an F16 GGUF, and the final quantized file at once. For non-GGUF sources,
`conversion_outtype` accepts direct `f16`, `q8_0`, or `f32` conversion and Q2_K–Q6_K
targets, which first convert to F16. GGUF inputs can be requantized to `q8_0`,
`q2_k`, `q2_k_s`, `q3_k_s`, `q3_k_m`, `q3_k_l`, `q4_k_s`, `q4_k_m`, `q5_k_s`,
`q5_k_m`, and `q6_k`; selecting `f16` or `f32` leaves an existing GGUF unchanged.
Q4_K_M is a practical starting point; Q2/Q3 can noticeably reduce quality.
Requantizing an already-quantized GGUF is disabled by default;
`allow_requantize = true` explicitly opts into that quality risk. Delete old
cache directories manually when they are no longer needed.

## Configure Cortex

Add a model with:

```toml
[[models]]
id = "qwen-native"
runtime = "cortex-cpp"
model_path = "/models/qwen-instruct-q4_k_m.gguf"
capabilities = ["text_generation", "embeddings"]
gpu_layers = -1
optimization_preset = "low_vram"
context_size = 2048
batch_size = 128
ubatch_size = 32
default = true
```

The `low_vram` profile caps context/batches and uses Q8 KV cache. For the
native C++ host, `gpu_layers = -1` under that profile becomes the memory-fit
sentinel `-2`; the pinned llama.cpp fitter estimates the selected context and
available device memory, then chooses a GPU layer count with a 1 GiB safety
margin. Set `gpu_layers = 0` for CPU-only operation, or `-2` directly to request
fitting without the profile. External `llama.cpp` servers keep manual layer
counts for compatibility.

`optimization_preset = "super_low"` starts with a 1024-token context, batch
and micro-batch ceilings of 64/16, Q4 KV cache, one request slot, mmap, and no
mlock. The dashboard pre-fills these values; advanced fields remain editable,
including context, batches, parallelism, cache precision, threads, GPU layers,
mmap, and mlock. Larger settings increase memory use. With Cortex C++,
`gpu_layers = -1` selects memory-fit; external llama.cpp still requires a manual
count. Q4 KV and short starting contexts can trade quality or conversation
length for lower memory use.

Only text generation and embeddings are currently supported by this native
adapter. Projector-based vision, audio, video, and other tasks need a different
runtime. Cortex starts one persistent native process for the selected model;
requests share a model context and are serialized at the inference boundary.

The process listens only on `127.0.0.1`, exposes `/health`, chat/completion
routes, and embeddings when enabled, and is started/stopped by Cortex's runtime
manager. Chat/completion accepts OpenAI-style `frequency_penalty` and
`presence_penalty` values in `[-2, 2]` along with temperature, top-p, top-k,
seed, max-token, and stop controls. Non-streaming chat/completion responses
include prompt/completion token usage. Streaming requests can opt into a final
usage-only SSE chunk with `stream_options: {"include_usage": true}`; without the
flag, the stream stays free of that extra chunk. Cortex can use this metadata for
local Activity counters without retaining prompts or generated content. Do not
expose the private backend port directly.

## Third-party notices

The native inference core is from [llama.cpp](https://github.com/ggml-org/llama.cpp)
(MIT license); HTTP handling uses
[cpp-httplib](https://github.com/yhirose/cpp-httplib) (MIT); JSON handling uses
[nlohmann/json](https://github.com/nlohmann/json) (MIT). Their source and license
files are fetched into the local build tree by CMake and are not copied into this
repository.
